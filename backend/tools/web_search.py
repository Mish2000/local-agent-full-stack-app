# web_search.py
import asyncio
import datetime as dt
import html
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Tuple, Optional
from zoneinfo import ZoneInfo

import httpx

# -------------------- Config / constants --------------------

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_BASE = os.getenv("TAVILY_BASE", "https://api.tavily.com")
TAVILY_SEARCH_PATH = "/search"

# Resolve aggregator links (Google/Bing) to publisher URLs for cleaner citations
RESOLVE_REDIRECTS = os.getenv("RESOLVE_REDIRECTS", "1") not in ("0", "false", "False", "no", "NO")

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
ACCEPT_LANG = "he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7"
TIMEZONE = os.getenv("TIMEZONE", "Asia/Jerusalem")

# Hosts we should NOT treat as publishers (wrappers, CDNs, trackers, socials)
_BLOCK_HOSTS = {
    "news.google.com", "google.com", "www.google.com",
    "googleusercontent.com", "lh3.googleusercontent.com", "gstatic.com",
    "fonts.googleapis.com", "fonts.gstatic.com",
    "googletagmanager.com", "google-analytics.com",
    "bing.com", "www.bing.com",
    "facebook.com", "www.facebook.com",
    "twitter.com", "t.co",
    "youtube.com", "www.youtube.com", "youtu.be",
}
# Domains we never want as final citations (link shorteners, trackers, etc.)
EXCLUDE_DOMAINS: set[str] = {
    "r.search.yahoo.com",
    "duckduckgo.com", "duckduckgo.com/l/?",
    "l.facebook.com", "lm.facebook.com", "t.co", "bit.ly",
    "feedproxy.google.com", "feeds.feedburner.com", "microsoftstart.msn.com",
}

# Credible domains get a boost (news, tech, crypto, official sources)
DOMAIN_TRUST: dict[str, float] = {
    # global news
    "reuters.com": 1.00, "apnews.com": 0.98, "bloomberg.com": 0.98, "bbc.com": 0.96,
    "nytimes.com": 0.95, "wsj.com": 0.95, "ft.com": 0.95, "theguardian.com": 0.93,
    # business/tech
    "theverge.com": 0.90, "arstechnica.com": 0.90, "techcrunch.com": 0.88, "wired.com": 0.88,
    # crypto-native
    "coindesk.com": 0.95, "cointelegraph.com": 0.92, "theblock.co": 0.90,
    "decrypt.co": 0.86, "messari.io": 0.86,
    # data/reference
    "coingecko.com": 0.90, "coinmarketcap.com": 0.85,
    # official polygon/POL (example)
    "polygon.technology": 1.00, "blog.polygon.technology": 1.00,
    # regulators / filings
    "sec.gov": 0.98, "europa.eu": 0.90, "bis.org": 0.90,
}

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".ico")
_CONTENT_NS = "{http://purl.org/rss/1.0/modules/content/}"  # for <content:encoded>

EN_STOP = {
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "by", "with", "from", "is", "are", "was", "were",
    "this", "that", "these", "those", "as", "at", "about", "into", "over", "after", "before", "latest", "today", "now", "news"
}
HE_STOP = {
    "של", "על", "אל", "אל־", "את", "עם", "או", "אבל", "גם", "כי", "כך", "לא", "כן", "מי", "מתי", "איפה", "כמה",
    "היום", "עכשיו", "חדשות", "האחרונות", "ב", "ל", "כ", "ו", "ה", "ש", "מ", "מה", "זה", "זו"
}

# -------------------- Tiny utils --------------------

def _norm_result(title: str, url: str, snippet: str, score: float, provider: str) -> Dict[str, Any]:
    return {
        "title": (title or url or "").strip(),
        "url": (url or "").strip(),
        "snippet": (snippet or "").strip(),
        "score": float(score),
        "provider": provider,
    }

def _strip_html(s: str) -> str:
    s = html.unescape(s or "")
    return re.sub(r"<[^>]+>", "", s).strip()

def _iso_date_from_rfc822(s: str) -> str:
    try:
        dtm = parsedate_to_datetime(s)
        return dtm.date().isoformat()
    except Exception:
        return ""

def _is_agg(u: str) -> bool:
    return u.startswith("https://news.google.com/") or u.startswith("https://www.bing.com/")

def _is_image_url(u: str) -> bool:
    try:
        p = urllib.parse.urlparse(u)
        path = p.path.lower()
        if any(path.endswith(ext) for ext in _IMG_EXTS):
            return True
        if ("=w" in p.query or "=h" in p.query) and "googleusercontent.com" in (p.netloc or ""):
            return True
        return False
    except Exception:
        return False

def _host(u: str) -> str:
    try:
        return urllib.parse.urlparse(u).netloc.lower()
    except Exception:
        return ""

def _looks_publisher(u: str) -> bool:
    h = _host(u)
    if not h:
        return False
    if h in _BLOCK_HOSTS or any(h.endswith("." + b) for b in _BLOCK_HOSTS):
        return False
    if _is_image_url(u):
        return False
    return True

def _extract_publisher_from_gnews_html(body: str) -> str:
    """Parse Google News HTML for a meta refresh or publisher <a href=...> anchors only."""
    if not body:
        return ""
    m = re.search(r'<meta[^>]+http-equiv=["\']refresh["\'][^>]*content=["\'][^;]*;\s*url=([^"\']+)["\']', body, re.I)
    if m:
        cand = html.unescape(m.group(1))
        if _looks_publisher(cand):
            return cand
    for href in re.findall(r'<a\s[^>]*href=["\'](https?://[^"\']+)["\']', body, re.I):
        href = html.unescape(href)
        if _looks_publisher(href):
            return href
    m2 = re.search(r'(https?://[^\s"\'<>]+)', body)
    if m2:
        cand = html.unescape(m2.group(1))
        if _looks_publisher(cand):
            return cand
    return ""

def _first_publisher_anchor(html_snippet: str) -> str:
    """Return first <a href=...> that looks like a publisher URL."""
    if not html_snippet:
        return ""
    for href in re.findall(r'<a\s[^>]*href=["\'](https?://[^"\']+)["\']', html_snippet, re.I):
        href = html.unescape(href)
        if _looks_publisher(href):
            return href
    return ""

async def _resolve_redirects_bulk(urls: List[str], timeout: float) -> Dict[str, str]:
    """Resolve aggregator links to final publisher URLs; best-effort."""
    out: Dict[str, str] = {}
    if not urls:
        return out

    headers = {
        "User-Agent": UA,
        "Accept-Language": ACCEPT_LANG,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://news.google.com/",
    }

    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        async def resolve(u: str):
            try:
                if u.startswith("https://news.google.com/"):
                    sep = "&" if "?" in u else "?"
                    u2 = f"{u}{sep}hl=he&gl=IL&ceid=IL:he"
                    r = await client.get(u2)
                    cand = _extract_publisher_from_gnews_html(r.text or "")
                    if cand and _looks_publisher(cand):
                        out[u] = cand
                    else:
                        final = str(r.url) or u
                        out[u] = final if _looks_publisher(final) else u
                else:
                    r = await client.get(u)
                    final = str(r.url) or u
                    out[u] = final if _looks_publisher(final) else u
            except Exception:
                out[u] = u

        await asyncio.gather(*(resolve(u) for u in urls))

    return out

def _is_hebrew(s: str) -> bool:
    return any("\u0590" <= ch <= "\u05FF" for ch in (s or ""))

def _norm_title(t: str) -> str:
    t = (t or "").strip().lower()
    t = re.sub(r"[\s\-–—_:|]+", " ", t)
    t = re.sub(r"[^a-zA-Z0-9\u0590-\u05FF ]+", "", t)  # Hebrew+Latin+digits
    return re.sub(r"\s+", " ", t).strip()

def _tokenize(text: str) -> List[str]:
    t = _norm_title(text)
    return [w for w in t.split() if w and w not in EN_STOP and w not in HE_STOP]

def _relevance_score(query: str, title: str, snippet: str = "") -> float:
    qtok = set(_tokenize(query))
    if not qtok:
        return 0.0
    cand = " ".join([title or "", snippet or ""])
    stok = set(_tokenize(cand))
    if not stok:
        return 0.0
    inter = qtok.intersection(stok)
    return len(inter) / max(1, len(qtok))

def _apply_relevance_filter(items: List[Dict[str, Any]], query: str, min_score: float = 0.20) -> List[Dict[str, Any]]:
    scored = []
    for it in items:
        # Keep aggregators even if score is low; they carry fresh news
        u = it.get("url", "")
        if _is_excluded_domain(u):
            continue

        s = _relevance_score(query, it.get("title", ""), it.get("snippet", "") or it.get("content", ""))
        it["rel_score"] = round(s, 3)

        # If the URL is an aggregator, accept it with a lower floor
        floor = 0.12 if _is_agg(u) else min_score
        if s >= floor:
            scored.append(it)

    # If we filtered too aggressively, keep top-3 by provider score as a safety net
    if len(scored) < 3 and items:
        fallback = sorted(items, key=lambda x: x.get("score", 0.0), reverse=True)[:3]
        for fb in fallback:
            if fb not in scored and not _is_excluded_domain(fb.get("url", "")):
                fb["rel_score"] = max(fb.get("rel_score", 0.0), 0.10)
                scored.append(fb)
    return scored


def _parse_date_heuristic(it: Dict[str, Any]) -> Optional[dt.date]:
    pub = (it.get("published_date") or "").strip()
    if pub:
        try:
            return dt.date.fromisoformat(pub[:10])
        except Exception:
            pass
    sn = (it.get("snippet") or "")
    m = re.match(r"\[(\d{4})-(\d{2})-(\d{2})\]\s", sn)
    if m:
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            pass
    u = (it.get("url") or "")
    m = re.search(r"/(20\d{2})(?:/(\d{1,2})(?:/(\d{1,2}))?)?/", u)
    if m:
        y = int(m.group(1))
        month = int(m.group(2)) if m.group(2) else 1
        day = int(m.group(3)) if m.group(3) else 1
        try:
            return dt.date(y, max(1, min(12, month)), max(1, min(28, day)))
        except Exception:
            pass
    return None

def _median_date(items: List[Dict[str, Any]]) -> Optional[dt.date]:
    ds = [d for d in (_parse_date_heuristic(it) for it in items) if d]
    if not ds:
        return None
    ds.sort()
    return ds[len(ds) // 2]

def _filter_by_age(items: List[Dict[str, Any]], max_age_days: int) -> List[Dict[str, Any]]:
    cutoff = dt.date.today() - dt.timedelta(days=max_age_days)
    out: List[Dict[str, Any]] = []
    for it in items:
        d = _parse_date_heuristic(it)
        if d is None:
            out.append(it)  # keep but down-rank later
        elif d >= cutoff:
            out.append(it)
    return out

def _extract_domain(u: str) -> str:
    try:
        p = urllib.parse.urlparse(u)
        return p.netloc.lower()
    except Exception:
        return ""

def _is_excluded_domain(u: str) -> bool:
    d = _extract_domain(u)
    if not d:
        return False
    # Never exclude Google/Bing/Yahoo aggregator links; we want them as a fallback if publisher resolution fails
    if _is_agg(u):
        return False
    return any(d == ex or d.endswith(f".{ex}") for ex in EXCLUDE_DOMAINS)

def _root_domain(u: str) -> str:
    d = _extract_domain(u)
    if d.count(".") >= 2:
        parts = d.split(".")
        return ".".join(parts[-2:])
    return d

def _domain_trust(u: str) -> float:
    rd = _root_domain(u)
    return DOMAIN_TRUST.get(rd, 0.50)  # neutral default

def _days_ago(d: Optional[dt.date]) -> Optional[int]:
    if not d:
        return None
    return (dt.date.today() - d).days

def _diversify(items: List[Dict[str, Any]], per_domain: int = 2, cap: int = 10) -> List[Dict[str, Any]]:
    """Keep at most `per_domain` per host on the first pass, then fill up to `cap`."""
    seen = defaultdict(int)
    primary, spill = [], []
    for it in items:
        h = _host(it.get("url", ""))
        if not h:
            spill.append(it)
            continue
        if seen[h] < per_domain:
            primary.append(it); seen[h] += 1
        else:
            spill.append(it)
    out = primary[:cap]
    for it in spill:
        if len(out) >= cap:
            break
        out.append(it)
    return out

def _annotate_debug_fields(items: List[Dict[str, Any]]) -> None:
    for it in items:
        d = _parse_date_heuristic(it)
        it["_debug_age_days"] = _days_ago(d)
        it["_debug_domain"] = _root_domain(it.get("url", ""))
        it["_debug_trust"] = round(_domain_trust(it.get("url", "")), 3)

def _sort_results(items: List[Dict[str, Any]], *, prefer_new: bool) -> List[Dict[str, Any]]:
    def key(it: Dict[str, Any]):
        d = _parse_date_heuristic(it)
        # Freshness: newer is better; if unknown date -> treat as older than ~month
        fresh = -( _days_ago(d) if (prefer_new and d) else 999 )
        rel = float(it.get("rel_score") or 0.0)
        trust = _domain_trust(it.get("url", ""))
        prov = float(it.get("score") or 0.0)
        return (fresh, rel, trust, prov)
    return sorted(items, key=key, reverse=True)

def _augment_query_recency(query: str) -> List[str]:
    today = dt.date.today()
    month_name = today.strftime("%B")
    y = today.year
    return [
        f"{query} {month_name} {y}",
        f"{query} {y}",
        f"{query} latest",
        f"{query} news",
    ]

def _looks_time_sensitive(text: str) -> bool:
    pats = [
        r"\b(היום|אתמול|מחר|מעודכן|עדכני|חדש|בשבוע\s+האחרון|בחודש\s+האחרון)\b",
        r"\b(today|yesterday|tomorrow|latest|current|this\s+week|last\s+week|this\s+month|last\s+month|202[4-9]|20[3-9]\d)\b",
        r"\b(price|rate|launch|released?|announc(?:e|ed|ement)|deadline|version|results?)\b",
        r"\bשער\b",
    ]
    return any(re.search(p, text or "", flags=re.IGNORECASE) for p in pats)

def _max_age_days(tr: Optional[str], d: Optional[int]) -> Optional[int]:
    if d and d > 0:
        return int(d)
    tr = (tr or "").lower()
    return {"d": 1, "day": 1, "w": 7, "week": 7, "m": 31, "month": 31, "y": 365, "year": 365}.get(tr)

# -------------------- Smart facts (local) --------------------

def _intent_current_date(q: str) -> bool:
    pats = [
        r"\bwhat(?:'s| is)?\s+the\s+date\b",
        r"\bwhat\s+date\s+is\s+it\b",
        r"\btoday(?:'s)?\s+date\b",
        r"\bcurrent\s+date\b",
        r"\bמה\s+התאריך\b",
        r"\bאיזה\s+תאריך\s+(היום|כיום)\b",
        r"\bמה\s+היום\b",
    ]
    return any(re.search(p, q, re.IGNORECASE) for p in pats)

def _intent_fx_query(q: str) -> Optional[Tuple[str, str]]:
    """
    Detect FX intent and return (base, quote) like ("USD", "ILS").
    """
    qn = _norm_title(q)
    m = re.search(r"\b([a-z]{3})\s*(?:to|/)\s*([a-z]{3})\b", qn)
    if m:
        return m.group(1).upper(), m.group(2).upper()
    aliases = {
        "dollar": "USD", "usd": "USD", "דולר": "USD",
        "shekel": "ILS", "nis": "ILS", "ils": "ILS", "שקל": "ILS", "ש\"ח": "ILS",
        "euro": "EUR", "eur": "EUR", "אירו": "EUR",
        "pound": "GBP", "gbp": "GBP", "לירה": "GBP",
    }
    looks_fx = any(s in qn for s in ("rate", "exchange", "fx", "שער", "כמה", "worth", "value", "מחיר"))
    if not looks_fx:
        return None
    mentioned: set[str] = set()
    for k, code in aliases.items():
        if k in qn:
            mentioned.add(code)
    if not mentioned:
        return None
    if "USD" in mentioned and "ILS" in mentioned: return "USD", "ILS"
    if "EUR" in mentioned and "ILS" in mentioned: return "EUR", "ILS"
    if "GBP" in mentioned and "ILS" in mentioned: return "GBP", "ILS"
    if "USD" in mentioned and "EUR" in mentioned: return "USD", "EUR"
    if "USD" in mentioned and "GBP" in mentioned: return "USD", "GBP"
    if "EUR" in mentioned and "GBP" in mentioned: return "EUR", "GBP"
    if "USD" in mentioned: return "USD", "ILS"
    if "EUR" in mentioned: return "EUR", "ILS"
    if "GBP" in mentioned: return "GBP", "ILS"
    if "ILS" in mentioned: return "USD", "ILS"
    return None

async def _fx_fetch_rate(base: str, quote: str, timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    url = f"https://api.exchangerate.host/latest?base={base}&symbols={quote}"
    try:
        async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent": UA}) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return None
            data = r.json()
            rate = (data.get("rates") or {}).get(quote)
            if rate is None:
                return None
            ts = data.get("date")
            return {"base": base, "quote": quote, "rate": float(rate), "as_of": ts}
    except Exception:
        return None

# --- Crypto smart-fact (CoinGecko) ---

_CRYPTO_ALIASES = {
    "bitcoin": "bitcoin", "btc": "bitcoin", "ביטקוין": "bitcoin",
    "ethereum": "ethereum", "ether": "ethereum", "eth": "ethereum", "אתריום": "ethereum",
    "sol": "solana", "solana": "solana",
    "ada": "cardano", "cardano": "cardano",
    "doge": "dogecoin", "dogecoin": "dogecoin",
    "pol": "polygon-ecosystem-token", "polygon": "polygon-ecosystem-token",
}

def _intent_crypto_query(q: str) -> Optional[Tuple[str, str]]:
    """
    Detect crypto price intent. Returns (coin_id, vs_currency) suitable for CoinGecko.
    """
    qn = _norm_title(q)
    coin_id: Optional[str] = None
    for k, cid in _CRYPTO_ALIASES.items():
        if re.search(rf"\b{k}\b", qn):
            coin_id = cid
            break
    if not coin_id:
        return None
    if any(v in qn for v in ("ils", "nis", "שקל", "ש\"ח", "shekel")):
        vs = "ils"
    elif "eur" in qn or "euro" in qn or "אירו" in qn:
        vs = "eur"
    else:
        vs = "ils"  # locale default
    return coin_id, vs

async def _crypto_fetch_price(coin_id: str, vs_currency: str, timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    url = (
        "https://api.coingecko.com/api/v3/simple/price"
        f"?ids={urllib.parse.quote_plus(coin_id)}&vs_currencies={urllib.parse.quote_plus(vs_currency)}"
    )
    try:
        async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent": UA}) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return None
            data = r.json()
            px = (data.get(coin_id) or {}).get(vs_currency)
            if px is None:
                return None
            return {"coin_id": coin_id, "vs_currency": vs_currency, "price": float(px)}
    except Exception:
        return None

# -------------------- Providers --------------------

class WebSearcher:
    """
    Multi-provider search:
      - Tavily (if key)
      - Google News RSS (he-IL/IL)
      - Bing News RSS (he/IL)
      - Yahoo News RSS (generic)
      - DuckDuckGo Instant Answer (fallback)
    Plus: smart facts (date / FX / crypto).

    Returns from search(...):
      items, provider_label, debug_dict, timed_out
    """

    def __init__(self) -> None:
        self.have_tavily = bool(TAVILY_API_KEY)

    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        timeout: float = 20.0,
        time_budget_sec: float = 45.0,
        days: int | None = None,
        topic: str | None = None,
        time_range: Optional[str] = None,
        auto_parameters: Optional[bool] = None,
        **_ignored: Any,
    ):
        t_start = time.monotonic()
        debug: Dict[str, Any] = {
            "topic": topic,
            "days": days,
            "time_range": time_range,
            "auto_parameters": auto_parameters,
            "timings_ms": {},
            "by_provider_count": {},
            "notes": [],
        }
        timed_out = False

        # ---------------- Smart facts (run BEFORE any provider calls) ----------------
        if _intent_current_date(query):
            now = dt.datetime.now(ZoneInfo(TIMEZONE))
            iso = now.strftime("%Y-%m-%d")
            human = now.strftime("%B %d, %Y")
            items = [{
                "title": "Current Date",
                "url": f"local://date/{iso}",
                "snippet": f"Today is {human} ({iso}) in {TIMEZONE}.",
                "content": f"Today is {human} ({iso}) in {TIMEZONE}.",
                "published_date": iso,
                "published": iso,  # <-- alias for main.py
                "score": 1.0,
                "provider": "local-fact",
            }]
            debug["notes"].append("smart_fact=date")
            debug["returned"] = 1
            debug["elapsed_ms"] = int((time.monotonic() - t_start) * 1000)
            return items, "local-fact", debug, False

        fx = _intent_fx_query(query)
        if fx:
            base, quote = fx
            ts0 = time.monotonic()
            fxdata = await _fx_fetch_rate(base, quote, timeout=min(8.0, timeout))
            debug["timings_ms"]["fx"] = int((time.monotonic() - ts0) * 1000)
            if fxdata:
                items = [{
                    "title": f"{base}/{quote} rate",
                    "url": f"local://fx/{base}{quote}",
                    "snippet": f"{base}→{quote} ≈ {fxdata['rate']:.6f} as of {fxdata['as_of']}",
                    "content": f"Spot FX: 1 {base} = {fxdata['rate']:.6f} {quote} (as of {fxdata['as_of']}).",
                    "published_date": fxdata["as_of"],
                    "published": fxdata["as_of"],  # <-- alias
                    "score": 1.0,
                    "provider": "local-fact",
                }]
                debug["notes"].append(f"smart_fact=fx {base}->{quote}")
                debug["returned"] = 1
                debug["elapsed_ms"] = int((time.monotonic() - t_start) * 1000)
                return items, "local-fact", debug, False

        crypto = _intent_crypto_query(query)
        if crypto:
            coin_id, vs = crypto
            ts0 = time.monotonic()
            cdata = await _crypto_fetch_price(coin_id, vs, timeout=min(8.0, timeout))
            debug["timings_ms"]["crypto"] = int((time.monotonic() - ts0) * 1000)
            if cdata:
                vs_u = vs.upper()
                items = [{
                    "title": f"{coin_id.replace('-', ' ').title()} price ({vs_u})",
                    "url": f"local://crypto/{coin_id}:{vs_u}",
                    "snippet": f"≈ {cdata['price']:.6f} {vs_u}",
                    "content": f"1 {coin_id} ≈ {cdata['price']:.6f} {vs_u}.",
                    "score": 1.0,
                    "provider": "local-fact",
                }]
                debug["notes"].append(f"smart_fact=crypto {coin_id}->{vs_u}")
                debug["returned"] = 1
                debug["elapsed_ms"] = int((time.monotonic() - t_start) * 1000)
                return items, "local-fact", debug, False

        # ---------------- Recency inference ----------------
        tr_inferred: Optional[str] = None
        if topic is None and _looks_time_sensitive(query):
            topic = "news"
            if re.search(r"\b(היום|today|now)\b", query, re.I):
                tr_inferred = "day"
            elif re.search(r"\b(השבוע|this\s+week|last\s+week|week)\b", query, re.I):
                tr_inferred = "week"
            elif re.search(r"\b(בחודש\s+האחרון|this\s+month|last\s+month|month)\b", query, re.I):
                tr_inferred = "month"
            else:
                tr_inferred = "month"

        # Prefer explicit time_range if supplied from caller
        effective_tr = (time_range or tr_inferred)
        max_age_days = _max_age_days(effective_tr, days)
        debug["notes"].append(
            f"recency_infer topic={topic or 'general'} tr={effective_tr or '-'} days={max_age_days or '-'}")

        variants = [query]
        if topic == "news":
            variants += _augment_query_recency(query)

        # ---- Pass 1
        results, provider_used, _ = await self._search_pass(
            variants=variants,
            query=query,
            max_results=max_results,
            timeout=timeout,
            prefer_news=(topic == "news"),
            max_age_days=max_age_days,
            debug=debug,
            force_advanced=(topic == "news"),
            auto_parameters=auto_parameters,
            time_range=effective_tr,
        )

        # ---- If still old, try stricter recency
        med = _median_date(results) if results else None
        if (topic == "news") and max_age_days and med:
            cutoff = dt.date.today() - dt.timedelta(days=max_age_days)
            if med < cutoff:
                debug["notes"].append(f"recency_requery median={med.isoformat()} cutoff={cutoff.isoformat()}")
                results2, provider2, _ = await self._search_pass(
                    variants=_augment_query_recency(query),
                    query=query,
                    max_results=max_results,
                    timeout=timeout,
                    prefer_news=True,
                    max_age_days=max_age_days,
                    debug=debug,
                    force_advanced=True,
                )
                if results2:
                    results, provider_used = results2, provider2

        # Final relevance + sort + diversity
        if results:
            results = _apply_relevance_filter(results, query, min_score=0.32)
            results = _sort_results(results, prefer_new=bool(max_age_days))
            results = _diversify(results, per_domain=2, cap=max_results)
            _annotate_debug_fields(results)

        # Budget check
        if not results and (time.monotonic() - t_start) >= time_budget_sec:
            timed_out = True
            debug["notes"].append("timeout_reached=1")

        debug["returned"] = len(results)
        debug["elapsed_ms"] = int((time.monotonic() - t_start) * 1000)

        if re.search(r"\b(write|story|poem|haiku|joke|role ?play|lyrics)\b", (query or ""), re.IGNORECASE):
            debug["notes"].append("Creative prompt detected — web search skipped")
            return [], "none", debug, False

        return results, provider_used, debug, timed_out

    async def _search_pass(
        self,
        *,
        variants: List[str],
        query: str,
        max_results: int,
        timeout: float,
        prefer_news: bool,
        max_age_days: Optional[int],
        debug: Dict[str, Any],
        force_advanced: bool = False,
        auto_parameters: Optional[bool] = None,  # <-- NEW
        time_range: Optional[str] = None,  # <-- NEW
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        provider_mix: List[str] = []
        results: List[Dict[str, Any]] = []

        for qv in variants:
            batch: List[Dict[str, Any]] = []

            # Tavily
            if self.have_tavily and len(batch) < max_results:
                ts = time.monotonic()
                ok, res, d = await self._tavily_search(
                    qv, max_results, timeout,
                    topic=("news" if prefer_news else None),
                    time_range=(time_range if prefer_news else None),
                    days=(max_age_days if prefer_news else None),
                    auto_parameters=(True if auto_parameters is None else bool(auto_parameters)),
                    force_advanced=(force_advanced or prefer_news),
                )
                debug["timings_ms"]["tavily"] = int((time.monotonic() - ts) * 1000)
                debug["by_provider_count"]["tavily"] = len(res) if ok else 0
                debug["notes"] += d
                if ok and res:
                    batch.extend(res)
                    provider_mix.append("tavily")

            # Google News
            if len(batch) < max_results:
                ts = time.monotonic()
                ok, res, d = await self._google_news_rss(qv, max_results - len(batch), timeout)
                debug["timings_ms"]["google-news"] = int((time.monotonic() - ts) * 1000)
                debug["by_provider_count"]["google-news"] = len(res) if ok else 0
                debug["notes"] += d
                if ok and res:
                    batch.extend(res)
                    provider_mix.append("google-news")

            # Bing News
            if len(batch) < max_results:
                ts = time.monotonic()
                ok, res, d = await self._bing_news_rss(qv, max_results - len(batch), timeout)
                debug["timings_ms"]["bing-news"] = int((time.monotonic() - ts) * 1000)
                debug["by_provider_count"]["bing-news"] = len(res) if ok else 0
                debug["notes"] += d
                if ok and res:
                    batch.extend(res)
                    provider_mix.append("bing-news")

            # Yahoo News
            if len(batch) < max_results:
                ts = time.monotonic()
                ok, res, d = await self._yahoo_news_rss(qv, max_results - len(batch), timeout)
                debug["timings_ms"]["yahoo-news"] = int((time.monotonic() - ts) * 1000)
                debug["by_provider_count"]["yahoo-news"] = len(res) if ok else 0
                debug["notes"] += d
                if ok and res:
                    batch.extend(res)
                    provider_mix.append("yahoo-news")

            # DuckDuckGo fallback
            if len(batch) < max_results:
                ts = time.monotonic()
                ok, res, d = await self._duckduckgo_fallback(qv, timeout)
                debug["timings_ms"]["duckduckgo"] = int((time.monotonic() - ts) * 1000)
                debug["by_provider_count"]["duckduckgo"] = len(res) if ok else 0
                debug["notes"] += d
                if ok and res:
                    seen = {r.get("url") for r in batch if r.get("url")}
                    for r in res:
                        if r.get("url") and r["url"] not in seen and not _is_excluded_domain(r["url"]):
                            batch.append(r)
                            seen.add(r["url"])
                    provider_mix.append("duckduckgo")

            if batch:
                # Resolve aggregators → publishers
                if RESOLVE_REDIRECTS:
                    agg_urls = [r["url"] for r in batch if _is_agg(r.get("url", ""))]
                    if agg_urls:
                        ts = time.monotonic()
                        mapping = await _resolve_redirects_bulk(agg_urls, timeout)
                        debug["timings_ms"]["resolve_redirects"] = int((time.monotonic() - ts) * 1000)
                        fixed = 0
                        for r in batch:
                            u = r.get("url", "")
                            if u in mapping and mapping[u] and _looks_publisher(mapping[u]):
                                r["url"] = mapping[u]
                                fixed += 1
                        debug["notes"].append(f"agg_fixed={fixed}")

                # Dedupe
                batch = _dedupe_results(batch)

                # Age filter
                if max_age_days:
                    batch = _filter_by_age(batch, max_age_days)

                # Relevance & sort
                batch = _apply_relevance_filter(batch, query, min_score=0.20)
                batch = _sort_results(batch, prefer_new=bool(max_age_days))

                _annotate_debug_fields(batch)

                if batch:
                    results = batch[:max_results]
                    break  # stop at first variant that yields solid results

        provider_used = "mixed" if len(set(provider_mix)) > 1 else (provider_mix[0] if provider_mix else "none")
        return results, provider_used, debug

    # ---------------- Tavily ----------------
    async def _tavily_search(
        self,
        query: str,
        max_results: int,
        timeout: float,
        *,
        topic: str | None = None,
        time_range: str | None = None,
        days: int | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        auto_parameters: bool = True,
        force_advanced: bool = False,
    ) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        if not self.have_tavily:
            return True, [], ["tavily disabled (no key)"]
        url = f"{TAVILY_BASE}{TAVILY_SEARCH_PATH}"
        headers = {
            "Authorization": f"Bearer {TAVILY_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": UA,
            "Accept-Language": ACCEPT_LANG,
        }
        payload: Dict[str, Any] = {
            "query": query,
            "search_depth": "advanced" if (force_advanced or (topic == "news")) else "basic",
            "max_results": max(1, min(int(max_results), 10)),
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
            "include_image_descriptions": False,
            "auto_parameters": bool(auto_parameters),
            "api_key": TAVILY_API_KEY or "",
        }
        if topic:
            payload["topic"] = topic  # "news" | "general"
        if time_range:
            payload["time_range"] = time_range  # "day" | "week" | "month" | "year"
        if (days is not None) and days > 0:
            payload["days"] = int(days)
        if include_domains:
            payload["include_domains"] = include_domains[:300]
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        dbg: List[str] = [
            f"tavily depth={payload['search_depth']} topic={payload.get('topic', '-')} tr={payload.get('time_range', '-')} days={payload.get('days', '-')}"
        ]
        try:
            async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
                r = await client.post(url, json=payload)
                dbg.append(f"tavily status={r.status_code}")
                if r.status_code != 200:
                    dbg.append(f"tavily fail={r.text[:400]}")
                    return False, [], dbg
                data = r.json()
        except Exception as e:
            dbg.append(f"tavily exception={type(e).__name__}: {e}")
            return False, [], dbg

        out: List[Dict[str, Any]] = []
        for i, it in enumerate((data or {}).get("results") or []):
            title = (it.get("title") or "").strip()
            link = (it.get("url") or "").strip()
            content = (it.get("content") or "").strip()
            if not link or _is_excluded_domain(link):
                continue
            score = float(it.get("score") or (1.0 - min(i * 0.1, 0.9)))
            item = _norm_result(title, link, content, score, "tavily")
            pub = (it.get("published_date") or "").strip()
            if pub:
                item["published_date"] = pub
                item["published"] = pub
            out.append(item)
        dbg.append(f"tavily results={len(out)}")
        return True, out, dbg

    # ---------------- Google News RSS ----------------
    async def _google_news_rss(
        self, query: str, max_results: int, timeout: float
    ) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        dbg: List[str] = []
        if max_results <= 0:
            return True, [], dbg
        q = urllib.parse.quote_plus(query)
        rss_url = f"https://news.google.com/rss/search?q={q}&hl=he-IL&gl=IL&ceid=IL:he"
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                headers={"User-Agent": UA, "Accept-Language": ACCEPT_LANG},
                follow_redirects=True,
            ) as client:
                r = await client.get(rss_url)
                dbg.append(f"gnews status={r.status_code}")
                if r.status_code != 200 or not r.text:
                    return False, [], dbg
                root = ET.fromstring(r.text)
        except Exception as e:
            dbg.append(f"gnews exception={type(e).__name__}: {e}")
            return False, [], dbg

        items: List[Dict[str, Any]] = []
        desc_used = 0
        src_fallback = 0
        for i, item in enumerate(root.findall(".//item")[: max_results]):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()  # aggregator
            content_html = item.findtext(f"{_CONTENT_NS}encoded") or ""
            desc_html = item.findtext("description") or ""
            html_blob = content_html or desc_html

            desc_text = _strip_html(html_blob or desc_html)
            pub = _iso_date_from_rfc822(item.findtext("pubDate") or "")
            snippet = f"[{pub}] {desc_text or title}".strip() if pub else (desc_text or title)

            pub_url = _first_publisher_anchor(html_blob) or _first_publisher_anchor(desc_html)
            if pub_url and _looks_publisher(pub_url):
                link = pub_url
                desc_used += 1
            else:
                src_el = item.find("source")
                src_url = (src_el.get("url") if src_el is not None else "") or ""
                if src_url and _looks_publisher(src_url):
                    link = src_url
                    src_fallback += 1

            if not link:
                continue
            score = 1.0 - min(i * 0.05, 0.9)
            item = _norm_result(title, link, snippet, score, "google-news")
            if pub:
                item["published_date"] = pub
                item["published"] = pub
            items.append(item)

        dbg.append(f"gnews results={len(items)}")
        dbg.append(f"gnews_desc_links_used={desc_used}")
        dbg.append(f"gnews_source_url_fallback={src_fallback}")
        return True, items, dbg

    # ---------------- Bing News RSS ----------------
    async def _bing_news_rss(
        self, query: str, max_results: int, timeout: float
    ) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        dbg: List[str] = []
        if max_results <= 0:
            return True, [], dbg
        q = urllib.parse.quote_plus(query)
        rss_url = f"https://www.bing.com/news/search?q={q}&format=rss&setlang=he&cc=IL"
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                headers={"User-Agent": UA, "Accept-Language": ACCEPT_LANG},
                follow_redirects=True,
            ) as client:
                r = await client.get(rss_url)
                dbg.append(f"bing status={r.status_code}")
                if r.status_code != 200 or not r.text:
                    return False, [], dbg
                root = ET.fromstring(r.text)
        except Exception as e:
            dbg.append(f"bing exception={type(e).__name__}: {e}")
            return False, [], dbg

        items: List[Dict[str, Any]] = []
        for i, item in enumerate(root.findall(".//item")[: max_results]):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            desc = _strip_html(item.findtext("description") or "")
            pub = _iso_date_from_rfc822(item.findtext("pubDate") or "")
            snippet = f"[{pub}] {desc or title}".strip() if pub else (desc or title)
            if not link:
                continue
            score = 1.0 - min(i * 0.05, 0.9)
            item = _norm_result(title, link, snippet, score, "bing-news")
            if pub:
                item["published_date"] = pub
                item["published"] = pub
            items.append(item)
        dbg.append(f"bing results={len(items)}")
        return True, items, dbg

    # ---------------- Yahoo News RSS ----------------
    async def _yahoo_news_rss(
        self, query: str, max_results: int, timeout: float
    ) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        dbg: List[str] = []
        if max_results <= 0:
            return True, [], dbg
        q = urllib.parse.quote_plus(query)
        rss_url = f"https://news.search.yahoo.com/rss?p={q}&fr=news"
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                headers={"User-Agent": UA, "Accept-Language": ACCEPT_LANG},
                follow_redirects=True,
            ) as client:
                r = await client.get(rss_url)
                dbg.append(f"yahoo status={r.status_code}")
                if r.status_code != 200 or not r.text:
                    return False, [], dbg
                root = ET.fromstring(r.text)
        except Exception as e:
            dbg.append(f"yahoo exception={type(e).__name__}: {e}")
            return False, [], dbg

        items: List[Dict[str, Any]] = []
        for i, item in enumerate(root.findall(".//item")[: max_results]):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            desc = _strip_html(item.findtext("description") or "")
            pub = _iso_date_from_rfc822(item.findtext("pubDate") or "")
            snippet = f"[{pub}] {desc or title}".strip() if pub else (desc or title)
            if not link:
                continue
            score = 1.0 - min(i * 0.05, 0.9)
            item = _norm_result(title, link, snippet, score, "yahoo-news")
            if pub:
                item["published_date"] = pub
                item["published"] = pub
            items.append(item)
        dbg.append(f"yahoo results={len(items)}")
        return True, items, dbg

    # ---------------- DuckDuckGo IA ----------------
    async def _duckduckgo_fallback(self, query: str, timeout: float) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        dbg: List[str] = []
        q = urllib.parse.quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={q}&format=json&no_redirect=1&skip_disambig=1"
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                headers={"User-Agent": UA, "Accept-Language": ACCEPT_LANG},
                follow_redirects=True,
            ) as client:
                r = await client.get(url)
                dbg.append(f"ddg status={r.status_code}")
                if r.status_code != 200:
                    return False, [], dbg
                data = r.json()
        except Exception as e:
            dbg.append(f"ddg exception={type(e).__name__}: {e}")
            return False, [], dbg

        out: List[Dict[str, Any]] = []
        abs_text = str(data.get("AbstractText") or "").strip()
        abs_url = str(data.get("AbstractURL") or "").strip()
        heading = str(data.get("Heading") or "").strip()
        if abs_text and abs_url:
            out.append(_norm_result(heading or abs_url, abs_url, abs_text, 1.0, "duckduckgo"))

        for i, rel in enumerate(data.get("RelatedTopics", [])[:5]):
            if isinstance(rel, dict):
                txt = str(rel.get("Text") or "").strip()
                first = str(rel.get("FirstURL") or "").strip()
                if first:
                    out.append(_norm_result("", first, txt, 0.9 - i * 0.1, "duckduckgo"))
        dbg.append(f"ddg results={len(out)}")
        return True, out, dbg

# -------------------- Dedupe (URL+title) --------------------

def _dedupe_results(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Dedupe by canonical URL host+path and fuzzy-normalized title."""
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        u = (it.get("url") or "").strip()
        if not u:
            continue
        try:
            p = urllib.parse.urlparse(u)
            key_u = f"{p.netloc.lower()}{p.path}"
        except Exception:
            key_u = u
        tnorm = _norm_title(it.get("title") or "")[:120]
        if (key_u in seen_urls) or (tnorm and tnorm in seen_titles):
            continue
        seen_urls.add(key_u)
        if tnorm:
            seen_titles.add(tnorm)
        out.append(it)
    return out
