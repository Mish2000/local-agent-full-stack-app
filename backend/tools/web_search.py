import os
import re
import html
import time
import asyncio
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple
from email.utils import parsedate_to_datetime

import httpx

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

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".ico")
_CONTENT_NS = "{http://purl.org/rss/1.0/modules/content/}"  # for <content:encoded>


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
        dt = parsedate_to_datetime(s)
        return dt.date().isoformat()
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

    async with httpx.AsyncClient(
        timeout=timeout,
        headers=headers,
        follow_redirects=True,
    ) as client:
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


def _query_variants(q: str) -> List[str]:
    """Generate robust query variants (he/en) to avoid 0-results cases."""
    q = (q or "").strip()
    base = [q]
    # Normalize punctuation/spaces
    q1 = re.sub(r"[“”\"'’]+", "", q)
    q1 = re.sub(r"\s+", " ", q1)
    if q1 and q1 != q:
        base.append(q1)

    # Add news-y wrapper terms
    if _is_hebrew(q):
        base += [
            f"חדשות {q1}",
            f"{q1} עדכני",
            f"{q1} 2025",
            "החדשות האחרונות בינה מלאכותית",
            "חדשות LLM מתקדמים",
        ]
        # English fallbacks for tough queries (help Bing/Google News)
        base += [
            "most advanced LLM today",
            "state of the art large language model 2025",
            "latest AI model 2025",
            "OpenAI model latest",
        ]
    else:
        base += [
            f"{q1} latest",
            f"{q1} 2025",
            "latest AI news",
            "most advanced LLM today",
            "state of the art LLM",
        ]
        # Hebrew fallback
        base += ["מודל שפה גדול מתקדם ביותר", "חדשות AI עדכני"]

    # Deduplicate while keeping order
    seen = set()
    out = []
    for s in base:
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


class WebSearcher:
    """
    Provider chain (no key required for RSS paths). With time-budget loop.

    Providers per attempt:
      - Tavily (if key)
      - Google News RSS (he-IL/IL)
      - Bing News RSS (he/IL)
      - Yahoo News RSS (generic)
      - DuckDuckGo Instant Answer

    search(...) guarantees: either >=1 result or timed_out=True in debug after time_budget_sec.
    """

    def __init__(self) -> None:
        self.have_tavily = bool(TAVILY_API_KEY)

    async def search(
        self,
        query: str,
        max_results: int = 5,
        timeout: float = 15.0,
        time_budget_sec: float = 60.0,
    ) -> Tuple[List[Dict[str, Any]], str, List[str], bool]:
        start = time.monotonic()
        overall_debug: List[str] = []
        provider_mix: List[str] = []
        results: List[Dict[str, Any]] = []

        variants = _query_variants(query)
        attempt = 0
        timed_out = False

        for qv in variants:
            attempt += 1
            if (time.monotonic() - start) > time_budget_sec:
                timed_out = True
                overall_debug.append("timeout_reached=1")
                break

            batch_debug: List[str] = [f"attempt={attempt}", f"q='{qv}'"]
            batch_results: List[Dict[str, Any]] = []
            batch_providers: List[str] = []

            # Tavily
            if self.have_tavily and len(batch_results) < max_results:
                ok, res, d = await self._tavily_search(qv, max_results - len(batch_results), timeout)
                batch_debug += d
                if ok and res:
                    batch_results.extend(res)
                    batch_providers.append("tavily")

            # Google News
            if len(batch_results) < max_results:
                ok, res, d = await self._google_news_rss(qv, max_results - len(batch_results), timeout)
                batch_debug += d
                if ok and res:
                    batch_results.extend(res)
                    batch_providers.append("google-news")

            # Bing News
            if len(batch_results) < max_results:
                ok, res, d = await self._bing_news_rss(qv, max_results - len(batch_results), timeout)
                batch_debug += d
                if ok and res:
                    batch_results.extend(res)
                    batch_providers.append("bing-news")

            # Yahoo News RSS
            if len(batch_results) < max_results:
                ok, res, d = await self._yahoo_news_rss(qv, max_results - len(batch_results), timeout)
                batch_debug += d
                if ok and res:
                    batch_results.extend(res)
                    batch_providers.append("yahoo-news")

            # DuckDuckGo IA
            if len(batch_results) < max_results:
                ok, res, d = await self._duckduckgo_fallback(qv, timeout)
                batch_debug += d
                if ok and res:
                    seen = {r["url"] for r in batch_results if r.get("url")}
                    for r in res:
                        if r.get("url") and r["url"] not in seen:
                            batch_results.append(r)
                            seen.add(r["url"])
                    batch_providers.append("duckduckgo")

            # Resolve aggregator links
            pub_fixed = 0
            if RESOLVE_REDIRECTS and batch_results:
                agg_urls = [r["url"] for r in batch_results if _is_agg(r.get("url", ""))]
                if agg_urls:
                    mapping = await _resolve_redirects_bulk(agg_urls, timeout)
                    fixed = 0
                    for r in batch_results:
                        u = r.get("url", "")
                        if u in mapping and mapping[u]:
                            new_u = mapping[u]
                            if new_u != u:
                                fixed += 1
                                if _looks_publisher(new_u):
                                    pub_fixed += 1
                                r["url"] = new_u
                    batch_debug.append(f"resolved_redirects={fixed}")
            batch_debug.append(f"publisher_urls={pub_fixed}")

            overall_debug += batch_debug

            if batch_results:
                results = batch_results[:max_results]
                provider_mix = batch_providers
                break

            # Small pause to be polite before next variant
            await asyncio.sleep(0.3)

        if not results and (time.monotonic() - start) > time_budget_sec:
            timed_out = True
            overall_debug.append("timeout_reached=1")

        provider_used = "mixed" if len(set(provider_mix)) > 1 else (provider_mix[0] if provider_mix else "none")
        return results, provider_used, overall_debug, timed_out

    # ---------------- Tavily ----------------
    async def _tavily_search(
        self, query: str, max_results: int, timeout: float
    ) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
        url = f"{TAVILY_BASE}{TAVILY_SEARCH_PATH}"
        headers = {
            "Authorization": f"Bearer {TAVILY_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": UA,
            "Accept-Language": ACCEPT_LANG,
        }
        payload = {
            "query": query,
            "search_depth": "basic",
            "max_results": max(1, min(int(max_results), 10)),
            "include_answer": False,
            "include_images": False,
        }
        dbg: List[str] = []
        try:
            async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
                r = await client.post(url, json=payload)
                dbg.append(f"tavily status={r.status_code}")
                if r.status_code != 200:
                    try:
                        dbg.append(f"tavily body={r.text[:300]}")
                    except Exception:
                        pass
                    return False, [], dbg
                data = r.json()
        except Exception as e:
            dbg.append(f"tavily exception={type(e).__name__}: {e}")
            return False, [], dbg

        items: List[Dict[str, Any]] = []
        for i, it in enumerate(data.get("results", []) or []):
            title = str(it.get("title") or "").strip()
            link = str(it.get("url") or "").strip()
            content = str(it.get("content") or "").strip()
            if not link:
                continue
            score = float(it.get("score") or (1.0 - min(i * 0.1, 0.9)))
            items.append(_norm_result(title, link, content, score, "tavily"))
        dbg.append(f"tavily results={len(items)}")
        return True, items, dbg

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
            items.append(_norm_result(title, link, snippet, score, "google-news"))

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
            items.append(_norm_result(title, link, snippet, score, "bing-news"))
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
            items.append(_norm_result(title, link, snippet, score, "yahoo-news"))
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
