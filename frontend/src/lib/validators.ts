export type PwdLevel = "invalid" | "weak" | "medium" | "strong";

export function validateEmail(email: string): boolean {
    if (!email || email.length > 254) return false;
    // must start with a letter; common-lenient but robust
    const re = /^[A-Za-z][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,24}$/;
    return re.test(email);
}

/**
 * Assess password strength per your spec.
 * - 8..20 chars, ≥1 uppercase, ≥1 digit => at least "weak"
 * - score bumps: len>=12, has special, has both lower+upper, >=2 digits, >=2 uppercase
 * Mapping:
 *   invalid -> score 0, weak -> 1, medium -> 2, strong -> >=3
 */
export function assessPassword(pw: string): {
    level: PwdLevel;
    score: number;     // 0..5 (internal), used to compute percent
    percent: number;   // 0, 33, 66, 100 (for the bar)
    label: string;     // i18n-friendly label
} {
    const len = pw.length;
    const hasUpper = /[A-Z]/.test(pw);
    const hasLower = /[a-z]/.test(pw);
    const digits = (pw.match(/\d/g) || []).length;
    const uppers = (pw.match(/[A-Z]/g) || []).length;
    const hasDigit = digits >= 1;
    const hasSpecial = /[^A-Za-z0-9]/.test(pw);

    const meetsMin = len >= 8 && len <= 20 && hasUpper && hasDigit;

    if (!meetsMin) {
        return { level: "invalid", score: 0, percent: 0, label: "Invalid password" };
    }

    // start at weak
    let score = 1;

    if (len >= 12) score++;
    if (hasSpecial) score++;
    if (hasLower && hasUpper) score++;
    if (digits >= 2) score++;
    if (uppers >= 2) score++;

    let level: PwdLevel = "weak";
    if (score >= 3) level = "medium";
    if (score >= 4) level = "strong";

    const percent = level === "weak" ? 33 : level === "medium" ? 66 : 100;
    const label =
        level === "weak" ? "Weak password" :
            level === "medium" ? "Medium password" : "Strong password";

    return { level, score, percent, label };
}
