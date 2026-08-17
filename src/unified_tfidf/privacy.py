from __future__ import annotations

import hashlib
import re


PATTERNS = (
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
    re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
    re.compile(r"(?i)(微信|vx|wechat|QQ|联系(?:方式|人)?|手机号|电话)\s*[:：]?\s*[A-Za-z0-9_-]{5,}"),
    re.compile(r"(订单号|单号|设备标识|序列号|IMEI)\s*[:：]?\s*[A-Za-z0-9*_-]{5,}", re.I),
)


def sanitize_text(value: object) -> tuple[str, int]:
    text = "" if value is None else str(value)
    replacements = 0
    for pattern in PATTERNS:
        text, count = pattern.subn("[已脱敏]", text)
        replacements += count
    return text, replacements


def anonymous_user(value: object) -> str:
    normalized = "" if value is None else str(value).strip()
    if not normalized:
        return ""
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"用户_{digest}"


def contains_pii(value: object) -> bool:
    text = "" if value is None else str(value)
    return any(pattern.search(text) for pattern in PATTERNS)
