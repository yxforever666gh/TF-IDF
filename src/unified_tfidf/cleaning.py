from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .privacy import anonymous_user, sanitize_text


DEFAULT_COMMENTS = {
    "该用户未填写评价内容",
    "该用户觉得商品非常好，给出5星好评",
    "该用户觉得商品还不错",
}
BLACKCAT_PATTERNS = (
    re.compile(
        r"您投诉的商家\s+.{1,40}?\s+未匹配成功[，,]?\s*我们将尽快帮您寻找商家[，,]\s*"
        r"推进其处理投诉[，,]\s*请保持手机畅通并留意官方消息推送[。.]?"
    ),
    re.compile(r"已分配商家(?:\s+|[:：]\s*)?(?:Apple(?:支持|客服|官方)?|苹果(?:支持|客服|官方)?|京东(?:客服|官方|自营)?|天猫(?:客服|官方)?|淘宝(?:客服|官方)?)?"),
    re.compile(r"距离商家最新回复已经\d+个自然日.*?系统已自动变更为已完成状态。?"),
    re.compile(r"您好[，,]您反馈的问题我们已经收到.*?请耐心等待[。.]?"),
    re.compile(r"您好[，,]您的问题已受理[，,]很抱歉给您带来不便[。.]?"),
)
SPACE_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]+>")
NORMALIZE_HASH_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_json(value: object) -> dict[str, Any]:
    try:
        parsed = json.loads("" if value is None else str(value))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", html.unescape(text))
    text = TAG_RE.sub(" ", text).replace("\u200b", " ").replace("\ufeff", " ")
    return SPACE_RE.sub(" ", text).strip()


def clean_blackcat_text(value: object) -> str:
    text = normalize_text(value).replace("已隐藏内容", " ")
    for pattern in BLACKCAT_PATTERNS:
        text = pattern.sub(" ", text)
    return normalize_text(text).lstrip("，。！？、.;；：: ")


def text_hash(value: object) -> str:
    normalized = NORMALIZE_HASH_RE.sub("", normalize_text(value)).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def content_status(text: str) -> str:
    compact = NORMALIZE_HASH_RE.sub("", text)
    if not compact:
        return "空文本"
    if text in DEFAULT_COMMENTS:
        return "平台模板"
    if len(compact) < 10:
        return "低信息"
    return "有效"


def parse_date(value: object, fallback_year: str | int | None = None) -> tuple[str, str]:
    text = normalize_text(value)
    text = re.sub(r"(\d{4})年(\d{1,2})月(\d{1,2})日", r"\1-\2-\3", text)
    if fallback_year and re.fullmatch(r"\d{1,2}-\d{1,2}(?:\s+\d{1,2}:\d{2})?", text):
        text = f"{fallback_year}-{text}"
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return "", "无法解析"
    parsed_date = parsed.date()
    return parsed_date.isoformat(), "日期异常" if parsed_date > date.today() else "有效"


def date_from_text(value: object) -> str:
    text = normalize_text(value)
    patterns = (
        r"(20\d{2})年(\d{1,2})月(\d{1,2})日?",
        r"(20\d{2})[./-](\d{1,2})[./-](\d{1,2})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            year, month, day = map(int, match.groups())
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                continue
    return ""


def finish_records(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    clean_values: list[str] = []
    replacement_counts: list[int] = []
    for value in output["clean_text"]:
        sanitized, count = sanitize_text(value)
        clean_values.append(normalize_text(sanitized))
        replacement_counts.append(count)
    output["clean_text"] = clean_values
    output["pii_replacements"] = replacement_counts
    output["username"] = output["username"].map(anonymous_user)
    output["content_status"] = output["clean_text"].map(content_status)
    output["text_hash"] = output["clean_text"].map(text_hash)
    output["duplicate_group_id"] = "dup_" + output["text_hash"].str[:16]
    output = output.sort_values("record_id", kind="stable").reset_index(drop=True)
    output["is_primary_record"] = ~output["text_hash"].duplicated()
    output["prediction_scope"] = output["record_type"].map(
        lambda value: "正式" if value in {"投诉", "主评论"} else "探索性"
    )
    if output["record_id"].duplicated().any():
        duplicates = output.loc[output["record_id"].duplicated(), "record_id"].tolist()[:5]
        raise ValueError(f"duplicate record_id values: {duplicates}")
    return output
