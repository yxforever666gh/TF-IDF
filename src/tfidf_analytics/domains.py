from __future__ import annotations

import re

import pandas as pd


DOMAIN_TASKS = {
    "complaint": ["product_topics", "service_topics", "severity", "resolution", "user_request"],
    "ecommerce": ["sentiment", "product_topics", "service_topics"],
}

QUALITY_GATES = {
    "complaint": {
        "product_topics": 0.65,
        "service_topics": 0.60,
        "severity": 0.60,
        "resolution": 0.60,
        "user_request": 0.55,
    },
    "ecommerce": {"sentiment": 0.70, "product_topics": 0.65, "service_topics": 0.60},
}

OLDER_IPHONE_RE = re.compile(
    r"(?:iphone|苹果(?:手机)?)[\s-]*(?:15|14|13|12|11|x(?:s|r)?|8|7|6)(?:\s*(?:pro|max|plus|mini))?",
    re.IGNORECASE,
)
IPHONE16_RE = re.compile(
    r"(?:iphone|苹果(?:手机)?)[\s-]*16(?:\s*(?:pro|max|plus))?",
    re.IGNORECASE,
)


def task_specs(all_specs: dict, domain: str) -> dict:
    return {name: all_specs[name] for name in DOMAIN_TASKS[domain]}


def select_domain_records(frame: pd.DataFrame, domain: str) -> pd.DataFrame:
    if domain == "complaint":
        output = frame.loc[frame["platform"].eq("黑猫")].copy()
        output["scope_status"] = output.apply(
            lambda row: complaint_scope(row.get("clean_text", ""), row.get("product_model", "")), axis=1
        )
        output["prediction_domain"] = "supervised_domain"
    elif domain == "ecommerce":
        output = frame.loc[frame["platform"].isin(["京东", "淘宝"])].copy()
        output["scope_status"] = "in_scope"
        output["prediction_domain"] = output["platform"].map(
            {"京东": "supervised_domain", "淘宝": "domain_transfer"}
        )
        output["rating_sentiment"] = output["original_rating"].map(rating_sentiment)
        output["sentiment_source"] = output["original_rating"].map(
            lambda value: "original_rating" if rating_sentiment(value) else ""
        )
    else:
        raise ValueError(f"unknown domain: {domain}")
    return output.reset_index(drop=True)


def select_domain_annotations(frame: pd.DataFrame, domain: str) -> pd.DataFrame:
    platform = "黑猫" if domain == "complaint" else "京东"
    return frame.loc[frame["platform"].eq(platform)].reset_index(drop=True)


def complaint_scope(text: object, product_model: object = "") -> str:
    body = str(text or "")
    model = str(product_model or "")
    if IPHONE16_RE.search(body):
        return "in_scope"
    if OLDER_IPHONE_RE.search(body):
        return "out_of_scope"
    if IPHONE16_RE.search(model):
        return "in_scope"
    return "review"


def rating_sentiment(value: object) -> str:
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return ""
    if rating >= 4:
        return "正向"
    if rating >= 3:
        return "中性或无法判断"
    if rating >= 1:
        return "负向"
    return ""
