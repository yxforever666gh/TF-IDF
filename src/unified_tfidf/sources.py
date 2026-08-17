from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .cleaning import clean_blackcat_text, date_from_text, finish_records, normalize_text, parse_date, parse_json
from .paths import ProjectPaths


BASE_COLUMNS = [
    "record_id", "platform", "source_type", "record_type", "source_id", "parent_id", "username",
    "review_date", "date_status", "clean_text", "product_model", "spec", "original_rating",
    "rating_source", "useful_count", "repeat_purchase", "has_image", "public_status",
]


def _readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)


def _row(**kwargs: Any) -> dict[str, Any]:
    result = {column: "" for column in BASE_COLUMNS}
    result.update(kwargs)
    return result


def read_blackcat(path: Path) -> pd.DataFrame:
    connection = _readonly(path)
    try:
        rows = pd.read_sql_query(
            "SELECT complaint_id, complaint_date, collected_at, model, title, body, demand, public_status FROM complaints",
            connection,
        ).to_dict("records")
    finally:
        connection.close()
    output = []
    for item in rows:
        body = clean_blackcat_text(item.get("body", ""))
        demand = clean_blackcat_text(item.get("demand", ""))
        text = body
        if demand and demand not in body:
            text = normalize_text(body + " " + demand)
        inferred_date = item.get("complaint_date") or date_from_text(text) or item.get("collected_at")
        review_date, date_status = parse_date(inferred_date)
        output.append(_row(
            record_id=f"blackcat:{item['complaint_id']}", platform="黑猫", source_type="投诉", record_type="投诉",
            source_id=str(item["complaint_id"]), review_date=review_date, date_status=date_status,
            clean_text=text, product_model=normalize_text(item.get("model", "")),
            public_status=normalize_text(item.get("public_status", "")), rating_source="ml",
        ))
    return pd.DataFrame(output)


def _format_spec(value: object) -> str:
    parsed = value if isinstance(value, dict) else parse_json(value)
    if parsed:
        return " ".join(f"{normalize_text(key)}:{normalize_text(item)}" for key, item in parsed.items() if normalize_text(item))
    return normalize_text(value)


def read_jd(path: Path) -> pd.DataFrame:
    connection = _readonly(path)
    try:
        comments = pd.read_sql_query(
            "SELECT comment_fp, sku, comment_id, normalized_json FROM comments", connection
        ).to_dict("records")
        replies = pd.read_sql_query(
            "SELECT reply_id, comment_fp, normalized_json FROM replies", connection
        ).to_dict("records")
    finally:
        connection.close()
    output: list[dict[str, Any]] = []
    parents: dict[str, dict[str, Any]] = {}
    for item in comments:
        value = parse_json(item["normalized_json"])
        comment_id = str(item["comment_id"])
        review_date, date_status = parse_date(value.get("comment_date"))
        parent = _row(
            record_id=f"jd:{comment_id}", platform="京东", source_type="电商", record_type="主评论",
            source_id=comment_id, username=value.get("user_nickname", ""), review_date=review_date,
            date_status=date_status, clean_text=normalize_text(value.get("content", "")),
            product_model="", spec=value.get("sku_text") or _format_spec(value.get("sku_info", {})),
            original_rating=value.get("score", ""), rating_source="original",
            useful_count=value.get("main_praise_count", 0), repeat_purchase=value.get("buy_count", 0),
            has_image=int(bool(value.get("images"))),
        )
        output.append(parent)
        parents[str(item["comment_fp"])] = parent
    for item in replies:
        parent = parents.get(str(item["comment_fp"]))
        if not parent:
            continue
        value = parse_json(item["normalized_json"])
        reply_id = str(item["reply_id"])
        parent_year = str(parent["review_date"])[:4] if parent["review_date"] else None
        review_date, date_status = parse_date(value.get("reply_date"), parent_year)
        output.append(_row(
            record_id=f"jd-reply:{parent['source_id']}:{reply_id}", platform="京东", source_type="电商",
            record_type="回复", source_id=reply_id, parent_id=parent["record_id"], username=value.get("user_nickname", ""),
            review_date=review_date, date_status=date_status, clean_text=normalize_text(value.get("content", "")),
            product_model=parent["product_model"], spec=parent["spec"], original_rating=parent["original_rating"],
            rating_source="inherited", useful_count=value.get("praise_count", 0),
        ))
    return pd.DataFrame(output)


def read_taobao(path: Path) -> pd.DataFrame:
    connection = _readonly(path)
    try:
        comments = pd.read_sql_query(
            "SELECT c.*, i.title FROM comments c JOIN items i ON c.item_id=i.item_id", connection
        ).to_dict("records")
        replies = pd.read_sql_query(
            "SELECT r.*, c.rating parent_rating, c.sku_json parent_spec, c.comment_date parent_comment_date, i.title "
            "FROM replies r JOIN comments c ON r.item_id=c.item_id AND r.target_comment_id=c.comment_id "
            "JOIN items i ON i.item_id=c.item_id",
            connection,
        ).to_dict("records")
    finally:
        connection.close()
    output: list[dict[str, Any]] = []
    for item in comments:
        comment_id = str(item["comment_id"])
        item_id = str(item["item_id"])
        review_date, date_status = parse_date(item.get("comment_date"))
        append = parse_json(item.get("append_json")).get("content", "")
        clean = normalize_text(" ".join(value for value in (item.get("content", ""), append) if value))
        output.append(_row(
            record_id=f"taobao:{item_id}:{comment_id}", platform="淘宝", source_type="电商", record_type="主评论",
            source_id=comment_id, username=item.get("nickname", ""), review_date=review_date, date_status=date_status,
            clean_text=clean, product_model=normalize_text(item.get("title", "")).replace("Apple/苹果 ", ""),
            spec=_format_spec(item.get("sku_json", "")), original_rating=item.get("rating", ""),
            rating_source="original" if item.get("rating") not in (None, "") else "ml",
            useful_count=item.get("like_count", 0), has_image=int(bool(parse_json(item.get("media_json")).get("pictures"))),
        ))
    for item in replies:
        item_id, parent_id, reply_id = map(str, (item["item_id"], item["target_comment_id"], item["reply_id"]))
        parent_date, _ = parse_date(item.get("parent_comment_date"))
        review_date, date_status = parse_date(item.get("reply_date"), parent_date[:4] if parent_date else None)
        output.append(_row(
            record_id=f"taobao-reply:{item_id}:{parent_id}:{reply_id}", platform="淘宝", source_type="电商",
            record_type="商家回复" if int(item.get("is_author") or 0) else "回复", source_id=reply_id,
            parent_id=f"taobao:{item_id}:{parent_id}", username=item.get("nickname", ""),
            review_date=review_date, date_status=date_status, clean_text=normalize_text(item.get("content", "")),
            product_model=normalize_text(item.get("title", "")).replace("Apple/苹果 ", ""),
            spec=_format_spec(item.get("parent_spec", "")), original_rating=item.get("parent_rating", ""),
            rating_source="inherited", useful_count=item.get("like_count", 0),
        ))
    return pd.DataFrame(output)


def load_all_sources(paths: ProjectPaths) -> tuple[pd.DataFrame, dict[str, str]]:
    database_paths = {
        "blackcat": paths.locate("blackcat-complaint.sqlite3"),
        "jd": paths.locate("jingdong-comment.sqlite3"),
        "taobao": paths.locate("taobao_comments.sqlite3"),
    }
    from .cleaning import file_sha256

    hashes = {name: file_sha256(path) for name, path in database_paths.items()}
    combined = pd.concat(
        [read_blackcat(database_paths["blackcat"]), read_jd(database_paths["jd"]), read_taobao(database_paths["taobao"])],
        ignore_index=True,
    )
    return finish_records(combined), hashes
