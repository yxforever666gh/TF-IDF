from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .cleaning import normalize_text, text_hash
from .paths import ProjectPaths
from .privacy import sanitize_text


PRODUCT_MAP = {
    "电池与充电": "电池与充电",
    "屏幕与显示": "屏幕与触控",
    "屏幕与触控": "屏幕与触控",
    "性能与发热": "发热与性能",
    "发热与性能": "发热与性能",
    "信号与通信": "信号与通信",
    "影像与音频": "影像与音频",
    "外观与结构": "外观与结构",
    "系统与软件": "系统与软件",
    "真伪与新旧": "正品与激活",
    "正品与激活": "正品与激活",
    "配件与兼容": "配件与兼容",
    "存储与容量": "其他产品问题",
    "综合产品体验": "其他产品问题",
    "其他产品问题": "其他产品问题",
    "无产品问题": "无产品问题",
    "无": "无产品问题",
}
SERVICE_MAP = {
    "物流与包装": "物流与包装",
    "价格与优惠": "价格与营销",
    "价格与营销": "价格与营销",
    "客服与咨询": "客服与推诿",
    "客服与推诿": "客服与推诿",
    "检测争议": "检测争议",
    "维修与保修": "维修问题",
    "维修问题": "维修问题",
    "保修争议": "保修争议",
    "退换货与退款": "退换货退款",
    "退换货退款": "退换货退款",
    "账号与数据": "账号与数据",
    "以旧换新与回收": "以旧换新与回收",
    "无服务问题": "无服务问题",
    "无": "无服务问题",
}
SENTIMENT_MAP = {
    "正向": "正向", "中性/无法判断": "中性或无法判断", "褒贬混合": "混合",
    "混合": "混合", "负向": "负向",
}
RESOLUTION_MAP = {
    "不适用": "不适用", "无法判断": "无法判断", "处理中": "处理中",
    "未解决": "未解决", "未解决/仍存在": "未解决", "存在但可接受": "部分解决",
    "部分解决": "部分解决", "已解决": "已解决", "已解决/已缓解": "已解决",
}
REQUEST_MAP = {
    "无明确诉求": "无明确诉求", "解释与检测": "解释与检测", "免费维修": "维修相关诉求",
    "付费维修争议": "维修相关诉求", "更换设备": "更换设备", "退款退货": "退款退货",
    "赔偿补偿": "赔偿补偿", "道歉整改": "道歉整改", "加快处理": "加快处理", "其他诉求": "其他诉求",
}
STAGE_MAP = {
    "购买决策": "购买决策", "收货验机": "收货验机", "使用阶段": "使用阶段",
    "售后/权益处理": "售后处理", "无法判断": "无法判断",
}
SPLIT_RE = re.compile(r"[；;、|,/]+")


def _find_master(path: Path, identifier: str, expected_rows: int) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    candidates: list[pd.DataFrame] = []
    for sheet in workbook.sheet_names:
        preview = pd.read_excel(path, sheet_name=sheet, header=None, nrows=8, dtype=str)
        for header_index, row in preview.iterrows():
            values = {str(value).strip() for value in row.tolist() if pd.notna(value)}
            if identifier not in values:
                continue
            frame = pd.read_excel(path, sheet_name=sheet, header=int(header_index), dtype=str).dropna(how="all")
            if identifier in frame.columns and len(frame) == expected_rows:
                candidates.append(frame)
    if not candidates:
        raise ValueError(f"unable to find {expected_rows}-row master sheet in {path}")
    # Some workbooks intentionally duplicate the cumulative master as an editable template.
    return candidates[-1].copy()


def _split_labels(value: object, mapping: dict[str, str], default: str) -> str:
    values: list[str] = []
    for raw in SPLIT_RE.split("" if pd.isna(value) else str(value)):
        label = mapping.get(raw.strip())
        if label and label not in values:
            values.append(label)
    return ";".join(values or [default])


def _safe_text(value: object) -> str:
    text, _ = sanitize_text(normalize_text(value))
    return text


def read_jd_annotations(path: Path) -> pd.DataFrame:
    source = _find_master(path, "comment_id", 1271)
    rows = []
    for item in source.to_dict("records"):
        text = _safe_text(item.get("原始评论", ""))
        append = _safe_text(item.get("追评内容", ""))
        if append and append not in {"0", "1"}:
            text = normalize_text(text + " " + append)
        valid = "有效" if item.get("有效性") == "有效" else "低信息或模板"
        rows.append({
            "record_id": f"jd:{item['comment_id']}", "platform": "京东", "clean_text": text,
            "model_text": build_model_text("京东", text, item.get("评分", ""), item.get("规格", ""), ""),
            "text_hash": text_hash(text), "duplicate_group_id": "dup_" + text_hash(text)[:16],
            "content_validity": valid, "product_scope": "iPhone16",
            "sentiment": SENTIMENT_MAP.get(str(item.get("整体情绪", "")), ""),
            "product_topics": _split_labels(item.get("产品主题（多选）"), PRODUCT_MAP, "无产品问题"),
            "service_topics": _split_labels(item.get("服务主题（多选）"), SERVICE_MAP, "无服务问题"),
            "severity": str(item.get("问题严重度", "")),
            "resolution": RESOLUTION_MAP.get(str(item.get("问题状态", "")), ""),
            "user_request": "", "purchase_stage": STAGE_MAP.get(str(item.get("购买使用阶段", "")), ""),
            "mask_content_validity": 1, "mask_product_scope": 1, "mask_sentiment": 1,
            "mask_product_topics": int(valid == "有效"), "mask_service_topics": int(valid == "有效"),
            "mask_severity": 1, "mask_resolution": 1, "mask_user_request": 0,
            "mask_purchase_stage": int(valid == "有效"),
        })
    return pd.DataFrame(rows)


def _scope(value: object) -> str:
    return {"iPhone16": "iPhone16", "其他3C": "其他3C", "无法确认": "无法确认", "无关": "无关"}.get(str(value), "无法确认")


def read_blackcat_annotations(path: Path) -> pd.DataFrame:
    source = _find_master(path, "complaint_id", 1502)
    rows = []
    for item in source.to_dict("records"):
        text = _safe_text(item.get("清洗后投诉文本", ""))
        duplicate_target = "" if pd.isna(item.get("重复指向")) else str(item.get("重复指向", "")).strip()
        group = f"blackcat-ref:{duplicate_target}" if duplicate_target else "dup_" + text_hash(text)[:16]
        included = str(item.get("是否纳入分析", "")) == "是"
        product_values = _split_labels(item.get("主要产品问题"), PRODUCT_MAP, "无产品问题")
        secondary = _split_labels(item.get("次要产品问题"), PRODUCT_MAP, "无产品问题")
        product_values = ";".join(dict.fromkeys((product_values + ";" + secondary).split(";")))
        if "无产品问题" in product_values.split(";") and len(product_values.split(";")) > 1:
            product_values = ";".join(value for value in product_values.split(";") if value != "无产品问题")
        rows.append({
            "record_id": f"blackcat:{item['complaint_id']}", "platform": "黑猫", "clean_text": text,
            "model_text": build_model_text("黑猫", text, "", "", item.get("原始平台状态", "")),
            "text_hash": text_hash(text), "duplicate_group_id": group,
            "content_validity": "有效" if included else "低信息或模板",
            "product_scope": _scope(item.get("产品范围")),
            "sentiment": SENTIMENT_MAP.get(str(item.get("情绪", "")), ""),
            "product_topics": product_values,
            "service_topics": _split_labels(item.get("售后服务问题"), SERVICE_MAP, "无服务问题"),
            "severity": str(item.get("严重度", "")),
            "resolution": RESOLUTION_MAP.get(str(item.get("解决结果", "")), ""),
            "user_request": REQUEST_MAP.get(str(item.get("用户诉求", "")), "其他诉求"),
            "purchase_stage": "",
            "mask_content_validity": 1, "mask_product_scope": 1, "mask_sentiment": 1,
            "mask_product_topics": int(included), "mask_service_topics": int(included),
            "mask_severity": 1, "mask_resolution": 1, "mask_user_request": int(included),
            "mask_purchase_stage": 0,
        })
    return pd.DataFrame(rows)


def build_model_text(platform: object, text: object, rating: object, spec: object, status: object) -> str:
    pieces = [f"【平台】{normalize_text(platform)}"]
    if normalize_text(rating):
        pieces.append(f"【评分】{normalize_text(rating)}")
    if normalize_text(spec):
        pieces.append(f"【规格】{normalize_text(spec)}")
    if normalize_text(status):
        pieces.append(f"【状态】{normalize_text(status)}")
    pieces.append(f"【文本】{normalize_text(text)}")
    return "\n".join(pieces)


def load_annotations(paths: ProjectPaths) -> pd.DataFrame:
    jd = read_jd_annotations(paths.locate("*1271条.xlsx"))
    blackcat = read_blackcat_annotations(paths.locate("*1502条.xlsx"))
    output = pd.concat([jd, blackcat], ignore_index=True)
    if output["record_id"].duplicated().any():
        raise ValueError("duplicate annotated record IDs")
    return output
