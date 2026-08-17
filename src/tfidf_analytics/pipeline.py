from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from unified_tfidf.annotations import build_model_text, load_annotations
from unified_tfidf.cleaning import file_sha256
from unified_tfidf.dataset import assert_no_leakage, assign_group_splits
from unified_tfidf.privacy import contains_pii
from unified_tfidf.sources import load_all_sources

from .domains import DOMAIN_TASKS, QUALITY_GATES, select_domain_annotations, select_domain_records, task_specs
from .model import evaluate_domain, predict_domain, train_domain
from .paths import ProjectPaths


EXPECTED_PLATFORM_COUNTS = {"黑猫": 6391, "京东": 2444, "淘宝": 3934}
EXPECTED_RECORD_TYPES = {"投诉": 6391, "主评论": 3671, "回复": 2392, "商家回复": 315}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def _add_model_text(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["model_text"] = [
        build_model_text(row.platform, row.clean_text, row.original_rating, row.spec, row.public_status)
        for row in output.itertuples(index=False)
    ]
    return output


def prepare(paths: ProjectPaths, domain: str) -> dict[str, Any]:
    paths.ensure_runtime_dirs(domain)
    all_records, source_hashes = load_all_sources(paths)
    platform_counts = {str(key): int(value) for key, value in all_records["platform"].value_counts().items()}
    record_counts = {str(key): int(value) for key, value in all_records["record_type"].value_counts().items()}
    if platform_counts != EXPECTED_PLATFORM_COUNTS:
        raise AssertionError(f"unexpected source counts: {platform_counts}")
    if record_counts != EXPECTED_RECORD_TYPES:
        raise AssertionError(f"unexpected record type counts: {record_counts}")

    records = _add_model_text(select_domain_records(all_records, domain))
    annotations = assign_group_splits(
        select_domain_annotations(load_annotations(paths), domain),
        seed=int(paths.settings.get("random_seed", 42)),
    )
    assert_no_leakage(annotations)
    expected_annotations = 1502 if domain == "complaint" else 1271
    if len(annotations) != expected_annotations:
        raise AssertionError(f"expected {expected_annotations} {domain} annotations, got {len(annotations)}")

    _write_csv(records, paths.domain_prepared(domain))
    _write_csv(records, paths.domain_output(domain) / "cleaned_records.csv")
    _write_csv(annotations, paths.domain_annotations(domain))

    database_paths = {
        "blackcat": paths.locate("blackcat-complaint.sqlite3"),
        "jd": paths.locate("jingdong-comment.sqlite3"),
        "taobao": paths.locate("taobao_comments.sqlite3"),
    }
    after_hashes = {name: file_sha256(path) for name, path in database_paths.items()}
    if source_hashes != after_hashes:
        raise RuntimeError("source SQLite changed during read-only preparation")

    audit = {
        "domain": domain,
        "source_rows": int(len(records)),
        "platform_counts": {str(key): int(value) for key, value in records["platform"].value_counts().items()},
        "record_type_counts": {str(key): int(value) for key, value in records["record_type"].value_counts().items()},
        "annotation_rows": int(len(annotations)),
        "content_status": {str(key): int(value) for key, value in records["content_status"].value_counts().items()},
        "scope_status": {str(key): int(value) for key, value in records["scope_status"].value_counts().items()},
        "pii_replacements": int(pd.to_numeric(records["pii_replacements"], errors="coerce").fillna(0).sum()),
        "source_sha256": source_hashes,
        "source_hashes_unchanged": True,
    }
    _write_json(audit, paths.domain_output(domain) / "preparation_report.json")
    return audit


def train(paths: ProjectPaths, domain: str) -> dict[str, Any]:
    if not paths.domain_annotations(domain).exists():
        prepare(paths, domain)
    annotations = _read_csv(paths.domain_annotations(domain))
    for column in annotations.columns:
        if column.startswith("mask_"):
            annotations[column] = pd.to_numeric(annotations[column], errors="coerce").fillna(0).astype(int)
    specs = task_specs(paths.task_specs, domain)
    bundle, report, assigned = train_domain(
        annotations,
        specs,
        domain,
        seed=int(paths.settings.get("random_seed", 42)),
    )
    assert_no_leakage(assigned)
    _write_csv(assigned, paths.domain_annotations(domain))
    joblib.dump(bundle, paths.domain_model(domain), compress=3)
    _write_json(report, paths.domain_output(domain) / "training_report.json")
    return report


def _quality(domain: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for task, minimum in QUALITY_GATES[domain].items():
        item = evaluation["tasks"][task]
        reasons: list[str] = []
        if item["macro_f1"] < minimum:
            reasons.append(f"macro_f1 {item['macro_f1']:.4f} < {minimum:.2f}")
        if domain == "ecommerce" and task == "sentiment":
            low_support = {label: count for label, count in item["support"].items() if count < 20}
            if low_support:
                reasons.append(f"test support < 20: {low_support}")
        passed = not reasons
        output[task] = {
            "status": "approved" if passed else "baseline_only",
            "passed": passed,
            "minimum_macro_f1": minimum,
            "macro_f1": item["macro_f1"],
            "reasons": reasons,
        }
    return output


def evaluate(paths: ProjectPaths, domain: str) -> dict[str, Any]:
    if not paths.domain_model(domain).exists():
        train(paths, domain)
    bundle = joblib.load(paths.domain_model(domain))
    assigned = _read_csv(paths.domain_annotations(domain))
    for column in assigned.columns:
        if column.startswith("mask_"):
            assigned[column] = pd.to_numeric(assigned[column], errors="coerce").fillna(0).astype(int)
    report = evaluate_domain(bundle, assigned, task_specs(paths.task_specs, domain))
    report["quality_gates"] = _quality(domain, report)
    bundle["quality"] = report["quality_gates"]
    joblib.dump(bundle, paths.domain_model(domain), compress=3)
    _write_json(report, paths.domain_output(domain) / "evaluation_report.json")
    training = json.loads((paths.domain_output(domain) / "training_report.json").read_text(encoding="utf-8"))
    _write_model_card(paths.domain_output(domain) / "model_card.md", domain, training, report)
    return report


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1"})


def _review_queue(frame: pd.DataFrame, domain: str, tasks: list[str]) -> pd.DataFrame:
    reasons: list[str] = []
    for row in frame.itertuples(index=False):
        current: list[str] = []
        if row.content_status != "有效":
            current.append("content_quality")
        if row.date_status != "有效":
            current.append("date")
        if int(row.pii_replacements or 0) > 0:
            current.append("pii_redacted")
        if domain == "complaint" and row.scope_status != "in_scope":
            current.append(f"scope_{row.scope_status}")
        if domain == "ecommerce" and row.record_type != "主评论":
            current.append("reply_excluded_from_kpi")
        if domain == "ecommerce" and row.prediction_domain == "domain_transfer":
            current.append("domain_transfer")
        for task in tasks:
            if float(getattr(row, f"confidence_{task}")) < 0.55:
                current.append(f"low_confidence_{task}")
        reasons.append(";".join(dict.fromkeys(current)))
    output = frame.copy()
    output["review_reason"] = reasons
    return output.loc[output["review_reason"].ne("")].copy()


def predict(paths: ProjectPaths, domain: str, input_path: Path | None = None, output_path: Path | None = None) -> dict[str, Any]:
    if not paths.domain_model(domain).exists():
        train(paths, domain)
        evaluate(paths, domain)
    bundle = joblib.load(paths.domain_model(domain))
    if not bundle.get("quality"):
        evaluate(paths, domain)
        bundle = joblib.load(paths.domain_model(domain))
    if input_path is None:
        if not paths.domain_prepared(domain).exists():
            prepare(paths, domain)
        records = _read_csv(paths.domain_prepared(domain))
    else:
        records = _read_csv(input_path)
        required = {"clean_text"}
        missing = required - set(records.columns)
        if missing:
            raise ValueError(f"input CSV missing columns: {sorted(missing)}")
        records = records.copy()
        for column, default in {
            "platform": "黑猫" if domain == "complaint" else "京东",
            "original_rating": "", "spec": "", "public_status": "", "record_type": "投诉" if domain == "complaint" else "主评论",
        }.items():
            if column not in records:
                records[column] = default
        if "model_text" not in records:
            records = _add_model_text(records)
    predictions = predict_domain(bundle, records["model_text"])
    all_predictions = pd.concat([records.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
    for task in DOMAIN_TASKS[domain]:
        status = bundle["quality"][task]["status"]
        all_predictions[f"quality_{task}"] = status
        all_predictions[f"approved_{task}"] = (
            all_predictions[f"predicted_{task}"] if status == "approved" else ""
        )

    if input_path is not None:
        destination = output_path or input_path.with_name(input_path.stem + "_predictions.csv")
        _write_csv(all_predictions, destination)
        return {"domain": domain, "rows": int(len(all_predictions)), "output": str(destination)}

    _write_csv(all_predictions, paths.domain_output(domain) / "all_predictions.csv")
    primary = _truthy(all_predictions["is_primary_record"])
    eligible = primary & all_predictions["content_status"].eq("有效")
    if domain == "complaint":
        eligible &= all_predictions["record_type"].eq("投诉") & all_predictions["scope_status"].eq("in_scope")
    else:
        eligible &= all_predictions["record_type"].eq("主评论")
    analysis_ready = all_predictions.loc[eligible].copy()
    _write_csv(analysis_ready, paths.domain_output(domain) / "analysis_ready.csv")
    review_queue = _review_queue(all_predictions, domain, DOMAIN_TASKS[domain])
    _write_csv(review_queue, paths.domain_output(domain) / "review_queue.csv")
    return {
        "domain": domain,
        "all_predictions": int(len(all_predictions)),
        "analysis_ready": int(len(analysis_ready)),
        "review_queue": int(len(review_queue)),
        "quality": bundle["quality"],
    }


def _write_model_card(path: Path, domain: str, training: dict[str, Any], evaluation: dict[str, Any]) -> None:
    title = "投诉分析" if domain == "complaint" else "电商评论分析"
    limitations = (
        "仅适用于规则确认属于 iPhone 16 的黑猫投诉；无法确认和旧型号记录进入复核队列。"
        if domain == "complaint"
        else "仅京东人工标注参与训练；淘宝输出属于 domain_transfer，回复不进入默认评论 KPI。"
    )
    lines = [
        f"# {title} TF-IDF 模型卡",
        "",
        "## 用途与数据",
        "",
        f"- 人工标注数：{training['annotation_rows']}。确定性 70/15/15 重复组划分，随机种子 42。",
        "- 候选特征：字符 TF-IDF、词级 TF-IDF、二者组合。模型仅使用验证集选型，冻结测试集只评估一次。",
        f"- 平台限制：{limitations}",
        "",
        "## 冻结测试集结果",
        "",
        "| 任务 | Macro F1 | 门槛 | 状态 | 测试支持数 |",
        "|---|---:|---:|---|---|",
    ]
    for task, item in evaluation["tasks"].items():
        quality = evaluation["quality_gates"][task]
        support = ", ".join(f"{label}:{count}" for label, count in item["support"].items())
        lines.append(
            f"| {task} | {item['macro_f1']:.4f} | {quality['minimum_macro_f1']:.2f} | {quality['status']} | {support} |"
        )
    lines.extend([
        "",
        "## 类别召回率与混淆矩阵",
        "",
        "完整的逐类别 precision/recall/F1、支持数、混淆矩阵及多标签阈值见同目录 `evaluation_report.json`。",
        "",
        "## 使用限制",
        "",
        "`baseline_only` 任务只保留在 `all_predictions.csv`，不会填入 `analysis_ready.csv` 的 `approved_*` 字段。",
        "模型不能替代人工投诉处置、产品质量鉴定或法律判断。输入分布变化后应重新评估。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def verify(paths: ProjectPaths, domains: tuple[str, ...] = ("complaint", "ecommerce")) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "passed", "domains": {}}
    for domain in domains:
        output_dir = paths.domain_output(domain)
        required = [
            "cleaned_records.csv", "all_predictions.csv", "analysis_ready.csv", "review_queue.csv",
            "training_report.json", "evaluation_report.json", "model_card.md",
        ]
        missing = [name for name in required if not (output_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"{domain} missing outputs: {missing}")
        cleaned = _read_csv(output_dir / "cleaned_records.csv")
        predictions = _read_csv(output_dir / "all_predictions.csv")
        analysis = _read_csv(output_dir / "analysis_ready.csv")
        if cleaned["record_id"].duplicated().any():
            raise AssertionError(f"{domain}: duplicate record_id")
        if any(contains_pii(value) for value in cleaned["clean_text"]):
            raise AssertionError(f"{domain}: residual PII in clean_text")
        if domain == "complaint" and not analysis["scope_status"].eq("in_scope").all():
            raise AssertionError("complaint analysis contains out-of-scope records")
        if domain == "ecommerce":
            if not analysis["record_type"].eq("主评论").all():
                raise AssertionError("ecommerce analysis contains replies")
            original_missing = predictions["original_rating"].eq("")
            if not predictions.loc[original_missing, "rating_sentiment"].eq("").all():
                raise AssertionError("rating sentiment synthesized without original rating")
        evaluation = json.loads((output_dir / "evaluation_report.json").read_text(encoding="utf-8"))
        for task, quality in evaluation["quality_gates"].items():
            if quality["status"] == "baseline_only" and not analysis[f"approved_{task}"].eq("").all():
                raise AssertionError(f"{domain}/{task}: baseline predictions leaked into approved field")
        assigned = _read_csv(paths.domain_annotations(domain))
        assert_no_leakage(assigned)
        report["domains"][domain] = {
            "cleaned_rows": int(len(cleaned)),
            "prediction_rows": int(len(predictions)),
            "analysis_rows": int(len(analysis)),
            "quality": evaluation["quality_gates"],
        }
    _write_json(report, paths.output / "verification_report.json")
    return report


def run_all(paths: ProjectPaths, domain: str) -> dict[str, Any]:
    return {
        "prepare": prepare(paths, domain),
        "train": train(paths, domain),
        "evaluate": evaluate(paths, domain),
        "predict": predict(paths, domain),
    }
