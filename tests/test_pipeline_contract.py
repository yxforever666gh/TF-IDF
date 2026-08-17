from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import pytest

from tfidf_analytics.domains import complaint_scope, rating_sentiment
from tfidf_analytics.model import evaluate_domain, predict_domain, train_domain
from tfidf_analytics.paths import ProjectPaths
from unified_tfidf.cleaning import file_sha256
from unified_tfidf.dataset import assert_no_leakage
from unified_tfidf.privacy import contains_pii


def _paths() -> ProjectPaths:
    return ProjectPaths.create()


LOCAL_DATA = ProjectPaths.create().data_root.exists()
requires_local_data = pytest.mark.skipif(not LOCAL_DATA, reason="private raw_data is not present")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def _count(path: Path, table: str) -> int:
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        return int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    finally:
        connection.close()


@requires_local_data
def test_source_row_contracts() -> None:
    paths = _paths()
    assert _count(paths.locate("blackcat-complaint.sqlite3"), "complaints") == 6391
    jd = paths.locate("jingdong-comment.sqlite3")
    taobao = paths.locate("taobao_comments.sqlite3")
    assert (_count(jd, "comments"), _count(jd, "replies")) == (1271, 1173)
    assert (_count(taobao, "comments"), _count(taobao, "replies")) == (2400, 1534)


def test_scope_and_rating_rules() -> None:
    assert complaint_scope("iPhone 16 Pro 屏幕故障") == "in_scope"
    assert complaint_scope("iPhone 15 Pro 电池问题") == "out_of_scope"
    assert complaint_scope("苹果手机售后问题") == "review"
    assert rating_sentiment(5) == "正向"
    assert rating_sentiment(3) == "中性或无法判断"
    assert rating_sentiment(1) == "负向"
    assert rating_sentiment("") == ""


@requires_local_data
def test_domain_outputs_and_quality_gates() -> None:
    paths = _paths()
    expected = {"complaint": 6391, "ecommerce": 6378}
    for domain, rows in expected.items():
        output = paths.domain_output(domain)
        cleaned = _read(output / "cleaned_records.csv")
        predictions = _read(output / "all_predictions.csv")
        analysis = _read(output / "analysis_ready.csv")
        assert len(cleaned) == rows
        assert len(predictions) == rows
        assert cleaned["record_id"].is_unique
        assert not any(contains_pii(text) for text in cleaned["clean_text"])
        assert_no_leakage(_read(paths.domain_annotations(domain)))
        evaluation = json.loads((output / "evaluation_report.json").read_text(encoding="utf-8"))
        for task, gate in evaluation["quality_gates"].items():
            if gate["status"] == "baseline_only":
                assert analysis[f"approved_{task}"].eq("").all()
        if domain == "complaint":
            assert analysis["scope_status"].eq("in_scope").all()
        else:
            assert analysis["record_type"].eq("主评论").all()
            missing = predictions["original_rating"].eq("")
            assert predictions.loc[missing, "rating_sentiment"].eq("").all()


@requires_local_data
def test_predictions_are_deterministic() -> None:
    paths = _paths()
    for domain in ("complaint", "ecommerce"):
        bundle = joblib.load(paths.domain_model(domain))
        sample = _read(paths.domain_prepared(domain)).head(8)
        first = predict_domain(bundle, sample["model_text"])
        second = predict_domain(bundle, sample["model_text"])
        pd.testing.assert_frame_equal(first, second, check_exact=True)


@requires_local_data
def test_source_hashes_match_preparation_reports() -> None:
    paths = _paths()
    current = {
        "blackcat": file_sha256(paths.locate("blackcat-complaint.sqlite3")),
        "jd": file_sha256(paths.locate("jingdong-comment.sqlite3")),
        "taobao": file_sha256(paths.locate("taobao_comments.sqlite3")),
    }
    for domain in ("complaint", "ecommerce"):
        audit = json.loads(
            (paths.domain_output(domain) / "preparation_report.json").read_text(encoding="utf-8")
        )
        assert audit["source_sha256"] == current
        assert audit["source_hashes_unchanged"] is True


def test_synthetic_training_and_prediction_end_to_end() -> None:
    rows = []
    for index in range(120):
        label = str(index % 2)
        rows.append({
            "record_id": f"synthetic:{index}",
            "text_hash": f"hash-{index}",
            "duplicate_group_id": f"group-{index}",
            "model_text": f"样例类别{label} 文本编号{index}",
            "severity": label,
            "mask_severity": 1,
        })
    annotations = pd.DataFrame(rows)
    specs = {"severity": {"kind": "single", "labels": ["0", "1"]}}
    bundle, training, assigned = train_domain(annotations, specs, "complaint", seed=42)
    evaluation = evaluate_domain(bundle, assigned, specs)
    predicted = predict_domain(bundle, annotations["model_text"].head(5))
    assert training["annotation_rows"] == 120
    assert 0 <= evaluation["tasks"]["severity"]["macro_f1"] <= 1
    assert predicted["predicted_severity"].isin(["0", "1"]).all()
