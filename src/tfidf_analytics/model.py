from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, multilabel_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from unified_tfidf.dataset import SPLITS, assert_no_leakage, assign_group_splits


def _vectorizer(name: str, rows: int):
    min_df = 1 if rows < 100 else 2
    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=min_df,
        max_df=0.995,
        max_features=60000,
        sublinear_tf=True,
    )
    word = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.995,
        max_features=40000,
        sublinear_tf=True,
    )
    if name == "char":
        return char
    if name == "word":
        return word
    if name == "char_word":
        return FeatureUnion([("char", char), ("word", word)])
    raise ValueError(name)


def _single_estimator(name: str, seed: int):
    if name == "linear_svc":
        return LinearSVC(class_weight="balanced", random_state=seed, dual="auto")
    if name == "sgd_logistic":
        return SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            max_iter=3000,
            tol=1e-4,
            random_state=seed,
        )
    raise ValueError(name)


def _multi_estimator(seed: int):
    base = SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        max_iter=3000,
        tol=1e-4,
        random_state=seed,
    )
    return OneVsRestClassifier(base)


def _probabilities(estimator, matrix) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        values = estimator.predict_proba(matrix)
        if values.ndim == 1:
            return np.column_stack([1 - values, values])
        return values
    scores = estimator.decision_function(matrix)
    if np.ndim(scores) == 1:
        scores = np.column_stack([-scores, scores])
    return softmax(scores, axis=1)


def _split_multi(value: object) -> list[str]:
    return [item for item in str(value or "").split(";") if item]


def _tune_thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> list[float]:
    thresholds: list[float] = []
    for index in range(y_true.shape[1]):
        best_threshold, best_score = 0.5, -1.0
        for threshold in np.arange(0.20, 0.81, 0.05):
            score = f1_score(y_true[:, index], probabilities[:, index] >= threshold, zero_division=0)
            if score > best_score:
                best_threshold, best_score = float(round(threshold, 2)), float(score)
        thresholds.append(best_threshold)
    return thresholds


def _ensure_one_label(binary: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    output = binary.copy()
    empty = np.where(output.sum(axis=1) == 0)[0]
    if len(empty):
        output[empty, probabilities[empty].argmax(axis=1)] = 1
    return output


def train_domain(
    annotations: pd.DataFrame,
    specs: dict,
    domain: str,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    assigned = assign_group_splits(annotations, seed=seed)
    assert_no_leakage(assigned)
    bundle: dict[str, Any] = {
        "schema_version": "0.1.0",
        "domain": domain,
        "seed": seed,
        "tasks": {},
        "quality": {},
    }
    report: dict[str, Any] = {
        "domain": domain,
        "seed": seed,
        "annotation_rows": int(len(assigned)),
        "split_counts": {name: int(assigned["split"].eq(name).sum()) for name in SPLITS},
        "tasks": {},
    }

    for task, spec in specs.items():
        subset = assigned.loc[assigned[f"mask_{task}"].eq(1)].copy()
        train = subset.loc[subset["split"].eq("train")]
        validation = subset.loc[subset["split"].eq("validation")]
        if train.empty or validation.empty:
            raise ValueError(f"{domain}/{task} has an empty train or validation split")
        candidates: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        if spec["kind"] == "single":
            for feature in ("char", "word", "char_word"):
                vectorizer = _vectorizer(feature, len(train))
                train_matrix = vectorizer.fit_transform(train["model_text"].fillna(""))
                validation_matrix = vectorizer.transform(validation["model_text"].fillna(""))
                for classifier in ("linear_svc", "sgd_logistic"):
                    estimator = _single_estimator(classifier, seed)
                    estimator.fit(train_matrix, train[task].astype(str))
                    predicted = estimator.predict(validation_matrix)
                    score = float(f1_score(validation[task].astype(str), predicted, average="macro", zero_division=0))
                    candidate = {"feature": feature, "classifier": classifier, "validation_macro_f1": score}
                    candidates.append(candidate)
                    if best is None or score > best["score"]:
                        best = {"score": score, "feature": feature, "classifier": classifier}
        else:
            labeler = MultiLabelBinarizer(classes=spec["labels"])
            train_y = labeler.fit_transform(train[task].map(_split_multi))
            validation_y = labeler.transform(validation[task].map(_split_multi))
            for feature in ("char", "word", "char_word"):
                vectorizer = _vectorizer(feature, len(train))
                train_matrix = vectorizer.fit_transform(train["model_text"].fillna(""))
                validation_matrix = vectorizer.transform(validation["model_text"].fillna(""))
                estimator = _multi_estimator(seed)
                estimator.fit(train_matrix, train_y)
                probabilities = _probabilities(estimator, validation_matrix)
                thresholds = _tune_thresholds(validation_y, probabilities)
                predicted = _ensure_one_label(probabilities >= np.asarray(thresholds), probabilities)
                score = float(f1_score(validation_y, predicted, average="macro", zero_division=0))
                candidate = {
                    "feature": feature,
                    "classifier": "ovr_sgd_logistic",
                    "validation_macro_f1": score,
                    "thresholds": thresholds,
                }
                candidates.append(candidate)
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "feature": feature,
                        "classifier": "ovr_sgd_logistic",
                        "thresholds": thresholds,
                    }

        if best is None:
            raise RuntimeError(f"no candidate model for {domain}/{task}")
        refit = subset.loc[subset["split"].isin(["train", "validation"])]
        final_vectorizer = _vectorizer(best["feature"], len(refit))
        refit_matrix = final_vectorizer.fit_transform(refit["model_text"].fillna(""))
        if spec["kind"] == "single":
            final_estimator = _single_estimator(best["classifier"], seed)
            final_estimator.fit(refit_matrix, refit[task].astype(str))
            labels = list(final_estimator.classes_)
            thresholds = None
        else:
            labeler = MultiLabelBinarizer(classes=spec["labels"])
            refit_y = labeler.fit_transform(refit[task].map(_split_multi))
            final_estimator = _multi_estimator(seed)
            final_estimator.fit(refit_matrix, refit_y)
            labels = list(labeler.classes_)
            thresholds = best["thresholds"]
        bundle["tasks"][task] = {
            "kind": spec["kind"],
            "labels": labels,
            "feature": best["feature"],
            "classifier": best["classifier"],
            "thresholds": thresholds,
            "vectorizer": final_vectorizer,
            "estimator": final_estimator,
        }
        report["tasks"][task] = {
            "labelled_rows": int(len(subset)),
            "selected_feature": best["feature"],
            "selected_classifier": best["classifier"],
            "validation_macro_f1": best["score"],
            "candidates": sorted(candidates, key=lambda item: item["validation_macro_f1"], reverse=True),
        }
    return bundle, report, assigned


def predict_domain(bundle: dict[str, Any], texts: pd.Series) -> pd.DataFrame:
    output = pd.DataFrame(index=texts.index)
    clean_texts = texts.fillna("").astype(str)
    for task, item in bundle["tasks"].items():
        matrix = item["vectorizer"].transform(clean_texts)
        probabilities = _probabilities(item["estimator"], matrix)
        if item["kind"] == "single":
            indices = probabilities.argmax(axis=1)
            labels = np.asarray(item["estimator"].classes_, dtype=object)
            output[f"predicted_{task}"] = labels[indices]
            output[f"confidence_{task}"] = probabilities.max(axis=1)
        else:
            thresholds = np.asarray(item["thresholds"], dtype=float)
            binary = _ensure_one_label(probabilities >= thresholds, probabilities)
            labels = np.asarray(item["labels"], dtype=object)
            output[f"predicted_{task}"] = [";".join(labels[row.astype(bool)]) for row in binary]
            output[f"confidence_{task}"] = probabilities.max(axis=1)
    return output


def evaluate_domain(bundle: dict[str, Any], assigned: pd.DataFrame, specs: dict) -> dict[str, Any]:
    report: dict[str, Any] = {"domain": bundle["domain"], "split": "frozen_test", "tasks": {}}
    for task, spec in specs.items():
        test = assigned.loc[
            assigned["split"].eq("test") & assigned[f"mask_{task}"].eq(1)
        ].copy()
        if test.empty:
            raise ValueError(f"{bundle['domain']}/{task} has an empty frozen test split")
        item = bundle["tasks"][task]
        matrix = item["vectorizer"].transform(test["model_text"].fillna(""))
        probabilities = _probabilities(item["estimator"], matrix)
        if spec["kind"] == "single":
            predicted = item["estimator"].classes_[probabilities.argmax(axis=1)]
            truth = test[task].astype(str).to_numpy()
            labels = spec["labels"]
            details = classification_report(
                truth, predicted, labels=labels, output_dict=True, zero_division=0
            )
            macro_f1 = float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0))
            support = {label: int((truth == label).sum()) for label in labels}
            matrix_value = confusion_matrix(truth, predicted, labels=labels).tolist()
            task_report = {
                "macro_f1": macro_f1,
                "support": support,
                "per_class": {label: details[label] for label in labels},
                "labels": labels,
                "confusion_matrix": matrix_value,
            }
        else:
            labeler = MultiLabelBinarizer(classes=spec["labels"])
            truth = labeler.fit_transform(test[task].map(_split_multi))
            predicted = _ensure_one_label(
                probabilities >= np.asarray(item["thresholds"], dtype=float), probabilities
            )
            labels = list(labeler.classes_)
            details = classification_report(
                truth, predicted, target_names=labels, output_dict=True, zero_division=0
            )
            macro_f1 = float(f1_score(truth, predicted, average="macro", zero_division=0))
            support = {label: int(truth[:, index].sum()) for index, label in enumerate(labels)}
            matrices = multilabel_confusion_matrix(truth, predicted).tolist()
            task_report = {
                "macro_f1": macro_f1,
                "support": support,
                "per_class": {label: details[label] for label in labels},
                "labels": labels,
                "confusion_matrix_per_label": dict(zip(labels, matrices, strict=True)),
                "thresholds": dict(zip(labels, item["thresholds"], strict=True)),
            }
        if not math.isfinite(task_report["macro_f1"]):
            task_report["macro_f1"] = 0.0
        report["tasks"][task] = task_report
    return report
