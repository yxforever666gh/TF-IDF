from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_root: Path

    @classmethod
    def create(cls, data_root: str | Path | None = None) -> "ProjectPaths":
        root = Path(__file__).resolve().parents[2]
        configured = data_root or os.environ.get("TFIDF_DATA_ROOT") or root.parent / "raw_data"
        return cls(root=root, data_root=Path(configured).resolve())

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def output(self) -> Path:
        return self.root / "data" / "output"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def settings(self) -> dict:
        return json.loads((self.root / "config" / "settings.json").read_text(encoding="utf-8"))

    @property
    def task_specs(self) -> dict:
        return json.loads((self.root / "config" / "task_labels.json").read_text(encoding="utf-8"))

    def domain_output(self, domain: str) -> Path:
        return self.output / domain

    def domain_model(self, domain: str) -> Path:
        return self.artifacts / domain / "model.joblib"

    def domain_prepared(self, domain: str) -> Path:
        return self.processed / f"{domain}_prepared.csv"

    def domain_annotations(self, domain: str) -> Path:
        return self.processed / f"{domain}_annotations.csv"

    def ensure_runtime_dirs(self, domain: str) -> None:
        self.processed.mkdir(parents=True, exist_ok=True)
        self.domain_output(domain).mkdir(parents=True, exist_ok=True)
        self.domain_model(domain).parent.mkdir(parents=True, exist_ok=True)

    def locate(self, pattern: str) -> Path:
        matches = sorted(self.data_root.rglob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one {pattern!r} below {self.data_root}, found {len(matches)}"
            )
        return matches[0]
