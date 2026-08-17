from __future__ import annotations

import hashlib

import pandas as pd


SPLITS = ("train", "validation", "test")


def assign_group_splits(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assign duplicate groups deterministically to an approximate 70/15/15 split."""
    output = frame.copy().reset_index(drop=True)
    parent = list(range(len(output)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    # Exact normalized duplicates are always one component.
    for _, indices in output.groupby("text_hash", sort=False).groups.items():
        values = list(indices)
        for index in values[1:]:
            union(values[0], index)

    # Black Cat duplicate references point at the canonical complaint ID.
    record_index = {str(value): index for index, value in enumerate(output["record_id"])}
    for index, group in enumerate(output["duplicate_group_id"].fillna("").astype(str)):
        if group.startswith("blackcat-ref:"):
            target = "blackcat:" + group.split(":", 1)[1]
            if target in record_index:
                union(index, record_index[target])

    components: dict[int, list[int]] = {}
    for index in range(len(output)):
        components.setdefault(find(index), []).append(index)
    canonical = {}
    for indices in components.values():
        key = min(
            [str(output.loc[index, "text_hash"]) for index in indices]
            + [str(output.loc[index, "record_id"]) for index in indices]
        )
        for index in indices:
            canonical[index] = "group_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    output["duplicate_group_id"] = pd.Series(canonical)
    group_keys = output["duplicate_group_id"]

    def choose(group: str) -> str:
        digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        return "train" if value < 0.70 else ("validation" if value < 0.85 else "test")

    output["split"] = group_keys.map(choose)
    return output


def assert_no_leakage(frame: pd.DataFrame) -> None:
    for column in ("record_id", "text_hash", "duplicate_group_id"):
        sets = {
            split: set(frame.loc[frame["split"].eq(split), column].dropna().astype(str))
            for split in SPLITS
        }
        for index, left in enumerate(SPLITS):
            for right in SPLITS[index + 1 :]:
                overlap = sets[left] & sets[right]
                if overlap:
                    raise AssertionError(f"{column} leakage between {left} and {right}: {len(overlap)}")
