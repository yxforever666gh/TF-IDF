from __future__ import annotations

import argparse
import json
from pathlib import Path

from .paths import ProjectPaths
from .pipeline import evaluate, predict, prepare, run_all, train, verify


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tfidf-analytics")
    parser.add_argument("--data-root", help="包含三个 SQLite 与两个人工标注工作簿的本地数据目录")
    subparsers = parser.add_subparsers(dest="domain", required=True)
    for domain in ("complaint", "ecommerce"):
        domain_parser = subparsers.add_parser(domain)
        commands = domain_parser.add_subparsers(dest="command", required=True)
        commands.add_parser("prepare")
        commands.add_parser("train")
        commands.add_parser("evaluate")
        predict_parser = commands.add_parser("predict")
        predict_parser.add_argument("--input", type=Path, help="可选：对外部脱敏 CSV 预测")
        predict_parser.add_argument("--output", type=Path, help="外部预测结果路径")
        commands.add_parser("run-all")
    subparsers.add_parser("verify")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = ProjectPaths.create(args.data_root)
    if args.domain == "verify":
        result = verify(paths)
    elif args.command == "prepare":
        result = prepare(paths, args.domain)
    elif args.command == "train":
        result = train(paths, args.domain)
    elif args.command == "evaluate":
        result = evaluate(paths, args.domain)
    elif args.command == "predict":
        result = predict(paths, args.domain, args.input, args.output)
    elif args.command == "run-all":
        result = run_all(paths, args.domain)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
