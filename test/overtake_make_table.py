#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: float) -> str:
    return f"{v:.2f}"


def _fmt_pct(v: float) -> str:
    return f"{v:+.1f}%"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render markdown table from overtake_analyze.py JSON summaries.")
    parser.add_argument("summaries", nargs="+", help="One or more summary.json paths")
    args = parser.parse_args(argv)

    rows: List[Dict[str, Any]] = [_load(Path(p)) for p in args.summaries]
    rows.sort(key=lambda r: r.get("label", ""))

    device_name = next((r.get("device_name") for r in rows if r.get("device_name")), None)
    matrix_size = next((r.get("matrix_size") for r in rows if r.get("matrix_size")), None)
    dtype = next((r.get("dtype") for r in rows if r.get("dtype")), None)

    if device_name or matrix_size or dtype:
        header_bits = []
        if device_name:
            header_bits.append(f"GPU: {device_name}")
        if matrix_size:
            header_bits.append(f"matmul: {matrix_size}x{matrix_size}")
        if dtype:
            header_bits.append(f"dtype: {dtype}")
        print("\n".join(header_bits))
        print()

    print("| Scenario | Native it/s (pre) | Native it/s (during) | Δ (during vs pre) |")
    print("|---|---:|---:|---:|")
    for row in rows:
        label = row.get("label", "")
        pre = float(row.get("pre_mean_iters_per_sec", 0.0))
        during = float(row.get("during_mean_iters_per_sec", 0.0))
        drop = float(row.get("drop_percent", 0.0))
        print(f"| {label} | {_fmt(pre)} | {_fmt(during)} | {_fmt_pct(drop)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

