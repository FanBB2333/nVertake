#!/usr/bin/env python3
"""
Run a small model/parameter matrix for nVertake PyTorch priority injection.

The experiment compares the same transformers/PEFT workloads in two modes:
- baseline: plain Python, expected to use PyTorch's default CUDA stream
- nvertake_run: `python -m nvertake.cli run ...`, expected to use nVertake's
  non-default high-priority CUDA stream
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = REPO_ROOT / "verification" / "verify_transformers_priority.py"


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    scenario: str
    model_size: str
    batch_size: int
    seq_len: int
    steps: int
    lora_rank: int = 4


@dataclass
class ModeResult:
    mode: str
    returncode: int
    command: List[str]
    elapsed_seconds: float
    stdout_json: Optional[Any]
    stderr_tail: str


@dataclass
class CaseResult:
    case: MatrixCase
    baseline: ModeResult
    nvertake_run: ModeResult
    effective: bool
    reason: str


CASES: List[MatrixCase] = [
    MatrixCase(
        case_id="inference_tiny_batch2_seq32",
        scenario="inference",
        model_size="tiny",
        batch_size=2,
        seq_len=32,
        steps=2,
    ),
    MatrixCase(
        case_id="inference_small_batch4_seq48",
        scenario="inference",
        model_size="small",
        batch_size=4,
        seq_len=48,
        steps=2,
    ),
    MatrixCase(
        case_id="inference_medium_batch2_seq64",
        scenario="inference",
        model_size="medium",
        batch_size=2,
        seq_len=64,
        steps=1,
    ),
    MatrixCase(
        case_id="lora_tiny_rank4_batch2_seq32",
        scenario="lora",
        model_size="tiny",
        batch_size=2,
        seq_len=32,
        steps=2,
        lora_rank=4,
    ),
    MatrixCase(
        case_id="lora_small_rank8_batch2_seq48",
        scenario="lora",
        model_size="small",
        batch_size=2,
        seq_len=48,
        steps=2,
        lora_rank=8,
    ),
    MatrixCase(
        case_id="lora_medium_rank4_batch1_seq64",
        scenario="lora",
        model_size="medium",
        batch_size=1,
        seq_len=64,
        steps=1,
        lora_rank=4,
    ),
]


def _json_payload(stdout: str) -> Optional[Any]:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min((idx for idx in (text.find("["), text.find("{")) if idx >= 0), default=-1)
        if start < 0:
            return None
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            return None


def _tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _case_args(case: MatrixCase) -> List[str]:
    return [
        str(VERIFY_SCRIPT),
        "--scenario",
        case.scenario,
        "--model-size",
        case.model_size,
        "--batch-size",
        str(case.batch_size),
        "--seq-len",
        str(case.seq_len),
        "--steps",
        str(case.steps),
        "--lora-rank",
        str(case.lora_rank),
        "--json",
    ]


def _run_mode(case: MatrixCase, mode: str) -> ModeResult:
    if mode == "baseline":
        command = [sys.executable] + _case_args(case)
    elif mode == "nvertake_run":
        command = [
            sys.executable,
            "-m",
            "nvertake.cli",
            "--nice",
            "0",
            "run",
        ] + _case_args(case) + ["--expect-priority"]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    return ModeResult(
        mode=mode,
        returncode=completed.returncode,
        command=command,
        elapsed_seconds=elapsed,
        stdout_json=_json_payload(completed.stdout),
        stderr_tail=_tail(completed.stderr),
    )


def _first_result(mode_result: ModeResult) -> Optional[Dict[str, Any]]:
    if not isinstance(mode_result.stdout_json, list) or not mode_result.stdout_json:
        return None
    first = mode_result.stdout_json[0]
    return first if isinstance(first, dict) else None


def _is_effective(baseline: ModeResult, nvertake_run: ModeResult) -> tuple[bool, str]:
    baseline_result = _first_result(baseline)
    nvertake_result = _first_result(nvertake_run)
    if baseline.returncode != 0:
        return False, f"baseline failed with returncode={baseline.returncode}"
    if nvertake_run.returncode != 0:
        return False, f"nvertake_run failed with returncode={nvertake_run.returncode}"
    if baseline_result is None:
        return False, "baseline JSON result missing"
    if nvertake_result is None:
        return False, "nvertake JSON result missing"
    if baseline_result.get("current_is_default_before") is not True:
        return False, "baseline did not start on default stream"
    if baseline_result.get("current_is_default_after") is not True:
        return False, "baseline did not end on default stream"
    if nvertake_result.get("current_is_default_before") is not False:
        return False, "nvertake did not start on non-default stream"
    if nvertake_result.get("current_is_default_after") is not False:
        return False, "nvertake did not end on non-default stream"
    if nvertake_result.get("current_stream_before") == nvertake_result.get("default_stream"):
        return False, "nvertake current stream equals default stream"
    return True, "baseline used default stream; nVertake used non-default priority stream"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="verification/results/priority_matrix_latest.json",
        help="Path for JSON summary.",
    )
    args = parser.parse_args()

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    results: List[CaseResult] = []
    for case in CASES:
        print(f"[matrix] running {case.case_id} baseline", flush=True)
        baseline = _run_mode(case, "baseline")
        print(f"[matrix] running {case.case_id} nvertake_run", flush=True)
        nvertake_run = _run_mode(case, "nvertake_run")
        effective, reason = _is_effective(baseline, nvertake_run)
        results.append(
            CaseResult(
                case=case,
                baseline=baseline,
                nvertake_run=nvertake_run,
                effective=effective,
                reason=reason,
            )
        )

    payload: Dict[str, Any] = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_cases": len(results),
        "effective_cases": sum(1 for result in results if result.effective),
        "all_effective": all(result.effective for result in results),
        "definition_of_effective": (
            "baseline reports current_is_default_before/after=true, while "
            "nvertake_run reports current_is_default_before/after=false and "
            "the selected current stream differs from default_stream"
        ),
        "results": [
            {
                "case": asdict(result.case),
                "effective": result.effective,
                "reason": result.reason,
                "baseline": asdict(result.baseline),
                "nvertake_run": asdict(result.nvertake_run),
            }
            for result in results
        ],
    }

    output_path = REPO_ROOT / args.output
    _write_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[matrix] wrote {output_path}")
    return 0 if payload["all_effective"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
