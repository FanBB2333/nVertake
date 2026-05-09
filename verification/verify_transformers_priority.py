#!/usr/bin/env python3
"""
Verify nVertake auto-priority on common Hugging Face workloads.

The script uses randomly initialized tiny models, so it does not download
weights or datasets. It checks that typical transformers inference and PEFT
LoRA training run while the active CUDA stream is nVertake's non-default stream.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch


@dataclass
class ScenarioResult:
    scenario: str
    ok: bool
    cuda_available: bool
    device: int
    current_is_default_before: Optional[bool]
    current_is_default_after: Optional[bool]
    current_stream_before: Optional[int]
    current_stream_after: Optional[int]
    default_stream: Optional[int]
    elapsed_seconds: float
    details: Dict[str, Any]


def _stream_state(device: int) -> Dict[str, Optional[int]]:
    if not torch.cuda.is_available():
        return {
            "current_stream": None,
            "default_stream": None,
            "current_is_default": None,
        }
    current = torch.cuda.current_stream(device)
    default = torch.cuda.default_stream(device)
    return {
        "current_stream": int(current.cuda_stream),
        "default_stream": int(default.cuda_stream),
        "current_is_default": bool(current.cuda_stream == default.cuda_stream),
    }


def _assert_priority_stream(result: ScenarioResult, expect_priority: bool) -> None:
    if not expect_priority:
        return
    if not result.cuda_available:
        raise AssertionError(f"{result.scenario}: CUDA is not available")
    if result.current_is_default_before is not False:
        raise AssertionError(
            f"{result.scenario}: expected non-default current stream before workload, "
            f"got current_is_default={result.current_is_default_before}"
        )
    if result.current_is_default_after is not False:
        raise AssertionError(
            f"{result.scenario}: expected non-default current stream after workload, "
            f"got current_is_default={result.current_is_default_after}"
        )


def _tiny_gpt2_model(device: int):
    from transformers import AutoModelForCausalLM, GPT2Config

    config = GPT2Config(
        vocab_size=256,
        n_positions=64,
        n_ctx=64,
        n_embd=96,
        n_layer=2,
        n_head=4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_config(config)
    return model.to(f"cuda:{device}")


def _random_batch(device: int, batch_size: int = 2, seq_len: int = 32) -> Dict[str, torch.Tensor]:
    input_ids = torch.randint(
        low=3,
        high=255,
        size=(batch_size, seq_len),
        device=f"cuda:{device}",
        dtype=torch.long,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def run_transformers_inference(device: int, expect_priority: bool, steps: int) -> ScenarioResult:
    before = _stream_state(device)
    start = time.perf_counter()
    model = _tiny_gpt2_model(device)
    model.eval()
    batch = _random_batch(device)

    with torch.inference_mode():
        for _ in range(steps):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            _ = torch.argmax(logits[:, -1, :], dim=-1)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    after = _stream_state(device)

    result = ScenarioResult(
        scenario="transformers_inference",
        ok=True,
        cuda_available=torch.cuda.is_available(),
        device=device,
        current_is_default_before=before["current_is_default"],
        current_is_default_after=after["current_is_default"],
        current_stream_before=before["current_stream"],
        current_stream_after=after["current_stream"],
        default_stream=after["default_stream"],
        elapsed_seconds=elapsed,
        details={
            "steps": steps,
            "logits_shape": list(logits.shape),
            "model_class": type(model).__name__,
        },
    )
    _assert_priority_stream(result, expect_priority)
    return result


def run_peft_lora_training(device: int, expect_priority: bool, steps: int) -> ScenarioResult:
    from peft import LoraConfig, TaskType, get_peft_model

    before = _stream_state(device)
    start = time.perf_counter()
    model = _tiny_gpt2_model(device)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn", "c_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.train()
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=1e-3,
    )

    losses: List[float] = []
    for _ in range(steps):
        batch = _random_batch(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    after = _stream_state(device)

    result = ScenarioResult(
        scenario="peft_lora_training",
        ok=True,
        cuda_available=torch.cuda.is_available(),
        device=device,
        current_is_default_before=before["current_is_default"],
        current_is_default_after=after["current_is_default"],
        current_stream_before=before["current_stream"],
        current_stream_after=after["current_stream"],
        default_stream=after["default_stream"],
        elapsed_seconds=elapsed,
        details={
            "steps": steps,
            "losses": losses,
            "trainable_parameters": int(
                sum(param.numel() for param in model.parameters() if param.requires_grad)
            ),
            "model_class": type(model).__name__,
        },
    )
    _assert_priority_stream(result, expect_priority)
    return result


def _selected_scenarios(value: str) -> Iterable[str]:
    if value == "all":
        return ("inference", "lora")
    return (value,)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("all", "inference", "lora"),
        default="all",
        help="Verification scenario to run.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--steps", type=int, default=3, help="Workload iterations.")
    parser.add_argument(
        "--expect-priority",
        action="store_true",
        help="Fail unless the active stream is non-default before and after each workload.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON results.",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA is not available; run this verification on a CUDA host.", file=sys.stderr)
        return 2

    torch.cuda.set_device(args.device)
    # Make output deterministic enough for regression comparison.
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = True

    results: List[ScenarioResult] = []
    try:
        for scenario in _selected_scenarios(args.scenario):
            if scenario == "inference":
                results.append(
                    run_transformers_inference(
                        device=args.device,
                        expect_priority=args.expect_priority,
                        steps=args.steps,
                    )
                )
            elif scenario == "lora":
                results.append(
                    run_peft_lora_training(
                        device=args.device,
                        expect_priority=args.expect_priority,
                        steps=args.steps,
                    )
                )
            else:
                raise ValueError(f"Unsupported scenario: {scenario}")
    except Exception as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1

    payload = [asdict(result) for result in results]
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for result in results:
            print(
                f"{result.scenario}: ok={result.ok} "
                f"current_is_default_before={result.current_is_default_before} "
                f"current_is_default_after={result.current_is_default_after} "
                f"elapsed={result.elapsed_seconds:.3f}s details={result.details}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
