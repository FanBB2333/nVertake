# Transformers Priority Verification

Date: 2026-05-09

Host GPU: NVIDIA GeForce RTX 3090 Ti

Environment:
- torch: 2.9.1+cu128
- transformers: installed
- peft: 0.19.1
- accelerate: installed

## What Was Verified

The verification uses tiny randomly initialized GPT-2 models, so it does not
download model weights or datasets.

Scenarios:
- transformers causal-LM inference
- PEFT LoRA causal-LM training
- direct `nvertake run`
- launcher-compatible `nvertake exec`
- `torchrun` with explicit rendezvous address

## Commands And Results

Baseline without nVertake:

```bash
python3 verification/verify_transformers_priority.py --scenario all --steps 2 --json
```

Result:
- `transformers_inference`: `current_is_default_before=true`, `current_is_default_after=true`
- `peft_lora_training`: `current_is_default_before=true`, `current_is_default_after=true`

With `nvertake run`:

```bash
python3 -m nvertake.cli --nice 0 run verification/verify_transformers_priority.py --scenario all --steps 2 --expect-priority --json
```

Result:
- `transformers_inference`: `current_is_default_before=false`, `current_is_default_after=false`
- `peft_lora_training`: `current_is_default_before=false`, `current_is_default_after=false`

With `nvertake exec`:

```bash
python3 -m nvertake.cli --nice 0 exec python3 verification/verify_transformers_priority.py --scenario all --steps 2 --expect-priority --json
```

Result:
- `transformers_inference`: `current_is_default_before=false`, `current_is_default_after=false`
- `peft_lora_training`: `current_is_default_before=false`, `current_is_default_after=false`

With `torchrun` through `nvertake exec`:

```bash
python3 -m nvertake.cli --nice 0 exec torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29601 verification/verify_transformers_priority.py --scenario inference --steps 1 --expect-priority --json
python3 -m nvertake.cli --nice 0 exec torchrun --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29602 verification/verify_transformers_priority.py --scenario lora --steps 1 --expect-priority --json
```

Result:
- `transformers_inference`: `current_is_default_before=false`, `current_is_default_after=false`
- `peft_lora_training`: `current_is_default_before=false`, `current_is_default_after=false`

## Interpretation

For normal PyTorch/transformers flows that launch kernels on the current CUDA
stream, nVertake's low-intrusion injection is effective: the workload runs with
a non-default CUDA stream selected by nVertake instead of PyTorch's default
stream.

This validates common single-process transformers inference and PEFT LoRA
training smoke paths, plus a `torchrun` launcher path. It does not prove a hard
GPU quota or cover workloads that explicitly switch to their own CUDA streams
after startup.

## Note

`torchrun --standalone` hung on this host in c10d TCPStore rendezvous address
resolution. Retrying with explicit `--master_addr=127.0.0.1 --master_port=...`
worked and preserved the nVertake priority injection.
