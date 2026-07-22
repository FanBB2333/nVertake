# Testing nVertake

nVertake's test and benchmark flow is built around real NVIDIA GPUs with
CUDA-enabled PyTorch. FakeGPU is not part of the default validation story for
this repository.

## Test Matrix

### 1. CPU / Unit Tests

File: `test/test_nvertake.py`

Scope:
- CLI argument parsing
- scheduler wiring
- PyTorch auto-priority env injection
- memory target calculations
- utility helpers

Run:

```bash
bash test/run_tests_summary.sh unit
```

These tests do not require a GPU.

### 2. Real-GPU Integration Tests

File: `test/test_gpu_priority.py`

Scope:
- high-priority CUDA stream creation
- low-intrusion PyTorch stream auto-injection
- stream-priority behavior under contention
- optional `nvidia-smi pmon` observation of SM utilization

Run:

```bash
bash test/run_tests_summary.sh gpu --device 0
```

Stricter latency assertions:

```bash
bash test/run_tests_summary.sh strict --device 0
```

These modes require:
- an NVIDIA GPU
- a CUDA-enabled PyTorch install
- a valid CUDA device index

### 3. PMON-Based Process Observation

File: `test/test_gpu_priority.py`

Scope:
- verify that `nvidia-smi pmon` can observe two competing GPU processes
- optionally assert that the nVertake-enabled process wins more SM share

Run:

```bash
bash test/run_tests_summary.sh pmon --device 0
```

Stricter PMON assertions:

```bash
bash test/run_tests_summary.sh strict-pmon --device 0
```

Extra requirement:
- `nvidia-smi pmon` must be available on `PATH`

### 4. Resident-vs-Invader Benchmark

Files:
- `test/run_overtake_benchmark.sh`
- `test/overtake_resident.py`
- `test/overtake_invader.py`
- `test/overtake_analyze.py`
- `test/overtake_make_table.py`

Purpose:
- run a stable resident GEMM loop
- launch an invader workload with and without `nvertake`
- compare how much the resident throughput drops during contention

Run:

```bash
bash test/run_tests_summary.sh overtake
```

Artifacts:
- JSONL resident logs under `test/overtake_output/<timestamp>/`
- per-scenario JSON summaries
- a markdown table printed to stdout

Interpretation:
- `invader_no_nvertake` is the baseline competing workload
- `invader_with_nvertake` enables `inject_priority`
- a smaller throughput drop for the resident during the `with_nvertake` case is
  the signal you are looking for

### 5. MPS Weighted-Share Experiment

File: `verification/run_mps_share_experiment.py`

Purpose:

- launch two saturated GEMM processes through the public `nvertake` CLI
- compare an equal `50/50` active-thread configuration with `25/75`
- report the target process's share of combined measured TFLOP/s
- verify that daemon startup, GPU remapping, client probing, and cleanup work
  together

Run on a native Linux host with NVIDIA MPS:

```bash
bash test/run_tests_summary.sh mps-share \
  --device 0 \
  --output verification/results/mps_share_latest.json
```

The experiment is not available in WSL. If the MPS client cannot connect, the
CLI exits before starting the worker and includes the tail of `control.log` and
`server.log` in the error.

## Recommended Workflow

1. Start with `bash test/run_tests_summary.sh unit`.
2. On a real GPU machine, run `bash test/run_tests_summary.sh gpu --device 0`.
3. If `nvidia-smi pmon` is available, run `bash test/run_tests_summary.sh pmon --device 0`.
4. Run `bash test/run_tests_summary.sh overtake` to collect benchmark data you
   can publish in the root README.
5. On a native Linux MPS host, run the weighted-share experiment.

## Notes

- Negative nice values may require elevated privileges or `CAP_SYS_NICE`.
- The benchmark scripts intentionally write fresh artifacts instead of shipping
  canned result files in the repository.
- If you want durable benchmark numbers in version control, copy the generated
  markdown table or summary JSON into a dedicated report after the run.
