# Testing nVertake

nVertake's test and benchmark flow is built around real NVIDIA GPUs with
CUDA-enabled PyTorch. FakeGPU is not part of the default validation story for
this repository.

## Test Matrix

### 1. CPU / Unit Tests

File: `test/test_nvertake.py`

Scope:
- CLI argument parsing
- Green Context partition calculation, multi-process launch, and concurrent task wiring
- read-only driver diagnostics and dry-run allocation plans
- YAML validation, multi-GPU grouping, logs, runtime reports, and monitor output
- GPU leases, stale-run reconciliation, SSH retry/reattach, and cancellation state
- PyTorch memory-fraction wiring and multi-sample throughput calibration
- static-MPS capability/chunk planning and nvidia-smi/DCGM telemetry parsing
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

### 5. Green Context SM-Share Experiment

File: `verification/run_green_context_share_experiment.py`

Purpose:

- run two saturated PyTorch GEMM callables in one process
- compare driver-selected SM allocations for requested `50/50` and `25/75`
- report each lane's exact SM count and measured throughput
- exercise context creation, per-thread binding, synchronization, and cleanup

Run on a CUDA system whose driver exposes Green Context APIs:

```bash
bash test/run_tests_summary.sh green-share \
  --device 0 \
  --output verification/results/green_share_latest.json
```

This experiment works on the tested native Linux and WSL hosts. Exact TFLOP/s
ratios are workload- and architecture-dependent even when the SM partition is
fixed.

### 6. Multi-Process Green Context Experiment

Files:

- `verification/run_green_process_share_experiment.py`
- `verification/green_process_gemm_worker.py`

Purpose:

- launch three or more real Python processes through `green-procs`
- verify that all workers report one consistent, complete SM partition map
- report distinct PIDs, exact SM counts, and measured throughput share
- exercise native Linux and WSL without an MPS daemon

Run:

```bash
bash test/run_tests_summary.sh green-procs-share \
  --device 0 \
  --shares 20,30,50 \
  --output verification/results/green_process_share_latest.json
```

Use any number of positive weights. The experiment repeats the same saturated
worker file once per weight; the public CLI also accepts different Python files.

### 7. MPS Weighted-Share Experiment

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

### 8. YAML Launch, Memory Cap, Monitor, and Calibration Smoke Test

The example worker publishes `GEMM/s` through `report_throughput()`, so one run
exercises the YAML launcher, independent logs, PyTorch memory fractions, live
JSON state, and automatic calibration:

```bash
nvertake launch examples/jobs.yaml --dry-run
nvertake launch examples/jobs.yaml --calibrate
nvertake monitor
```

While the launch is active, `nvertake monitor --watch` displays PID, assigned
SM count, framebuffer memory, reported throughput, and state.
`nvertake monitor --profile --json` adds process SM/memory utilization and
device/DCGM telemetry. After an injected launcher failure,
`nvertake list --refresh` reconciles the stale report. The final report is
stored below `examples/runs/<run-id>/report.json`.

## Recommended Workflow

1. Start with `bash test/run_tests_summary.sh unit`.
2. On a real GPU machine, run `bash test/run_tests_summary.sh gpu --device 0`.
3. If `nvidia-smi pmon` is available, run `bash test/run_tests_summary.sh pmon --device 0`.
4. Run `bash test/run_tests_summary.sh overtake` to collect benchmark data you
   can publish in the root README.
5. Run the two-task Green Context SM-share experiment.
6. Run the multi-process Green Context experiment with at least three weights.
7. On a native Linux MPS host, run the MPS weighted-share experiment.
8. Run the YAML launch smoke test and inspect its final JSON report.

## Notes

- Negative nice values may require elevated privileges or `CAP_SYS_NICE`.
- The benchmark scripts intentionally write fresh artifacts instead of shipping
  canned result files in the repository.
- If you want durable benchmark numbers in version control, copy the generated
  markdown table or summary JSON into a dedicated report after the run.
