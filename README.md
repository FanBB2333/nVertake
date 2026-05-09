# nVertake

A Python package for low-intrusion CUDA stream priority, GPU memory reservation,
and resource-contention experiments on NVIDIA GPUs.

## Features

- **PyTorch Priority Hook**: Run common PyTorch training and inference workloads on a high-priority CUDA stream with little or no source change
- **Memory Reservation**: Reserve GPU memory to prevent other processes from claiming it
- **Dynamic Memory Management**: Maintain constant memory usage even during script execution
- **Contention Experiments**: Measure whether a change actually improves saturated GPU resource share before relying on it

## Installation

```bash
pip install -e .
```

## Usage

### Run with Elevated Priority

Run a Python script with nVertake's PyTorch stream-priority hook:

```bash
nvertake run train.py --epochs 100
```

For PyTorch scripts, `nvertake run` enables a startup hook by default. The hook
patches PyTorch after import and sets the current CUDA stream to a high-priority
stream, so most single-process training scripts do not need source changes.

For launchers such as `torchrun` or `accelerate`, use `exec` so child worker
processes inherit the same environment:

```bash
nvertake exec torchrun --nproc_per_node=4 train.py
```

Disable the PyTorch hook for `run` or `exec` if your program manages CUDA
streams explicitly:

```bash
nvertake --no-torch-priority run train.py
```

### Memory Reservation

Reserve 95% of GPU memory while running your script:

```bash
nvertake --filled 0.95 run train.py
```

This prevents other processes from claiming GPU memory, even when your script temporarily frees memory during execution.

### Standalone Memory Fill

Just fill GPU memory without running a script (useful for reserving GPU):

```bash
nvertake --filled 0.95
```

Press Ctrl+C to release the memory.

### Specify GPU Device

```bash
nvertake --device 1 --filled 0.8 run inference.py
```

### Show GPU Info

```bash
nvertake info
```

### Python Decorator (In-Process)

If you want to inject priority inside an existing Python process (instead of using `nvertake run ...`), use `inject_priority`:

```python
from nvertake import inject_priority

@inject_priority(device=0, nice_value=-10)
def train():
    # Your CUDA workload here
    ...

train()
```

For an even smaller source change, call the PyTorch hook once near process
startup:

```python
from nvertake import enable_torch_priority

enable_torch_priority()
```

## CLI Options

```
nvertake [OPTIONS] [COMMAND]

Options:
  --filled, -f RATIO    Fill GPU memory to this ratio (0.0-1.0)
  --device, -d GPU_ID   GPU device to use (default: 0)
  --nice, -n VALUE      Nice value for CPU priority (default: -10)
  --no-torch-priority   Disable PyTorch high-priority stream auto-injection
  --quiet, -q           Suppress info messages
  --version, -V         Show version

Commands:
  run SCRIPT [ARGS...]  Run a Python script with elevated priority
  exec COMMAND [ARGS...] Run any command with elevated priority environment
  info                  Show GPU information
```

## How It Works

### Priority Scheduling

1. **CPU Priority**: Uses `os.nice()` to increase process priority (lower nice value = higher priority)
2. **PyTorch Auto-Injection**: `nvertake run` prepends a small `sitecustomize` hook to the child Python process. When PyTorch is imported, the hook creates a high-priority CUDA stream and makes it the current stream for the selected device.
3. **In-Process API**: `@inject_priority` and `enable_torch_priority()` remain available when you prefer an explicit one-line code change.

This is intentionally low-intrusion, not a hard GPU quota or preemption system.
CUDA stream priority can influence how queued CUDA work is scheduled, but it does
not give an unprivileged process a guaranteed GPU capacity share. Lowering the
CPU nice value may also require elevated OS permissions; if that fails, the CUDA
stream hook can still be used inside your own process.

Code that manually switches CUDA streams, launches work from other Python
threads, or uses non-PyTorch CUDA APIs may need explicit integration.

### Memory Reservation

1. Calculates target memory based on fill ratio
2. Allocates buffer tensors to reach target
3. Runs a background thread to monitor and adjust buffer size
4. Maintains constant total GPU memory usage during script execution

## Testing

nVertake's validation targets real NVIDIA GPUs with CUDA-enabled PyTorch. The
repository does not depend on FakeGPU for its main test flow.

- CPU/unit checks: `bash test/run_tests_summary.sh unit`
- Real-GPU integration checks: `bash test/run_tests_summary.sh gpu --device 0`
- `nvidia-smi pmon` checks: `bash test/run_tests_summary.sh pmon --device 0`
- Resident-vs-invader benchmark: `bash test/run_tests_summary.sh overtake`
- Hugging Face smoke checks:
  `nvertake run verification/verify_transformers_priority.py --expect-priority`
- Sustained utilization contention experiment:
  `python3 verification/run_contention_util_experiment.py --output verification/results/contention_util_<date>.json`
- Own-process-only contention variants:
  `python3 verification/run_own_process_variants_experiment.py --output verification/results/own_process_variants_<date>.json`
- Plot-ready result summary:
  `verification/results/resource_contention_summary_20260509.json`

See `test/README.md` for the test matrix, prerequisites, and how to interpret
the generated artifacts.

## Experimental Results

The 2026-05-09 real-GPU contention runs were designed around saturated
resident-vs-invader workloads, because GPU utilization must be near full before
resource-share claims are meaningful.

The same-kernel two-process experiment saturated both tested loads
(`97.7-99.5%` mean GPU utilization), but the nVertake high-priority stream
invader did not gain share. Invader share changed by `-0.26` percentage points
for `gemm4096_fp16_batch12` and `-0.82` percentage points for
`gemm8192_fp16_batch4`, so this result does not prove stronger inter-process GPU
allocation under full utilization.

The own-process-only variant experiment kept the resident fixed at
`gemm8192_fp16_batch4` and changed only the invader implementation:

| Invader Variant | Mean GPU Util | Invader TFLOP/s | Resident TFLOP/s | Invader Share | Share Δ vs Control |
|---|---:|---:|---:|---:|---:|
| control, single eager normal stream | 99.5% | 36.92 | 49.52 | 42.71% | 0.00 pp |
| nVertake, single eager high-priority stream | 99.0% | 36.38 | 49.53 | 42.35% | -0.36 pp |
| nVertake, four eager high-priority streams | 98.3% | 35.96 | 48.78 | 42.43% | -0.28 pp |
| nVertake, CUDA Graph replay with 4096 GEMM | 99.9% | 213.80 | 38.77 | 84.65% | +41.94 pp |
| nVertake, smaller 4096 eager GEMM | 98.9% | 31.36 | 46.45 | 40.31% | -2.40 pp |
| two normal-priority invader replicas | 99.1% | 36.95 | 41.07 | 47.35% | +4.64 pp |
| two nVertake invader replicas | 99.6% | 39.02 | 41.41 | 48.52% | +5.80 pp |

Interpretation: the low-intrusion nVertake stream hook is effective at making
typical PyTorch/Hugging Face workloads run on a non-default high-priority stream,
but the measured full-utilization share gains came from changing the invader's
own execution pattern, especially CUDA Graph replay or extra own replicas. Treat
those as workload-level optimization or self-parallelism techniques, not as a
guaranteed GPU scheduling entitlement.

## Performance Test (Resident vs Invader)

This benchmark starts 2 programs:
- **resident**: a stable compute loop (throughput is reported as iterations/second)
- **invader**: launched after resident is running, with/without `nVertake` priority injection

Run:
```bash
bash test/run_overtake_benchmark.sh
```
or:
```bash
bash test/run_tests_summary.sh overtake
```

It runs on a real CUDA GPU, prints a markdown table you can paste into this
README or an issue, and saves logs under `test/overtake_output/<timestamp>/`.

### Overtake Result Template

The table below is the expected output shape. Replace it with numbers from your
own real-GPU run if you want to publish benchmark results.

| Scenario | Native it/s (pre) | Native it/s (during) | Δ (during vs pre) |
|---|---:|---:|---:|
| invader_no_nvertake | ... | ... | ... |
| invader_with_nvertake | ... | ... | ... |

## Requirements

- Python >= 3.8
- PyTorch
- NVIDIA GPU with CUDA support
- psutil

## License

MIT
