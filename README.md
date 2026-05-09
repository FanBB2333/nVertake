# nVertake

A Python package for preemptive scheduling on NVIDIA GPUs.

## Features

- **Priority Scheduling**: Elevate your GPU process priority using CPU nice values and CUDA stream priorities
- **Memory Reservation**: Reserve GPU memory to prevent other processes from claiming it
- **Dynamic Memory Management**: Maintain constant memory usage even during script execution

## Installation

```bash
pip install -e .
```

## Usage

### Run with Elevated Priority

Run a Python script with elevated GPU scheduling priority:

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

This is intentionally low-intrusion, not a hard GPU quota system. Code that
manually switches CUDA streams, launches work from other Python threads, or uses
non-PyTorch CUDA APIs may need explicit integration.

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

See `test/README.md` for the test matrix, prerequisites, and how to interpret
the generated artifacts.

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
