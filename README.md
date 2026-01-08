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

## CLI Options

```
nvertake [OPTIONS] [COMMAND]

Options:
  --filled, -f RATIO    Fill GPU memory to this ratio (0.0-1.0)
  --device, -d GPU_ID   GPU device to use (default: 0)
  --nice, -n VALUE      Nice value for CPU priority (default: -10)
  --quiet, -q           Suppress info messages
  --version, -V         Show version

Commands:
  run SCRIPT [ARGS...]  Run a Python script with elevated priority
  info                  Show GPU information
```

## How It Works

### Priority Scheduling

1. **CPU Priority**: Uses `os.nice()` to increase process priority (lower nice value = higher priority)
2. **CUDA Streams**: Creates high-priority CUDA streams for GPU task scheduling

### Memory Reservation

1. Calculates target memory based on fill ratio
2. Allocates buffer tensors to reach target
3. Runs a background thread to monitor and adjust buffer size
4. Maintains constant total GPU memory usage during script execution

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

It prints a markdown table you can paste into this README, and saves logs under `test/overtake_output/<timestamp>/`.

### Overtake Results (Fill With Your Run)

Example (FakeGPU, `MATRIX_SIZE=256`, `DTYPE=float32`, `PRE_SECONDS=3`, `INVADE_SECONDS=3`):

| Scenario | Native it/s (pre) | Native it/s (during) | Δ (during vs pre) |
|---|---:|---:|---:|
| invader_no_nvertake | 4.76 | 4.73 | -0.8% |
| invader_with_nvertake | 4.47 | 4.59 | +2.6% |

## Requirements

- Python >= 3.8
- PyTorch
- NVIDIA GPU with CUDA support
- psutil

## License

MIT
