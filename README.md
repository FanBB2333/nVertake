# nVertake

A Python package for CUDA Green Context SM partitioning, NVIDIA MPS GPU sharing,
low-intrusion CUDA stream priority, GPU memory reservation, and
resource-contention experiments.

## Features

- **PyTorch Priority Hook**: Run common PyTorch training and inference workloads on a high-priority CUDA stream with little or no source change
- **Multi-Process SM Shares**: Launch two, three, or more Python files together with a requested SM weight for each process
- **YAML and Multi-GPU Launches**: Keep scripts, arguments, environments, working directories, and GPU placement in one job file
- **Dry-Run Diagnostics**: Check driver support and preview exact SM counts without creating contexts or processes
- **Per-Process PyTorch Memory Caps**: Pair SM shares with caching-allocator memory fractions
- **Live Reports and Calibration**: Monitor local PIDs and adjust SM weights from workload-reported throughput
- **Driver-Level SM Partitioning**: Run two cooperating in-process tasks in separate CUDA Green Contexts, including on WSL
- **Weighted GPU Sharing**: Give cooperating CUDA processes different execution-resource ceilings through NVIDIA MPS
- **Memory Reservation**: Reserve GPU memory to prevent other processes from claiming it
- **Dynamic Memory Management**: Maintain constant memory usage even during script execution
- **Contention Experiments**: Measure whether a change actually improves saturated GPU resource share before relying on it

## Installation

```bash
pip install -e .
```

## Usage

### Check Driver Support and Preview Allocations

Inspect the selected physical GPU without creating a CUDA context or launching a
workload:

```bash
nvertake --device 0 doctor
nvertake --device 0 doctor --shares 20,30,50 --json
```

`doctor` reports the CUDA driver capability, allocatable SM count, partition
constraints, and the maximum number of SM resource groups. The maximum is a
driver-reported partition limit, not a promise that memory and work-queue
resources can sustain that many useful workloads.

Preview a `green-procs` split with the same driver path used by a real launch:

```bash
nvertake --device 0 green-procs \
  --shares 20,30,50 --dry-run \
  first.py second.py third.py
```

The JSON result includes `starts_processes: false` and the exact SM count for
each lane.

### Launch YAML Jobs and Monitor Them

[`examples/jobs.yaml`](examples/jobs.yaml) contains a complete single-GPU
configuration. A shortened form is:

```yaml
version: 1
logs_dir: ./runs
defaults:
  cwd: .
  device: 0
  env:
    PYTHONUNBUFFERED: "1"
jobs:
  - name: background
    script: worker.py
    args: ["--batch-size", "32"]
    sm_share: 30
    memory_share: 0.30
  - name: foreground
    script: worker.py
    args: ["--batch-size", "64"]
    sm_share: 70
    memory_share: 0.65
```

Validate scripts, directories, memory fractions, and driver-selected SM counts
without starting anything, then launch:

```bash
nvertake launch jobs.yaml --dry-run
nvertake launch jobs.yaml
```

Each launch creates a run directory with one log per job, one cooperative metric
file per job, and a live/final `report.json`. The launcher prints the run id and
report path. Inspect the latest local run, a particular run id, or a report file:

```bash
nvertake monitor
nvertake monitor RUN_ID --watch
nvertake monitor path/to/report.json --json
```

The monitor reports the job name, physical GPU, PID, assigned SM count,
framebuffer memory, latest workload throughput, and state. Add `device: 0` or
`device: 1` to each job to launch groups on both GPUs at the same time; see
[`examples/jobs-multi-gpu.yaml`](examples/jobs-multi-gpu.yaml).

Memory caps use `torch.cuda.set_per_process_memory_fraction` after the Green
Context is bound and before the user script runs. Set `memory_share` on every
job for a GPU; nVertake requires their sum to be at most `1.0`. This limits the
PyTorch caching allocator. Direct CUDA allocations and allocations made by
other libraries are outside this limit, and driver/context overhead still uses
memory.

### Calibrate Shares from Measured Throughput

No driver counter can define application throughput for an arbitrary script.
Jobs that opt into calibration therefore publish their own measurement:

```python
from nvertake import report_throughput

report_throughput(samples_per_second, unit="samples/s")
```

Enable the short concurrent benchmark in YAML:

```yaml
calibration:
  enabled: true
  duration: 5
  rounds: 2
  tolerance: 0.05
  damping: 0.5
```

During each round nVertake sets `NVERTAKE_CALIBRATION=1` and
`NVERTAKE_CALIBRATION_SECONDS`, collects the latest value from every job,
compares normalized throughput with each job's `target_share` (or `sm_share`),
and applies a damped correction before the real launch. Optional
`calibration_args` are appended only during those rounds. Calibration fails
clearly when a job does not report a positive value or when units differ on the
same GPU. Use calibration mode only when the script treats the calibration
environment as a short, side-effect-safe benchmark. `--no-calibrate` skips it,
and `--calibrate` enables it from the command line.

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

### Launch Multiple Python Processes with SM Shares

Launch every participating Python file in one command and give each process a
relative SM weight:

```bash
nvertake --device 0 green-procs \
  --shares 20,30,50 \
  --memory-shares 0.20,0.30,0.50 \
  --script-args '["--epochs", "5"]' \
  --script-args '["--batch-size", "64"]' \
  --script-args '[]' \
  first.py second.py third.py
```

The order of `--shares`, `--script-args`, and Python files is the same. Omit
`--script-args` entirely when none of the files needs arguments. The values are
relative weights, so `1,2,3` and `20,40,60` request the same split. There is no
fixed process count in nVertake; the practical limit is the number of resource
groups that the GPU driver can produce.

The launcher starts real OS processes, creates one Green Context in each, waits
until every context is ready, and then releases all files at the same time. Each
child sees the selected physical GPU as logical CUDA device `0`. The startup
message reports the child PIDs and exact driver-selected SM counts. If one child
fails, nVertake terminates the remaining children and returns a nonzero status.

Important constraints:

- This allocates SM sets, not exact wall-clock percentages. Throughput still
  depends on the kernels, memory bandwidth, and other GPU resources.
- All cooperating processes must be launched by the same `green-procs` command.
  It does not limit unrelated CUDA processes already running on the GPU.
- CUDA must not already be initialized before nVertake binds the context. Normal
  top-level initialization inside each launched file is safe. Do not make that
  file spawn additional CUDA worker processes; those descendants do not receive
  their own partition.
- Three or more lanes use the CUDA
  `CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING` mode for fine-grained
  allocation. This trades SM co-scheduling guarantees and advanced features
  such as large thread-block clusters for smaller partitions.
- Multi-process Green Context coordination is experimental. It worked on both
  tested machines, but NVIDIA primarily documents Green Context partitioning
  within one process; validate the real workload and driver before relying on it.

The corresponding Python API is `run_green_process_scripts(...)`.
See NVIDIA's [Green Context programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html)
for the underlying CUDA mechanism and its architecture-specific constraints.

### Partition SMs Between Two In-Process Tasks

CUDA Green Contexts are the strongest option in this project when both tasks
can run as Python callables in one process. The CUDA driver assigns each task a
separate set of SMs; no MPS daemon or kernel modification is required.

Put CUDA work inside callable functions. Do not initialize CUDA at module scope:

```python
# jobs.py
def _gemm(role, steps):
    import torch
    left = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    for _ in range(steps):
        torch.mm(left, right, out=output)
    torch.cuda.current_stream().synchronize()
    return {"role": role, "checksum": float(output[0, 0].item())}

def background(steps=100):
    return _gemm("background", steps)

def target(steps=100):
    return _gemm("target", steps)
```

Run both functions concurrently with a 25/75 requested split:

```bash
nvertake --device 0 green-run \
  --shares 25,75 \
  --task jobs:background \
  --task jobs:target \
  --task-kwargs '{"steps": 100}' \
  --task-kwargs '{"steps": 100}'
```

The JSON result reports both requested shares and the exact SM counts selected
by the driver. Hardware partition granularity means the actual split can differ
slightly: for example, the tested 110-SM Blackwell GPU maps 25/75 to 24/86 SMs.

Current constraints:

- exactly two cooperating callables in one Python process
- CUDA tensors must be created, used, and released within their own task lane
- CUDA tensors must not be exchanged between lanes or returned from a task
- changing shares requires completing the run and creating new contexts
- SM allocation is deterministic, but throughput and wall-clock shares remain
  workload-dependent

Use the Python API when a launcher needs more control:

```python
from nvertake import run_green_tasks

result = run_green_tasks(
    [background, target],
    device=0,
    shares=(25, 75),
    task_kwargs=({"steps": 100}, {"steps": 100}),
)
print(result.to_dict())
```

### Partition Separate Processes with MPS

Launch every competing task through nVertake and assign a lower active-thread
ceiling to the background task than to the target task:

```bash
# Background task: at most 25% of the GPU's active threads.
nvertake --device 0 --gpu-share 25 --mps-priority below-normal \
  exec python background.py

# Target task: at most 75%, with normal MPS client priority.
nvertake --device 0 --gpu-share 75 --mps-priority normal \
  run target.py --epochs 100
```

`--gpu-share` starts or reuses an nVertake-managed NVIDIA MPS daemon for the
selected physical GPU. The daemon maps that GPU to logical CUDA device 0 inside
each client. nVertake performs a short CUDA connection probe before launching
the task, so driver/MPS incompatibilities fail before user code starts.

The percentage is an execution-resource ceiling, not an exact wall-clock time
guarantee. For predictable contention, assign every cooperating process an
explicit share and keep the total near or below 100. A process can use less than
its ceiling when its workload cannot saturate the assigned resources.

The limit is fixed when a CUDA context is created. To change the share of a
running task, stop and relaunch that task with a new value.

Inspect and stop the per-device daemon with:

```bash
nvertake --device 0 mps status
nvertake --device 0 mps stop
```

`mps stop` refuses to stop a daemon with active clients. `--force` is available
for intentional interruption. `--filled` cannot be combined with MPS sharing;
memory reservation would run as a separate, unrestricted CUDA context and make
the requested compute split misleading.

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
  --gpu-share PERCENT   MPS active-thread ceiling (1-100)
  --mps-priority LEVEL  MPS priority hint: normal or below-normal
  --quiet, -q           Suppress info messages
  --version, -V         Show version

Commands:
  run SCRIPT [ARGS...]  Run a Python script with elevated priority
  exec COMMAND [ARGS...] Run any command with elevated priority environment
  doctor                Inspect Green Context driver capabilities
  launch JOBS.yaml      Launch a YAML job file across one or more GPUs
  monitor [RUN]         Inspect a live or completed JSON run report
  green-procs           Run Python files as weighted Green Context processes
  green-run             Run two callables in separate Green Context SM partitions
  info                  Show GPU information
  mps ACTION            Start, inspect, or stop the per-device MPS daemon
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

### MPS GPU Sharing

1. The MPS daemon is started with the selected physical GPU UUID and a 100%
   daemon-level ceiling.
2. Each child receives the same per-device MPS pipe directory and its own
   `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` value.
3. `CUDA_VISIBLE_DEVICES` is removed from clients because MPS already remaps the
   daemon's selected GPU to logical device 0.
4. `CUDA_MPS_CLIENT_PRIORITY` optionally maps a whole client to normal or
   below-normal priority. NVIDIA defines this as a scheduling hint.

This feature requires native Linux, a Volta-or-newer NVIDIA GPU, and
`nvidia-cuda-mps-control`. CUDA on WSL does not expose the Linux MPS daemon, so
`--gpu-share` reports the platform limitation there. The existing stream
priority hook remains available without MPS.

### CUDA Green Context SM Partitioning

1. nVertake asks the CUDA Driver API for the device's SM resource.
2. It splits that resource according to the requested weights, rounded to the
   driver-reported partition granularity.
3. `green-run` binds two contexts to Python worker threads in one process;
   `green-procs` binds one context in each launched Python process.
4. The tasks run concurrently, after which nVertake synchronizes and destroys
   the contexts.

This is a supported CUDA Driver API mechanism, not a driver bypass. It requires
a driver exposing the CUDA 12.4 Green Context API and a supported GPU. It works
on the tested native Linux and WSL systems. The Runtime API support used by
PyTorch remains experimental in nVertake, so test the actual workload before
relying on a particular throughput ratio.

`nvidia-smi` GPU utilization reaching 100% only means at least one kernel was
active throughout its sampling windows. It does not prove that every SM or
execution unit was fully occupied. Green Contexts control which SMs each lane
may use; they do not make an under-parallelized kernel fill those SMs.

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
- MPS 50/50 vs 25/75 share experiment:
  `bash test/run_tests_summary.sh mps-share --device 0 --output verification/results/mps_share_<date>.json`
- Green Context 50/50 vs 25/75 SM experiment:
  `bash test/run_tests_summary.sh green-share --device 0 --output verification/results/green_share_<date>.json`
- Multi-process Green Context experiment:
  `bash test/run_tests_summary.sh green-procs-share --device 0 --shares 20,30,50 --output verification/results/green_process_share_<date>.json`
- 2026-07-22 Green Context results:
  `verification/results/green_context_share_20260722_{406,gem12_wsl}.json`
- 2026-07-22 multi-process results:
  `verification/results/green_process_share{,_four}_20260722_{406,gem12_wsl}.json`
- Plot-ready result summary:
  `verification/results/resource_contention_summary_20260509.json`

See `test/README.md` for the test matrix, prerequisites, and how to interpret
the generated artifacts.

## Experimental Results

### Multiple Green Context processes (2026-07-22)

The public `green-procs` command launched separate Python PIDs running saturated
FP16 4096x4096 GEMM loops. The measured throughput distribution followed the
allocated SM distribution on native Linux and WSL:

| Host / GPU | Processes | Requested weights | Actual SM counts | Throughput shares |
|---|---:|---:|---:|---:|
| 406 / RTX PRO 5000 Blackwell | 3 | 20/30/50 | 22/34/54 | 22.25%/32.79%/44.96% |
| gem12 WSL / RTX 3090 Ti | 3 | 20/30/50 | 16/26/42 | 21.48%/31.81%/46.71% |
| 406 / RTX PRO 5000 Blackwell | 4 | 10/20/30/40 | 12/22/32/44 | 12.37%/19.39%/29.21%/39.02% |
| gem12 WSL / RTX 3090 Ti | 4 | 10/20/30/40 | 8/16/26/34 | 11.21%/20.04%/31.11%/37.64% |

The SM counts sum to the full device in every case. Small differences from the
requested weights come from the driver's partition granularity; throughput is
not expected to match SM percentage exactly.

### CUDA Green Context SM partitions (2026-07-22)

The public Green Context executor and `green-run` CLI were tested with two
concurrent FP16 4096x4096 GEMM loops. Each case used a synchronized three-second
measurement window:

| Host / GPU | Requested split | Actual SM split | Target TFLOP/s | Target throughput share |
|---|---:|---:|---:|---:|
| 406 / RTX PRO 5000 Blackwell | 50/50 | 56/54 | 86.72 | 49.21% |
| 406 / RTX PRO 5000 Blackwell | 25/75 | 24/86 | 140.45 | 78.06% |
| gem12 WSL / RTX 3090 Ti | 50/50 | 42/42 | 39.25 | 50.00% |
| gem12 WSL / RTX 3090 Ti | 25/75 | 20/64 | 63.27 | 76.15% |

Changing the requested split raised target-task throughput by about 61% on both
machines in this saturated GEMM test. The exact ratio is not a general
guarantee; kernels with insufficient blocks, memory-bandwidth limits, or other
bottlenecks can scale differently.

### Stream-priority and workload variants (2026-05-09)

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
- For `--gpu-share`: native Linux, Volta or newer, and NVIDIA MPS utilities
- For `green-run`: a driver exposing CUDA 12.4 Green Context APIs and a
  supported NVIDIA GPU
- For three or more `green-procs` lanes: a recent driver supporting
  fine-grained Green Context splits and combining resources from one split

## License

MIT
