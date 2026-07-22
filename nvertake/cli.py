"""
Command line interface for nVertake.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .memory import MemoryManager, fill_gpu_memory
from .mps import MPSControlError, MPSController
from .scheduler import PriorityScheduler
from .utils import logger, get_gpu_memory, get_gpu_count, validate_device


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for nVertake CLI."""
    parser = argparse.ArgumentParser(
        prog='nvertake',
        description='Preemptive scheduling and memory reservation for NVIDIA GPUs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a script with elevated GPU priority
  nvertake run train.py --epochs 100

  # Run a launcher such as torchrun with the same priority environment
  nvertake exec torchrun --nproc_per_node=4 train.py

  # Give two cooperating MPS tasks 25% and 75% active-thread ceilings
  nvertake --gpu-share 25 exec python background.py
  nvertake --gpu-share 75 run target.py

  # Inspect or stop the per-device MPS daemon
  nvertake --device 0 mps status
  nvertake --device 0 mps stop

  # Disable PyTorch high-priority stream auto-injection
  nvertake --no-torch-priority run train.py

  # Run with 95% memory reservation
  nvertake --filled 0.95 run train.py

  # Just fill GPU memory (standalone mode)
  nvertake --filled 0.95

  # Specify GPU device
  nvertake --device 1 --filled 0.8 run inference.py
        """,
    )
    
    parser.add_argument(
        '--version', '-V',
        action='version',
        version=f'nvertake {__version__}',
    )
    
    parser.add_argument(
        '--filled', '-f',
        type=float,
        default=None,
        metavar='RATIO',
        help='Fill GPU memory to this ratio (0.0-1.0). If no run command, '
             'just fills memory and waits.',
    )
    
    parser.add_argument(
        '--device', '-d',
        type=int,
        default=0,
        metavar='GPU_ID',
        help='GPU device to use (default: 0)',
    )
    
    parser.add_argument(
        '--nice', '-n',
        type=int,
        default=-10,
        metavar='VALUE',
        help='Nice value for CPU priority (-20 to 19, default: -10)',
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress info messages',
    )

    parser.add_argument(
        '--no-torch-priority',
        action='store_true',
        help='Disable PyTorch high-priority stream auto-injection for `run`/`exec`.',
    )

    parser.add_argument(
        '--gpu-share', '--mps-active-thread-percentage',
        dest='gpu_share',
        type=int,
        default=None,
        metavar='PERCENT',
        help='Enable NVIDIA MPS and cap this client to 1-100%% of active threads. '
             'Launch every competing task with an explicit share.',
    )

    parser.add_argument(
        '--mps-priority',
        choices=('normal', 'below-normal'),
        default=None,
        help='MPS client priority hint. This is not an execution-order guarantee.',
    )

    parser.add_argument(
        '--mps-pipe-directory',
        default=None,
        metavar='PATH',
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        '--mps-log-directory',
        default=None,
        metavar='PATH',
        help=argparse.SUPPRESS,
    )
    
    # Subcommand for running scripts
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    run_parser = subparsers.add_parser(
        'run',
        help='Run a Python script with elevated priority',
    )
    run_parser.add_argument(
        'script',
        help='Python script to run',
    )
    run_parser.add_argument(
        'script_args',
        nargs=argparse.REMAINDER,
        help='Arguments to pass to the script',
    )

    exec_parser = subparsers.add_parser(
        'exec',
        help='Run any command with elevated priority environment',
    )
    exec_parser.add_argument(
        'command_args',
        nargs=argparse.REMAINDER,
        help='Command and arguments to execute',
    )
    
    # Info subcommand
    subparsers.add_parser(
        'info',
        help='Show GPU information',
    )

    mps_parser = subparsers.add_parser(
        'mps',
        help='Manage the per-device NVIDIA MPS daemon',
    )
    mps_parser.add_argument(
        'mps_action',
        choices=('start', 'status', 'stop'),
        help='Daemon action',
    )
    mps_parser.add_argument(
        '--force',
        action='store_true',
        help='Allow `mps stop` while client processes are active',
    )
    
    return parser


def _prepend_env_path(existing: Optional[str], prefix: str) -> str:
    if not existing:
        return prefix
    return prefix + os.pathsep + existing


def _auto_priority_bootstrap_dir() -> str:
    return str(Path(__file__).resolve().parent / "_bootstrap")


def configure_auto_priority_env(
    env: dict,
    *,
    device: int,
    quiet: bool = False,
) -> dict:
    """
    Configure a child Python process to install nVertake's PyTorch hook.

    `CUDA_VISIBLE_DEVICES` maps the selected physical GPU to logical device 0
    inside the child process, so the injected stream targets device 0 there.
    """
    env["NVERTAKE_AUTO_PRIORITY"] = "1"
    env["NVERTAKE_AUTO_PRIORITY_DEVICE"] = "0"
    env["NVERTAKE_AUTO_PRIORITY_PHYSICAL_DEVICE"] = str(device)
    if quiet:
        env["NVERTAKE_AUTO_PRIORITY_QUIET"] = "1"
    env["PYTHONPATH"] = _prepend_env_path(
        env.get("PYTHONPATH"),
        _auto_priority_bootstrap_dir(),
    )
    return env


def _mps_requested(args: argparse.Namespace) -> bool:
    return args.gpu_share is not None or args.mps_priority is not None


def _mps_controller(args: argparse.Namespace, *, device: int) -> MPSController:
    return MPSController(
        device=device,
        pipe_directory=args.mps_pipe_directory,
        log_directory=args.mps_log_directory,
    )


def _build_child_env(args: argparse.Namespace, *, device: int) -> dict:
    env = os.environ.copy()
    if _mps_requested(args):
        controller = _mps_controller(args, device=device)
        started = controller.ensure_started()
        controller.client_environment(
            env,
            active_thread_percentage=args.gpu_share,
            client_priority=args.mps_priority,
        )
        try:
            probe = controller.probe_client(env)
        except (MPSControlError, OSError, subprocess.SubprocessError):
            if started:
                try:
                    controller.stop(force=True)
                except (MPSControlError, OSError, subprocess.SubprocessError) as cleanup_error:
                    logger.warning("Failed to stop MPS after client probe error: %s", cleanup_error)
            raise
        action = "started" if started else "reused"
        share = args.gpu_share if args.gpu_share is not None else 100
        logger.info(
            "MPS %s for GPU %s; client share=%s%%, visible SMs=%s",
            action,
            device,
            share,
            probe.get("sm_count", "unknown"),
        )
    else:
        env['CUDA_VISIBLE_DEVICES'] = str(device)
    if not args.no_torch_priority:
        configure_auto_priority_env(env, device=device, quiet=bool(args.quiet))
    return env


def cmd_info(args: argparse.Namespace) -> int:
    """Show GPU information."""
    gpu_count = get_gpu_count()
    
    if gpu_count == 0:
        print("No NVIDIA GPUs detected.")
        return 1
    
    print(f"Detected {gpu_count} NVIDIA GPU(s):\n")
    
    for i in range(gpu_count):
        try:
            mem = get_gpu_memory(i)
            print(f"GPU {i}:")
            print(f"  Total Memory:  {mem['total']:,} MiB")
            print(f"  Used Memory:   {mem['used']:,} MiB")
            print(f"  Free Memory:   {mem['free']:,} MiB")
            print(f"  Usage:         {mem['used']/mem['total']*100:.1f}%")
            print()
        except Exception as e:
            print(f"GPU {i}: Error getting info - {e}\n")
    
    return 0


def cmd_run(
    args: argparse.Namespace,
    fill_ratio: Optional[float] = None,
) -> int:
    """Run a script with priority scheduling and optional memory filling."""
    device = args.device
    nice_value = args.nice
    script = args.script
    script_args = args.script_args or []
    
    # Validate device
    if not _mps_requested(args) and not validate_device(device):
        logger.error(f"Invalid GPU device: {device}")
        return 1
    
    # Check script exists
    if not os.path.exists(script):
        logger.error(f"Script not found: {script}")
        return 1
    
    # Set up priority scheduler
    scheduler = PriorityScheduler(device=device, nice_value=nice_value)
    scheduler.set_cpu_priority()
    
    # Memory manager if --filled specified
    memory_manager: Optional[MemoryManager] = None
    if fill_ratio is not None:
        memory_manager = MemoryManager(
            device=device,
            fill_ratio=fill_ratio,
        )
        memory_manager.fill_memory()
        memory_manager.start_monitor()
        logger.info(f"Memory manager active, maintaining {fill_ratio*100:.1f}% usage")
    
    try:
        env = _build_child_env(args, device=device)
        
        # Build command
        cmd = [sys.executable, script] + script_args
        logger.info(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, env=env)
        return result.returncode
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Failed to run script: {e}")
        return 1
    finally:
        if memory_manager is not None:
            memory_manager.stop_monitor()
            memory_manager.release_memory()
        scheduler.restore_cpu_priority()


def cmd_exec(
    args: argparse.Namespace,
    fill_ratio: Optional[float] = None,
) -> int:
    """Run an arbitrary command with priority env and optional memory filling."""
    device = args.device
    nice_value = args.nice
    command_args = args.command_args or []

    if not command_args:
        logger.error("No command provided for `exec`")
        return 1

    if not _mps_requested(args) and not validate_device(device):
        logger.error(f"Invalid GPU device: {device}")
        return 1

    scheduler = PriorityScheduler(device=device, nice_value=nice_value)
    scheduler.set_cpu_priority()

    memory_manager: Optional[MemoryManager] = None
    if fill_ratio is not None:
        memory_manager = MemoryManager(
            device=device,
            fill_ratio=fill_ratio,
        )
        memory_manager.fill_memory()
        memory_manager.start_monitor()
        logger.info(f"Memory manager active, maintaining {fill_ratio*100:.1f}% usage")

    try:
        env = _build_child_env(args, device=device)
        logger.info(f"Running: {' '.join(command_args)}")
        result = subprocess.run(command_args, env=env)
        return result.returncode
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Failed to run command: {e}")
        return 1
    finally:
        if memory_manager is not None:
            memory_manager.stop_monitor()
            memory_manager.release_memory()
        scheduler.restore_cpu_priority()


def cmd_filled_only(args: argparse.Namespace) -> int:
    """Fill GPU memory without running a script."""
    device = args.device
    fill_ratio = args.filled
    
    if not validate_device(device):
        logger.error(f"Invalid GPU device: {device}")
        return 1
    
    logger.info(f"Filling GPU {device} to {fill_ratio*100:.1f}%...")
    fill_gpu_memory(device=device, fill_ratio=fill_ratio, block=True)
    return 0


def cmd_mps(args: argparse.Namespace) -> int:
    """Start, inspect, or stop the per-device NVIDIA MPS daemon."""
    controller = _mps_controller(args, device=args.device)
    try:
        if args.mps_action == 'start':
            started = controller.start()
            print("MPS daemon started." if started else "MPS daemon is already running.")
            print(f"Pipe directory: {controller.paths.pipe_directory}")
            print(f"Log directory:  {controller.paths.log_directory}")
            return 0

        if args.mps_action == 'stop':
            stopped = controller.stop(force=bool(args.force))
            print("MPS daemon stopped." if stopped else "MPS daemon is not running.")
            return 0

        status = controller.status()
        availability = "available" if status.available else "unavailable"
        state = "running" if status.running else "stopped"
        print(f"MPS: {state} ({availability})")
        print(f"Pipe directory: {status.pipe_directory}")
        print(f"Log directory:  {status.log_directory}")
        if status.server_pids:
            print("Server PIDs: " + ", ".join(str(pid) for pid in status.server_pids))
        if status.client_pids:
            print("Client PIDs: " + ", ".join(str(pid) for pid in status.client_pids))
        if status.detail:
            print(f"Detail: {status.detail}")
        return 0 if status.available else 1
    except MPSControlError as exc:
        logger.error(str(exc))
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for nVertake CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Set logging level
    if args.quiet:
        logger.setLevel('WARNING')
    
    # Validate --filled range
    if args.filled is not None and not (0.0 < args.filled <= 1.0):
        parser.error("--filled must be between 0.0 and 1.0")

    if args.gpu_share is not None and not (1 <= args.gpu_share <= 100):
        parser.error("--gpu-share must be an integer between 1 and 100")

    if _mps_requested(args) and args.command not in {'run', 'exec'}:
        parser.error("--gpu-share/--mps-priority can only be used with `run` or `exec`")

    if _mps_requested(args) and args.filled is not None:
        parser.error("--filled cannot be combined with NVIDIA MPS sharing")
    
    # Route to appropriate command
    if args.command == 'info':
        return cmd_info(args)
    elif args.command == 'mps':
        return cmd_mps(args)
    elif args.command == 'run':
        return cmd_run(args, fill_ratio=args.filled)
    elif args.command == 'exec':
        return cmd_exec(args, fill_ratio=args.filled)
    elif args.filled is not None:
        # Standalone --filled mode (no run command)
        return cmd_filled_only(args)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
