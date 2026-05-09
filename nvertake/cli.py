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

  # Disable PyTorch high-priority stream auto-injection
  nvertake --no-torch-priority run train.py

  # Run with 95%% memory reservation
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
    info_parser = subparsers.add_parser(
        'info',
        help='Show GPU information',
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


def _build_child_env(args: argparse.Namespace, *, device: int) -> dict:
    env = os.environ.copy()
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
    if not validate_device(device):
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

    if not validate_device(device):
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
    
    # Route to appropriate command
    if args.command == 'info':
        return cmd_info(args)
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
