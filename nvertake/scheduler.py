"""
Priority scheduling for NVIDIA GPU processes.
"""

import os
import sys
import subprocess
from contextlib import contextmanager
from typing import Optional, List

import torch

from .utils import logger


class PriorityScheduler:
    """
    Manages process priority for GPU workloads.
    
    Provides two mechanisms for priority scheduling:
    1. CPU process priority (nice value) - affects OS scheduler
    2. CUDA stream priority - affects GPU task scheduling
    """
    
    def __init__(self, device: int = 0, nice_value: int = -10):
        """
        Initialize the priority scheduler.
        
        Args:
            device: GPU device index to use
            nice_value: Nice value for CPU priority (-20 to 19, lower = higher priority)
        """
        self.device = device
        self.nice_value = nice_value
        self._original_nice = None
        self._high_priority_stream: Optional[torch.cuda.Stream] = None
        
    def set_cpu_priority(self) -> bool:
        """
        Set the CPU process priority using nice value.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._original_nice = os.nice(0)  # Get current nice value
            # nice() adds to current value, so we need to calculate the delta
            delta = self.nice_value - self._original_nice
            os.nice(delta)
            logger.info(f"Set process nice value to {self.nice_value}")
            return True
        except PermissionError:
            logger.warning(
                f"Cannot set nice value to {self.nice_value} (requires root). "
                "Running with default priority."
            )
            return False
        except OSError as e:
            logger.warning(f"Failed to set nice value: {e}")
            return False
    
    def restore_cpu_priority(self) -> None:
        """Restore the original CPU priority."""
        if self._original_nice is not None:
            try:
                current = os.nice(0)
                delta = self._original_nice - current
                os.nice(delta)
                logger.debug(f"Restored nice value to {self._original_nice}")
            except OSError:
                pass
    
    def get_high_priority_stream(self) -> torch.cuda.Stream:
        """
        Get or create a high-priority CUDA stream.
        
        CUDA streams have two priority levels:
        - High priority: -1
        - Low priority: 0
        
        Returns:
            A high-priority CUDA stream
        """
        if self._high_priority_stream is None:
            if torch.cuda.is_available():
                with torch.cuda.device(self.device):
                    # Priority -1 is high priority, 0 is low priority
                    self._high_priority_stream = torch.cuda.Stream(
                        device=self.device,
                        priority=-1  # High priority
                    )
                    logger.info(f"Created high-priority CUDA stream on device {self.device}")
            else:
                logger.warning("CUDA not available, cannot create high-priority stream")
        return self._high_priority_stream
    
    @contextmanager
    def priority_context(self):
        """
        Context manager that sets up priority scheduling.
        
        Usage:
            scheduler = PriorityScheduler()
            with scheduler.priority_context():
                # Your GPU code here
                pass
        """
        # Set CPU priority
        self.set_cpu_priority()
        
        # Get high-priority stream
        stream = self.get_high_priority_stream()
        
        try:
            if stream is not None:
                with torch.cuda.stream(stream):
                    yield stream
            else:
                yield None
        finally:
            self.restore_cpu_priority()


def run_with_priority(
    script_path: str,
    script_args: List[str],
    device: int = 0,
    nice_value: int = -10,
) -> int:
    """
    Run a Python script with elevated priority.
    
    Args:
        script_path: Path to the Python script to run
        script_args: Arguments to pass to the script
        device: GPU device to use
        nice_value: Nice value for CPU priority
        
    Returns:
        Exit code of the script
    """
    scheduler = PriorityScheduler(device=device, nice_value=nice_value)
    
    # Set our process priority (child will inherit)
    scheduler.set_cpu_priority()
    
    # Set CUDA_VISIBLE_DEVICES if specific device requested
    env = os.environ.copy()
    if device >= 0:
        env['CUDA_VISIBLE_DEVICES'] = str(device)
        logger.info(f"Set CUDA_VISIBLE_DEVICES={device}")
    
    # Build command
    cmd = [sys.executable, script_path] + script_args
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, env=env)
        return result.returncode
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Failed to run script: {e}")
        return 1
    finally:
        scheduler.restore_cpu_priority()
