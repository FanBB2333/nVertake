"""
GPU memory management and reservation for nVertake.
"""

import signal
import sys
import threading
import time
from typing import Optional, List

import torch

from .utils import logger, get_gpu_memory


class MemoryManager:
    """
    Manages GPU memory reservation to prevent other processes from claiming it.
    
    This class can:
    1. Fill a portion of GPU memory
    2. Maintain constant memory usage while a script runs
    3. Dynamically adjust buffer size to maintain target memory level
    """
    
    def __init__(
        self,
        device: int = 0,
        fill_ratio: float = 0.95,
        check_interval: float = 0.5,
    ):
        """
        Initialize the memory manager.
        
        Args:
            device: GPU device index
            fill_ratio: Target ratio of GPU memory to occupy (0.0 to 1.0)
            check_interval: Seconds between memory checks in monitor mode
        """
        self.device = device
        self.fill_ratio = fill_ratio
        self.check_interval = check_interval
        
        self._buffer_tensors: List[torch.Tensor] = []
        self._target_memory_mib: Optional[int] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
    def calculate_target_memory(self) -> int:
        """
        Calculate target memory in MiB based on fill ratio.
        
        Returns:
            Target memory in MiB
        """
        mem_info = get_gpu_memory(self.device)
        total = mem_info['total']
        return int(total * self.fill_ratio)
    
    def _allocate_tensor(self, size_mib: int) -> Optional[torch.Tensor]:
        """
        Allocate a tensor of specified size on GPU.
        
        Args:
            size_mib: Size in MiB
            
        Returns:
            Allocated tensor or None on failure
        """
        if size_mib <= 0:
            return None
            
        try:
            # float32 = 4 bytes per element
            num_elements = (size_mib * 1024 * 1024) // 4
            with torch.cuda.device(self.device):
                tensor = torch.empty(
                    int(num_elements),
                    dtype=torch.float32,
                    device=f'cuda:{self.device}'
                )
            return tensor
        except RuntimeError as e:
            logger.debug(f"Failed to allocate {size_mib} MiB: {e}")
            return None
    
    def fill_memory(self) -> int:
        """
        Fill GPU memory up to the target ratio.
        
        Returns:
            Amount of memory allocated in MiB
        """
        with self._lock:
            mem_info = get_gpu_memory(self.device)
            self._target_memory_mib = self.calculate_target_memory()
            
            # Calculate how much we need to allocate
            current_used = mem_info['used']
            need_to_allocate = self._target_memory_mib - current_used
            
            if need_to_allocate <= 0:
                logger.info(
                    f"GPU {self.device} already using {current_used} MiB, "
                    f"target is {self._target_memory_mib} MiB"
                )
                return 0
            
            logger.info(
                f"Allocating {need_to_allocate} MiB on GPU {self.device} "
                f"(target: {self._target_memory_mib} MiB, {self.fill_ratio*100:.1f}%)"
            )
            
            # Allocate in chunks to be more flexible
            chunk_size = min(1024, need_to_allocate)  # 1GB chunks max
            allocated = 0
            
            while allocated < need_to_allocate:
                remaining = need_to_allocate - allocated
                size = min(chunk_size, remaining)
                
                tensor = self._allocate_tensor(size)
                if tensor is None:
                    # Try smaller chunks
                    if chunk_size > 64:
                        chunk_size //= 2
                        continue
                    else:
                        break
                
                self._buffer_tensors.append(tensor)
                allocated += size
            
            logger.info(f"Allocated {allocated} MiB on GPU {self.device}")
            return allocated
    
    def _adjust_memory(self) -> None:
        """Adjust buffer size to maintain target memory level."""
        if self._target_memory_mib is None:
            return
            
        with self._lock:
            mem_info = get_gpu_memory(self.device)
            current_used = mem_info['used']
            diff = self._target_memory_mib - current_used
            
            if diff > 64:  # Need to allocate more (threshold: 64 MiB)
                tensor = self._allocate_tensor(diff)
                if tensor is not None:
                    self._buffer_tensors.append(tensor)
                    logger.debug(f"Allocated additional {diff} MiB")
            elif diff < -64:  # Need to free some (threshold: 64 MiB)
                # Free some buffer tensors
                to_free = abs(diff)
                freed = 0
                while self._buffer_tensors and freed < to_free:
                    tensor = self._buffer_tensors.pop()
                    tensor_size = tensor.numel() * 4 // (1024 * 1024)
                    del tensor
                    freed += tensor_size
                if freed > 0:
                    torch.cuda.empty_cache()
                    logger.debug(f"Freed {freed} MiB")
    
    def _monitor_loop(self) -> None:
        """Background thread loop to maintain memory level."""
        while not self._stop_event.is_set():
            try:
                self._adjust_memory()
            except Exception as e:
                logger.debug(f"Error in memory monitor: {e}")
            self._stop_event.wait(self.check_interval)
    
    def start_monitor(self) -> None:
        """Start the background memory monitor thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
            
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="nvertake-memory-monitor"
        )
        self._monitor_thread.start()
        logger.info("Started memory monitor thread")
    
    def stop_monitor(self) -> None:
        """Stop the background memory monitor thread."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        logger.info("Stopped memory monitor thread")
    
    def release_memory(self) -> None:
        """Release all allocated buffer tensors."""
        with self._lock:
            count = len(self._buffer_tensors)
            self._buffer_tensors.clear()
            torch.cuda.empty_cache()
            logger.info(f"Released {count} buffer tensors")
    
    def __enter__(self):
        """Context manager entry."""
        self.fill_memory()
        self.start_monitor()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitor()
        self.release_memory()
        return False


def fill_gpu_memory(
    device: int = 0,
    fill_ratio: float = 0.95,
    block: bool = True,
) -> Optional[MemoryManager]:
    """
    Fill GPU memory and optionally block until interrupted.
    
    This is the standalone mode for `nvertake --filled` without a run command.
    
    Args:
        device: GPU device index
        fill_ratio: Ratio of GPU memory to fill
        block: If True, block until KeyboardInterrupt
        
    Returns:
        MemoryManager instance if not blocking, None otherwise
    """
    manager = MemoryManager(device=device, fill_ratio=fill_ratio)
    manager.fill_memory()
    
    if block:
        logger.info(
            f"GPU {device} memory filled to {fill_ratio*100:.1f}%. "
            "Press Ctrl+C to exit."
        )
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Received interrupt signal, releasing memory...")
            manager.release_memory()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            manager.release_memory()
        
        return None
    
    return manager
