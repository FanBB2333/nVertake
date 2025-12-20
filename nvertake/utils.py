"""
Utility functions for nVertake.
"""

import logging
import os
import subprocess
from typing import Dict, Optional

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nvertake")


def get_default_device() -> torch.device:
    """Get the default CUDA device if available, otherwise CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_gpu_memory(device: int = 0) -> Dict[str, int]:
    """
    Get GPU memory statistics for a specific device.
    
    Args:
        device: GPU device index
        
    Returns:
        Dictionary with 'total', 'used', and 'free' memory in MiB
    """
    if isinstance(device, torch.device):
        device = device.index if device.index is not None else 0
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
             '--format=csv,noheader,nounits', f'--id={device}'],
            capture_output=True,
            text=True,
            check=True
        )
        values = result.stdout.strip().split(',')
        return {
            'total': int(values[0].strip()),
            'used': int(values[1].strip()),
            'free': int(values[2].strip()),
        }
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError) as e:
        logger.warning(f"Failed to get GPU memory via nvidia-smi: {e}")
        # Fallback to torch.cuda
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            total = torch.cuda.get_device_properties(device).total_memory // (1024 * 1024)
            allocated = torch.cuda.memory_allocated(device) // (1024 * 1024)
            reserved = torch.cuda.memory_reserved(device) // (1024 * 1024)
            return {
                'total': total,
                'used': reserved,
                'free': total - reserved,
            }
        raise RuntimeError("No GPU available or nvidia-smi not found")


def get_gpu_count() -> int:
    """Get the number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def validate_device(device: int) -> bool:
    """Check if a GPU device index is valid."""
    gpu_count = get_gpu_count()
    return 0 <= device < gpu_count


def bytes_to_mib(bytes_val: int) -> float:
    """Convert bytes to MiB."""
    return bytes_val / (1024 * 1024)


def mib_to_bytes(mib_val: float) -> int:
    """Convert MiB to bytes."""
    return int(mib_val * 1024 * 1024)
