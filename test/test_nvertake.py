"""
Unit tests for nVertake package.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path to import nvertake
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestUtils(unittest.TestCase):
    """Tests for utility functions."""
    
    def test_bytes_to_mib(self):
        """Test bytes to MiB conversion."""
        from nvertake.utils import bytes_to_mib
        self.assertEqual(bytes_to_mib(1024 * 1024), 1.0)
        self.assertEqual(bytes_to_mib(2 * 1024 * 1024), 2.0)
    
    def test_mib_to_bytes(self):
        """Test MiB to bytes conversion."""
        from nvertake.utils import mib_to_bytes
        self.assertEqual(mib_to_bytes(1.0), 1024 * 1024)
        self.assertEqual(mib_to_bytes(2.0), 2 * 1024 * 1024)
    
    @patch('nvertake.utils.torch')
    def test_get_gpu_count_no_cuda(self, mock_torch):
        """Test GPU count when CUDA is not available."""
        from nvertake.utils import get_gpu_count
        mock_torch.cuda.is_available.return_value = False
        self.assertEqual(get_gpu_count(), 0)
    
    @patch('nvertake.utils.torch')
    def test_validate_device_no_cuda(self, mock_torch):
        """Test device validation when CUDA is not available."""
        from nvertake.utils import validate_device
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.device_count.return_value = 0
        self.assertFalse(validate_device(0))


class TestScheduler(unittest.TestCase):
    """Tests for PriorityScheduler."""
    
    @patch('nvertake.scheduler.torch')
    def test_scheduler_init(self, mock_torch):
        """Test scheduler initialization."""
        from nvertake.scheduler import PriorityScheduler
        scheduler = PriorityScheduler(device=1, nice_value=-5)
        self.assertEqual(scheduler.device, 1)
        self.assertEqual(scheduler.nice_value, -5)
    
    @patch('nvertake.scheduler.os.nice')
    def test_set_cpu_priority(self, mock_nice):
        """Test setting CPU priority."""
        from nvertake.scheduler import PriorityScheduler
        mock_nice.return_value = 0
        
        scheduler = PriorityScheduler(nice_value=-10)
        result = scheduler.set_cpu_priority()
        
        # Should call nice twice: once to get current, once to set
        self.assertEqual(mock_nice.call_count, 2)
        self.assertTrue(result)
    
    @patch('nvertake.scheduler.os.nice')
    def test_set_cpu_priority_permission_error(self, mock_nice):
        """Test handling permission error when setting priority."""
        from nvertake.scheduler import PriorityScheduler
        mock_nice.side_effect = PermissionError("Permission denied")
        
        scheduler = PriorityScheduler(nice_value=-10)
        result = scheduler.set_cpu_priority()
        
        self.assertFalse(result)


class TestMemoryManager(unittest.TestCase):
    """Tests for MemoryManager."""
    
    def test_memory_manager_init(self):
        """Test memory manager initialization."""
        from nvertake.memory import MemoryManager
        manager = MemoryManager(device=2, fill_ratio=0.8)
        self.assertEqual(manager.device, 2)
        self.assertEqual(manager.fill_ratio, 0.8)
    
    @patch('nvertake.memory.get_gpu_memory')
    def test_calculate_target_memory(self, mock_get_gpu_memory):
        """Test target memory calculation."""
        from nvertake.memory import MemoryManager
        mock_get_gpu_memory.return_value = {'total': 10000, 'used': 1000, 'free': 9000}
        
        manager = MemoryManager(fill_ratio=0.95)
        target = manager.calculate_target_memory()
        
        self.assertEqual(target, 9500)  # 10000 * 0.95


class TestCLI(unittest.TestCase):
    """Tests for CLI."""
    
    def test_create_parser(self):
        """Test parser creation."""
        from nvertake.cli import create_parser
        parser = create_parser()
        self.assertIsNotNone(parser)
    
    def test_parse_run_command(self):
        """Test parsing run command."""
        from nvertake.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(['run', 'test.py', '--arg1', 'value1'])
        
        self.assertEqual(args.command, 'run')
        self.assertEqual(args.script, 'test.py')
        self.assertEqual(args.script_args, ['--arg1', 'value1'])
    
    def test_parse_filled_option(self):
        """Test parsing --filled option."""
        from nvertake.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(['--filled', '0.95', 'run', 'test.py'])
        
        self.assertEqual(args.filled, 0.95)
        self.assertEqual(args.command, 'run')
    
    def test_parse_device_option(self):
        """Test parsing --device option."""
        from nvertake.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(['--device', '2', 'run', 'test.py'])
        
        self.assertEqual(args.device, 2)
    
    def test_parse_info_command(self):
        """Test parsing info command."""
        from nvertake.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(['info'])
        
        self.assertEqual(args.command, 'info')


if __name__ == '__main__':
    unittest.main()
