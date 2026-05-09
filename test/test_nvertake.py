"""
Unit tests for nVertake package.
"""

import os
import sys
import unittest
from contextlib import contextmanager
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

    @patch("nvertake.scheduler.PriorityScheduler")
    def test_inject_priority_decorator(self, mock_scheduler_cls):
        """Test @inject_priority decorator wiring."""
        from nvertake.scheduler import inject_priority

        @contextmanager
        def dummy_ctx():
            yield None

        scheduler = MagicMock()
        scheduler.priority_context.return_value = dummy_ctx()
        mock_scheduler_cls.return_value = scheduler

        @inject_priority(device=2, nice_value=-5)
        def add_one(x):
            return x + 1

        self.assertEqual(add_one(41), 42)
        mock_scheduler_cls.assert_called_once_with(device=2, nice_value=-5)
        scheduler.priority_context.assert_called_once()


class TestAutoPriority(unittest.TestCase):
    """Tests for low-intrusion PyTorch priority injection."""

    def test_enable_torch_priority_sets_high_priority_stream(self):
        from nvertake import auto_priority

        class DummyDeviceContext:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        class DummyStream:
            def __init__(self, device, priority):
                self.device = device
                self.priority = priority

        class DummyCuda:
            def __init__(self):
                self.selected_stream = None

            def is_available(self):
                return True

            def current_device(self):
                return 0

            def get_stream_priority_range(self):
                return (0, -1)

            def Stream(self, device, priority):
                return DummyStream(device, priority)

            def device(self, device):
                return DummyDeviceContext()

            def set_stream(self, stream):
                self.selected_stream = stream

            def current_stream(self, device_arg=None):
                return self.selected_stream

        class DummyTorch:
            cuda = DummyCuda()

        auto_priority._STREAMS.clear()

        with patch.dict(os.environ, {}, clear=False):
            self.assertTrue(auto_priority.enable_torch_priority(DummyTorch, device=3))

        stream = DummyTorch.cuda.selected_stream
        self.assertIsNotNone(stream)
        self.assertEqual(stream.device, 3)
        self.assertEqual(stream.priority, -1)


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

    def test_parse_exec_command(self):
        """Test parsing exec command."""
        from nvertake.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(['exec', 'torchrun', '--nproc_per_node=2', 'train.py'])

        self.assertEqual(args.command, 'exec')
        self.assertEqual(args.command_args, ['torchrun', '--nproc_per_node=2', 'train.py'])
    
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

    def test_parse_no_torch_priority_option(self):
        """Test parsing --no-torch-priority option."""
        from nvertake.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(['--no-torch-priority', 'run', 'test.py'])

        self.assertTrue(args.no_torch_priority)
    
    def test_parse_info_command(self):
        """Test parsing info command."""
        from nvertake.cli import create_parser
        parser = create_parser()
        args = parser.parse_args(['info'])
        
        self.assertEqual(args.command, 'info')

    def test_configure_auto_priority_env(self):
        """Test child env setup for PyTorch auto-priority injection."""
        from nvertake.cli import configure_auto_priority_env

        env = {"PYTHONPATH": "/existing"}
        configured = configure_auto_priority_env(env, device=2, quiet=True)

        self.assertEqual(configured["NVERTAKE_AUTO_PRIORITY"], "1")
        self.assertEqual(configured["NVERTAKE_AUTO_PRIORITY_DEVICE"], "0")
        self.assertEqual(configured["NVERTAKE_AUTO_PRIORITY_PHYSICAL_DEVICE"], "2")
        self.assertEqual(configured["NVERTAKE_AUTO_PRIORITY_QUIET"], "1")
        self.assertTrue(configured["PYTHONPATH"].endswith(os.pathsep + "/existing"))


if __name__ == '__main__':
    unittest.main()
