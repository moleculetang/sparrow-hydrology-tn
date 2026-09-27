import unittest
from unittest.mock import patch

from d29_platform import runtime


class RuntimeResourceGateTests(unittest.TestCase):
    def measurements(self, memory_percent):
        return dict(physical_used_percent=memory_percent,
                    commit_used_percent=memory_percent,
                    physical_available_bytes=10_000_000_000,
                    commit_available_bytes=10_000_000_000)

    def test_full_cpu_allowed_while_memory_is_below_gate(self):
        with patch.object(runtime, 'resources', return_value=self.measurements(89.9)), \
             patch.object(runtime, 'cpu_percent', return_value=100.):
            self.assertTrue(runtime.dispatch_allowed(reserve_bytes=9_000_000_000)[0])

    def test_memory_pause_and_hysteresis_remain(self):
        with patch.object(runtime, 'cpu_percent', return_value=100.):
            with patch.object(runtime, 'resources', return_value=self.measurements(90.)):
                self.assertFalse(runtime.dispatch_allowed()[0])
            with patch.object(runtime, 'resources', return_value=self.measurements(85.)):
                self.assertFalse(runtime.dispatch_allowed(paused=True)[0])
            with patch.object(runtime, 'resources', return_value=self.measurements(84.9)):
                self.assertTrue(runtime.dispatch_allowed(paused=True)[0])


if __name__ == '__main__':
    unittest.main()
