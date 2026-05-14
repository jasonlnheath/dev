"""Tests for load balancer configuration correctness."""

import json
import os
import unittest


class TestLBConfig(unittest.TestCase):
    """Verify lb_config.json has correct backend entries."""

    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "lb_config.json")

    def test_config_loads(self):
        """Config file is valid JSON."""
        with open(self.CONFIG_PATH) as f:
            cfg = json.load(f)
        self.assertIsInstance(cfg, dict)

    def test_has_backends(self):
        """Config has at least 2 backends."""
        with open(self.CONFIG_PATH) as f:
            cfg = json.load(f)
        self.assertIn("backends", cfg)
        self.assertGreaterEqual(len(cfg["backends"]), 2)

    def test_local_backend_port_8033(self):
        """Local backend routes to port 8033."""
        with open(self.CONFIG_PATH) as f:
            cfg = json.load(f)
        local = [b for b in cfg["backends"] if b.get("name") == "local"]
        self.assertEqual(len(local), 1)
        self.assertEqual(local[0]["port"], 8033)
        self.assertEqual(local[0]["host"], "127.0.0.1")

    def test_remote_backend_correct_ip(self):
        """Remote (Charlie's) backend uses correct IP 192.168.1.201."""
        with open(self.CONFIG_PATH) as f:
            cfg = json.load(f)
        remote = [b for b in cfg["backends"] if b.get("name") == "remote"]
        self.assertEqual(len(remote), 1)
        self.assertEqual(remote[0]["host"], "192.168.1.201")
        self.assertEqual(remote[0]["port"], 8033)

    def test_health_check_interval(self):
        """Health checks run every 5 seconds."""
        with open(self.CONFIG_PATH) as f:
            cfg = json.load(f)
        self.assertEqual(cfg.get("health_interval"), 5)

    def test_request_timeout_sufficient(self):
        """Request timeout is high enough for long generations (600s)."""
        with open(self.CONFIG_PATH) as f:
            cfg = json.load(f)
        self.assertGreaterEqual(cfg.get("request_timeout", 0), 300)

    def test_max_tokens_cap(self):
        """Max tokens per request is capped at 25K."""
        with open(self.CONFIG_PATH) as f:
            cfg = json.load(f)
        self.assertEqual(cfg.get("max_tokens_per_request"), 25000)


if __name__ == "__main__":
    unittest.main()
