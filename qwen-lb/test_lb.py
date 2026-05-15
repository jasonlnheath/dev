"""
Tests for the Qwen Multi-Node Load Balancer.

Run with: python -m pytest test_lb.py -v
or:       python test_lb.py

Tests cover:
- Config loading
- Subnet trust checking (LAN, Meshnet, loopback)
- API key authentication
- Health check polling
- Least-busy-first routing
- Failover on backend failure
- SSE streaming pass-through
- Non-streaming request forwarding
- Aggregate /health endpoint
"""

import ipaddress
import json
import os
import sys
import threading
import time
import unittest
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import patch, MagicMock

# Ensure the module can be imported
sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Helpers: Fake llama-server backends for testing
# ---------------------------------------------------------------------------

class FakeBackendHandler(BaseHTTPRequestHandler):
    """Minimal fake llama-server that responds to /health and /v1/chat/completions."""

    slots_idle = 2
    slots_processing = 0
    _lock = threading.Lock()

    def log_message(self, format, *args):
        pass  # silence request logs

    def do_GET(self):
        if self.path == "/health":
            with self._lock:
                idle = self.slots_idle
                processing = self.slots_processing
            body = json.dumps({
                "status": "ok",
                "slots_idle": idle,
                "slots_processing": processing,
            })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "qwen", "object": "model"}]})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length)) if length else {}
            stream = data.get("stream", False)

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                # Send a couple SSE events then done
                for i in range(3):
                    chunk = json.dumps({"choices": [{"delta": {"content": f"word{i}"}}]})
                    event = f"data: {chunk}\n\n"
                    self._send_chunked(event.encode())
                self._send_chunked(b"data: [DONE]\n\n")
                self._send_chunked(b"")  # end of chunked transfer
            else:
                body = json.dumps({
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "hello"}, "index": 0}],
                })
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _send_chunked(self, data: bytes):
        if not data:
            self.wfile.write(b"0\r\n\r\n")
        else:
            self.wfile.write(f"{len(data):x}\r\n".encode())
            self.wfile.write(data)
            self.wfile.write(b"\r\n")
        self.wfile.flush()


def start_fake_backend(port, slots_idle=2, slots_processing=0):
    """Start a fake backend on the given port, return (server, thread)."""
    handler = type(
        f"FakeBackend_{port}",
        (FakeBackendHandler,),
        {"slots_idle": slots_idle, "slots_processing": slots_processing},
    )
    server = HTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, t


# ---------------------------------------------------------------------------
# Test config fixture
# ---------------------------------------------------------------------------

TEST_CONFIG = {
    "listen_port": 0,  # 0 = OS picks a free port
    "listen_host": "127.0.0.1",
    "health_interval": 2,
    "request_timeout": 10,
    "auth": {
        "api_key": "test-key-12345",
        "trusted_subnets": [
            "192.168.0.0/16",
            "10.0.0.0/8",
            "172.16.0.0/12",
            "127.0.0.0/8",
        ],
    },
    "backends": [],
}


def write_test_config(path, backends, **overrides):
    """Write a test config file, return the full config dict."""
    cfg = {**TEST_CONFIG, **overrides}
    cfg["backends"] = backends
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


# ===========================================================================
# TESTS
# ===========================================================================

class TestConfigLoading(unittest.TestCase):
    """Config file parsing and defaults."""

    def test_load_valid_config(self):
        from lb import load_config
        path = os.path.join(os.path.dirname(__file__), "_test_cfg_valid.json")
        write_test_config(path, [{"host": "127.0.0.1", "port": 9999, "name": "test"}])
        cfg = load_config(path)
        self.assertEqual(cfg["listen_port"], 0)
        self.assertEqual(len(cfg["backends"]), 1)
        os.unlink(path)

    def test_missing_config_raises(self):
        from lb import load_config
        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/path.json")

    def test_config_defaults(self):
        from lb import load_config
        path = os.path.join(os.path.dirname(__file__), "_test_cfg_defaults.json")
        with open(path, "w") as f:
            json.dump({"backends": [{"host": "127.0.0.1", "port": 8034, "name": "local"}]}, f)
        cfg = load_config(path)
        # Defaults should be filled in
        self.assertEqual(cfg["listen_host"], "0.0.0.0")
        self.assertEqual(cfg["listen_port"], 8033)
        self.assertEqual(cfg["health_interval"], 5)
        self.assertEqual(cfg["request_timeout"], 600)
        os.unlink(path)


class TestSubnetTrust(unittest.TestCase):
    """Trusted subnet detection for auth bypass."""

    def setUp(self):
        from lb import is_trusted_ip
        self.is_trusted = is_trusted_ip

    def test_loopback_trusted(self):
        self.assertTrue(self.is_trusted("127.0.0.1"))

    def test_lan_192_trusted(self):
        self.assertTrue(self.is_trusted("192.168.1.100"))

    def test_meshnet_10_trusted(self):
        """NordVPN Meshnet uses 10.x.x.x range."""
        self.assertTrue(self.is_trusted("10.0.0.45"))

    def test_docker_172_trusted(self):
        self.assertTrue(self.is_trusted("172.17.0.5"))

    def test_public_ip_not_trusted(self):
        self.assertFalse(self.is_trusted("203.0.113.50"))

    def test_another_public_not_trusted(self):
        self.assertFalse(self.is_trusted("8.8.8.8"))


class TestApiKeyAuth(unittest.TestCase):
    """API key authentication middleware."""

    def setUp(self):
        from lb import check_auth
        self.check_auth = check_auth

    def test_trusted_ip_no_key_needed(self):
        # Returns True = allowed (no key required)
        self.assertTrue(self.check_auth("127.0.0.1", None, "test-key"))

    def test_meshnet_ip_no_key_needed(self):
        self.assertTrue(self.check_auth("10.0.0.50", None, "test-key"))

    def test_untrusted_ip_no_key_rejected(self):
        self.assertFalse(self.check_auth("203.0.113.50", None, "test-key"))

    def test_untrusted_ip_valid_key(self):
        self.assertTrue(self.check_auth("203.0.113.50", "test-key", "test-key"))

    def test_untrusted_ip_wrong_key(self):
        self.assertFalse(self.check_auth("203.0.113.50", "wrong-key", "test-key"))

    def test_bearer_token_works(self):
        self.assertTrue(self.check_auth("203.0.113.50", "Bearer test-key", "test-key"))


class TestHealthChecks(unittest.TestCase):
    """Backend health polling and state tracking."""

    def test_healthy_backend_tracked(self):
        from lb import BackendState
        state = BackendState("127.0.0.1", 19999)
        # Simulate a health check response
        state.update_health(True, slots_idle=2, slots_processing=0)
        self.assertTrue(state.healthy)
        self.assertEqual(state.slots_idle, 2)

    def test_unhealthy_backend_tracked(self):
        from lb import BackendState
        state = BackendState("127.0.0.1", 19998)
        state.update_health(False)
        self.assertFalse(state.healthy)
        self.assertEqual(state.slots_idle, 0)

    def test_aggregate_health(self):
        from lb import BackendState
        states = [
            BackendState("a", 1),
            BackendState("b", 2),
        ]
        states[0].update_health(True, slots_idle=2, slots_processing=0)
        states[1].update_health(False)
        # One healthy, one not
        healthy_count = sum(1 for s in states if s.healthy)
        self.assertEqual(healthy_count, 1)


class TestRouting(unittest.TestCase):
    """Least-busy-first backend selection."""

    def setUp(self):
        from lb import BackendState, pick_backend
        self.BackendState = BackendState
        self.pick_backend = pick_backend

    def test_picks_most_idle(self):
        states = [
            self.BackendState("a", 1),
            self.BackendState("b", 2),
        ]
        states[0].update_health(True, slots_idle=1, slots_processing=1)
        states[1].update_health(True, slots_idle=2, slots_processing=0)
        chosen = self.pick_backend(states)
        self.assertEqual(chosen.port, 2)

    def test_skips_unhealthy(self):
        states = [
            self.BackendState("a", 1),
            self.BackendState("b", 2),
        ]
        states[0].update_health(True, slots_idle=2, slots_processing=0)
        states[1].update_health(False)
        chosen = self.pick_backend(states)
        self.assertEqual(chosen.port, 1)

    def test_all_unhealthy_returns_none(self):
        states = [
            self.BackendState("a", 1),
            self.BackendState("b", 2),
        ]
        states[0].update_health(False)
        states[1].update_health(False)
        chosen = self.pick_backend(states)
        self.assertIsNone(chosen)

    def test_tiebreak_fewer_recent_requests(self):
        states = [
            self.BackendState("a", 1),
            self.BackendState("b", 2),
        ]
        states[0].update_health(True, slots_idle=2, slots_processing=0)
        states[1].update_health(True, slots_idle=2, slots_processing=0)
        states[0].request_count = 5
        states[1].request_count = 2
        chosen = self.pick_backend(states)
        self.assertEqual(chosen.port, 2)  # fewer recent requests


class TestIntegration(unittest.TestCase):
    """Full proxy integration with fake backends."""

    @classmethod
    def setUpClass(cls):
        """Start two fake backends and the proxy."""
        # Start fake backends
        cls.backend1_port = 19091
        cls.backend2_port = 19092
        cls.srv1, _ = start_fake_backend(cls.backend1_port, slots_idle=2)
        cls.srv2, _ = start_fake_backend(cls.backend2_port, slots_idle=1)

        # Write config pointing at fake backends
        cls.config_path = os.path.join(
            os.path.dirname(__file__), "_test_cfg_integration.json"
        )
        write_test_config(
            cls.config_path,
            backends=[
                {"host": "127.0.0.1", "port": cls.backend1_port, "name": "b1"},
                {"host": "127.0.0.1", "port": cls.backend2_port, "name": "b2"},
            ],
        )

    @classmethod
    def tearDownClass(cls):
        cls.srv1.shutdown()
        cls.srv2.shutdown()
        if os.path.exists(cls.config_path):
            os.unlink(cls.config_path)

    def _get_proxy(self):
        """Import and create proxy instance (tests lb.py exists and works)."""
        from lb import LoadBalancer
        cfg = load_config_local(self.config_path)
        lb = LoadBalancer(cfg)
        return lb

    def test_non_streaming_request(self):
        """Proxy forwards a non-streaming chat completion."""
        import urllib.request
        from lb import LoadBalancer

        cfg = load_config_local(self.config_path)
        lb = LoadBalancer(cfg)

        # Start proxy on a free port
        lb.start(blocking=False)
        time.sleep(1)  # let health checks run

        try:
            proxy_port = lb.server.server_address[1]
            url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
            data = json.dumps({
                "model": "qwen",
                "messages": [{"role": "user", "content": "hi"}],
            }).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read())
            self.assertIn("choices", result)
            self.assertEqual(result["choices"][0]["message"]["content"], "hello")
        finally:
            lb.stop()

    def test_streaming_request(self):
        """Proxy forwards SSE streaming correctly."""
        import urllib.request
        from lb import LoadBalancer

        cfg = load_config_local(self.config_path)
        lb = LoadBalancer(cfg)
        lb.start(blocking=False)
        time.sleep(1)

        try:
            proxy_port = lb.server.server_address[1]
            url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
            data = json.dumps({
                "model": "qwen",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            }).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = resp.read().decode()
            # Should contain SSE data lines
            self.assertIn("data: ", raw)
            self.assertIn("[DONE]", raw)
        finally:
            lb.stop()

    def test_models_endpoint(self):
        """Proxy forwards /v1/models to a healthy backend."""
        import urllib.request
        from lb import LoadBalancer

        cfg = load_config_local(self.config_path)
        lb = LoadBalancer(cfg)
        lb.start(blocking=False)
        time.sleep(1)

        try:
            proxy_port = lb.server.server_address[1]
            url = f"http://127.0.0.1:{proxy_port}/v1/models"
            with urllib.request.urlopen(url, timeout=5) as resp:
                result = json.loads(resp.read())
            self.assertIn("data", result)
        finally:
            lb.stop()

    def test_health_endpoint(self):
        """Proxy /health returns aggregate backend status."""
        import urllib.request
        from lb import LoadBalancer

        cfg = load_config_local(self.config_path)
        lb = LoadBalancer(cfg)
        lb.start(blocking=False)
        time.sleep(1)

        try:
            proxy_port = lb.server.server_address[1]
            url = f"http://127.0.0.1:{proxy_port}/health"
            with urllib.request.urlopen(url, timeout=5) as resp:
                result = json.loads(resp.read())
            self.assertIn("backends", result)
            self.assertEqual(len(result["backends"]), 2)
        finally:
            lb.stop()

    def test_auth_blocks_untrusted(self):
        """Untrusted IP without API key gets 401."""
        import urllib.request
        from lb import LoadBalancer, ProxyHandler

        cfg = load_config_local(self.config_path)
        # Ensure API key is set for this test
        cfg["auth"]["api_key"] = "test-key-12345"
        cfg["_trusted_networks"] = [ipaddress.ip_network(s) for s in cfg["auth"]["trusted_subnets"]]
        lb = LoadBalancer(cfg)
        lb.start(blocking=False)
        time.sleep(1)

        try:
            proxy_port = lb.server.server_address[1]
            url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
            data = json.dumps({
                "model": "qwen",
                "messages": [{"role": "user", "content": "hi"}],
            }).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            # Patch ProxyHandler._get_client_ip to return a public IP
            with patch.object(ProxyHandler, "_get_client_ip", return_value="203.0.113.50"):
                try:
                    urllib.request.urlopen(req, timeout=5)
                    self.fail("Should have raised HTTPError 401")
                except urllib.error.HTTPError as e:
                    self.assertEqual(e.code, 401)
        finally:
            lb.stop()


class TestWatchdog(unittest.TestCase):
    """GPU watchdog state machine."""

    def test_import_watchdog(self):
        """Watchdog module can be imported."""
        from watchdog import WatchdogStateMachine
        self.assertIsNotNone(WatchdogStateMachine)

    def test_initial_state_running(self):
        from watchdog import WatchdogStateMachine
        wd = WatchdogStateMachine("qwen36-server.bat")
        self.assertEqual(wd.state, "RUNNING")

    def test_vram_low_triggers_yield(self):
        from watchdog import WatchdogStateMachine
        wd = WatchdogStateMachine("qwen36-server.bat")
        wd.on_vram_check(free_mb=2048)  # first low
        wd.on_vram_check(free_mb=2048)  # second low — triggers yield
        self.assertEqual(wd.state, "YIELDING")

    def test_vram_ok_stays_running(self):
        from watchdog import WatchdogStateMachine
        wd = WatchdogStateMachine("qwen36-server.bat")
        wd.on_vram_check(free_mb=8000)  # above threshold
        self.assertEqual(wd.state, "RUNNING")

    def test_single_low_doesnt_trigger(self):
        """Need 2 consecutive low readings before yielding."""
        from watchdog import WatchdogStateMachine
        wd = WatchdogStateMachine("qwen36-server.bat")
        wd.on_vram_check(free_mb=2048)  # first low
        self.assertEqual(wd.state, "RUNNING")  # not yet
        wd.on_vram_check(free_mb=2048)  # second low
        self.assertEqual(wd.state, "YIELDING")

    def test_idle_to_recovery(self):
        from watchdog import WatchdogStateMachine
        wd = WatchdogStateMachine("qwen36-server.bat")
        wd.state = "IDLE"
        wd.vram_free_start = time.time() - 61  # 61 seconds of free VRAM
        wd.on_vram_check(free_mb=12000)
        self.assertEqual(wd.state, "RECOVERING")


def load_config_local(path):
    """Local helper to load config for integration tests."""
    from lb import load_config
    return load_config(path)


# ===========================================================================
# BACKPRESSURE TESTS
# ===========================================================================

class TestBackpressureBackendState(unittest.TestCase):
    """Tests for queue depth tracking in BackendState."""

    def setUp(self):
        from lb import BackendState
        self.BackendState = BackendState

    def test_default_queue_depth_is_five(self):
        """Default max queue depth should be 5."""
        state = self.BackendState("127.0.0.1", 19999)
        self.assertEqual(state.max_queue_depth, 5)

    def test_custom_queue_depth(self):
        """max_queue_depth should be configurable."""
        state = self.BackendState("127.0.0.1", 19999, max_queue_depth=3)
        self.assertEqual(state.max_queue_depth, 3)

    def test_not_saturated_when_empty(self):
        """Backend should NOT be saturated when queue is empty."""
        state = self.BackendState("127.0.0.1", 19999, max_queue_depth=5)
        state.update_health(True, slots_idle=2, slots_processing=0)
        self.assertFalse(state.is_saturated())

    def test_not_saturated_under_limit(self):
        """Backend should NOT be saturated when queue depth < max."""
        state = self.BackendState("127.0.0.1", 19999, max_queue_depth=5)
        state.update_health(True, slots_idle=0, slots_processing=3)
        state.active_requests = 1
        # 3 slots + 1 active = 4 < 5
        self.assertFalse(state.is_saturated())

    def test_saturated_at_limit(self):
        """Backend should be saturated when queue depth == max."""
        state = self.BackendState("127.0.0.1", 19999, max_queue_depth=5)
        state.update_health(True, slots_idle=0, slots_processing=3)
        state.active_requests = 2
        # 3 slots + 2 active = 5 == 5
        self.assertTrue(state.is_saturated())

    def test_saturated_over_limit(self):
        """Backend should be saturated when queue depth > max."""
        state = self.BackendState("127.0.0.1", 19999, max_queue_depth=5)
        state.update_health(True, slots_idle=0, slots_processing=4)
        state.active_requests = 2
        # 4 slots + 2 active = 6 > 5
        self.assertTrue(state.is_saturated())

    def test_active_requests_increment(self):
        """increment_active should increase counter."""
        state = self.BackendState("127.0.0.1", 19999)
        self.assertEqual(state.active_requests, 0)
        state.increment_active()
        self.assertEqual(state.active_requests, 1)
        state.increment_active()
        state.increment_active()
        self.assertEqual(state.active_requests, 3)

    def test_active_requests_decrement(self):
        """decrement_active should decrease counter."""
        state = self.BackendState("127.0.0.1", 19999)
        state.active_requests = 3
        state.decrement_active()
        self.assertEqual(state.active_requests, 2)
        state.decrement_active()
        state.decrement_active()
        self.assertEqual(state.active_requests, 0)

    def test_active_requests_can_go_negative(self):
        """decrement_active can go below zero if over-called (no guards)."""
        state = self.BackendState("127.0.0.1", 19999)
        state.active_requests = 1
        state.decrement_active()
        state.decrement_active()
        self.assertEqual(state.active_requests, -1)

    def test_to_dict_includes_active_requests(self):
        """BackendState.to_dict() should include active_requests and queue_depth."""
        state = self.BackendState("127.0.0.1", 19999, max_queue_depth=5)
        state.update_health(True, slots_idle=1, slots_processing=2)
        state.active_requests = 1
        d = state.to_dict()
        self.assertEqual(d["active_requests"], 1)
        self.assertEqual(d["queue_depth"], 3)  # 2 processing + 1 active


class TestBackpressureIntegration(unittest.TestCase):
    """Integration tests for backpressure in the proxy."""

    @classmethod
    def setUpClass(cls):
        """Start a fake backend and proxy with low queue depth."""
        cls.backend_port = 19093
        cls.srv, _ = start_fake_backend(cls.backend_port, slots_idle=2)

        cls.config_path = os.path.join(
            os.path.dirname(__file__), "_test_cfg_backpressure.json"
        )
        write_test_config(
            cls.config_path,
            backends=[{"host": "127.0.0.1", "port": cls.backend_port, "name": "test"}],
            max_queue_depth=2,  # Low queue depth for testing
        )

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        if os.path.exists(cls.config_path):
            os.unlink(cls.config_path)

    def test_request_succeeds_when_not_saturated(self):
        """Normal request should succeed when queue is under limit."""
        import urllib.request
        from lb import LoadBalancer

        cfg = load_config_local(self.config_path)
        lb = LoadBalancer(cfg)
        lb.start(blocking=False)
        time.sleep(1)

        try:
            proxy_port = lb.server.server_address[1]
            url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
            data = json.dumps({
                "model": "qwen",
                "messages": [{"role": "user", "content": "hi"}],
            }).encode()
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                result = json.loads(resp.read())
                self.assertIn("choices", result)
        finally:
            lb.stop()

    def test_health_shows_active_requests(self):
        """Proxy /health should include active_requests in response."""
        import urllib.request
        from lb import LoadBalancer

        cfg = load_config_local(self.config_path)
        lb = LoadBalancer(cfg)
        lb.start(blocking=False)
        time.sleep(1)

        try:
            proxy_port = lb.server.server_address[1]
            url = f"http://127.0.0.1:{proxy_port}/health"
            with urllib.request.urlopen(url, timeout=5) as resp:
                result = json.loads(resp.read())
                self.assertIn("active_requests", result)
                # Backend state should also include queue_depth
                for b in result["backends"]:
                    self.assertIn("queue_depth", b)
                    self.assertIn("active_requests", b)
        finally:
            lb.stop()

    def test_request_rejected_when_saturated(self):
        """Request should get 429 when backend queue is full."""
        import urllib.request
        from lb import LoadBalancer

        cfg = load_config_local(self.config_path)
        lb = LoadBalancer(cfg)
        lb.start(blocking=False)
        time.sleep(1)

        try:
            proxy_port = lb.server.server_address[1]
            # Force backend into saturated state
            # Set slots_processing + active_requests >= max_queue_depth
            for state in lb.states:
                state.update_health(True, slots_idle=0, slots_processing=2)
                state.active_requests = 0

            url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
            data = json.dumps({
                "model": "qwen",
                "messages": [{"role": "user", "content": "hi"}],
            }).encode()
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            try:
                urllib.request.urlopen(req, timeout=5)
                self.fail("Should have raised HTTPError 429")
            except urllib.error.HTTPError as e:
                self.assertEqual(e.code, 429)
        finally:
            lb.stop()


if __name__ == "__main__":
    unittest.main()
