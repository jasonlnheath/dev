"""
Qwen Multi-Node Load Balancer — Python proxy with API key auth.

Routes requests to the least-busy llama-server backend.
Trusted subnets (LAN + Meshnet) bypass auth; others need an API key.

Usage: python lb.py [config.json]
"""

import http.client
import ipaddress
import json
import os
import sys
import threading
import time
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULTS = {
    "listen_host": "0.0.0.0",
    "listen_port": 8033,
    "health_interval": 5,
    "request_timeout": 600,
    "auth": {
        "api_key": "",
        "trusted_subnets": [
            "192.168.0.0/16",
            "10.0.0.0/8",
            "172.16.0.0/12",
            "127.0.0.0/8",
        ],
    },
    "backends": [],
}


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        cfg = json.load(f)
    # Merge defaults (shallow — auth merged separately)
    merged = {**DEFAULTS, **cfg}
    if "auth" not in cfg:
        merged["auth"] = DEFAULTS["auth"]
    else:
        merged["auth"] = {**DEFAULTS["auth"], **cfg["auth"]}
    # Parse subnets into ipaddress objects
    merged["_trusted_networks"] = [
        ipaddress.ip_network(s) for s in merged["auth"].get("trusted_subnets", [])
    ]
    return merged


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def is_trusted_ip(ip_str: str, networks=None) -> bool:
    """Check if an IP is in any trusted subnet."""
    if networks is None:
        networks = [ipaddress.ip_network(s) for s in DEFAULTS["auth"]["trusted_subnets"]]
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return any(addr in net for net in networks)


def check_auth(client_ip: str, api_key_header: str | None, expected_key: str, networks=None) -> bool:
    """Return True if the request is allowed."""
    if is_trusted_ip(client_ip, networks):
        return True
    if not expected_key:
        return True  # no key configured = open
    if not api_key_header:
        return False
    # Strip "Bearer " prefix if present
    key = api_key_header.removeprefix("Bearer ").strip()
    return key == expected_key


# ---------------------------------------------------------------------------
# Backend state
# ---------------------------------------------------------------------------


class BackendState:
    def __init__(self, host: str, port: int, name: str = "", priority: int = 999, max_queue_depth: int = 5):
        self.host = host
        self.port = port
        self.name = name or f"{host}:{port}"
        self.priority = priority
        self.max_queue_depth = max_queue_depth
        self.healthy = False
        self.slots_idle = 0
        self.slots_processing = 0
        self.active_requests = 0
        self.request_count = 0
        self.last_check = 0.0
        self._lock = threading.Lock()

    def is_saturated(self) -> bool:
        """Check if backend has reached max queue depth."""
        with self._lock:
            return self.slots_processing + self.active_requests >= self.max_queue_depth

    def increment_active(self):
        with self._lock:
            self.active_requests += 1

    def decrement_active(self):
        with self._lock:
            self.active_requests -= 1

    def update_health(self, healthy: bool, slots_idle: int = 0, slots_processing: int = 0):
        with self._lock:
            self.healthy = healthy
            self.slots_idle = slots_idle
            self.slots_processing = slots_processing
            self.last_check = time.time()

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "host": self.host,
                "port": self.port,
                "healthy": self.healthy,
                "slots_idle": self.slots_idle,
                "slots_processing": self.slots_processing,
                "active_requests": self.active_requests,
                "queue_depth": self.slots_processing + self.active_requests,
                "last_check_ago": round(time.time() - self.last_check, 1) if self.last_check else None,
            }


def pick_backend(states: list[BackendState], config: dict = None) -> BackendState | None:
    """Select a backend based on routing mode.
    
    Priority fallback: use highest-priority backends first, fall back to lower
    priority only when all higher-priority backends are fully saturated.
    Least-busy (default): pick the healthy backend with most idle slots.
    Both modes check is_saturated() to respect queue depth limits.
    """
    healthy = [s for s in states if s.healthy]
    if not healthy:
        return None

    routing = config.get("routing", "least_busy") if config else "least_busy"

    if routing == "priority_fallback":
        # Group by priority, lowest number = highest priority
        by_priority = {}
        for s in healthy:
            by_priority.setdefault(s.priority, []).append(s)

        for prio in sorted(by_priority.keys()):
            group = by_priority[prio]
            # Check if any in this priority group has capacity (not saturated)
            available = [s for s in group if not s.is_saturated()]
            if available:
                # Pick the one with most idle slots among same-priority
                available.sort(key=lambda s: (-s.slots_idle, s.request_count))
                return available[0]
            # All backends at this priority are fully saturated — fall through to next
        return None  # Everything is saturated

    # Default: least-busy (most idle slots, then fewest requests)
    # Filter out saturated backends first
    available = [s for s in healthy if not s.is_saturated()]
    if not available:
        return None
    available.sort(key=lambda s: (-s.slots_idle, s.request_count))
    return available[0]


# ---------------------------------------------------------------------------
# Health checker thread
# ---------------------------------------------------------------------------


class HealthChecker:
    def __init__(self, states: list[BackendState], interval: int = 5):
        self.states = states
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        while not self._stop.is_set():
            for state in self.states:
                self._check_one(state)
            self._stop.wait(self.interval)

    def _check_one(self, state: BackendState):
        healthy = False
        slots_idle = 0
        slots_processing = 0
        try:
            conn = http.client.HTTPConnection(state.host, state.port, timeout=5)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            if resp.status == 200:
                healthy = True
                data = json.loads(resp.read())
                slots_idle = data.get("slots_idle", 0)
                slots_processing = data.get("slots_processing", 0)
            conn.close()

            # If /health didn't include slot data, try /slots
            if healthy and slots_idle == 0 and slots_processing == 0:
                conn = http.client.HTTPConnection(state.host, state.port, timeout=5)
                conn.request("GET", "/slots")
                resp = conn.getresponse()
                if resp.status == 200:
                    slots = json.loads(resp.read())
                    slots_idle = sum(1 for s in slots if not s.get("is_processing", False))
                    slots_processing = sum(1 for s in slots if s.get("is_processing", False))
                conn.close()
        except Exception:
            healthy = False

        state.update_health(healthy, slots_idle, slots_processing)


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


class ProxyHandler(BaseHTTPRequestHandler):
    """Forward requests to the best available backend."""

    lb: "LoadBalancer" = None  # set by LoadBalancer

    def log_message(self, format, *args):
        pass

    def _get_client_ip(self) -> str:
        return self.client_address[0]

    def _check_auth(self) -> bool:
        cfg = self.lb.config
        client_ip = self._get_client_ip()
        api_key = self.headers.get("X-API-Key") or self.headers.get("Authorization")
        expected = cfg["auth"].get("api_key", "")
        return check_auth(client_ip, api_key, expected, cfg.get("_trusted_networks"))

    def do_GET(self):
        if not self._check_auth():
            self.send_error(401, "Unauthorized")
            return

        if self.path == "/health":
            self._handle_aggregate_health()
            return

        self._forward("GET")

    def do_POST(self):
        if not self._check_auth():
            print("DEBUG: POST auth rejected", file=sys.stderr)
            self.send_error(401, "Unauthorized")
            return
        print(f"DEBUG: POST received path={self.path} client={self.client_address}", file=sys.stderr)
        self._forward("POST")

    def _handle_aggregate_health(self):
        backends = [s.to_dict() for s in self.lb.states]
        healthy = sum(1 for b in backends if b["healthy"])
        total_slots = sum(b["slots_idle"] + b["slots_processing"] for b in backends)
        idle_slots = sum(b["slots_idle"] for b in backends)
        active_requests = sum(b["active_requests"] for b in backends)
        body = json.dumps({
            "status": "ok" if healthy > 0 else "degraded",
            "healthy_backends": healthy,
            "total_backends": len(backends),
            "total_slots": total_slots,
            "idle_slots": idle_slots,
            "active_requests": active_requests,
            "backends": backends,
        })
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    # Hard cap on max_tokens to prevent slow 10K-token generations on local Qwen
    MAX_TOKENS_CAP = 25000

    # Status codes that trigger a retry on another backend
    RETRY_STATUS_CODES = {400, 502, 504}

    def _forward(self, method: str, _retried: set | None = None):
        backend = pick_backend(self.lb.states, self.lb.config)
        if backend is None:
            healthy = [s for s in self.lb.states if s.healthy]
            if healthy and all(s.is_saturated() for s in healthy):
                self.send_error(429, "All backends saturated")
            else:
                self.send_error(503, "No healthy backends")
            return

        # Track which backends we've already tried for retry logic
        if _retried is None:
            _retried = set()

        # Increment active_requests first (count ourselves), then check saturation.
        # This prevents the race condition where multiple requests slip through
        # between health polls because slots_processing is stale.
        backend.request_count += 1
        backend.increment_active()
        already_decremented = False
        try:
            sat = backend.is_saturated()
            print(f"DEBUG: {backend.name} active={backend.active_requests} proc={backend.slots_processing} max_qd={backend.max_queue_depth} saturated={sat}", file=sys.stderr)
            if sat:
                already_decremented = True
                backend.decrement_active()
                # Retry on another backend instead of immediately returning 429
                _retried.add(backend.name)
                remaining = [s for s in self.lb.states if s.healthy and s.name not in _retried]
                if remaining:
                    return self._forward(method, _retried)
                self.send_error(429, "Queue full (all backends saturated)")
                return

            # Read request body if present
            body = None
            content_length = self.headers.get("Content-Length")
            if content_length:
                body = self.rfile.read(int(content_length))

            # Clamp max_tokens for POST requests
            if body and method == "POST":
                body = self._clamp_body_max_tokens(body)

            conn = http.client.HTTPConnection(
                backend.host, backend.port,
                timeout=self.lb.config.get("request_timeout", 600),
            )
            conn.request(method, self.path, body=body, headers=self._filtered_headers())
            resp = conn.getresponse()

            # Retry on retriable errors (e.g. 400 context overflow, 502, 504)
            if resp.status in self.RETRY_STATUS_CODES and method == "POST":
                error_body = resp.read()
                conn.close()
                already_decremented = True
                backend.decrement_active()
                _retried.add(backend.name)
                remaining = [s for s in self.lb.states if s.healthy and s.name not in _retried]
                if remaining:
                    print(f"RETRY: {backend.name} returned {resp.status}, trying next backend", file=sys.stderr)
                    return self._forward(method, _retried)
                # No more backends — return the original error
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.getheader("Content-Type", "text/plain"))
                self.end_headers()
                self.wfile.write(error_body)
                return

            # Streaming?
            ct = resp.getheader("Content-Type", "")
            is_streaming = "text/event-stream" in ct

            self.send_response(resp.status)
            for key, val in resp.getheaders():
                if key.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(key, val)
            self.end_headers()

            if is_streaming:
                self._stream_response(conn, resp)
            else:
                data = resp.read()
                self.wfile.write(data)

            conn.close()
        except Exception as e:
            # Connection error — retry on another backend
            already_decremented = True
            backend.decrement_active()
            _retried.add(backend.name)
            remaining = [s for s in self.lb.states if s.healthy and s.name not in _retried]
            if remaining:
                print(f"RETRY: {backend.name} connection error ({e}), trying next backend", file=sys.stderr)
                return self._forward(method, _retried)
            self.send_error(502, f"Backend error: {e}")
        finally:
            if not already_decremented:
                backend.decrement_active()

    def _stream_response(self, conn, resp):
        """Pipe SSE chunks from backend to client."""
        try:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _filtered_headers(self) -> dict:
        """Forward relevant headers, drop hop-by-hop ones."""
        skip = {"host", "connection", "transfer-encoding", "x-api-key", "authorization"}
        headers = {}
        for key in self.headers:
            if key.lower() not in skip:
                headers[key] = self.headers[key]
        return headers

    def _clamp_body_max_tokens(self, body: bytes) -> bytes:
        """Clamp max_tokens in JSON body to config cap. Returns (possibly new) body."""
        cap = self.lb.config.get("max_tokens_per_request", self.MAX_TOKENS_CAP)
        if cap <= 0:
            return body
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return body
        modified = False
        for key in ("max_tokens", "n_predict"):
            val = data.get(key)
            if isinstance(val, (int, float)) and val > cap:
                data[key] = cap
                modified = True
        if modified:
            return json.dumps(data).encode()
        return body


# ---------------------------------------------------------------------------
# LoadBalancer
# ---------------------------------------------------------------------------


class LoadBalancer:
    def __init__(self, config: dict):
        self.config = config
        max_queue = config.get("max_queue_depth", 5)
        self.states = [
            BackendState(b["host"], b["port"], b.get("name", ""), b.get("priority", 999), max_queue)
            for b in config["backends"]
        ]
        self._checker: HealthChecker | None = None
        self.server: ThreadingHTTPServer | None = None

    def start(self, blocking: bool = True):
        host = self.config["listen_host"]
        port = self.config["listen_port"]

         # Custom server class with SO_REUSEADDR to handle TIME_WAIT sockets
        class ReusableHTTPServer(ThreadingHTTPServer):
            allow_reuse_address = True

        handler = type("Handler", (ProxyHandler,), {"lb": self})
        self.server = ReusableHTTPServer((host, port), handler)

        # Start health checker
        self._checker = HealthChecker(self.states, self.config.get("health_interval", 5))
        self._checker.start()

        # Give health checks time to populate
        time.sleep(0.5)

        if blocking:
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        else:
            t = threading.Thread(target=self.server.serve_forever, daemon=True)
            t.start()

    def stop(self):
        if self._checker:
            self._checker.stop()
        if self.server:
            self.server.shutdown()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "lb_config.json"
    )
    cfg = load_config(config_path)
    lb = LoadBalancer(cfg)
    print(f"Qwen LB listening on {cfg['listen_host']}:{cfg['listen_port']}")
    print(f"Backends: {len(cfg['backends'])}")
    print(f"Trusted subnets: {cfg['auth'].get('trusted_subnets', [])}")
    lb.start()


if __name__ == "__main__":
    main()
