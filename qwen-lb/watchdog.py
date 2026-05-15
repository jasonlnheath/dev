"""
GPU Watchdog for Machine 2 — monitors VRAM, yields to games, recovers when idle.

State machine:
  RUNNING  → VRAM low 2x in a row? → YIELDING
  YIELDING → process exited?       → IDLE
  IDLE     → VRAM free >= threshold for 60s? → RECOVERING
  RECOVERING → /health OK?         → RUNNING

Usage: python watchdog.py [--config lb_config.json]
"""

import json
import os
import subprocess
import sys
import time


class WatchdogStateMachine:
    """Tracks the lifecycle of llama-server on a gaming PC."""

    def __init__(
        self,
        server_cmd: str = "",
        vram_threshold_mb: int = 4096,
        cooldown_seconds: int = 60,
        trigger_count: int = 2,
        poll_interval: int = 10,
        host: str = "127.0.0.1",
        port: int = 8033,
    ):
        self.server_cmd = server_cmd
        self.vram_threshold_mb = vram_threshold_mb
        self.cooldown_seconds = cooldown_seconds
        self.trigger_count = trigger_count
        self.poll_interval = poll_interval
        self.host = host
        self.port = port

        self.state = "RUNNING"
        self.low_vram_count = 0
        self.vram_free_start: float | None = None
        self.process: subprocess.Popen | None = None
        self._stop = False

    def on_vram_check(self, free_mb: int):
        """Called each poll with the current free VRAM in MB."""
        below = free_mb < self.vram_threshold_mb

        if self.state == "RUNNING":
            if below:
                self.low_vram_count += 1
                if self.low_vram_count >= self.trigger_count:
                    self.state = "YIELDING"
                    self.low_vram_count = 0
            else:
                self.low_vram_count = 0

        elif self.state == "IDLE":
            if below:
                self.vram_free_start = None
            else:
                if self.vram_free_start is None:
                    self.vram_free_start = time.time()
                elapsed = time.time() - self.vram_free_start
                if elapsed >= self.cooldown_seconds:
                    self.state = "RECOVERING"
                    self.vram_free_start = None

    def on_process_exited(self):
        """Called when the llama-server process exits."""
        if self.state == "YIELDING":
            self.state = "IDLE"

    def on_health_ok(self):
        """Called when /health returns 200."""
        if self.state == "RECOVERING":
            self.state = "RUNNING"

    # -----------------------------------------------------------------------
    # Live polling (used when running as a daemon)
    # -----------------------------------------------------------------------

    def _read_vram(self) -> int:
        """Read free VRAM via nvidia-smi. Returns MB."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Take the first GPU
            return int(result.stdout.strip().split("\n")[0].strip())
        except Exception:
            return 0

    def _kill_server(self):
        """Kill the llama-server process."""
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", "llama-server.exe"],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass

    def _start_server(self):
        """Launch llama-server."""
        if not self.server_cmd:
            return
        self.process = subprocess.Popen(
            self.server_cmd,
            shell=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    def _check_health(self) -> bool:
        """Check if llama-server is responding."""
        import http.client
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            conn.close()
            return resp.status == 200
        except Exception:
            return False

    def run(self):
        """Main watchdog loop."""
        print(f"Watchdog starting (state={self.state}, threshold={self.vram_threshold_mb}MB)")
        while not self._stop:
            free_mb = self._read_vram()
            self.on_vram_check(free_mb)

            if self.state == "YIELDING":
                print(f"VRAM low ({free_mb}MB), killing llama-server")
                self._kill_server()
                time.sleep(2)
                self.on_process_exited()

            elif self.state == "RECOVERING":
                print("VRAM free, restarting llama-server")
                self._start_server()
                # Wait for health
                for _ in range(30):
                    if self._check_health():
                        self.on_health_ok()
                        print("llama-server healthy, back to RUNNING")
                        break
                    time.sleep(2)

            time.sleep(self.poll_interval)

    def stop(self):
        self._stop = True


def load_watchdog_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    return cfg.get("watchdog", {})


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "lb_config.json"))
    parser.add_argument("--server-cmd", default="")
    args = parser.parse_args()

    wd_cfg = {}
    if os.path.exists(args.config):
        wd_cfg = load_watchdog_config(args.config)

    wd = WatchdogStateMachine(
        server_cmd=args.server_cmd or wd_cfg.get("server_cmd", ""),
        vram_threshold_mb=wd_cfg.get("vram_free_threshold_mb", 4096),
        cooldown_seconds=wd_cfg.get("cooldown_seconds", 60),
        trigger_count=wd_cfg.get("trigger_count", 2),
        poll_interval=wd_cfg.get("poll_interval", 10),
    )
    try:
        wd.run()
    except KeyboardInterrupt:
        wd.stop()
        print("\nWatchdog stopped")


if __name__ == "__main__":
    main()
