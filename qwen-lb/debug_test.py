#!/usr/bin/env python3
"""Debug test: trace what's happening with backpressure"""
import json
import time
import urllib.request
import urllib.error
import concurrent.futures
import threading

LB = "http://localhost:8080"
auth = "Bearer qwen-lb-2f1cf189d556f21c4b2f1c6ccc5121fd"

# Thread-safe counter for each status code
status_counts = {}
status_lock = threading.Lock()

def log_status(code):
    with status_lock:
        status_counts[code] = status_counts.get(code, 0) + 1

def make_request(i):
    data = json.dumps({
        "model": "qwen3.6-35b",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 4,
        "stream": False
    }).encode()
    req = urllib.request.Request(
        f"{LB}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": auth}
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        code = resp.status
        log_status(code)
        print(f"  Req {i:2d}: HTTP {code}")
    except urllib.error.HTTPError as e:
        code = e.code
        log_status(code)
        print(f"  Req {i:2d}: HTTP {code}")
    except Exception as e:
        log_status("TIMEOUT")
        print(f"  Req {i:2d}: TIMEOUT ({str(e)[:40]})")

# Start Phase 2: 50 concurrent requests
print("Starting 50 concurrent requests...\n")
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(make_request, i) for i in range(50)]
    for f in concurrent.futures.as_completed(futures):
        pass

print(f"\nStatus counts: {dict(sorted(status_counts.items()))}")
print(f"Total: {sum(status_counts.values())}")

# Check LB health
print("\nLB health:")
resp = urllib.request.urlopen(f"{LB}/health", timeout=5)
health = json.loads(resp.read())
for b in health['backends']:
    print(f"  {b['name']}: idle={b['slots_idle']} processing={b['slots_processing']} active_req={b['active_requests']} queue_depth={b['queue_depth']}")
