#!/usr/bin/env python3
"""Quick test: does backpressure actually reject requests with 429?"""
import json
import time
import urllib.request
import urllib.error
import concurrent.futures

LB = "http://localhost:8080"
PROMPT = "Say hello"

def make_request(i):
    """Single request - returns (status_code, time_to_response)"""
    data = json.dumps({
        "model": "qwen3.6-35b",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 4,
        "stream": False
    }).encode()
    req = urllib.request.Request(
        f"{LB}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer qwen-lb-2f1cf189d556f21c4b2f1c6ccc5121fd"}
    )
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return ("OK", resp.status, time.time() - t0)
    except urllib.error.HTTPError as e:
        return ("429/5xx", e.code, time.time() - t0)
    except Exception as e:
        return ("FAIL", str(e)[:50], time.time() - t0)

print("=== BACKPRESSURE TEST ===\n")

# Phase 1: Local backend saturation test
# Send 10 concurrent requests. Local has 2 slots + queue of 5 = 7 total capacity.
# Requests 8+ should get 429'd
print("Phase 1: 10 concurrent requests (local queue_depth=5, 2 slots)")
print("Expected: ~3 requests rejected (429), rest should succeed or time out\n")

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(make_request, i) for i in range(10)]
    for f in concurrent.futures.as_completed(futures):
        results.append(f.result())

ok = sum(1 for r in results if r[0] == "OK")
rejected = sum(1 for r in results if r[0] == "429/5xx")
failed = sum(1 for r in results if r[0] == "FAIL")
print(f"Results: {ok} OK, {rejected} rejected (429/5xx), {failed} failed/timeout")
print(f"429 rate: {rejected/len(results)*100:.0f}%")

# Phase 2: Heavy load test
print("\nPhase 2: 50 concurrent requests")
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(make_request, i) for i in range(50)]
    for f in concurrent.futures.as_completed(futures):
        results.append(f.result())

ok = sum(1 for r in results if r[0] == "OK")
rejected = sum(1 for r in results if r[0] == "429/5xx")
failed = sum(1 for r in results if r[0] == "FAIL")
print(f"Results: {ok} OK, {rejected} rejected (429/5xx), {failed} failed/timeout")
print(f"429 rate: {rejected/len(results)*100:.0f}%")
if rejected == 0:
    print("⚠️  NO 429s returned! Backpressure NOT working!")

# Phase 3: Check LB health during load
print("\nPhase 3: LB health status")
import urllib.request
resp = urllib.request.urlopen(f"{LB}/health", timeout=5)
health = json.loads(resp.read())
print(json.dumps(health, indent=2))

print(f"\n=== DONE ===")
