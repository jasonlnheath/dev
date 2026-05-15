#!/usr/bin/env python3
"""Inline test for backpressure on port 8088."""
import json, sys, time, threading, urllib.request, concurrent.futures
sys.path.insert(0, '.')
from lb import LoadBalancer, ProxyHandler, ThreadingHTTPServer

cfg = json.load(open('lb_config.json'))
cfg['listen_port'] = 8088

class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

lb = LoadBalancer(cfg)
handler = type('Handler', (ProxyHandler,), {'lb': lb})
server = ReusableHTTPServer(('0.0.0.0', 8088), handler)
lb._checker = None  # Don't use inline health checker
lb.server = server
server.server_activate()

# Manually mark backends as healthy with proper slot counts
for state in lb.states:
    state.healthy = True
    state.slots_idle = 2
    state.slots_processing = 0

def serve_forever():
    server.serve_forever()

t = threading.Thread(target=serve_forever, daemon=True)
t.start()
time.sleep(1)

# Check health
print("=== Health check ===")
resp = urllib.request.urlopen('http://localhost:8088/health', timeout=5)
print('Health:', resp.read().decode()[:200])

# Single request test
print("\n=== Single request test ===")
try:
    req = urllib.request.Request('http://localhost:8088/completions')
    resp = urllib.request.urlopen(req, timeout=30)
    print(f'Status: {resp.status}')
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')

# Backpressure test: 10 concurrent requests
print("\n=== Sending 10 concurrent requests ===")
def hit(url):
    try:
        r = urllib.request.urlopen(url, timeout=60)
        return r.status
    except urllib.error.HTTPError as e:
        return (e.code, e.read().decode()[:50])
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:60]}"

urls = ['http://localhost:8088/completions'] * 10
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
    results = list(ex.map(hit, urls))

ok = sum(1 for r in results if r == 200)
rejected = sum(1 for r in results if isinstance(r, tuple) and r[0] == 429)
other = len(results) - ok - rejected
print(f'Results: {ok} OK, {rejected} rejected (429), {other} other')

# Check LB internals
print(f'\nLB local backend request_count: {lb.states[0].request_count}')
print(f'LB local active_requests: {lb.states[0].active_requests}')
print(f'LB local slots_processing: {lb.states[0].slots_processing}')
print(f'LB local max_queue_depth: {lb.states[0].max_queue_depth}')
print(f'LB local is_saturated: {lb.states[0].is_saturated()}')

server.shutdown()
print('\nDONE')
