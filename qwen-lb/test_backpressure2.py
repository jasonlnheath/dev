#!/usr/bin/env python3
"""Proper backpressure test using /completion endpoint."""
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

# Run the actual health checker
lb._checker = None
lb.server = server
server.server_activate()

# Manually mark backends as healthy with real slot counts from llama.cpp
import http.client
try:
    conn = http.client.HTTPConnection('127.0.0.1', 8033, timeout=5)
    conn.request('GET', '/slots')
    resp = conn.getresponse()
    slots = json.loads(resp.read().decode())
    active_slots = sum(1 for s in slots if s.get('is_processing', False))
    idle_slots = len(slots) - active_slots
    print(f"Backend slots: {len(slots)} total, {active_slots} processing, {idle_slots} idle")
    lb.states[0].healthy = True
    lb.states[0].slots_idle = idle_slots
    lb.states[0].slots_processing = active_slots
    lb.states[0].max_queue_depth = 3  # More realistic for 2 slots
except Exception as e:
    print(f"Could not check slots: {e}")
    lb.states[0].healthy = True
    lb.states[0].slots_idle = 2
    lb.states[0].slots_processing = 0

# Start LB
def serve_forever():
    server.serve_forever()
t = threading.Thread(target=serve_forever, daemon=True)
t.start()
time.sleep(1)

# Health check
print(f"\nHealth: {lb.states[0].name} healthy={lb.states[0].healthy}, "
      f"slots_idle={lb.states[0].slots_idle}, slots_processing={lb.states[0].slots_processing}, "
      f"max_qd={lb.states[0].max_queue_depth}, saturated={lb.states[0].is_saturated()}")

# Test with /completion endpoint
print("\n=== Single request to /completion ===")
try:
    req = urllib.request.Request(
        'http://localhost:8088/completion',
        data=json.dumps({"prompt": "hello", "n_predict": 10}).encode(),
        headers={"Content-Type": "application/json"},
        method='POST'
    )
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read().decode())
    print(f'Status: 200, content: {data.get("content", "")[:50]}')
except urllib.error.HTTPError as e:
    print(f'Error: HTTP {e.code}: {e.read().decode()[:100]}')
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')

# Backpressure test: send 15 concurrent requests to /completion
print(f"\n=== Sending 15 concurrent requests to /completion ===")
def hit(url, body):
    try:
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method='POST')
        r = urllib.request.urlopen(req, timeout=60)
        return r.status
    except urllib.error.HTTPError as e:
        return (e.code, e.read().decode()[:30])
    except Exception as e:
        return f"{type(e).__name__}"

payload = json.dumps({"prompt": "say something brief", "n_predict": 20}).encode()
urls = ['http://localhost:8088/completion'] * 15
with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
    results = list(ex.map(hit, urls, [payload]*15))

ok = sum(1 for r in results if r == 200)
rejected = sum(1 for r in results if isinstance(r, tuple) and r[0] == 429)
not_found = sum(1 for r in results if isinstance(r, tuple) and r[0] == 404)
other = len(results) - ok - rejected - not_found
print(f'Results: {ok} OK, {rejected} rejected (429), {not_found} not found, {other} other')

print(f'\nLB state after: active={lb.states[0].active_requests}, '
      f'proc={lb.states[0].slots_processing}, saturated={lb.states[0].is_saturated()}')

server.shutdown()
print('\nDONE')
