#!/usr/bin/env python3
"""Debug: manually test pick_backend and is_saturated logic"""
import sys
sys.path.insert(0, '/mnt/c/dev/qwen-lb')
from lb import BackendState, pick_backend

# Simulate config
config = {
    "routing": "priority_fallback",
    "max_queue_depth": 5
}

# Create backends with same config as lb_config.json
local = BackendState("127.0.0.1", 8033, "local", priority=1, max_queue_depth=5)
remote = BackendState("192.168.1.201", 8033, "remote", priority=2, max_queue_depth=5)

# Simulate health state: local has 2 slots processing (from health check)
local.update_health(True, slots_idle=0, slots_processing=2)
remote.update_health(True, slots_idle=2, slots_processing=0)

states = [local, remote]

print(f"Local: idle={local.slots_idle} proc={local.slots_processing} active={local.active_requests} saturated={local.is_saturated()}")
print(f"Remote: idle={remote.slots_idle} proc={remote.slots_processing} active={remote.active_requests} saturated={remote.is_saturated()}")
print()

# Simulate 10 requests coming in
for i in range(10):
    backend = pick_backend(states, config)
    if backend is None:
        print(f"Req {i}: pick_backend returned None → 503")
        continue
    
    # Increment first (like the fixed code)
    backend.request_count += 1
    backend.increment_active()
    
    sat = backend.is_saturated()
    print(f"Req {i}: picked {backend.name}, active={backend.active_requests}, saturated={sat}")
    
    if sat:
        backend.decrement_active()
        print(f"         → 429 (queue full)")
    else:
        print(f"         → forwarded (would be 200)")

print()
print(f"After 10 requests:")
print(f"Local: active={local.active_requests} saturated={local.is_saturated()}")
print(f"Remote: active={remote.active_requests} saturated={remote.is_saturated()}")
