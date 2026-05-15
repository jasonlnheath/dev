#!/usr/bin/env python3
"""Ultra-fast stress test — 64 tok max, 15s timeout per request."""
import json, time, statistics, urllib.request, urllib.error, concurrent.futures

LB_URL = "http://127.0.0.1:8080/v1/chat/completions"
CHARLIE_URL = "http://192.168.1.201:8033/v1/chat/completions"
LOCAL_URL = "http://127.0.0.1:8033/v1/chat/completions"
PROMPTS = [
    "What is 2+2? Answer in one word.",
    "Name a color. One word.",
    "List one fruit. One word.",
    "What day comes after Monday? One word.",
    "Name a number under 10. One word.",
]

LB_HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer qwen-lb-2f1cf189d556f21c4b2f1c6ccc5121fd"}

def send(url, prompt, headers=None, timeout=15):
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "temperature": 0.7,
    }).encode()
    hdrs = dict(LB_HEADERS) if headers is None else dict(headers)
    t0 = time.time()
    try:
        req = urllib.request.Request(url, data=payload, headers=hdrs)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            elapsed = time.time() - t0
            toks = data["usage"]["completion_tokens"]
            return {"elapsed": elapsed, "toks": toks, "tok_s": toks/elapsed if elapsed > 0 else 0, "status": "ok", "http": resp.status}
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        return {"elapsed": elapsed, "toks": 0, "tok_s": 0, "status": f"429", "http": e.code}
    except Exception as e:
        elapsed = time.time() - t0
        return {"elapsed": elapsed, "toks": 0, "tok_s": 0, "status": str(e)[:40], "http": 0}

def run_batch(name, url, n_requests, headers=None, timeout=15):
    print(f"\n{'='*60}")
    print(f"  {name}: {n_requests} requests")
    print(f"{'='*60}")
    
    results = []
    start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_requests) as pool:
        futs = {}
        for i in range(n_requests):
            f = pool.submit(send, url, PROMPTS[i % len(PROMPTS)], headers, timeout)
            futs[f] = i
        
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            results.append(r)
    
    wall = time.time() - start
    ok = [x for x in results if x['status'] == 'ok']
    rej = [x for x in results if x['status'] == '429']
    errs = [x for x in results if x['status'] not in ('ok', '429')]
    
    print(f"  Wall time: {wall:.1f}s")
    print(f"  Total:     {len(results)}")
    print(f"  OK:        {len(ok)}")
    print(f"  429:       {len(rej)}")
    print(f"  Errors:    {len(errs)}")
    
    if ok:
        rates = [x['tok_s'] for x in ok]
        total_toks = sum(x['toks'] for x in ok)
        tp = total_toks / wall if wall > 0 else 0
        
        print(f"\n  Throughput: {tp:.1f} tok/s total")
        print(f"  Req/s:      {len(results)/wall:.1f}")
        print(f"\n  Token/sec:")
        print(f"    Mean:   {statistics.mean(rates):.1f}")
        print(f"    Median: {statistics.median(rates):.1f}")
        print(f"    Min:    {min(rates):.1f}")
        print(f"    Max:    {max(rates):.1f}")
        if len(rates) > 5:
            s = sorted(rates)
            print(f"    P10:    {s[len(s)//10]:.1f}")
            print(f"    P90:    {s[9*len(s)//10]:.1f}")
        
        times = [x['elapsed'] for x in ok]
        print(f"\n  Latency:")
        print(f"    Mean:   {statistics.mean(times):.2f}s")
        print(f"    Median: {statistics.median(times):.2f}s")
        print(f"    Min:    {min(times):.2f}s")
        print(f"    Max:    {max(times):.2f}s")
    
    if errs:
        ec = {}
        for x in errs:
            ec[x['status']] = ec.get(x['status'], 0) + 1
        print(f"\n  Errors:")
        for s, c in sorted(ec.items()):
            print(f"    {s}: {c}")
    
    return {"throughput": tp if ok else 0, "rejections": len(rej), "total": len(results), "ok": len(ok)}

def main():
    print(f"\n{'#'*60}")
    print(f"# QWEN LB BACKPRESSURE STRESS TEST (ultra-fast)")
    print(f"{'#'*60}")
    
    # Phase 1: Charlie single-threaded baseline
    r1 = run_batch("PHASE 1: Charlie single (5 reqs)", CHARLIE_URL, 5)
    
    # Phase 2: Local single-threaded baseline  
    r2 = run_batch("PHASE 2: Local single (5 reqs)", LOCAL_URL, 5)
    
    # Phase 3: Charlie concurrent (4 workers)
    r3 = run_batch("PHASE 3: Charlie concurrent (4)", CHARLIE_URL, 4)
    
    # Phase 4: LB moderate (10 concurrent)
    r4 = run_batch("PHASE 4: LB moderate (10)", LB_URL, 10, headers=LB_HEADERS)
    
    # Phase 5: LB heavy (50 concurrent) — expect backpressure!
    r5 = run_batch("PHASE 5: LB heavy (50)", LB_URL, 50, headers=LB_HEADERS)
    
    # Phase 6: LB extreme (100 concurrent) — max stress
    r6 = run_batch("PHASE 6: LB extreme (100)", LB_URL, 100, headers=LB_HEADERS)
    
    # Final comparison
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*60}")
    phases = [
        ("Charlie single", r1),
        ("Local single", r2),
        ("Charlie 4-conn", r3),
        ("LB 10-conn", r4),
        ("LB 50-conn", r5),
        ("LB 100-conn", r6),
    ]
    print(f"\n  {'Phase':25s} {'Throughput':>12s} {'Rejections':>12s} {'OK/Total':>12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    for name, r in phases:
        tp = f"{r['throughput']:.1f}" if 'throughput' in r else "N/A"
        rej = str(r.get('rejections', 0))
        ok = r.get('ok', 0)
        total = r.get('total', 0)
        ratio = f"{ok}/{total}"
        print(f"  {name:25s} {tp:>12s} {rej:>12s} {ratio:>12s}")
    
    print(f"\n{'#'*60}")
    print(f"# DONE")
    print(f"{'#'*60}\n")

if __name__ == "__main__":
    main()
