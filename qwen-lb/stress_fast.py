#!/usr/bin/env python3
"""Fast stress test — short responses (128 tok) so we can measure throughput quickly."""
import json, time, statistics, urllib.request, urllib.error, concurrent.futures

LB_URL = "http://127.0.0.1:8080/v1/chat/completions"
CHARLIE_URL = "http://192.168.1.201:8033/v1/chat/completions"
PROMPTS = [
    "Explain quantum entanglement in 150 words.",
    "Summarize the key points of special relativity in 150 words.",
    "Describe how neural networks learn in 150 words.",
    "What is the halting problem and why does it matter? Explain in 150 words.",
    "Compare SQL and NoSQL databases in 150 words.",
]

LB_HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer qwen-lb-2f1cf189d556f21c4b2f1c6ccc5121fd"}

def send(url, prompt, headers=None):
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.7,
    }).encode()
    hdrs = dict(LB_HEADERS) if headers is None else dict(headers)
    t0 = time.time()
    try:
        req = urllib.request.Request(url, data=payload, headers=hdrs)
        with urllib.request.urlopen(req, timeout=120) as resp:
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

def phase(name, url, max_workers, n_rounds, headers=None):
    print(f"\n{'='*60}")
    print(f"  {name}: {max_workers} concurrent, {n_rounds} rounds")
    print(f"{'='*60}")
    
    all_results = []
    start = time.time()
    
    for r in range(n_rounds):
        batch = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for i in range(max_workers):
                f = pool.submit(send, url, PROMPTS[i % len(PROMPTS)], headers)
                batch.append(f)
            for f in concurrent.futures.as_completed(batch):
                all_results.append(f.result())
        
        ok = [x for x in all_results if x['status'] == 'ok']
        rej = [x for x in all_results if x['status'] == '429']
        wall = time.time() - start
        
        if r + 1 == n_rounds:
            # Final round stats
            if ok:
                rates = [x['tok_s'] for x in ok]
                total_toks = sum(x['toks'] for x in ok)
                tp = total_toks / wall if wall > 0 else 0
                print(f"\n  Results:")
                print(f"    Duration:     {wall:.1f}s")
                print(f"    Total reqs:   {len(all_results)}")
                print(f"    Successful:   {len(ok)}")
                print(f"    Rejected:     {len(rej)}")
                print(f"    Throughput:   {tp:.1f} tok/s total")
                print(f"    Req/s:        {len(all_results)/wall:.1f}")
                print(f"\n    Token/sec:")
                print(f"      Mean:   {statistics.mean(rates):.1f}")
                print(f"      Median: {statistics.median(rates):.1f}")
                print(f"      Min:    {min(rates):.1f}")
                print(f"      Max:    {max(rates):.1f}")
                if len(rates) > 5:
                    s = sorted(rates)
                    print(f"      P10:    {s[len(s)//10]:.1f}")
                    print(f"      P90:    {s[9*len(s)//10]:.1f}")
                print(f"\n    Latency:")
                times = [x['elapsed'] for x in ok]
                print(f"      Mean:   {statistics.mean(times):.2f}s")
                print(f"      Median: {statistics.median(times):.2f}s")
                print(f"      Min:    {min(times):.2f}s")
                print(f"      Max:    {max(times):.2f}s")
            else:
                print(f"  No successful requests!")
        
        # Progress every round
        if len(ok) > 0:
            avg = statistics.mean([x['tok_s'] for x in ok])
            print(f"  Round {r+1}/{n_rounds}: {len(ok)} ok, {len(rej)} 429, {avg:.0f} tok/s avg")

def main():
    print(f"\n{'#'*60}")
    print(f"# QWEN LB BACKPRESSURE STRESS TEST (fast)")
    print(f"{'#'*60}")
    
    # Phase 1: Charlie single-threaded baseline
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Charlie single-threaded (5 requests)")
    print(f"{'='*60}")
    results = []
    for i in range(5):
        r = send(CHARLIE_URL, PROMPTS[i % len(PROMPTS)])
        results.append(r)
        print(f"  [{i+1}/5] {r['elapsed']:.1f}s | {r['toks']} tok | {r['tok_s']:.1f} tok/s")
    
    ok = [x for x in results if x['status'] == 'ok']
    if ok:
        avg = statistics.mean([x['tok_s'] for x in ok])
        print(f"\n  Charlie baseline: {avg:.1f} tok/s (single-threaded)")
    
    # Phase 2: Charlie concurrent (4 workers, 3 rounds)
    phase("PHASE 2: Charlie concurrent (4 workers)", CHARLIE_URL, 4, 3)
    
    # Phase 3: LB moderate (10 concurrent, 3 rounds)
    phase("PHASE 3: LB moderate (10 concurrent)", LB_URL, 10, 3, headers=LB_HEADERS)
    
    # Phase 4: LB heavy (50 concurrent, 3 rounds) — expect 429s
    phase("PHASE 4: LB heavy (50 concurrent)", LB_URL, 50, 3, headers=LB_HEADERS)
    
    # Phase 5: LB extreme (100 concurrent, 2 rounds) — max stress
    phase("PHASE 5: LB extreme (100 concurrent)", LB_URL, 100, 2, headers=LB_HEADERS)
    
    print(f"\n{'#'*60}")
    print(f"# DONE")
    print(f"{'#'*60}\n")

if __name__ == "__main__":
    main()
