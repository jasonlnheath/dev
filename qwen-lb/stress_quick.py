#!/usr/bin/env python3
"""Fast stress test for LB backpressure.

Phases:
1. Charlie baseline (single-threaded, 5 requests)
2. Charlie concurrent (4 workers, 30s)
3. LB moderate (10 concurrent, 30s)
4. LB heavy (50 concurrent, 30s) — expect 429s
5. LB extreme (100 concurrent, 20s) — expect lots of 429s
"""
import json
import time
import sys
import urllib.request
import urllib.error
import statistics
import concurrent.futures

LB_URL = "http://127.0.0.1:8080/v1/chat/completions"
LB_HEALTH = "http://127.0.0.1:8080/health"
CHARLIE_URL = "http://192.168.1.201:8033/v1/chat/completions"

PROMPTS = [
    "Write a detailed explanation of how transformer attention mechanisms work, including the mathematical formulation of scaled dot-product attention, multi-head attention, and positional encoding. Discuss the computational complexity and memory requirements.",
    "Explain the differences between batch normalization, layer normalization, and instance normalization in deep learning. For each, provide the mathematical formula, discuss when it's most appropriate to use, and explain why it helps with training stability.",
    "Describe the training process for a large language model from scratch. Include details about data preprocessing, tokenization, model architecture choices, loss functions, optimizer settings, learning rate scheduling, gradient clipping, and distributed training strategies.",
    "What are the trade-offs between greedy decoding, beam search, and nucleus (top-p) sampling for text generation? Provide a mathematical explanation of how each method selects the next token and discuss their respective strengths and weaknesses in terms of diversity, coherence, and computational cost.",
    "Explain how RLHF (Reinforcement Learning from Human Feedback) works for aligning large language models. Cover preference data collection, reward model training, PPO algorithm details, KL divergence penalties, and the challenges of reward hacking and distribution shift.",
]

def send_request(url, prompt, headers=None, timeout=300):
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
    }).encode()
    
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    
    t0 = time.time()
    try:
        req = urllib.request.Request(url, data=payload, headers=hdrs)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            elapsed = time.time() - t0
            gen_n = data.get("usage", {}).get("completion_tokens", 0)
            tok_s = gen_n / elapsed if elapsed > 0 else 0
            return {
                "elapsed": elapsed, "tokens": gen_n, "tok_s": tok_s,
                "status": "ok", "http_status": resp.status,
            }
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        return {"elapsed": elapsed, "tokens": 0, "tok_s": 0,
                "status": f"error_{e.code}", "http_status": e.code}
    except Exception as e:
        elapsed = time.time() - t0
        return {"elapsed": elapsed, "tokens": 0, "tok_s": 0,
                "status": str(e)[:60], "http_status": 0}

def run_phase(name, url, concurrent, duration, headers=None):
    print(f"\n{'='*70}")
    print(f"  PHASE: {name}")
    print(f"  Concurrent: {concurrent}, Duration: {duration}s")
    print(f"{'='*70}")
    
    results = []
    start = time.time()
    task_id = [0]
    
    while time.time() - start < duration:
        batch = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as pool:
            for _ in range(concurrent):
                task_id[0] += 1
                prompt = PROMPTS[(task_id[0] - 1) % len(PROMPTS)]
                f = pool.submit(send_request, url, prompt, headers)
                batch.append(f)
            
            for f in concurrent.futures.as_completed(batch):
                results.append(f.result())
        
        elapsed = time.time() - start
        ok = sum(1 for r in results if r['status'] == 'ok')
        rej = sum(1 for r in results if r['http_status'] == 429)
        err = len(results) - ok - rej
        
        # Print health every 5s
        if int(elapsed) % 5 == 0 and int(elapsed) > 0:
            try:
                req = urllib.request.Request(LB_HEALTH, headers={
                    "Authorization": "Bearer qwen-lb-2f1cf189d556f21c4b2f1c6ccc5121fd"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    h = json.loads(resp.read())
                backends = {b['name']: b for b in h.get('backends', [])}
                print(f"  [t+{int(elapsed):3d}s] total={len(results):4d} ok={ok:4d} "
                      f"429={rej:4d} err={err:3d}", end="")
                for n, b in backends.items():
                    qd = b.get('slots_processing', 0) + b.get('active_requests', 0)
                    print(f" | {n}: proc={b['slots_processing']} active={b.get('active_requests',0)} queue={qd}", end="")
                print()
            except:
                pass
        
        # Print progress every 10 results
        if len(results) % 10 == 0 and len(results) > 0:
            ok_results = [r for r in results if r['status'] == 'ok']
            if ok_results:
                avg_tok_s = statistics.mean([r['tok_s'] for r in ok_results])
                print(f"  [t+{int(elapsed):3d}s] {len(results):4d} done, "
                      f"{avg_tok_s:.1f} tok/s avg so far")
    
    wall = time.time() - start
    return results, wall


def phase_charlie_single():
    """Single-threaded baseline on Charlie."""
    print(f"\n{'='*70}")
    print(f"  PHASE: Charlie single-threaded baseline (5 requests)")
    print(f"{'='*70}")
    
    times = []
    tokens = []
    for i in range(5):
        r = send_request(CHARLIE_URL, PROMPTS[i % len(PROMPTS)])
        times.append(r['elapsed'])
        tokens.append(r['tokens'])
        tok_s = r['tok_s']
        print(f"  [{i+1}/5] {r['elapsed']:5.1f}s | {r['tokens']:4d} tok | {tok_s:5.1f} tok/s")
    
    avg_tok_s = statistics.mean([t/e for t, e in zip(tokens, times)])
    print(f"\n  Charlie single-threaded baseline: {avg_tok_s:.1f} tok/s")
    return avg_tok_s


def report(name, results, wall):
    ok = [r for r in results if r['status'] == 'ok']
    rej = [r for r in results if r['http_status'] == 429]
    errors = [r for r in results if r['status'] not in ('ok',)]
    
    if not ok:
        print(f"  No successful requests!")
        return {'throughput': 0, 'mean_tok_s': 0, 'rejections': len(rej), 'total': len(results)}
    
    tok_rates = [r['tok_s'] for r in ok]
    times = [r['elapsed'] for r in ok]
    total_tokens = sum(r['tokens'] for r in ok)
    throughput = total_tokens / wall if wall > 0 else 0
    
    print(f"\n  Results ({name}):")
    print(f"    Duration:       {wall:.1f}s")
    print(f"    Total requests: {len(results)}")
    print(f"    Successful:     {len(ok)}")
    print(f"    Rejected (429): {len(rej)}")
    print(f"    Errors:         {len(errors)}")
    print(f"    Throughput:     {throughput:.1f} tok/s total")
    print(f"    Req/s:          {len(results)/wall:.1f}")
    print(f"\n    Token/sec stats:")
    print(f"      Mean:   {statistics.mean(tok_rates):.1f}")
    print(f"      Median: {statistics.median(tok_rates):.1f}")
    print(f"      Min:    {min(tok_rates):.1f}")
    print(f"      Max:    {max(tok_rates):.1f}")
    if len(tok_rates) > 5:
        s = sorted(tok_rates)
        p10, p90 = s[len(s)//10], s[9*len(s)//10]
        print(f"      P10:    {p10:.1f}")
        print(f"      P90:    {p90:.1f}")
    
    print(f"\n    Latency stats:")
    print(f"      Mean:   {statistics.mean(times):.1f}s")
    print(f"      Median: {statistics.median(times):.1f}s")
    print(f"      Min:    {min(times):.1f}s")
    print(f"      Max:    {max(times):.1f}s")
    
    if errors:
        ec = {}
        for r in errors:
            ec[r['status']] = ec.get(r['status'], 0) + 1
        print(f"\n    Errors:")
        for s, c in sorted(ec.items()):
            print(f"      {s}: {c}")
    
    return {
        'throughput': throughput,
        'mean_tok_s': statistics.mean(tok_rates),
        'rejections': len(rej),
        'total': len(results),
    }


def main():
    print(f"\n{'#'*70}")
    print(f"# QWEN LB BACKPRESSURE STRESS TEST")
    print(f"{'#'*70}")
    
    results = {}
    
    # Phase 1: Charlie single-threaded baseline
    results['charlie_single'] = phase_charlie_single()
    
    # Phase 2: Charlie concurrent (4 workers)
    r, wall = run_phase("Charlie concurrent (4 workers)", CHARLIE_URL, 4, 30)
    results['charlie_concurrent4'] = report("Charlie concurrent 4", r, wall)
    
    # Phase 3: LB moderate (10 concurrent)
    r, wall = run_phase("LB moderate (10 concurrent)", LB_URL, 10, 30, {
        "Authorization": "Bearer qwen-lb-2f1cf189d556f21c4b2f1c6ccc5121fd"})
    results['lb_moderate'] = report("LB moderate 10", r, wall)
    
    # Phase 4: LB heavy (50 concurrent) — expect backpressure
    r, wall = run_phase("LB heavy (50 concurrent)", LB_URL, 50, 30, {
        "Authorization": "Bearer qwen-lb-2f1cf189d556f21c4b2f1c6ccc5121fd"})
    results['lb_heavy'] = report("LB heavy 50", r, wall)
    
    # Phase 5: LB extreme (100 concurrent) — max stress
    r, wall = run_phase("LB extreme (100 concurrent)", LB_URL, 100, 20, {
        "Authorization": "Bearer qwen-lb-2f1cf189d556f21c4b2f1c6ccc5121fd"})
    results['lb_extreme'] = report("LB extreme 100", r, wall)
    
    # Final comparison table
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"\n  {'Scenario':30s} {'Throughput':>12s} {'Rejections':>12s} {'Req/s':>10s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
    
    for name, r in results.items():
        if isinstance(r, dict):
            tp = f"{r.get('throughput', 0):.1f}"
            rej = str(r.get('rejections', 0))
            req_s = f"{r.get('total', 0) / 30:.1f}" if r.get('total') else "N/A"
        else:
            tp = f"{r:.1f}"
            rej = "N/A"
            req_s = "N/A"
        print(f"  {name:30s} {tp:>12s} {rej:>12s} {req_s:>10s}")
    
    print(f"\n{'#'*70}")
    print(f"# STRESS TEST COMPLETE")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
