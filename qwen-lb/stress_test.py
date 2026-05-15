#!/usr/bin/env python3
"""
Stress test for LB backpressure implementation.

Measures:
  - Tokens/sec on Charlie's server (direct and via LB)
  - 429 rejection behavior under saturation
  - Latency distribution under load
  - Sustained throughput over time
  - Recovery after saturation clears

Usage:
  python stress_test.py                          # default: 50 concurrent, 60s
  python stress_test.py --concurrent 100 --duration 120
"""
import json
import time
import sys
import urllib.request
import urllib.error
import statistics
import threading
from collections import defaultdict
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

class StressTester:
    def __init__(self, concurrent=50, duration=60, target="lb"):
        self.concurrent = concurrent
        self.duration = duration
        self.target = target  # "lb" or "charlie"
        self.url = LB_URL if target == "lb" else CHARLIE_URL
        
        # Results tracking
        self.results = []
        self.lock = threading.Lock()
        self.rejections = 0
        self.start_time = 0
        self.heartbeat = threading.Thread(target=self._heartbeat, daemon=True)
    
    def _send_request(self, prompt, task_id):
        payload = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
        }).encode()
        
        headers = {"Content-Type": "application/json"}
        if self.target == "lb":
            headers["X-API-Key"] = "qwen-lb-2f1cf189d556f21c4b2f1c6ccc5121fd"
        
        t0 = time.time()
        try:
            req = urllib.request.Request(self.url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
                elapsed = time.time() - t0
                gen_n = data.get("usage", {}).get("completion_tokens", 0)
                tok_s = gen_n / elapsed if elapsed > 0 else 0
                content_len = len(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
                return {
                    "task_id": task_id,
                    "elapsed": elapsed,
                    "tokens": gen_n,
                    "tok_s": tok_s,
                    "content_len": content_len,
                    "status": "ok",
                    "http_status": 200,
                    "backend": self.target,
                }
        except urllib.error.HTTPError as e:
            elapsed = time.time() - t0
            if e.code == 429:
                return {
                    "task_id": task_id,
                    "elapsed": elapsed,
                    "tokens": 0,
                    "tok_s": 0,
                    "content_len": 0,
                    "status": "rejected",
                    "http_status": 429,
                    "backend": self.target,
                }
            else:
                return {
                    "task_id": task_id,
                    "elapsed": elapsed,
                    "tokens": 0,
                    "tok_s": 0,
                    "content_len": 0,
                    "status": f"error_{e.code}",
                    "http_status": e.code,
                    "backend": self.target,
                }
        except Exception as e:
            elapsed = time.time() - t0
            return {
                "task_id": task_id,
                "elapsed": elapsed,
                "tokens": 0,
                "tok_s": 0,
                "content_len": 0,
                "status": str(e)[:50],
                "http_status": 0,
                "backend": self.target,
            }

    def _heartbeat(self):
        """Log health status every 2 seconds during the test."""
        while time.time() - self.start_time < self.duration:
            try:
                req = urllib.request.Request(
                    LB_HEALTH,
                    headers={"Authorization": "Bearer qwen-lb-2f1cf189d556f21c4b2f1c6ccc5121fd"}
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                with self.lock:
                    self.latest_health = data
            except:
                pass
            time.sleep(2)

    def _health_log(self):
        """Log periodic health status."""
        print("\n--- Health Log ---")
        if hasattr(self, 'latest_health'):
            h = self.latest_health
            backends = {b['name']: b for b in h.get('backends', [])}
            for name, b in backends.items():
                queue_depth = b.get('slots_processing', 0) + b.get('active_requests', 0)
                print(f"  {name:10s}: idle={b['slots_idle']:2d} proc={b['slots_processing']:2d} "
                      f"active={b.get('active_requests', 0):2d} queue={queue_depth}")
            print(f"  total: active_requests={h.get('active_requests', 0)}")
        else:
            print("  (no health data)")

    def run(self):
        print(f"\n{'='*70}")
        print(f"  STRESS TEST: {self.target.upper()}")
        print(f"  Concurrent requests: {self.concurrent}")
        print(f"  Duration: {self.duration}s")
        print(f"  Endpoint: {self.url}")
        print(f"{'='*70}")
        
        self.start_time = time.time()
        self.heartbeat.start()
        
        completed = 0
        total_tasks = 0
        results = []
        
        t_end = self.start_time + self.duration
        
        while time.time() < t_end:
            # Launch a batch of concurrent requests
            batch_size = min(self.concurrent, self.concurrent)
            futures = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
                for i in range(batch_size):
                    total_tasks += 1
                    prompt = PROMPTS[(total_tasks - 1) % len(PROMPTS)]
                    f = pool.submit(self._send_request, prompt, total_tasks)
                    futures.append(f)
                
                for f in concurrent.futures.as_completed(futures):
                    r = f.result()
                    with self.lock:
                        results.append(r)
                        if r['status'] == 'rejected':
                            self.rejections += 1
                    completed += 1
            
            if total_tasks % self.concurrent == 0:
                self._health_log()
        
        elapsed_total = time.time() - self.start_time
        self.results = results
        return self._report(elapsed_total, total_tasks)

    def _report(self, elapsed_total, total_tasks):
        ok_results = [r for r in self.results if r['status'] == 'ok']
        rejected = [r for r in self.results if r['status'] == 'rejected']
        errors = [r for r in self.results if r['status'] not in ('ok', 'rejected')]
        
        if not ok_results:
            print("\nNo successful results!")
            return {}
        
        times = [r['elapsed'] for r in ok_results]
        tok_rates = [r['tok_s'] for r in ok_results]
        tokens = [r['tokens'] for r in ok_results]
        
        total_tokens = sum(tokens)
        total_throughput = total_tokens / elapsed_total if elapsed_total > 0 else 0
        
        # Per-backend breakdown if possible
        print(f"\n{'='*70}")
        print(f"  RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"\n  Duration:          {elapsed_total:.1f}s wall")
        print(f"  Total requests:    {total_tasks}")
        print(f"  Successful:        {len(ok_results)}")
        print(f"  Rejected (429):    {len(rejected)}")
        print(f"  Errors:            {len(errors)}")
        print(f"\n  Total tokens gen:  {total_tokens}")
        print(f"  Throughput:        {total_throughput:.1f} tok/s (total)")
        
        if self.target == 'lb':
            # Estimate per-backend throughput
            print(f"\n  NOTE: With 2 backends and priority_fallback,")
            print(f"  throughput is split between local and remote.")
            print(f"  See per-backend breakdown below.")
        
        print(f"\n  Token/sec statistics (successful requests):")
        print(f"    Mean:    {statistics.mean(tok_rates):.1f} tok/s")
        print(f"    Median:  {statistics.median(tok_rates):.1f} tok/s")
        print(f"    Min:     {min(tok_rates):.1f} tok/s")
        print(f"    Max:     {max(tok_rates):.1f} tok/s")
        print(f"    StdDev:  {statistics.stdev(tok_rates):.1f}")
        if len(tok_rates) > 5:
            p10 = sorted(tok_rates)[len(tok_rates)//10]
            p90 = sorted(tok_rates)[9*len(tok_rates)//10]
            print(f"    P10:     {p10:.1f} tok/s")
            print(f"    P90:     {p90:.1f} tok/s")
        
        print(f"\n  Latency statistics:")
        print(f"    Mean:    {statistics.mean(times):.1f}s")
        print(f"    Median:  {statistics.median(times):.1f}s")
        print(f"    Min:     {min(times):.1f}s")
        print(f"    Max:     {max(times):.1f}s")
        print(f"    StdDev:  {statistics.stdev(times):.1f}")
        if len(times) > 5:
            p10 = sorted(times)[len(times)//10]
            p90 = sorted(times)[9*len(times)//10]
            print(f"    P10:     {p10:.1f}s")
            print(f"    P90:     {p90:.1f}s")
        
        print(f"\n  Average tokens per request: {statistics.mean(tokens):.1f}")
        
        if errors:
            print(f"\n  Errors ({len(errors)}):")
            err_counts = defaultdict(int)
            for r in errors:
                err_counts[r['status']] += 1
            for status, count in sorted(err_counts.items()):
                print(f"    {status}: {count}")
        
        if rejected:
            print(f"\n  429 Rejections ({len(rejected)}):")
            print(f"    Rejection rate: {len(rejected)/total_tasks*100:.1f}%")
        
        return {
            'total_tokens': total_tokens,
            'throughput': total_throughput,
            'mean_tok_s': statistics.mean(tok_rates),
            'median_tok_s': statistics.median(tok_rates),
            'rejections': len(rejected),
            'total_requests': total_tasks,
        }


def stress_charlie_baseline():
    """Baseline: measure Charlie's server directly (single-threaded, steady load)."""
    print(f"\n{'='*70}")
    print(f"  BASELINE: Charlie's server (single-threaded)")
    print(f"{'='*70}")
    
    times = []
    tokens = []
    n = 10
    
    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        payload = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.7,
        }).encode()
        headers = {"Content-Type": "application/json"}
        
        t0 = time.time()
        req = urllib.request.Request(CHARLIE_URL, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        elapsed = time.time() - t0
        gen_n = data.get("usage", {}).get("completion_tokens", 0)
        tok_s = gen_n / elapsed if elapsed > 0 else 0
        
        times.append(elapsed)
        tokens.append(gen_n)
        
        print(f"  [{i+1:2d}/{n}] {elapsed:5.1f}s | {gen_n:4d} tok | {tok_s:5.1f} tok/s")
    
    avg_tok_s = statistics.mean([t/e for t, e in zip(tokens, times)])
    print(f"\n  Baseline avg: {avg_tok_s:.1f} tok/s (single-threaded, direct to Charlie)")
    print(f"  Baseline avg latency: {statistics.mean(times):.1f}s")
    
    return avg_tok_s


def stress_lb_saturated():
    """Stress test via LB at high concurrency to trigger backpressure."""
    tester = StressTester(concurrent=100, duration=60, target="lb")
    return tester.run()


def stress_lb_moderate():
    """Moderate load via LB (no backpressure expected)."""
    tester = StressTester(concurrent=10, duration=60, target="lb")
    return tester.run()


def main():
    import concurrent.futures
    
    concurrent_mode = "--concurrent" in sys.argv
    target = "lb"
    duration = 60
    
    if "--concurrent" in sys.argv:
        idx = sys.argv.index("--concurrent")
        if idx + 1 < len(sys.argv):
            concurrent = int(sys.argv[idx + 1])
        else:
            concurrent = 100
    else:
        concurrent = 50
    
    if "--duration" in sys.argv:
        idx = sys.argv.index("--duration")
        if idx + 1 < len(sys.argv):
            duration = int(sys.argv[idx + 1])
    
    if "--target" in sys.argv:
        idx = sys.argv.index("--target")
        if idx + 1 < len(sys.argv):
            target = sys.argv[idx + 1]
    
    print(f"\n{'#'*70}")
    print(f"# QWEN LB BACKPRESSURE STRESS TEST")
    print(f"{'#'*70}")
    
    results = {}
    
    # Step 1: Baseline - Charlie's server direct (single-threaded)
    baseline = stress_charlie_baseline()
    results['charlie_baseline'] = {'tok_s': baseline}
    
    # Step 2: Baseline - Charlie's server (multi-threaded, concurrent=4 to match slots)
    print(f"\n{'='*70}")
    print(f"  BASELINE: Charlie's server (multi-threaded, concurrent=4)")
    print(f"{'='*70}")
    
    tester = StressTester(concurrent=4, duration=30, target="charlie")
    results['charlie_concurrent4'] = tester.run()
    
    # Step 3: Moderate LB load (no backpressure expected, 10 concurrent)
    print(f"\n{'='*70}")
    print(f"  TEST: LB moderate load (10 concurrent)")
    print(f"{'='*70}")
    results['lb_moderate'] = stress_lb_moderate()
    
    # Step 4: Heavy LB load (expect some backpressure)
    print(f"\n{'='*70}")
    print(f"  TEST: LB heavy load (50 concurrent)")
    print(f"{'='*70}")
    results['lb_heavy'] = StressTester(concurrent=50, duration=60, target="lb").run()
    
    # Step 5: Extreme LB load (expect lots of backpressure)
    print(f"\n{'='*70}")
    print(f"  TEST: LB extreme load (100 concurrent)")
    print(f"{'='*70}")
    results['lb_extreme'] = StressTester(concurrent=100, duration=45, target="lb").run()
    
    # Final comparison
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"\n  {'Scenario':25s} {'Throughput':>12s} {'Rejections':>12s} {'Req/s':>12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    
    for name, r in results.items():
        if 'throughput' in r:
            tok_s = f"{r['throughput']:.1f}"
            rej = f"{r['rejections']}"
            req_s = f"{r['total_requests'] / (duration if 'duration' in r else 60):.1f}"
        else:
            tok_s = f"{r.get('tok_s', 0):.1f}"
            rej = "N/A"
            req_s = "N/A"
        print(f"  {name:25s} {tok_s:>12s} {rej:>12s} {req_s:>12s}")
    
    print(f"\n{'='*70}")
    print(f"  TEST COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
