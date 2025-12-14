#!/usr/bin/env python3
"""
Benchmark script to measure generation throughput (tokens/s).
Measures performance on available device (CPU, MPS, or CUDA).
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def benchmark_model(model_name: str, device: str, num_runs: int = 5, max_tokens: int = 100):
    """Benchmark a model's generation throughput."""
    print(f"\n🔧 Loading {model_name} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    model.eval()
    
    # Warmup run
    prompt = "Once upon a time in a magical kingdom"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Benchmark runs
    total_tokens = 0
    total_time = 0
    
    print(f"📊 Running {num_runs} benchmark iterations...")
    
    for i in range(num_runs):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Sync for accurate timing on GPU/MPS
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        
        end_time = time.perf_counter()
        
        generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        elapsed = end_time - start_time
        
        total_tokens += generated_tokens
        total_time += elapsed
        
        tokens_per_sec = generated_tokens / elapsed
        print(f"  Run {i+1}: {generated_tokens} tokens in {elapsed:.2f}s = {tokens_per_sec:.1f} tok/s")
    
    avg_throughput = total_tokens / total_time
    return avg_throughput


def main():
    print("=" * 60)
    print("🚀 THROUGHPUT BENCHMARK - Ethical AI Storyteller")
    print("=" * 60)
    
    # Detect available device
    if torch.cuda.is_available():
        gpu_device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        gpu_device = "mps"
        gpu_name = "Apple Silicon (MPS)"
    else:
        gpu_device = None
        gpu_name = None
    
    print(f"\n📱 Detected devices:")
    print(f"  - CPU: Available")
    if gpu_device:
        print(f"  - GPU: {gpu_name} ({gpu_device})")
    else:
        print(f"  - GPU: Not available")
    
    models = ["gpt2", "distilgpt2"]
    results = {}
    
    for model_name in models:
        results[model_name] = {}
        
        # Benchmark on CPU
        print(f"\n{'='*40}")
        print(f"Testing {model_name} on CPU...")
        cpu_throughput = benchmark_model(model_name, "cpu", num_runs=3, max_tokens=50)
        results[model_name]["cpu"] = cpu_throughput
        print(f"✅ {model_name} CPU: {cpu_throughput:.1f} tokens/s")
        
        # Benchmark on GPU if available
        if gpu_device:
            print(f"\nTesting {model_name} on {gpu_device.upper()}...")
            gpu_throughput = benchmark_model(model_name, gpu_device, num_runs=5, max_tokens=100)
            results[model_name]["gpu"] = gpu_throughput
            print(f"✅ {model_name} GPU: {gpu_throughput:.1f} tokens/s")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("📊 THROUGHPUT RESULTS (tokens/s)")
    print("=" * 60)
    print(f"{'Model':<20} {'CPU':<15} {'GPU':<15}")
    print("-" * 50)
    
    for model_name in models:
        cpu = results[model_name].get("cpu", 0)
        gpu = results[model_name].get("gpu", 0)
        gpu_str = f"{gpu:.1f}" if gpu else "N/A"
        print(f"{model_name:<20} {cpu:.1f}{'':>9} {gpu_str}")
    
    # Save results
    import json
    output = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device_info": {
            "cpu": "Available",
            "gpu": gpu_name if gpu_device else "Not available",
            "gpu_type": gpu_device
        },
        "results": results,
        "config": {
            "num_runs_cpu": 3,
            "num_runs_gpu": 5,
            "max_tokens_cpu": 50,
            "max_tokens_gpu": 100
        }
    }
    
    with open("results/evaluation/throughput_benchmark.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to results/evaluation/throughput_benchmark.json")


if __name__ == "__main__":
    main()

