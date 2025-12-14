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




if __name__ == "__main__":
    main()

