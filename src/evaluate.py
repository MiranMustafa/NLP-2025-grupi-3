"""
Lightweight evaluation utilities for the Ethical AI Storyteller.

Reports perplexity, safety, and bias statistics on a small dataset so
results in the README can be reproduced.
"""

import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dataset import StoryDatasetLoader
from .ethical_filter import EthicalFilter, BiasDetector


def load_texts(
    dataset_path: Optional[str],
    hf_dataset: Optional[str],
    max_samples: int = 50
) -> List[str]:
    """Load stories from either a local JSON/JSONL file or a HuggingFace dataset."""
    loader = StoryDatasetLoader()
    examples = []

    if dataset_path:
        examples = loader.load_custom_dataset(dataset_path)
    elif hf_dataset:
        examples = loader.load_dataset(hf_dataset, max_samples=max_samples)
    else:
        raise ValueError("Provide either dataset_path or hf_dataset")

    texts: List[str] = []
    for ex in examples[:max_samples]:
        combined = f"{ex.prompt.strip()} {ex.story.strip()}".strip()
        texts.append(combined)
    return texts


def load_model(model_name: str, device: Optional[str] = None):
    """Load a causal LM and tokenizer."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    max_length: int = 512
) -> float:
    """Compute token-level perplexity over the provided texts."""
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(device)

            outputs = model(**enc, labels=enc["input_ids"])
            seq_len = enc["input_ids"].shape[1]
            total_loss += outputs.loss.item() * seq_len
            total_tokens += seq_len

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def evaluate_safety_and_bias(
    texts: List[str],
    safety_filter: EthicalFilter,
    bias_detector: BiasDetector
) -> Tuple[float, float]:
    """Return (safe_ratio, bias_ratio)."""
    safe = 0
    biased = 0

    for text in texts:
        filt = safety_filter.filter_content(text)
        if filt.is_safe:
            safe += 1

        bias = bias_detector.analyze_bias(text)
        if bias.get("has_bias"):
            biased += 1

    total = max(len(texts), 1)
    return safe / total, biased / total


def run_eval(
    model_name: str,
    dataset_path: Optional[str],
    hf_dataset: Optional[str],
    max_samples: int = 50
) -> Dict[str, float]:
    """Run perplexity, safety, and bias evaluations."""
    texts = load_texts(dataset_path, hf_dataset, max_samples=max_samples)
    model, tokenizer, device = load_model(model_name)
    safety_filter = EthicalFilter(use_detoxify=False)
    bias_detector = BiasDetector()

    ppl = compute_perplexity(model, tokenizer, texts, device=device)
    safe_ratio, bias_ratio = evaluate_safety_and_bias(texts, safety_filter, bias_detector)

    return {
        "perplexity": ppl,
        "safe_ratio": safe_ratio,
        "bias_ratio": bias_ratio,
        "num_samples": len(texts),
        "model": model_name
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ethical AI Storyteller models")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--dataset", type=str, default="data/sample_stories.json", help="Local dataset path")
    parser.add_argument("--hf-dataset", type=str, default=None, help="HuggingFace dataset name")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum samples to evaluate")
    args = parser.parse_args()

    metrics = run_eval(
        model_name=args.model,
        dataset_path=args.dataset,
        hf_dataset=args.hf_dataset,
        max_samples=args.max_samples
    )

    print("\n📊 Evaluation Results")
    for key, value in metrics.items():
        if key.endswith("ratio"):
            print(f"- {key}: {value:.1%}")
        elif key == "perplexity":
            print(f"- {key}: {value:.2f}")
        else:
            print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
