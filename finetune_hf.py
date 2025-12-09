#!/usr/bin/env python3
"""
Fine-tune on HuggingFace storytelling datasets.

Usage:
    python finetune_hf.py --dataset tiny_stories --samples 500 --epochs 2
    python finetune_hf.py --dataset writing_prompts --samples 1000 --epochs 3
    python finetune_hf.py --dataset fairy_tales --samples 300 --epochs 2
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.finetune import StorytellerFineTuner, TrainingConfig
from src.dataset import StoryDatasetLoader


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on HuggingFace datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["tiny_stories", "writing_prompts", "fairy_tales", "roc_stories"],
        help="HuggingFace dataset to use"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Maximum samples to load (default: 500)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: models/finetuned-{dataset})"
    )
    parser.add_argument(
        "--combine-with-local",
        type=str,
        default=None,
        help="Also include local dataset (e.g., data/sample_stories.json)"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir or f"models/finetuned-{args.dataset}"
    
    print("=" * 60)
    print(f"🎓 Fine-Tuning on {args.dataset.upper()}")
    print("=" * 60)
    print(f"  Samples: {args.samples}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Output: {output_dir}")
    print("=" * 60)
    
    # Configure training
    config = TrainingConfig(
        output_dir=output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=20,
        eval_steps=50,
        save_steps=100
    )
    
    # Initialize trainer
    trainer = StorytellerFineTuner(config)
    
    # Load data
    print(f"\n📚 Loading {args.dataset} from HuggingFace...")
    
    train_loader, eval_loader = trainer.load_training_data(
        dataset_path=args.combine_with_local,
        hf_dataset=args.dataset,
        max_samples=args.samples,
        use_generated_prompts=True,
        num_generated=50
    )
    
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Evaluation batches: {len(eval_loader)}")
    
    # Train
    print("\n🚀 Starting fine-tuning...")
    history = trainer.train(train_loader, eval_loader)
    
    # Test
    print("\n" + "=" * 60)
    print("🧪 Testing fine-tuned model:")
    print("=" * 60)
    
    test_prompts = [
        "Once upon a time,",
        "In a magical kingdom,",
        "The brave hero discovered",
    ]
    
    for prompt in test_prompts:
        result = trainer.generate_sample(prompt, max_length=80)
        print(f"\n📝 {prompt}")
        print(f"   {result}")
    
    print("\n" + "=" * 60)
    print(f"✅ Fine-tuning complete!")
    print(f"📁 Model saved to: {output_dir}/final")
    print("=" * 60)
    
    # Usage instructions
    print("\n📖 To use your fine-tuned model:")
    print(f"""
from src.storyteller import EthicalStoryTeller

storyteller = EthicalStoryTeller(
    model_name="{output_dir}/final"
)

result = storyteller.generate_story(
    prompt="Your prompt here",
    max_length=150
)
print(result.story)
""")


if __name__ == "__main__":
    main()

