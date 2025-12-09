"""
Fine-Tuning Module for Ethical AI Storyteller.

This module provides functionality to fine-tune GPT-2 models
on domain-specific storytelling datasets.

Features:
- Support for multiple dataset sources
- Configurable training parameters
- Training progress tracking
- Model checkpointing
- Evaluation during training
"""

import os
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed
)
from tqdm import tqdm

from .dataset import (
    StoryDatasetLoader,
    StoryPromptGenerator,
    StoryDataset,
    StoryExample
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    # Model settings
    model_name: str = "gpt2"
    output_dir: str = "models/finetuned-storyteller"
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Data settings
    max_length: int = 256
    train_split: float = 0.9
    
    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    
    # Other
    seed: int = 42
    fp16: bool = False  # Mixed precision training
    gradient_accumulation_steps: int = 1


class StorytellerFineTuner:
    """
    Fine-tune GPT-2 models on storytelling datasets.
    
    Supports:
    - Custom JSON datasets
    - HuggingFace datasets
    - Mixed dataset training
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize the fine-tuner.
        
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        
        # Set seed for reproducibility
        set_seed(self.config.seed)
        
        # Detect device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self._load_model()
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float("inf")
        
    def _load_model(self) -> None:
        """Load the pretrained model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.model.to(self.device)
        logger.info("Model loaded successfully")
    
    def prepare_dataset(
        self,
        examples: List[StoryExample],
        is_train: bool = True
    ) -> DataLoader:
        """
        Prepare a DataLoader from story examples.
        
        Args:
            examples: List of StoryExample objects
            is_train: Whether this is training data
            
        Returns:
            DataLoader for the dataset
        """
        dataset = StoryDataset(
            examples=examples,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=is_train,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False
        )
    
    def load_training_data(
        self,
        dataset_path: Optional[str] = None,
        hf_dataset: Optional[str] = None,
        max_samples: Optional[int] = None,
        use_generated_prompts: bool = True,
        num_generated: int = 100
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Load and prepare training data.
        
        Args:
            dataset_path: Path to custom JSON dataset
            hf_dataset: Name of HuggingFace dataset to use
            max_samples: Maximum samples to load
            use_generated_prompts: Whether to augment with generated prompts
            num_generated: Number of prompts to generate
            
        Returns:
            Tuple of (train_dataloader, eval_dataloader)
        """
        loader = StoryDatasetLoader()
        all_examples = []
        
        # Load custom dataset
        if dataset_path:
            logger.info(f"Loading custom dataset from {dataset_path}")
            examples = loader.load_custom_dataset(dataset_path, genre_field="genre")
            all_examples.extend(examples)
            logger.info(f"Loaded {len(examples)} examples from custom dataset")
        
        # Load HuggingFace dataset
        if hf_dataset:
            logger.info(f"Loading HuggingFace dataset: {hf_dataset}")
            try:
                examples = loader.load_dataset(
                    hf_dataset,
                    max_samples=max_samples or 1000
                )
                all_examples.extend(examples)
                logger.info(f"Loaded {len(examples)} examples from {hf_dataset}")
            except Exception as e:
                logger.warning(f"Could not load {hf_dataset}: {e}")
        
        # Generate additional prompts with sample completions
        if use_generated_prompts:
            logger.info(f"Generating {num_generated} additional prompts...")
            generator = StoryPromptGenerator(seed=self.config.seed)
            
            # Sample completions to pair with prompts
            completions = [
                "and discovered a hidden world beyond imagination.",
                "which changed everything they thought they knew.",
                "leading to an adventure that would be remembered forever.",
                "but what they found was beyond their wildest dreams.",
                "and the journey that followed tested their courage.",
                "revealing secrets that had been buried for centuries.",
                "setting in motion events that would reshape their destiny.",
                "and in that moment, everything became clear.",
            ]
            
            for _ in range(num_generated):
                prompt, genre = generator.generate_prompt()
                import random
                story = random.choice(completions)
                
                all_examples.append(StoryExample(
                    story_id=f"generated_{len(all_examples)}",
                    prompt=prompt,
                    story=story,
                    genre=genre,
                    metadata={"source": "generated"}
                ))
        
        if not all_examples:
            raise ValueError("No training data loaded! Provide dataset_path or hf_dataset")
        
        logger.info(f"Total examples: {len(all_examples)}")
        
        # Split into train/eval
        import random
        random.seed(self.config.seed)
        random.shuffle(all_examples)
        
        split_idx = int(len(all_examples) * self.config.train_split)
        train_examples = all_examples[:split_idx]
        eval_examples = all_examples[split_idx:]
        
        logger.info(f"Train examples: {len(train_examples)}")
        logger.info(f"Eval examples: {len(eval_examples)}")
        
        train_dataloader = self.prepare_dataset(train_examples, is_train=True)
        eval_dataloader = self.prepare_dataset(eval_examples, is_train=False)
        
        return train_dataloader, eval_dataloader
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Fine-tune the model.
        
        Args:
            train_dataloader: Training data
            eval_dataloader: Evaluation data (optional)
            
        Returns:
            Training history
        """
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Calculate total steps
        num_update_steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        total_steps = num_update_steps_per_epoch * self.config.num_epochs
        
        # Initialize optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training history
        history = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": []
        }
        
        logger.info("=" * 60)
        logger.info("Starting fine-tuning")
        logger.info(f"  Num epochs: {self.config.num_epochs}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info("=" * 60)
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                leave=True
            )
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                # Gradient accumulation
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update weights
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = epoch_loss / num_batches
                        current_lr = scheduler.get_last_lr()[0]
                        
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{current_lr:.2e}"
                        })
                        
                        history["train_loss"].append(avg_loss)
                        history["learning_rate"].append(current_lr)
                    
                    # Evaluation
                    if eval_dataloader and self.global_step % self.config.eval_steps == 0:
                        eval_loss = self.evaluate(eval_dataloader)
                        history["eval_loss"].append(eval_loss)
                        
                        logger.info(f"Step {self.global_step}: eval_loss = {eval_loss:.4f}")
                        
                        # Save best model
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_model(os.path.join(self.config.output_dir, "best"))
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        checkpoint_dir = os.path.join(
                            self.config.output_dir,
                            f"checkpoint-{self.global_step}"
                        )
                        self.save_model(checkpoint_dir)
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")
            
            # Evaluate at end of epoch
            if eval_dataloader:
                eval_loss = self.evaluate(eval_dataloader)
                logger.info(f"Epoch {epoch + 1} eval loss: {eval_loss:.4f}")
                history["eval_loss"].append(eval_loss)
        
        # Save final model
        self.save_model(os.path.join(self.config.output_dir, "final"))
        logger.info(f"Training complete! Model saved to {self.config.output_dir}")
        
        return history
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evaluate the model.
        
        Args:
            dataloader: Evaluation data
            
        Returns:
            Average loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_model(self, path: str) -> None:
        """Save the model and tokenizer."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    
    def generate_sample(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8
    ) -> str:
        """
        Generate a sample story to test the fine-tuned model.
        
        Args:
            prompt: Starting prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        self.model.eval()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length + inputs["input_ids"].shape[1],
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def finetune_on_stories(
    dataset_path: str = "data/sample_stories.json",
    output_dir: str = "models/finetuned-storyteller",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to fine-tune on storytelling data.
    
    Args:
        dataset_path: Path to the training dataset
        output_dir: Where to save the fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_samples: Maximum samples to use
        
    Returns:
        Training history
    """
    # Create config
    config = TrainingConfig(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Initialize trainer
    trainer = StorytellerFineTuner(config)
    
    # Load data
    train_loader, eval_loader = trainer.load_training_data(
        dataset_path=dataset_path,
        max_samples=max_samples,
        use_generated_prompts=True,
        num_generated=50
    )
    
    # Train
    history = trainer.train(train_loader, eval_loader)
    
    # Generate sample
    logger.info("\n" + "=" * 60)
    logger.info("Testing fine-tuned model:")
    logger.info("=" * 60)
    
    sample = trainer.generate_sample(
        "In a magical kingdom, a young wizard discovered",
        max_length=100
    )
    logger.info(f"\nGenerated: {sample}")
    
    return history


def main():
    """Main function for command-line fine-tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on storytelling data")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sample_stories.json",
        help="Path to training dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/finetuned-storyteller",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum training samples"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎓 Storyteller Fine-Tuning")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)
    
    history = finetune_on_stories(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples
    )
    
    print("\n✅ Fine-tuning complete!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
