"""
Unit tests for finetune.py module.

Tests cover:
- TrainingConfig
- StorytellerFineTuner initialization
- Dataset preparation
- Training loop (mocked)
- Model saving
- Edge cases and error handling
"""

import pytest
import torch
import os
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.finetune import (
    TrainingConfig,
    StorytellerFineTuner
)
from src.dataset import StoryExample


class TestTrainingConfig:
    """Test suite for TrainingConfig dataclass."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.model_name == "gpt2"
        assert config.output_dir == "models/finetuned-storyteller"
        assert config.num_epochs == 3
        assert config.batch_size == 4
        assert config.learning_rate == 5e-5
        assert config.seed == 42
    
    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = TrainingConfig(
            model_name="gpt2-medium",
            num_epochs=5,
            batch_size=8,
            learning_rate=1e-4
        )
        
        assert config.model_name == "gpt2-medium"
        assert config.num_epochs == 5
        assert config.batch_size == 8
        assert config.learning_rate == 1e-4


class TestStorytellerFineTuner:
    """Test suite for StorytellerFineTuner class."""
    
    @patch('src.finetune.AutoModelForCausalLM')
    @patch('src.finetune.AutoTokenizer')
    def test_initialization_default(self, mock_tokenizer, mock_model):
        """Test default initialization."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        
        config = TrainingConfig()
        trainer = StorytellerFineTuner(config)
        
        assert trainer.config == config
        assert trainer.device in ["cuda", "mps", "cpu"]
        assert trainer.global_step == 0
        assert trainer.best_eval_loss == float("inf")
    
    @patch('src.finetune.AutoModelForCausalLM')
    @patch('src.finetune.AutoTokenizer')
    def test_initialization_custom_config(self, mock_tokenizer, mock_model):
        """Test initialization with custom config."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        
        config = TrainingConfig(model_name="gpt2-medium", num_epochs=5)
        trainer = StorytellerFineTuner(config)
        
        assert trainer.config.model_name == "gpt2-medium"
        assert trainer.config.num_epochs == 5
    
    @patch('src.finetune.AutoModelForCausalLM')
    @patch('src.finetune.AutoTokenizer')
    def test_prepare_dataset(self, mock_tokenizer, mock_model):
        """Test dataset preparation."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        
        config = TrainingConfig()
        trainer = StorytellerFineTuner(config)
        
        examples = [
            StoryExample(
                story_id="1",
                prompt="Test",
                story="Story",
                genre="fantasy"
            )
        ]
        
        dataloader = trainer.prepare_dataset(examples, is_train=True)
        
        assert dataloader is not None
        assert dataloader.batch_size == config.batch_size
    
    @patch('src.finetune.AutoModelForCausalLM')
    @patch('src.finetune.AutoTokenizer')
    @patch('src.finetune.StoryDatasetLoader')
    def test_load_training_data_from_file(self, mock_loader_class, mock_tokenizer, mock_model):
        """Test loading training data from file."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        
        # Mock dataset loader
        mock_loader = MagicMock()
        mock_example = StoryExample(
            story_id="1",
            prompt="Test",
            story="Story",
            genre="fantasy"
        )
        mock_loader.load_custom_dataset.return_value = [mock_example]
        mock_loader_class.return_value = mock_loader
        
        config = TrainingConfig()
        trainer = StorytellerFineTuner(config)
        
        train_loader, eval_loader = trainer.load_training_data(
            dataset_path="test.json",
            max_samples=10
        )
        
        assert train_loader is not None
        assert eval_loader is not None
    
    @patch('src.finetune.AutoModelForCausalLM')
    @patch('src.finetune.AutoTokenizer')
    @patch('src.finetune.StoryDatasetLoader')
    def test_load_training_data_no_data(self, mock_loader_class, mock_tokenizer, mock_model):
        """Test loading training data with no data sources."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        
        mock_loader = MagicMock()
        mock_loader.load_custom_dataset.return_value = []
        mock_loader_class.return_value = mock_loader
        
        config = TrainingConfig()
        trainer = StorytellerFineTuner(config)
        
        with pytest.raises(ValueError, match="No training data loaded"):
            trainer.load_training_data(
                dataset_path=None,
                hf_dataset=None,
                use_generated_prompts=False
            )
    
    @patch('src.finetune.AutoModelForCausalLM')
    @patch('src.finetune.AutoTokenizer')
    def test_evaluate(self, mock_tokenizer, mock_model):
        """Test model evaluation."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        
        # Mock model
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(2.5)
        mock_model_instance.return_value = mock_output
        mock_model.return_value = mock_model_instance
        
        config = TrainingConfig()
        trainer = StorytellerFineTuner(config)
        
        # Create mock dataloader
        mock_dataloader = MagicMock()
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3]])
        }
        mock_dataloader.__iter__ = MagicMock(return_value=iter([mock_batch]))
        
        eval_loss = trainer.evaluate(mock_dataloader)
        
        assert isinstance(eval_loss, float)
        assert eval_loss > 0
    
    @patch('src.finetune.AutoModelForCausalLM')
    @patch('src.finetune.AutoTokenizer')
    def test_save_model(self, mock_tokenizer, mock_model, tmp_path):
        """Test saving model."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        
        mock_model_instance = MagicMock()
        mock_model_instance.save_pretrained = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.save_pretrained = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        config = TrainingConfig(output_dir=str(tmp_path))
        trainer = StorytellerFineTuner(config)
        
        save_path = str(tmp_path / "test_model")
        trainer.save_model(save_path)
        
        mock_model_instance.save_pretrained.assert_called_once()
        mock_tokenizer_instance.save_pretrained.assert_called_once()
    
    @patch('src.finetune.AutoModelForCausalLM')
    @patch('src.finetune.AutoTokenizer')
    def test_generate_sample(self, mock_tokenizer, mock_model):
        """Test generating sample with fine-tuned model."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_instance.generate = MagicMock(return_value=mock_output)
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer_instance.decode = MagicMock(return_value="Generated text")
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        config = TrainingConfig()
        trainer = StorytellerFineTuner(config)
        
        sample = trainer.generate_sample("Test prompt", max_length=50)
        
        assert isinstance(sample, str)
        assert len(sample) > 0
    
    @patch('src.finetune.AutoModelForCausalLM')
    @patch('src.finetune.AutoTokenizer')
    @patch('torch.optim.AdamW')
    @patch('src.finetune.get_linear_schedule_with_warmup')
    def test_train_basic(self, mock_scheduler, mock_optimizer, mock_tokenizer, mock_model):
        """Test basic training loop (mocked)."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        
        mock_model_instance = MagicMock()
        mock_model_instance.train = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(2.0)
        mock_model_instance.return_value = mock_output
        mock_model.return_value = mock_model_instance
        
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.step = MagicMock()
        mock_optimizer_instance.zero_grad = MagicMock()
        mock_optimizer.return_value = mock_optimizer_instance
        
        mock_scheduler_instance = MagicMock()
        mock_scheduler_instance.step = MagicMock()
        mock_scheduler_instance.get_last_lr = MagicMock(return_value=[1e-5])
        mock_scheduler.return_value = mock_scheduler_instance
        
        config = TrainingConfig(num_epochs=1, batch_size=2, save_steps=1000, eval_steps=1000)
        trainer = StorytellerFineTuner(config)
        
        # Create mock dataloader
        mock_dataloader = MagicMock()
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3], [4, 5, 6]])
        }
        mock_dataloader.__iter__ = MagicMock(return_value=iter([mock_batch]))
        mock_dataloader.__len__ = MagicMock(return_value=1)
        
        history = trainer.train(mock_dataloader, eval_dataloader=None)
        
        assert "train_loss" in history
        assert "eval_loss" in history
        assert "learning_rate" in history


class TestFinetuneEdgeCases:
    """Test edge cases for finetune module."""
    
    @patch('src.finetune.AutoModelForCausalLM')
    @patch('src.finetune.AutoTokenizer')
    def test_prepare_dataset_empty(self, mock_tokenizer, mock_model):
        """Test preparing dataset with empty examples."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        
        config = TrainingConfig()
        trainer = StorytellerFineTuner(config)
        
        dataloader = trainer.prepare_dataset([], is_train=True)
        
        assert dataloader is not None
        assert len(list(dataloader)) == 0
    
    @patch('src.finetune.AutoModelForCausalLM')
    @patch('src.finetune.AutoTokenizer')
    def test_evaluate_empty_dataloader(self, mock_tokenizer, mock_model):
        """Test evaluation with empty dataloader."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_model.return_value = mock_model_instance
        
        config = TrainingConfig()
        trainer = StorytellerFineTuner(config)
        
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter([]))
        
        # Should handle empty dataloader gracefully
        with pytest.raises((ZeroDivisionError, ValueError)):
            trainer.evaluate(mock_dataloader)

