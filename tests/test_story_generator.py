"""
Unit tests for story_generator.py module.

Tests cover:
- Model initialization
- Story generation with various parameters
- Genre-based generation
- Story continuation
- Edge cases and error handling
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.story_generator import StoryGenerator


class TestStoryGenerator:
    """Test suite for StoryGenerator class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        with patch('src.story_generator.AutoModelForCausalLM') as mock_model, \
             patch('src.story_generator.AutoTokenizer') as mock_tokenizer:
            
            mock_tokenizer.return_value.pad_token = None
            mock_tokenizer.return_value.eos_token = "<|endoftext|>"
            
            generator = StoryGenerator()
            
            assert generator.model_name == "gpt2"
            assert generator.device in ["cuda", "mps", "cpu"]
            assert generator.seed == 42
    
    def test_initialization_custom_model(self):
        """Test initialization with custom model."""
        with patch('src.story_generator.AutoModelForCausalLM') as mock_model, \
             patch('src.story_generator.AutoTokenizer') as mock_tokenizer:
            
            mock_tokenizer.return_value.pad_token = None
            mock_tokenizer.return_value.eos_token = "<|endoftext|>"
            
            generator = StoryGenerator(model_name="gpt2-medium")
            assert generator.model_name == "gpt2-medium"
    
    def test_initialization_custom_device(self):
        """Test initialization with custom device."""
        with patch('src.story_generator.AutoModelForCausalLM') as mock_model, \
             patch('src.story_generator.AutoTokenizer') as mock_tokenizer:
            
            mock_tokenizer.return_value.pad_token = None
            mock_tokenizer.return_value.eos_token = "<|endoftext|>"
            
            generator = StoryGenerator(device="cpu")
            assert generator.device == "cpu"
    
    def test_initialization_custom_seed(self):
        """Test initialization with custom seed."""
        with patch('src.story_generator.AutoModelForCausalLM') as mock_model, \
             patch('src.story_generator.AutoTokenizer') as mock_tokenizer:
            
            mock_tokenizer.return_value.pad_token = None
            mock_tokenizer.return_value.eos_token = "<|endoftext|>"
            
            generator = StoryGenerator(seed=123)
            assert generator.seed == 123
    
    def test_pad_token_setup(self):
        """Test that pad token is set correctly."""
        with patch('src.story_generator.AutoModelForCausalLM') as mock_model, \
             patch('src.story_generator.AutoTokenizer') as mock_tokenizer:
            
            mock_tokenizer.return_value.pad_token = None
            mock_tokenizer.return_value.eos_token = "<|endoftext|>"
            mock_tokenizer.return_value.pad_token_id = 50256
            
            generator = StoryGenerator()
            assert generator.tokenizer.pad_token == "<|endoftext|>"
    
    @patch('src.story_generator.AutoModelForCausalLM')
    @patch('src.story_generator.AutoTokenizer')
    def test_generate_story_basic(self, mock_tokenizer, mock_model):
        """Test basic story generation."""
        # Setup mocks
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        mock_tokenizer.return_value.eos_token_id = 50256
        
        # Mock tokenizer encoding
        mock_tokenizer.return_value.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
        }
        
        # Mock model generation
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        
        # Create mock output
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        mock_model_instance.generate = MagicMock(return_value=mock_output)
        mock_model.return_value = mock_model_instance
        
        # Mock decode
        mock_tokenizer.return_value.decode = MagicMock(return_value="Once upon a time there was a story")
        
        generator = StoryGenerator()
        result = generator.generate_story("Once upon a time", max_length=50)
        
        assert "prompt" in result
        assert "stories" in result
        assert "full_texts" in result
        assert result["prompt"] == "Once upon a time"
    
    @patch('src.story_generator.AutoModelForCausalLM')
    @patch('src.story_generator.AutoTokenizer')
    def test_generate_story_empty_prompt(self, mock_tokenizer, mock_model):
        """Test story generation with empty prompt."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        mock_tokenizer.return_value.eos_token_id = 50256
        
        mock_tokenizer.return_value.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]])
        }
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3]])
        mock_model_instance.generate = MagicMock(return_value=mock_output)
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value.decode = MagicMock(return_value="Generated story")
        
        generator = StoryGenerator()
        result = generator.generate_story("", max_length=50)
        
        assert result["prompt"] == ""
        assert len(result["stories"]) > 0
    
    @patch('src.story_generator.AutoModelForCausalLM')
    @patch('src.story_generator.AutoTokenizer')
    def test_generate_story_different_temperatures(self, mock_tokenizer, mock_model):
        """Test story generation with different temperature values."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        mock_tokenizer.return_value.eos_token_id = 50256
        
        mock_tokenizer.return_value.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_instance.generate = MagicMock(return_value=mock_output)
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value.decode = MagicMock(return_value="Story text")
        
        generator = StoryGenerator()
        
        # Test low temperature
        result1 = generator.generate_story("Once", temperature=0.1, max_length=50)
        assert result1["generation_params"]["temperature"] == 0.1
        
        # Test high temperature
        result2 = generator.generate_story("Once", temperature=1.5, max_length=50)
        assert result2["generation_params"]["temperature"] == 1.5
    
    @patch('src.story_generator.AutoModelForCausalLM')
    @patch('src.story_generator.AutoTokenizer')
    def test_generate_with_genre_valid(self, mock_tokenizer, mock_model):
        """Test genre-based generation with valid genre."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        mock_tokenizer.return_value.eos_token_id = 50256
        
        mock_tokenizer.return_value.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_instance.generate = MagicMock(return_value=mock_output)
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value.decode = MagicMock(return_value="Fantasy story")
        
        generator = StoryGenerator()
        result = generator.generate_with_genre("fantasy", max_length=50)
        
        assert result["genre"] == "fantasy"
        assert "magical kingdom" in result["prompt"].lower()
    
    @patch('src.story_generator.AutoModelForCausalLM')
    @patch('src.story_generator.AutoTokenizer')
    def test_generate_with_genre_invalid(self, mock_tokenizer, mock_model):
        """Test genre-based generation with invalid genre."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        
        generator = StoryGenerator()
        
        with pytest.raises(ValueError, match="Unknown genre"):
            generator.generate_with_genre("invalid_genre", max_length=50)
    
    @patch('src.story_generator.AutoModelForCausalLM')
    @patch('src.story_generator.AutoTokenizer')
    def test_generate_with_genre_custom_prompt(self, mock_tokenizer, mock_model):
        """Test genre-based generation with custom prompt addition."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        mock_tokenizer.return_value.eos_token_id = 50256
        
        mock_tokenizer.return_value.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_instance.generate = MagicMock(return_value=mock_output)
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value.decode = MagicMock(return_value="Story")
        
        generator = StoryGenerator()
        result = generator.generate_with_genre(
            "fantasy",
            custom_prompt="a young wizard",
            max_length=50
        )
        
        assert "young wizard" in result["prompt"]
    
    @patch('src.story_generator.AutoModelForCausalLM')
    @patch('src.story_generator.AutoTokenizer')
    def test_continue_story(self, mock_tokenizer, mock_model):
        """Test story continuation."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        mock_tokenizer.return_value.eos_token_id = 50256
        
        mock_tokenizer.return_value.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_instance.generate = MagicMock(return_value=mock_output)
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value.decode = MagicMock(return_value="Once upon a time continuation")
        
        generator = StoryGenerator()
        result = generator.continue_story("Once upon a time", continuation_length=50)
        
        assert "prompt" in result
        assert result["prompt"] == "Once upon a time"
    
    @patch('src.story_generator.AutoModelForCausalLM')
    @patch('src.story_generator.AutoTokenizer')
    def test_get_token_probabilities(self, mock_tokenizer, mock_model):
        """Test getting token probabilities."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        
        mock_tokenizer.return_value.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]])
        }
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        
        # Mock logits output
        mock_logits = torch.randn(1, 3, 50257)  # batch, seq_len, vocab_size
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_model_instance.return_value = mock_output
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer.return_value.convert_ids_to_tokens = MagicMock(
            return_value=["token1", "token2", "token3"]
        )
        
        generator = StoryGenerator()
        result = generator.get_token_probabilities("test text", top_k=5)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert "position" in result[0]
        assert "current_token" in result[0]
        assert "top_predictions" in result[0]
    
    def test_available_models(self):
        """Test that available models are defined."""
        assert hasattr(StoryGenerator, "AVAILABLE_MODELS")
        assert "gpt2" in StoryGenerator.AVAILABLE_MODELS
        assert "gpt2-medium" in StoryGenerator.AVAILABLE_MODELS
    
    def test_genre_prompts(self):
        """Test that genre prompts are defined."""
        assert hasattr(StoryGenerator, "GENRE_PROMPTS")
        assert "fantasy" in StoryGenerator.GENRE_PROMPTS
        assert "sci-fi" in StoryGenerator.GENRE_PROMPTS
        assert "mystery" in StoryGenerator.GENRE_PROMPTS
    
    @patch('src.story_generator.AutoModelForCausalLM')
    @patch('src.story_generator.AutoTokenizer')
    def test_generate_story_edge_cases(self, mock_tokenizer, mock_model):
        """Test edge cases in story generation."""
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value.pad_token_id = 50256
        mock_tokenizer.return_value.eos_token_id = 50256
        
        mock_tokenizer.return_value.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]])
        }
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model_instance.to = MagicMock(return_value=mock_model_instance)
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2]])
        mock_model_instance.generate = MagicMock(return_value=mock_output)
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value.decode = MagicMock(return_value="Story")
        
        generator = StoryGenerator()
        
        # Test with very short max_length
        result = generator.generate_story("test", max_length=1)
        assert "prompt" in result
        
        # Test with very long max_length
        result = generator.generate_story("test", max_length=10000)
        assert "prompt" in result
        
        # Test with zero temperature
        result = generator.generate_story("test", temperature=0.0, max_length=50)
        assert result["generation_params"]["temperature"] == 0.0

