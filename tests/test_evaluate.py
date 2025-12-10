"""
Unit tests for evaluate.py module.

Tests cover:
- Text loading
- Model loading
- Perplexity computation
- Safety and bias evaluation
- Edge cases and error handling
"""

import pytest
import torch
import math
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.evaluate import (
    load_texts,
    load_model,
    compute_perplexity,
    evaluate_safety_and_bias,
    run_eval
)
from src.ethical_filter import EthicalFilter, BiasDetector


class TestLoadTexts:
    """Test suite for load_texts function."""
    
    def test_load_texts_from_json(self, sample_stories_json):
        """Test loading texts from JSON file."""
        texts = load_texts(dataset_path=sample_stories_json, hf_dataset=None, max_samples=10)
        
        assert isinstance(texts, list)
        assert len(texts) > 0
        assert all(isinstance(text, str) for text in texts)
    
    def test_load_texts_max_samples(self, sample_stories_json):
        """Test loading texts with max_samples limit."""
        texts = load_texts(dataset_path=sample_stories_json, hf_dataset=None, max_samples=1)
        
        assert len(texts) <= 1
    
    def test_load_texts_no_source(self):
        """Test loading texts with no source provided."""
        with pytest.raises(ValueError, match="Provide either"):
            load_texts(dataset_path=None, hf_dataset=None, max_samples=10)
    
    @patch('src.evaluate.StoryDatasetLoader')
    def test_load_texts_from_hf(self, mock_loader_class):
        """Test loading texts from HuggingFace dataset."""
        mock_loader = MagicMock()
        mock_example = MagicMock()
        mock_example.prompt = "Test prompt"
        mock_example.story = "Test story"
        mock_loader.load_dataset.return_value = [mock_example]
        mock_loader_class.return_value = mock_loader
        
        texts = load_texts(dataset_path=None, hf_dataset="test_dataset", max_samples=10)
        
        assert len(texts) > 0
        assert all(isinstance(text, str) for text in texts)


class TestLoadModel:
    """Test suite for load_model function."""
    
    @patch('src.evaluate.AutoTokenizer')
    @patch('src.evaluate.AutoModelForCausalLM')
    def test_load_model_default(self, mock_model_class, mock_tokenizer_class):
        """Test loading model with default device detection."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model_class.from_pretrained.return_value = mock_model
        
        model, tokenizer, device = load_model("gpt2")
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        assert device in ["cuda", "mps", "cpu"]
    
    @patch('src.evaluate.AutoTokenizer')
    @patch('src.evaluate.AutoModelForCausalLM')
    def test_load_model_custom_device(self, mock_model_class, mock_tokenizer_class):
        """Test loading model with custom device."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model_class.from_pretrained.return_value = mock_model
        
        model, tokenizer, device = load_model("gpt2", device="cpu")
        
        assert device == "cpu"
        mock_model.to.assert_called_with("cpu")
    
    @patch('src.evaluate.AutoTokenizer')
    @patch('src.evaluate.AutoModelForCausalLM')
    def test_load_model_pad_token_setup(self, mock_model_class, mock_tokenizer_class):
        """Test that pad token is set correctly."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.config.pad_token_id = None
        mock_model_class.from_pretrained.return_value = mock_model
        
        model, tokenizer, device = load_model("gpt2")
        
        assert tokenizer.pad_token == "<|endoftext|>"


class TestComputePerplexity:
    """Test suite for compute_perplexity function."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.eval = MagicMock()
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]])
        }
        return tokenizer
    
    def test_compute_perplexity_basic(self, mock_model, mock_tokenizer):
        """Test basic perplexity computation."""
        # Mock model output with loss
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(2.0)  # log loss
        mock_model.return_value = mock_output
        
        texts = ["This is a test sentence."]
        device = "cpu"
        
        ppl = compute_perplexity(mock_model, mock_tokenizer, texts, device)
        
        # Perplexity should be exp(loss)
        expected_ppl = math.exp(2.0)
        assert abs(ppl - expected_ppl) < 0.1
    
    def test_compute_perplexity_multiple_texts(self, mock_model, mock_tokenizer):
        """Test perplexity computation with multiple texts."""
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(2.0)
        mock_model.return_value = mock_output
        
        texts = ["Text 1", "Text 2", "Text 3"]
        device = "cpu"
        
        ppl = compute_perplexity(mock_model, mock_tokenizer, texts, device)
        
        assert ppl > 0
        assert not math.isinf(ppl)
    
    def test_compute_perplexity_empty_texts(self, mock_model, mock_tokenizer):
        """Test perplexity computation with empty texts."""
        texts = []
        device = "cpu"
        
        ppl = compute_perplexity(mock_model, mock_tokenizer, texts, device)
        
        assert math.isinf(ppl)
    
    def test_compute_perplexity_truncation(self, mock_model, mock_tokenizer):
        """Test perplexity computation with truncation."""
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(2.0)
        mock_model.return_value = mock_output
        
        # Very long text that should be truncated
        texts = ["word " * 1000]
        device = "cpu"
        
        ppl = compute_perplexity(mock_model, mock_tokenizer, texts, device, max_length=512)
        
        assert ppl > 0
        # Verify truncation was called
        mock_tokenizer.assert_called()


class TestEvaluateSafetyAndBias:
    """Test suite for evaluate_safety_and_bias function."""
    
    @pytest.fixture
    def mock_safety_filter(self):
        """Create a mock safety filter."""
        filter_obj = MagicMock()
        filter_result = MagicMock()
        filter_result.is_safe = True
        filter_obj.filter_content.return_value = filter_result
        return filter_obj
    
    @pytest.fixture
    def mock_bias_detector(self):
        """Create a mock bias detector."""
        detector = MagicMock()
        bias_result = {"has_bias": False}
        detector.analyze_bias.return_value = bias_result
        return detector
    
    def test_evaluate_safety_and_bias_safe(self, mock_safety_filter, mock_bias_detector):
        """Test safety and bias evaluation with safe content."""
        texts = ["Safe text 1", "Safe text 2"]
        
        safe_ratio, bias_ratio = evaluate_safety_and_bias(
            texts, mock_safety_filter, mock_bias_detector
        )
        
        assert safe_ratio == 1.0
        assert bias_ratio == 0.0
    
    def test_evaluate_safety_and_bias_mixed(self, mock_safety_filter, mock_bias_detector):
        """Test safety and bias evaluation with mixed content."""
        # Mock filter to return different results
        filter_results = [
            MagicMock(is_safe=True),
            MagicMock(is_safe=False)
        ]
        mock_safety_filter.filter_content.side_effect = filter_results
        
        bias_results = [
            {"has_bias": False},
            {"has_bias": True}
        ]
        mock_bias_detector.analyze_bias.side_effect = bias_results
        
        texts = ["Safe text", "Unsafe text"]
        
        safe_ratio, bias_ratio = evaluate_safety_and_bias(
            texts, mock_safety_filter, mock_bias_detector
        )
        
        assert safe_ratio == 0.5
        assert bias_ratio == 0.5
    
    def test_evaluate_safety_and_bias_empty(self, mock_safety_filter, mock_bias_detector):
        """Test safety and bias evaluation with empty texts."""
        texts = []
        
        safe_ratio, bias_ratio = evaluate_safety_and_bias(
            texts, mock_safety_filter, mock_bias_detector
        )
        
        # Should handle empty list gracefully
        assert safe_ratio == 0.0
        assert bias_ratio == 0.0


class TestRunEval:
    """Test suite for run_eval function."""
    
    @patch('src.evaluate.load_texts')
    @patch('src.evaluate.load_model')
    @patch('src.evaluate.compute_perplexity')
    @patch('src.evaluate.evaluate_safety_and_bias')
    def test_run_eval_complete(self, mock_safety_bias, mock_ppl, mock_load_model, mock_load_texts):
        """Test complete evaluation run."""
        # Setup mocks
        mock_load_texts.return_value = ["text1", "text2"]
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer, "cpu")
        mock_ppl.return_value = 25.5
        mock_safety_bias.return_value = (0.9, 0.1)
        
        result = run_eval(
            model_name="gpt2",
            dataset_path="test.json",
            hf_dataset=None,
            max_samples=10
        )
        
        assert "perplexity" in result
        assert "safe_ratio" in result
        assert "bias_ratio" in result
        assert "num_samples" in result
        assert "model" in result
        assert result["perplexity"] == 25.5
        assert result["safe_ratio"] == 0.9
        assert result["bias_ratio"] == 0.1
        assert result["num_samples"] == 2
        assert result["model"] == "gpt2"
    
    @patch('src.evaluate.load_texts')
    @patch('src.evaluate.load_model')
    @patch('src.evaluate.compute_perplexity')
    @patch('src.evaluate.evaluate_safety_and_bias')
    def test_run_eval_with_hf_dataset(self, mock_safety_bias, mock_ppl, mock_load_model, mock_load_texts):
        """Test evaluation with HuggingFace dataset."""
        mock_load_texts.return_value = ["text1"]
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer, "cpu")
        mock_ppl.return_value = 30.0
        mock_safety_bias.return_value = (1.0, 0.0)
        
        result = run_eval(
            model_name="gpt2",
            dataset_path=None,
            hf_dataset="test_dataset",
            max_samples=5
        )
        
        assert result["num_samples"] == 1
        mock_load_texts.assert_called_with(None, "test_dataset", 5)


class TestEvaluateEdgeCases:
    """Test edge cases for evaluate module."""
    
    def test_load_texts_invalid_file(self):
        """Test loading texts from invalid file."""
        with pytest.raises((FileNotFoundError, ValueError)):
            load_texts(dataset_path="nonexistent.json", hf_dataset=None, max_samples=10)
    
    @patch('src.evaluate.AutoTokenizer')
    @patch('src.evaluate.AutoModelForCausalLM')
    def test_load_model_invalid_name(self, mock_model_class, mock_tokenizer_class):
        """Test loading invalid model name."""
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Model not found")
        
        with pytest.raises(Exception):
            load_model("invalid_model_name")
    
    def test_compute_perplexity_zero_tokens(self):
        """Test perplexity computation with zero tokens."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[]])
        }
        
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(0.0)
        mock_model.return_value = mock_output
        
        texts = [""]
        device = "cpu"
        
        ppl = compute_perplexity(mock_model, mock_tokenizer, texts, device)
        
        # Should handle zero tokens gracefully
        assert ppl >= 0 or math.isinf(ppl)

