"""
Unit tests for explainability.py module.

Tests cover:
- Attention visualization
- Token importance analysis
- Generation explanation
- Edge cases and error handling
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from src.explainability import (
    AttentionVisualizer,
    TokenImportanceAnalyzer,
    GenerationExplainer
)


class TestAttentionVisualizer:
    """Test suite for AttentionVisualizer class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.return_value = MagicMock(attentions=[
            torch.randn(1, 12, 10, 10)  # batch, heads, seq, seq
        ] * 12)  # 12 layers
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
        tokenizer.convert_ids_to_tokens = MagicMock(
            return_value=["token1", "token2", "token3", "token4", "token5"]
        )
        return tokenizer
    
    def test_initialization(self, mock_model, mock_tokenizer):
        """Test AttentionVisualizer initialization."""
        visualizer = AttentionVisualizer(mock_model, mock_tokenizer, "cpu")
        
        assert visualizer.model == mock_model
        assert visualizer.tokenizer == mock_tokenizer
        assert visualizer.device == "cpu"
    
    def test_get_attention_weights(self, mock_model, mock_tokenizer):
        """Test getting attention weights."""
        visualizer = AttentionVisualizer(mock_model, mock_tokenizer, "cpu")
        
        # Mock model output
        mock_output = MagicMock()
        mock_output.attentions = [
            torch.randn(1, 12, 10, 10) for _ in range(12)
        ]
        mock_model.return_value = mock_output
        
        result = visualizer.get_attention_weights("test text")
        
        assert "tokens" in result
        assert "attention_weights" in result
        assert "num_layers" in result
        assert "num_heads" in result
        assert "sequence_length" in result
    
    def test_get_attention_weights_specific_layer(self, mock_model, mock_tokenizer):
        """Test getting attention weights for specific layer."""
        visualizer = AttentionVisualizer(mock_model, mock_tokenizer, "cpu")
        
        mock_output = MagicMock()
        mock_output.attentions = [
            torch.randn(1, 12, 10, 10) for _ in range(12)
        ]
        mock_model.return_value = mock_output
        
        result = visualizer.get_attention_weights("test text", layer=5)
        
        assert result["attention_weights"].shape[0] == 1  # Single layer
    
    @patch('matplotlib.pyplot.subplots')
    @patch('seaborn.heatmap')
    def test_plot_attention_heatmap(self, mock_heatmap, mock_subplots, mock_model, mock_tokenizer):
        """Test plotting attention heatmap."""
        visualizer = AttentionVisualizer(mock_model, mock_tokenizer, "cpu")
        
        mock_output = MagicMock()
        mock_output.attentions = [
            torch.randn(1, 12, 10, 10) for _ in range(12)
        ]
        mock_model.return_value = mock_output
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        fig = visualizer.plot_attention_heatmap("test text")
        
        assert fig == mock_fig
        mock_heatmap.assert_called_once()
    
    @patch('matplotlib.pyplot.subplots')
    @patch('seaborn.heatmap')
    def test_plot_attention_flow(self, mock_heatmap, mock_subplots, mock_model, mock_tokenizer):
        """Test plotting attention flow."""
        visualizer = AttentionVisualizer(mock_model, mock_tokenizer, "cpu")
        
        mock_output = MagicMock()
        mock_output.attentions = [
            torch.randn(1, 12, 10, 10) for _ in range(12)
        ]
        mock_model.return_value = mock_output
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        fig = visualizer.plot_attention_flow("test text")
        
        assert fig == mock_fig
        mock_heatmap.assert_called_once()


class TestTokenImportanceAnalyzer:
    """Test suite for TokenImportanceAnalyzer class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.transformer = MagicMock()
        model.transformer.wte = MagicMock(return_value=torch.randn(1, 5, 768))
        model.return_value = MagicMock(logits=torch.randn(1, 5, 50257))
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
        tokenizer.convert_ids_to_tokens = MagicMock(
            return_value=["token1", "token2", "token3", "token4", "token5"]
        )
        tokenizer.pad_token_id = 0
        return tokenizer
    
    def test_initialization(self, mock_model, mock_tokenizer):
        """Test TokenImportanceAnalyzer initialization."""
        analyzer = TokenImportanceAnalyzer(mock_model, mock_tokenizer, "cpu")
        
        assert analyzer.model == mock_model
        assert analyzer.tokenizer == mock_tokenizer
        assert analyzer.device == "cpu"
    
    def test_compute_leave_one_out_importance(self, mock_model, mock_tokenizer):
        """Test computing leave-one-out importance."""
        analyzer = TokenImportanceAnalyzer(mock_model, mock_tokenizer, "cpu")
        
        # Mock model outputs
        baseline_logits = torch.randn(50257)
        masked_logits = torch.randn(50257)
        
        def mock_forward(*args, **kwargs):
            output = MagicMock()
            if "input_ids" in kwargs:
                output.logits = torch.stack([masked_logits])
            else:
                output.logits = torch.stack([baseline_logits])
            return output
        
        mock_model.side_effect = mock_forward
        
        result = analyzer.compute_leave_one_out_importance("test text")
        
        assert "tokens" in result
        assert "importance_scores" in result
        assert "method" in result
        assert "most_important" in result
        assert result["method"] == "leave_one_out"
        assert len(result["importance_scores"]) == len(result["tokens"])
    
    @patch('matplotlib.pyplot.subplots')
    def test_plot_token_importance(self, mock_subplots, mock_model, mock_tokenizer):
        """Test plotting token importance."""
        analyzer = TokenImportanceAnalyzer(mock_model, mock_tokenizer, "cpu")
        
        importance_data = {
            "tokens": ["token1", "token2", "token3"],
            "importance_scores": [0.5, 0.3, 0.2]
        }
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        fig = analyzer.plot_token_importance(importance_data)
        
        assert fig == mock_fig


class TestGenerationExplainer:
    """Test suite for GenerationExplainer class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.return_value = MagicMock(
            logits=torch.randn(1, 5, 50257)
        )
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
        tokenizer.convert_ids_to_tokens = MagicMock(
            return_value=["token1", "token2", "token3", "token4", "token5"]
        )
        tokenizer.decode = MagicMock(return_value="decoded")
        return tokenizer
    
    def test_initialization(self, mock_model, mock_tokenizer):
        """Test GenerationExplainer initialization."""
        explainer = GenerationExplainer(mock_model, mock_tokenizer, "cpu")
        
        assert explainer.model == mock_model
        assert explainer.tokenizer == mock_tokenizer
        assert explainer.device == "cpu"
    
    def test_explain_next_token(self, mock_model, mock_tokenizer):
        """Test explaining next token prediction."""
        explainer = GenerationExplainer(mock_model, mock_tokenizer, "cpu")
        
        result = explainer.explain_next_token("test text", top_k=10)
        
        assert "input_text" in result
        assert "top_candidates" in result
        assert "predicted_token" in result
        assert "prediction_confidence" in result
        assert "entropy" in result
        assert "uncertainty_level" in result
        assert len(result["top_candidates"]) == 10
    
    def test_explain_next_token_top_k(self, mock_model, mock_tokenizer):
        """Test explaining next token with different top_k."""
        explainer = GenerationExplainer(mock_model, mock_tokenizer, "cpu")
        
        result = explainer.explain_next_token("test text", top_k=5)
        
        assert len(result["top_candidates"]) == 5
    
    def test_get_uncertainty_level(self, mock_model, mock_tokenizer):
        """Test uncertainty level categorization."""
        explainer = GenerationExplainer(mock_model, mock_tokenizer, "cpu")
        
        assert explainer._get_uncertainty_level(0.2) == "low"
        assert explainer._get_uncertainty_level(0.5) == "medium"
        assert explainer._get_uncertainty_level(0.8) == "high"
    
    def test_trace_generation(self, mock_model, mock_tokenizer):
        """Test tracing generation process."""
        explainer = GenerationExplainer(mock_model, mock_tokenizer, "cpu")
        
        # Mock tokenizer to return different tokens
        token_ids = [10, 20, 30, 40, 50]
        mock_tokenizer.decode = MagicMock(side_effect=lambda x: f"token{token_ids.pop(0) if token_ids else 0}")
        
        # Mock multinomial sampling
        with patch('torch.multinomial') as mock_multinomial:
            mock_multinomial.return_value = torch.tensor([10])
            
            trace = explainer.trace_generation("test", num_tokens=3, temperature=1.0)
            
            assert len(trace) == 3
            assert all("step" in t for t in trace)
            assert all("current_text" in t for t in trace)
            assert all("generated_token" in t for t in trace)
            assert all("token_probability" in t for t in trace)
            assert all("uncertainty" in t for t in trace)
    
    @patch('matplotlib.pyplot.subplots')
    def test_visualize_generation_trace(self, mock_subplots, mock_model, mock_tokenizer):
        """Test visualizing generation trace."""
        explainer = GenerationExplainer(mock_model, mock_tokenizer, "cpu")
        
        trace = [
            {
                "step": 1,
                "token_probability": 0.8,
                "generated_token": "token1",
                "uncertainty": "low"
            },
            {
                "step": 2,
                "token_probability": 0.6,
                "generated_token": "token2",
                "uncertainty": "medium"
            }
        ]
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        fig = explainer.visualize_generation_trace(trace)
        
        assert fig == mock_fig


class TestExplainabilityEdgeCases:
    """Test edge cases for explainability module."""
    
    def test_attention_visualizer_empty_text(self):
        """Test attention visualization with empty text."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[]])}
        mock_tokenizer.convert_ids_to_tokens = MagicMock(return_value=[])
        
        visualizer = AttentionVisualizer(mock_model, mock_tokenizer, "cpu")
        
        mock_output = MagicMock()
        mock_output.attentions = [torch.randn(1, 12, 0, 0)]
        mock_model.return_value = mock_output
        
        result = visualizer.get_attention_weights("")
        
        assert result["sequence_length"] == 0
    
    def test_token_importance_empty_text(self):
        """Test token importance with empty text."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[]])}
        mock_tokenizer.convert_ids_to_tokens = MagicMock(return_value=[])
        mock_tokenizer.pad_token_id = 0
        
        analyzer = TokenImportanceAnalyzer(mock_model, mock_tokenizer, "cpu")
        
        mock_model.return_value = MagicMock(logits=torch.randn(1, 0, 50257))
        
        result = analyzer.compute_leave_one_out_importance("")
        
        assert len(result["tokens"]) == 0
        assert len(result["importance_scores"]) == 0
    
    def test_generation_explainer_empty_text(self):
        """Test generation explanation with empty text."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[]])}
        mock_tokenizer.decode = MagicMock(return_value="")
        
        explainer = GenerationExplainer(mock_model, mock_tokenizer, "cpu")
        
        mock_model.return_value = MagicMock(logits=torch.randn(1, 0, 50257))
        
        result = explainer.explain_next_token("")
        
        assert "input_text" in result
        assert result["input_text"] == ""

