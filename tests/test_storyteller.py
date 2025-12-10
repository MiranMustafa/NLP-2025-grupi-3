"""
Integration tests for storyteller.py module.

Tests cover:
- Full integration of all components
- Story generation with ethical filtering
- Bias detection integration
- Explainability integration
- Edge cases and error handling
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.storyteller import EthicalStoryTeller, StoryResult
from src.ethical_filter import ContentRating


class TestEthicalStoryTeller:
    """Test suite for EthicalStoryTeller integration class."""
    
    @patch('src.storyteller.StoryGenerator')
    @patch('src.storyteller.EthicalFilter')
    @patch('src.storyteller.BiasDetector')
    @patch('src.storyteller.AttentionVisualizer')
    @patch('src.storyteller.TokenImportanceAnalyzer')
    @patch('src.storyteller.GenerationExplainer')
    def test_initialization_default(self, mock_gen_explainer, mock_token_analyzer,
                                     mock_attention_viz, mock_bias, mock_filter, mock_generator):
        """Test default initialization."""
        mock_gen = MagicMock()
        mock_gen.device = "cpu"
        mock_gen.model_name = "gpt2"
        mock_generator.return_value = mock_gen
        
        storyteller = EthicalStoryTeller()
        
        assert storyteller.generator == mock_gen
        assert storyteller.enable_ethical_filter == True
        assert storyteller.enable_bias_detection == True
    
    @patch('src.storyteller.StoryGenerator')
    @patch('src.storyteller.EthicalFilter')
    @patch('src.storyteller.BiasDetector')
    @patch('src.storyteller.AttentionVisualizer')
    @patch('src.storyteller.TokenImportanceAnalyzer')
    @patch('src.storyteller.GenerationExplainer')
    def test_initialization_disable_filters(self, mock_gen_explainer, mock_token_analyzer,
                                            mock_attention_viz, mock_bias, mock_filter, mock_generator):
        """Test initialization with filters disabled."""
        mock_gen = MagicMock()
        mock_gen.device = "cpu"
        mock_gen.model_name = "gpt2"
        mock_generator.return_value = mock_gen
        
        storyteller = EthicalStoryTeller(
            enable_ethical_filter=False,
            enable_bias_detection=False
        )
        
        assert storyteller.enable_ethical_filter == False
        assert storyteller.enable_bias_detection == False
        assert storyteller.ethical_filter is None
        assert storyteller.bias_detector is None
    
    @patch('src.storyteller.StoryGenerator')
    @patch('src.storyteller.EthicalFilter')
    @patch('src.storyteller.BiasDetector')
    @patch('src.storyteller.AttentionVisualizer')
    @patch('src.storyteller.TokenImportanceAnalyzer')
    @patch('src.storyteller.GenerationExplainer')
    def test_generate_story_basic(self, mock_gen_explainer, mock_token_analyzer,
                                   mock_attention_viz, mock_bias, mock_filter, mock_generator):
        """Test basic story generation."""
        # Setup mocks
        mock_gen = MagicMock()
        mock_gen.device = "cpu"
        mock_gen.model_name = "gpt2"
        mock_gen.GENRE_PROMPTS = {"fantasy": "In a magical kingdom, "}
        mock_gen.generate_story.return_value = {
            "full_texts": ["In a magical kingdom, a brave knight saved the day."],
            "stories": ["a brave knight saved the day."]
        }
        mock_generator.return_value = mock_gen
        
        mock_filter_instance = MagicMock()
        from src.ethical_filter import FilterResult, ContentRating
        mock_filter_instance.filter_content.return_value = FilterResult(
            is_safe=True,
            rating=ContentRating.SAFE,
            toxicity_scores={},
            flagged_content=[],
            suggestions=[],
            confidence=0.9
        )
        mock_filter.return_value = mock_filter_instance
        
        mock_bias_instance = MagicMock()
        mock_bias_instance.analyze_bias.return_value = {"has_bias": False}
        mock_bias.return_value = mock_bias_instance
        
        storyteller = EthicalStoryTeller()
        result = storyteller.generate_story("a brave knight", genre="fantasy", max_length=50)
        
        assert isinstance(result, StoryResult)
        assert result.is_safe == True
        assert result.genre == "fantasy"
        assert len(result.story) > 0
    
    @patch('src.storyteller.StoryGenerator')
    @patch('src.storyteller.EthicalFilter')
    @patch('src.storyteller.BiasDetector')
    @patch('src.storyteller.AttentionVisualizer')
    @patch('src.storyteller.TokenImportanceAnalyzer')
    @patch('src.storyteller.GenerationExplainer')
    def test_generate_story_invalid_genre(self, mock_gen_explainer, mock_token_analyzer,
                                          mock_attention_viz, mock_bias, mock_filter, mock_generator):
        """Test story generation with invalid genre."""
        mock_gen = MagicMock()
        mock_gen.device = "cpu"
        mock_gen.GENRE_PROMPTS = {"fantasy": "In a magical kingdom, "}
        mock_generator.return_value = mock_gen
        
        storyteller = EthicalStoryTeller()
        
        with pytest.raises(ValueError, match="Unknown genre"):
            storyteller.generate_story("test", genre="invalid_genre")
    
    @patch('src.storyteller.StoryGenerator')
    @patch('src.storyteller.EthicalFilter')
    @patch('src.storyteller.BiasDetector')
    @patch('src.storyteller.AttentionVisualizer')
    @patch('src.storyteller.TokenImportanceAnalyzer')
    @patch('src.storyteller.GenerationExplainer')
    def test_generate_story_with_retries(self, mock_gen_explainer, mock_token_analyzer,
                                         mock_attention_viz, mock_bias, mock_filter, mock_generator):
        """Test story generation with retries for safe content."""
        mock_gen = MagicMock()
        mock_gen.device = "cpu"
        mock_gen.model_name = "gpt2"
        mock_gen.GENRE_PROMPTS = {}
        mock_gen.generate_story.return_value = {
            "full_texts": ["Generated story"],
            "stories": ["Generated story"]
        }
        mock_generator.return_value = mock_gen
        
        mock_filter_instance = MagicMock()
        from src.ethical_filter import FilterResult, ContentRating
        
        # First attempt unsafe, second safe
        unsafe_result = FilterResult(
            is_safe=False,
            rating=ContentRating.MODERATE,
            toxicity_scores={},
            flagged_content=[],
            suggestions=[],
            confidence=0.5
        )
        safe_result = FilterResult(
            is_safe=True,
            rating=ContentRating.SAFE,
            toxicity_scores={},
            flagged_content=[],
            suggestions=[],
            confidence=0.9
        )
        mock_filter_instance.filter_content.side_effect = [unsafe_result, safe_result]
        mock_filter.return_value = mock_filter_instance
        
        mock_bias_instance = MagicMock()
        mock_bias_instance.analyze_bias.return_value = {"has_bias": False}
        mock_bias.return_value = mock_bias_instance
        
        storyteller = EthicalStoryTeller()
        result = storyteller.generate_story(
            "test",
            max_length=50,
            ensure_safe=True,
            max_retries=3
        )
        
        assert result.is_safe == True
        assert mock_gen.generate_story.call_count == 2
    
    @patch('src.storyteller.StoryGenerator')
    @patch('src.storyteller.EthicalFilter')
    @patch('src.storyteller.BiasDetector')
    @patch('src.storyteller.AttentionVisualizer')
    @patch('src.storyteller.TokenImportanceAnalyzer')
    @patch('src.storyteller.GenerationExplainer')
    def test_continue_story(self, mock_gen_explainer, mock_token_analyzer,
                            mock_attention_viz, mock_bias, mock_filter, mock_generator):
        """Test story continuation."""
        mock_gen = MagicMock()
        mock_gen.device = "cpu"
        mock_gen.model_name = "gpt2"
        mock_gen.GENRE_PROMPTS = {}
        mock_gen.generate_story.return_value = {
            "full_texts": ["Once upon a time continuation"],
            "stories": ["continuation"]
        }
        mock_generator.return_value = mock_gen
        
        mock_filter_instance = MagicMock()
        from src.ethical_filter import FilterResult, ContentRating
        mock_filter_instance.filter_content.return_value = FilterResult(
            is_safe=True,
            rating=ContentRating.SAFE,
            toxicity_scores={},
            flagged_content=[],
            suggestions=[],
            confidence=0.9
        )
        mock_filter.return_value = mock_filter_instance
        
        mock_bias_instance = MagicMock()
        mock_bias_instance.analyze_bias.return_value = {"has_bias": False}
        mock_bias.return_value = mock_bias_instance
        
        storyteller = EthicalStoryTeller()
        result = storyteller.continue_story("Once upon a time", continuation_length=50)
        
        assert isinstance(result, StoryResult)
        assert len(result.story) > 0
    
    @patch('src.storyteller.StoryGenerator')
    @patch('src.storyteller.EthicalFilter')
    @patch('src.storyteller.BiasDetector')
    @patch('src.storyteller.AttentionVisualizer')
    @patch('src.storyteller.TokenImportanceAnalyzer')
    @patch('src.storyteller.GenerationExplainer')
    def test_explain_generation(self, mock_gen_explainer, mock_token_analyzer,
                                mock_attention_viz, mock_bias, mock_filter, mock_generator):
        """Test generation explanation."""
        mock_gen = MagicMock()
        mock_gen.device = "cpu"
        mock_generator.return_value = mock_gen
        
        mock_explainer = MagicMock()
        mock_explainer.explain_next_token.return_value = {
            "predicted_token": "the",
            "confidence": 0.8
        }
        mock_explainer.trace_generation.return_value = [
            {"step": 1, "current_text": "test", "generated_token": "the"}
        ]
        mock_gen_explainer.return_value = mock_explainer
        
        storyteller = EthicalStoryTeller()
        result = storyteller.explain_generation("test text")
        
        assert "next_token" in result
        assert "generation_trace" in result
        assert "continued_text" in result
    
    @patch('src.storyteller.StoryGenerator')
    @patch('src.storyteller.EthicalFilter')
    @patch('src.storyteller.BiasDetector')
    @patch('src.storyteller.AttentionVisualizer')
    @patch('src.storyteller.TokenImportanceAnalyzer')
    @patch('src.storyteller.GenerationExplainer')
    def test_get_available_genres(self, mock_gen_explainer, mock_token_analyzer,
                                  mock_attention_viz, mock_bias, mock_filter, mock_generator):
        """Test getting available genres."""
        mock_gen = MagicMock()
        mock_gen.GENRE_PROMPTS = {"fantasy": "test", "sci-fi": "test"}
        mock_generator.return_value = mock_gen
        
        storyteller = EthicalStoryTeller()
        genres = storyteller.get_available_genres()
        
        assert isinstance(genres, list)
        assert "fantasy" in genres
        assert "sci-fi" in genres
    
    @patch('src.storyteller.StoryGenerator')
    @patch('src.storyteller.EthicalFilter')
    @patch('src.storyteller.BiasDetector')
    @patch('src.storyteller.AttentionVisualizer')
    @patch('src.storyteller.TokenImportanceAnalyzer')
    @patch('src.storyteller.GenerationExplainer')
    def test_get_genre_prompt(self, mock_gen_explainer, mock_token_analyzer,
                              mock_attention_viz, mock_bias, mock_filter, mock_generator):
        """Test getting genre prompt."""
        mock_gen = MagicMock()
        mock_gen.GENRE_PROMPTS = {"fantasy": "In a magical kingdom, "}
        mock_generator.return_value = mock_gen
        
        storyteller = EthicalStoryTeller()
        prompt = storyteller.get_genre_prompt("fantasy")
        
        assert prompt == "In a magical kingdom, "
    
    @patch('src.storyteller.StoryGenerator')
    @patch('src.storyteller.EthicalFilter')
    @patch('src.storyteller.BiasDetector')
    @patch('src.storyteller.AttentionVisualizer')
    @patch('src.storyteller.TokenImportanceAnalyzer')
    @patch('src.storyteller.GenerationExplainer')
    def test_get_genre_prompt_invalid(self, mock_gen_explainer, mock_token_analyzer,
                                      mock_attention_viz, mock_bias, mock_filter, mock_generator):
        """Test getting prompt for invalid genre."""
        mock_gen = MagicMock()
        mock_gen.GENRE_PROMPTS = {"fantasy": "test"}
        mock_generator.return_value = mock_gen
        
        storyteller = EthicalStoryTeller()
        
        with pytest.raises(ValueError, match="Unknown genre"):
            storyteller.get_genre_prompt("invalid")
    
    @patch('src.storyteller.StoryGenerator')
    @patch('src.storyteller.EthicalFilter')
    @patch('src.storyteller.BiasDetector')
    @patch('src.storyteller.AttentionVisualizer')
    @patch('src.storyteller.TokenImportanceAnalyzer')
    @patch('src.storyteller.GenerationExplainer')
    def test_generate_story_with_bias(self, mock_gen_explainer, mock_token_analyzer,
                                      mock_attention_viz, mock_bias, mock_filter, mock_generator):
        """Test story generation with bias detection."""
        mock_gen = MagicMock()
        mock_gen.device = "cpu"
        mock_gen.model_name = "gpt2"
        mock_gen.GENRE_PROMPTS = {}
        mock_gen.generate_story.return_value = {
            "full_texts": ["Generated story"],
            "stories": ["Generated story"]
        }
        mock_generator.return_value = mock_gen
        
        mock_filter_instance = MagicMock()
        from src.ethical_filter import FilterResult, ContentRating
        mock_filter_instance.filter_content.return_value = FilterResult(
            is_safe=True,
            rating=ContentRating.SAFE,
            toxicity_scores={},
            flagged_content=[],
            suggestions=[],
            confidence=0.9
        )
        mock_filter.return_value = mock_filter_instance
        
        mock_bias_instance = MagicMock()
        mock_bias_instance.analyze_bias.return_value = {
            "has_bias": True,
            "recommendations": ["Consider balanced representation"]
        }
        mock_bias.return_value = mock_bias_instance
        
        storyteller = EthicalStoryTeller()
        result = storyteller.generate_story("test", max_length=50)
        
        assert result.is_safe == True
        assert len(result.warnings) > 0
        assert "bias" in result.bias_analysis or result.bias_analysis.get("has_bias", False)


class TestStoryResult:
    """Test suite for StoryResult dataclass."""
    
    def test_story_result_creation(self):
        """Test creating a StoryResult."""
        from src.ethical_filter import ContentRating
        
        result = StoryResult(
            story="Generated story",
            prompt="Test prompt",
            genre="fantasy",
            is_safe=True,
            content_rating="safe",
            ethical_analysis={},
            bias_analysis={},
            generation_params={},
            warnings=[]
        )
        
        assert result.story == "Generated story"
        assert result.prompt == "Test prompt"
        assert result.genre == "fantasy"
        assert result.is_safe == True
        assert result.content_rating == "safe"
    
    def test_story_result_with_warnings(self):
        """Test StoryResult with warnings."""
        from src.ethical_filter import ContentRating
        
        result = StoryResult(
            story="Story",
            prompt="Prompt",
            genre=None,
            is_safe=True,
            content_rating="mild",
            ethical_analysis={},
            bias_analysis={},
            generation_params={},
            warnings=["Warning 1", "Warning 2"]
        )
        
        assert len(result.warnings) == 2
        assert "Warning 1" in result.warnings


class TestIntegrationEdgeCases:
    """Test edge cases for integration."""
    
    @patch('src.storyteller.StoryGenerator')
    @patch('src.storyteller.EthicalFilter')
    @patch('src.storyteller.BiasDetector')
    @patch('src.storyteller.AttentionVisualizer')
    @patch('src.storyteller.TokenImportanceAnalyzer')
    @patch('src.storyteller.GenerationExplainer')
    def test_generate_story_empty_prompt(self, mock_gen_explainer, mock_token_analyzer,
                                         mock_attention_viz, mock_bias, mock_filter, mock_generator):
        """Test story generation with empty prompt."""
        mock_gen = MagicMock()
        mock_gen.device = "cpu"
        mock_gen.model_name = "gpt2"
        mock_gen.GENRE_PROMPTS = {}
        mock_gen.generate_story.return_value = {
            "full_texts": ["Generated"],
            "stories": ["Generated"]
        }
        mock_generator.return_value = mock_gen
        
        mock_filter_instance = MagicMock()
        from src.ethical_filter import FilterResult, ContentRating
        mock_filter_instance.filter_content.return_value = FilterResult(
            is_safe=True,
            rating=ContentRating.SAFE,
            toxicity_scores={},
            flagged_content=[],
            suggestions=[],
            confidence=0.9
        )
        mock_filter.return_value = mock_filter_instance
        
        mock_bias_instance = MagicMock()
        mock_bias_instance.analyze_bias.return_value = {"has_bias": False}
        mock_bias.return_value = mock_bias_instance
        
        storyteller = EthicalStoryTeller()
        result = storyteller.generate_story("", max_length=50)
        
        assert isinstance(result, StoryResult)
        assert result.prompt == ""
    
    @patch('src.storyteller.StoryGenerator')
    @patch('src.storyteller.EthicalFilter')
    @patch('src.storyteller.BiasDetector')
    @patch('src.storyteller.AttentionVisualizer')
    @patch('src.storyteller.TokenImportanceAnalyzer')
    @patch('src.storyteller.GenerationExplainer')
    def test_generate_story_no_filter(self, mock_gen_explainer, mock_token_analyzer,
                                      mock_attention_viz, mock_bias, mock_filter, mock_generator):
        """Test story generation without ethical filter."""
        mock_gen = MagicMock()
        mock_gen.device = "cpu"
        mock_gen.model_name = "gpt2"
        mock_gen.GENRE_PROMPTS = {}
        mock_gen.generate_story.return_value = {
            "full_texts": ["Generated"],
            "stories": ["Generated"]
        }
        mock_generator.return_value = mock_gen
        
        storyteller = EthicalStoryTeller(enable_ethical_filter=False)
        result = storyteller.generate_story("test", max_length=50)
        
        assert isinstance(result, StoryResult)
        assert result.is_safe == True  # Default when no filter

