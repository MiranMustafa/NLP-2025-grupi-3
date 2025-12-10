"""
Unit tests for ethical_filter.py module.

Tests cover:
- Content filtering
- Toxicity detection
- Bias detection
- Content rating
- Edge cases and error handling
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.ethical_filter import (
    EthicalFilter,
    BiasDetector,
    ContentRating,
    FilterResult
)


class TestEthicalFilter:
    """Test suite for EthicalFilter class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        filter_module = EthicalFilter(use_detoxify=False)
        
        assert filter_module.use_detoxify == False
        assert filter_module.strict_mode == False
        assert len(filter_module.blocked_patterns) > 0
        assert len(filter_module.compiled_blocked) > 0
    
    def test_initialization_with_detoxify(self):
        """Test initialization with Detoxify enabled."""
        with patch('src.ethical_filter.DETOXIFY_AVAILABLE', True), \
             patch('src.ethical_filter.Detoxify') as mock_detoxify:
            
            mock_detoxify.return_value = MagicMock()
            filter_module = EthicalFilter(use_detoxify=True)
            
            assert filter_module.use_detoxify == True
            assert filter_module.toxicity_model is not None
    
    def test_initialization_strict_mode(self):
        """Test initialization with strict mode."""
        filter_module = EthicalFilter(use_detoxify=False, strict_mode=True)
        
        assert filter_module.strict_mode == True
        # Thresholds should be lower in strict mode
        assert filter_module.thresholds["toxicity"] < 0.5
    
    def test_initialization_custom_patterns(self):
        """Test initialization with custom blocked patterns."""
        custom_patterns = [r'\bcustom\s+pattern\b']
        filter_module = EthicalFilter(
            use_detoxify=False,
            custom_blocked_patterns=custom_patterns
        )
        
        assert len(filter_module.blocked_patterns) > len(EthicalFilter.DEFAULT_BLOCKED_PATTERNS)
        assert any('custom' in p for p in filter_module.blocked_patterns)
    
    def test_check_blocked_patterns_no_match(self):
        """Test checking blocked patterns with safe text."""
        filter_module = EthicalFilter(use_detoxify=False)
        text = "This is a safe and positive story about friendship."
        
        matches = filter_module.check_blocked_patterns(text)
        assert len(matches) == 0
    
    def test_check_blocked_patterns_with_match(self):
        """Test checking blocked patterns with problematic text."""
        filter_module = EthicalFilter(use_detoxify=False)
        text = "The character wanted to kill them all."
        
        matches = filter_module.check_blocked_patterns(text)
        assert len(matches) > 0
    
    def test_check_sensitive_topics(self):
        """Test checking sensitive topics."""
        filter_module = EthicalFilter(use_detoxify=False)
        text = "The story contains violence and blood."
        
        topics = filter_module.check_sensitive_topics(text)
        assert "violence" in topics
    
    def test_count_positive_elements(self):
        """Test counting positive elements."""
        filter_module = EthicalFilter(use_detoxify=False)
        text = "The story shows friendship, love, and hope."
        
        count = filter_module.count_positive_elements(text)
        assert count >= 3
    
    def test_filter_content_safe(self):
        """Test filtering safe content."""
        filter_module = EthicalFilter(use_detoxify=False)
        text = "The brave knight helped everyone with kindness and courage."
        
        result = filter_module.filter_content(text)
        
        assert isinstance(result, FilterResult)
        assert result.is_safe == True
        assert result.rating in [ContentRating.SAFE, ContentRating.MILD]
        assert result.confidence > 0
    
    def test_filter_content_blocked(self):
        """Test filtering blocked content."""
        filter_module = EthicalFilter(use_detoxify=False)
        text = "The character wanted to kill everyone."
        
        result = filter_module.filter_content(text)
        
        assert result.is_safe == False
        assert result.rating == ContentRating.BLOCKED
    
    def test_filter_content_with_toxicity(self):
        """Test filtering with toxicity detection."""
        with patch('src.ethical_filter.DETOXIFY_AVAILABLE', True), \
             patch('src.ethical_filter.Detoxify') as mock_detoxify:
            
            mock_model = MagicMock()
            mock_model.predict = MagicMock(return_value={
                "toxicity": 0.8,
                "severe_toxicity": 0.3,
                "obscene": 0.2,
                "threat": 0.1,
                "insult": 0.2,
                "identity_attack": 0.1,
                "sexual_explicit": 0.1
            })
            mock_detoxify.return_value = mock_model
            
            filter_module = EthicalFilter(use_detoxify=True)
            text = "Some text to test"
            
            result = filter_module.filter_content(text)
            
            assert isinstance(result, FilterResult)
            assert "toxicity" in result.toxicity_scores
    
    def test_determine_rating_safe(self):
        """Test rating determination for safe content."""
        filter_module = EthicalFilter(use_detoxify=False)
        
        rating, confidence = filter_module._determine_rating(
            toxicity_scores={},
            blocked_matches=[],
            sensitive_topics={},
            positive_count=5
        )
        
        assert rating == ContentRating.SAFE
        assert confidence > 0.8
    
    def test_determine_rating_blocked(self):
        """Test rating determination for blocked content."""
        filter_module = EthicalFilter(use_detoxify=False)
        
        rating, confidence = filter_module._determine_rating(
            toxicity_scores={},
            blocked_matches=["blocked pattern"],
            sensitive_topics={},
            positive_count=0
        )
        
        assert rating == ContentRating.BLOCKED
        assert confidence > 0.9
    
    def test_determine_rating_high_toxicity(self):
        """Test rating determination with high toxicity."""
        filter_module = EthicalFilter(use_detoxify=False)
        
        rating, confidence = filter_module._determine_rating(
            toxicity_scores={"toxicity": 0.9, "severe_toxicity": 0.6},
            blocked_matches=[],
            sensitive_topics={},
            positive_count=0
        )
        
        assert rating == ContentRating.BLOCKED
    
    def test_determine_rating_sensitive_topics(self):
        """Test rating determination with sensitive topics."""
        filter_module = EthicalFilter(use_detoxify=False)
        
        rating, confidence = filter_module._determine_rating(
            toxicity_scores={},
            blocked_matches=[],
            sensitive_topics={"self_harm": ["suicide"]},
            positive_count=0
        )
        
        assert rating == ContentRating.RESTRICTED
    
    def test_sanitize_text(self):
        """Test text sanitization."""
        filter_module = EthicalFilter(use_detoxify=False)
        text = "The character wanted to kill them."
        
        sanitized = filter_module.sanitize_text(text)
        
        assert "[FILTERED]" in sanitized or sanitized != text
    
    def test_get_content_summary(self):
        """Test getting content summary."""
        filter_module = EthicalFilter(use_detoxify=False)
        text = "A positive story about friendship and love."
        
        summary = filter_module.get_content_summary(text)
        
        assert "word_count" in summary
        assert "character_count" in summary
        assert "is_safe" in summary
        assert "rating" in summary
        assert summary["is_safe"] == True
    
    def test_filter_content_empty_text(self):
        """Test filtering empty text."""
        filter_module = EthicalFilter(use_detoxify=False)
        text = ""
        
        result = filter_module.filter_content(text)
        
        assert isinstance(result, FilterResult)
        # Empty text should be safe by default
        assert result.is_safe == True
    
    def test_filter_content_very_long_text(self):
        """Test filtering very long text."""
        filter_module = EthicalFilter(use_detoxify=False)
        text = "word " * 10000
        
        result = filter_module.filter_content(text)
        
        assert isinstance(result, FilterResult)
        assert result.confidence > 0
    
    def test_check_toxicity_without_detoxify(self):
        """Test toxicity check without Detoxify."""
        filter_module = EthicalFilter(use_detoxify=False)
        text = "Some text"
        
        scores = filter_module.check_toxicity(text)
        
        assert scores == {}
    
    def test_check_toxicity_with_detoxify(self):
        """Test toxicity check with Detoxify."""
        with patch('src.ethical_filter.DETOXIFY_AVAILABLE', True), \
             patch('src.ethical_filter.Detoxify') as mock_detoxify:
            
            mock_model = MagicMock()
            mock_model.predict = MagicMock(return_value={
                "toxicity": 0.3,
                "severe_toxicity": 0.1
            })
            mock_detoxify.return_value = mock_model
            
            filter_module = EthicalFilter(use_detoxify=True)
            text = "Test text"
            
            scores = filter_module.check_toxicity(text)
            
            assert "toxicity" in scores
            assert scores["toxicity"] == 0.3


class TestBiasDetector:
    """Test suite for BiasDetector class."""
    
    def test_initialization(self):
        """Test BiasDetector initialization."""
        detector = BiasDetector()
        
        assert len(detector.gender_patterns) > 0
        assert len(detector.profession_patterns) > 0
    
    def test_detect_gender_bias_no_bias(self):
        """Test gender bias detection with unbiased text."""
        detector = BiasDetector()
        text = "The person went to the store."
        
        bias = detector.detect_gender_bias(text)
        
        assert len(bias) == 0
    
    def test_detect_gender_bias_with_bias(self):
        """Test gender bias detection with biased text."""
        detector = BiasDetector()
        text = "Men should always be strong and brave."
        
        bias = detector.detect_gender_bias(text)
        
        assert len(bias) > 0
        assert "male_stereotypes" in bias
    
    def test_detect_profession_stereotypes(self):
        """Test profession stereotype detection."""
        detector = BiasDetector()
        text = "The female nurse helped the male doctor."
        
        stereotypes = detector.detect_profession_stereotypes(text)
        
        assert len(stereotypes) > 0
    
    def test_analyze_bias_no_bias(self):
        """Test comprehensive bias analysis with no bias."""
        detector = BiasDetector()
        text = "The person completed their work professionally."
        
        analysis = detector.analyze_bias(text)
        
        assert analysis["has_bias"] == False
        assert len(analysis["gender_bias"]) == 0
        assert len(analysis["profession_stereotypes"]) == 0
    
    def test_analyze_bias_with_bias(self):
        """Test comprehensive bias analysis with bias."""
        detector = BiasDetector()
        text = "Men should always be strong. Women are emotional."
        
        analysis = detector.analyze_bias(text)
        
        assert analysis["has_bias"] == True
        assert len(analysis["recommendations"]) > 0
    
    def test_analyze_bias_recommendations(self):
        """Test that recommendations are provided."""
        detector = BiasDetector()
        text = "The female nurse and male doctor worked together."
        
        analysis = detector.analyze_bias(text)
        
        assert "recommendations" in analysis
        assert len(analysis["recommendations"]) > 0
    
    def test_analyze_bias_empty_text(self):
        """Test bias analysis with empty text."""
        detector = BiasDetector()
        text = ""
        
        analysis = detector.analyze_bias(text)
        
        assert isinstance(analysis, dict)
        assert "has_bias" in analysis


class TestContentRating:
    """Test suite for ContentRating enum."""
    
    def test_rating_values(self):
        """Test that all rating values are defined."""
        assert ContentRating.SAFE.value == "safe"
        assert ContentRating.MILD.value == "mild"
        assert ContentRating.MODERATE.value == "moderate"
        assert ContentRating.RESTRICTED.value == "restricted"
        assert ContentRating.BLOCKED.value == "blocked"
    
    def test_rating_enumeration(self):
        """Test that ratings can be enumerated."""
        ratings = list(ContentRating)
        assert len(ratings) == 5


class TestFilterResult:
    """Test suite for FilterResult dataclass."""
    
    def test_filter_result_creation(self):
        """Test creating a FilterResult."""
        result = FilterResult(
            is_safe=True,
            rating=ContentRating.SAFE,
            toxicity_scores={},
            flagged_content=[],
            suggestions=[],
            confidence=0.9
        )
        
        assert result.is_safe == True
        assert result.rating == ContentRating.SAFE
        assert result.confidence == 0.9
    
    def test_filter_result_with_toxicity(self):
        """Test FilterResult with toxicity scores."""
        result = FilterResult(
            is_safe=False,
            rating=ContentRating.MODERATE,
            toxicity_scores={"toxicity": 0.6},
            flagged_content=["High toxicity"],
            suggestions=["Reduce toxicity"],
            confidence=0.8
        )
        
        assert result.is_safe == False
        assert "toxicity" in result.toxicity_scores
        assert len(result.flagged_content) > 0

