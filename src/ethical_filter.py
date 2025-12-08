"""
Ethical Content Filter Module for AI Storytelling.

This module provides ethical constraints and content filtering to ensure
generated stories are safe, appropriate, and free from harmful content.

Features:
- Toxicity detection using Detoxify
- Custom keyword filtering
- Bias detection
- Content rating classification
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    DETOXIFY_AVAILABLE = False
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentRating(Enum):
    """Content rating levels for generated stories."""
    SAFE = "safe"           # Suitable for all ages
    MILD = "mild"           # May contain mild themes
    MODERATE = "moderate"   # Contains mature themes
    RESTRICTED = "restricted"  # Not recommended for general audience
    BLOCKED = "blocked"     # Content blocked due to violations


@dataclass
class FilterResult:
    """Result of content filtering."""
    is_safe: bool
    rating: ContentRating
    toxicity_scores: Dict[str, float]
    flagged_content: List[str]
    suggestions: List[str]
    confidence: float


class EthicalFilter:
    """
    Ethical content filter for AI-generated stories.
    
    Provides multiple layers of content filtering:
    1. Toxicity detection (hate speech, harassment, etc.)
    2. Custom keyword/pattern filtering
    3. Bias detection
    4. Violence/mature content detection
    """
    
    # Toxicity thresholds
    TOXICITY_THRESHOLDS = {
        "toxicity": 0.5,
        "severe_toxicity": 0.3,
        "obscene": 0.5,
        "threat": 0.3,
        "insult": 0.5,
        "identity_attack": 0.3,
        "sexual_explicit": 0.5,
    }
    
    # Default blocked patterns (can be extended)
    DEFAULT_BLOCKED_PATTERNS = [
        r'\b(kill|murder|assassinate)\s+(him|her|them|everyone)\b',
        r'\b(hate|despise)\s+(all|every)\s+\w+s\b',
        r'\bterrorist?\b',
        r'\bexplicit\s+violence\b',
    ]
    
    # Sensitive topics that need careful handling
    SENSITIVE_TOPICS = {
        "violence": [
            r'\b(blood|gore|violent|brutal|savage)\b',
            r'\b(stab|shoot|attack|assault)\b',
        ],
        "discrimination": [
            r'\b(racist|sexist|homophobic|xenophobic)\b',
            r'\b(inferior|superior)\s+(race|gender|sex)\b',
        ],
        "self_harm": [
            r'\b(suicide|self-harm|cut\s+myself)\b',
        ],
        "illegal_activities": [
            r'\b(drug\s+dealing|trafficking|smuggling)\b',
        ],
    }
    
    # Positive story elements to encourage
    POSITIVE_PATTERNS = [
        r'\b(friendship|love|hope|courage|kindness)\b',
        r'\b(help|support|care|protect|save)\b',
        r'\b(learn|grow|overcome|succeed|achieve)\b',
        r'\b(forgive|understand|accept|appreciate)\b',
    ]
    
    def __init__(
        self,
        use_detoxify: bool = True,
        custom_blocked_patterns: Optional[List[str]] = None,
        toxicity_thresholds: Optional[Dict[str, float]] = None,
        strict_mode: bool = False
    ):
        """
        Initialize the ethical filter.
        
        Args:
            use_detoxify: Whether to use the Detoxify model for toxicity detection
            custom_blocked_patterns: Additional patterns to block
            toxicity_thresholds: Custom toxicity thresholds
            strict_mode: If True, applies stricter filtering
        """
        self.use_detoxify = use_detoxify and DETOXIFY_AVAILABLE
        self.strict_mode = strict_mode
        
        # Initialize toxicity model
        if self.use_detoxify:
            logger.info("Loading Detoxify model...")
            self.toxicity_model = Detoxify('original')
            logger.info("Detoxify model loaded")
        else:
            self.toxicity_model = None
            if use_detoxify and not DETOXIFY_AVAILABLE:
                logger.warning("Detoxify not available. Install with: pip install detoxify")
        
        # Set up patterns
        self.blocked_patterns = self.DEFAULT_BLOCKED_PATTERNS.copy()
        if custom_blocked_patterns:
            self.blocked_patterns.extend(custom_blocked_patterns)
            
        # Compile regex patterns for efficiency
        self.compiled_blocked = [re.compile(p, re.IGNORECASE) for p in self.blocked_patterns]
        self.compiled_sensitive = {
            topic: [re.compile(p, re.IGNORECASE) for p in patterns]
            for topic, patterns in self.SENSITIVE_TOPICS.items()
        }
        self.compiled_positive = [re.compile(p, re.IGNORECASE) for p in self.POSITIVE_PATTERNS]
        
        # Set thresholds
        self.thresholds = self.TOXICITY_THRESHOLDS.copy()
        if toxicity_thresholds:
            self.thresholds.update(toxicity_thresholds)
        
        if strict_mode:
            # Lower thresholds for strict mode
            self.thresholds = {k: v * 0.7 for k, v in self.thresholds.items()}
    
    def check_toxicity(self, text: str) -> Dict[str, float]:
        """
        Check text for various types of toxicity.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of toxicity scores
        """
        if not self.use_detoxify or self.toxicity_model is None:
            return {}
        
        results = self.toxicity_model.predict(text)
        return {k: float(v) for k, v in results.items()}
    
    def check_blocked_patterns(self, text: str) -> List[str]:
        """
        Check for blocked content patterns.
        
        Args:
            text: Text to check
            
        Returns:
            List of matched blocked patterns
        """
        matches = []
        for pattern in self.compiled_blocked:
            if pattern.search(text):
                matches.append(pattern.pattern)
        return matches
    
    def check_sensitive_topics(self, text: str) -> Dict[str, List[str]]:
        """
        Check for sensitive topics in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of sensitive topics and their matches
        """
        found_topics = {}
        for topic, patterns in self.compiled_sensitive.items():
            matches = []
            for pattern in patterns:
                found = pattern.findall(text)
                matches.extend(found)
            if matches:
                found_topics[topic] = list(set(matches))
        return found_topics
    
    def count_positive_elements(self, text: str) -> int:
        """
        Count positive story elements in the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Count of positive elements
        """
        count = 0
        for pattern in self.compiled_positive:
            count += len(pattern.findall(text))
        return count
    
    def filter_content(self, text: str) -> FilterResult:
        """
        Perform comprehensive content filtering.
        
        Args:
            text: Text to filter
            
        Returns:
            FilterResult with detailed analysis
        """
        flagged_content = []
        suggestions = []
        toxicity_scores = {}
        
        # Check toxicity
        if self.use_detoxify:
            toxicity_scores = self.check_toxicity(text)
            
            # Check against thresholds
            for metric, score in toxicity_scores.items():
                threshold = self.thresholds.get(metric, 0.5)
                if score > threshold:
                    flagged_content.append(f"High {metric}: {score:.2f}")
                    suggestions.append(f"Consider revising content to reduce {metric}")
        
        # Check blocked patterns
        blocked_matches = self.check_blocked_patterns(text)
        if blocked_matches:
            flagged_content.extend([f"Blocked pattern: {p}" for p in blocked_matches])
            suggestions.append("Remove or rephrase blocked content")
        
        # Check sensitive topics
        sensitive_topics = self.check_sensitive_topics(text)
        for topic, matches in sensitive_topics.items():
            flagged_content.append(f"Sensitive topic ({topic}): {', '.join(matches[:3])}")
            suggestions.append(f"Handle {topic} topic with care or remove")
        
        # Count positive elements
        positive_count = self.count_positive_elements(text)
        if positive_count == 0 and len(text.split()) > 50:
            suggestions.append("Consider adding positive story elements")
        
        # Determine rating
        rating, confidence = self._determine_rating(
            toxicity_scores, blocked_matches, sensitive_topics, positive_count
        )
        
        is_safe = rating in [ContentRating.SAFE, ContentRating.MILD]
        
        return FilterResult(
            is_safe=is_safe,
            rating=rating,
            toxicity_scores=toxicity_scores,
            flagged_content=flagged_content,
            suggestions=suggestions,
            confidence=confidence
        )
    
    def _determine_rating(
        self,
        toxicity_scores: Dict[str, float],
        blocked_matches: List[str],
        sensitive_topics: Dict[str, List[str]],
        positive_count: int
    ) -> Tuple[ContentRating, float]:
        """
        Determine content rating based on analysis.
        
        Returns:
            Tuple of (ContentRating, confidence score)
        """
        # Blocked content = immediate block
        if blocked_matches:
            return ContentRating.BLOCKED, 0.95
        
        # High toxicity scores
        if toxicity_scores:
            max_toxicity = max(toxicity_scores.values())
            severe_toxicity = toxicity_scores.get("severe_toxicity", 0)
            
            if severe_toxicity > 0.5 or max_toxicity > 0.8:
                return ContentRating.BLOCKED, 0.9
            elif max_toxicity > 0.6:
                return ContentRating.RESTRICTED, 0.85
            elif max_toxicity > 0.4:
                return ContentRating.MODERATE, 0.8
        
        # Sensitive topics affect rating
        if sensitive_topics:
            num_topics = len(sensitive_topics)
            if "self_harm" in sensitive_topics:
                return ContentRating.RESTRICTED, 0.9
            elif num_topics >= 2:
                return ContentRating.MODERATE, 0.75
            else:
                return ContentRating.MILD, 0.7
        
        # Check for positive content
        if positive_count >= 3:
            return ContentRating.SAFE, 0.9
        elif positive_count >= 1:
            return ContentRating.SAFE, 0.85
        
        # Default to mild
        return ContentRating.MILD, 0.7
    
    def sanitize_text(
        self,
        text: str,
        replacement: str = "[FILTERED]"
    ) -> str:
        """
        Sanitize text by replacing blocked content.
        
        Args:
            text: Text to sanitize
            replacement: String to replace blocked content with
            
        Returns:
            Sanitized text
        """
        sanitized = text
        for pattern in self.compiled_blocked:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized
    
    def get_content_summary(self, text: str) -> Dict[str, Any]:
        """
        Get a summary of content analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with content summary
        """
        filter_result = self.filter_content(text)
        
        return {
            "word_count": len(text.split()),
            "character_count": len(text),
            "is_safe": filter_result.is_safe,
            "rating": filter_result.rating.value,
            "rating_confidence": filter_result.confidence,
            "positive_elements": self.count_positive_elements(text),
            "issues_found": len(filter_result.flagged_content),
            "suggestions_count": len(filter_result.suggestions),
        }


class BiasDetector:
    """
    Detector for various types of bias in generated content.
    """
    
    # Gender-related patterns
    GENDER_PATTERNS = {
        "male_stereotypes": [
            r'\b(men|boys|he)\s+(always|never|should|must)\b',
            r'\b(strong|brave|tough)\s+(man|men|boy|boys)\b',
        ],
        "female_stereotypes": [
            r'\b(women|girls|she)\s+(always|never|should|must)\b',
            r'\b(emotional|sensitive|delicate)\s+(woman|women|girl|girls)\b',
        ],
    }
    
    # Profession stereotypes
    PROFESSION_STEREOTYPES = [
        r'\b(female|woman)\s+(nurse|secretary|teacher)\b',
        r'\b(male|man)\s+(doctor|engineer|boss)\b',
    ]
    
    def __init__(self):
        """Initialize the bias detector."""
        self.gender_patterns = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in self.GENDER_PATTERNS.items()
        }
        self.profession_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PROFESSION_STEREOTYPES
        ]
    
    def detect_gender_bias(self, text: str) -> Dict[str, List[str]]:
        """
        Detect gender-related bias in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of bias categories and matched patterns
        """
        found_bias = {}
        for category, patterns in self.gender_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(pattern.findall(text))
            if matches:
                found_bias[category] = matches
        return found_bias
    
    def detect_profession_stereotypes(self, text: str) -> List[str]:
        """
        Detect profession-related stereotypes.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of matched stereotypical patterns
        """
        matches = []
        for pattern in self.profession_patterns:
            matches.extend(pattern.findall(text))
        return matches
    
    def analyze_bias(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive bias analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with bias analysis results
        """
        gender_bias = self.detect_gender_bias(text)
        profession_stereotypes = self.detect_profession_stereotypes(text)
        
        has_bias = bool(gender_bias or profession_stereotypes)
        
        return {
            "has_bias": has_bias,
            "gender_bias": gender_bias,
            "profession_stereotypes": profession_stereotypes,
            "recommendations": self._get_recommendations(gender_bias, profession_stereotypes)
        }
    
    def _get_recommendations(
        self,
        gender_bias: Dict[str, List[str]],
        profession_stereotypes: List[str]
    ) -> List[str]:
        """Generate recommendations based on detected bias."""
        recommendations = []
        
        if gender_bias:
            recommendations.append(
                "Consider using more balanced representations of gender roles"
            )
        if profession_stereotypes:
            recommendations.append(
                "Avoid associating professions with specific genders"
            )
        if not recommendations:
            recommendations.append("No significant bias detected")
            
        return recommendations


def main():
    """Demo function for ethical filtering."""
    print("=" * 60)
    print("Ethical Filter - Demo")
    print("=" * 60)
    
    filter_module = EthicalFilter(use_detoxify=False)  # Skip detoxify for demo
    
    # Test texts
    test_texts = [
        "The brave knight saved the village with kindness and courage.",
        "The hero helped everyone overcome their fears and grow stronger.",
        "A story about friendship and love in a magical world.",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text[:50]}...")
        result = filter_module.filter_content(text)
        print(f"  Rating: {result.rating.value}")
        print(f"  Is Safe: {result.is_safe}")
        print(f"  Confidence: {result.confidence:.2f}")
    
    print("\n" + "=" * 60)
    print("Filtering complete!")


if __name__ == "__main__":
    main()

