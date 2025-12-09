"""
Main Storyteller Module - Integrates all components.

This module brings together story generation, ethical filtering,
and explainability into a unified interface.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .story_generator import StoryGenerator
from .ethical_filter import EthicalFilter, BiasDetector, ContentRating, FilterResult
from .explainability import (
    AttentionVisualizer,
    TokenImportanceAnalyzer,
    GenerationExplainer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StoryResult:
    """Complete result from story generation."""
    story: str
    prompt: str
    genre: Optional[str]
    is_safe: bool
    content_rating: str
    ethical_analysis: Dict[str, Any]
    bias_analysis: Dict[str, Any]
    generation_params: Dict[str, Any]
    warnings: List[str]


class EthicalStoryTeller:
    """
    Main class for ethical AI storytelling.
    
    Combines story generation with ethical constraints and explainability.
    
    Features:
    - Multiple genre support
    - Real-time content filtering
    - Bias detection
    - Explainable generation
    - Interactive story continuation
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        enable_ethical_filter: bool = True,
        enable_bias_detection: bool = True,
        strict_mode: bool = False,
        seed: Optional[int] = 42
    ):
        """
        Initialize the Ethical Story Teller.
        
        Args:
            model_name: Name of the pretrained model
            device: Device to use (auto-detected if None)
            enable_ethical_filter: Whether to enable content filtering
            enable_bias_detection: Whether to enable bias detection
            strict_mode: Apply stricter content filtering
            seed: Random seed for reproducibility
        """
        logger.info("Initializing Ethical Story Teller...")
        
        # Initialize story generator
        self.generator = StoryGenerator(
            model_name=model_name,
            device=device,
            seed=seed
        )
        
        self.device = self.generator.device
        
        # Initialize ethical filter
        self.enable_ethical_filter = enable_ethical_filter
        if enable_ethical_filter:
            self.ethical_filter = EthicalFilter(
                use_detoxify=True,
                strict_mode=strict_mode
            )
        else:
            self.ethical_filter = None
        
        # Initialize bias detector
        self.enable_bias_detection = enable_bias_detection
        if enable_bias_detection:
            self.bias_detector = BiasDetector()
        else:
            self.bias_detector = None
        
        # Initialize explainability components
        self.attention_visualizer = AttentionVisualizer(
            self.generator.model,
            self.generator.tokenizer,
            self.device
        )
        self.token_analyzer = TokenImportanceAnalyzer(
            self.generator.model,
            self.generator.tokenizer,
            self.device
        )
        self.generation_explainer = GenerationExplainer(
            self.generator.model,
            self.generator.tokenizer,
            self.device
        )
        
        logger.info("Ethical Story Teller initialized successfully")
    
    def generate_story(
        self,
        prompt: str,
        genre: Optional[str] = None,
        max_length: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        num_variations: int = 1,
        ensure_safe: bool = True,
        max_retries: int = 3,
        **kwargs
    ) -> StoryResult:
        """
        Generate an ethical story.
        
        Args:
            prompt: Starting prompt for the story
            genre: Optional genre (fantasy, sci-fi, mystery, etc.)
            max_length: Maximum length in tokens
            temperature: Creativity temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            num_variations: Number of story variations to generate
            ensure_safe: Retry generation if content is unsafe
            max_retries: Maximum retries for safe content
            **kwargs: Additional generation parameters
            
        Returns:
            StoryResult with the generated story and analysis
        """
        warnings = []
        
        # Use genre prompt if specified
        if genre:
            if genre.lower() not in self.generator.GENRE_PROMPTS:
                available = ", ".join(self.generator.GENRE_PROMPTS.keys())
                raise ValueError(f"Unknown genre: {genre}. Available: {available}")
            
            full_prompt = self.generator.GENRE_PROMPTS[genre.lower()]
            if prompt:
                full_prompt += prompt
        else:
            full_prompt = prompt
        
        # Generate story (with retries if needed)
        best_story = None
        best_filter_result = None
        
        for attempt in range(max_retries if ensure_safe else 1):
            # Generate
            result = self.generator.generate_story(
                prompt=full_prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_variations,
                **kwargs
            )
            
            story_text = result["full_texts"][0]
            
            # Check ethical filter
            if self.enable_ethical_filter:
                filter_result = self.ethical_filter.filter_content(story_text)
                
                if filter_result.is_safe:
                    best_story = story_text
                    best_filter_result = filter_result
                    break
                elif attempt == max_retries - 1:
                    # Last attempt, use best available
                    if best_story is None or filter_result.confidence > best_filter_result.confidence:
                        best_story = story_text
                        best_filter_result = filter_result
                    warnings.append(f"Could not generate fully safe content after {max_retries} attempts")
                else:
                    # Store as candidate
                    if best_story is None:
                        best_story = story_text
                        best_filter_result = filter_result
                    # Increase temperature to get different content
                    temperature = min(1.2, temperature + 0.1)
            else:
                best_story = story_text
                break
        
        # Default filter result if not using filter
        if best_filter_result is None:
            best_filter_result = FilterResult(
                is_safe=True,
                rating=ContentRating.MILD,
                toxicity_scores={},
                flagged_content=[],
                suggestions=[],
                confidence=1.0
            )
        
        # Bias detection
        bias_analysis = {}
        if self.enable_bias_detection:
            bias_analysis = self.bias_detector.analyze_bias(best_story)
            if bias_analysis["has_bias"]:
                warnings.extend(bias_analysis["recommendations"])
        
        # Build ethical analysis dict
        ethical_analysis = {
            "is_safe": best_filter_result.is_safe,
            "rating": best_filter_result.rating.value,
            "confidence": best_filter_result.confidence,
            "toxicity_scores": best_filter_result.toxicity_scores,
            "flagged_content": best_filter_result.flagged_content,
            "suggestions": best_filter_result.suggestions
        }
        
        return StoryResult(
            story=best_story,
            prompt=full_prompt,
            genre=genre,
            is_safe=best_filter_result.is_safe,
            content_rating=best_filter_result.rating.value,
            ethical_analysis=ethical_analysis,
            bias_analysis=bias_analysis,
            generation_params={
                "model": self.generator.model_name,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "max_length": max_length
            },
            warnings=warnings
        )
    
    def continue_story(
        self,
        story_so_far: str,
        continuation_length: int = 100,
        **kwargs
    ) -> StoryResult:
        """
        Continue an existing story.
        
        Args:
            story_so_far: The existing story text
            continuation_length: Length of continuation
            **kwargs: Additional generation parameters
            
        Returns:
            StoryResult with the continued story
        """
        return self.generate_story(
            prompt=story_so_far,
            max_length=continuation_length,
            **kwargs
        )
    
    def explain_generation(
        self,
        text: str,
        num_trace_tokens: int = 10
    ) -> Dict[str, Any]:
        """
        Get explanation for how a story would be continued.
        
        Args:
            text: Current story text
            num_trace_tokens: Number of tokens to trace
            
        Returns:
            Dictionary with generation explanation
        """
        # Next token prediction
        next_token_info = self.generation_explainer.explain_next_token(text)
        
        # Generation trace
        trace = self.generation_explainer.trace_generation(
            text, 
            num_tokens=num_trace_tokens
        )
        
        return {
            "next_token": next_token_info,
            "generation_trace": trace,
            "continued_text": trace[-1]["current_text"] if trace else text
        }
    
    def visualize_attention(
        self,
        text: str,
        layer: int = -1,
        save_path: Optional[str] = None
    ):
        """
        Create attention visualization for the text.
        
        Args:
            text: Text to visualize
            layer: Layer to visualize
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        return self.attention_visualizer.plot_attention_heatmap(
            text, layer=layer, save_path=save_path
        )
    
    def analyze_token_importance(
        self,
        text: str,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze which tokens are most important.
        
        Args:
            text: Text to analyze
            save_path: Path to save visualization
            
        Returns:
            Dictionary with importance analysis
        """
        importance_data = self.token_analyzer.compute_leave_one_out_importance(text)
        
        if save_path:
            self.token_analyzer.plot_token_importance(
                importance_data, save_path=save_path
            )
        
        return importance_data
    
    def get_available_genres(self) -> List[str]:
        """Get list of available story genres."""
        return list(self.generator.GENRE_PROMPTS.keys())
    
    def get_genre_prompt(self, genre: str) -> str:
        """Get the prompt template for a genre."""
        if genre.lower() not in self.generator.GENRE_PROMPTS:
            raise ValueError(f"Unknown genre: {genre}")
        return self.generator.GENRE_PROMPTS[genre.lower()]


def main():
    """Demo the Ethical Story Teller."""
    print("=" * 70)
    print("🌟 Ethical AI Story Teller - Demo 🌟")
    print("=" * 70)
    
    # Initialize
    storyteller = EthicalStoryTeller(
        model_name="gpt2",
        enable_ethical_filter=False,  # Skip detoxify for demo
        enable_bias_detection=True
    )
    
    print("\nAvailable genres:", ", ".join(storyteller.get_available_genres()))
    
    # Generate a story
    print("\n📖 Generating a fantasy story...")
    result = storyteller.generate_story(
        prompt="A young wizard discovers",
        genre="fantasy",
        max_length=150,
        temperature=0.9
    )
    
    print(f"\nGenre: {result.genre}")
    print(f"Content Rating: {result.content_rating}")
    print(f"Is Safe: {result.is_safe}")
    print(f"\n📜 Story:\n{result.story}")
    
    if result.warnings:
        print(f"\n⚠️ Warnings: {result.warnings}")
    
    print("\n" + "=" * 70)
    print("Demo complete! ✨")


if __name__ == "__main__":
    main()

