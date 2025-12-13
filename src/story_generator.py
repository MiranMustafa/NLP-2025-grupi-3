"""
Story Generation Module using Pretrained Language Models.

This module provides the core story generation functionality using
GPT-2 based models with support for various generation strategies.
"""

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)
from typing import Optional, List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StoryGenerator:
    """
    A story generation class using pretrained GPT-2 models.
    
    Supports multiple generation strategies including:
    - Greedy decoding
    - Beam search
    - Top-k sampling
    - Top-p (nucleus) sampling
    - Temperature-based sampling
    """
    
    # Available pretrained models
    AVAILABLE_MODELS = {
        "gpt2": "gpt2",
        "gpt2-medium": "gpt2-medium",
        "gpt2-large": "gpt2-large",
        "distilgpt2": "distilgpt2",
    }
    
    # Story genre prompts for guided generation
    GENRE_PROMPTS = {
        "fantasy": "In a magical kingdom far beyond the mountains, ",
        "sci-fi": "In the year 3000, humanity had finally mastered interstellar travel. ",
        "mystery": "The detective examined the crime scene carefully, noticing ",
        "romance": "When their eyes met across the crowded room, ",
        "horror": "The old mansion stood silent in the moonlight, hiding ",
        "adventure": "The brave explorer set out on a journey to discover ",
        "fairy_tale": "Once upon a time, in a land of wonder and magic, ",
        "historical": "In the ancient city, during the reign of the great empire, ",
    }
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        seed: Optional[int] = 42,
        use_quantization: bool = False
    ):
        """
        Initialize the story generator.
        
        Args:
            model_name: Name of the pretrained model to use
            device: Device to run the model on ('cuda', 'mps', 'cpu', or None for auto)
            seed: Random seed for reproducibility
            use_quantization: Enable 8-bit quantization for reduced memory (CPU only)
        """
        self.model_name = self.AVAILABLE_MODELS.get(model_name, model_name)
        self.use_quantization = use_quantization
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Set seed for reproducibility
        if seed is not None:
            set_seed(seed)
            self.seed = seed
        
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the pretrained model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load with quantization if enabled
        if self.use_quantization and self.device == "cpu":
            logger.info("Loading model with dynamic quantization for efficiency...")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            # Apply dynamic quantization for faster CPU inference
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            logger.info("Quantization applied successfully")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
        
    def generate_story(
        self,
        prompt: str,
        max_length: int = 200,
        min_length: int = 50,
        num_return_sequences: int = 1,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
        num_beams: int = 1,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a story based on the given prompt.
        
        Args:
            prompt: The starting text for the story
            max_length: Maximum length of generated text (in tokens)
            min_length: Minimum length of generated text (in tokens)
            num_return_sequences: Number of story variations to generate
            temperature: Sampling temperature (higher = more creative)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling (vs greedy/beam search)
            num_beams: Number of beams for beam search
            no_repeat_ngram_size: Size of n-grams to avoid repeating
            early_stopping: Stop when all beams reach EOS
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing generated stories and metadata
        """
        # Encode the prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        prompt_length = inputs["input_ids"].shape[1]
        
        # Generation configuration
        generation_config = {
            "max_length": max_length + prompt_length,
            "min_length": min_length + prompt_length,
            "num_return_sequences": num_return_sequences,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "early_stopping": early_stopping,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "output_attentions": return_attention,
            "return_dict_in_generate": True,
        }
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # Decode generated sequences
        generated_texts = []
        for seq in outputs.sequences:
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            # Remove the prompt from the output
            story_text = text[len(prompt):].strip()
            generated_texts.append(story_text)
        
        result = {
            "prompt": prompt,
            "stories": generated_texts,
            "full_texts": [prompt + " " + story for story in generated_texts],
            "model": self.model_name,
            "generation_params": {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "max_length": max_length,
            }
        }
        
        if return_attention and hasattr(outputs, "attentions"):
            result["attentions"] = outputs.attentions
            
        return result
    
    def generate_with_genre(
        self,
        genre: str,
        custom_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a story with a specific genre.
        
        Args:
            genre: The story genre (fantasy, sci-fi, mystery, etc.)
            custom_prompt: Optional custom addition to the genre prompt
            **kwargs: Additional arguments passed to generate_story
            
        Returns:
            Dictionary containing generated stories and metadata
        """
        if genre.lower() not in self.GENRE_PROMPTS:
            available = ", ".join(self.GENRE_PROMPTS.keys())
            raise ValueError(f"Unknown genre: {genre}. Available: {available}")
        
        prompt = self.GENRE_PROMPTS[genre.lower()]
        if custom_prompt:
            prompt += custom_prompt
            
        result = self.generate_story(prompt, **kwargs)
        result["genre"] = genre
        
        return result
    
    def continue_story(
        self,
        story_so_far: str,
        continuation_length: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Continue an existing story.
        
        Args:
            story_so_far: The existing story text
            continuation_length: Length of continuation to generate
            **kwargs: Additional arguments passed to generate_story
            
        Returns:
            Dictionary containing the continued story
        """
        return self.generate_story(
            prompt=story_so_far,
            max_length=continuation_length,
            **kwargs
        )
    
    def get_token_probabilities(
        self,
        text: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get the top-k token probabilities for each position in the text.
        
        Useful for explainability and understanding model behavior.
        
        Args:
            text: Input text to analyze
            top_k: Number of top tokens to return per position
            
        Returns:
            List of dictionaries with token probabilities
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        results = []
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        for i, (token, token_probs) in enumerate(zip(tokens, probs[0])):
            top_probs, top_indices = torch.topk(token_probs, top_k)
            top_tokens = self.tokenizer.convert_ids_to_tokens(top_indices.tolist())
            
            results.append({
                "position": i,
                "current_token": token,
                "top_predictions": [
                    {"token": t, "probability": p.item()}
                    for t, p in zip(top_tokens, top_probs)
                ]
            })
        
        return results


def main():
    """Demo function showing story generation capabilities."""
    print("=" * 60)
    print("Ethical AI Storyteller - Demo")
    print("=" * 60)
    
    # Initialize generator
    generator = StoryGenerator(model_name="gpt2")
    
    # Generate a fantasy story
    print("\n📖 Generating a fantasy story...\n")
    result = generator.generate_with_genre(
        genre="fantasy",
        max_length=150,
        temperature=0.9
    )
    
    print(f"Genre: {result['genre']}")
    print(f"Prompt: {result['prompt']}")
    print(f"\nGenerated Story:\n{result['full_texts'][0]}")
    
    print("\n" + "=" * 60)
    print("Generation complete!")


if __name__ == "__main__":
    main()

