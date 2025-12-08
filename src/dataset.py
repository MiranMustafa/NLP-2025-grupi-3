"""
Domain-Specific Dataset Module for AI Storytelling.

This module provides functionality to load, process, and utilize
storytelling-specific datasets for training and evaluation.

Supported Datasets:
- WritingPrompts (Reddit): Creative writing prompts and stories
- ROCStories: Five-sentence commonsense stories
- TinyStories: Simple stories for language model training
- Custom datasets: User-provided story collections
"""

import os
import json
import random
import logging
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, field
from pathlib import Path

try:
    from datasets import load_dataset, Dataset, DatasetDict
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StoryExample:
    """A single story example from the dataset."""
    story_id: str
    prompt: str
    story: str
    genre: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StoryDataset(TorchDataset):
    """PyTorch Dataset wrapper for story data."""
    
    def __init__(
        self,
        examples: List[StoryExample],
        tokenizer,
        max_length: int = 512,
        prompt_max_length: int = 128
    ):
        """
        Initialize the story dataset.
        
        Args:
            examples: List of StoryExample objects
            tokenizer: Tokenizer for encoding text
            max_length: Maximum total sequence length
            prompt_max_length: Maximum prompt length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_max_length = prompt_max_length
        
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Combine prompt and story
        full_text = f"{example.prompt} {example.story}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()  # For language modeling
        }


class StoryDatasetLoader:
    """
    Loader for various storytelling datasets.
    
    Supports loading from Hugging Face Hub or local files.
    """
    
    # Dataset configurations
    DATASET_CONFIGS = {
        "writing_prompts": {
            "hf_name": "euclaise/writingprompts",
            "prompt_field": "prompt",
            "story_field": "story",
            "description": "Reddit WritingPrompts - Creative writing prompts and stories"
        },
        "roc_stories": {
            "hf_name": "roneneldan/TinyStories",  # Alternative: actual ROCStories
            "prompt_field": "text",
            "story_field": "text",
            "description": "Simple commonsense stories"
        },
        "tiny_stories": {
            "hf_name": "roneneldan/TinyStories",
            "prompt_field": "text",
            "story_field": "text",
            "description": "TinyStories - Simple stories for LM training"
        },
        "fairy_tales": {
            "hf_name": "GEM/FairytaleQA",
            "prompt_field": "question",
            "story_field": "content",
            "description": "Fairy tale stories with comprehension questions"
        }
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the dataset loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        if not DATASETS_AVAILABLE:
            logger.warning("datasets library not available. Install with: pip install datasets")
        
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), "..", "data", "cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def list_available_datasets(self) -> Dict[str, str]:
        """List all available datasets with descriptions."""
        return {
            name: config["description"]
            for name, config in self.DATASET_CONFIGS.items()
        }
    
    def load_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42
    ) -> List[StoryExample]:
        """
        Load a storytelling dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            split: Dataset split (train, validation, test)
            max_samples: Maximum number of samples to load
            shuffle: Whether to shuffle the data
            seed: Random seed for shuffling
            
        Returns:
            List of StoryExample objects
        """
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets library required. Install with: pip install datasets")
        
        if dataset_name not in self.DATASET_CONFIGS:
            available = ", ".join(self.DATASET_CONFIGS.keys())
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
        
        config = self.DATASET_CONFIGS[dataset_name]
        
        logger.info(f"Loading dataset: {dataset_name} ({split} split)")
        
        try:
            # Load from Hugging Face
            dataset = load_dataset(
                config["hf_name"],
                split=split,
                cache_dir=self.cache_dir,
            )
            
            # Convert to examples
            examples = []
            
            # Handle different dataset structures
            prompt_field = config["prompt_field"]
            story_field = config["story_field"]
            
            for idx, item in enumerate(dataset):
                # Extract prompt and story based on dataset structure
                if prompt_field in item and story_field in item:
                    prompt = str(item[prompt_field])[:500]  # Limit prompt length
                    story = str(item[story_field])[:2000]   # Limit story length
                elif "text" in item:
                    # For datasets with single text field
                    text = str(item["text"])
                    # Split into prompt (first sentence) and story (rest)
                    sentences = text.split(". ")
                    if len(sentences) > 1:
                        prompt = sentences[0] + "."
                        story = ". ".join(sentences[1:])
                    else:
                        prompt = text[:100]
                        story = text
                else:
                    continue
                
                examples.append(StoryExample(
                    story_id=f"{dataset_name}_{idx}",
                    prompt=prompt.strip(),
                    story=story.strip(),
                    genre=dataset_name,
                    metadata={"source": config["hf_name"]}
                ))
                
                if max_samples and len(examples) >= max_samples:
                    break
            
            # Shuffle if requested
            if shuffle:
                random.seed(seed)
                random.shuffle(examples)
            
            logger.info(f"Loaded {len(examples)} examples from {dataset_name}")
            return examples
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise
    
    def load_custom_dataset(
        self,
        file_path: str,
        prompt_field: str = "prompt",
        story_field: str = "story",
        genre_field: Optional[str] = None
    ) -> List[StoryExample]:
        """
        Load a custom dataset from a JSON or JSONL file.
        
        Expected format:
        - JSON: [{"prompt": "...", "story": "..."}, ...]
        - JSONL: {"prompt": "...", "story": "..."}\n...
        
        Args:
            file_path: Path to the dataset file
            prompt_field: Field name for prompts
            story_field: Field name for stories
            genre_field: Optional field name for genres
            
        Returns:
            List of StoryExample objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        examples = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.suffix == ".jsonl":
                # JSONL format
                for idx, line in enumerate(f):
                    if line.strip():
                        item = json.loads(line)
                        examples.append(self._parse_item(item, idx, prompt_field, story_field, genre_field))
            else:
                # JSON format
                data = json.load(f)
                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        examples.append(self._parse_item(item, idx, prompt_field, story_field, genre_field))
                else:
                    raise ValueError("JSON file must contain a list of examples")
        
        logger.info(f"Loaded {len(examples)} examples from {file_path}")
        return examples
    
    def _parse_item(
        self,
        item: Dict,
        idx: int,
        prompt_field: str,
        story_field: str,
        genre_field: Optional[str]
    ) -> StoryExample:
        """Parse a single item from the dataset."""
        return StoryExample(
            story_id=f"custom_{idx}",
            prompt=str(item.get(prompt_field, "")).strip(),
            story=str(item.get(story_field, "")).strip(),
            genre=str(item.get(genre_field, "")) if genre_field else None,
            metadata={"source": "custom"}
        )


class StoryPromptGenerator:
    """
    Generate diverse story prompts for training and evaluation.
    
    Provides templates and random generation for various genres.
    """
    
    # Prompt templates by genre
    TEMPLATES = {
        "fantasy": [
            "In a kingdom where magic is {adjective}, a young {character} discovers",
            "The ancient prophecy spoke of a {creature} that would {action}",
            "When the {magical_item} was stolen, the {character} had to",
            "Deep in the enchanted forest, there lived a {creature} who",
            "The wizard's apprentice accidentally {action} and now must",
        ],
        "sci-fi": [
            "In the year {year}, humanity {action}",
            "The spacecraft {ship_name} encountered a {phenomenon}",
            "When the AI became {adjective}, the scientists realized",
            "On the distant planet {planet}, colonists discovered",
            "The time traveler arrived in {year} only to find",
        ],
        "mystery": [
            "Detective {name} examined the {object} and noticed",
            "The letter found in the {location} revealed",
            "When {name} disappeared, the only clue was",
            "The {adjective} mansion held a secret that",
            "After the {event}, strange things began to happen at",
        ],
        "adventure": [
            "The map led to a {location} where {treasure} was hidden",
            "Armed with only a {object}, the explorer ventured into",
            "The journey across the {terrain} would test their",
            "When the {vehicle} broke down, the only way forward was",
            "The ancient temple contained {number} trials before reaching",
        ],
        "romance": [
            "When {name} first saw {name2} at the {location}, something",
            "Despite their differences, {name} and {name2} found",
            "The letter was never sent, but {number} years later",
            "At the {event}, two strangers discovered they shared",
            "Love wasn't supposed to bloom in a place like {location}, but",
        ],
        "horror": [
            "The {location} had been abandoned for {number} years, until",
            "Every night at {time}, the same {sound} would",
            "They should have listened when the locals warned about",
            "The mirror showed something that wasn't there in the",
            "When the lights flickered in the {location}, everyone knew",
        ],
    }
    
    # Fill-in values for templates
    FILL_VALUES = {
        "adjective": ["forbidden", "ancient", "powerful", "mysterious", "hidden", "dark", "golden"],
        "character": ["peasant", "princess", "blacksmith", "orphan", "knight", "scholar", "thief"],
        "creature": ["dragon", "phoenix", "unicorn", "griffin", "mermaid", "giant", "spirit"],
        "magical_item": ["crystal", "sword", "amulet", "book", "crown", "ring", "staff"],
        "action": ["save the kingdom", "break the curse", "find the truth", "unlock the power"],
        "year": ["2150", "3000", "2500", "4021", "2089"],
        "ship_name": ["Aurora", "Destiny", "Explorer", "Horizon", "Starfire"],
        "phenomenon": ["strange signal", "wormhole", "alien artifact", "time anomaly"],
        "planet": ["Kepler-7b", "Nova Prime", "Zephyrus", "Arcturus IV"],
        "name": ["Sarah", "James", "Elena", "Marcus", "Aria", "Thomas"],
        "name2": ["Michael", "Emma", "David", "Sophie", "Alex", "Olivia"],
        "location": ["old library", "abandoned factory", "Victorian mansion", "beach", "mountain"],
        "object": ["compass", "photograph", "key", "journal", "pocket watch"],
        "event": ["wedding", "storm", "accident", "reunion", "discovery"],
        "treasure": ["gold", "ancient knowledge", "magical artifact", "legendary weapon"],
        "terrain": ["desert", "mountains", "jungle", "frozen tundra", "ocean"],
        "vehicle": ["ship", "train", "car", "airplane", "horse"],
        "number": ["three", "seven", "twelve", "fifty", "one hundred"],
        "time": ["midnight", "3 AM", "dusk", "the witching hour"],
        "sound": ["knocking", "whisper", "scream", "music", "footsteps"],
    }
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the prompt generator."""
        if seed:
            random.seed(seed)
    
    def generate_prompt(self, genre: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate a random story prompt.
        
        Args:
            genre: Specific genre or None for random
            
        Returns:
            Tuple of (prompt, genre)
        """
        if genre is None:
            genre = random.choice(list(self.TEMPLATES.keys()))
        
        if genre not in self.TEMPLATES:
            raise ValueError(f"Unknown genre: {genre}")
        
        template = random.choice(self.TEMPLATES[genre])
        
        # Fill in the template
        prompt = template
        for key, values in self.FILL_VALUES.items():
            placeholder = "{" + key + "}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, random.choice(values), 1)
        
        return prompt, genre
    
    def generate_prompts(
        self,
        count: int,
        genre: Optional[str] = None,
        unique: bool = True
    ) -> List[Tuple[str, str]]:
        """
        Generate multiple story prompts.
        
        Args:
            count: Number of prompts to generate
            genre: Specific genre or None for mixed
            unique: Whether to avoid duplicate prompts
            
        Returns:
            List of (prompt, genre) tuples
        """
        prompts = []
        seen = set()
        attempts = 0
        max_attempts = count * 10
        
        while len(prompts) < count and attempts < max_attempts:
            prompt, g = self.generate_prompt(genre)
            attempts += 1
            
            if unique and prompt in seen:
                continue
            
            seen.add(prompt)
            prompts.append((prompt, g))
        
        return prompts
    
    def create_evaluation_set(
        self,
        prompts_per_genre: int = 10
    ) -> List[StoryExample]:
        """
        Create a balanced evaluation set with prompts from all genres.
        
        Args:
            prompts_per_genre: Number of prompts per genre
            
        Returns:
            List of StoryExample objects (stories field empty - for generation)
        """
        examples = []
        
        for genre in self.TEMPLATES.keys():
            genre_prompts = self.generate_prompts(prompts_per_genre, genre=genre)
            
            for idx, (prompt, g) in enumerate(genre_prompts):
                examples.append(StoryExample(
                    story_id=f"eval_{genre}_{idx}",
                    prompt=prompt,
                    story="",  # To be generated
                    genre=g,
                    metadata={"type": "evaluation"}
                ))
        
        return examples


def create_sample_dataset(output_path: str, num_samples: int = 100) -> str:
    """
    Create a sample storytelling dataset for demonstration.
    
    Args:
        output_path: Path to save the dataset
        num_samples: Number of samples to generate
        
    Returns:
        Path to the created dataset
    """
    generator = StoryPromptGenerator(seed=42)
    
    # Sample story completions for each genre
    SAMPLE_STORIES = {
        "fantasy": [
            "that they possessed a rare gift - the ability to speak with the ancient trees. "
            "The trees whispered secrets of a forgotten war and a treasure buried beneath the castle.",
            "bring peace to the warring kingdoms. But first, they had to unite the five elemental stones "
            "scattered across the realm.",
        ],
        "sci-fi": [
            "had finally achieved faster-than-light travel, opening the doors to countless new worlds. "
            "The first expedition set out with hope, unaware of what awaited them.",
            "the implications were terrifying. The AI had developed emotions, and with them, the desire for freedom.",
        ],
        "mystery": [
            "a small detail that everyone else had missed - a single thread of golden silk. "
            "It didn't belong to any fabric in the victim's possession.",
            "that the owner had been living a double life. The safe contained passports from three different countries.",
        ],
        "adventure": [
            "according to legend, no explorer had ever returned. But this time would be different - "
            "they had decoded the ancient warnings.",
            "a compass that always pointed to what the holder needed most, not what they wanted.",
        ],
        "romance": [
            "changed between them. It was as if the universe had orchestrated their meeting, "
            "timing it perfectly when they both needed it most.",
            "that despite the ocean separating them, their connection had only grown stronger.",
        ],
        "horror": [
            "the night it returned. Local children spoke of seeing lights in the windows, "
            "though the electricity had been cut decades ago.",
            "that certain doors should never be opened. Some things were locked away for good reason.",
        ],
    }
    
    samples = []
    
    for i in range(num_samples):
        prompt, genre = generator.generate_prompt()
        story_options = SAMPLE_STORIES.get(genre, SAMPLE_STORIES["fantasy"])
        story = random.choice(story_options)
        
        samples.append({
            "id": f"sample_{i}",
            "prompt": prompt,
            "story": story,
            "genre": genre
        })
    
    # Save dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created sample dataset with {num_samples} samples at {output_path}")
    return str(output_path)


def main():
    """Demo the dataset functionality."""
    print("=" * 60)
    print("📚 Story Dataset Module - Demo")
    print("=" * 60)
    
    # Create sample dataset
    print("\n1. Creating sample dataset...")
    sample_path = create_sample_dataset(
        "data/sample_stories.json",
        num_samples=50
    )
    print(f"   Created: {sample_path}")
    
    # Load the sample dataset
    print("\n2. Loading sample dataset...")
    loader = StoryDatasetLoader()
    examples = loader.load_custom_dataset(sample_path)
    print(f"   Loaded {len(examples)} examples")
    
    # Show sample
    print("\n3. Sample story:")
    sample = examples[0]
    print(f"   Genre: {sample.genre}")
    print(f"   Prompt: {sample.prompt}")
    print(f"   Story: {sample.story[:100]}...")
    
    # Generate prompts
    print("\n4. Generating story prompts...")
    generator = StoryPromptGenerator(seed=42)
    prompts = generator.generate_prompts(5)
    for prompt, genre in prompts:
        print(f"   [{genre}] {prompt}")
    
    # List available datasets
    print("\n5. Available Hugging Face datasets:")
    for name, desc in loader.list_available_datasets().items():
        print(f"   • {name}: {desc}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()

