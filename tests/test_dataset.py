"""
Unit tests for dataset.py module.

Tests cover:
- Dataset loading
- StoryExample creation
- StoryDataset class
- StoryDatasetLoader
- StoryPromptGenerator
- Edge cases and error handling
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.dataset import (
    StoryExample,
    StoryDataset,
    StoryDatasetLoader,
    StoryPromptGenerator
)


class TestStoryExample:
    """Test suite for StoryExample dataclass."""
    
    def test_story_example_creation(self):
        """Test creating a StoryExample."""
        example = StoryExample(
            story_id="test_1",
            prompt="Once upon a time",
            story="there was a brave knight.",
            genre="fantasy"
        )
        
        assert example.story_id == "test_1"
        assert example.prompt == "Once upon a time"
        assert example.story == "there was a brave knight."
        assert example.genre == "fantasy"
        assert isinstance(example.metadata, dict)
    
    def test_story_example_with_metadata(self):
        """Test StoryExample with custom metadata."""
        example = StoryExample(
            story_id="test_1",
            prompt="Test",
            story="Story",
            genre="fantasy",
            metadata={"source": "test", "rating": 5}
        )
        
        assert example.metadata["source"] == "test"
        assert example.metadata["rating"] == 5
    
    def test_story_example_default_metadata(self):
        """Test StoryExample with default metadata."""
        example = StoryExample(
            story_id="test_1",
            prompt="Test",
            story="Story"
        )
        
        assert isinstance(example.metadata, dict)
        assert example.genre is None


class TestStoryDataset:
    """Test suite for StoryDataset class."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": MagicMock(squeeze=MagicMock(return_value=MagicMock())),
            "attention_mask": MagicMock(squeeze=MagicMock(return_value=MagicMock()))
        }
        return tokenizer
    
    def test_dataset_initialization(self, mock_tokenizer):
        """Test dataset initialization."""
        examples = [
            StoryExample(
                story_id="1",
                prompt="Test",
                story="Story",
                genre="fantasy"
            )
        ]
        
        dataset = StoryDataset(
            examples=examples,
            tokenizer=mock_tokenizer,
            max_length=256
        )
        
        assert len(dataset) == 1
        assert dataset.max_length == 256
    
    def test_dataset_length(self, mock_tokenizer):
        """Test dataset length."""
        examples = [
            StoryExample(story_id=str(i), prompt="Test", story="Story")
            for i in range(10)
        ]
        
        dataset = StoryDataset(examples=examples, tokenizer=mock_tokenizer)
        assert len(dataset) == 10
    
    def test_dataset_getitem(self, mock_tokenizer):
        """Test dataset item access."""
        examples = [
            StoryExample(
                story_id="1",
                prompt="Test prompt",
                story="Test story",
                genre="fantasy"
            )
        ]
        
        dataset = StoryDataset(examples=examples, tokenizer=mock_tokenizer)
        
        # Mock tokenizer to return proper format
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(squeeze=MagicMock(return_value=MagicMock())),
            "attention_mask": MagicMock(squeeze=MagicMock(return_value=MagicMock()))
        }
        
        item = dataset[0]
        
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
    
    def test_dataset_empty(self, mock_tokenizer):
        """Test dataset with no examples."""
        dataset = StoryDataset(examples=[], tokenizer=mock_tokenizer)
        assert len(dataset) == 0


class TestStoryDatasetLoader:
    """Test suite for StoryDatasetLoader class."""
    
    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = StoryDatasetLoader()
        
        assert loader.cache_dir is not None
        assert isinstance(loader.DATASET_CONFIGS, dict)
        assert len(loader.DATASET_CONFIGS) > 0
    
    def test_loader_initialization_custom_cache(self):
        """Test loader initialization with custom cache dir."""
        cache_dir = "/tmp/test_cache"
        loader = StoryDatasetLoader(cache_dir=cache_dir)
        
        assert loader.cache_dir == cache_dir
    
    def test_list_available_datasets(self):
        """Test listing available datasets."""
        loader = StoryDatasetLoader()
        datasets = loader.list_available_datasets()
        
        assert isinstance(datasets, dict)
        assert len(datasets) > 0
    
    def test_load_custom_dataset_json(self, sample_stories_json):
        """Test loading custom JSON dataset."""
        loader = StoryDatasetLoader()
        examples = loader.load_custom_dataset(sample_stories_json)
        
        assert len(examples) > 0
        assert all(isinstance(ex, StoryExample) for ex in examples)
        assert examples[0].story_id.startswith("test_")
    
    def test_load_custom_dataset_jsonl(self, tmp_path):
        """Test loading custom JSONL dataset."""
        # Create JSONL file
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            f.write('{"prompt": "Test", "story": "Story", "genre": "fantasy"}\n')
            f.write('{"prompt": "Test2", "story": "Story2", "genre": "sci-fi"}\n')
        
        loader = StoryDatasetLoader()
        examples = loader.load_custom_dataset(str(jsonl_file))
        
        assert len(examples) == 2
        assert all(isinstance(ex, StoryExample) for ex in examples)
    
    def test_load_custom_dataset_custom_fields(self, tmp_path):
        """Test loading custom dataset with custom field names."""
        json_file = tmp_path / "test.json"
        data = [
            {
                "text": "The prompt",
                "content": "The story",
                "type": "fantasy"
            }
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        
        loader = StoryDatasetLoader()
        examples = loader.load_custom_dataset(
            str(json_file),
            prompt_field="text",
            story_field="content",
            genre_field="type"
        )
        
        assert len(examples) > 0
        assert examples[0].prompt == "The prompt"
        assert examples[0].story == "The story"
        assert examples[0].genre == "fantasy"
    
    def test_load_custom_dataset_file_not_found(self):
        """Test loading non-existent dataset file."""
        loader = StoryDatasetLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_custom_dataset("nonexistent_file.json")
    
    def test_load_custom_dataset_invalid_json(self, tmp_path):
        """Test loading invalid JSON file."""
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, "w", encoding="utf-8") as f:
            f.write("not valid json")
        
        loader = StoryDatasetLoader()
        
        with pytest.raises(json.JSONDecodeError):
            loader.load_custom_dataset(str(invalid_file))
    
    def test_load_custom_dataset_invalid_format(self, tmp_path):
        """Test loading JSON file with invalid format."""
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, "w", encoding="utf-8") as f:
            json.dump({"not": "a list"}, f)
        
        loader = StoryDatasetLoader()
        
        with pytest.raises(ValueError, match="must contain a list"):
            loader.load_custom_dataset(str(invalid_file))
    
    @patch('src.dataset.DATASETS_AVAILABLE', True)
    @patch('src.dataset.load_dataset')
    def test_load_dataset_huggingface(self, mock_load_dataset):
        """Test loading HuggingFace dataset."""
        # Mock dataset
        mock_dataset = [
            {"prompt": "Test prompt", "story": "Test story"},
            {"prompt": "Test prompt 2", "story": "Test story 2"}
        ]
        mock_load_dataset.return_value = mock_dataset
        
        loader = StoryDatasetLoader()
        examples = loader.load_dataset("writing_prompts", max_samples=2)
        
        assert len(examples) > 0
        assert all(isinstance(ex, StoryExample) for ex in examples)
    
    @patch('src.dataset.DATASETS_AVAILABLE', False)
    def test_load_dataset_no_datasets_library(self):
        """Test loading dataset when datasets library is not available."""
        loader = StoryDatasetLoader()
        
        with pytest.raises(RuntimeError, match="datasets library required"):
            loader.load_dataset("writing_prompts")
    
    @patch('src.dataset.DATASETS_AVAILABLE', True)
    @patch('src.dataset.load_dataset')
    def test_load_dataset_invalid_name(self, mock_load_dataset):
        """Test loading invalid dataset name."""
        loader = StoryDatasetLoader()
        
        with pytest.raises(ValueError, match="Unknown dataset"):
            loader.load_dataset("invalid_dataset_name")


class TestStoryPromptGenerator:
    """Test suite for StoryPromptGenerator class."""
    
    def test_generator_initialization(self):
        """Test prompt generator initialization."""
        generator = StoryPromptGenerator()
        assert generator is not None
    
    def test_generator_initialization_with_seed(self):
        """Test prompt generator initialization with seed."""
        generator = StoryPromptGenerator(seed=42)
        assert generator is not None
    
    def test_generate_prompt_default(self):
        """Test generating a prompt with default (random) genre."""
        generator = StoryPromptGenerator(seed=42)
        prompt, genre = generator.generate_prompt()
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert genre in generator.TEMPLATES.keys()
    
    def test_generate_prompt_specific_genre(self):
        """Test generating a prompt with specific genre."""
        generator = StoryPromptGenerator(seed=42)
        prompt, genre = generator.generate_prompt(genre="fantasy")
        
        assert isinstance(prompt, str)
        assert genre == "fantasy"
    
    def test_generate_prompt_invalid_genre(self):
        """Test generating prompt with invalid genre."""
        generator = StoryPromptGenerator()
        
        with pytest.raises(ValueError, match="Unknown genre"):
            generator.generate_prompt(genre="invalid_genre")
    
    def test_generate_prompts_single(self):
        """Test generating a single prompt."""
        generator = StoryPromptGenerator(seed=42)
        prompts = generator.generate_prompts(1)
        
        assert len(prompts) == 1
        assert isinstance(prompts[0], tuple)
        assert len(prompts[0]) == 2
    
    def test_generate_prompts_multiple(self):
        """Test generating multiple prompts."""
        generator = StoryPromptGenerator(seed=42)
        prompts = generator.generate_prompts(5)
        
        assert len(prompts) == 5
        assert all(isinstance(p, tuple) for p in prompts)
    
    def test_generate_prompts_with_genre(self):
        """Test generating prompts with specific genre."""
        generator = StoryPromptGenerator(seed=42)
        prompts = generator.generate_prompts(3, genre="sci-fi")
        
        assert len(prompts) == 3
        assert all(genre == "sci-fi" for _, genre in prompts)
    
    def test_generate_prompts_unique(self):
        """Test generating unique prompts."""
        generator = StoryPromptGenerator(seed=42)
        prompts = generator.generate_prompts(10, unique=True)
        
        prompt_texts = [p[0] for p in prompts]
        assert len(prompt_texts) == len(set(prompt_texts))
    
    def test_create_evaluation_set(self):
        """Test creating evaluation set."""
        generator = StoryPromptGenerator(seed=42)
        examples = generator.create_evaluation_set(prompts_per_genre=2)
        
        assert len(examples) > 0
        assert all(isinstance(ex, StoryExample) for ex in examples)
        assert all(ex.story == "" for ex in examples)  # Stories should be empty
    
    def test_create_evaluation_set_balanced(self):
        """Test that evaluation set is balanced across genres."""
        generator = StoryPromptGenerator(seed=42)
        examples = generator.create_evaluation_set(prompts_per_genre=3)
        
        genres = [ex.genre for ex in examples]
        genre_counts = {g: genres.count(g) for g in set(genres)}
        
        # All genres should have similar counts
        assert all(count >= 3 for count in genre_counts.values())
    
    def test_templates_exist(self):
        """Test that templates are defined for all genres."""
        generator = StoryPromptGenerator()
        
        assert hasattr(generator, "TEMPLATES")
        assert len(generator.TEMPLATES) > 0
        assert "fantasy" in generator.TEMPLATES
        assert "sci-fi" in generator.TEMPLATES
    
    def test_fill_values_exist(self):
        """Test that fill values are defined."""
        generator = StoryPromptGenerator()
        
        assert hasattr(generator, "FILL_VALUES")
        assert len(generator.FILL_VALUES) > 0


class TestDatasetEdgeCases:
    """Test edge cases for dataset module."""
    
    def test_story_example_empty_strings(self):
        """Test StoryExample with empty strings."""
        example = StoryExample(
            story_id="",
            prompt="",
            story=""
        )
        
        assert example.story_id == ""
        assert example.prompt == ""
        assert example.story == ""
    
    def test_load_custom_dataset_empty_file(self, tmp_path):
        """Test loading empty JSON file."""
        empty_file = tmp_path / "empty.json"
        with open(empty_file, "w", encoding="utf-8") as f:
            json.dump([], f)
        
        loader = StoryDatasetLoader()
        examples = loader.load_custom_dataset(str(empty_file))
        
        assert len(examples) == 0
    
    def test_load_custom_dataset_missing_fields(self, tmp_path):
        """Test loading dataset with missing fields."""
        json_file = tmp_path / "test.json"
        data = [
            {"prompt": "Test"}  # Missing story field
        ]
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        
        loader = StoryDatasetLoader()
        examples = loader.load_custom_dataset(str(json_file))
        
        # Should still work, with empty story
        assert len(examples) > 0
        assert examples[0].story == ""
    
    def test_generate_prompts_zero_count(self):
        """Test generating zero prompts."""
        generator = StoryPromptGenerator()
        prompts = generator.generate_prompts(0)
        
        assert len(prompts) == 0
    
    def test_generate_prompts_very_large_count(self):
        """Test generating very large number of prompts."""
        generator = StoryPromptGenerator(seed=42)
        prompts = generator.generate_prompts(100, unique=False)
        
        assert len(prompts) == 100

