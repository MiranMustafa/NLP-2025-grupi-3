"""
Shared pytest fixtures and utilities for testing.
"""

import pytest
import torch
import tempfile
import json
import os
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

# Test data fixtures
@pytest.fixture
def sample_story_data():
    """Sample story data for testing."""
    return {
        "id": "test_1",
        "prompt": "Once upon a time",
        "story": "there was a brave knight who saved the kingdom.",
        "genre": "fantasy"
    }

@pytest.fixture
def sample_stories_json(tmp_path):
    """Create a temporary JSON file with sample stories."""
    data = [
        {
            "id": "test_1",
            "prompt": "In a magical kingdom",
            "story": "a young wizard discovered their powers.",
            "genre": "fantasy"
        },
        {
            "id": "test_2",
            "prompt": "In the year 3000",
            "story": "humanity reached the stars.",
            "genre": "sci-fi"
        }
    ]
    file_path = tmp_path / "test_stories.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return str(file_path)

@pytest.fixture
def mock_model():
    """Mock transformer model for testing."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    model.generate = MagicMock(return_value=MagicMock(
        sequences=torch.randint(0, 1000, (1, 10))
    ))
    return model

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token_id = 50256
    tokenizer.eos_token_id = 50256
    tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
    tokenizer.decode = MagicMock(return_value="decoded text")
    tokenizer.convert_ids_to_tokens = MagicMock(return_value=["token1", "token2"])
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    return tokenizer

@pytest.fixture
def device():
    """Get available device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

@pytest.fixture
def safe_text():
    """Sample safe text for testing."""
    return "The brave knight helped everyone with kindness and courage."

@pytest.fixture
def unsafe_text():
    """Sample unsafe text for testing."""
    return "This text contains blocked patterns that should be filtered."

@pytest.fixture
def empty_text():
    """Empty text for edge case testing."""
    return ""

@pytest.fixture
def very_long_text():
    """Very long text for edge case testing."""
    return "word " * 10000

@pytest.fixture
def special_characters_text():
    """Text with special characters for edge case testing."""
    return "Text with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"

@pytest.fixture
def unicode_text():
    """Text with unicode characters for edge case testing."""
    return "Unicode text: 你好世界 🌟 émoji 🎭"

@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test outputs."""
    return tmp_path

@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def mock_detoxify():
    """Mock Detoxify model for testing."""
    mock_model = MagicMock()
    mock_model.predict = MagicMock(return_value={
        "toxicity": 0.1,
        "severe_toxicity": 0.05,
        "obscene": 0.1,
        "threat": 0.05,
        "insult": 0.1,
        "identity_attack": 0.05,
        "sexual_explicit": 0.1
    })
    return mock_model

@pytest.fixture
def sample_story_example():
    """Sample StoryExample for testing."""
    from src.dataset import StoryExample
    return StoryExample(
        story_id="test_1",
        prompt="Once upon a time",
        story="there was a brave knight.",
        genre="fantasy",
        metadata={"source": "test"}
    )

@pytest.fixture
def sample_filter_result():
    """Sample FilterResult for testing."""
    from src.ethical_filter import FilterResult, ContentRating
    return FilterResult(
        is_safe=True,
        rating=ContentRating.SAFE,
        toxicity_scores={},
        flagged_content=[],
        suggestions=[],
        confidence=0.9
    )

