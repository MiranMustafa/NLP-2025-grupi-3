"""
Performance and edge case tests.

Tests cover:
- Performance benchmarks
- Stress testing with large inputs
- Memory usage
- Concurrent operations
- Edge cases with extreme values
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from src.ethical_filter import EthicalFilter, BiasDetector
from src.dataset import StoryDatasetLoader, StoryPromptGenerator


class TestPerformance:
    """Performance tests."""
    
    def test_ethical_filter_performance(self):
        """Test ethical filter performance with multiple texts."""
        filter_module = EthicalFilter(use_detoxify=False)
        
        texts = ["Safe text " * 100] * 100  # 100 texts
        
        start_time = time.time()
        for text in texts:
            result = filter_module.filter_content(text)
        end_time = time.time()
        
        elapsed = end_time - start_time
        # Should process 100 texts in reasonable time (< 5 seconds)
        assert elapsed < 5.0
        assert elapsed > 0
    
    def test_bias_detector_performance(self):
        """Test bias detector performance."""
        detector = BiasDetector()
        
        texts = ["Test text " * 50] * 100
        
        start_time = time.time()
        for text in texts:
            detector.analyze_bias(text)
        end_time = time.time()
        
        elapsed = end_time - start_time
        # Should be fast (< 2 seconds for 100 texts)
        assert elapsed < 2.0
    
    def test_prompt_generator_performance(self):
        """Test prompt generator performance."""
        generator = StoryPromptGenerator(seed=42)
        
        start_time = time.time()
        prompts = generator.generate_prompts(1000, unique=False)
        end_time = time.time()
        
        elapsed = end_time - start_time
        # Should generate 1000 prompts quickly (< 3 seconds)
        assert elapsed < 3.0
        assert len(prompts) == 1000


class TestEdgeCases:
    """Edge case tests with extreme values."""
    
    def test_ethical_filter_very_long_text(self):
        """Test ethical filter with very long text."""
        filter_module = EthicalFilter(use_detoxify=False)
        # Very long text (100k words)
        text = "word " * 100000
        
        result = filter_module.filter_content(text)
        
        assert isinstance(result, type(filter_module.filter_content("test")))
        assert result.confidence >= 0
    
    def test_ethical_filter_special_characters(self):
        """Test ethical filter with special characters."""
        filter_module = EthicalFilter(use_detoxify=False)
        text = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"
        
        result = filter_module.filter_content(text)
        
        assert isinstance(result, type(filter_module.filter_content("test")))
    
    def test_ethical_filter_unicode(self):
        """Test ethical filter with unicode characters."""
        filter_module = EthicalFilter(use_detoxify=False)
        text = "你好世界 🌟 émoji 🎭 日本語"
        
        result = filter_module.filter_content(text)
        
        assert isinstance(result, type(filter_module.filter_content("test")))
    
    def test_ethical_filter_newlines(self):
        """Test ethical filter with many newlines."""
        filter_module = EthicalFilter(use_detoxify=False)
        text = "\n" * 1000 + "some text" + "\n" * 1000
        
        result = filter_module.filter_content(text)
        
        assert isinstance(result, type(filter_module.filter_content("test")))
    
    def test_prompt_generator_extreme_count(self):
        """Test prompt generator with extreme count."""
        generator = StoryPromptGenerator(seed=42)
        
        # Should handle large counts
        prompts = generator.generate_prompts(100, unique=False)
        
        assert len(prompts) == 100
    
    def test_prompt_generator_all_genres(self):
        """Test prompt generator with all genres."""
        generator = StoryPromptGenerator(seed=42)
        
        genres = list(generator.TEMPLATES.keys())
        for genre in genres:
            prompt, g = generator.generate_prompt(genre=genre)
            assert g == genre
            assert len(prompt) > 0


class TestConcurrentOperations:
    """Tests for concurrent operations (if applicable)."""
    
    def test_multiple_filter_instances(self):
        """Test creating multiple filter instances."""
        filters = [EthicalFilter(use_detoxify=False) for _ in range(10)]
        
        text = "Test text"
        results = [f.filter_content(text) for f in filters]
        
        assert len(results) == 10
        assert all(r.is_safe for r in results)
    
    def test_multiple_detector_instances(self):
        """Test creating multiple detector instances."""
        detectors = [BiasDetector() for _ in range(10)]
        
        text = "Test text"
        results = [d.analyze_bias(text) for d in detectors]
        
        assert len(results) == 10
        assert all("has_bias" in r for r in results)


class TestMemoryUsage:
    """Tests for memory usage patterns."""
    
    def test_large_dataset_loading(self, tmp_path):
        """Test loading large dataset."""
        # Create large JSON file
        large_data = [
            {
                "id": f"sample_{i}",
                "prompt": "Test prompt " * 10,
                "story": "Test story " * 100,
                "genre": "fantasy"
            }
            for i in range(1000)
        ]
        
        json_file = tmp_path / "large.json"
        import json
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(large_data, f)
        
        loader = StoryDatasetLoader()
        examples = loader.load_custom_dataset(str(json_file))
        
        assert len(examples) == 1000
        assert all(ex.story_id.startswith("sample_") for ex in examples)


class TestBoundaryConditions:
    """Tests for boundary conditions."""
    
    def test_ethical_filter_zero_length_text(self):
        """Test with zero-length text."""
        filter_module = EthicalFilter(use_detoxify=False)
        result = filter_module.filter_content("")
        
        assert result.is_safe == True  # Empty should be safe
    
    def test_ethical_filter_single_character(self):
        """Test with single character."""
        filter_module = EthicalFilter(use_detoxify=False)
        result = filter_module.filter_content("a")
        
        assert isinstance(result, type(filter_module.filter_content("test")))
    
    def test_ethical_filter_only_whitespace(self):
        """Test with only whitespace."""
        filter_module = EthicalFilter(use_detoxify=False)
        result = filter_module.filter_content("   \n\t   ")
        
        assert isinstance(result, type(filter_module.filter_content("test")))
    
    def test_prompt_generator_zero_count(self):
        """Test generating zero prompts."""
        generator = StoryPromptGenerator(seed=42)
        prompts = generator.generate_prompts(0)
        
        assert len(prompts) == 0
    
    def test_prompt_generator_single_prompt(self):
        """Test generating single prompt."""
        generator = StoryPromptGenerator(seed=42)
        prompts = generator.generate_prompts(1)
        
        assert len(prompts) == 1


class TestErrorRecovery:
    """Tests for error recovery and resilience."""
    
    def test_ethical_filter_malformed_text(self):
        """Test with malformed text."""
        filter_module = EthicalFilter(use_detoxify=False)
        
        # Text with null bytes, control characters, etc.
        malformed_texts = [
            "text\x00with\x01null",
            "text\n\n\nwith\nmany\nnewlines",
            "text\t\twith\ttabs",
        ]
        
        for text in malformed_texts:
            result = filter_module.filter_content(text)
            assert isinstance(result, type(filter_module.filter_content("test")))
    
    def test_dataset_loader_missing_optional_fields(self, tmp_path):
        """Test loading dataset with missing optional fields."""
        import json
        data = [
            {"prompt": "Test"},  # Missing story
            {"story": "Story"},  # Missing prompt
            {"prompt": "Test", "story": "Story"}  # Complete
        ]
        
        json_file = tmp_path / "partial.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
        
        loader = StoryDatasetLoader()
        examples = loader.load_custom_dataset(str(json_file))
        
        # Should handle missing fields gracefully
        assert len(examples) == 3
        assert examples[0].story == ""  # Missing story becomes empty
        assert examples[1].prompt == ""  # Missing prompt becomes empty

