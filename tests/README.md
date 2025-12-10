# Test Suite for Ethical AI Storyteller

This directory contains comprehensive tests for the Ethical AI Storyteller project.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared pytest fixtures
├── README.md                # This file
├── test_story_generator.py  # Tests for story generation
├── test_ethical_filter.py   # Tests for content filtering
├── test_dataset.py          # Tests for dataset handling
├── test_explainability.py   # Tests for explainability features
├── test_evaluate.py         # Tests for evaluation utilities
├── test_finetune.py         # Tests for fine-tuning
└── test_storyteller.py      # Integration tests
```

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test File
```bash
pytest tests/test_story_generator.py
```

### Run Specific Test Class
```bash
pytest tests/test_story_generator.py::TestStoryGenerator
```

### Run Specific Test Function
```bash
pytest tests/test_story_generator.py::TestStoryGenerator::test_initialization_default
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run Only Fast Tests (Skip Slow Integration Tests)
```bash
pytest tests/ -m "not slow"
```

## Test Coverage

### Unit Tests
- **test_story_generator.py**: Tests for story generation functionality
  - Model initialization
  - Story generation with various parameters
  - Genre-based generation
  - Story continuation
  - Edge cases

- **test_ethical_filter.py**: Tests for ethical filtering
  - Content filtering
  - Toxicity detection
  - Bias detection
  - Content rating
  - Edge cases

- **test_dataset.py**: Tests for dataset handling
  - Dataset loading (JSON, JSONL, HuggingFace)
  - StoryExample creation
  - StoryDataset class
  - StoryPromptGenerator
  - Edge cases

- **test_explainability.py**: Tests for explainability features
  - Attention visualization
  - Token importance analysis
  - Generation explanation
  - Edge cases

- **test_evaluate.py**: Tests for evaluation utilities
  - Text loading
  - Model loading
  - Perplexity computation
  - Safety and bias evaluation

- **test_finetune.py**: Tests for fine-tuning
  - TrainingConfig
  - StorytellerFineTuner
  - Dataset preparation
  - Training loop (mocked)
  - Model saving

### Integration Tests
- **test_storyteller.py**: Full integration tests
  - Complete story generation pipeline
  - Ethical filtering integration
  - Bias detection integration
  - Explainability integration
  - Edge cases

## Test Categories

### Edge Cases Covered
- Empty inputs
- Very long inputs
- Invalid inputs
- Missing files
- Invalid model names
- Invalid genres
- Zero-length sequences
- Empty datasets

### Error Cases Covered
- File not found errors
- Invalid JSON format
- Model loading errors
- Invalid parameter values
- Missing dependencies

### Performance Considerations
- Tests use mocking to avoid loading large models
- Fast execution for CI/CD pipelines
- Separate slow tests for full integration

## Fixtures

Shared fixtures are defined in `conftest.py`:
- `sample_story_data`: Sample story data
- `sample_stories_json`: Temporary JSON file with stories
- `mock_model`: Mock transformer model
- `mock_tokenizer`: Mock tokenizer
- `device`: Available device for testing
- `safe_text`: Sample safe text
- `unsafe_text`: Sample unsafe text
- `empty_text`: Empty text for edge cases
- `very_long_text`: Very long text for edge cases
- `temp_dir`: Temporary directory for outputs

## Writing New Tests

When adding new functionality, follow these guidelines:

1. **Create test file** if testing a new module
2. **Use existing fixtures** from `conftest.py` when possible
3. **Mock external dependencies** (models, APIs) for fast tests
4. **Test edge cases** (empty, None, very long inputs)
5. **Test error cases** (invalid inputs, missing files)
6. **Use descriptive test names** that explain what is being tested
7. **Group related tests** in test classes

### Example Test Structure

```python
class TestNewFeature:
    """Test suite for NewFeature class."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        feature = NewFeature()
        
        # Act
        result = feature.do_something()
        
        # Assert
        assert result is not None
    
    def test_edge_case_empty_input(self):
        """Test with empty input."""
        feature = NewFeature()
        result = feature.do_something("")
        assert result == expected_value
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines:
- Fast execution (< 5 minutes for full suite)
- No external dependencies required (mocked)
- Deterministic results (seeded random)
- Clear error messages

## Known Limitations

- Some tests require mocking due to large model sizes
- Integration tests may be slower
- Some edge cases may not be fully covered yet

## Contributing

When contributing tests:
1. Ensure all tests pass locally
2. Add tests for new functionality
3. Update this README if adding new test categories
4. Keep test execution time reasonable
5. Use appropriate mocking to avoid external dependencies

