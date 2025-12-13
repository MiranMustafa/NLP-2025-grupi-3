"""
Fast smoke tests (no model downloads).

These tests should run quickly and avoid loading large models.
"""


def test_imports_smoke():
    """Basic import smoke test."""
    from src.storyteller import EthicalStoryTeller  # noqa: F401
    from src.dataset import StoryDatasetLoader  # noqa: F401
    from src.ethical_filter import EthicalFilter  # noqa: F401


def test_dataset_sample_loads():
    """The bundled sample dataset should load."""
    from src.dataset import StoryDatasetLoader

    loader = StoryDatasetLoader()
    examples = loader.load_custom_dataset("data/sample_stories.json")
    assert len(examples) > 0
    assert all(hasattr(ex, "prompt") and hasattr(ex, "story") for ex in examples)


def test_offline_storyteller_initializes_without_loading_model():
    """Offline + lazy mode should not download/load on init."""
    from src.storyteller import EthicalStoryTeller

    st = EthicalStoryTeller(
        model_name="gpt2",
        offline_mode=True,
        lazy_model=True,
        use_detoxify=False,
    )
    assert st.generator.is_loaded() is False

