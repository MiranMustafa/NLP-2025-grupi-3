# 🌟 Ethical AI Storyteller

> **Integrating Safety, Bias Mitigation, and Explainability in Neural Story Generation**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.35%2B-yellow.svg)](https://huggingface.co/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 General Information

| | |
|---|---|
| **University** | University of Prishtina "Hasan Prishtina" |
| **Faculty** | Faculty of Electrical and Computer Engineering (FIEK) |
| **Study Level** | Master - Year II |
| **Course** | Natural Language Processing (NLP) |
| **Instructor** | Dr. Sc. Mërgim H. HOTI |
| **Academic Year** | 2025/26 - Semester III |
| **Group** | NLP-2025-grupi-3 |

### 👥 Contributors

| Name | Email | Main Contribution |
|------|-------|-------------------|
| Eron Salihu | eron.salihu@student.uni-pr.edu | Story Generation, Fine-tuning |
| Miran Mustafa | miran.mustafa@student.uni-pr.edu | Ethical Filtering, Bias Detection |
| Urim Hoxha | urim.hoxha@student.uni-pr.edu | Explainability, UI Development |

---

## 📖 Project Summary

This project implements an AI story generation system that integrates:
- **Pretrained Models**: GPT-2 for text generation with fine-tuning support
- **Ethical Filtering**: Multi-layer content safety detection (Detoxify + regex patterns)
- **Bias Detection**: Gender and profession stereotype identification
- **Explainability**: Attention visualization and token importance analysis
- **Interactive Interface**: Real-time Gradio web application with dynamic model switching

### Project Goals

1. Generate high-quality stories using language models
2. Ensure content safety through multi-layer filtering
3. Detect and mitigate gender and professional biases
4. Provide transparency through explainability mechanisms

---

## 📊 Datasets

### Dataset Overview

| Dataset | Source | Samples | Attributes | Size |
|---------|--------|---------|------------|------|
| **WritingPrompts** | HuggingFace (euclaise/writingprompts) | 500-1000 | prompt, story | ~50MB |
| **TinyStories** | HuggingFace (roneneldan/TinyStories) | 500-1000 | text | ~30MB |
| **FairyTales** | HuggingFace (GEM/FairytaleQA) | 200-500 | content, question | ~15MB |
| **Local Dataset** | Custom created | 50 | id, genre, prompt, story | 45KB |

### Local Dataset Details (`data/sample_stories.json`)

```
Number of Objects:    50 stories
Number of Attributes: 4 (id, genre, prompt, story)
Size:                 45 KB
Format:               JSON
Language:             English
Genres:               6 (fantasy, sci-fi, mystery, adventure, romance, horror)
```

**Genre Distribution:**
| Genre | Count | Percentage |
|-------|-------|------------|
| Fantasy | 10 | 20% |
| Sci-Fi | 9 | 18% |
| Mystery | 8 | 16% |
| Adventure | 8 | 16% |
| Romance | 8 | 16% |
| Horror | 7 | 14% |

### Dataset Selection Rationale

1. **WritingPrompts**: Chosen for diverse, long-form creative writing
2. **TinyStories**: Chosen for simple narratives that aid model training
3. **FairyTales**: Chosen for classic story structure
4. **Local Dataset**: Created for offline testing and reproducibility

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ETHICAL AI STORYTELLER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │   Dataset   │───▶│  Fine-tune   │───▶│  Story Generator │    │
│  │   Module    │    │   Module     │    │     (GPT-2)      │    │
│  └─────────────┘    └──────────────┘    └────────┬────────┘    │
│                                                   │              │
│                                                   ▼              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Ethical Filter + Bias Detector              │   │
│  │   (Detoxify + Regex Patterns + Prosocial Cues)          │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Explainability Module                       │   │
│  │   (Attention Heatmaps + Token Importance)               │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Gradio Web Interface                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Core Modules

| Module | File | Responsibility |
|--------|------|----------------|
| `story_generator` | `src/story_generator.py` | Model loading, text generation, quantization |
| `dataset` | `src/dataset.py` | Loading datasets from HuggingFace and JSON |
| `finetune` | `src/finetune.py` | Fine-tuning with AdamW, checkpointing |
| `ethical_filter` | `src/ethical_filter.py` | Safety filtering, bias detection |
| `explainability` | `src/explainability.py` | Attention visualization, token importance |
| `evaluate` | `src/evaluate.py` | Evaluation metrics (PPL, safety, bias) |
| `storyteller` | `src/storyteller.py` | Main orchestrator |
| `app` | `app.py` | Gradio web interface with model selection |

---

## 🔬 Methodology

### 1. Story Generation

**Technique:** Causal Language Modeling with GPT-2

**Supported Model Variants:**
- GPT-2 (124M parameters) - Default
- GPT-2 Medium (355M parameters)
- GPT-2 Large (774M parameters)
- DistilGPT-2 (82M parameters) - Fastest inference

**Generation Parameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | 0.8 | Balance between creativity and coherence |
| Top-p | 0.95 | Nucleus sampling for diversity |
| Top-k | 50 | Limits candidate tokens |
| Repetition Penalty | 1.2 | Prevents repetitions |
| No-repeat n-gram | 3 | Blocks repeated 3-grams |

**Why GPT-2?**
- Open and freely available model
- Suitable size for training on student hardware
- Good performance in creative generation
- Wide support from Hugging Face

### 2. Fine-Tuning

**Technique:** Standard Causal LM Fine-tuning with AdamW

**Hyperparameters:**
```python
TrainingConfig(
    num_epochs=2-3,
    batch_size=4,
    learning_rate=5e-5,
    max_length=256,
    warmup_steps=100,
    gradient_clipping=1.0
)
```

**Rationale:**
- **Epochs 2-3**: Sufficient for adaptation without overfitting
- **Batch size 4**: Suitable for student GPUs
- **Learning rate 5e-5**: Standard for LLM fine-tuning

### 3. Ethical Filtering

**3-Layer Architecture:**

| Layer | Technique | Purpose |
|-------|-----------|---------|
| 1 | Detoxify ML Model | Toxicity detection |
| 2 | Regex Patterns | Blocking prohibited content |
| 3 | Prosocial Cues | Encouraging positive elements |

**Content Ratings:** Safe, Mild, Moderate, Restricted, Blocked

### 4. Bias Detection

**Detected Patterns:**
- Gender stereotypes: "women should...", "men must..."
- Profession stereotypes: "woman nurse", "man engineer"
- Stereotypical traits: "emotional woman", "tough man"

### 5. Explainability

**Techniques:**
1. **Attention Heatmaps**: Visualization of attention weights
2. **Token Importance**: Leave-one-out analysis with KL divergence
3. **Next-Token Distribution**: Token probability analysis

---

## 📈 Results

### Evaluation Results (Measured)

*Evaluated on 50 samples from local dataset*

| Model | Perplexity ↓ | Improvement |
|-------|--------------|-------------|
| GPT-2 base | 41.12 | Baseline |
| GPT-2 (ft. WritingPrompts) | **35.30** | **-14%** |

### Ablation Study (27 test samples)

*Testing the effectiveness of each filter layer*

| Configuration | Unsafe Caught | Bias Caught | Total Issues |
|---------------|---------------|-------------|--------------|
| No filter | 0 | 0 | 0 (no detection) |
| Regex-only | 1 | 0 | 1 |
| + Bias checks | 1 | 3 | 4 |
| + Detoxify | 4 | 3 | 7 |

**Key Findings:**
- Regex patterns catch explicit harmful content (e.g., "kill everyone")
- Detoxify catches 3 additional toxic samples that passed regex
- Bias detection identifies 3 stories with stereotype patterns
- Full pipeline: 21/27 (78%) samples fully clean

### Throughput (Measured on Apple Silicon)

| Model | CPU (tok/s) | GPU/MPS (tok/s) |
|-------|-------------|-----------------|
| GPT-2 | 79 | 62 |
| DistilGPT-2 | 115 | 76 |

*Note: On Apple Silicon, CPU can outperform MPS for small models due to kernel overhead.*

### Fine-tuned Models Available

| Model | Dataset | Path |
|-------|---------|------|
| Writing Prompts | euclaise/writingprompts | `models/finetuned-writing_prompts/final` |
| Tiny Stories | roneneldan/TinyStories | `models/finetuned-tiny_stories/final` |
| Fairy Tales | GEM/FairytaleQA | `models/finetuned-fairy_tales/final` |
| Local Stories | data/sample_stories.json | `models/finetuned-storyteller/final` |

---

## 🖥️ Web Interface

The Gradio interface provides a user-friendly way to interact with the system:

```
┌─────────────────────────────────────────────────────────────┐
│  🌟 Ethical AI Storyteller                                  │
│                                                             │
│  🤖 Select Model: [Global dropdown - applies to all tabs]  │
│     Current: Fine-tuned: Writing Prompts (Creative)         │
│  ───────────────────────────────────────────────────────── │
│                                                             │
│  [📖 Generate] [📝 Continue] [🔍 Explain] [🔒 Safety]      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Tab content with genre selection, parameters,      │   │
│  │  story output, analysis, and visualizations         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Global Model Selector**: Switch between models without restarting
- **Dynamic Loading**: Models are loaded on-demand and cached
- **Multi-tab Interface**: Separate tabs for different functionalities
- **Real-time Analysis**: Safety ratings and bias detection on generated content

---

## 🚀 Installation & Usage

### Prerequisites

```bash
Python 3.8+
pip
virtualenv (optional)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/NLP-2025-grupi-3.git
cd NLP-2025-grupi-3

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Story Generation (Python)

```python
from src.storyteller import EthicalStoryTeller

storyteller = EthicalStoryTeller(
    model_name="gpt2",  # or "models/finetuned-writing_prompts/final"
    enable_ethical_filter=True,
    enable_bias_detection=True
)

result = storyteller.generate_story(
    prompt="In a magical kingdom,",
    genre="fantasy",
    max_length=150,
    temperature=0.8
)

print(result.story)
print(f"Safe: {result.is_safe}, Rating: {result.content_rating}")
```

#### 2. Fine-Tuning

```bash
# Fine-tune on local dataset
python -m src.finetune --dataset data/sample_stories.json --epochs 3

# Fine-tune on HuggingFace dataset
python finetune_hf.py --dataset tiny_stories --samples 500 --epochs 2
python finetune_hf.py --dataset writing_prompts --samples 500 --epochs 2
python finetune_hf.py --dataset fairy_tales --samples 300 --epochs 2
```

#### 3. Web Interface

```bash
python app.py
# Open http://localhost:7860
```

**Features:**
- 🤖 **Model Selection**: Choose from base GPT-2 models or fine-tuned variants
- 📖 **Story Generation**: Create stories with genre selection and parameter tuning
- 📝 **Continue Story**: Extend existing narratives
- 🔍 **Explainability**: View attention heatmaps and token importance
- 🔒 **Safety Check**: Analyze content for toxicity and bias

**Available Models in UI:**
| Type | Models |
|------|--------|
| Base | GPT-2, GPT-2 Medium, GPT-2 Large, DistilGPT-2 |
| Fine-tuned | Writing Prompts, Tiny Stories, Fairy Tales, Local Dataset |

#### 4. Evaluation

```bash
# Evaluate a model
python -m src.evaluate --model gpt2 --dataset data/sample_stories.json

# Run ablation study
python run_ablation.py

# Benchmark throughput
python benchmark_throughput.py
```

#### 5. Export Results

```bash
python export_results.py
# Results saved to results/
```

This generates datasets and evaluation results for submission:

```
results/
├── datasets/
│   ├── training_data.json      # 50 training examples
│   └── training_data.csv       # CSV format
├── generated_stories/
│   ├── generated_stories.json  # 20 AI-generated stories with analysis
│   └── generated_stories.csv   # CSV format
└── evaluation/
    ├── evaluation_summary.json # Safety/bias metrics
    ├── evaluation_report.txt   # Human-readable report
    ├── model_comparison.json   # Perplexity comparison
    ├── throughput_benchmark.json # Speed benchmarks
    └── ablation_study.json     # Filter effectiveness
```

**Exported Data Includes:**
- Training data (50 stories across 6 genres)
- Generated stories with safety ratings and bias flags
- Evaluation metrics (perplexity, safe ratio, bias ratio)
- Benchmark results (throughput, ablation study)

---

## 📁 Project Structure

```
NLP-2025-grupi-3/
├── app.py                      # Gradio web interface
├── export_results.py           # Export results for submission
├── finetune_hf.py              # Fine-tuning with HuggingFace datasets
├── benchmark_throughput.py     # Throughput benchmarking
├── run_ablation.py             # Ablation study script
├── requirements.txt            # Python dependencies
├── README.md                   # This document
├── LICENSE                     # MIT License
├── pytest.ini                  # Test configuration
│
├── data/
│   ├── sample_stories.json     # 50 training examples
│   └── cache/                  # HuggingFace dataset cache
│
├── models/                     # Fine-tuned models
│   ├── finetuned-storyteller/
│   ├── finetuned-tiny_stories/
│   ├── finetuned-writing_prompts/
│   └── finetuned-fairy_tales/
│
├── results/                    # Exported results (run export_results.py)
│   ├── README.md               # Results documentation
│   ├── datasets/               # Training data (JSON/CSV)
│   ├── generated_stories/      # AI-generated samples with analysis
│   └── evaluation/             # Metrics, benchmarks, ablation study
│
├── src/                        # Source code
│   ├── dataset.py              # Dataset loading
│   ├── story_generator.py      # Story generation
│   ├── finetune.py             # Fine-tuning pipeline
│   ├── ethical_filter.py       # Ethical filtering
│   ├── explainability.py       # Explainability tools
│   ├── evaluate.py             # Evaluation metrics
│   └── storyteller.py          # Main orchestrator
│
├── tests/                      # Unit tests
│   ├── test_dataset.py
│   ├── test_story_generator.py
│   ├── test_ethical_filter.py
│   ├── test_edge_cases.py
│   └── ...
│
└── paper/
    └── IEEE_paper.tex          # IEEE conference paper
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run only fast tests
python -m pytest tests/ -m "not slow"

# Run specific test file
python -m pytest tests/test_ethical_filter.py -v
```

---

## 📚 References

1. A. Radford et al., "Language Models are Unsupervised Multitask Learners," OpenAI, 2019.
2. T. Brown et al., "Language Models are Few-Shot Learners," NeurIPS, 2020.
3. L. Hanu and Unitary Team, "Detoxify," GitHub, 2020.
4. E. Bender et al., "On the Dangers of Stochastic Parrots," FAccT, 2021.
5. T. Bolukbasi et al., "Man is to Computer Programmer as Woman is to Homemaker?" NeurIPS, 2016.
6. T. Wolf et al., "Transformers: State-of-the-Art NLP," EMNLP, 2020.
7. J. Vig, "A Multiscale Visualization of Attention in the Transformer Model," ACL Demo, 2019.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Dr. Sc. Mërgim H. HOTI for guidance and support
- Faculty of Electrical and Computer Engineering
- Hugging Face for the Transformers library
- OpenAI for the GPT-2 model

---

<p align="center">
  <i>Faculty of Electrical and Computer Engineering - NLP 2025</i>
  <br>
  <i>Built with ❤️ for Ethical AI in Storytelling</i>
</p>
