# Exported Results - Ethical AI Storyteller

## Export Date
2025-12-13 15:12:35

## Directory Structure

```
results/
├── datasets/
│   ├── training_data.json      # Training dataset (JSON format)
│   └── training_data.csv       # Training dataset (CSV format)
├── generated_stories/
│   ├── generated_stories.json  # Generated stories with analysis
│   └── generated_stories.csv   # Generated stories (CSV format)
└── evaluation/
    ├── evaluation_summary.json # Evaluation metrics
    ├── evaluation_report.txt   # Human-readable report
    └── finetuned_models.json   # Fine-tuned model information
```

## Dataset Description

### Training Data
- 50 curated story examples across 6 genres
- Fields: id, genre, prompt, story

### Generated Stories
- AI-generated stories with safety and bias analysis
- Fields: id, genre, prompt, story, is_safe, content_rating, confidence, has_bias

### Evaluation Metrics
- Safe content ratio
- Bias-free ratio
- Content rating distribution
- Genre distribution

## Usage

These datasets can be used for:
1. Reproducibility verification
2. Further analysis
3. Model comparison studies

## Project Information
- Course: Natural Language Processing (NLP) 2025
- Institution: Faculty of Electrical and Computer Engineering, University of Prishtina
- Team: NLP-2025-grupi-3
