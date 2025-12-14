#!/usr/bin/env python3
"""
Export Project Results and Datasets for Submission.

This script exports:
1. Generated stories with safety/bias analysis
2. Evaluation metrics and results
3. Training data used for fine-tuning
4. Model performance statistics

Output: results/ directory with JSON and CSV files
"""

import os
import sys
import json
import csv
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def create_output_dir():
    """Create output directory for results."""
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (output_dir / "generated_stories").mkdir(exist_ok=True)
    (output_dir / "evaluation").mkdir(exist_ok=True)
    (output_dir / "datasets").mkdir(exist_ok=True)
    
    return output_dir


def export_sample_dataset(output_dir: Path):
    """Export the sample training dataset."""
    print("📊 Exporting training dataset...")
    
    # Copy sample stories
    src_file = Path("data/sample_stories.json")
    if src_file.exists():
        import shutil
        shutil.copy(src_file, output_dir / "datasets" / "training_data.json")
        
        # Also create CSV version
        with open(src_file, 'r') as f:
            data = json.load(f)
        
        csv_path = output_dir / "datasets" / "training_data.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'genre', 'prompt', 'story'])
            writer.writeheader()
            for item in data:
                writer.writerow(item)
        
        print(f"  ✓ Exported {len(data)} training examples")
        return len(data)
    return 0


def generate_and_export_stories(output_dir: Path, num_stories: int = 20):
    """Generate sample stories and export with analysis."""
    print(f"📖 Generating {num_stories} sample stories with analysis...")
    
    try:
        from src.storyteller import EthicalStoryTeller
        
        # Check for fine-tuned model
        model_paths = [
            "models/finetuned-writing_prompts/final",
            "models/finetuned-tiny_stories/final",
            "models/finetuned-storyteller/final",
            "gpt2"
        ]
        
        model_name = "gpt2"
        for path in model_paths:
            if os.path.exists(path):
                model_name = path
                break
        
        print(f"  Using model: {model_name}")
        
        storyteller = EthicalStoryTeller(
            model_name=model_name,
            enable_ethical_filter=True,
            enable_bias_detection=True
        )
        
        genres = ["fantasy", "sci-fi", "mystery", "adventure", "romance", "horror"]
        results = []
        
        for i in range(num_stories):
            genre = genres[i % len(genres)]
            
            result = storyteller.generate_story(
                prompt="",
                genre=genre,
                max_length=150,
                temperature=0.8
            )
            
            story_data = {
                "id": f"generated_{i+1}",
                "genre": genre,
                "prompt": result.prompt,
                "story": result.story,
                "is_safe": result.is_safe,
                "content_rating": result.content_rating,
                "confidence": result.ethical_analysis.get('confidence', 0),
                "has_bias": result.bias_analysis.get('has_bias', False) if result.bias_analysis else False,
                "model": model_name,
                "timestamp": datetime.now().isoformat()
            }
            results.append(story_data)
            print(f"  ✓ Generated story {i+1}/{num_stories} [{genre}]")
        
        # Save as JSON
        json_path = output_dir / "generated_stories" / "generated_stories.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save as CSV
        csv_path = output_dir / "generated_stories" / "generated_stories.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['id', 'genre', 'prompt', 'story', 'is_safe', 'content_rating', 
                         'confidence', 'has_bias', 'model', 'timestamp']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"  ✓ Exported {len(results)} generated stories")
        return results
        
    except Exception as e:
        print(f"  ⚠ Could not generate stories: {e}")
        return []


def export_evaluation_results(output_dir: Path, generated_stories: list):
    """Export evaluation metrics and statistics."""
    print("📈 Exporting evaluation results...")
    
    if not generated_stories:
        print("  ⚠ No generated stories to evaluate")
        return
    
    # Calculate statistics
    total = len(generated_stories)
    safe_count = sum(1 for s in generated_stories if s.get('is_safe', False))
    bias_count = sum(1 for s in generated_stories if s.get('has_bias', False))
    
    # Genre distribution
    genre_counts = {}
    for s in generated_stories:
        genre = s.get('genre', 'unknown')
        genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    # Rating distribution
    rating_counts = {}
    for s in generated_stories:
        rating = s.get('content_rating', 'unknown')
        rating_counts[rating] = rating_counts.get(rating, 0) + 1
    
    evaluation_summary = {
        "metadata": {
            "export_date": datetime.now().isoformat(),
            "total_samples": total,
            "model_used": generated_stories[0].get('model', 'unknown') if generated_stories else 'unknown'
        },
        "safety_metrics": {
            "safe_ratio": safe_count / total if total > 0 else 0,
            "safe_count": safe_count,
            "unsafe_count": total - safe_count
        },
        "bias_metrics": {
            "bias_ratio": bias_count / total if total > 0 else 0,
            "biased_count": bias_count,
            "unbiased_count": total - bias_count
        },
        "content_ratings": rating_counts,
        "genre_distribution": genre_counts,
        "average_confidence": sum(s.get('confidence', 0) for s in generated_stories) / total if total > 0 else 0
    }
    
    # Save evaluation summary
    eval_path = output_dir / "evaluation" / "evaluation_summary.json"
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    # Create a readable report
    report_path = output_dir / "evaluation" / "evaluation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ETHICAL AI STORYTELLER - EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {evaluation_summary['metadata']['model_used']}\n")
        f.write(f"Total Samples: {total}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("SAFETY METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Safe Content Rate: {evaluation_summary['safety_metrics']['safe_ratio']:.1%}\n")
        f.write(f"Safe Stories: {safe_count}/{total}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("BIAS METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Bias-Free Rate: {1 - evaluation_summary['bias_metrics']['bias_ratio']:.1%}\n")
        f.write(f"Biased Stories: {bias_count}/{total}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("CONTENT RATINGS\n")
        f.write("-" * 40 + "\n")
        for rating, count in rating_counts.items():
            f.write(f"  {rating}: {count} ({count/total:.1%})\n")
        
        f.write("\n" + "-" * 40 + "\n")
        f.write("GENRE DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        for genre, count in genre_counts.items():
            f.write(f"  {genre}: {count}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 60 + "\n")
    
    print(f"  ✓ Evaluation summary exported")
    print(f"  ✓ Safe content rate: {evaluation_summary['safety_metrics']['safe_ratio']:.1%}")
    print(f"  ✓ Bias-free rate: {1 - evaluation_summary['bias_metrics']['bias_ratio']:.1%}")
    
    return evaluation_summary


def export_model_info(output_dir: Path):
    """Export information about fine-tuned models."""
    print("🤖 Exporting model information...")
    
    models_dir = Path("models")
    model_info = []
    
    if models_dir.exists():
        for model_path in models_dir.iterdir():
            if model_path.is_dir():
                final_path = model_path / "final"
                if final_path.exists():
                    config_path = final_path / "config.json"
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        
                        model_info.append({
                            "name": model_path.name,
                            "path": str(final_path),
                            "architecture": config.get("architectures", ["unknown"])[0],
                            "vocab_size": config.get("vocab_size", 0),
                            "n_layer": config.get("n_layer", 0),
                            "n_head": config.get("n_head", 0),
                        })
    
    if model_info:
        info_path = output_dir / "evaluation" / "finetuned_models.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2)
        print(f"  ✓ Exported info for {len(model_info)} fine-tuned models")
    else:
        print("  ⚠ No fine-tuned models found")
    
    return model_info


def create_submission_readme(output_dir: Path):
    """Create a README for the exported results."""
    readme_content = f"""# Exported Results - Ethical AI Storyteller

## Export Date
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

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
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("📝 Created results README")


def main():
    """Main export function."""
    print("=" * 60)
    print("🚀 ETHICAL AI STORYTELLER - RESULTS EXPORT")
    print("=" * 60)
    print()
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"📁 Output directory: {output_dir.absolute()}\n")
    
    # Export training dataset
    export_sample_dataset(output_dir)
    print()
    
    # Generate and export stories
    generated_stories = generate_and_export_stories(output_dir, num_stories=20)
    print()
    
    # Export evaluation results
    export_evaluation_results(output_dir, generated_stories)
    print()
    
    # Export model info
    export_model_info(output_dir)
    print()
    
    # Create README
    create_submission_readme(output_dir)
    print()
    
    print("=" * 60)
    print("✅ EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nFiles to submit:")
    print("  - results/datasets/training_data.json")
    print("  - results/datasets/training_data.csv")
    print("  - results/generated_stories/generated_stories.json")
    print("  - results/generated_stories/generated_stories.csv")
    print("  - results/evaluation/evaluation_summary.json")
    print("  - results/evaluation/evaluation_report.txt")
    print()


if __name__ == "__main__":
    main()

