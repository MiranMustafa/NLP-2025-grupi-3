"""
Gradio Web Interface for Ethical AI Storyteller.

This module provides an interactive web interface for the storytelling system.
Features:
- Real-time story generation
- Genre selection
- Parameter adjustment
- Ethical analysis display
- Explainability visualization

Compatibility: Works with Gradio 3.x and 4.x
"""

import gradio as gr
import sys
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Check Gradio version for compatibility
GRADIO_VERSION = tuple(map(int, gr.__version__.split('.')[:2]))
print(f"📦 Gradio version: {gr.__version__}")

from src.storyteller import EthicalStoryTeller
from src.ethical_filter import EthicalFilter

# Global storyteller instance (initialized on first use)
storyteller = None
safety_filter = None
current_model_name = None

# Available base models
BASE_MODELS = {
    "gpt2": "GPT-2 (124M) - Fast, good quality",
    "gpt2-medium": "GPT-2 Medium (355M) - Better quality",
    "gpt2-large": "GPT-2 Large (774M) - Best quality, slower",
    "distilgpt2": "DistilGPT-2 (82M) - Fastest inference",
}

# Fine-tuned model paths
FINETUNED_MODELS = {
    "models/finetuned-writing_prompts/final": "Fine-tuned: Writing Prompts (Creative)",
    "models/finetuned-tiny_stories/final": "Fine-tuned: Tiny Stories (Simple)",
    "models/finetuned-fairy_tales/final": "Fine-tuned: Fairy Tales (Classic)",
    "models/finetuned-storyteller/final": "Fine-tuned: Local Dataset (Custom)",
}


def get_available_models():
    """Get list of available models (base + existing fine-tuned)."""
    models = {}
    
    # Add base models
    for model_id, description in BASE_MODELS.items():
        models[model_id] = description
    
    # Add fine-tuned models that exist
    for model_path, description in FINETUNED_MODELS.items():
        if os.path.exists(model_path):
            models[model_path] = description
    
    return models


def get_storyteller(model_name: str = None):
    """Get or initialize the storyteller with the specified model."""
    global storyteller, current_model_name
    
    # Default model selection
    if model_name is None:
        # Try to use a fine-tuned model by default
        for path in FINETUNED_MODELS.keys():
            if os.path.exists(path):
                model_name = path
                break
        if model_name is None:
            model_name = "gpt2"
    
    # Reinitialize if model changed
    if storyteller is None or current_model_name != model_name:
        print(f"🔄 Loading model: {model_name}... (this may take a moment)")
        
        storyteller = EthicalStoryTeller(
            model_name=model_name,
            enable_ethical_filter=True,
            enable_bias_detection=True,
            strict_mode=True
        )
        current_model_name = model_name
        print(f"✅ Model loaded: {model_name}")
    
    return storyteller


def get_safety_filter():
    """
    Lazily create and cache the ethical filter.
    
    Uses Detoxify if available; otherwise falls back gracefully so the UI keeps working.
    """
    global safety_filter
    if safety_filter is None:
        try:
            safety_filter = EthicalFilter(use_detoxify=True)
        except Exception as exc:  # Detoxify or torch may be unavailable
            print(f"⚠️ Detoxify unavailable ({exc}); using regex-only filter.")
            safety_filter = EthicalFilter(use_detoxify=False)
    return safety_filter


def generate_story(
    prompt: str,
    genre: str,
    model_choice: str,
    max_length: int,
    temperature: float,
    top_p: float,
    top_k: int
):
    """Generate a story with the given parameters."""
    try:
        st = get_storyteller(model_choice)
        
        # Handle genre selection
        genre_value = genre if genre != "None (Custom Prompt)" else None
        
        result = st.generate_story(
            prompt=prompt,
            genre=genre_value,
            max_length=int(max_length),
            temperature=temperature,
            top_p=top_p,
            top_k=int(top_k),
            ensure_safe=True
        )
        
        # Format the output
        story_output = result.story
        
        # Create analysis summary
        analysis = f"""
### 📊 Content Analysis
- **Safety Rating**: {'✅ Safe' if result.is_safe else '⚠️ Needs Review'}
- **Content Rating**: {result.content_rating.upper()}
- **Confidence**: {result.ethical_analysis['confidence']:.1%}

### ⚙️ Generation Settings
- **Model**: {result.generation_params['model']}
- **Temperature**: {result.generation_params['temperature']}
- **Top-p**: {result.generation_params['top_p']}
- **Top-k**: {result.generation_params['top_k']}
"""
        
        if result.warnings:
            analysis += "\n### ⚠️ Warnings\n"
            for warning in result.warnings:
                analysis += f"- {warning}\n"
        
        if result.bias_analysis and result.bias_analysis.get('has_bias'):
            analysis += "\n### 🔍 Bias Detection\n"
            for rec in result.bias_analysis.get('recommendations', []):
                analysis += f"- {rec}\n"
        
        return story_output, analysis
        
    except Exception as e:
        return f"Error generating story: {str(e)}", "Error occurred during generation."


def continue_story(
    current_story: str,
    continuation_length: int,
    temperature: float
):
    """Continue an existing story."""
    try:
        st = get_storyteller()
        
        result = st.continue_story(
            story_so_far=current_story,
            continuation_length=int(continuation_length),
            temperature=temperature
        )
        
        return result.story
        
    except Exception as e:
        return f"Error continuing story: {str(e)}"


def explain_next_token(text: str):
    """Explain what the model predicts next."""
    try:
        st = get_storyteller()
        
        explanation = st.explain_generation(text, num_trace_tokens=5)
        next_token = explanation['next_token']
        
        output = f"""
### 🔮 Next Token Prediction

**Input**: "{text}"

**Predicted Next Token**: `{next_token['predicted_token']}`
**Confidence**: {next_token['prediction_confidence']:.2%}
**Uncertainty Level**: {next_token['uncertainty_level'].upper()}

### 📊 Top 5 Candidates:
"""
        
        for i, candidate in enumerate(next_token['top_candidates'][:5], 1):
            bar_length = int(candidate['probability'] * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            output += f"\n{i}. `{candidate['token_decoded']}` {bar} {candidate['probability']:.1%}"
        
        return output
        
    except Exception as e:
        return f"Error explaining: {str(e)}"


def visualize_attention(text: str, layer: int):
    """Generate an attention heatmap figure."""
    try:
        st = get_storyteller()
        fig = st.visualize_attention(text, layer=layer)
        return fig
    except Exception as e:
        # Return None so Gradio can show a friendly message elsewhere
        print(f"Attention visualization failed: {e}")
        return None


def visualize_token_importance(text: str):
    """Generate a token-importance bar chart figure."""
    try:
        st = get_storyteller()
        importance = st.analyze_token_importance(text)
        fig = st.token_analyzer.plot_token_importance(importance)
        return fig
    except Exception as e:
        print(f"Token importance visualization failed: {e}")
        return None


def check_content_safety(text: str):
    """Check content for safety and ethical concerns."""
    try:
        filter_module = get_safety_filter()
        result = filter_module.filter_content(text)
        
        output = f"""
### 🔒 Safety Analysis

**Overall Rating**: {result.rating.value.upper()}
**Is Safe**: {'✅ Yes' if result.is_safe else '⚠️ Needs Review'}
**Confidence**: {result.confidence:.1%}

### 🧪 Toxicity Scores:
"""
        
        if result.toxicity_scores:
            for metric, score in result.toxicity_scores.items():
                bar_length = int(score * 30)
                color = "🟢" if score < 0.3 else "🟡" if score < 0.6 else "🔴"
                output += f"\n{color} **{metric}**: {score:.2%}"
        else:
            output += "\n*Toxicity analysis not available*"
        
        if result.flagged_content:
            output += "\n\n### ⚠️ Flagged Content:\n"
            for flag in result.flagged_content:
                output += f"- {flag}\n"
        
        if result.suggestions:
            output += "\n### 💡 Suggestions:\n"
            for suggestion in result.suggestions:
                output += f"- {suggestion}\n"
        
        return output
        
    except Exception as e:
        return f"Error checking content: {str(e)}"


# Create the Gradio interface
def create_interface():
    """Create the Gradio web interface."""
    
    # Custom CSS for styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .story-output {
        font-size: 1.1em;
        line-height: 1.6;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
    }
    """
    
    with gr.Blocks() as demo:
        
        gr.Markdown("""
        # 🌟 Ethical AI Storyteller
        
        Generate creative stories with built-in ethical constraints and explainability.
        
        **Features:**
        - 📖 Multiple genre support
        - 🔒 Content safety filtering
        - 🔍 Bias detection
        - 💡 Explainable AI
        
        ---
        """)
        
        with gr.Tabs():
            # Tab 1: Story Generation
            with gr.TabItem("📖 Generate Story"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Model selection dropdown
                        available_models = get_available_models()
                        model_choices = list(available_models.keys())
                        model_labels = [f"{available_models[m]}" for m in model_choices]
                        
                        # Default to first fine-tuned model if available
                        default_model = model_choices[0]
                        for m in model_choices:
                            if m.startswith("models/"):
                                default_model = m
                                break
                        
                        model_dropdown = gr.Dropdown(
                            choices=model_choices,
                            value=default_model,
                            label="🤖 Select Model",
                            info="Choose base GPT-2 or fine-tuned models"
                        )
                        
                        genre_dropdown = gr.Dropdown(
                            choices=[
                                "None (Custom Prompt)",
                                "fantasy",
                                "sci-fi",
                                "mystery",
                                "romance",
                                "horror",
                                "adventure",
                                "fairy_tale",
                                "historical"
                            ],
                            value="fantasy",
                            label="🎭 Select Genre"
                        )
                        
                        prompt_input = gr.Textbox(
                            label="✍️ Your Prompt (optional - adds to genre prompt)",
                            placeholder="Enter your story starting point...",
                            lines=3
                        )
                        
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            max_length = gr.Slider(
                                minimum=50,
                                maximum=500,
                                value=200,
                                step=10,
                                label="📏 Max Length (tokens)"
                            )
                            
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=1.5,
                                value=0.8,
                                step=0.1,
                                label="🌡️ Temperature (creativity)"
                            )
                            
                            top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.95,
                                step=0.05,
                                label="🎯 Top-p (nucleus sampling)"
                            )
                            
                            top_k = gr.Slider(
                                minimum=1,
                                maximum=100,
                                value=50,
                                step=5,
                                label="🔢 Top-k"
                            )
                        
                        generate_btn = gr.Button(
                            "✨ Generate Story",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        story_output = gr.Textbox(
                            label="📜 Generated Story",
                            lines=12,
                            max_lines=20
                        )
                        
                        analysis_output = gr.Markdown(
                            label="📊 Analysis"
                        )
                
                generate_btn.click(
                    fn=generate_story,
                    inputs=[
                        prompt_input,
                        genre_dropdown,
                        model_dropdown,
                        max_length,
                        temperature,
                        top_p,
                        top_k
                    ],
                    outputs=[story_output, analysis_output]
                )
            
            # Tab 2: Continue Story
            with gr.TabItem("📝 Continue Story"):
                gr.Markdown("### Continue an existing story")
                
                current_story_input = gr.Textbox(
                    label="📖 Current Story",
                    placeholder="Paste your story here to continue it...",
                    lines=8
                )
                
                with gr.Row():
                    cont_length = gr.Slider(
                        minimum=20,
                        maximum=200,
                        value=100,
                        label="📏 Continuation Length"
                    )
                    cont_temp = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.8,
                        label="🌡️ Temperature"
                    )
                
                continue_btn = gr.Button("➡️ Continue Story", variant="primary")
                
                continued_output = gr.Textbox(
                    label="📜 Continued Story",
                    lines=10
                )
                
                continue_btn.click(
                    fn=continue_story,
                    inputs=[current_story_input, cont_length, cont_temp],
                    outputs=continued_output
                )
            
            # Tab 3: Explainability
            with gr.TabItem("🔍 Explain"):
                gr.Markdown("### Understand Model Predictions")
                
                explain_input = gr.Textbox(
                    label="📝 Enter Text",
                    placeholder="Enter text to see what the model predicts next...",
                    lines=3
                )

                explain_layer = gr.Slider(
                    minimum=-1,
                    maximum=23,
                    value=-1,
                    step=1,
                    label="Transformer Layer (-1 = last layer)"
                )
                
                explain_btn = gr.Button("🔮 Run Explainability", variant="primary")
                
                explain_output = gr.Markdown(
                    label="🔍 Explanation"
                )

                attention_plot = gr.Plot(label="🕸️ Attention Heatmap")
                importance_plot = gr.Plot(label="📊 Token Importance")
                
                explain_btn.click(
                    fn=lambda text, layer: (
                        explain_next_token(text),
                        visualize_attention(text, int(layer)),
                        visualize_token_importance(text)
                    ),
                    inputs=[explain_input, explain_layer],
                    outputs=[explain_output, attention_plot, importance_plot]
                )
            
            # Tab 4: Safety Check
            with gr.TabItem("🔒 Safety Check"):
                gr.Markdown("### Check Content Safety")
                
                safety_input = gr.Textbox(
                    label="📝 Content to Check",
                    placeholder="Enter text to analyze for safety...",
                    lines=5
                )
                
                safety_btn = gr.Button("🔒 Check Safety", variant="primary")
                
                safety_output = gr.Markdown(
                    label="🔍 Safety Analysis"
                )
                
                safety_btn.click(
                    fn=check_content_safety,
                    inputs=safety_input,
                    outputs=safety_output
                )
        
        # Model info section
        with gr.Accordion("📚 Available Models", open=False):
            model_info = "### Base Models\n"
            for model_id, desc in BASE_MODELS.items():
                model_info += f"- **{model_id}**: {desc}\n"
            
            model_info += "\n### Fine-tuned Models\n"
            for model_path, desc in FINETUNED_MODELS.items():
                status = "✅ Available" if os.path.exists(model_path) else "❌ Not found"
                model_info += f"- **{desc}**: {status}\n"
            
            gr.Markdown(model_info)
        
        gr.Markdown("""
        ---
        
        ### 📚 About This Project
        
        This is an NLP course project demonstrating ethical AI storytelling with:
        - **GPT-2** pretrained language models (base and fine-tuned variants)
        - **Detoxify** for toxicity detection
        - **Custom bias detection** algorithms
        - **Attention visualization** for explainability
        
        *Faculty of Electrical and Computer Engineering - NLP 2025*
        """)
    
    return demo


if __name__ == "__main__":
    print("🚀 Starting Ethical AI Storyteller...")
    print("📝 Note: The model will be loaded on first use.")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
