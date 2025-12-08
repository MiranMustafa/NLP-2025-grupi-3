"""
Explainability Module for AI Storytelling.

This module provides interpretability tools for understanding
how the story generation model makes decisions.

Features:
- Attention visualization
- Token importance analysis
- Generation probability analysis
- Saliency maps for input tokens
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Visualize attention patterns in transformer models.
    
    Helps understand which parts of the input the model focuses on
    when generating each new token.
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cpu"
    ):
        """
        Initialize the attention visualizer.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            device: Device to run computations on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def get_attention_weights(
        self,
        text: str,
        layer: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract attention weights from the model.
        
        Args:
            text: Input text
            layer: Specific layer to extract (None for all layers)
            
        Returns:
            Dictionary with attention weights and metadata
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Stack all attention layers
        attentions = torch.stack(outputs.attentions)  # (layers, batch, heads, seq, seq)
        
        if layer is not None:
            attentions = attentions[layer:layer+1]
            
        # Average over heads
        avg_attention = attentions.mean(dim=2)  # (layers, batch, seq, seq)
        
        return {
            "tokens": tokens,
            "attention_weights": avg_attention.cpu().numpy(),
            "num_layers": len(outputs.attentions),
            "num_heads": outputs.attentions[0].shape[1],
            "sequence_length": len(tokens)
        }
    
    def plot_attention_heatmap(
        self,
        text: str,
        layer: int = -1,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a heatmap of attention weights.
        
        Args:
            text: Input text
            layer: Layer to visualize (-1 for last layer)
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        attention_data = self.get_attention_weights(text)
        tokens = attention_data["tokens"]
        
        # Get specified layer
        if layer == -1:
            layer = attention_data["num_layers"] - 1
        
        attention_matrix = attention_data["attention_weights"][layer, 0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            attention_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="YlOrRd",
            ax=ax,
            square=True,
            cbar_kws={"label": "Attention Weight"}
        )
        
        ax.set_title(f"Attention Weights (Layer {layer + 1})", fontsize=14)
        ax.set_xlabel("Key Tokens", fontsize=12)
        ax.set_ylabel("Query Tokens", fontsize=12)
        
        # Rotate labels
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_attention_flow(
        self,
        text: str,
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize attention flow across layers.
        
        Shows how attention patterns evolve through the model layers.
        
        Args:
            text: Input text
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        attention_data = self.get_attention_weights(text)
        tokens = attention_data["tokens"]
        num_layers = attention_data["num_layers"]
        
        # Get average attention for each layer (summed over query positions)
        layer_attention = []
        for layer_idx in range(num_layers):
            # Sum attention that each token receives from all positions
            received_attention = attention_data["attention_weights"][layer_idx, 0].sum(axis=0)
            layer_attention.append(received_attention)
        
        layer_attention = np.array(layer_attention)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            layer_attention,
            xticklabels=tokens,
            yticklabels=[f"Layer {i+1}" for i in range(num_layers)],
            cmap="viridis",
            ax=ax,
            cbar_kws={"label": "Total Attention Received"}
        )
        
        ax.set_title("Attention Flow Across Layers", fontsize=14)
        ax.set_xlabel("Tokens", fontsize=12)
        ax.set_ylabel("Layer", fontsize=12)
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            
        return fig


class TokenImportanceAnalyzer:
    """
    Analyze importance of input tokens for generation.
    
    Uses gradient-based methods to determine which input tokens
    most influence the model's output.
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cpu"
    ):
        """
        Initialize the token importance analyzer.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            device: Device to run computations on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def compute_gradient_importance(
        self,
        text: str,
        target_position: int = -1
    ) -> Dict[str, Any]:
        """
        Compute token importance using gradient analysis.
        
        Args:
            text: Input text
            target_position: Position to analyze (-1 for last token)
            
        Returns:
            Dictionary with importance scores
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Get embeddings with gradients
        embeddings = self.model.transformer.wte(inputs["input_ids"])
        embeddings.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(inputs_embeds=embeddings)
        logits = outputs.logits
        
        # Get loss for target position
        if target_position == -1:
            target_position = logits.shape[1] - 1
            
        # Use max logit as target
        target_logit = logits[0, target_position].max()
        
        # Backward pass
        target_logit.backward()
        
        # Get gradient importance (L2 norm of gradients)
        gradients = embeddings.grad[0]
        importance = torch.norm(gradients, dim=-1).detach().cpu().numpy()
        
        # Normalize
        importance = importance / importance.sum()
        
        return {
            "tokens": tokens,
            "importance_scores": importance.tolist(),
            "target_position": target_position,
            "most_important": [
                {"token": tokens[i], "score": float(importance[i])}
                for i in np.argsort(importance)[::-1][:5]
            ]
        }
    
    def compute_leave_one_out_importance(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Compute token importance using leave-one-out analysis.
        
        Measures how much the output changes when each token is removed.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with importance scores
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(**inputs)
            baseline_logits = baseline_output.logits[0, -1]
            baseline_probs = torch.softmax(baseline_logits, dim=-1)
        
        importance_scores = []
        
        for i in range(len(tokens)):
            # Create input with token i masked
            masked_ids = inputs["input_ids"].clone()
            masked_ids[0, i] = self.tokenizer.pad_token_id
            
            with torch.no_grad():
                masked_output = self.model(input_ids=masked_ids)
                masked_logits = masked_output.logits[0, -1]
                masked_probs = torch.softmax(masked_logits, dim=-1)
            
            # Compute KL divergence between baseline and masked
            kl_div = torch.sum(
                baseline_probs * (torch.log(baseline_probs + 1e-10) - torch.log(masked_probs + 1e-10))
            ).item()
            
            importance_scores.append(abs(kl_div))
        
        # Normalize
        total = sum(importance_scores)
        if total > 0:
            importance_scores = [s / total for s in importance_scores]
        
        return {
            "tokens": tokens,
            "importance_scores": importance_scores,
            "method": "leave_one_out",
            "most_important": [
                {"token": tokens[i], "score": importance_scores[i]}
                for i in np.argsort(importance_scores)[::-1][:5]
            ]
        }
    
    def plot_token_importance(
        self,
        importance_data: Dict[str, Any],
        figsize: Tuple[int, int] = (12, 4),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a bar chart of token importance.
        
        Args:
            importance_data: Output from importance analysis
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        tokens = importance_data["tokens"]
        scores = importance_data["importance_scores"]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.RdYlGn(np.array(scores) / max(scores))
        
        bars = ax.bar(range(len(tokens)), scores, color=colors)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        
        ax.set_xlabel("Tokens", fontsize=12)
        ax.set_ylabel("Importance Score", fontsize=12)
        ax.set_title("Token Importance Analysis", fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            
        return fig


class GenerationExplainer:
    """
    Explain the story generation process step by step.
    
    Provides insights into why specific tokens were chosen
    during generation.
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cpu"
    ):
        """
        Initialize the generation explainer.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            device: Device to run computations on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def explain_next_token(
        self,
        text: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Explain the next token prediction.
        
        Args:
            text: Input text
            top_k: Number of top candidates to show
            
        Returns:
            Dictionary with prediction explanation
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1]
            probs = torch.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k)
        top_tokens = self.tokenizer.convert_ids_to_tokens(top_indices.tolist())
        
        candidates = [
            {
                "token": token,
                "token_decoded": self.tokenizer.decode([idx]),
                "probability": prob.item(),
                "log_probability": np.log(prob.item())
            }
            for token, idx, prob in zip(top_tokens, top_indices.tolist(), top_probs)
        ]
        
        # Compute entropy (uncertainty)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy
        
        return {
            "input_text": text,
            "top_candidates": candidates,
            "predicted_token": candidates[0]["token_decoded"],
            "prediction_confidence": candidates[0]["probability"],
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "uncertainty_level": self._get_uncertainty_level(normalized_entropy)
        }
    
    def _get_uncertainty_level(self, normalized_entropy: float) -> str:
        """Categorize uncertainty level."""
        if normalized_entropy < 0.3:
            return "low"
        elif normalized_entropy < 0.6:
            return "medium"
        else:
            return "high"
    
    def trace_generation(
        self,
        prompt: str,
        num_tokens: int = 10,
        temperature: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Trace the generation process token by token.
        
        Args:
            prompt: Starting prompt
            num_tokens: Number of tokens to generate and trace
            temperature: Sampling temperature
            
        Returns:
            List of explanations for each generated token
        """
        current_text = prompt
        trace = []
        
        for step in range(num_tokens):
            # Explain next token
            explanation = self.explain_next_token(current_text)
            
            # Sample next token (using temperature)
            inputs = self.tokenizer(current_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1).item()
            
            next_token = self.tokenizer.decode([next_token_id])
            
            trace.append({
                "step": step + 1,
                "current_text": current_text,
                "generated_token": next_token,
                "token_probability": probs[next_token_id].item(),
                "top_alternatives": explanation["top_candidates"][:5],
                "uncertainty": explanation["uncertainty_level"]
            })
            
            current_text += next_token
        
        return trace
    
    def visualize_generation_trace(
        self,
        trace: List[Dict[str, Any]],
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize the generation trace.
        
        Args:
            trace: Output from trace_generation
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        steps = [t["step"] for t in trace]
        probabilities = [t["token_probability"] for t in trace]
        tokens = [t["generated_token"] for t in trace]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color by uncertainty
        colors = []
        for t in trace:
            if t["uncertainty"] == "low":
                colors.append("#2ecc71")
            elif t["uncertainty"] == "medium":
                colors.append("#f39c12")
            else:
                colors.append("#e74c3c")
        
        bars = ax.bar(steps, probabilities, color=colors)
        
        # Add token labels
        for bar, token in zip(bars, tokens):
            height = bar.get_height()
            ax.annotate(
                repr(token)[1:-1],  # Remove quotes but show escapes
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=45
            )
        
        ax.set_xlabel("Generation Step", fontsize=12)
        ax.set_ylabel("Token Probability", fontsize=12)
        ax.set_title("Story Generation Trace", fontsize=14)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2ecc71", label="Low Uncertainty"),
            Patch(facecolor="#f39c12", label="Medium Uncertainty"),
            Patch(facecolor="#e74c3c", label="High Uncertainty"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            
        return fig


def create_explainability_report(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = "cpu",
    output_dir: str = "."
) -> Dict[str, Any]:
    """
    Create a comprehensive explainability report for a text.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Text to analyze
        device: Device to use
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary with all analysis results
    """
    # Initialize analyzers
    attention_viz = AttentionVisualizer(model, tokenizer, device)
    token_analyzer = TokenImportanceAnalyzer(model, tokenizer, device)
    gen_explainer = GenerationExplainer(model, tokenizer, device)
    
    report = {
        "input_text": text,
        "analyses": {}
    }
    
    # Attention analysis
    try:
        attention_data = attention_viz.get_attention_weights(text)
        report["analyses"]["attention"] = {
            "num_layers": attention_data["num_layers"],
            "num_heads": attention_data["num_heads"],
            "sequence_length": attention_data["sequence_length"]
        }
    except Exception as e:
        logger.warning(f"Attention analysis failed: {e}")
    
    # Token importance
    try:
        importance_data = token_analyzer.compute_leave_one_out_importance(text)
        report["analyses"]["token_importance"] = {
            "most_important_tokens": importance_data["most_important"]
        }
    except Exception as e:
        logger.warning(f"Token importance analysis failed: {e}")
    
    # Next token prediction
    try:
        next_token = gen_explainer.explain_next_token(text)
        report["analyses"]["next_token_prediction"] = {
            "predicted": next_token["predicted_token"],
            "confidence": next_token["prediction_confidence"],
            "uncertainty": next_token["uncertainty_level"]
        }
    except Exception as e:
        logger.warning(f"Next token analysis failed: {e}")
    
    return report

