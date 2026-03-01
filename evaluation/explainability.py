"""
MindGuard Explainability
========================
SHAP and LIME-based word-level explanations for model predictions.
Shows counselors WHICH words drove the severity classification.

This transparency is critical for:
1. Building trust with counselors
2. Catching model errors
3. Supporting human decision-making
4. Research documentation
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')


# ============================================================
# SHAP Explainer
# ============================================================

class MindGuardExplainer:
    """
    SHAP-based explainability for MindGuard predictions.
    
    Shows which words contributed most to each severity classification.
    Uses partition SHAP for transformer models (more efficient than KernelSHAP).
    """
    
    LABEL_MAP = {
        0: "No Concern",
        1: "Mild",
        2: "Moderate", 
        3: "Severe Crisis"
    }
    
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        method: str = "shap",
        num_features: int = 15,
        max_evals: int = 500,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.method = method
        self.num_features = num_features
        self.max_evals = max_evals
        
        self.model.eval()
    
    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities for a list of texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        all_probs = []
        self.model.eval()
        
        with torch.no_grad():
            for text in texts:
                enc = self.tokenizer(
                    text, return_tensors='pt', max_length=256,
                    padding='max_length', truncation=True
                )
                outputs = self.model(
                    enc['input_ids'].to(self.device),
                    enc['attention_mask'].to(self.device)
                )
                probs = torch.softmax(outputs['logits'], dim=-1)
                all_probs.append(probs[0].cpu().numpy())
        
        return np.array(all_probs)
    
    def explain_with_shap(self, text: str) -> Dict:
        """
        Generate SHAP explanation for a single text.
        
        Returns word-level importance scores showing which words
        pushed the prediction toward each severity level.
        """
        try:
            import shap
        except ImportError:
            return self._fallback_attention_explanation(text)
        
        # Create SHAP explainer
        explainer = shap.Explainer(
            self._predict_proba,
            self.tokenizer,
            output_names=list(self.LABEL_MAP.values()),
        )
        
        # Generate SHAP values
        shap_values = explainer([text], max_evals=self.max_evals)
        
        # Get the predicted class
        probs = self._predict_proba(text)
        predicted_class = int(np.argmax(probs[0]))
        
        # Extract word-level importances for predicted class
        if hasattr(shap_values, 'values') and shap_values.values is not None:
            values = shap_values.values[0]
            if len(values.shape) > 1:
                # Multi-class: get values for predicted class
                class_values = values[:, predicted_class]
            else:
                class_values = values
            
            # Get corresponding words/tokens
            if hasattr(shap_values, 'data'):
                words = shap_values.data[0] if hasattr(shap_values.data[0], '__iter__') else [text]
            else:
                words = text.split()
            
            # Create word importance pairs
            word_importances = []
            for i, (word, importance) in enumerate(zip(words, class_values)):
                if isinstance(word, str) and word.strip():
                    word_importances.append({
                        'word': word.strip(),
                        'importance': float(importance),
                        'direction': 'risk' if importance > 0 else 'protective',
                    })
            
            # Sort by absolute importance
            word_importances.sort(key=lambda x: abs(x['importance']), reverse=True)
            
            return {
                'method': 'SHAP',
                'predicted_class': predicted_class,
                'predicted_label': self.LABEL_MAP[predicted_class],
                'confidence': float(probs[0][predicted_class]),
                'probabilities': {self.LABEL_MAP[i]: float(p) for i, p in enumerate(probs[0])},
                'word_importances': word_importances[:self.num_features],
                'all_importances': word_importances,
                'shap_values_raw': shap_values,
            }
        
        return self._fallback_attention_explanation(text)
    
    def explain_with_lime(self, text: str) -> Dict:
        """
        Generate LIME explanation for a single text.
        
        LIME creates local linear approximations to explain
        individual predictions.
        """
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError:
            return self._fallback_attention_explanation(text)
        
        explainer = LimeTextExplainer(
            class_names=list(self.LABEL_MAP.values()),
            split_expression=r'\W+',
        )
        
        exp = explainer.explain_instance(
            text,
            self._predict_proba,
            num_features=self.num_features,
            num_samples=200,
        )
        
        # Get prediction
        probs = self._predict_proba(text)
        predicted_class = int(np.argmax(probs[0]))
        
        # Extract word importances
        word_importances = []
        for word, importance in exp.as_list(label=predicted_class):
            word_importances.append({
                'word': word,
                'importance': float(importance),
                'direction': 'risk' if importance > 0 else 'protective',
            })
        
        return {
            'method': 'LIME',
            'predicted_class': predicted_class,
            'predicted_label': self.LABEL_MAP[predicted_class],
            'confidence': float(probs[0][predicted_class]),
            'probabilities': {self.LABEL_MAP[i]: float(p) for i, p in enumerate(probs[0])},
            'word_importances': word_importances,
        }
    
    def _fallback_attention_explanation(self, text: str) -> Dict:
        """
        Fallback: use attention weights for explainability
        when SHAP/LIME are not available.
        """
        enc = self.tokenizer(
            text, return_tensors='pt', max_length=256,
            padding='max_length', truncation=True
        )
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                enc['input_ids'].to(self.device),
                enc['attention_mask'].to(self.device)
            )
            probs = torch.softmax(outputs['logits'], dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
        
        # Try to get attention weights
        word_importances = []
        if outputs.get('attentions') is not None:
            # Average attention from last layer
            attn = outputs['attentions'][-1]  # Last layer
            attn_avg = attn.mean(dim=1)[0]    # Average over heads
            attn_weights = attn_avg[0]          # CLS token attention
            
            tokens = self.tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
            
            for token, weight in zip(tokens, attn_weights):
                if token not in ['[PAD]', '[CLS]', '[SEP]'] and not token.startswith('##'):
                    word_importances.append({
                        'word': token,
                        'importance': float(weight),
                        'direction': 'attention',
                    })
            
            word_importances.sort(key=lambda x: x['importance'], reverse=True)
        else:
            # Simple keyword-based fallback
            risk_words = {
                'die', 'kill', 'suicide', 'dead', 'end', 'pain', 'hurt',
                'hopeless', 'worthless', 'alone', 'nobody', 'never',
                'depressed', 'anxiety', 'crying', 'empty', 'numb',
            }
            
            for word in text.lower().split():
                clean_word = word.strip('.,!?;:"\'-()[]{}')
                if clean_word in risk_words:
                    word_importances.append({
                        'word': clean_word,
                        'importance': 0.8,
                        'direction': 'risk',
                    })
        
        return {
            'method': 'attention_fallback',
            'predicted_class': predicted_class,
            'predicted_label': self.LABEL_MAP[predicted_class],
            'confidence': float(probs[0][predicted_class]),
            'probabilities': {self.LABEL_MAP[i]: float(p) for i, p in enumerate(probs[0])},
            'word_importances': word_importances[:self.num_features],
        }
    
    def explain(self, text: str) -> Dict:
        """
        Generate explanation using configured method.
        Falls back gracefully if primary method fails.
        """
        if self.method == 'shap':
            try:
                return self.explain_with_shap(text)
            except Exception as e:
                print(f"⚠️ SHAP failed ({str(e)[:80]}), trying LIME...")
                try:
                    return self.explain_with_lime(text)
                except Exception:
                    return self._fallback_attention_explanation(text)
        elif self.method == 'lime':
            try:
                return self.explain_with_lime(text)
            except Exception:
                return self._fallback_attention_explanation(text)
        else:
            return self._fallback_attention_explanation(text)
    
    def format_explanation(self, explanation: Dict) -> str:
        """Format explanation for display."""
        lines = []
        lines.append(f"🔍 Explanation ({explanation['method']})")
        lines.append(f"   Prediction: {explanation['predicted_label']} "
                     f"(confidence: {explanation['confidence']:.1%})")
        
        lines.append(f"\n   📊 Probabilities:")
        for label, prob in explanation['probabilities'].items():
            bar = "█" * int(prob * 30)
            lines.append(f"      {label:>15}: {bar} {prob:.1%}")
        
        lines.append(f"\n   🔑 Key Words (top {len(explanation['word_importances'])})")
        for wi in explanation['word_importances']:
            direction = "🔴 ↑Risk" if wi['direction'] == 'risk' else "🟢 ↓Protective"
            lines.append(f"      {wi['word']:>20}: {wi['importance']:+.4f} ({direction})")
        
        return '\n'.join(lines)
