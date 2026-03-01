"""
MindGuard Model — Mental-BERT Fine-Tuning
==========================================
Fine-tunes mental-bert-base-uncased for multi-class severity classification
aligned with CSSRS levels: [No Concern / Mild / Moderate / Severe Crisis]

Ethical Note: This model outputs severity BANDS, never binary labels.
All predictions must be reviewed by a trained professional.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoConfig,
)
from typing import Optional, Dict

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# ============================================================
# Model Definition
# ============================================================

class MindGuardClassifier(nn.Module):
    """
    Mental-BERT based classifier for mental health severity detection.
    
    Architecture:
        mental-bert-base-uncased → Dropout → Linear(768, 4)
    
    Outputs severity probabilities across 4 CSSRS-aligned levels.
    """
    
    LABEL_MAP = {
        0: "No Concern",
        1: "Mild", 
        2: "Moderate",
        3: "Severe Crisis"
    }
    
    DISCLAIMER = (
        "⚠️ SCREENING TOOL ONLY — This prediction is NOT a diagnosis. "
        "A trained human professional MUST review all flagged responses. "
        "This model cannot replace clinical assessment."
    )
    
    def __init__(
        self,
        model_name: str = "mental/mental-bert-base-uncased",
        num_labels: int = 4,
        dropout: float = 0.3,
        freeze_embeddings: bool = False,
        freeze_lower_layers: int = 0,
        use_lora: bool = True,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_lora = use_lora
        
        # Load pre-trained model with classification head
        try:
            self.config = AutoConfig.from_pretrained(
                model_name, 
                num_labels=num_labels,
                output_attentions=True,
                output_hidden_states=True,
            )
            self.bert = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=self.config,
            )
        except Exception as e:
            print(f"⚠️  Could not load '{model_name}'. Falling back to 'bert-base-uncased'")
            print(f"   Error: {str(e)[:100]}")
            self.config = AutoConfig.from_pretrained(
                'bert-base-uncased',
                num_labels=num_labels,
                output_attentions=True,
                output_hidden_states=True,
            )
            self.bert = AutoModelForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                config=self.config,
            )
        
        # Custom dropout (higher than default for regularization)
        self.dropout = nn.Dropout(dropout)
        
        # Freeze embeddings if specified
        if freeze_embeddings:
            for param in self.bert.base_model.embeddings.parameters():
                param.requires_grad = False
            print(f"   🔒 Froze embedding layers")
        
        # Freeze lower transformer layers if specified
        if freeze_lower_layers > 0 and not self.use_lora: # LoRA typically manages freezing
            encoder = self.bert.base_model.encoder
            for i, layer in enumerate(encoder.layer):
                if i < freeze_lower_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"   🔒 Froze bottom {freeze_lower_layers} transformer layers")
            
        # Apply LoRA if requested and available
        if self.use_lora:
            if PEFT_AVAILABLE:
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    target_modules=["query", "value"]  # common targets for BERT
                )
                self.bert = get_peft_model(self.bert, lora_config)
                print("   🚀 Applied LoRA (Parameter-Efficient Fine-Tuning)")
            else:
                print("   ⚠️  LoRA requested but 'peft' library is not installed. Proceeding with full fine-tuning.")
                self.use_lora = False
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n   📐 Model Parameters:")
        print(f"      Total:     {total_params:>12,}")
        print(f"      Trainable: {trainable_params:>12,}")
        print(f"      Frozen:    {total_params - trainable_params:>12,}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict:
        """
        Forward pass.
        
        Returns dict with:
            - logits: (batch_size, num_labels)
            - loss: if labels provided
            - attentions: attention weights for explainability
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return {
            'loss': outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else None,
            'logits': outputs.logits,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None,
        }
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict:
        """
        Predict severity level with confidence scores.
        
        Returns:
            - predicted_label: int (0-3)
            - label_name: str
            - confidence: float (0-1)
            - probabilities: dict of label → probability
            - disclaimer: str (always included)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            
            predicted = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, predicted].item()
            
            prob_dict = {
                self.LABEL_MAP[i]: round(probs[0, i].item(), 4)
                for i in range(self.num_labels)
            }
        
        return {
            'predicted_label': predicted,
            'label_name': self.LABEL_MAP[predicted],
            'confidence': round(confidence, 4),
            'probabilities': prob_dict,
            'disclaimer': self.DISCLAIMER,
            'limitations': self._get_limitations(),
        }
    
    def _get_limitations(self) -> list:
        """What this model CANNOT do — included in every output."""
        return [
            "Cannot diagnose any mental health condition",
            "Cannot replace professional clinical assessment",
            "Cannot make treatment recommendations",
            "Cannot provide crisis intervention",
            "Should never operate without human oversight",
            "Does not give binary 'suicidal / not suicidal' labels",
            "May have different accuracy across demographic groups",
            "Performance on text outside training distribution is unknown",
        ]
    
    def save_model(self, path: str):
        """Save model weights and config."""
        self.bert.save_pretrained(path)
        print(f"   💾 Model saved to: {path}")
    
    def load_model(self, path: str):
        """Load model weights from checkpoint."""
        if self.use_lora and PEFT_AVAILABLE:
            from peft import PeftModel
            self.bert = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=self.config,
            )
            self.bert = PeftModel.from_pretrained(self.bert, path)
            print(f"   📂 LoRA adapters loaded from: {path}")
        else:
            self.bert = AutoModelForSequenceClassification.from_pretrained(
                path,
                config=self.config,
            )
            print(f"   📂 Model loaded from: {path}")


def get_tokenizer(model_name: str = "mental/mental-bert-base-uncased"):
    """Get the tokenizer for Mental-BERT (with fallback)."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        print(f"⚠️  Could not load tokenizer for '{model_name}'. Using 'bert-base-uncased'")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer
