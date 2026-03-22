import sys
import io
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Optional
import re
import numpy as np

# Fix Windows console encoding for emoji/unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.mental_bert import MindGuardClassifier, get_tokenizer

class EmotionalAnalyzer:
    """
    Analyzes text for both core severity (Mental-BERT) and an expanded 
    set of emotional dimensions (Heuristic + Keyword scoring).
    Designed for community-level aggregate analysis.
    """
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.dimensions = self.config.get('emotional_dimensions', {})
        
        # Load core Mental-BERT model if available
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_core_model()

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load the community configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # Fallback to default location
        default_path = Path(__file__).parent.parent / 'config' / 'community_config.yaml'
        if default_path.exists():
            with open(default_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        print("[WARN] community_config.yaml not found.")
        return {}

    def _load_core_model(self):
        """Attempt to load the fine-tuned Mental-BERT model."""
        try:
            model_path = Path(__file__).parent.parent / "checkpoints" / "best_model"
            if model_path.exists():
                self.tokenizer = get_tokenizer()
                self.model = MindGuardClassifier(num_labels=4).to(self.device)
                self.model.load_model(str(model_path))
                self.model.eval()
                print("[OK] Core Mental-BERT model loaded for Emotional Analyzer.")
            else:
                print("[WARN] Core model not found at checkpoints/best_model. Operating in keyword-only mode.")
        except Exception as e:
            print(f"[WARN] Failed to load core model: {str(e)[:100]}. Operating in keyword-only mode.")

    def analyze_text(self, text: str) -> Dict:
        """
        Produce a comprehensive emotional profile for a single text.
        Combines Mental-BERT core risk with dimensional analysis.
        """
        if not text or len(text.strip()) < 10:
            return {"error": "Text too short for reliable analysis (min 10 chars)."}

        result = {
            "text_length": len(text.split()),
            "core_severity": self._get_core_severity(text),
            "dimensions": self._score_dimensions(text),
            "flags": []
        }
        
        # Generate any immediate action flags based on the profile
        result["flags"] = self._generate_flags(result)
        
        return result

    def _get_core_severity(self, text: str) -> Dict:
        """Get the CSSRS-aligned severity score (0-3) from Mental-BERT or fallback."""
        if self.model and self.tokenizer:
            try:
                enc = self.tokenizer(text, return_tensors='pt', max_length=256, padding='max_length', truncation=True)
                with torch.no_grad():
                    pred_out = self.model.predict(enc['input_ids'].to(self.device), enc['attention_mask'].to(self.device))
                return {
                    "level": pred_out['label_name'],
                    "level_idx": pred_out['predicted_label'],
                    "confidence": pred_out['confidence'],
                    "source": "Mental-BERT"
                }
            except Exception as e:
                print(f"Prediction error: {e}")
        
        # Fallback keyword scoring if model fails or is absent
        return self._fallback_severity_score(text)

    def _fallback_severity_score(self, text: str) -> Dict:
        """Simple keyword fallback matching gradio_app.py logic."""
        text_lower = text.lower()
        severe_keywords = ['kill myself', 'suicide', 'want to die', 'end it all', 'overdose']
        moderate_keywords = ['depressed', 'hopeless', 'worthless', 'self harm', 'breakdown']
        mild_keywords = ['stressed', 'anxious', 'overwhelmed', 'lonely', 'sad']
        
        if any(kw in text_lower for kw in severe_keywords):
            return {"level": "Severe Crisis", "level_idx": 3, "confidence": 0.85, "source": "Keyword Fallback"}
        elif any(kw in text_lower for kw in moderate_keywords):
            return {"level": "Moderate", "level_idx": 2, "confidence": 0.75, "source": "Keyword Fallback"}
        elif any(kw in text_lower for kw in mild_keywords):
            return {"level": "Mild", "level_idx": 1, "confidence": 0.65, "source": "Keyword Fallback"}
        return {"level": "No Concern", "level_idx": 0, "confidence": 0.90, "source": "Keyword Fallback"}

    def _score_dimensions(self, text: str) -> Dict[str, Dict]:
        """Score each configured emotional dimension from 0-100."""
        text_lower = text.lower()
        scores = {}
        
        for dim_key, dim_data in self.dimensions.items():
            score = 0
            primary_triggers = []
            
            # High intensity keywords (+40 each)
            for kw in dim_data.get('keywords_high', []):
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    score += 40
                    primary_triggers.append(kw)
            
            # Moderate intensity keywords (+20 each)
            for kw in dim_data.get('keywords_moderate', []):
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    score += 20
                    if len(primary_triggers) < 3:
                        primary_triggers.append(kw)
                        
            # Mild intensity keywords (+10 each)
            for kw in dim_data.get('keywords_mild', []):
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    score += 10
            
            # Cap at 100
            final_score = min(100, score)
            
            scores[dim_key] = {
                "score": final_score,
                "label": dim_data.get('label', dim_key.title()),
                "icon": dim_data.get('icon', '-'),
                "color": dim_data.get('color', '#888888'),
                "triggers_found": primary_triggers[:3]
            }
            
        return scores

    def _generate_flags(self, result: Dict) -> List[Dict]:
        """Generate specific alerts or flags based on the combined profile."""
        flags = []
        
        # 1. Core Severity Alert
        if result['core_severity']['level_idx'] == 3:
            flags.append({"type": "critical", "message": "High Crisis Implied (Severe Severity Level)"})
            
        # 2. Self-Harm Specific Alert
        if 'self_harm' in result['dimensions'] and result['dimensions']['self_harm']['score'] >= 40:
            flags.append({"type": "critical", "message": "Explicit self-harm indicators detected."})
            
        # 3. High Multi-Morbidity (High scores across several negative dimensions)
        neg_dims = ['anxiety', 'depression', 'loneliness', 'anger', 'burnout', 'grief']
        high_neg_count = sum(1 for d in neg_dims if d in result['dimensions'] and result['dimensions'][d]['score'] >= 60)
        
        if high_neg_count >= 3:
            flags.append({"type": "warning", "message": "Complex distress state across multiple emotions."})
            
        return flags

    def analyze_batch(self, texts: List[str]) -> Dict:
        """
        Analyze a batch of texts and compute aggregate statistics for a community.
        This ensures privacy by returning only the macro view, not individual texts.
        """
        valid_texts = [t for t in texts if isinstance(t, str) and len(t.strip()) >= 10]
        if not valid_texts:
            return {"error": "No valid texts provided for batch analysis."}
            
        results = [self.analyze_text(text) for text in valid_texts]
        
        # Aggregate logic
        total = len(results)
        severity_counts = {"No Concern": 0, "Mild": 0, "Moderate": 0, "Severe Crisis": 0}
        avg_dimensions = {k: 0 for k in self.dimensions.keys()}
        total_critical_flags = 0
        
        for res in results:
            if "error" in res: continue
            
            # Severity counts
            sev_level = res['core_severity']['level']
            if sev_level in severity_counts:
                severity_counts[sev_level] += 1
                
            # Dimensional averages
            for dim_key, dim_data in res['dimensions'].items():
                avg_dimensions[dim_key] += dim_data['score']
                
            # Flags
            for flag in res['flags']:
                if flag['type'] == 'critical':
                    total_critical_flags += 1

        # Finalize averages
        for dim_key in avg_dimensions:
            avg_dimensions[dim_key] = round(avg_dimensions[dim_key] / total, 1)
            
        return {
            "summary": {
                "total_analyzed": total,
                "critical_flags_generated": total_critical_flags,
                "overall_health_score": self._calculate_health_score(severity_counts, total)
            },
            "severity_distribution": severity_counts,
            "average_emotional_dimensions": avg_dimensions,
            # Note: We do NOT return the individual `results` list to 
            # enforce the 'aggregate_only' privacy constraint at the API boundary.
        }
        
    def _calculate_health_score(self, counts: Dict[str, int], total: int) -> int:
        """Calculate a single 0-100 overall community health score (0=Crisis, 100=Excellent)."""
        if total == 0: return 100
        # Weighting: Severe=5, Moderate=3, Mild=1, None=0
        penalty = (counts.get("Severe Crisis", 0) * 5) + \
                  (counts.get("Moderate", 0) * 3) + \
                  (counts.get("Mild", 0) * 1)
                  
        # Max possible penalty is if everyone is severe
        max_penalty = total * 5
        
        # Invert and scale to 100
        raw_score = 100 - ((penalty / max_penalty) * 100)
        return int(max(0, min(100, raw_score)))
