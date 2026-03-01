"""
MindGuard Gradio App — Counselor Interface
===========================================
A simple, privacy-preserving interface for school counselors to screen
anonymous student check-in responses.

HARD ETHICAL REQUIREMENTS:
1. NEVER stores or logs any user input text
2. ALL outputs include a human-review disclaimer
3. Model outputs severity BANDS, never binary labels
4. Includes what the model CANNOT do in every output
"""

import sys
import json
import warnings
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

try:
    from model.mental_bert import MindGuardClassifier, get_tokenizer
    from evaluation.explainability import MindGuardExplainer
except ImportError:
    MindGuardClassifier = None
    MindGuardExplainer = None

# ============================================================
# App Configuration
# ============================================================

DISCLAIMER = """
⚠️ **IMPORTANT DISCLAIMER**

This is a **screening tool ONLY** — it is NOT a diagnostic system.

✅ This tool CAN: Flag text for counselor review based on language patterns
❌ This tool CANNOT:
- Diagnose any mental health condition
- Replace professional clinical assessment
- Make treatment recommendations
- Provide crisis intervention
- Operate without human oversight

**A trained human professional MUST review ALL flagged responses.**

If someone is in immediate danger, please contact:
- **988 Suicide & Crisis Lifeline**: Call/text 988
- **Crisis Text Line**: Text HOME to 741741
"""

SEVERITY_COLORS = {
    "No Concern": "#4CAF50",
    "Mild": "#FFC107",
    "Moderate": "#FF9800",
    "Severe Crisis": "#F44336",
}

SEVERITY_ACTIONS = {
    "No Concern": "**Action:** Routine check-in. No immediate follow-up needed.",
    "Mild": "**Action:** Counselor awareness. Schedule a general wellness check-in.",
    "Moderate": "**Action:** Follow up within 48 hours. Review in context of student's history.",
    "Severe Crisis": "**Action:** ⚠️ IMMEDIATE counselor review required. Follow your school's crisis protocol.",
}


# Global state for production model
MODEL = None
TOKENIZER = None
EXPLAINER = None

def load_production_model():
    """Attempt to load the trained model, returning True if successful."""
    global MODEL, TOKENIZER, EXPLAINER
    if MindGuardClassifier is None:
        print("⚠️  Model code not reachable. Running in demo mode.")
        return False
        
    model_path = Path(__file__).parent.parent / "checkpoints" / "best_model"
    if not model_path.exists():
        print("⚠️  No trained model found at checkpoints/best_model. Running in demo mode.")
        return False
        
    print("⏳ Loading fine-tuned Mental-BERT model...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        TOKENIZER = get_tokenizer()
        MODEL = MindGuardClassifier(num_labels=4).to(device)
        MODEL.load_model(str(model_path))
        MODEL.eval()
        
        EXPLAINER = MindGuardExplainer(MODEL, TOKENIZER, device, method='attention_fallback', num_features=8)
        print("✅ Production model loaded successfully!")
        return True
    except Exception as e:
        print(f"⚠️  Failed to load model: {str(e)[:100]}. Running in demo mode.")
        MODEL = None
        return False

def create_app():
    """
    Create a demonstration version of the Gradio app
    that works without a trained model (using keyword-based scoring).
    """
    try:
        import gradio as gr
    except ImportError:
        print("Install Gradio: pip install gradio")
        return None
    
    def analyze_text_demo(text: str) -> str:
        """
        Demo analysis using keyword-based scoring.
        In production, this would use the trained Mental-BERT model.
        
        NOTE: Input text is NEVER logged or stored.
        """
        if not text or len(text.strip()) < 10:
            return "⚠️ Please enter a longer text (at least 10 characters) for analysis."
        
        if MODEL is not None:
            try:
                # Production ML inference
                device = MODEL.bert.device
                enc = TOKENIZER(text, return_tensors='pt', max_length=256, padding='max_length', truncation=True)
                pred_out = MODEL.predict(enc['input_ids'].to(device), enc['attention_mask'].to(device))
                
                severity = pred_out['label_name']
                confidence = pred_out['confidence']
                
                explain_dict = EXPLAINER.explain(text)
                triggered_words = [wi['word'] for wi in explain_dict['word_importances'] if wi['direction'] == 'risk']
                mild_hits = [] 
                moderate_hits = []
                severe_hits = [] 
                
            except Exception as e:
                return f"⚠️ **Model Error:** {str(e)}. Please check standard output."
        else:
            # Keyword-based severity scoring (demo version)
            text_lower = text.lower()
            
            severe_keywords = [
                'kill myself', 'end my life', 'suicide', 'want to die',
                'no reason to live', 'better off dead', 'end it all',
                'goodbye', 'final', 'last day', 'overdose',
            ]
            
            moderate_keywords = [
                'depressed', 'hopeless', 'worthless', 'can\'t go on',
                'no point', 'empty', 'numb', 'crying', 'can\'t eat',
                'can\'t sleep', 'self harm', 'cutting', 'panic',
                'breakdown', 'falling apart', 'burden', 'disappear',
            ]
            
            mild_keywords = [
                'stressed', 'anxious', 'overwhelmed', 'worried',
                'lonely', 'sad', 'frustrated', 'struggling',
                'exhausted', 'isolated', 'tough', 'hard time',
            ]
            
            # Score
            severe_hits = [kw for kw in severe_keywords if kw in text_lower]
            moderate_hits = [kw for kw in moderate_keywords if kw in text_lower]
            mild_hits = [kw for kw in mild_keywords if kw in text_lower]
            
            if severe_hits:
                severity = "Severe Crisis"
                confidence = min(0.65 + len(severe_hits) * 0.1, 0.95)
                triggered_words = severe_hits + moderate_hits
            elif moderate_hits:
                severity = "Moderate"
                confidence = min(0.55 + len(moderate_hits) * 0.08, 0.90)
                triggered_words = moderate_hits + mild_hits
            elif mild_hits:
                severity = "Mild"
                confidence = min(0.50 + len(mild_hits) * 0.07, 0.85)
                triggered_words = mild_hits
            else:
                severity = "No Concern"
                confidence = 0.70
                triggered_words = []
        
        # Build response
        color = SEVERITY_COLORS[severity]
        action = SEVERITY_ACTIONS[severity]
        
        response = f"""
## 📊 Screening Result

### Severity Level: **{severity}**
### Confidence: {confidence:.0%}

{action}

---

### 🔑 Triggered Words
"""
        if triggered_words:
            for word in triggered_words:
                response += f"- `{word}` — flagged as severity indicator\n"
        else:
            response += "- No specific crisis keywords detected\n"
        
        response += f"""
---

### 📈 Severity Probabilities (Demo)
| Level | Probability |
|-------|-------------|
| No Concern | {'%.0f%%' % (100 - confidence*100 if severity != 'No Concern' else confidence*100)} |
| Mild | {'%.0f%%' % (confidence*100*0.2 if severity == 'Mild' else 10)} |
| Moderate | {'%.0f%%' % (confidence*100*0.3 if severity == 'Moderate' else 10)} |
| Severe Crisis | {'%.0f%%' % (confidence*100 if severity == 'Severe Crisis' else 5)} |

---

### ⚠️ What This Model CANNOT Do
- ❌ Diagnose any mental health condition
- ❌ Replace professional clinical assessment
- ❌ Make treatment recommendations
- ❌ Provide crisis intervention
- ❌ Operate without human oversight
- ❌ Give binary "suicidal / not suicidal" labels

### 🔒 Privacy Notice
Your input text was **NOT stored or logged**. It was processed in-memory only.

---
{DISCLAIMER}
"""
        return response
    
    # Build Gradio Interface
    with gr.Blocks(
        title="MindGuard — Mental Health Screening Tool",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
        .disclaimer-box {
            background: #FFF3E0;
            border: 2px solid #FF9800;
            border-radius: 8px;
            padding: 16px;
            margin: 12px 0;
        }
        .header-text {
            text-align: center;
            color: #1565C0;
        }
        footer {display: none !important;}
        """
    ) as app:
        
        gr.Markdown("""
        # 🧠 MindGuard — Mental Health Screening Tool
        ### A privacy-preserving screening tool for school counselors
        
        **Purpose:** Screen anonymous student check-in responses for severity indicators.
        **This is NOT a diagnostic tool.** All flagged responses must be reviewed by a trained professional.
        """)
        
        gr.Markdown(DISCLAIMER, elem_classes=["disclaimer-box"])
        
        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="📝 Paste Anonymous Student Text",
                    placeholder="Enter the anonymous check-in response here...\n\n(This text is NOT stored or logged)",
                    lines=8,
                    max_lines=20,
                )
                
                with gr.Row():
                    analyze_btn = gr.Button(
                        "🔍 Screen Text", 
                        variant="primary",
                        size="lg"
                    )
                    clear_btn = gr.Button("🗑️ Clear", size="lg")
            
            with gr.Column(scale=1):
                output = gr.Markdown(
                    label="📊 Screening Result",
                    value="*Enter text and click 'Screen Text' to begin.*"
                )
        
        gr.Markdown(f"""
        ---
        ### 📞 Crisis Resources
        - **988 Suicide & Crisis Lifeline**: Call or text **988**
        - **Crisis Text Line**: Text **HOME** to **741741**
        - **SAMHSA Helpline**: **1-800-662-4357**
        
        ---
        *MindGuard v1.0 — A research tool by [Your Name]. Not for clinical use.*
        *Mode: {"**Production (Mental-BERT)**" if MODEL else "**Demo (Keyword Fallback)**"}*
        """)
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_text_demo,
            inputs=[text_input],
            outputs=[output],
        )
        
        clear_btn.click(
            fn=lambda: ("", "*Enter text and click 'Screen Text' to begin.*"),
            inputs=[],
            outputs=[text_input, output],
        )
    
    return app


def main():
    """Launch the Gradio app."""
    load_production_model()
    app = create_app()
    if app:
        print("\n🚀 Launching MindGuard Screening Tool...")
        print("⚠️  REMINDER: This is a screening tool ONLY.")
        print("   All flagged responses MUST be reviewed by a trained professional.\n")
        app.launch(
            server_port=7860,
            share=False,  # NEVER share publicly
            show_error=True,
        )


if __name__ == '__main__':
    main()
