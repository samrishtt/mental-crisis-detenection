# 🧠 MindGuard: Privacy-Preserving Mental Health Crisis Detection

> *A human-in-the-loop NLP screening tool for adolescent mental health triage in under-resourced school settings.*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ethics First](https://img.shields.io/badge/Ethics-First-green.svg)](#ethical-framework)

## ⚠️ Important Disclaimer

**This is a research screening tool, NOT a diagnostic system.** It is designed to assist — never replace — trained school counselors. All flagged responses **must** be reviewed by a qualified human professional. This tool cannot and does not diagnose mental health conditions.

### What This Model CANNOT Do
- ❌ Diagnose any mental health condition
- ❌ Replace professional clinical assessment
- ❌ Make treatment recommendations
- ❌ Provide crisis intervention
- ❌ Operate without human oversight
- ❌ Give binary "suicidal / not suicidal" labels

## 🏗️ Project Structure

```
mindguard/
├── config/
│   └── config.yaml              # All hyperparameters and settings
├── data/
│   ├── pipeline.py              # Dataset loading, cleaning, splitting
│   └── augmentation.py          # Text augmentation for minority classes
├── model/
│   ├── mental_bert.py           # Mental-BERT fine-tuning model
│   └── loss.py                  # Weighted + asymmetric loss functions
├── training/
│   ├── trainer.py               # Training loop with early stopping
│   └── metrics.py               # F1, AUC-ROC, precision/recall
├── evaluation/
│   ├── bias_audit.py            # Demographic bias testing
│   └── explainability.py        # SHAP/LIME word-level explanations
├── app/
│   └── gradio_app.py            # Counselor-facing Gradio interface
├── docs/
│   ├── pilot_framework.md       # Ethical pilot study design
│   └── research_paper_outline.md # arXiv paper structure
├── notebooks/
│   └── 01_data_exploration.py   # Data exploration & visualization
├── requirements.txt
└── README.md
```

## 🎯 Research Framing

**Title:** MindGuard: A Privacy-Preserving, Human-in-the-Loop NLP Screening Tool for Adolescent Mental Health Triage in Under-Resourced School Settings

**Target Venues:** Regeneron ISEF, Conrad Challenge, arXiv (cs.CL / cs.CY)

## 🏷️ Severity Classification (CSSRS-Aligned)

| Level | Label | Description | Action |
|-------|-------|-------------|--------|
| 0 | No Concern | No indicators of distress | Routine check-in |
| 1 | Mild | General stress, low mood | Counselor awareness |
| 2 | Moderate | Significant distress signals | Counselor follow-up within 48h |
| 3 | Severe Crisis | Acute crisis indicators | Immediate counselor intervention |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run data pipeline
python -m data.pipeline

# Train model
python -m training.trainer

# Launch counselor interface
python -m app.gradio_app
```

## 📜 Ethical Framework

This project adheres to strict ethical guidelines:
1. **No text logging** — user inputs are never stored
2. **Human-in-the-loop** — all flags require counselor review
3. **Severity bands only** — no binary suicidal/not labels
4. **Bias audited** — tested for demographic fairness
5. **Privacy-preserving** — designed for anonymous text only
6. **Asymmetric error costs** — missing a crisis > false alarm

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.
