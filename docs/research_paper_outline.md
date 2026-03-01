# Research Paper Outline

## MindGuard: A Privacy-Preserving, Human-in-the-Loop NLP Screening Tool for Adolescent Mental Health Triage in Under-Resourced School Settings

**Target Venues:** Regeneron ISEF, Conrad Challenge, arXiv (cs.CL / cs.CY)

---

## Structured Abstract

**Background:** Adolescent mental health crises are increasing, yet under-resourced schools lack systematic screening tools. Existing NLP approaches focus on binary classification without human oversight.

**Objective:** We present MindGuard, a privacy-preserving, human-in-the-loop NLP screening tool that classifies anonymous student text into four severity bands aligned with the Columbia Suicide Severity Rating Scale (CSSRS).

**Methods:** We fine-tune Mental-BERT on Reddit mental health corpora using a novel asymmetric loss function that penalizes false negatives on severe cases more heavily than false positives. We incorporate SHAP-based explainability and conduct a comprehensive demographic bias audit.

**Results:** [To be filled after experiments] Our model achieves [X]% macro F1 and [X]% crisis recall. Bias audit shows < [X]% prediction flip rate across demographic groups. Counselor-facing interface provides word-level explanations for every prediction.

**Conclusions:** MindGuard demonstrates that responsible NLP can augment school counselor workflows without replacing clinical judgment. Our ethical framework ensures privacy preservation, human oversight, and demographic fairness.

**Keywords:** mental health NLP, crisis detection, CSSRS, transformer, bias audit, explainable AI, human-in-the-loop, adolescent mental health, school counseling

---

## 1. Introduction

### 1.1 Problem Statement
- Rising adolescent mental health crisis (CDC data)
- School counselor-to-student ratios (ASCA recommendations vs. reality)
- Gap in systematic screening tools for under-resourced settings
- Ethical concerns with existing automated mental health detection

### 1.2 Research Gap
- Existing work: binary classification (suicidal vs. not) — ethically problematic
- Lack of severity-graded output aligned with clinical scales
- Missing human-in-the-loop design in current NLP mental health tools
- Insufficient bias auditing in mental health NLP
- No privacy-preserving design for school settings

### 1.3 Contributions
1. **CSSRS-aligned severity classification** (4 levels, not binary)
2. **Asymmetric loss function** that reflects clinical priority (missing a crisis > false alarm)
3. **Comprehensive bias audit** with counterfactual fairness testing
4. **Word-level explainability** for counselor trust and transparency
5. **Privacy-preserving design** (no text logging, anonymous input)
6. **Ethical pilot framework** for school deployment

---

## 2. Related Work

### 2.1 NLP for Mental Health Detection
- CLPsych Shared Tasks (Zirikly et al., 2019; Macavaney et al., 2021)
- Reddit-based mental health classification (Yates et al., 2017; Losada et al., 2020)
- BERT-based approaches (Ji et al., 2022; Matero et al., 2019)
- Mental-BERT (Ji et al., 2022) — domain-specific pre-training

### 2.2 Suicide Risk Assessment Scales
- Columbia Suicide Severity Rating Scale (CSSRS) (Posner et al., 2011)
- PHQ-9 and its NLP adaptations
- Clinical vs. computational severity assessment

### 2.3 Fairness and Bias in Mental Health NLP
- Demographic bias in language models (Blodgett et al., 2020)
- Fairness in clinical NLP (Chen et al., 2020)
- Counterfactual fairness approaches (Kusner et al., 2017)

### 2.4 Explainable AI in Healthcare
- SHAP (Lundberg & Lee, 2017)
- LIME (Ribeiro et al., 2016)
- Clinical decision support explanations (Tonekaboni et al., 2019)

### 2.5 Human-in-the-Loop AI
- Human oversight in clinical AI (Cai et al., 2019)
- Collaborative human-AI decision making (Bansal et al., 2021)
- Appropriate trust calibration (Lee & See, 2004)

---

## 3. Ethical Framework

### 3.1 Design Principles
- Privacy by design (no text storage)
- Human oversight mandate
- Severity bands, not binary labels
- Asymmetric error treatment
- Demographic fairness auditing
- Transparency through explainability

### 3.2 What the Model Cannot Do
- Cannot diagnose mental health conditions
- Cannot replace clinical assessment
- Cannot operate without human oversight
- Cannot provide crisis intervention

### 3.3 Alignment with AI Ethics Guidelines
- EU AI Act (high-risk classification)
- OECD AI Principles
- APA Guidelines for Technology-Based Interventions

---

## 4. Methods

### 4.1 Dataset
- **Sources:** Reddit mental health subreddits (r/SuicideWatch, r/depression, r/mentalhealth, r/CasualConversation)
- **Labeling scheme:** CSSRS-aligned severity mapping
  - No Concern (0), Mild (1), Moderate (2), Severe Crisis (3)
- **Preprocessing:** URL removal, username anonymization, text normalization
- **Class distribution:** [Table with counts and imbalance ratio]
- **Splits:** Stratified 80/10/10 train/val/test

### 4.2 Model Architecture
- **Base model:** mental-bert-base-uncased (Ji et al., 2022)
- **Classification head:** Dropout(0.3) → Linear(768, 4)
- **Why Mental-BERT:** Domain-specific pre-training on mental health text
- **Parameter-Efficient Fine-Tuning:** LoRA (Low-Rank Adaptation) via HuggingFace PEFT to enable efficient training and adaptation on consumer hardware while preserving robust pre-trained representations
- **Tokenization:** WordPiece, max length 256

### 4.3 Training
- **Optimizer:** AdamW (lr=2e-5, weight_decay=0.01)
- **Scheduler:** Linear warmup (10% of steps)
- **Loss function:** Combined Asymmetric CE + Focal Loss
  - False negative penalty: 3x for severe class
  - Class weight multipliers: [1.0, 1.5, 2.5, 5.0]
- **Early stopping:** Patience 3, monitoring val F1 (macro)
- **Mixed precision:** FP16 for GPU efficiency

### 4.4 Evaluation Metrics
- **Primary:** F1 (macro), Crisis Recall
- **Secondary:** AUC-ROC, per-class precision/recall
- **Error analysis:** False positive/negative breakdown by severity
- **Bias metrics:** Counterfactual flip rate, per-demographic F1

### 4.5 Explainability
- **Method:** SHAP (partition explainer for transformers)
- **Output:** Top-15 words with importance scores
- **Visualization:** Highlighted text with color-coded importance

### 4.6 Bias Audit
- **Counterfactual test:** Same text ± demographic terms
- **Demographic categories:** Gender, race, religion
- **Threshold:** < 10% flip rate for fairness

---

## 5. Results

### 5.1 Overall Performance
- [Table: accuracy, F1 macro/weighted, AUC-ROC]

### 5.2 Per-Class Performance
- [Table: precision, recall, F1 per severity level]

### 5.3 Crisis Detection Performance
- Crisis recall: [X]%
- Crisis false negative analysis
- Confusion matrix visualization

### 5.4 Comparison with Baselines
- Mental-BERT vs. BERT-base
- Asymmetric loss vs. standard CE
- Effect of class weighting

### 5.5 Explainability Examples
- [Figure: SHAP word highlights for each severity level]
- [Example predictions with explanations]

### 5.6 Bias Audit Results
- [Table: flip rates by demographic group]
- [Figure: per-demographic F1 scores]
- Interpretation and limitations

---

## 6. Discussion

### 6.1 Key Findings
- Model performance relative to clinical utility
- Importance of asymmetric loss for crisis detection
- Explainability as a trust-building mechanism

### 6.2 Clinical Implications
- How counselors can use severity bands
- Integration with existing school workflows
- Time savings and workload reduction

### 6.3 Ethical Implications
- Privacy preservation in practice
- Appropriate trust calibration
- Risk of over-reliance

---

## 7. Limitations

### 7.1 Data Limitations
- Reddit text ≠ student check-in text (domain shift)
- Subreddit labels as proxy for clinical severity
- Self-reported text inherent biases

### 7.2 Model Limitations
- Performance on out-of-distribution text
- Sarcasm, irony, and implicit language
- Non-English text
- Cultural context

### 7.3 Evaluation Limitations
- No clinical validation (no IRB-approved student data)
- Bias audit covers limited demographic categories
- Counterfactual test is a proxy for real-world fairness

### 7.4 Deployment Limitations
- Requires trained counselor for all interpretations
- Cannot handle multimedia (images, voice)
- Single-turn analysis (no conversation history)

---

## 8. Future Work

1. **Clinical validation:** IRB-approved pilot with real student data
2. **Multilingual support:** Expand beyond English
3. **Temporal modeling:** Track severity changes over time
4. **Multimodal:** Integrate voice/tone analysis
5. **Federated learning:** Train on distributed school data without centralization
6. **Active learning:** Counselor feedback loop for model improvement
7. **Cross-cultural adaptation:** Cultural norms affect expression of distress

---

## 9. Conclusion

MindGuard demonstrates that responsible, human-in-the-loop NLP can support mental health screening in schools without compromising privacy or clinical standards. By using severity bands instead of binary labels, asymmetric loss functions, and comprehensive bias auditing, we show that ethical constraints and model performance are not in conflict — they are complementary.

---

## References

[To be formatted in ACL style]

- Blodgett, S. L., et al. (2020). Language (Technology) is Power.
- Ji, S., et al. (2022). Mental-BERT: Publicly Available Pretrained Language Models for Mental Healthcare.
- Kusner, M. J., et al. (2017). Counterfactual Fairness.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions.
- Macavaney, S., et al. (2021). Community-level Research on Suicidality Prediction.
- Posner, K., et al. (2011). The Columbia–Suicide Severity Rating Scale.
- Ribeiro, M. T., et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier.
- Yates, A., et al. (2017). Depression and Self-Harm Risk Assessment in Online Forums.
- Zirikly, A., et al. (2019). CLPsych 2019 Shared Task.

---

## Appendices

### A. Ethical Pilot Framework
See `docs/pilot_framework.md`

### B. Model Card
Following Mitchell et al. (2019) model card format:
- **Model Name:** MindGuard v1.0
- **Task:** Mental health severity classification (4-class)
- **Intended Use:** School counselor screening aid (anonymous text)
- **Out-of-Scope Use:** Clinical diagnosis, public deployment, unsupervised use
- **Training Data:** Reddit mental health corpora (publicly available)
- **Evaluation:** Macro F1, crisis recall, bias audit
- **Ethical Considerations:** See Section 3
- **Limitations:** See Section 7

### C. CSSRS Alignment Table
| MindGuard Level | CSSRS Level | Description |
|-----------------|-------------|-------------|
| No Concern (0)  | Level 0     | No indicators |
| Mild (1)        | Level 1     | Wish to be dead |
| Moderate (2)    | Level 2-3   | Non-specific thoughts / Active ideation without plan |
| Severe Crisis (3)| Level 4-5  | Active ideation with plan/intent |
