"""
MindGuard Training Metrics
==========================
Comprehensive metric computation for mental health crisis detection.
Includes macro F1, per-class precision/recall, AUC-ROC, and
false positive/negative analysis with emphasis on crisis classes.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)


LABEL_MAP = {
    0: "No Concern",
    1: "Mild",
    2: "Moderate",
    3: "Severe Crisis"
}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    num_classes: int = 4,
) -> Dict:
    """
    Compute comprehensive metrics for mental health severity classification.
    
    Returns a dict with:
        - Overall: accuracy, F1 macro, F1 weighted
        - Per-class: precision, recall, F1
        - AUC-ROC (macro, per-class) if probabilities provided
        - False positive/negative analysis
        - Crisis-specific recall (most critical metric)
    """
    metrics = {}
    
    # ---- Overall Metrics ----
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    # ---- Per-Class Metrics ----
    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class'] = {}
    for i in range(min(num_classes, len(per_class_p))):
        label_name = LABEL_MAP.get(i, f"Class_{i}")
        metrics['per_class'][label_name] = {
            'precision': float(per_class_p[i]),
            'recall': float(per_class_r[i]),
            'f1': float(per_class_f1[i]),
        }
    
    # ---- Crisis-Specific Recall (MOST CRITICAL) ----
    # Missing a severe crisis = worst outcome
    if 3 in y_true or 3 in y_pred:
        crisis_mask_true = (y_true == 3)
        if crisis_mask_true.sum() > 0:
            crisis_correct = ((y_pred == 3) & crisis_mask_true).sum()
            metrics['crisis_recall'] = float(crisis_correct / crisis_mask_true.sum())
            metrics['crisis_missed'] = int(crisis_mask_true.sum() - crisis_correct)
            metrics['crisis_total'] = int(crisis_mask_true.sum())
        else:
            metrics['crisis_recall'] = None
            metrics['crisis_missed'] = 0
            metrics['crisis_total'] = 0
    
    # ---- AUC-ROC (if probabilities available) ----
    if y_probs is not None and y_probs.shape[1] == num_classes:
        try:
            metrics['auc_roc_macro'] = float(roc_auc_score(
                y_true, y_probs, multi_class='ovr', average='macro'
            ))
            
            # Per-class AUC
            metrics['auc_roc_per_class'] = {}
            for i in range(num_classes):
                try:
                    binary_true = (y_true == i).astype(int)
                    if binary_true.sum() > 0 and binary_true.sum() < len(binary_true):
                        auc = float(roc_auc_score(binary_true, y_probs[:, i]))
                        metrics['auc_roc_per_class'][LABEL_MAP[i]] = auc
                except Exception:
                    pass
        except Exception as e:
            metrics['auc_roc_macro'] = None
            metrics['auc_roc_note'] = f"Could not compute: {str(e)[:100]}"
    
    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    metrics['confusion_matrix'] = cm.tolist()
    
    # ---- False Positive / False Negative Analysis ----
    metrics['error_analysis'] = compute_error_analysis(y_true, y_pred, num_classes)
    
    return metrics


def compute_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 4,
) -> Dict:
    """
    Detailed false positive/negative analysis.
    
    Key insight: For crisis detection, we care most about:
    - False Negatives on Severe: Model says "No Concern" when student is in crisis
    - False Positives on Severe: Model says "Crisis" when student is fine
    
    Both matter, but FN on Severe is the WORST outcome.
    """
    analysis = {}
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    for i in range(num_classes):
        label = LABEL_MAP.get(i, f"Class_{i}")
        
        # True Positives
        tp = cm[i, i]
        
        # False Negatives: True is class i, predicted as something else
        fn = cm[i, :].sum() - tp
        
        # False Positives: Predicted as class i, but true is something else
        fp = cm[:, i].sum() - tp
        
        # True Negatives
        tn = cm.sum() - tp - fp - fn
        
        analysis[label] = {
            'true_positives': int(tp),
            'false_negatives': int(fn),
            'false_positives': int(fp),
            'true_negatives': int(tn),
        }
        
        # For the Severe Crisis class, add specific details
        if i == 3:
            # What did severe cases get misclassified as?
            misclassified_as = {}
            for j in range(num_classes):
                if j != i and cm[i, j] > 0:
                    misclassified_as[LABEL_MAP[j]] = int(cm[i, j])
            analysis[label]['misclassified_as'] = misclassified_as
            
            # How dangerous are the misclassifications?
            # Severe → No Concern is worst; Severe → Moderate is less bad
            if fn > 0:
                analysis[label]['severity_of_errors'] = {
                    'critical_misses': int(cm[i, 0]) if cm.shape[1] > 0 else 0,  # Severe → No Concern
                    'significant_misses': int(cm[i, 1]) if cm.shape[1] > 1 else 0,  # Severe → Mild
                    'minor_misses': int(cm[i, 2]) if cm.shape[1] > 2 else 0,  # Severe → Moderate
                }
    
    return analysis


def format_metrics_report(metrics: Dict) -> str:
    """Format metrics into a human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("  MindGuard — Evaluation Metrics Report")
    lines.append("=" * 60)
    
    lines.append(f"\n📊 Overall Performance:")
    lines.append(f"   Accuracy:    {metrics.get('accuracy', 0):.4f}")
    lines.append(f"   F1 (Macro):  {metrics.get('f1_macro', 0):.4f}")
    lines.append(f"   F1 (Weighted): {metrics.get('f1_weighted', 0):.4f}")
    
    if metrics.get('auc_roc_macro') is not None:
        lines.append(f"   AUC-ROC:     {metrics['auc_roc_macro']:.4f}")
    
    lines.append(f"\n🏥 Per-Class Performance:")
    lines.append(f"   {'Class':>15} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
    lines.append(f"   {'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    for label_name, class_metrics in metrics.get('per_class', {}).items():
        lines.append(
            f"   {label_name:>15} | "
            f"{class_metrics['precision']:>10.4f} | "
            f"{class_metrics['recall']:>10.4f} | "
            f"{class_metrics['f1']:>10.4f}"
        )
    
    # Crisis recall (MOST IMPORTANT)
    if metrics.get('crisis_recall') is not None:
        lines.append(f"\n🚨 CRISIS DETECTION:")
        lines.append(f"   Severe Crisis Recall: {metrics['crisis_recall']:.4f}")
        lines.append(f"   Crises Detected:      {metrics['crisis_total'] - metrics['crisis_missed']}/{metrics['crisis_total']}")
        lines.append(f"   Crises MISSED:        {metrics['crisis_missed']}")
        if metrics['crisis_missed'] > 0:
            lines.append(f"   ⚠️  {metrics['crisis_missed']} severe crisis cases were MISSED by the model!")
    
    # Error analysis for Severe class
    error = metrics.get('error_analysis', {}).get('Severe Crisis', {})
    if error:
        lines.append(f"\n📋 Severe Crisis Error Analysis:")
        lines.append(f"   True Positives:  {error.get('true_positives', 0)}")
        lines.append(f"   False Negatives: {error.get('false_negatives', 0)} (MISSED crises)")
        lines.append(f"   False Positives: {error.get('false_positives', 0)} (false alarms)")
        
        severity = error.get('severity_of_errors', {})
        if severity:
            lines.append(f"   Error Severity Breakdown:")
            lines.append(f"     Critical (→ No Concern): {severity.get('critical_misses', 0)}")
            lines.append(f"     Significant (→ Mild):    {severity.get('significant_misses', 0)}")
            lines.append(f"     Minor (→ Moderate):      {severity.get('minor_misses', 0)}")
    
    lines.append(f"\n{'='*60}")
    lines.append("⚠️  This model is a screening tool ONLY.")
    lines.append("   All predictions must be reviewed by a trained professional.")
    lines.append(f"{'='*60}")
    
    return '\n'.join(lines)
