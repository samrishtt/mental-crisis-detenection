"""
MindGuard Training Loop
=======================
Complete training pipeline with:
- Weighted loss for class imbalance
- Asymmetric loss (crisis-aware)
- Early stopping
- Learning rate scheduling
- Mixed precision training
- Comprehensive logging and evaluation

Ethical Note: This trains a screening model, not a diagnostic tool.
"""

import os
import sys
import json
import time
import yaml
import copy
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.mental_bert import MindGuardClassifier, get_tokenizer
from model.loss import AsymmetricCrisisLoss, CombinedCrisisLoss
from training.metrics import compute_metrics, format_metrics_report, LABEL_MAP


# ============================================================
# Dataset
# ============================================================

class MindGuardDataset(Dataset):
    """PyTorch Dataset for MindGuard text classification."""
    
    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer,
        max_length: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:
    """Stop training when validation metric stops improving."""
    
    def __init__(self, patience: int = 3, mode: str = 'max', min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_model_state = None
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
    
    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta


# ============================================================
# Trainer
# ============================================================

class MindGuardTrainer:
    """
    Complete training pipeline for MindGuard.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🖥️  Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        
        # Paths
        self.checkpoint_dir = Path(self.config.get('paths', {}).get('model_dir', 'checkpoints'))
        self.results_dir = Path(self.config.get('paths', {}).get('results_dir', 'results'))
        self.data_dir = Path(self.config.get('paths', {}).get('data_dir', 'data/processed'))
        
        for d in [self.checkpoint_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Training config
        train_cfg = self.config.get('training', {})
        self.batch_size = train_cfg.get('batch_size', 16)
        self.lr = train_cfg.get('learning_rate', 2e-5)
        self.num_epochs = train_cfg.get('num_epochs', 10)
        self.warmup_ratio = train_cfg.get('warmup_ratio', 0.1)
        self.max_grad_norm = train_cfg.get('max_grad_norm', 1.0)
        self.weight_decay = train_cfg.get('weight_decay', 0.01)
        self.fp16 = train_cfg.get('fp16', True) and self.device.type == 'cuda'
        self.grad_accum = train_cfg.get('gradient_accumulation_steps', 2)
        
        # Early stopping
        es_cfg = train_cfg.get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get('patience', 3),
            mode=es_cfg.get('mode', 'max'),
        )
        
        # Initialize model and tokenizer
        model_cfg = self.config.get('model', {})
        self.model_name = model_cfg.get('name', 'mental/mental-bert-base-uncased')
        self.max_length = self.config.get('data', {}).get('preprocessing', {}).get('max_length', 256)
        
        print(f"\n🔧 Loading tokenizer: {self.model_name}")
        self.tokenizer = get_tokenizer(self.model_name)
        
        print(f"🧠 Loading model: {self.model_name}")
        self.model = MindGuardClassifier(
            model_name=self.model_name,
            num_labels=model_cfg.get('num_labels', 4),
            dropout=model_cfg.get('dropout', 0.3),
            freeze_embeddings=model_cfg.get('freeze_embeddings', False),
            freeze_lower_layers=model_cfg.get('freeze_lower_layers', 0),
            use_lora=model_cfg.get('use_lora', True),
        ).to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': [],
            'val_crisis_recall': [],
            'learning_rates': [],
        }
    
    def _load_config(self, config_path):
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        for p in ['config/config.yaml', '../config/config.yaml']:
            if Path(p).exists():
                with open(p, 'r') as f:
                    return yaml.safe_load(f)
        return {}
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load processed data and create DataLoaders."""
        print("\n📂 Loading processed data...")
        
        train_df = pd.read_csv(self.data_dir / 'train.csv')
        val_df = pd.read_csv(self.data_dir / 'val.csv')
        test_df = pd.read_csv(self.data_dir / 'test.csv')
        
        text_col = 'text_clean' if 'text_clean' in train_df.columns else 'text'
        
        # Create datasets
        train_ds = MindGuardDataset(
            train_df[text_col].tolist(), train_df['label'].tolist(),
            self.tokenizer, self.max_length
        )
        val_ds = MindGuardDataset(
            val_df[text_col].tolist(), val_df['label'].tolist(),
            self.tokenizer, self.max_length
        )
        test_ds = MindGuardDataset(
            test_df[text_col].tolist(), test_df['label'].tolist(),
            self.tokenizer, self.max_length
        )
        
        # Create DataLoaders
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, 
                                   num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size * 2, shuffle=False,
                                 num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size * 2, shuffle=False,
                                  num_workers=0, pin_memory=True)
        
        print(f"   Train: {len(train_ds):,} samples ({len(train_loader)} batches)")
        print(f"   Val:   {len(val_ds):,} samples ({len(val_loader)} batches)")
        print(f"   Test:  {len(test_ds):,} samples ({len(test_loader)} batches)")
        
        # Compute class weights from training data
        label_counts = train_df['label'].value_counts().sort_index()
        n_classes = len(label_counts)
        total = len(train_df)
        
        weights = torch.zeros(n_classes)
        multipliers = self.config.get('training', {}).get(
            'class_weight_multipliers', {0: 1.0, 1: 1.5, 2: 2.5, 3: 5.0}
        )
        
        for label, count in label_counts.items():
            weights[label] = (total / (n_classes * count)) * multipliers.get(label, 1.0)
        
        # Normalize
        weights = weights / weights.min()
        self.class_weights = weights.to(self.device)
        
        print(f"\n⚖️  Class weights: {[f'{w:.2f}' for w in self.class_weights.tolist()]}")
        
        return train_loader, val_loader, test_loader
    
    def setup_training(self, train_loader: DataLoader):
        """Set up optimizer, scheduler, and loss function."""
        # Optimizer with weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        optimizer_grouped_params = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        self.optimizer = AdamW(optimizer_grouped_params, lr=self.lr)
        
        # Learning rate scheduler
        total_steps = len(train_loader) * self.num_epochs // self.grad_accum
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Loss function — asymmetric, crisis-aware
        train_cfg = self.config.get('training', {}).get('asymmetric_loss', {})
        self.criterion = CombinedCrisisLoss(
            class_weights=self.class_weights,
            fn_weight=train_cfg.get('fn_weight', 3.0),
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.fp16)
        
        print(f"\n⚙️  Training Setup:")
        print(f"   Optimizer:       AdamW (lr={self.lr}, wd={self.weight_decay})")
        print(f"   Scheduler:       Linear warmup ({warmup_steps} steps)")
        print(f"   Loss:            Combined (Asymmetric CE + Focal)")
        print(f"   Mixed Precision: {'✅' if self.fp16 else '❌'}")
        print(f"   Grad Accum:      {self.grad_accum} steps")
        print(f"   Total Steps:     {total_steps}")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        
        pbar = tqdm(train_loader, desc=f"   Epoch {epoch+1}/{self.num_epochs} [Train]",
                     leave=True)
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with autocast(enabled=self.fp16):
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs['logits'], labels)
                loss = loss / self.grad_accum
            
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.grad_accum
            
            preds = torch.argmax(outputs['logits'], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f"{loss.item() * self.grad_accum:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / len(train_loader)
        metrics = compute_metrics(
            np.array(all_labels), np.array(all_preds), num_classes=4
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split_name: str = "Val") -> Dict:
        """Evaluate model on a data split."""
        self.model.eval()
        total_loss = 0
        all_preds, all_labels, all_probs = [], [], []
        
        pbar = tqdm(loader, desc=f"   [{split_name}]", leave=True)
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with autocast(enabled=self.fp16):
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs['logits'], labels)
            
            total_loss += loss.item()
            
            probs = torch.softmax(outputs['logits'], dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        metrics = compute_metrics(
            np.array(all_labels), np.array(all_preds),
            np.array(all_probs), num_classes=4
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def train(self):
        """Full training loop."""
        print("\n" + "=" * 60)
        print("  🧠 MindGuard Training Pipeline")
        print("  Crisis-Aware Mental Health Severity Classification")
        print("=" * 60)
        
        # Load data
        train_loader, val_loader, test_loader = self.load_data()
        
        # Setup
        self.setup_training(train_loader)
        
        # Training
        print(f"\n🏋️ Starting training for {self.num_epochs} epochs...")
        best_val_f1 = 0
        
        for epoch in range(self.num_epochs):
            print(f"\n{'─' * 50}")
            print(f"  Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'─' * 50}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.evaluate(val_loader, "Val")
            
            # Log
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_f1'].append(train_metrics['f1_macro'])
            self.history['val_f1'].append(val_metrics['f1_macro'])
            self.history['val_crisis_recall'].append(val_metrics.get('crisis_recall'))
            self.history['learning_rates'].append(self.scheduler.get_last_lr()[0])
            
            print(f"\n   📊 Epoch {epoch+1} Summary:")
            print(f"   Train Loss: {train_metrics['loss']:.4f}  |  Val Loss: {val_metrics['loss']:.4f}")
            print(f"   Train F1:   {train_metrics['f1_macro']:.4f}  |  Val F1:   {val_metrics['f1_macro']:.4f}")
            if val_metrics.get('crisis_recall') is not None:
                print(f"   🚨 Crisis Recall: {val_metrics['crisis_recall']:.4f}")
            
            # Early stopping
            self.early_stopping(val_metrics['f1_macro'], self.model)
            
            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                self.model.save_model(str(self.checkpoint_dir / 'best_model'))
                print(f"   💾 New best model saved! (F1: {best_val_f1:.4f})")
            
            if self.early_stopping.should_stop:
                print(f"\n   ⏹️ Early stopping triggered at epoch {epoch+1}")
                break
        
        # Restore best model
        if self.early_stopping.best_model_state is not None:
            self.model.load_state_dict(self.early_stopping.best_model_state)
            print("\n   📂 Restored best model weights")
        
        # Final evaluation on test set
        print(f"\n{'=' * 60}")
        print("  📋 Final Evaluation on Test Set")
        print(f"{'=' * 60}")
        
        test_metrics = self.evaluate(test_loader, "Test")
        report = format_metrics_report(test_metrics)
        print(report)
        
        # Save results
        self._save_results(test_metrics)
        self._plot_training_curves()
        
        return test_metrics
    
    def _save_results(self, test_metrics: Dict):
        """Save training results and metrics."""
        results = {
            'test_metrics': {k: v for k, v in test_metrics.items() 
                           if not isinstance(v, np.ndarray)},
            'training_history': self.history,
            'config': self.config,
        }
        
        results_path = self.results_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n   📊 Results saved to: {results_path}")
    
    def _plot_training_curves(self):
        """Plot training and validation curves."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('MindGuard Training Curves', fontsize=14, fontweight='bold')
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        axes[0].plot(epochs, self.history['train_loss'], 'b-o', label='Train', markersize=4)
        axes[0].plot(epochs, self.history['val_loss'], 'r-o', label='Validation', markersize=4)
        axes[0].set_title('Loss', fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # F1 curves
        axes[1].plot(epochs, self.history['train_f1'], 'b-o', label='Train', markersize=4)
        axes[1].plot(epochs, self.history['val_f1'], 'r-o', label='Validation', markersize=4)
        axes[1].set_title('F1 Score (Macro)', fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # Crisis recall
        crisis_recalls = [r for r in self.history['val_crisis_recall'] if r is not None]
        if crisis_recalls:
            crisis_epochs = [i+1 for i, r in enumerate(self.history['val_crisis_recall']) if r is not None]
            axes[2].plot(crisis_epochs, crisis_recalls, 'r-o', label='Crisis Recall', markersize=4)
            axes[2].axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Target (0.95)')
            axes[2].set_title('🚨 Crisis Recall', fontweight='bold')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Recall')
            axes[2].legend()
            axes[2].grid(alpha=0.3)
            axes[2].set_ylim([0, 1.05])
        
        plt.tight_layout()
        save_path = self.results_dir / 'training_curves.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   📈 Training curves saved to: {save_path}")


# ============================================================
# Entry Point
# ============================================================

def main():
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / 'config' / 'config.yaml'
    
    trainer = MindGuardTrainer(
        config_path=str(config_path) if config_path.exists() else None
    )
    test_metrics = trainer.train()
    
    return test_metrics


if __name__ == '__main__':
    main()
