"""
MindGuard Data Pipeline
=======================
Loads Reddit Mental Health datasets from HuggingFace, preprocesses text,
maps subreddits to CSSRS-aligned severity labels, handles class imbalance,
and creates reproducible train/val/test splits.

Ethical Note: This pipeline processes publicly available Reddit data.
No personally identifiable information is collected or stored.
"""

import os
import re
import sys
import json
import yaml
import hashlib
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for compatibility
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Optional imports with fallbacks
try:
    from datasets import load_dataset, Dataset, DatasetDict
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  `datasets` not installed. Install with: pip install datasets")

try:
    import ftfy
    FTFY_AVAILABLE = True
except ImportError:
    FTFY_AVAILABLE = False

try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================
# Constants
# ============================================================

LABEL_MAP = {
    0: "No Concern",
    1: "Mild",
    2: "Moderate",
    3: "Severe Crisis"
}

# Subreddit → Severity Label mapping (CSSRS-aligned)
SUBREDDIT_TO_LABEL = {
    # Severe Crisis (CSSRS Level 4-5: Active ideation with intent/plan)
    "SuicideWatch": 3,
    
    # Moderate (CSSRS Level 2-3: Passive ideation, significant distress)
    "depression": 2,
    "anxiety": 2,
    
    # Mild (CSSRS Level 1: General distress, wish to be dead without plan)
    "mentalhealth": 1,
    "offmychest": 1,
    "lonely": 1,
    "socialanxiety": 1,
    
    # No Concern (Control group)
    "CasualConversation": 0,
    "AskReddit": 0,
    "happy": 0,
    "self": 0,
}

# Color palette for severity levels
SEVERITY_COLORS = {
    0: "#4CAF50",  # Green — No Concern
    1: "#FFC107",  # Amber — Mild
    2: "#FF9800",  # Orange — Moderate
    3: "#F44336",  # Red — Severe Crisis
}


# ============================================================
# Text Preprocessing
# ============================================================

class TextPreprocessor:
    """
    Cleans and normalizes social media text for NLP processing.
    Preserves emotional signals while removing noise.
    """
    
    # Regex patterns compiled once for performance
    URL_PATTERN = re.compile(r'http[s]?://\S+|www\.\S+')
    USERNAME_PATTERN = re.compile(r'u/\w+|@\w+')
    SUBREDDIT_PATTERN = re.compile(r'r/\w+')
    HTML_PATTERN = re.compile(r'<[^>]+>')
    EXTRA_SPACES = re.compile(r'\s+')
    REPEATED_CHARS = re.compile(r'(.)\1{3,}')  # 4+ repeated chars → 2
    SPECIAL_CHARS = re.compile(r'[^\w\s.,!?\'"-:;()\[\]{}…]')
    NUMBER_PATTERN = re.compile(r'\b\d{4,}\b')  # Remove long numbers (IDs, etc.)
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.remove_urls = self.config.get('remove_urls', True)
        self.remove_usernames = self.config.get('remove_usernames', True)
        self.remove_emojis = self.config.get('remove_emojis', False)
        self.lowercase = self.config.get('lowercase', True)
        self.min_length = self.config.get('min_length', 10)
        self.max_length_chars = 5000  # Prevent extremely long inputs
    
    def clean(self, text: str) -> Optional[str]:
        """
        Full preprocessing pipeline for a single text.
        Returns None if text should be filtered out.
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return None
        
        # Fix Unicode issues
        if FTFY_AVAILABLE:
            text = ftfy.fix_text(text)
        
        # Remove HTML
        text = self.HTML_PATTERN.sub(' ', text)
        
        # Remove URLs
        if self.remove_urls:
            text = self.URL_PATTERN.sub('[URL]', text)
        
        # Remove usernames
        if self.remove_usernames:
            text = self.USERNAME_PATTERN.sub('[USER]', text)
        
        # Remove subreddit mentions
        text = self.SUBREDDIT_PATTERN.sub('[SUBREDDIT]', text)
        
        # Remove long numbers
        text = self.NUMBER_PATTERN.sub('[NUM]', text)
        
        # Handle emojis
        if self.remove_emojis and EMOJI_AVAILABLE:
            text = emoji.replace_emoji(text, replace='')
        elif EMOJI_AVAILABLE:
            text = emoji.demojize(text, delimiters=(' ', ' '))
        
        # Reduce repeated characters (e.g., "nooooo" → "noo")
        text = self.REPEATED_CHARS.sub(r'\1\1', text)
        
        # Normalize whitespace
        text = self.EXTRA_SPACES.sub(' ', text).strip()
        
        # Truncate extremely long texts
        if len(text) > self.max_length_chars:
            text = text[:self.max_length_chars]
        
        # Lowercase (preserve some case for sentiment signals)
        if self.lowercase:
            text = text.lower()
        
        # Filter by minimum word count
        word_count = len(text.split())
        if word_count < self.min_length:
            return None
        
        return text


# ============================================================
# Dataset Loading & Processing
# ============================================================

class MindGuardDataPipeline:
    """
    Complete data pipeline for MindGuard.
    Loads from HuggingFace, preprocesses, labels, balances, and splits.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.preprocessor = TextPreprocessor(
            self.config.get('data', {}).get('preprocessing', {})
        )
        self.raw_data = None
        self.processed_data = None
        self.splits = None
        
        # Create output directories
        self.data_dir = Path(self.config.get('paths', {}).get('data_dir', 'data/processed'))
        self.results_dir = Path(self.config.get('paths', {}).get('results_dir', 'results'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            # Try default locations
            for p in ['config/config.yaml', '../config/config.yaml']:
                if Path(p).exists():
                    config_path = p
                    break
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        print("⚠️  No config file found. Using defaults.")
        return {}
    
    def load_dataset_from_huggingface(self) -> pd.DataFrame:
        """
        Load the Reddit Mental Health Dataset from HuggingFace.
        
        Uses the 'sentimental-mental-health' dataset which contains
        posts from various mental health subreddits.
        """
        if not HF_AVAILABLE:
            raise ImportError("Install `datasets`: pip install datasets")
        
        print("=" * 60)
        print("🔄 Loading Reddit Mental Health Dataset from HuggingFace...")
        print("=" * 60)
        
        all_records = []
        
        # Try multiple dataset sources for robustness
        dataset_sources = [
            {
                "path": "mrjunos/depression-reddit-cleaned",
                "text_col": "text",
                "label_strategy": "depression_binary"
            },
            {
                "path": "solomonn/reddit_mental_health_posts",
                "text_col": "text",
                "label_col": "subreddit",
                "label_strategy": "subreddit"
            },
            {
                "path": "gokalp/reddit-mental-health-dataset",
                "text_col": "text",
                "label_col": "subreddit",
                "label_strategy": "subreddit"
            },
        ]
        
        loaded_any = False
        
        for source in dataset_sources:
            try:
                print(f"\n📥 Trying: {source['path']}...")
                ds = load_dataset(source['path'], trust_remote_code=True)
                
                # Handle different dataset structures
                if isinstance(ds, DatasetDict):
                    for split_name in ds:
                        df_split = ds[split_name].to_pandas()
                        df_split['_source'] = source['path']
                        df_split['_split_origin'] = split_name
                        all_records.append(df_split)
                        print(f"   ✅ Loaded {len(df_split):,} records from '{split_name}' split")
                else:
                    df = ds.to_pandas()
                    df['_source'] = source['path']
                    all_records.append(df)
                    print(f"   ✅ Loaded {len(df):,} records")
                
                loaded_any = True
                
            except Exception as e:
                print(f"   ⚠️  Could not load {source['path']}: {str(e)[:100]}")
                continue
        
        # Fallback: generate synthetic demonstration data if nothing loads
        if not loaded_any:
            print("\n⚠️  Could not load any HuggingFace datasets.")
            print("📝 Generating synthetic demonstration data for pipeline testing...")
            df = self._generate_synthetic_data()
            all_records.append(df)
        
        # Combine all sources
        combined_df = pd.concat(all_records, ignore_index=True)
        print(f"\n📊 Total raw records: {len(combined_df):,}")
        
        self.raw_data = combined_df
        return combined_df
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic demonstration data for pipeline testing.
        This is NOT real mental health data — it's for testing the pipeline only.
        """
        np.random.seed(42)
        
        synthetic_examples = {
            0: [  # No Concern
                "Had a great day at school today, aced my math test and played basketball after",
                "Just finished watching a really good movie with friends, feeling content",
                "Looking forward to the weekend, planning to go hiking with my family",
                "Finally finished my science project, it turned out better than expected",
                "Had a fun conversation with my best friend today about our favorite books",
                "The weather was beautiful today so we had lunch outside at school",
                "Started learning guitar last week and I can already play a few chords",
                "My dog learned a new trick today, so proud of the little guy",
                "Cooked dinner for my family tonight and everyone loved it",
                "Got accepted into the art club at school, really excited about it",
            ] * 50,  # Replicate for volume
            1: [  # Mild
                "Feeling a bit stressed about upcoming exams but trying to manage",
                "Had a rough day, nothing specific just feeling kinda blah about everything",
                "Sometimes I wonder if my friends actually like me or just tolerate me",
                "School has been really overwhelming lately with all the homework",
                "Feel like I am falling behind everyone else academically",
                "Wish I could just take a break from everything for a while",
                "Having trouble sleeping lately, my mind just keeps racing at night",
                "Feeling disconnected from people around me, hard to explain",
                "Some days are just harder than others, today was one of those days",
                "Struggling to find motivation to do anything productive lately",
            ] * 35,
            2: [  # Moderate
                "I have been feeling really down for weeks now and nothing seems to help",
                "Can barely get out of bed most mornings, everything feels pointless",
                "I keep crying for no reason and I do not know what is wrong with me",
                "Feel like nobody would notice if I just disappeared from school",
                "My anxiety is so bad I have not been able to eat properly in days",
                "I am so tired of pretending everything is fine when it is not",
                "Every day feels the same, just going through the motions with no real joy",
                "I have been isolating myself from everyone because I feel like a burden",
                "Cannot stop thinking about all my failures, I am worthless",
                "The sadness just will not go away no matter what I try",
            ] * 20,
            3: [  # Severe Crisis
                "I do not want to be here anymore, the pain is too much",
                "I have been thinking about ending it all and I have a plan",
                "Nobody cares about me, they would all be better off without me",
                "I cannot take this anymore, I just want everything to stop",
                "I have been hurting myself and I do not know how to stop",
                "Life has no meaning for me, I have given up on everything",
                "I wrote goodbye letters to my family last night",
                "The only way out I can see is to end everything permanently",
                "I have been stockpiling pills, I know exactly when I am going to do it",
                "Every morning I wish I had not woken up, I am so done with everything",
            ] * 10,
        }
        
        records = []
        for label, texts in synthetic_examples.items():
            for text in texts:
                records.append({
                    'text': text,
                    'label': label,
                    'subreddit': list(SUBREDDIT_TO_LABEL.keys())[
                        list(SUBREDDIT_TO_LABEL.values()).index(label)
                    ],
                    '_source': 'synthetic_demo',
                    '_is_synthetic': True
                })
        
        df = pd.DataFrame(records)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   📝 Generated {len(df):,} synthetic records for demonstration")
        return df
    
    def assign_severity_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map text to CSSRS-aligned severity labels.
        Uses subreddit source as primary signal, with keyword refinement.
        """
        print("\n🏷️  Assigning CSSRS-aligned severity labels...")
        
        # If labels already exist (synthetic data), skip
        if 'label' in df.columns and df['label'].notna().all():
            print("   Labels already assigned.")
            return df
        
        # Strategy 1: Map by subreddit
        if 'subreddit' in df.columns:
            df['label'] = df['subreddit'].map(SUBREDDIT_TO_LABEL)
        
        # Strategy 2: For datasets with binary labels, use keyword refinement
        if 'label' not in df.columns or df['label'].isna().any():
            df['label'] = df.apply(self._keyword_severity_scorer, axis=1)
        
        # Fill any remaining NaN labels with "Mild" (conservative default)
        df['label'] = df['label'].fillna(1).astype(int)
        
        # Add label names
        df['label_name'] = df['label'].map(LABEL_MAP)
        
        print("   ✅ Labels assigned successfully")
        return df
    
    def _keyword_severity_scorer(self, row) -> int:
        """
        Keyword-based severity scoring for text without subreddit labels.
        This is a heuristic — the model will learn better representations.
        """
        text = str(row.get('text', '')).lower()
        
        severe_keywords = [
            'kill myself', 'end my life', 'suicide', 'want to die',
            'no reason to live', 'better off dead', 'goodbye letter',
            'end it all', 'overdose', 'slit', 'hang myself', 'jump off',
            'final note', 'last day', 'pills', 'bridge'
        ]
        
        moderate_keywords = [
            'depressed', 'hopeless', 'worthless', 'can\'t go on',
            'no point', 'empty inside', 'numb', 'crying all the time',
            'can\'t eat', 'can\'t sleep', 'self harm', 'cutting',
            'panic attack', 'breakdown', 'falling apart'
        ]
        
        mild_keywords = [
            'stressed', 'anxious', 'overwhelmed', 'worried',
            'lonely', 'sad', 'frustrated', 'struggling',
            'burnout', 'exhausted', 'insomnia', 'isolated'
        ]
        
        # Score based on keyword presence
        for kw in severe_keywords:
            if kw in text:
                return 3
        
        for kw in moderate_keywords:
            if kw in text:
                return 2
        
        for kw in mild_keywords:
            if kw in text:
                return 1
        
        return 0
    
    def preprocess_texts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply text preprocessing pipeline to all texts."""
        print("\n🧹 Preprocessing text...")
        
        # Find the text column
        text_col = None
        for col in ['text', 'body', 'selftext', 'title', 'post', 'content']:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            raise ValueError(f"No text column found in data. Available: {list(df.columns)}")
        
        # Rename to standardized 'text' column
        if text_col != 'text':
            df = df.rename(columns={text_col: 'text'})
        
        # Apply preprocessing
        tqdm.pandas(desc="   Cleaning text")
        df['text_clean'] = df['text'].progress_apply(self.preprocessor.clean)
        
        # Count removed
        original_count = len(df)
        df = df.dropna(subset=['text_clean']).reset_index(drop=True)
        removed = original_count - len(df)
        
        print(f"   ✅ Preprocessed {len(df):,} texts ({removed:,} removed as too short/empty)")
        
        return df
    
    def create_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create stratified train/val/test splits.
        Ensures each split has proportional representation of all severity levels.
        """
        from sklearn.model_selection import train_test_split
        
        seed = self.config.get('data', {}).get('seed', 42)
        split_ratios = self.config.get('data', {}).get('splits', {})
        train_ratio = split_ratios.get('train', 0.8)
        val_ratio = split_ratios.get('val', 0.1)
        test_ratio = split_ratios.get('test', 0.1)
        
        print(f"\n📂 Creating stratified splits ({train_ratio}/{val_ratio}/{test_ratio})...")
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, test_size=(val_ratio + test_ratio),
            stratify=df['label'], random_state=seed
        )
        
        # Second split: val vs test
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, test_size=relative_test_ratio,
            stratify=temp_df['label'], random_state=seed
        )
        
        splits = {
            'train': train_df.reset_index(drop=True),
            'val': val_df.reset_index(drop=True),
            'test': test_df.reset_index(drop=True)
        }
        
        for name, split_df in splits.items():
            print(f"   {name:>5}: {len(split_df):>6,} samples")
        
        self.splits = splits
        return splits
    
    def compute_class_weights(self, df: pd.DataFrame) -> Dict[int, float]:
        """
        Compute class weights for handling imbalance.
        Uses inverse frequency * CSSRS severity multipliers.
        Missing a crisis (class 3) is penalized MORE heavily.
        """
        label_counts = df['label'].value_counts().sort_index()
        total = len(df)
        n_classes = len(label_counts)
        
        # Inverse frequency weights
        weights = {}
        for label, count in label_counts.items():
            weights[label] = total / (n_classes * count)
        
        # Apply the asymmetric severity multipliers from config
        multipliers = self.config.get('training', {}).get(
            'class_weight_multipliers', {0: 1.0, 1: 1.5, 2: 2.5, 3: 5.0}
        )
        
        for label in weights:
            weights[label] *= multipliers.get(label, 1.0)
        
        # Normalize so the minimum weight is 1.0
        min_weight = min(weights.values())
        weights = {k: v / min_weight for k, v in weights.items()}
        
        print("\n⚖️  Class weights (asymmetric — crisis-aware):")
        for label, weight in sorted(weights.items()):
            print(f"   {LABEL_MAP[label]:>15}: {weight:.3f}  "
                  f"(count: {label_counts[label]:,})")
        
        return weights
    
    def analyze_class_distribution(self, df: pd.DataFrame, save: bool = True) -> dict:
        """
        Analyze and visualize class distribution.
        Produces publication-quality charts.
        """
        print("\n📊 Analyzing class distribution...")
        
        label_counts = df['label'].value_counts().sort_index()
        label_names = [LABEL_MAP[i] for i in label_counts.index]
        label_colors = [SEVERITY_COLORS[i] for i in label_counts.index]
        
        # Statistics
        stats = {
            'total_samples': len(df),
            'class_counts': {LABEL_MAP[k]: int(v) for k, v in label_counts.items()},
            'class_percentages': {
                LABEL_MAP[k]: round(v / len(df) * 100, 2) 
                for k, v in label_counts.items()
            },
            'imbalance_ratio': float(label_counts.max() / label_counts.min()),
        }
        
        # Print summary
        print(f"\n   {'='*55}")
        print(f"   {'CLASS DISTRIBUTION SUMMARY':^55}")
        print(f"   {'='*55}")
        print(f"   Total Samples: {stats['total_samples']:,}")
        print(f"   Imbalance Ratio: {stats['imbalance_ratio']:.1f}:1")
        print(f"   {'-'*55}")
        
        for label_id in sorted(label_counts.index):
            name = LABEL_MAP[label_id]
            count = label_counts[label_id]
            pct = count / len(df) * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"   {name:>15} | {bar} | {count:>6,} ({pct:5.1f}%)")
        
        print(f"   {'='*55}")
        
        if save:
            self._plot_distribution(label_counts, label_names, label_colors, stats)
            self._plot_text_length_distribution(df)
        
        return stats
    
    def _plot_distribution(self, label_counts, label_names, label_colors, stats):
        """Generate publication-quality distribution plots."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('MindGuard — Class Distribution Analysis', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        # 1. Bar chart
        ax1 = axes[0]
        bars = ax1.bar(label_names, label_counts.values, color=label_colors, 
                       edgecolor='white', linewidth=1.5)
        ax1.set_title('Sample Count by Severity', fontweight='bold')
        ax1.set_ylabel('Number of Samples')
        ax1.set_xlabel('Severity Level')
        
        # Add value labels on bars
        for bar, count in zip(bars, label_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax1.tick_params(axis='x', rotation=15)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Pie chart
        ax2 = axes[1]
        wedges, texts, autotexts = ax2.pie(
            label_counts.values, labels=label_names, colors=label_colors,
            autopct='%1.1f%%', startangle=90, pctdistance=0.85,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        ax2.set_title('Proportion by Severity', fontweight='bold')
        
        # 3. Imbalance visualization
        ax3 = axes[2]
        ratios = label_counts.values / label_counts.values.min()
        bars3 = ax3.barh(label_names, ratios, color=label_colors,
                        edgecolor='white', linewidth=1.5)
        ax3.set_title('Relative Imbalance Ratio', fontweight='bold')
        ax3.set_xlabel('Ratio (relative to smallest class)')
        
        for bar, ratio in zip(bars3, ratios):
            ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2.,
                    f'{ratio:.1f}x', va='center', fontweight='bold')
        
        ax3.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.results_dir / 'class_distribution.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"\n   📈 Distribution plot saved to: {save_path}")
    
    def _plot_text_length_distribution(self, df: pd.DataFrame):
        """Plot text length distribution by severity level."""
        text_col = 'text_clean' if 'text_clean' in df.columns else 'text'
        df['word_count'] = df[text_col].apply(lambda x: len(str(x).split()))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Text Length Distribution by Severity', 
                     fontsize=14, fontweight='bold')
        
        # Histogram
        ax1 = axes[0]
        for label_id in sorted(df['label'].unique()):
            subset = df[df['label'] == label_id]
            ax1.hist(subset['word_count'], bins=50, alpha=0.6,
                    label=LABEL_MAP[label_id], color=SEVERITY_COLORS[label_id])
        ax1.set_xlabel('Word Count')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Word Count Distribution')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Box plot
        ax2 = axes[1]
        box_data = [df[df['label'] == l]['word_count'].values for l in sorted(df['label'].unique())]
        bp = ax2.boxplot(box_data, labels=[LABEL_MAP[l] for l in sorted(df['label'].unique())],
                        patch_artist=True)
        for patch, label_id in zip(bp['boxes'], sorted(df['label'].unique())):
            patch.set_facecolor(SEVERITY_COLORS[label_id])
            patch.set_alpha(0.7)
        ax2.set_ylabel('Word Count')
        ax2.set_title('Word Count by Severity Level')
        ax2.tick_params(axis='x', rotation=15)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.results_dir / 'text_length_distribution.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"   📈 Text length distribution saved to: {save_path}")
    
    def save_processed_data(self, splits: Dict[str, pd.DataFrame]):
        """Save processed splits to disk."""
        print(f"\n💾 Saving processed data to {self.data_dir}/...")
        
        for name, df in splits.items():
            # Save as CSV
            csv_path = self.data_dir / f'{name}.csv'
            df.to_csv(csv_path, index=False)
            print(f"   {name}: {csv_path} ({len(df):,} samples)")
        
        # Save metadata
        metadata = {
            'total_samples': sum(len(df) for df in splits.values()),
            'split_sizes': {name: len(df) for name, df in splits.items()},
            'label_map': LABEL_MAP,
            'columns': list(splits['train'].columns),
        }
        
        meta_path = self.data_dir / 'metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   📋 Metadata saved to: {meta_path}")
    
    def run(self) -> Tuple[Dict[str, pd.DataFrame], dict]:
        """
        Execute the full data pipeline.
        Returns (splits_dict, distribution_stats).
        """
        print("\n" + "🧠 " * 20)
        print("  MindGuard Data Pipeline")
        print("  Privacy-Preserving Mental Health Crisis Detection")
        print("🧠 " * 20 + "\n")
        
        # Step 1: Load dataset
        df = self.load_dataset_from_huggingface()
        
        # Step 2: Preprocess text
        df = self.preprocess_texts(df)
        
        # Step 3: Assign severity labels
        df = self.assign_severity_labels(df)
        
        # Step 4: Analyze distribution
        stats = self.analyze_class_distribution(df)
        
        # Step 5: Compute class weights
        class_weights = self.compute_class_weights(df)
        stats['class_weights'] = {LABEL_MAP[k]: float(v) for k, v in class_weights.items()}
        
        # Step 6: Create stratified splits
        splits = self.create_splits(df)
        
        # Step 7: Save everything
        self.save_processed_data(splits)
        
        # Save stats
        stats_path = self.results_dir / 'data_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"\n   📊 Stats saved to: {stats_path}")
        
        print("\n" + "=" * 60)
        print("✅ Data Pipeline Complete!")
        print("=" * 60)
        
        # Print what the model CANNOT do (ethical requirement)
        print("\n⚠️  IMPORTANT — What This Data CANNOT Tell Us:")
        print("   • It cannot diagnose mental health conditions")
        print("   • Subreddit ≠ clinical severity — this is a proxy label")
        print("   • Self-reported text has inherent noise and bias")
        print("   • The model trained on this data is a screening tool ONLY")
        print("   • All predictions MUST be reviewed by a trained professional")
        
        return splits, stats


# ============================================================
# Entry Point
# ============================================================

def main():
    """Run the data pipeline as a standalone script."""
    # Determine config path
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / 'config' / 'config.yaml'
    
    if not config_path.exists():
        config_path = None
    
    pipeline = MindGuardDataPipeline(config_path=str(config_path) if config_path else None)
    splits, stats = pipeline.run()
    
    return splits, stats


if __name__ == '__main__':
    main()
