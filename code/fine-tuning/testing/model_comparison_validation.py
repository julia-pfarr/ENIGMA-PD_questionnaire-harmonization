"""
Model Comparison Validation Analysis
=====================================

Compares fine-tuned harmonizer models vs base embedding models on unseen
questionnaires from the second round of expert survey (validation set).

For each construct:
  1. Fine-tuned ContrastiveModel predicts dimension assignments
  2. Base SentenceTransformer predicts via cosine similarity to dimension descriptions
  3. Both are evaluated against expert consensus (round-2 survey)
  4. Cohen's kappa and accuracy are reported and compared

Outputs (saved to output/fine-tuning/testing/):
  validation_predictions_all.csv   -- per-item predictions from both models
  validation_kappa_comparison.csv  -- summary metrics per construct
  validation_perdim_accuracy.csv   -- per-dimension accuracy breakdown
  figures/                         -- comparison plots

Usage:
  python model_comparison_validation.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
FT_CODE_DIR  = SCRIPT_DIR.parent / 'semantic-learning_harmonize'
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_DIMENSION_CONFIG = PROJECT_ROOT / 'input' / 'shared' / 'dimension_descriptions.yaml'

sys.path.insert(0, str(FT_CODE_DIR))

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

INPUT_ITEMS     = PROJECT_ROOT / 'input'  / 'model-testing'   / 'validation_complete_round2.csv'
EXPERT_ROUND2   = PROJECT_ROOT / 'output' / 'expert-survey'   / 'round-2' / 'item_assignments_round2.csv'
MODELS_DIR      = PROJECT_ROOT / 'output' / 'fine-tuning'     / 'models-final'
OUTPUT_DIR      = PROJECT_ROOT / 'output' / 'fine-tuning'     / 'testing' / 'base-vs-finetuned'
FIGURES_DIR     = OUTPUT_DIR / 'figures'
INVENTORY_PATH  = PROJECT_ROOT / 'output' / 'fine-tuning' / 'harmonized_inventory.csv'

# kNN blending configuration
# knn_alpha: 1.0 = pure prototype, 0.0 = pure kNN, 0.5 = equal blend
KNN_ALPHA = {
    'depression':     0.5,
    'apathy':         1.0,
    'psychosis':      1.0,
    'anxiety':        1.0,
    'sleep':          1.0,
    'impulse_control': 1.0,
}
KNN_K = 5

# ---------------------------------------------------------------------------
# Configuration: best base model per construct (from original analysis)
# ---------------------------------------------------------------------------

BASE_MODELS = {
    'depression':     'BAAI/bge-large-en-v1.5',
    'apathy':         'BAAI/bge-large-en-v1.5',
    'psychosis':      'hkunlp/instructor-large',
    'anxiety':        'sentence-transformers/gtr-t5-large',
    'sleep':          'princeton-nlp/sup-simcse-roberta-large',
    'impulse_control':'sentence-transformers/paraphrase-MiniLM-L6-v2',
}

def load_dimension_descriptions(path: Path = DEFAULT_DIMENSION_CONFIG) -> dict:
    """Load construct -> dimension_id -> {label, description, keywords} from
    the shared dimension config.

    To adapt this pipeline to your own constructs, edit a copy of
    input/shared/dimension_descriptions.yaml and pass its path here --
    no need to touch this script.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dimension config not found at {path}. This file defines the "
            "constructs/dimensions items get assigned to -- see "
            "input/shared/dimension_descriptions.yaml for the expected format."
        )
    with open(path) as f:
        config = yaml.safe_load(f)
    return config['dimension_descriptions']


# Dimension descriptions used in original analysis (identical to expert survey prompt)
DIMENSION_DESCRIPTIONS = load_dimension_descriptions()

# Instruction strings for instructor-style models
INSTRUCTOR_ITEM_INSTRUCTION = "Represent this mental health questionnaire item for semantic similarity:"
INSTRUCTOR_DIM_INSTRUCTION  = "Represent this mental health symptom dimension description:"


# ===========================================================================
# Data loading
# ===========================================================================

def load_validation_items(path: Path) -> pd.DataFrame:
    """Load validation items from validation_test.csv."""
    df = pd.read_csv(path)
    return df


def load_expert_assignments(path: Path) -> pd.DataFrame:
    """
    Load round-2 expert assignments.
    Normalises the 'pychosis' typo to 'psychosis'.
    """
    df = pd.read_csv(path)
    df['Construct'] = df['Construct'].str.strip().replace({'pychosis': 'psychosis'})
    return df


def get_expert_ground_truth(expert_df: pd.DataFrame, construct: str) -> pd.DataFrame:
    """Return item_id / expert_dimension / metadata for one construct."""
    sub = expert_df[expert_df['Construct'] == construct].copy()
    sub = sub.rename(columns={
        'Item ID':          'item_id',
        'Question Text':    'item_text',
        'Questionnaire':    'questionnaire',
        'Assigned Dimension': 'expert_dimension',
        'Dimension Label':  'expert_label',
        'P_i':              'expert_p_i',
    })
    return sub[['item_id', 'questionnaire', 'item_text', 'expert_dimension', 'expert_label', 'expert_p_i']].copy()


# ===========================================================================
# Base model predictions
# ===========================================================================

def predict_base_model(
    item_texts: list,
    construct: str,
    model_name: str,
) -> tuple:
    """
    Assign items to dimensions using cosine similarity to dimension descriptions.

    Returns
    -------
    assignments     : np.ndarray of dimension IDs (int)
    all_similarities: np.ndarray shape (n_items, n_dims)
    dim_ids         : list of int
    dim_labels      : list of str
    """
    dim_defs   = DIMENSION_DESCRIPTIONS[construct]
    dim_ids    = sorted(dim_defs.keys())
    dim_labels = [dim_defs[d]['label'] for d in dim_ids]
    dim_texts  = [f"{dim_defs[d]['label']}: {dim_defs[d]['description']}" for d in dim_ids]

    print(f"    Loading base model: {model_name}")
    model = SentenceTransformer(model_name)

    is_instructor = 'instructor' in model_name.lower()

    if is_instructor:
        item_inputs = [[INSTRUCTOR_ITEM_INSTRUCTION, t] for t in item_texts]
        dim_inputs  = [[INSTRUCTOR_DIM_INSTRUCTION, t]  for t in dim_texts]
        item_embs   = model.encode(item_inputs, show_progress_bar=True,  convert_to_numpy=True)
        dim_embs    = model.encode(dim_inputs,  show_progress_bar=False, convert_to_numpy=True)
    else:
        item_embs = model.encode(item_texts, show_progress_bar=True,  convert_to_numpy=True)
        dim_embs  = model.encode(dim_texts,  show_progress_bar=False, convert_to_numpy=True)

    # Normalise and compute cosine similarity
    item_embs = item_embs / (np.linalg.norm(item_embs, axis=1, keepdims=True) + 1e-10)
    dim_embs  = dim_embs  / (np.linalg.norm(dim_embs,  axis=1, keepdims=True) + 1e-10)
    similarities = item_embs @ dim_embs.T   # (n_items, n_dims)

    best_idx    = np.argmax(similarities, axis=1)
    assignments = np.array([dim_ids[i] for i in best_idx])

    return assignments, similarities, dim_ids, dim_labels


# ===========================================================================
# Base model + kNN predictions
# ===========================================================================

def predict_base_knn(
    item_texts: list,
    construct: str,
    model_name: str,
    dim_ids: list,
    knn_k: int = KNN_K,
    inventory_path: Path = INVENTORY_PATH,
) -> tuple:
    """
    Predict dimensions using base model embeddings + kNN against inventory items.

    Unlike predict_base_model (cosine similarity to description text), this
    uses the base model to encode real labeled inventory items and classifies
    by weighted nearest-neighbour vote.

    Returns
    -------
    predictions   : np.ndarray of dimension IDs (int)
    probabilities : np.ndarray shape (n_items, n_dims)
    pool_size     : int
    """
    inv = pd.read_csv(inventory_path, skipinitialspace=True)
    inv = inv[inv['construct'] == construct].copy()
    if 'item_text' not in inv.columns:
        # item_text is stripped from the public harmonized_inventory.csv (copyright)
        raise ValueError(
            f"harmonized_inventory.csv has no 'item_text' column (stripped for "
            f"copyright) -- cannot build a kNN pool for construct '{construct}'"
        )
    inv['dimension'] = pd.to_numeric(inv['dimension'], errors='coerce')
    inv = inv[inv['dimension'].isin(dim_ids)].dropna(subset=['item_text', 'dimension'])

    if len(inv) == 0:
        raise ValueError(f"No inventory items found for construct '{construct}'")

    pool_texts = inv['item_text'].tolist()
    pool_dims  = inv['dimension'].astype(int).tolist()
    print(f"    Loading base model for kNN: {model_name}  (pool={len(pool_dims)} items)")

    model = SentenceTransformer(model_name)
    is_instructor = 'instructor' in model_name.lower()

    if is_instructor:
        item_inputs = [[INSTRUCTOR_ITEM_INSTRUCTION, t] for t in item_texts]
        pool_inputs = [[INSTRUCTOR_ITEM_INSTRUCTION, t] for t in pool_texts]
        item_embs = model.encode(item_inputs, convert_to_numpy=True, show_progress_bar=True)
        pool_embs = model.encode(pool_inputs, convert_to_numpy=True, show_progress_bar=False)
    else:
        item_embs = model.encode(item_texts, convert_to_numpy=True, show_progress_bar=True)
        pool_embs = model.encode(pool_texts, convert_to_numpy=True, show_progress_bar=False)

    item_embs = item_embs / (np.linalg.norm(item_embs, axis=1, keepdims=True) + 1e-10)
    pool_embs = pool_embs / (np.linalg.norm(pool_embs, axis=1, keepdims=True) + 1e-10)

    proba       = knn_predict_proba(item_embs, pool_embs, pool_dims, dim_ids, k=knn_k)
    predictions = np.array([dim_ids[int(np.argmax(proba[i]))] for i in range(len(item_texts))])

    return predictions, proba, len(pool_dims)


# ===========================================================================
# Fine-tuned model predictions
# ===========================================================================

import contextlib
import json as _json

@contextlib.contextmanager
def _fix_tokenizer_config(encoder_dir: Path):
    """
    Temporarily convert extra_special_tokens from list → dict in tokenizer_config.json.

    Older sentence-transformers saved T5 sentinel tokens as a JSON list; newer
    transformers versions call .keys() on this value and crash. Since we only
    use these encoders for inference (not seq2seq generation), an empty dict is
    safe and correct.
    """
    config_path = encoder_dir / 'tokenizer_config.json'
    if not config_path.exists():
        yield
        return

    with open(config_path) as f:
        original_text = f.read()
    data = _json.loads(original_text)

    if isinstance(data.get('extra_special_tokens'), list):
        data['extra_special_tokens'] = {}
        with open(config_path, 'w') as f:
            _json.dump(data, f, indent=2)
        try:
            yield
        finally:
            with open(config_path, 'w') as f:
                f.write(original_text)
    else:
        yield


def predict_finetuned_model(
    item_texts: list,
    construct: str,
    models_dir: Path,
) -> tuple:
    """
    Load and run the fine-tuned model directly (bypasses inventory lookup).
    Tries ContrastiveModel first; falls back to PrototypeModel.

    Returns
    -------
    predictions  : np.ndarray of dimension IDs (int)
    probabilities: np.ndarray shape (n_items, n_dims)
    dimensions   : list of dimension IDs
    dim_labels   : list of str
    """
    from semantic_learning import ContrastiveModel, PrototypeModel

    contrastive_path = models_dir / construct / 'contrastive_model.pkl'
    prototype_path   = models_dir / construct / 'prototype_model.pkl'

    if contrastive_path.exists():
        model_path   = contrastive_path
        encoder_dir  = models_dir / construct / 'contrastive_model_encoder'
        model_class  = ContrastiveModel
        model_type   = 'contrastive'
    elif prototype_path.exists():
        model_path   = prototype_path
        encoder_dir  = models_dir / construct / 'prototype_model_encoder'
        model_class  = PrototypeModel
        model_type   = 'prototype'
    else:
        raise FileNotFoundError(
            f"No fine-tuned model found for {construct} in {models_dir / construct}"
        )

    print(f"    Loading fine-tuned {model_type} model from: {model_path}")
    with _fix_tokenizer_config(encoder_dir):
        model = model_class.load(str(model_path), device='cpu')

    predictions   = model.predict(item_texts)
    probabilities = model.predict_proba(item_texts)

    return predictions, probabilities, model.dimensions, model.dimension_labels, model


# ===========================================================================
# kNN blending helpers
# ===========================================================================

def build_knn_pool(
    construct: str,
    encoder,
    dimensions: list,
    inventory_path: Path = INVENTORY_PATH,
) -> tuple:
    """
    Load inventory items for a given construct and encode them with the
    fine-tuned encoder.

    Returns
    -------
    pool_embs : np.ndarray  shape (n_pool, emb_dim), L2-normalised
    pool_dims : list of int  corresponding dimension IDs
    n_pool    : int          number of items in pool (for logging)
    """
    inv = pd.read_csv(inventory_path, skipinitialspace=True)
    inv = inv[inv['construct'] == construct].copy()
    if 'item_text' not in inv.columns:
        # item_text is stripped from the public harmonized_inventory.csv
        # (copyright); no pool to draw neighbours from without it.
        return None, None, 0
    # Keep only rows whose dimension is among the model's known dimensions
    inv['dimension'] = pd.to_numeric(inv['dimension'], errors='coerce')
    inv = inv[inv['dimension'].isin(dimensions)].dropna(subset=['item_text', 'dimension'])

    if len(inv) == 0:
        return None, None, 0

    texts     = inv['item_text'].tolist()
    pool_dims = inv['dimension'].astype(int).tolist()

    pool_embs = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    pool_embs = pool_embs / (np.linalg.norm(pool_embs, axis=1, keepdims=True) + 1e-10)

    return pool_embs, pool_dims, len(pool_dims)


def knn_predict_proba(
    item_embs: np.ndarray,
    pool_embs: np.ndarray,
    pool_dims: list,
    dimensions: list,
    k: int = 5,
) -> np.ndarray:
    """
    Compute kNN dimension probability distribution for each item.

    Uses cosine similarity (item_embs already L2-normalised) against the pool,
    takes the top-k neighbours, and weights their dimension votes by softmax of
    their similarities.

    Returns
    -------
    proba : np.ndarray  shape (n_items, n_dims), rows sum to ~1
    """
    dim_to_idx = {d: i for i, d in enumerate(dimensions)}
    k = min(k, len(pool_dims))

    sims  = item_embs @ pool_embs.T   # (n_items, n_pool)
    proba = np.zeros((len(item_embs), len(dimensions)))

    for i in range(len(item_embs)):
        top_k_idx  = np.argsort(sims[i])[-k:][::-1]
        top_sims   = sims[i][top_k_idx]
        exp_s      = np.exp(top_sims - top_sims.max())
        weights    = exp_s / exp_s.sum()

        for w, j in zip(weights, top_k_idx):
            d = pool_dims[j]
            if d in dim_to_idx:
                proba[i, dim_to_idx[d]] += w

    return proba


def predict_knn_blended(
    item_texts: list,
    ft_probs: np.ndarray,
    ft_dims: list,
    model,
    construct: str,
    knn_alpha: float = KNN_ALPHA,
    knn_k: int = KNN_K,
    inventory_path: Path = INVENTORY_PATH,
) -> tuple:
    """
    Blend fine-tuned prototype probabilities with kNN probabilities.

    Parameters
    ----------
    item_texts  : validation item texts
    ft_probs    : prototype model probabilities  (n_items, n_dims)
    ft_dims     : ordered list of positive dimension IDs
    model       : loaded ContrastiveModel / PrototypeModel
    knn_alpha   : 1.0 → pure prototype, 0.0 → pure kNN, 0.5 → equal blend
    knn_k       : number of nearest neighbours

    Returns
    -------
    knn_preds : np.ndarray  predicted dimension IDs (int)
    knn_probs : np.ndarray  blended probabilities (n_items, n_dims)
    pool_size : int
    """
    # Encode validation items using the fine-tuned encoder (same normalisation
    # as ContrastiveModel.predict_proba uses internally)
    item_embs = model.encoder.encode(item_texts, convert_to_numpy=True, show_progress_bar=False)
    item_embs = item_embs / (np.linalg.norm(item_embs, axis=1, keepdims=True) + 1e-10)

    pool_embs, pool_dims, pool_size = build_knn_pool(
        construct, model.encoder, ft_dims, inventory_path
    )

    if pool_embs is None or pool_size == 0:
        print(f"    WARNING: kNN pool is empty for {construct}. Falling back to pure prototype.")
        return (
            np.array([ft_dims[int(np.argmax(ft_probs[i]))] for i in range(len(item_texts))]),
            ft_probs,
            0,
        )

    print(f"    kNN pool: {pool_size} inventory items for {construct}")

    knn_proba = knn_predict_proba(item_embs, pool_embs, pool_dims, ft_dims, k=knn_k)

    blended   = knn_alpha * ft_probs + (1.0 - knn_alpha) * knn_proba
    knn_preds = np.array([ft_dims[int(np.argmax(blended[i]))] for i in range(len(item_texts))])

    return knn_preds, blended, pool_size


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    exclude_minus1: bool = False,
) -> dict:
    """
    Compute Cohen's kappa and accuracy.

    Parameters
    ----------
    y_true         : expert dimension assignments (may include -1)
    y_pred         : model dimension assignments
    exclude_minus1 : if True, items where expert assigned -1 are excluded
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    if exclude_minus1:
        mask   = y_true != -1
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    n = len(y_true)

    if n < 2 or len(np.unique(y_true)) < 2:
        return {'kappa': np.nan, 'accuracy': np.nan, 'n_items': n}

    # Guard: kappa requires the union of labels to have ≥2 classes
    try:
        kappa = cohen_kappa_score(y_true, y_pred)
    except ValueError:
        kappa = np.nan

    acc = accuracy_score(y_true, y_pred)
    n_correct = int((y_true == y_pred).sum())

    return {
        'kappa':     kappa,
        'accuracy':  acc,
        'n_items':   n,
        'n_correct': n_correct,
    }


def per_dimension_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dim_ids: list,
) -> dict:
    """Return {dim_id: accuracy} for each positive dimension (excluding -1)."""
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    accs   = {}
    for d in dim_ids:
        if d == -1:
            continue
        mask = y_true == d
        if mask.sum() == 0:
            accs[d] = np.nan
        else:
            accs[d] = float((y_pred[mask] == d).mean())
    return accs


# ===========================================================================
# Figures
# ===========================================================================

def _kappa_interpretation(k):
    if np.isnan(k):
        return 'n/a'
    if k < 0:
        return 'poor'
    if k < 0.20:
        return 'slight'
    if k < 0.40:
        return 'fair'
    if k < 0.60:
        return 'moderate'
    if k < 0.80:
        return 'substantial'
    return 'almost perfect'


def make_comparison_figures(summary_df: pd.DataFrame, perdim_df: pd.DataFrame, output_dir: Path):
    """Generate all comparison figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    constructs = summary_df['construct'].tolist()
    n = len(constructs)
    x = np.arange(n)
    width = 0.35

    BASE_COLOR     = '#5B9BD5'
    BASE_KNN_COLOR = '#9B59B6'
    FT_COLOR       = '#ED7D31'
    DELTA_POS      = '#70AD47'
    DELTA_NEG      = '#FF4444'

    def _bar_labels(ax, bars):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7)

    # -----------------------------------------------------------------------
    # Figure 1: Kappa + Accuracy — 3 models (base, base+kNN, fine-tuned)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Base vs Base+kNN vs Fine-Tuned Model Performance\n(incl. 'does not fit')",
                 fontsize=15, fontweight='bold')

    panel_configs = [
        ('kappa',    "Cohen's κ", axes[0]),
        ('accuracy', 'Accuracy',  axes[1]),
    ]

    w3 = 0.25
    for metric_key, title, ax in panel_configs:
        base_vals     = summary_df[f'base_{metric_key}'].values
        base_knn_vals = summary_df[f'base_knn_{metric_key}'].values
        ft_vals       = summary_df[f'ft_{metric_key}'].values

        bars_base     = ax.bar(x - w3, base_vals,     w3, label='Base (descriptions)', color=BASE_COLOR,     alpha=0.85, edgecolor='white')
        bars_base_knn = ax.bar(x,      base_knn_vals, w3, label='Base + kNN pool',     color=BASE_KNN_COLOR, alpha=0.85, edgecolor='white')
        bars_ft       = ax.bar(x + w3, ft_vals,       w3, label='Fine-tuned',          color=FT_COLOR,       alpha=0.85, edgecolor='white')

        _bar_labels(ax, bars_base)
        _bar_labels(ax, bars_base_knn)
        _bar_labels(ax, bars_ft)

        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', '\n') for c in constructs], fontsize=9)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11, pad=8)
        ax.axhline(0.6, color='gray', linestyle='--', linewidth=0.8, alpha=0.5,
                   label='κ = 0.60 (substantial)')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig1_kappa_accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved fig1_kappa_accuracy_comparison.png")

    # -----------------------------------------------------------------------
    # Figure 2: Improvement (Δ) per construct — base+kNN and fine-tuned vs base
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Improvement over Base Model (description similarity)",
                 fontsize=14, fontweight='bold')

    for ax, (delta_col, ylabel, title) in zip(axes, [
        ('delta_kappa',    "Δ Cohen's κ", "Change in Cohen's κ"),
        ('delta_accuracy', "Δ Accuracy",  "Change in Accuracy"),
    ]):
        bknn_col  = 'delta_base_knn_' + delta_col.split('delta_')[1]
        bknn_vals = summary_df[bknn_col].values
        ft_vals   = summary_df[delta_col].values

        w2 = 0.3
        bknn_colors = [BASE_KNN_COLOR if d >= 0 else DELTA_NEG for d in bknn_vals]
        ft_colors   = [FT_COLOR       if d >= 0 else DELTA_NEG for d in ft_vals]

        bars_bknn = ax.bar(x - w2 / 2, bknn_vals, w2, color=bknn_colors, alpha=0.85, edgecolor='white', label='Base+kNN')
        bars_ft   = ax.bar(x + w2 / 2, ft_vals,   w2, color=ft_colors,   alpha=0.85, edgecolor='white', label='Fine-tuned')

        for bars in (bars_bknn, bars_ft):
            for bar, val in zip(bars, bars.datavalues if hasattr(bars, 'datavalues') else [b.get_height() for b in bars]):
                if not np.isnan(val):
                    va = 'bottom' if val >= 0 else 'top'
                    offset = 0.004 if val >= 0 else -0.004
                    ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                            f'{val:+.2f}', ha='center', va=va, fontsize=8, fontweight='bold')

        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', '\n') for c in constructs], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)

        all_vals = np.concatenate([bknn_vals[~np.isnan(bknn_vals)], ft_vals[~np.isnan(ft_vals)]])
        ymax = max(0.05, np.max(np.abs(all_vals)) * 1.35) if len(all_vals) else 0.1
        ax.set_ylim(-ymax, ymax)
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig2_improvement.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved fig2_improvement.png")

    # -----------------------------------------------------------------------
    # Figure 3: Per-dimension accuracy heatmap for each construct
    # -----------------------------------------------------------------------
    if perdim_df is not None and len(perdim_df) > 0:
        for construct in constructs:
            sub = perdim_df[perdim_df['construct'] == construct].copy()
            if len(sub) == 0:
                continue

            dims = sorted(sub['dimension'].unique())
            models_list = ['base', 'finetuned']
            heat_data = pd.DataFrame(index=models_list,
                                     columns=[f"Dim {d}" for d in dims],
                                     dtype=float)

            for d in dims:
                row = sub[sub['dimension'] == d].iloc[0]
                heat_data.loc['base',      f"Dim {d}"] = row['base_accuracy']
                heat_data.loc['finetuned', f"Dim {d}"] = row['ft_accuracy']

            # Add dimension labels as column annotations
            dim_labels_map = {d: DIMENSION_DESCRIPTIONS[construct][d]['label'] for d in dims
                              if d in DIMENSION_DESCRIPTIONS[construct]}

            col_labels = [f"Dim {d}\n{dim_labels_map.get(d, '')}" for d in dims]

            fig, ax = plt.subplots(figsize=(max(6, len(dims) * 2), 3))
            sns.heatmap(
                heat_data.astype(float), ax=ax,
                annot=True, fmt='.2f', cmap='YlGn',
                vmin=0, vmax=1, linewidths=0.5,
                cbar_kws={'label': 'Accuracy'},
                xticklabels=col_labels,
                yticklabels=['Base model', 'Fine-tuned'],
            )
            ax.set_title(f"{construct.replace('_', ' ').title()} — Per-Dimension Accuracy",
                         fontsize=12, pad=10)
            ax.tick_params(axis='x', labelsize=8)
            plt.tight_layout()
            fig.savefig(output_dir / f'fig3_{construct}_perdim_accuracy.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved fig3_{construct}_perdim_accuracy.png")

    # -----------------------------------------------------------------------
    # Figure 4: Summary scatter — base kappa vs fine-tuned kappa
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    base_k     = summary_df['base_kappa'].values
    base_knn_k = summary_df['base_knn_kappa'].values
    ft_k       = summary_df['ft_kappa'].values

    ax.scatter(base_k, ft_k,       s=100, zorder=5, color=FT_COLOR,       edgecolors='black', linewidth=0.7, label='Fine-tuned')
    ax.scatter(base_k, base_knn_k, s=100, zorder=5, color=BASE_KNN_COLOR, edgecolors='black', linewidth=0.7, label='Base+kNN', marker='D')

    for i, c in enumerate(constructs):
        ax.annotate(c.replace('_', ' '), (base_k[i], ft_k[i]),
                    textcoords='offset points', xytext=(6, 4), fontsize=8)
        ax.annotate(c.replace('_', ' '), (base_k[i], base_knn_k[i]),
                    textcoords='offset points', xytext=(6, -10), fontsize=8, color=BASE_KNN_COLOR)

    lim = [0, 1]
    ax.plot(lim, lim, 'k--', linewidth=1, alpha=0.4, label='No change')
    ax.set_xlabel("Base model (descriptions) — Cohen's κ", fontsize=11)
    ax.set_ylabel("Other model — Cohen's κ", fontsize=11)
    ax.set_title("Cohen's κ: Base vs Base+kNN vs Fine-Tuned", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / 'fig4_kappa_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved fig4_kappa_scatter.png")


def make_three_model_figures(summary_df: pd.DataFrame, perdim_df: pd.DataFrame, output_dir: Path):
    """
    Figures comparing all three models: base, fine-tuned, and kNN-blended.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    constructs = summary_df['construct'].tolist()
    n = len(constructs)
    x = np.arange(n)
    width = 0.25

    BASE_COLOR = '#5B9BD5'
    FT_COLOR   = '#ED7D31'
    KNN_COLOR  = '#70AD47'
    DELTA_POS  = '#70AD47'
    DELTA_NEG  = '#FF4444'

    # -----------------------------------------------------------------------
    # Figure 5: κ and accuracy for all 3 models (pos dims only)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Base vs Fine-Tuned vs kNN-Blended (k={KNN_K}, α per construct)\n"
                 "incl. 'does not fit'",
                 fontsize=13, fontweight='bold')

    for ax, (metric, ylabel) in zip(axes, [
        ('kappa',    "Cohen's κ"),
        ('accuracy', 'Accuracy'),
    ]):
        base_vals = summary_df[f'base_{metric}'].values
        ft_vals   = summary_df[f'ft_{metric}'].values
        knn_vals  = summary_df[f'knn_{metric}'].values

        bars_b = ax.bar(x - width, base_vals, width, label='Base',        color=BASE_COLOR, alpha=0.85, edgecolor='white')
        bars_f = ax.bar(x,         ft_vals,   width, label='Fine-tuned',   color=FT_COLOR,  alpha=0.85, edgecolor='white')
        bars_k = ax.bar(x + width, knn_vals,  width, label='kNN-blended',  color=KNN_COLOR, alpha=0.85, edgecolor='white')

        for bars in (bars_b, bars_f, bars_k):
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                            f'{h:.2f}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', '\n') for c in constructs], fontsize=9)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12)
        ax.axhline(0.6, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig5_three_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved fig5_three_model_comparison.png")

    # -----------------------------------------------------------------------
    # Figure 6: Δκ for ft and kNN vs base (side-by-side)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Improvement over Base Model (Δ Cohen's κ, positive dims only)",
                 fontsize=13, fontweight='bold')

    for ax, (model_tag, label, color) in zip(axes, [
        ('ft',  'Fine-tuned',  FT_COLOR),
        ('knn', 'kNN-blended', KNN_COLOR),
    ]):
        col = f'delta_{model_tag}_kappa' if model_tag == 'knn' else 'delta_kappa'
        deltas = summary_df[col].values
        bar_colors = [DELTA_POS if d >= 0 else DELTA_NEG for d in deltas]

        bars = ax.bar(x, deltas, color=bar_colors, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, deltas):
            if not np.isnan(val):
                va = 'bottom' if val >= 0 else 'top'
                offset = 0.005 if val >= 0 else -0.005
                ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                        f'{val:+.3f}', ha='center', va=va, fontsize=9, fontweight='bold')

        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('_', '\n') for c in constructs], fontsize=9)
        ax.set_ylabel("Δ Cohen's κ vs base", fontsize=10)
        ax.set_title(f"{label} vs Base", fontsize=12)

        ymax = max(0.05, np.nanmax(np.abs(deltas)) * 1.3)
        ax.set_ylim(-ymax, ymax)

        pos_patch = mpatches.Patch(color=DELTA_POS, label='Improvement')
        neg_patch = mpatches.Patch(color=DELTA_NEG, label='Degradation')
        ax.legend(handles=[pos_patch, neg_patch], fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'fig6_knn_improvement.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved fig6_knn_improvement.png")

    # -----------------------------------------------------------------------
    # Figure 7: Per-dimension accuracy heatmap including kNN (3 rows)
    # -----------------------------------------------------------------------
    if perdim_df is not None and len(perdim_df) > 0:
        for construct in constructs:
            sub = perdim_df[perdim_df['construct'] == construct].copy()
            if len(sub) == 0:
                continue

            dims = sorted(sub['dimension'].unique())
            models_list = ['base', 'base_knn', 'finetuned', 'knn_blended']
            heat_data = pd.DataFrame(index=models_list,
                                     columns=[f"Dim {d}" for d in dims],
                                     dtype=float)

            for d in dims:
                row = sub[sub['dimension'] == d].iloc[0]
                heat_data.loc['base',        f"Dim {d}"] = row['base_accuracy']
                heat_data.loc['base_knn',    f"Dim {d}"] = row.get('base_knn_accuracy', np.nan)
                heat_data.loc['finetuned',   f"Dim {d}"] = row['ft_accuracy']
                heat_data.loc['knn_blended', f"Dim {d}"] = row.get('knn_accuracy', np.nan)

            dim_labels_map = {d: DIMENSION_DESCRIPTIONS[construct][d]['label'] for d in dims
                              if d in DIMENSION_DESCRIPTIONS[construct]}
            col_labels = [f"Dim {d}\n{dim_labels_map.get(d, '')}" for d in dims]

            fig, ax = plt.subplots(figsize=(max(6, len(dims) * 2.2), 4.0))
            sns.heatmap(
                heat_data.astype(float), ax=ax,
                annot=True, fmt='.2f', cmap='YlGn',
                vmin=0, vmax=1, linewidths=0.5,
                cbar_kws={'label': 'Accuracy'},
                xticklabels=col_labels,
                yticklabels=['Base', 'Base+kNN', 'Fine-tuned', 'kNN-blended'],
            )
            ax.set_title(f"{construct.replace('_', ' ').title()} — Per-Dimension Accuracy (4 models)",
                         fontsize=12, pad=10)
            ax.tick_params(axis='x', labelsize=8)
            plt.tight_layout()
            fig.savefig(output_dir / f'fig7_{construct}_perdim_3models.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved fig7_{construct}_perdim_3models.png")


# ===========================================================================
# Main pipeline
# ===========================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Model Comparison Validation Analysis")
    print("Base Models vs Fine-Tuned Models on Round-2 Expert Survey Data")
    print("=" * 70)

    # Load data
    val_df    = load_validation_items(INPUT_ITEMS)
    expert_df = load_expert_assignments(EXPERT_ROUND2)

    print(f"\nValidation items loaded: {len(val_df)}")
    print(f"Expert assignments (round-2) loaded: {len(expert_df)}")

    constructs_in_val = set(val_df['construct'].unique())
    constructs_to_run = [c for c in BASE_MODELS if c in constructs_in_val]
    print(f"Constructs to analyse: {constructs_to_run}")

    all_item_results = []
    perdim_rows      = []
    summary_rows     = []

    for construct in constructs_to_run:
        print(f"\n{'='*60}")
        print(f"CONSTRUCT: {construct.upper()}")
        print('='*60)

        # Validation items for this construct
        items = val_df[val_df['construct'] == construct].copy()

        # Expert ground truth
        gt = get_expert_ground_truth(expert_df, construct)

        # Merge on item_id
        merged = items.merge(gt, on='item_id', how='inner')

        if len(merged) == 0:
            print(f"  WARNING: No matching items between validation set and expert assignments. Skipping.")
            continue

        n_minus1 = int((merged['expert_dimension'] == -1).sum())
        print(f"  Items matched: {len(merged)} (of which {n_minus1} are 'does not fit')")

        item_texts    = merged['item_text'].tolist()
        expert_dims   = merged['expert_dimension'].astype(int).tolist()
        questionnaires = merged['questionnaire_x'].tolist() if 'questionnaire_x' in merged.columns \
                         else merged['questionnaire'].tolist()

        # ------------------------------------------------------------------
        # Fine-tuned model
        # ------------------------------------------------------------------
        print(f"\n  [Fine-tuned model]")
        try:
            ft_preds, ft_probs, ft_dims, ft_dim_labels, ft_model = predict_finetuned_model(
                item_texts, construct, MODELS_DIR
            )
        except Exception as e:
            print(f"  ERROR running fine-tuned model: {e}")
            continue

        # Map ft_dims to labels dict for easy lookup
        ft_dim_label_map = dict(zip(ft_dims, ft_dim_labels))
        ft_pred_labels = [ft_dim_label_map.get(int(d), f"Dim {d}") for d in ft_preds]

        # ------------------------------------------------------------------
        # kNN-blended model
        # ------------------------------------------------------------------
        construct_alpha = KNN_ALPHA.get(construct, 0.75)
        print(f"\n  [kNN-blended model (α={construct_alpha}, k={KNN_K})]")
        try:
            knn_preds, knn_probs, knn_pool_size = predict_knn_blended(
                item_texts, ft_probs, ft_dims, ft_model,
                construct, construct_alpha, KNN_K
            )
        except Exception as e:
            print(f"  WARNING: kNN blending failed ({e}). Falling back to pure prototype.")
            knn_preds = ft_preds.copy()
            knn_probs = ft_probs.copy()
            knn_pool_size = 0

        knn_pred_labels = [ft_dim_label_map.get(int(d), f"Dim {d}") for d in knn_preds]

        # ------------------------------------------------------------------
        # Base model
        # ------------------------------------------------------------------
        print(f"\n  [Base model: {BASE_MODELS[construct]}]")
        try:
            base_assignments, base_sims, base_dim_ids, base_dim_labels = predict_base_model(
                item_texts, construct, BASE_MODELS[construct]
            )
        except Exception as e:
            print(f"  ERROR running base model: {e}")
            continue

        base_dim_label_map = dict(zip(base_dim_ids, base_dim_labels))

        # ------------------------------------------------------------------
        # Base model + kNN
        # ------------------------------------------------------------------
        print(f"\n  [Base model + kNN (k={KNN_K}): {BASE_MODELS[construct]}]")
        try:
            bknn_preds, bknn_probs, bknn_pool_size = predict_base_knn(
                item_texts, construct, BASE_MODELS[construct],
                base_dim_ids, KNN_K
            )
        except Exception as e:
            print(f"  WARNING: Base kNN failed ({e}). Skipping.")
            bknn_preds    = base_assignments.copy()
            bknn_probs    = np.zeros((len(item_texts), len(base_dim_ids)))
            bknn_pool_size = 0

        bknn_pred_labels = [base_dim_label_map.get(int(d), f"Dim {d}") for d in bknn_preds]

        # ------------------------------------------------------------------
        # Base kNN-blended: softmax(base_sims) blended with base kNN probs
        # Same alpha as fine-tuned kNN-blended; separates embedding effect
        # from prototype-vs-kNN effect.
        # ------------------------------------------------------------------
        base_sims_shifted = base_sims - base_sims.max(axis=1, keepdims=True)
        base_probs = np.exp(base_sims_shifted)
        base_probs = base_probs / base_probs.sum(axis=1, keepdims=True)
        bknn_blended = construct_alpha * base_probs + (1.0 - construct_alpha) * bknn_probs
        bknn_bl_preds = np.array([base_dim_ids[int(np.argmax(bknn_blended[i]))]
                                   for i in range(len(item_texts))])
        bknn_bl_pred_labels = [base_dim_label_map.get(int(d), f"Dim {d}") for d in bknn_bl_preds]

        # ------------------------------------------------------------------
        # Metrics (all items, incl. 'does not fit')
        # ------------------------------------------------------------------
        ft_all      = compute_metrics(expert_dims, ft_preds,         exclude_minus1=False)
        knn_all     = compute_metrics(expert_dims, knn_preds,        exclude_minus1=False)
        base_all    = compute_metrics(expert_dims, base_assignments,  exclude_minus1=False)
        bknn_all    = compute_metrics(expert_dims, bknn_preds,        exclude_minus1=False)
        bknn_bl_all = compute_metrics(expert_dims, bknn_bl_preds,     exclude_minus1=False)

        print(f"\n  --- Metrics (all items, incl. 'does not fit') ---")
        print(f"  Fine-tuned:           κ={ft_all['kappa']:.3f}  ({_kappa_interpretation(ft_all['kappa'])}),  "
              f"acc={ft_all['accuracy']:.3f}  ({ft_all['n_correct']}/{ft_all['n_items']})")
        print(f"  kNN-blended (FT):     κ={knn_all['kappa']:.3f}  ({_kappa_interpretation(knn_all['kappa'])}),  "
              f"acc={knn_all['accuracy']:.3f}  ({knn_all['n_correct']}/{knn_all['n_items']})")
        print(f"  Base model:           κ={base_all['kappa']:.3f}  ({_kappa_interpretation(base_all['kappa'])}),  "
              f"acc={base_all['accuracy']:.3f}  ({base_all['n_correct']}/{base_all['n_items']})")
        print(f"  Base+kNN (pure kNN):  κ={bknn_all['kappa']:.3f}  ({_kappa_interpretation(bknn_all['kappa'])}),  "
              f"acc={bknn_all['accuracy']:.3f}  ({bknn_all['n_correct']}/{bknn_all['n_items']})")
        print(f"  Base kNN-blended:     κ={bknn_bl_all['kappa']:.3f}  ({_kappa_interpretation(bknn_bl_all['kappa'])}),  "
              f"acc={bknn_bl_all['accuracy']:.3f}  ({bknn_bl_all['n_correct']}/{bknn_bl_all['n_items']})")

        # ------------------------------------------------------------------
        # Per-dimension accuracy
        # ------------------------------------------------------------------
        all_dim_ids_for_perdim = [d for d in base_dim_ids]
        ft_perdim      = per_dimension_accuracy(expert_dims, ft_preds,         all_dim_ids_for_perdim)
        knn_perdim     = per_dimension_accuracy(expert_dims, knn_preds,        all_dim_ids_for_perdim)
        base_perdim    = per_dimension_accuracy(expert_dims, base_assignments,  all_dim_ids_for_perdim)
        bknn_perdim    = per_dimension_accuracy(expert_dims, bknn_preds,        all_dim_ids_for_perdim)
        bknn_bl_perdim = per_dimension_accuracy(expert_dims, bknn_bl_preds,     all_dim_ids_for_perdim)

        for d in sorted(all_dim_ids_for_perdim):
            if d == -1:
                continue
            perdim_rows.append({
                'construct':          construct,
                'dimension':          d,
                'dim_label':          DIMENSION_DESCRIPTIONS[construct].get(d, {}).get('label', f'Dim {d}'),
                'n_expert':           int((np.array(expert_dims) == d).sum()),
                'base_accuracy':      round(base_perdim.get(d, np.nan),  4),
                'ft_accuracy':        round(ft_perdim.get(d, np.nan),    4),
                'knn_accuracy':       round(knn_perdim.get(d, np.nan),   4),
                'base_knn_accuracy':     round(bknn_perdim.get(d, np.nan),    4),
                'base_knn_bl_accuracy':  round(bknn_bl_perdim.get(d, np.nan), 4),
                'delta_ft_accuracy':  round(
                    ft_perdim.get(d, np.nan) - base_perdim.get(d, np.nan)
                    if not (np.isnan(ft_perdim.get(d, np.nan)) or np.isnan(base_perdim.get(d, np.nan)))
                    else np.nan, 4),
                'delta_knn_accuracy': round(
                    knn_perdim.get(d, np.nan) - base_perdim.get(d, np.nan)
                    if not (np.isnan(knn_perdim.get(d, np.nan)) or np.isnan(base_perdim.get(d, np.nan)))
                    else np.nan, 4),
                'delta_base_knn_accuracy': round(
                    bknn_perdim.get(d, np.nan) - base_perdim.get(d, np.nan)
                    if not (np.isnan(bknn_perdim.get(d, np.nan)) or np.isnan(base_perdim.get(d, np.nan)))
                    else np.nan, 4),
            })

        # ------------------------------------------------------------------
        # Build per-item result rows
        # ------------------------------------------------------------------
        for i, row in enumerate(merged.itertuples(index=False)):
            item_id = row.item_id
            exp_dim = int(row.expert_dimension)

            ft_pred    = int(ft_preds[i])
            knn_pred   = int(knn_preds[i])
            base_pred  = int(base_assignments[i])
            bknn_pred  = int(bknn_preds[i])

            # Fine-tuned confidence
            ft_pred_idx = ft_dims.index(ft_pred) if ft_pred in ft_dims else 0
            ft_conf = float(ft_probs[i, ft_pred_idx])

            # kNN-blended confidence
            knn_pred_idx = ft_dims.index(knn_pred) if knn_pred in ft_dims else 0
            knn_conf = float(knn_probs[i, knn_pred_idx])

            # Base+kNN confidence
            bknn_pred_idx = base_dim_ids.index(bknn_pred) if bknn_pred in base_dim_ids else 0
            bknn_conf = float(bknn_probs[i, bknn_pred_idx])

            # Base kNN-blended confidence
            bknn_bl_pred = int(bknn_bl_preds[i])
            bknn_bl_pred_idx = base_dim_ids.index(bknn_bl_pred) if bknn_bl_pred in base_dim_ids else 0
            bknn_bl_conf = float(bknn_blended[i, bknn_bl_pred_idx])

            ft_prob_dist  = {int(ft_dims[j]): round(float(ft_probs[i, j]), 4)
                             for j in range(len(ft_dims))}
            base_sim_dict = {base_dim_ids[j]: round(float(base_sims[i, j]), 4)
                             for j in range(len(base_dim_ids))}

            item_row = {
                'construct':             construct,
                'questionnaire':         questionnaires[i],
                'item_id':               item_id,
                'item_text':             row.item_text,
                'expert_dimension':      exp_dim,
                'expert_label':          row.expert_label,
                'expert_p_i':            row.expert_p_i,
                # Fine-tuned
                'ft_dimension':          ft_pred,
                'ft_dimension_label':    ft_pred_labels[i],
                'ft_confidence':         round(ft_conf, 4),
                'ft_prob_distribution':  str(ft_prob_dist),
                'ft_correct':            ft_pred == exp_dim,
                # kNN-blended (fine-tuned encoder)
                'knn_dimension':         knn_pred,
                'knn_dimension_label':   knn_pred_labels[i],
                'knn_confidence':        round(knn_conf, 4),
                'knn_correct':           knn_pred == exp_dim,
                # Base model (cosine to descriptions)
                'base_dimension':        base_pred,
                'base_dimension_label':  base_dim_label_map.get(base_pred, f'Dim {base_pred}'),
                'base_max_similarity':   round(float(np.max(base_sims[i])), 4),
                'base_correct':          base_pred == exp_dim,
                # Base model + kNN pool (pure kNN)
                'base_knn_dimension':       bknn_pred,
                'base_knn_dimension_label': bknn_pred_labels[i],
                'base_knn_confidence':      round(bknn_conf, 4),
                'base_knn_correct':         bknn_pred == exp_dim,
                # Base kNN-blended (base descriptions + base kNN)
                'base_knn_bl_dimension':       bknn_bl_pred,
                'base_knn_bl_dimension_label': bknn_bl_pred_labels[i],
                'base_knn_bl_confidence':      round(bknn_bl_conf, 4),
                'base_knn_bl_correct':         bknn_bl_pred == exp_dim,
                # Agreement
                'models_agree':          ft_pred == base_pred,
            }

            for d in base_dim_ids:
                item_row[f'base_sim_dim{d}'] = base_sim_dict[d]

            all_item_results.append(item_row)

        # ------------------------------------------------------------------
        # Summary row
        # ------------------------------------------------------------------
        def _r(v):
            return round(v, 4) if not np.isnan(v) else np.nan
        def _delta(a, b):
            return round(a - b, 4) if not (np.isnan(a) or np.isnan(b)) else np.nan

        summary_rows.append({
            'construct':              construct,
            'n_items_total':          len(merged),
            'n_items_expert_minus1':  n_minus1,
            # Fine-tuned
            'ft_kappa':               _r(ft_all['kappa']),
            'ft_accuracy':            _r(ft_all['accuracy']),
            'ft_n_correct':           ft_all['n_correct'],
            # kNN-blended (fine-tuned encoder)
            'knn_kappa':              _r(knn_all['kappa']),
            'knn_accuracy':           _r(knn_all['accuracy']),
            'knn_n_correct':          knn_all['n_correct'],
            'knn_alpha':              construct_alpha,
            'knn_pool_size':          knn_pool_size,
            # Base model (cosine to descriptions)
            'base_kappa':             _r(base_all['kappa']),
            'base_accuracy':          _r(base_all['accuracy']),
            'base_n_correct':         base_all['n_correct'],
            # Base model + kNN pool (pure kNN)
            'base_knn_kappa':         _r(bknn_all['kappa']),
            'base_knn_accuracy':      _r(bknn_all['accuracy']),
            'base_knn_n_correct':     bknn_all['n_correct'],
            'base_knn_pool_size':     bknn_pool_size,
            # Base kNN-blended (base descriptions blended with base kNN)
            'base_knn_bl_kappa':      _r(bknn_bl_all['kappa']),
            'base_knn_bl_accuracy':   _r(bknn_bl_all['accuracy']),
            'base_knn_bl_n_correct':  bknn_bl_all['n_correct'],
            # Deltas vs base (description similarity)
            'delta_kappa':            _delta(ft_all['kappa'],        base_all['kappa']),
            'delta_accuracy':         _delta(ft_all['accuracy'],     base_all['accuracy']),
            'delta_knn_kappa':        _delta(knn_all['kappa'],       base_all['kappa']),
            'delta_knn_accuracy':     _delta(knn_all['accuracy'],    base_all['accuracy']),
            'delta_base_knn_kappa':   _delta(bknn_all['kappa'],      base_all['kappa']),
            'delta_base_knn_accuracy':_delta(bknn_all['accuracy'],   base_all['accuracy']),
            'delta_base_knn_bl_kappa':    _delta(bknn_bl_all['kappa'],    base_all['kappa']),
            'delta_base_knn_bl_accuracy': _delta(bknn_bl_all['accuracy'], base_all['accuracy']),
        })

    # ==========================================================================
    # Save outputs
    # ==========================================================================

    results_df = pd.DataFrame(all_item_results)
    results_df.to_csv(OUTPUT_DIR / 'validation_predictions_all.csv', index=False)
    print(f"\nSaved: validation_predictions_all.csv  ({len(results_df)} rows)")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / 'validation_kappa_comparison.csv', index=False)
    print(f"Saved: validation_kappa_comparison.csv")

    perdim_df = pd.DataFrame(perdim_rows)
    perdim_df.to_csv(OUTPUT_DIR / 'validation_perdim_accuracy.csv', index=False)
    print(f"Saved: validation_perdim_accuracy.csv")

    # ==========================================================================
    # Print final summary table
    # ==========================================================================

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    def _fmt(v):
        return f"{v:.3f}" if not pd.isna(v) else "  n/a"

    print(f"\nCohen's Kappa (all items, incl. 'does not fit'):")
    print(f"{'Construct':<18} {'Base κ':>8} {'B+kNN κ':>9} {'B+kNN-bl κ':>12} {'FT κ':>8} {'FT-bl κ':>9}")
    print("-" * 70)
    for _, row in summary_df.iterrows():
        print(
            f"{row['construct']:<18} "
            f"{_fmt(row['base_kappa']):>8} "
            f"{_fmt(row['base_knn_kappa']):>9} "
            f"{_fmt(row['base_knn_bl_kappa']):>12} "
            f"{_fmt(row['ft_kappa']):>8} "
            f"{_fmt(row['knn_kappa']):>9}"
        )

    print(f"\nAccuracy (all items, incl. 'does not fit'):")
    print(f"{'Construct':<18} {'Base':>8} {'B+kNN':>9} {'B+kNN-bl':>12} {'FT':>8} {'FT-bl':>9}")
    print("-" * 70)
    for _, row in summary_df.iterrows():
        print(
            f"{row['construct']:<18} "
            f"{_fmt(row['base_accuracy']):>8} "
            f"{_fmt(row['base_knn_accuracy']):>9} "
            f"{_fmt(row['base_knn_bl_accuracy']):>12} "
            f"{_fmt(row['ft_accuracy']):>8} "
            f"{_fmt(row['knn_accuracy']):>9}"
        )

    # ==========================================================================
    # Generate figures
    # ==========================================================================
    print(f"\nGenerating figures...")
    make_comparison_figures(summary_df, perdim_df, FIGURES_DIR)
    make_three_model_figures(summary_df, perdim_df, FIGURES_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    return results_df, summary_df, perdim_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Override MODELS_DIR path')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override OUTPUT_DIR path')
    parser.add_argument('--dimension-config', type=str, default=None,
                        help='Path to a dimension_descriptions.yaml defining the '
                             f'constructs/dimensions to analyze (default: {DEFAULT_DIMENSION_CONFIG}). '
                             'Point this at your own copy to run on different constructs '
                             'without editing this script.')
    args = parser.parse_args()

    if args.model_dir:
        MODELS_DIR = Path(args.model_dir)
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        FIGURES_DIR = OUTPUT_DIR / 'figures'
    if args.dimension_config:
        DIMENSION_DESCRIPTIONS = load_dimension_descriptions(args.dimension_config)

    results_df, summary_df, perdim_df = main()
