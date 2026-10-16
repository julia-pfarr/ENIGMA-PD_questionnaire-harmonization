"""
Semantic Learning Script for ENIGMA-PD Harmonizer
=================================================

This script trains models that learn CLINICAL SEMANTICS from expert assignments.

Input: finetuning_dataset.csv (contains item texts, E_ik distributions, questionnaire info)

Models:
1. PROTOTYPE MODEL:
   - Learns dimension prototypes from expert-assigned items
   - Prototype = weighted average of embeddings (weighted by E_ik)
   - Fast, no GPU required

2. CONTRASTIVE MODEL:  
   - Fine-tunes embedding model with contrastive learning
   - Same dimension items → pull together
   - Different dimension items → push apart
   - Supports hard negative mining for boundary learning
   - Deeper semantic learning, GPU recommended

Both models use hybrid prototype + k-NN inference:
- Prototype similarity gives global structure (which dimension cluster?)
- k-NN similarity gives local structure (close to which specific items?)
- Combined score: α × prototype + (1-α) × k-NN
- The k-NN pool grows as new validated items are added to the inventory

Training signal: expert_proportions_e_ik (E_ik) - the soft label distribution from experts.

Usage:
------
# Prototype model (fast, CPU)
python semantic_learning.py \\
    --data finetuning_dataset.csv \\
    --construct depression \\
    --approach prototype \\
    --output ./models

# Contrastive with hard negative mining and k-NN
python semantic_learning.py \\
    --data finetuning_dataset.csv \\
    --construct depression \\
    --approach contrastive \\
    --hard-negative-boost 3.0 \\
    --knn-k 5 --knn-alpha 0.7 \\
    --device cuda --cv \\
    --output ./models
"""

import argparse
import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pickle
import warnings

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from scipy.optimize import minimize

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Contrastive model will not work.")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    warnings.warn("sentence-transformers not available.")


# =============================================================================
# Data Loading
# =============================================================================

def parse_e_ik(e_ik_str: str) -> Dict[int, float]:
    """Parse expert_proportions_e_ik column to dictionary."""
    if pd.isna(e_ik_str) or not e_ik_str:
        return {}
    try:
        return {int(k): float(v) for k, v in ast.literal_eval(e_ik_str).items()}
    except:
        return {}


def load_construct_data(data_path: str, construct: str) -> pd.DataFrame:
    """Load data for a specific construct from finetuning_dataset.csv."""
    df = pd.read_csv(data_path)
    df = df[df['construct'] == construct].copy()
    df['e_ik_dict'] = df['expert_proportions_e_ik'].apply(parse_e_ik)
    return df


def prepare_training_data(
    df: pd.DataFrame
) -> Tuple[List[str], np.ndarray, np.ndarray, List[int], List[str], List[str]]:
    """
    Prepare training data from finetuning_dataset.csv.
    
    Returns:
        item_texts, y_hard, y_soft, dimensions, dimension_labels, questionnaires
    """
    all_dims = set()
    for e_ik in df['e_ik_dict']:
        all_dims.update(e_ik.keys())
    
    dimensions = sorted([d for d in all_dims if d == -1]) + sorted([d for d in all_dims if d != -1])
    if -1 not in dimensions:
        dimensions = [-1] + dimensions
    
    n_items = len(df)
    n_dims = len(dimensions)
    
    item_texts = df['question_text'].tolist()
    questionnaires = df['questionnaire'].tolist()
    y_hard = df['final_dimension'].values.astype(int)
    
    y_soft = np.zeros((n_items, n_dims))
    for i, e_ik in enumerate(df['e_ik_dict']):
        for dim, prob in e_ik.items():
            dim_idx = dimensions.index(dim)
            y_soft[i, dim_idx] = prob
        if y_soft[i].sum() > 0:
            y_soft[i] /= y_soft[i].sum()
    
    dim_to_label = {}
    for _, row in df.iterrows():
        dim = row['final_dimension']
        label = row['final_dimension_label']
        if dim not in dim_to_label:
            dim_to_label[dim] = label
    if -1 not in dim_to_label:
        dim_to_label[-1] = "Does not fit"
    
    dimension_labels = [dim_to_label.get(d, f"Dimension {d}") for d in dimensions]
    return item_texts, y_hard, y_soft, dimensions, dimension_labels, questionnaires


# =============================================================================
# k-NN Scoring (shared by both models)
# =============================================================================

def compute_knn_scores(
    query_embeddings: np.ndarray,
    inventory_embeddings: np.ndarray,
    inventory_e_ik: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """
    Compute k-NN dimension scores for query items against an inventory.

    For each query item, finds the k nearest inventory items (by cosine
    similarity) and aggregates their E_ik distributions weighted by
    similarity.

    Parameters
    ----------
    query_embeddings : np.ndarray, shape (n_query, embedding_dim)
        Normalized embeddings for new items
    inventory_embeddings : np.ndarray, shape (n_inventory, embedding_dim)
        Normalized embeddings for inventory items
    inventory_e_ik : np.ndarray, shape (n_inventory, n_real_dims)
        E_ik distributions for inventory items (real dimensions only)
    k : int
        Number of neighbors

    Returns
    -------
    knn_scores : np.ndarray, shape (n_query, n_real_dims)
        Aggregated dimension scores from k nearest neighbors
    """
    n_query = len(query_embeddings)
    n_real_dims = inventory_e_ik.shape[1]

    # Cosine similarity between queries and inventory
    sim_matrix = query_embeddings @ inventory_embeddings.T  # (n_query, n_inventory)

    # Clamp k to inventory size
    actual_k = min(k, len(inventory_embeddings))

    knn_scores = np.zeros((n_query, n_real_dims))

    for i in range(n_query):
        # Find k nearest neighbors
        top_k_idx = np.argpartition(sim_matrix[i], -actual_k)[-actual_k:]
        top_k_sims = sim_matrix[i, top_k_idx]

        # Weight neighbors by similarity (softmax to normalize)
        # Use temperature=0.1 to sharpen the weighting toward closer neighbors
        exp_sims = np.exp((top_k_sims - top_k_sims.max()) / 0.1)
        neighbor_weights = exp_sims / (exp_sims.sum() + 1e-10)

        # Aggregate E_ik weighted by similarity
        for j, idx in enumerate(top_k_idx):
            knn_scores[i] += neighbor_weights[j] * inventory_e_ik[idx]

        # Normalize to sum to 1
        total = knn_scores[i].sum()
        if total > 0:
            knn_scores[i] /= total

    return knn_scores


# =============================================================================
# Prototype Model
# =============================================================================

class PrototypeModel:
    """
    Prototype-based model with hybrid prototype + k-NN inference.

    Training:
    1. Prototype embeddings for each dimension (weighted by E_ik)
    2. Calibration weights/biases
    3. Thresholds for "does not fit" (-1) detection

    Inference:
    - Prototype score: cosine similarity to dimension centroids
    - k-NN score: similarity-weighted vote from k nearest inventory items
    - Combined: α × prototype + (1-α) × k-NN
    - The inventory grows as new validated items are added
    """

    def __init__(
        self,
        model_name: str = "hkunlp/instructor-large",
        device: str = "cpu",
        knn_k: int = 5,
        knn_alpha: float = 0.7,
        class_balance: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.knn_k = knn_k
        self.knn_alpha = knn_alpha
        self.class_balance = class_balance

        self.encoder = None
        self.prototypes = None
        self.weights = None
        self.biases = None
        self.tau_max = 0.5
        self.tau_margin = 0.02
        self.p_minus1_base = 0.1
        self.temperature = 1.0
        self.dimensions = None
        self.dimension_labels = None
        self.n_real_dims = None
        self.real_dim_indices = None
        self.real_dim_labels = None
        self.minus1_idx = None
        self.fitted = False

        # Inventory for k-NN (grows over time)
        self.inventory_embeddings = None  # (n_items, embedding_dim)
        self.inventory_e_ik = None        # (n_items, n_real_dims)
        self.inventory_hard_labels = None  # (n_items,) dimension IDs
        self.inventory_item_ids = []       # item ID strings

    def fit(
        self,
        item_texts: List[str],
        y_soft: np.ndarray,
        dimensions: List[int],
        dimension_labels: List[str]
    ):
        """Learn prototypes, calibration, and store training inventory."""
        print(f"    Loading encoder: {self.model_name}")
        self.encoder = SentenceTransformer(self.model_name, device=self.device)

        self.dimensions = dimensions
        self.dimension_labels = dimension_labels
        self.minus1_idx = dimensions.index(-1) if -1 in dimensions else None
        self.real_dim_indices = [i for i, d in enumerate(dimensions) if d != -1]
        self.n_real_dims = len(self.real_dim_indices)
        self.real_dim_labels = [dimension_labels[i] for i in self.real_dim_indices]

        print(f"    Computing embeddings for {len(item_texts)} items...")
        embeddings = self.encoder.encode(item_texts, convert_to_numpy=True, show_progress_bar=True)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

        e_dims = y_soft[:, self.real_dim_indices]

        if self.minus1_idx is not None:
            e_minus1 = y_soft[:, self.minus1_idx]
        else:
            e_minus1 = np.zeros(len(item_texts))

        # Store training items as initial inventory
        print(f"    Storing {len(item_texts)} training items as k-NN inventory...")
        self.inventory_embeddings = embeddings.copy()
        self.inventory_e_ik = e_dims.copy()
        self.inventory_hard_labels = np.array([
            dimensions[np.argmax(y_soft[i])] for i in range(len(item_texts))
        ])
        self.inventory_item_ids = list(range(len(item_texts)))

        # Learn prototypes
        print(f"    Learning prototypes for {self.n_real_dims} dimensions...")
        embedding_dim = embeddings.shape[1]
        self.prototypes = np.zeros((self.n_real_dims, embedding_dim))

        for k in range(self.n_real_dims):
            weights_k = e_dims[:, k]
            if weights_k.sum() > 0:
                self.prototypes[k] = np.average(embeddings, axis=0, weights=weights_k + 1e-10)
                self.prototypes[k] /= (np.linalg.norm(self.prototypes[k]) + 1e-10)
                print(f"      {self.real_dim_labels[k][:40]}: {weights_k.sum():.1f} weighted items")

        similarities = embeddings @ self.prototypes.T
        max_sim = np.max(similarities, axis=1)
        sorted_sims = np.sort(similarities, axis=1)[:, ::-1]
        margin = sorted_sims[:, 0] - sorted_sims[:, 1] if self.n_real_dims > 1 else sorted_sims[:, 0]

        print(f"    Learning -1 thresholds from E_{{i,-1}}...")
        self._learn_thresholds(max_sim, margin, e_minus1)

        print(f"    Learning calibration weights from E_{{i,k}}...")
        self._learn_calibration(similarities, max_sim, margin, e_dims, e_minus1)

        self.fitted = True
        self._print_params()
        return self

    def _learn_thresholds(self, max_sim, margin, e_minus1):
        if e_minus1.sum() == 0:
            print(f"      No -1 votes in data, using defaults")
            return
        best_score = -np.inf
        for tau_max in np.arange(0.30, 0.75, 0.025):
            for tau_margin in np.arange(0.005, 0.12, 0.01):
                for p_base in [0.02, 0.05, 0.10, 0.15]:
                    p_minus1 = self._compute_p_minus1(max_sim, margin, tau_max, tau_margin, p_base)
                    if np.std(p_minus1) > 0 and np.std(e_minus1) > 0:
                        corr = np.corrcoef(p_minus1, e_minus1)[0, 1]
                        if not np.isnan(corr) and corr > best_score:
                            best_score = corr
                            self.tau_max = tau_max
                            self.tau_margin = tau_margin
                            self.p_minus1_base = p_base
        print(f"      Best correlation with E_{{i,-1}}: {best_score:.3f}")

    def _learn_calibration(self, similarities, max_sim, margin, e_dims, e_minus1):
        self.weights = np.ones(self.n_real_dims)
        self.biases = np.zeros(self.n_real_dims)

        def objective(params):
            w = params[:self.n_real_dims]
            b = params[self.n_real_dims:2*self.n_real_dims]
            temp = params[2*self.n_real_dims]
            calibrated = similarities * w + b
            scaled = calibrated / (temp + 1e-10)
            exp_scores = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            probs = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-10)
            if self.minus1_idx is not None:
                p_m1 = self._compute_p_minus1(max_sim, margin, self.tau_max, self.tau_margin, self.p_minus1_base)
                probs = probs * (1 - p_m1[:, np.newaxis])
            e_safe = e_dims + 1e-10
            e_safe = e_safe / e_safe.sum(axis=1, keepdims=True)
            p_safe = probs + 1e-10
            p_safe = p_safe / p_safe.sum(axis=1, keepdims=True)
            kl = np.sum(e_safe * (np.log(e_safe) - np.log(p_safe)), axis=1).mean()
            return kl + 0.01 * (np.sum((w - 1)**2) + np.sum(b**2))

        x0 = np.concatenate([np.ones(self.n_real_dims), np.zeros(self.n_real_dims), [1.0]])
        bounds = [(0.1, 5.0)] * self.n_real_dims + [(-1.0, 1.0)] * self.n_real_dims + [(0.1, 5.0)]
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500})
        self.weights = result.x[:self.n_real_dims]
        self.biases = result.x[self.n_real_dims:2*self.n_real_dims]
        self.temperature = result.x[2*self.n_real_dims]

    def _compute_p_minus1(self, max_sim, margin, tau_max, tau_margin, p_base):
        dist_max = (tau_max - max_sim) / (tau_max + 1e-10)
        dist_margin = (tau_margin - margin) / (tau_margin + 1e-10)
        dist = np.minimum(dist_max, dist_margin)
        return np.clip(p_base + (1 - p_base) / (1 + np.exp(-5 * dist)), 0.01, 0.99)

    def _print_params(self):
        print(f"\n    Learned parameters:")
        print(f"      Thresholds: τ_max={self.tau_max:.3f}, τ_margin={self.tau_margin:.3f}, p_base={self.p_minus1_base:.2f}")
        print(f"      Temperature: {self.temperature:.3f}")
        print(f"      k-NN: k={self.knn_k}, α={self.knn_alpha:.2f} (1.0=pure prototype, 0.0=pure k-NN)")
        print(f"      Inventory size: {len(self.inventory_embeddings)} items")
        for i, label in enumerate(self.real_dim_labels):
            print(f"        {label[:40]:<40}: w={self.weights[i]:.3f}, b={self.biases[i]:+.3f}")

    def predict(self, item_texts: List[str]) -> np.ndarray:
        proba = self.predict_proba(item_texts)
        pred_idx = np.argmax(proba, axis=1)
        return np.array([self.dimensions[i] for i in pred_idx])

    def predict_proba(self, item_texts: List[str]) -> np.ndarray:
        """
        Predict probability distribution using hybrid prototype + k-NN.

        If knn_alpha=1.0 or no inventory, falls back to pure prototype (same as v1).
        """
        embeddings = self.encoder.encode(item_texts, convert_to_numpy=True)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

        # --- Prototype scores ---
        similarities = embeddings @ self.prototypes.T
        max_sim = np.max(similarities, axis=1)
        sorted_sims = np.sort(similarities, axis=1)[:, ::-1]
        margin = sorted_sims[:, 0] - sorted_sims[:, 1] if self.n_real_dims > 1 else sorted_sims[:, 0]

        calibrated = similarities * self.weights + self.biases
        scaled = calibrated / (self.temperature + 1e-10)
        exp_scores = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        proto_probs = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-10)

        # --- k-NN scores ---
        use_knn = (
            self.knn_alpha < 1.0
            and self.inventory_embeddings is not None
            and len(self.inventory_embeddings) > 0
        )

        if use_knn:
            knn_probs = compute_knn_scores(
                embeddings, self.inventory_embeddings,
                self.inventory_e_ik, k=self.knn_k
            )
            # Blend: α × prototype + (1-α) × k-NN
            dim_probs = self.knn_alpha * proto_probs + (1 - self.knn_alpha) * knn_probs
        else:
            dim_probs = proto_probs

        # --- Class balance correction ---
        # Divide each dimension's score by its frequency in the current inventory,
        # then renormalize. Computed from inventory_e_ik so it updates automatically
        # as new items are added without retraining.
        if self.class_balance and self.inventory_e_ik is not None:
            freq = self.inventory_e_ik.sum(axis=0)          # (n_real_dims,)
            freq = freq / (freq.sum() + 1e-10)               # fraction per dimension
            dim_probs = dim_probs / (freq + 1e-10)
            dim_probs = dim_probs / (dim_probs.sum(axis=1, keepdims=True) + 1e-10)

        # --- P(-1) and final assembly ---
        p_minus1 = self._compute_p_minus1(max_sim, margin, self.tau_max, self.tau_margin, self.p_minus1_base)

        n_samples = len(item_texts)
        proba = np.zeros((n_samples, len(self.dimensions)))

        if self.minus1_idx is not None:
            proba[:, self.minus1_idx] = p_minus1
            for j, real_idx in enumerate(self.real_dim_indices):
                proba[:, real_idx] = dim_probs[:, j] * (1 - p_minus1)
        else:
            for j, real_idx in enumerate(self.real_dim_indices):
                proba[:, real_idx] = dim_probs[:, j]

        return proba

    def add_to_inventory(
        self,
        item_texts: List[str],
        hard_labels: List[int],
        e_ik: Optional[np.ndarray] = None,
        item_ids: Optional[List[str]] = None,
    ):
        """
        Add validated items to the k-NN inventory.

        Call this after expert review confirms dimension assignments
        for new questionnaire items. The k-NN pool grows, improving
        future predictions.

        Parameters
        ----------
        item_texts : list of str
            Item texts to add
        hard_labels : list of int
            Validated dimension IDs
        e_ik : np.ndarray, optional, shape (n_items, n_real_dims)
            Soft label distributions. If None, creates one-hot from hard_labels.
        item_ids : list of str, optional
            Item identifiers for tracking
        """
        # Compute embeddings
        new_embeddings = self.encoder.encode(item_texts, convert_to_numpy=True)
        new_embeddings = new_embeddings / (np.linalg.norm(new_embeddings, axis=1, keepdims=True) + 1e-10)

        # Build E_ik if not provided (one-hot from hard labels)
        if e_ik is None:
            e_ik = np.zeros((len(item_texts), self.n_real_dims))
            real_dims = [self.dimensions[i] for i in self.real_dim_indices]
            for i, label in enumerate(hard_labels):
                if label in real_dims:
                    dim_idx = real_dims.index(label)
                    e_ik[i, dim_idx] = 1.0

        # Append to inventory
        if self.inventory_embeddings is not None:
            self.inventory_embeddings = np.vstack([self.inventory_embeddings, new_embeddings])
            self.inventory_e_ik = np.vstack([self.inventory_e_ik, e_ik])
            self.inventory_hard_labels = np.concatenate([
                self.inventory_hard_labels, np.array(hard_labels)
            ])
        else:
            self.inventory_embeddings = new_embeddings
            self.inventory_e_ik = e_ik
            self.inventory_hard_labels = np.array(hard_labels)

        if item_ids:
            self.inventory_item_ids.extend(item_ids)
        else:
            start = len(self.inventory_item_ids)
            self.inventory_item_ids.extend([f"new_{start+i}" for i in range(len(item_texts))])

        print(f"    Added {len(item_texts)} items to inventory "
              f"(total: {len(self.inventory_embeddings)})")

    def save(self, path: str):
        encoder_path = str(Path(path).parent / f"{Path(path).stem}_encoder")
        self.encoder.save(encoder_path)

        state = {
            'prototypes': self.prototypes,
            'weights': self.weights,
            'biases': self.biases,
            'tau_max': self.tau_max,
            'tau_margin': self.tau_margin,
            'p_minus1_base': self.p_minus1_base,
            'temperature': self.temperature,
            'dimensions': self.dimensions,
            'dimension_labels': self.dimension_labels,
            'real_dim_indices': self.real_dim_indices,
            'real_dim_labels': self.real_dim_labels,
            'n_real_dims': self.n_real_dims,
            'minus1_idx': self.minus1_idx,
            'model_name': self.model_name,
            'encoder_path': encoder_path,
            'knn_k': self.knn_k,
            'knn_alpha': self.knn_alpha,
            'class_balance': self.class_balance,
            # Inventory
            'inventory_embeddings': self.inventory_embeddings,
            'inventory_e_ik': self.inventory_e_ik,
            'inventory_hard_labels': self.inventory_hard_labels,
            'inventory_item_ids': self.inventory_item_ids,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"    Saved model to {path} (inventory: {len(self.inventory_embeddings)} items)")

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'PrototypeModel':
        with open(path, 'rb') as f:
            state = pickle.load(f)
        model = cls(
            model_name=state['model_name'],
            device=device,
            knn_k=state.get('knn_k', 5),
            knn_alpha=state.get('knn_alpha', 0.7),
            class_balance=state.get('class_balance', False),
        )
        for key, value in state.items():
            if key not in ['encoder_path', 'model_name']:
                setattr(model, key, value)
        encoder_path = str(Path(path).parent / f"{Path(path).stem}_encoder")
        model.encoder = SentenceTransformer(encoder_path, device=device)
        model.fitted = True
        return model


# =============================================================================
# Contrastive Model
# =============================================================================

class ContrastiveModel:
    """
    Contrastive fine-tuning model with hybrid prototype + k-NN inference.

    Training:
    - Fine-tunes embedding model with contrastive learning
    - Supports hard negative mining for boundary learning
    - Then learns prototypes and calibration on fine-tuned embeddings

    Inference (same as PrototypeModel):
    - Prototype score: cosine similarity to dimension centroids
    - k-NN score: similarity-weighted vote from k nearest inventory items
    - Combined: α × prototype + (1-α) × k-NN
    """

    def __init__(
        self,
        model_name: str = "hkunlp/instructor-large",
        device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
        learning_rate: float = 2e-5,
        epochs: int = 5,
        batch_size: int = 8,
        margin: float = 0.3,
        hard_negative_boost: float = 3.0,
        hard_negative_percentile: float = 75.0,
        knn_k: int = 5,
        knn_alpha: float = 0.7,
        class_balance: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.margin = margin
        self.hard_negative_boost = hard_negative_boost
        self.hard_negative_percentile = hard_negative_percentile
        self.knn_k = knn_k
        self.knn_alpha = knn_alpha
        self.class_balance = class_balance

        self.encoder = None
        self.prototypes = None
        self.weights = None
        self.biases = None
        self.tau_max = 0.5
        self.tau_margin = 0.02
        self.p_minus1_base = 0.1
        self.temperature = 1.0
        self.dimensions = None
        self.dimension_labels = None
        self.real_dim_indices = None
        self.real_dim_labels = None
        self.n_real_dims = None
        self.minus1_idx = None
        self.fitted = False

        # Inventory for k-NN
        self.inventory_embeddings = None
        self.inventory_e_ik = None
        self.inventory_hard_labels = None
        self.inventory_item_ids = []

    def _create_pairs(
        self,
        item_texts: list,
        y_soft,
        real_dim_indices: list,
        minus1_col: int = None,
        is_minus1_override: np.ndarray = None,
    ) -> list:
        """
        Create contrastive pairs weighted by E_ik, with hard negative mining.

        Items with dimension -1 ("does not fit") are treated as explicit negatives
        against all real-dimension items, pushing them away from every real prototype
        during fine-tuning so the threshold-based -1 detector has accurate geometry.
        No positive pairs are created among -1 items — they are heterogeneous by
        definition and should not be pulled together.

        is_minus1_override: if provided, overrides the internal plurality-vote -1
        detection. Used to exclude held-out calibration -1 items from push pairs.
        """
        hard_negative_boost = self.hard_negative_boost
        hard_negative_percentile = self.hard_negative_percentile

        n = len(item_texts)
        e_dims = y_soft[:, real_dim_indices]

        # Identify -1 items: use explicit override if provided (for calibration split),
        # otherwise fall back to plurality vote in the full soft-label row.
        if is_minus1_override is not None:
            is_minus1 = is_minus1_override
        elif minus1_col is not None:
            is_minus1 = np.array([np.argmax(y_soft[i]) == minus1_col for i in range(n)])
        else:
            is_minus1 = np.zeros(n, dtype=bool)
        n_minus1 = int(is_minus1.sum())

        print(f"      Computing base embeddings for hard negative mining...")
        with torch.no_grad():
            base_embeddings = self.encoder.encode(
                item_texts, convert_to_numpy=True, show_progress_bar=False
            )
        norms = np.linalg.norm(base_embeddings, axis=1, keepdims=True)
        base_embeddings = base_embeddings / np.where(norms == 0, 1e-8, norms)
        base_sim_matrix = base_embeddings @ base_embeddings.T

        positive_pairs = []
        negative_pairs = []
        n_minus1_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                base_sim = float(base_sim_matrix[i, j])

                if not is_minus1[i] and not is_minus1[j]:
                    # Both real-dim items — existing logic unchanged
                    e_i = e_dims[i].copy()
                    e_j = e_dims[j].copy()
                    e_i = e_i / (e_i.sum() + 1e-10)
                    e_j = e_j / (e_j.sum() + 1e-10)
                    dim_i = np.argmax(e_i)
                    dim_j = np.argmax(e_j)
                    overlap = np.sum(np.minimum(e_i, e_j))

                    if dim_i == dim_j:
                        weight = overlap
                        if weight > 0.05:
                            positive_pairs.append((item_texts[i], item_texts[j], 1, weight))
                    else:
                        weight = e_i[dim_i] * e_j[dim_j]
                        if weight > 0.05:
                            negative_pairs.append({
                                'text_i': item_texts[i], 'text_j': item_texts[j],
                                'weight': weight, 'base_sim': base_sim,
                            })

                elif is_minus1[i] != is_minus1[j]:
                    # One -1 item, one real-dim item — always a negative pair.
                    # Weight = P(-1 | item_mi) × P(top_dim | item_ri), consistent
                    # with the real-vs-real negative weight formula.
                    mi = i if is_minus1[i] else j
                    ri = j if is_minus1[i] else i
                    e_real = e_dims[ri].copy()
                    e_real = e_real / (e_real.sum() + 1e-10)
                    weight = float(y_soft[mi, minus1_col]) * float(e_real[np.argmax(e_real)])
                    if weight > 0.05:
                        negative_pairs.append({
                            'text_i': item_texts[mi], 'text_j': item_texts[ri],
                            'weight': weight, 'base_sim': base_sim,
                        })
                        n_minus1_pairs += 1
                # Both -1: no pair — don't pull heterogeneous items together

        n_hard = 0
        if negative_pairs:
            neg_sims = np.array([p['base_sim'] for p in negative_pairs])
            hard_threshold = np.percentile(neg_sims, hard_negative_percentile)
            final_negative_pairs = []
            for p in negative_pairs:
                w = p['weight']
                if p['base_sim'] >= hard_threshold:
                    w *= hard_negative_boost
                    n_hard += 1
                final_negative_pairs.append((p['text_i'], p['text_j'], 0, w))
            print(f"      Hard negative threshold (sim >= {hard_threshold:.3f}): "
                  f"{n_hard}/{len(negative_pairs)} boosted x{hard_negative_boost}")
        else:
            final_negative_pairs = []

        pairs = positive_pairs + final_negative_pairs

        # Normalize weights so max = 1.0 (prevents gradient overflow
        # when boosted hard negative weights hit large-gradient architectures like T5)
        if pairs:
            max_weight = max(p[3] for p in pairs)
            if max_weight > 0:
                pairs = [(t1, t2, lbl, w / max_weight) for t1, t2, lbl, w in pairs]

        if n_minus1 > 0:
            print(f"      {n_minus1} 'does not fit' items → {n_minus1_pairs} additional negative pairs")
        print(f"      {len(pairs)} pairs: {len(positive_pairs)} positive, "
              f"{len(final_negative_pairs)} negative ({n_hard} hard)")
        return pairs

    def fit(
        self,
        item_texts: List[str],
        y_soft: np.ndarray,
        dimensions: List[int],
        dimension_labels: List[str]
    ):
        """Fine-tune with contrastive learning, then learn prototypes, calibration, and store inventory."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for contrastive learning")

        self.dimensions = dimensions
        self.dimension_labels = dimension_labels
        self.minus1_idx = dimensions.index(-1) if -1 in dimensions else None
        self.real_dim_indices = [i for i, d in enumerate(dimensions) if d != -1]
        self.n_real_dims = len(self.real_dim_indices)
        self.real_dim_labels = [dimension_labels[i] for i in self.real_dim_indices]

        print(f"    Loading encoder: {self.model_name}")
        self.encoder = SentenceTransformer(self.model_name, device=self.device)

        n = len(item_texts)

        # Split -1 items: ~70% pushed during contrastive training, ~30% held out
        # for threshold calibration. Held-out items are NOT pushed, so their
        # max_sim values reflect natural geometry — suitable reference for tau_max.
        calib_minus1_mask = np.zeros(n, dtype=bool)
        is_minus1_for_training = None
        if self.minus1_idx is not None:
            all_minus1 = np.where(
                [np.argmax(y_soft[i]) == self.minus1_idx for i in range(n)]
            )[0]
            if len(all_minus1) >= 4:
                n_calib = max(1, int(len(all_minus1) * 0.3))
                rng = np.random.default_rng(42)
                shuffled = rng.permutation(all_minus1)
                calib_idx = shuffled[:n_calib]
                calib_minus1_mask[calib_idx] = True
                # Training mask: -1 items not in the calibration hold-out
                is_minus1_for_training = np.array(
                    [np.argmax(y_soft[i]) == self.minus1_idx for i in range(n)]
                )
                is_minus1_for_training[calib_minus1_mask] = False
                print(f"    -1 split: {len(all_minus1) - n_calib} training "
                      f"(pushed) / {n_calib} calibration (held out)")
            else:
                # Too few to split — treat all as calibration (no push)
                calib_minus1_mask[all_minus1] = True
                is_minus1_for_training = np.zeros(n, dtype=bool)
                print(f"    Only {len(all_minus1)} -1 items — all held out for "
                      f"calibration, none used in push pairs")

        print(f"    Creating contrastive pairs from E_{{i,k}}...")
        pairs = self._create_pairs(
            item_texts, y_soft, self.real_dim_indices,
            minus1_col=self.minus1_idx,
            is_minus1_override=is_minus1_for_training,
        )
        n_pos = sum(1 for p in pairs if p[2] == 1)
        n_neg = len(pairs) - n_pos
        print(f"      {len(pairs)} pairs: {n_pos} positive, {n_neg} negative")

        if len(pairs) < 20:
            print(f"      Too few pairs - falling back to prototype learning only")
            return self._fit_prototype_only(item_texts, y_soft)

        # Fine-tuning loop (architecture-robust: works with BERT, T5, RoBERTa, MiniLM)
        print(f"    Fine-tuning encoder ({self.epochs} epochs)...")

        # Ensure all parameters are float32 (some models load in float16)
        for param in self.encoder.parameters():
            if param.dtype != torch.float32:
                param.data = param.data.float()

        self.encoder.train()
        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.learning_rate)

        # Gradient clipping max norm — essential for T5 and large models
        max_grad_norm = 1.0

        for epoch in range(self.epochs):
            np.random.shuffle(pairs)
            total_loss = 0
            n_batches = 0
            n_nan_skipped = 0

            for batch_start in range(0, len(pairs), self.batch_size):
                batch = pairs[batch_start:batch_start + self.batch_size]
                texts_i = [p[0] for p in batch]
                texts_j = [p[1] for p in batch]
                labels = torch.tensor([p[2] for p in batch], dtype=torch.float32, device=self.device)
                weights = torch.tensor([p[3] for p in batch], dtype=torch.float32, device=self.device)

                # Tokenize and move to device
                features_i = self.encoder.tokenize(texts_i)
                features_j = self.encoder.tokenize(texts_j)
                features_i = {k: v.to(self.device) for k, v in features_i.items()}
                features_j = {k: v.to(self.device) for k, v in features_j.items()}

                # Forward pass with gradients
                output_i = self.encoder.forward(features_i)
                output_j = self.encoder.forward(features_j)

                # Get sentence embeddings and ensure float32
                emb_i = output_i['sentence_embedding'].float()
                emb_j = output_j['sentence_embedding'].float()

                # Normalize
                emb_i = torch.nn.functional.normalize(emb_i, p=2, dim=1)
                emb_j = torch.nn.functional.normalize(emb_j, p=2, dim=1)

                # Cosine similarity
                sim = torch.sum(emb_i * emb_j, dim=1)

                # Contrastive loss
                pos_loss = labels * weights * (1 - sim) ** 2
                neg_loss = (1 - labels) * weights * torch.clamp(sim - self.margin, min=0) ** 2
                loss = (pos_loss + neg_loss).mean()

                # NaN detection — skip batch if loss is NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    n_nan_skipped += 1
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping — prevents explosion in T5 and deep models
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_grad_norm)

                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches if n_batches > 0 else float('nan')
            nan_msg = f" ({n_nan_skipped} NaN batches skipped)" if n_nan_skipped > 0 else ""
            print(f"      Epoch {epoch+1}/{self.epochs}: loss={avg_loss:.4f}{nan_msg}")

            # If all batches were NaN, stop early
            if n_batches == 0:
                print(f"      WARNING: All batches produced NaN — stopping fine-tuning")
                print(f"      Falling back to prototype learning on base embeddings")
                self.encoder.eval()
                return self._fit_prototype_only(item_texts, y_soft)

        self.encoder.eval()

        # Compute fine-tuned embeddings
        print(f"    Computing fine-tuned embeddings...")
        with torch.no_grad():
            embeddings = self.encoder.encode(item_texts, convert_to_numpy=True)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

        # Store training items as initial inventory
        e_dims = y_soft[:, self.real_dim_indices]
        print(f"    Storing {len(item_texts)} training items as k-NN inventory...")
        self.inventory_embeddings = embeddings.copy()
        self.inventory_e_ik = e_dims.copy()
        self.inventory_hard_labels = np.array([
            dimensions[np.argmax(y_soft[i])] for i in range(len(item_texts))
        ])
        self.inventory_item_ids = list(range(len(item_texts)))

        # Build threshold calibration mask: real-dim items + held-out -1 items.
        # Excludes training -1 items whose embeddings were artificially pushed
        # far from prototypes by the contrastive loss.
        if self.minus1_idx is not None:
            all_minus1_mask = np.array(
                [np.argmax(y_soft[i]) == self.minus1_idx for i in range(n)]
            )
            # real-dim items OR held-out calibration -1 items
            thresh_calib_mask = (~all_minus1_mask) | calib_minus1_mask
        else:
            thresh_calib_mask = None

        # Learn prototypes and calibration
        self._learn_prototypes_and_calibration(embeddings, y_soft, thresh_calib_mask)

        self.fitted = True
        return self

    def _fit_prototype_only(self, item_texts, y_soft):
        embeddings = self.encoder.encode(item_texts, convert_to_numpy=True)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        # Store inventory
        e_dims = y_soft[:, self.real_dim_indices]
        self.inventory_embeddings = embeddings.copy()
        self.inventory_e_ik = e_dims.copy()
        self.inventory_hard_labels = np.array([
            self.dimensions[np.argmax(y_soft[i])] for i in range(len(item_texts))
        ])
        self.inventory_item_ids = list(range(len(item_texts)))
        # No -1 push occurred, so all items represent natural geometry — no mask needed
        self._learn_prototypes_and_calibration(embeddings, y_soft, thresh_calib_mask=None)
        self.fitted = True
        return self

    def _learn_prototypes_and_calibration(self, embeddings, y_soft, thresh_calib_mask=None):
        """
        thresh_calib_mask: boolean array (n_items,). When provided, the tau_max/tau_margin
        grid search uses only the masked items (real-dim training items + held-out -1 items).
        This prevents artificially-pushed training -1 items from biasing tau_max too low.
        """
        e_dims = y_soft[:, self.real_dim_indices]
        if self.minus1_idx is not None:
            e_minus1 = y_soft[:, self.minus1_idx]
        else:
            e_minus1 = np.zeros(len(embeddings))

        print(f"    Learning prototypes...")
        embedding_dim = embeddings.shape[1]
        self.prototypes = np.zeros((self.n_real_dims, embedding_dim))
        for k in range(self.n_real_dims):
            w_k = e_dims[:, k]
            if w_k.sum() > 0:
                self.prototypes[k] = np.average(embeddings, axis=0, weights=w_k + 1e-10)
                self.prototypes[k] /= (np.linalg.norm(self.prototypes[k]) + 1e-10)

        similarities = embeddings @ self.prototypes.T
        max_sim = np.max(similarities, axis=1)
        sorted_sims = np.sort(similarities, axis=1)[:, ::-1]
        margin = sorted_sims[:, 0] - sorted_sims[:, 1] if self.n_real_dims > 1 else sorted_sims[:, 0]

        # Use calibration subset for threshold search to avoid bias from pushed -1 items.
        # If no mask, use all items (e.g., prototype-only fallback where no push occurred).
        if thresh_calib_mask is not None:
            calib_max_sim = max_sim[thresh_calib_mask]
            calib_margin  = margin[thresh_calib_mask]
            calib_e_minus1 = e_minus1[thresh_calib_mask]
            n_calib_minus1 = int((e_minus1[thresh_calib_mask] > 0.3).sum())
            print(f"    Learning thresholds from calibration subset "
                  f"({thresh_calib_mask.sum()} items, {n_calib_minus1} -1 items)...")
        else:
            calib_max_sim, calib_margin, calib_e_minus1 = max_sim, margin, e_minus1
            print(f"    Learning thresholds from E_{{i,-1}}...")

        if calib_e_minus1.sum() > 0:
            best_score = -np.inf
            for tau_max in np.arange(0.30, 0.75, 0.025):
                for tau_margin in np.arange(0.005, 0.12, 0.01):
                    for p_base in [0.02, 0.05, 0.10, 0.15]:
                        p_m1 = self._compute_p_minus1(calib_max_sim, calib_margin, tau_max, tau_margin, p_base)
                        if np.std(p_m1) > 0 and np.std(calib_e_minus1) > 0:
                            corr = np.corrcoef(p_m1, calib_e_minus1)[0, 1]
                            if not np.isnan(corr) and corr > best_score:
                                best_score = corr
                                self.tau_max, self.tau_margin, self.p_minus1_base = tau_max, tau_margin, p_base

        print(f"    Learning calibration from E_{{i,k}}...")
        self.weights = np.ones(self.n_real_dims)
        self.biases = np.zeros(self.n_real_dims)

        def objective(params):
            w = params[:self.n_real_dims]
            b = params[self.n_real_dims:2*self.n_real_dims]
            temp = params[2*self.n_real_dims]
            cal = similarities * w + b
            exp_s = np.exp(cal / temp - np.max(cal / temp, axis=1, keepdims=True))
            probs = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-10)
            if self.minus1_idx is not None:
                p_m1 = self._compute_p_minus1(max_sim, margin, self.tau_max, self.tau_margin, self.p_minus1_base)
                probs = probs * (1 - p_m1[:, np.newaxis])
            e_safe = (e_dims + 1e-10) / (e_dims + 1e-10).sum(axis=1, keepdims=True)
            p_safe = (probs + 1e-10) / (probs + 1e-10).sum(axis=1, keepdims=True)
            kl = np.sum(e_safe * (np.log(e_safe) - np.log(p_safe)), axis=1).mean()
            return kl + 0.01 * (np.sum((w-1)**2) + np.sum(b**2))

        x0 = np.concatenate([np.ones(self.n_real_dims), np.zeros(self.n_real_dims), [1.0]])
        bounds = [(0.1, 5.0)] * self.n_real_dims + [(-1.0, 1.0)] * self.n_real_dims + [(0.1, 5.0)]
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        self.weights = result.x[:self.n_real_dims]
        self.biases = result.x[self.n_real_dims:2*self.n_real_dims]
        self.temperature = result.x[2*self.n_real_dims]

        print(f"\n    Learned parameters:")
        print(f"      Thresholds: τ_max={self.tau_max:.3f}, τ_margin={self.tau_margin:.3f}, p_base={self.p_minus1_base:.2f}")
        print(f"      Temperature: {self.temperature:.3f}")
        print(f"      k-NN: k={self.knn_k}, α={self.knn_alpha:.2f}")
        print(f"      Inventory size: {len(self.inventory_embeddings)} items")
        for i, label in enumerate(self.real_dim_labels):
            print(f"      {label[:40]:<40}: w={self.weights[i]:.3f}, b={self.biases[i]:+.3f}")

    def _compute_p_minus1(self, max_sim, margin, tau_max, tau_margin, p_base):
        dist = np.minimum((tau_max - max_sim) / (tau_max + 1e-10),
                          (tau_margin - margin) / (tau_margin + 1e-10))
        return np.clip(p_base + (1 - p_base) / (1 + np.exp(-5 * dist)), 0.01, 0.99)

    def predict(self, item_texts: List[str]) -> np.ndarray:
        proba = self.predict_proba(item_texts)
        return np.array([self.dimensions[i] for i in np.argmax(proba, axis=1)])

    def predict_proba(self, item_texts: List[str]) -> np.ndarray:
        """Predict using hybrid prototype + k-NN."""
        emb = self.encoder.encode(item_texts, convert_to_numpy=True)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)

        # Prototype scores
        sim = emb @ self.prototypes.T
        max_s = np.max(sim, axis=1)
        sorted_s = np.sort(sim, axis=1)[:, ::-1]
        margin = sorted_s[:, 0] - sorted_s[:, 1] if self.n_real_dims > 1 else sorted_s[:, 0]

        cal = sim * self.weights + self.biases
        exp_s = np.exp(cal / self.temperature - np.max(cal / self.temperature, axis=1, keepdims=True))
        proto_probs = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-10)

        # k-NN scores
        use_knn = (
            self.knn_alpha < 1.0
            and self.inventory_embeddings is not None
            and len(self.inventory_embeddings) > 0
        )

        if use_knn:
            knn_probs = compute_knn_scores(
                emb, self.inventory_embeddings,
                self.inventory_e_ik, k=self.knn_k
            )
            dim_probs = self.knn_alpha * proto_probs + (1 - self.knn_alpha) * knn_probs
        else:
            dim_probs = proto_probs

        # --- Class balance correction ---
        if self.class_balance and self.inventory_e_ik is not None:
            freq = self.inventory_e_ik.sum(axis=0)
            freq = freq / (freq.sum() + 1e-10)
            dim_probs = dim_probs / (freq + 1e-10)
            dim_probs = dim_probs / (dim_probs.sum(axis=1, keepdims=True) + 1e-10)

        # P(-1) and assembly
        p_m1 = self._compute_p_minus1(max_s, margin, self.tau_max, self.tau_margin, self.p_minus1_base)

        proba = np.zeros((len(item_texts), len(self.dimensions)))
        if self.minus1_idx is not None:
            proba[:, self.minus1_idx] = p_m1
            for j, idx in enumerate(self.real_dim_indices):
                proba[:, idx] = dim_probs[:, j] * (1 - p_m1)
        else:
            for j, idx in enumerate(self.real_dim_indices):
                proba[:, idx] = dim_probs[:, j]
        return proba

    def add_to_inventory(
        self,
        item_texts: List[str],
        hard_labels: List[int],
        e_ik: Optional[np.ndarray] = None,
        item_ids: Optional[List[str]] = None,
    ):
        """
        Add validated items to the k-NN inventory.

        Call after expert review to grow the reference set.
        """
        new_embeddings = self.encoder.encode(item_texts, convert_to_numpy=True)
        new_embeddings = new_embeddings / (np.linalg.norm(new_embeddings, axis=1, keepdims=True) + 1e-10)

        if e_ik is None:
            e_ik = np.zeros((len(item_texts), self.n_real_dims))
            real_dims = [self.dimensions[i] for i in self.real_dim_indices]
            for i, label in enumerate(hard_labels):
                if label in real_dims:
                    dim_idx = real_dims.index(label)
                    e_ik[i, dim_idx] = 1.0

        if self.inventory_embeddings is not None:
            self.inventory_embeddings = np.vstack([self.inventory_embeddings, new_embeddings])
            self.inventory_e_ik = np.vstack([self.inventory_e_ik, e_ik])
            self.inventory_hard_labels = np.concatenate([
                self.inventory_hard_labels, np.array(hard_labels)
            ])
        else:
            self.inventory_embeddings = new_embeddings
            self.inventory_e_ik = e_ik
            self.inventory_hard_labels = np.array(hard_labels)

        if item_ids:
            self.inventory_item_ids.extend(item_ids)
        else:
            start = len(self.inventory_item_ids)
            self.inventory_item_ids.extend([f"new_{start+i}" for i in range(len(item_texts))])

        print(f"    Added {len(item_texts)} items to inventory "
              f"(total: {len(self.inventory_embeddings)})")

    def save(self, path: str):
        encoder_path = str(Path(path).parent / f"{Path(path).stem}_encoder")
        self.encoder.save(encoder_path)
        state = {
            'prototypes': self.prototypes, 'weights': self.weights, 'biases': self.biases,
            'tau_max': self.tau_max, 'tau_margin': self.tau_margin, 'p_minus1_base': self.p_minus1_base,
            'temperature': self.temperature, 'dimensions': self.dimensions,
            'dimension_labels': self.dimension_labels, 'real_dim_indices': self.real_dim_indices,
            'real_dim_labels': self.real_dim_labels, 'n_real_dims': self.n_real_dims,
            'minus1_idx': self.minus1_idx, 'model_name': self.model_name, 'encoder_path': encoder_path,
            'knn_k': self.knn_k, 'knn_alpha': self.knn_alpha,
            'class_balance': self.class_balance,
            'inventory_embeddings': self.inventory_embeddings,
            'inventory_e_ik': self.inventory_e_ik,
            'inventory_hard_labels': self.inventory_hard_labels,
            'inventory_item_ids': self.inventory_item_ids,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"    Saved model to {path} (inventory: {len(self.inventory_embeddings)} items)")

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'ContrastiveModel':
        with open(path, 'rb') as f:
            state = pickle.load(f)
        model = cls(
            model_name=state['model_name'],
            device=device,
            knn_k=state.get('knn_k', 5),
            knn_alpha=state.get('knn_alpha', 0.7),
            class_balance=state.get('class_balance', False),
        )
        for k, v in state.items():
            if k not in ['encoder_path', 'model_name']:
                setattr(model, k, v)
        encoder_path = str(Path(path).parent / f"{Path(path).stem}_encoder")
        model.encoder = SentenceTransformer(encoder_path, device=device)
        model.fitted = True
        return model


# =============================================================================
# Evaluation & Cross-Validation
# =============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kappa = cohen_kappa_score(y_true, y_pred)
    except:
        kappa = 0.0
    mask = y_true != -1
    acc_no_minus1 = accuracy_score(y_true[mask], y_pred[mask]) if mask.sum() > 0 else 0.0
    return {
        'accuracy': acc,
        'accuracy_no_minus1': acc_no_minus1,
        'f1_macro': f1,
        'kappa': kappa if not np.isnan(kappa) else 0.0,
        'n_samples': len(y_true),
        'n_correct': int((y_true == y_pred).sum())
    }


def run_logo_cv(
    df: pd.DataFrame,
    approach: str,
    model_name: str,
    device: str,
    contrastive_epochs: int = 5,
    learning_rate: float = 2e-5,
    margin: float = 0.3,
    hard_negative_boost: float = 3.0,
    hard_negative_percentile: float = 75.0,
    knn_k: int = 5,
    knn_alpha: float = 0.7,
    class_balance: bool = False,
) -> List[Dict]:
    """Run leave-one-questionnaire-out cross-validation."""
    questionnaires = df['questionnaire'].unique()
    print(f"\n  Running LOGO CV with {len(questionnaires)} folds")

    all_results = []

    for test_q in questionnaires:
        print(f"\n  === Fold: {test_q} (held out) ===")

        train_df = df[df['questionnaire'] != test_q]
        test_df = df[df['questionnaire'] == test_q]

        if len(test_df) < 1 or len(train_df) < 5:
            print(f"    Skipped (insufficient data)")
            continue

        texts_train, y_hard_train, y_soft_train, dimensions, dim_labels, _ = prepare_training_data(train_df)
        texts_test, y_hard_test, _, _, _, _ = prepare_training_data(test_df)

        print(f"    Train: {len(texts_train)} items, Test: {len(texts_test)} items")

        if approach == 'prototype':
            model = PrototypeModel(model_name=model_name, device=device,
                                   knn_k=knn_k, knn_alpha=knn_alpha,
                                   class_balance=class_balance)
        else:
            model = ContrastiveModel(
                model_name=model_name, device=device,
                epochs=contrastive_epochs, learning_rate=learning_rate,
                margin=margin, hard_negative_boost=hard_negative_boost,
                hard_negative_percentile=hard_negative_percentile,
                knn_k=knn_k, knn_alpha=knn_alpha,
                class_balance=class_balance,
            )

        model.fit(texts_train, y_soft_train, dimensions, dim_labels)

        y_pred = model.predict(texts_test)
        metrics = evaluate(y_hard_test, y_pred)

        print(f"\n    Results: Acc={metrics['accuracy']:.3f}, "
              f"Acc(no -1)={metrics['accuracy_no_minus1']:.3f}, F1={metrics['f1_macro']:.3f}")

        all_results.append({
            'fold': test_q,
            'n_train': len(train_df),
            'n_test': len(test_df),
            **metrics
        })

    if all_results:
        avg_acc = np.mean([r['accuracy'] for r in all_results])
        avg_acc_no_m1 = np.mean([r['accuracy_no_minus1'] for r in all_results])
        avg_f1 = np.mean([r['f1_macro'] for r in all_results])
        print(f"\n  ══════════════════════════════════════")
        print(f"  AVERAGE: Acc={avg_acc:.3f}, Acc(no -1)={avg_acc_no_m1:.3f}, F1={avg_f1:.3f}")
        print(f"  ══════════════════════════════════════")

    return all_results


# =============================================================================
# Main
# =============================================================================

ALL_CONSTRUCTS = ['depression', 'anxiety', 'psychosis', 'apathy', 'sleep', 'impulse_control']


def run_single_construct(
    df: pd.DataFrame,
    construct: str,
    approach: str,
    model_name: str,
    device: str,
    output_dir: Path,
    run_cv: bool,
    epochs: int,
    learning_rate: float = 2e-5,
    margin: float = 0.3,
    hard_negative_boost: float = 3.0,
    hard_negative_percentile: float = 75.0,
    knn_k: int = 5,
    knn_alpha: float = 0.7,
    class_balance: bool = False,
):
    """Run training for a single construct."""
    print(f"\n{'#' * 70}")
    print(f"# CONSTRUCT: {construct.upper()}")
    print(f"{'#' * 70}")

    item_texts, y_hard, y_soft, dimensions, dim_labels, _ = prepare_training_data(df)
    print(f"\nLoaded {len(df)} items from {len(df['questionnaire'].unique())} questionnaires")
    print(f"Dimensions: {[f'{d}={l[:30]}' for d, l in zip(dimensions, dim_labels)]}")

    construct_output_dir = output_dir / construct
    construct_output_dir.mkdir(parents=True, exist_ok=True)

    approaches = ['prototype', 'contrastive'] if approach == 'both' else [approach]

    for app in approaches:
        print(f"\n{'=' * 70}")
        print(f"APPROACH: {app.upper()}")
        print("=" * 70)

        if run_cv:
            results = run_logo_cv(
                df, app, model_name, device,
                contrastive_epochs=epochs, learning_rate=learning_rate,
                margin=margin, hard_negative_boost=hard_negative_boost,
                hard_negative_percentile=hard_negative_percentile,
                knn_k=knn_k, knn_alpha=knn_alpha,
                class_balance=class_balance,
            )
            results_df = pd.DataFrame(results)
            results_path = construct_output_dir / f'{app}_cv_results.csv'
            results_df.to_csv(results_path, index=False)
            print(f"\n  CV results saved to {results_path}")

        print(f"\n  Training final model on ALL data...")

        if app == 'prototype':
            model = PrototypeModel(model_name=model_name, device=device,
                                   knn_k=knn_k, knn_alpha=knn_alpha,
                                   class_balance=class_balance)
        else:
            model = ContrastiveModel(
                model_name=model_name, device=device,
                epochs=epochs, learning_rate=learning_rate,
                margin=margin, hard_negative_boost=hard_negative_boost,
                hard_negative_percentile=hard_negative_percentile,
                knn_k=knn_k, knn_alpha=knn_alpha,
                class_balance=class_balance,
            )

        model.fit(item_texts, y_soft, dimensions, dim_labels)

        model_path = construct_output_dir / f'{app}_model.pkl'
        model.save(str(model_path))

        y_pred = model.predict(item_texts)
        train_metrics = evaluate(y_hard, y_pred)
        print(f"\n  Training set performance: Acc={train_metrics['accuracy']:.3f}, F1={train_metrics['f1_macro']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Semantic learning for ENIGMA-PD harmonizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prototype with k-NN
  python semantic_learning.py --data finetuning_dataset.csv --construct depression \\
      --approach prototype --knn-k 5 --knn-alpha 0.7 --cv

  # Contrastive with hard negatives and k-NN
  python semantic_learning.py --data finetuning_dataset.csv --construct depression \\
      --approach contrastive --hard-negative-boost 3.0 --knn-k 5 --knn-alpha 0.7 \\
      --device cuda --cv

  # Pure prototype (no k-NN, same as v1 behavior)
  python semantic_learning.py --data finetuning_dataset.csv --construct depression \\
      --approach prototype --knn-alpha 1.0 --cv
        """
    )
    parser.add_argument('--data', '-d', required=True, help='Path to finetuning_dataset.csv')
    parser.add_argument('--construct', '-c', required=True,
                        choices=ALL_CONSTRUCTS + ['all'])
    parser.add_argument('--approach', '-a', choices=['prototype', 'contrastive', 'both'], default='prototype')
    parser.add_argument('--output', '-o', default='./models', help='Output directory')
    parser.add_argument('--model-name', default='hkunlp/instructor-large')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--cv', action='store_true', help='Run cross-validation')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning-rate', '--lr', type=float, default=2e-5)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--hard-negative-boost', type=float, default=3.0,
                        help='Weight multiplier for hard negatives (1.0=disabled)')
    parser.add_argument('--hard-negative-percentile', type=float, default=75.0,
                        help='Similarity percentile for hard negatives')
    parser.add_argument('--knn-k', type=int, default=5,
                        help='Number of nearest neighbors (default: 5)')
    parser.add_argument('--knn-alpha', type=float, default=0.7,
                        help='Prototype vs k-NN mixing weight (1.0=pure prototype, 0.0=pure k-NN)')
    parser.add_argument('--class-balance', action='store_true',
                        help='Divide dimension scores by their training frequency before '
                             'renormalizing, compensating for unequal class sizes. '
                             'Recomputed from the current inventory at prediction time, '
                             'so it stays correct as new items are added without retraining.')

    args = parser.parse_args()

    if args.device == 'cuda' and TORCH_AVAILABLE and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    if args.construct == 'all':
        constructs = ALL_CONSTRUCTS
    else:
        constructs = [args.construct]

    print("=" * 70)
    print(f"SEMANTIC LEARNING")
    print(f"Constructs: {', '.join(constructs)}")
    print(f"Approach: {args.approach}")
    print(f"Device: {args.device}")
    if args.approach in ('contrastive', 'both'):
        print(f"Epochs: {args.epochs}, LR: {args.learning_rate}, Margin: {args.margin}")
        print(f"Hard negative boost: {args.hard_negative_boost}, Percentile: {args.hard_negative_percentile}")
    print(f"k-NN: k={args.knn_k}, α={args.knn_alpha}")
    print(f"Class balance: {args.class_balance}")
    print("=" * 70)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(args.data)
    full_df['e_ik_dict'] = full_df['expert_proportions_e_ik'].apply(parse_e_ik)

    for construct in constructs:
        df = full_df[full_df['construct'] == construct].copy()
        if len(df) == 0:
            print(f"\nWARNING: No data for '{construct}', skipping...")
            continue
        run_single_construct(
            df=df, construct=construct, approach=args.approach,
            model_name=args.model_name, device=args.device,
            output_dir=output_dir, run_cv=args.cv,
            epochs=args.epochs, learning_rate=args.learning_rate,
            margin=args.margin, hard_negative_boost=args.hard_negative_boost,
            hard_negative_percentile=args.hard_negative_percentile,
            knn_k=args.knn_k, knn_alpha=args.knn_alpha,
            class_balance=args.class_balance,
        )


if __name__ == "__main__":
    main()