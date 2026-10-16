"""
Harmonizer: Production-Ready Questionnaire Harmonization Tool
==============================================================

This module provides the Harmonizer class for harmonizing questionnaire items
to standardized dimensions. It implements:

1. LOOKUP TABLE: Known items return stored assignments 
2. MODEL PREDICTION: New items get predicted assignments with confidence scores
3. AUTO-UPDATE: New predictions are automatically added to the inventory

Usage:
------
from harmonizer import Harmonizer

# Auto-downloads and caches the fine-tuned model for this construct from
# Hugging Face (see HF_MODEL_REPOS below) -- no local model_dir needed:
harmonizer = Harmonizer(
    inventory_path='harmonized_inventory.csv',
    auto_update=True
)

# Or use your own local copy instead:
harmonizer = Harmonizer(
    inventory_path='harmonized_inventory.csv',
    model_dir='./models',
    auto_update=True
)

results = harmonizer.harmonize(items_df=my_questionnaire, construct='depression')
"""

import json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
import pickle

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent.parent.parent
DEFAULT_DIMENSION_CONFIG = PROJECT_DIR / "input" / "shared" / "dimension_descriptions.yaml"

# Where each construct's fine-tuned model lives on Hugging Face, used to
# auto-download models when Harmonizer isn't given a local --model-dir.
# Point --hf-repo / the hf_repo_map constructor arg at your own repos to use
# your own fine-tuned models instead.
HF_MODEL_REPOS = {
    'depression':      'julia-pfarr/HarmoniCA_depression',
    'anxiety':         'julia-pfarr/HarmoniCA_anxiety',
    'psychosis':       'julia-pfarr/HarmoniCA_psychosis',
    'sleep':           'julia-pfarr/HarmoniCA_sleep',
    'impulse_control': 'julia-pfarr/HarmoniCA_impulse-control',
    'apathy':          'julia-pfarr/HarmoniCA_apathy',
}

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    from semantic_learning import ContrastiveModel, PrototypeModel
    SL_AVAILABLE = True
except ImportError:
    SL_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HarmonizationResult:
    """Result for a single item harmonization."""
    item_id: str
    item_text: str
    construct: str
    questionnaire: str
    dimension: int
    dimension_label: str
    confidence: float
    confidence_type: str
    source: str
    model_version: Optional[str] = None
    date_harmonized: Optional[str] = None
    probability_distribution: Optional[Dict[int, float]] = None


# =============================================================================
# Dimension Labels
# =============================================================================

def load_dimension_labels(path: Union[str, Path] = DEFAULT_DIMENSION_CONFIG) -> Dict[str, Dict[int, str]]:
    """Load construct -> dimension_id -> label from the shared dimension config.

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
    return config['dimension_labels']


DIMENSION_LABELS = load_dimension_labels()


# =============================================================================
# Harmonized Inventory
# =============================================================================

class HarmonizedInventory:
    """Manages the harmonized inventory of questionnaire items."""
    
    REQUIRED_COLUMNS = [
        'item_id', 'item_text', 'construct', 'questionnaire',
        'dimension', 'dimension_label', 'confidence', 'source'
    ]
    
    def __init__(self, path: Optional[str] = None):
        self.path = path
        
        if path and Path(path).exists():
            self.df = pd.read_csv(path)
            if 'item_text' not in self.df.columns:
                # Stripped from the public harmonized_inventory.csv (copyright).
                # Lookups/kNN pooling degrade gracefully without real item
                # wording; only item_text itself is affected.
                self.df['item_text'] = ''
            self._validate()
        else:
            self.df = pd.DataFrame(columns=self.REQUIRED_COLUMNS + [
                'confidence_type', 'model_version', 'date_added',
                'expert_agreement', 'rating_distribution', 'notes'
            ])

    def _validate(self):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Inventory missing required columns: {missing}")
    
    def lookup(self, item_id: str, construct: str) -> Optional[Dict]:
        """Look up an item by ID and construct."""
        mask = (self.df['item_id'] == item_id) & (self.df['construct'] == construct)
        matches = self.df[mask]
        
        if len(matches) > 0:
            return matches.iloc[0].to_dict()
        return None
    
    def add_item(self, item_id: str, item_text: str, construct: str,
                 questionnaire: str, dimension: int, dimension_label: str,
                 confidence: float, source: str, **kwargs):
        """Add a new item to the inventory."""
        existing = self.lookup(item_id, construct)
        if existing is not None:
            warnings.warn(f"Item {item_id} already exists for {construct}. Skipping.")
            return
        
        new_row = {
            'item_id': item_id,
            'item_text': item_text,
            'construct': construct,
            'questionnaire': questionnaire,
            'dimension': dimension,
            'dimension_label': dimension_label,
            'confidence': confidence,
            'source': source,
            'date_added': datetime.now().isoformat(),
            **kwargs
        }
        
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
    
    def save(self, path: Optional[str] = None):
        """Save inventory to CSV."""
        save_path = path or self.path
        if save_path:
            self.df.to_csv(save_path, index=False)
    
    def summary(self) -> pd.DataFrame:
        """Get summary statistics."""
        if len(self.df) == 0:
            return pd.DataFrame()
        
        rows = []
        for construct in self.df['construct'].unique():
            df_c = self.df[self.df['construct'] == construct]
            rows.append({
                'construct': construct,
                'n_items': len(df_c),
                'n_questionnaires': df_c['questionnaire'].nunique(),
                'n_expert': (df_c['source'] == 'expert_consensus').sum(),
                'n_model': (df_c['source'] == 'model_prediction').sum(),
                'n_does_not_fit': (df_c['dimension'] == -1).sum()
            })
        return pd.DataFrame(rows)
    
    def __len__(self):
        return len(self.df)


# =============================================================================
# Harmonizer
# =============================================================================

class Harmonizer:
    """
    Main class for harmonizing questionnaire items to dimensions.

    Parameters
    ----------
    inventory_path : str
        Path to harmonized inventory CSV
    model_dir : str, optional
        Directory containing trained models locally (one subdirectory per
        construct: <model_dir>/<construct>/<model_type>_model.pkl). If not
        given (the default), models are downloaded and cached from Hugging
        Face instead -- see hf_repo_map / HF_MODEL_REPOS.
    hf_repo_map : Dict[str, str], optional
        construct -> Hugging Face repo id, used to auto-download models when
        model_dir is not given (default: HF_MODEL_REPOS). Point this at your
        own repos to use your own fine-tuned models.
    auto_update : bool
        Whether to automatically add new predictions to inventory
    model_type : str
        'contrastive' or 'prototype' (from semantic_learning.py)
    device : str
        Device for inference ('cpu' or 'cuda')
    """

    def __init__(
        self,
        inventory_path: str,
        model_dir: Optional[str] = None,
        hf_repo_map: Optional[Dict[str, str]] = None,
        auto_update: bool = True,
        model_type: str = 'contrastive',
        device: str = 'cpu',
        knn_alpha: Optional[float] = None,
        class_balance: Optional[bool] = None,
    ):
        self.inventory_path = inventory_path
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.hf_repo_map = hf_repo_map if hf_repo_map is not None else HF_MODEL_REPOS
        self.auto_update = auto_update
        self.model_type = model_type
        self.device = device
        self.knn_alpha = knn_alpha        # None = use model's stored value
        self.class_balance = class_balance  # None = use model's stored value

        self.inventory = HarmonizedInventory(inventory_path)
        self.models = {}
        self.model_version = f"{model_type}_v1.0"

    def _download_from_huggingface(self, construct: str) -> Path:
        """Download (and cache, via huggingface_hub's own cache) a construct's
        model repo, returning the local snapshot directory."""
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "huggingface_hub is required to auto-download models "
                "(pip install huggingface_hub, already in requirements.txt), "
                "or pass model_dir / --model-dir to use a local copy instead."
            )
        repo_id = self.hf_repo_map.get(construct)
        if repo_id is None:
            raise ValueError(
                f"No Hugging Face repo known for construct '{construct}'. "
                f"Known constructs: {list(self.hf_repo_map.keys())}. "
                "Pass model_dir / --model-dir to use a local model instead, "
                "or extend hf_repo_map / HF_MODEL_REPOS."
            )
        print(f"Downloading model for '{construct}' from Hugging Face ({repo_id}) ...")
        local_dir = snapshot_download(repo_id=repo_id)
        return Path(local_dir)

    def _load_model(self, construct: str):
        """Load trained model for a construct, from model_dir if given,
        otherwise downloaded from Hugging Face."""
        if construct in self.models:
            return self.models[construct]

        if not SL_AVAILABLE:
            raise ImportError(
                "semantic_learning module required. "
                "Ensure semantic_learning.py is on your Python path."
            )

        if self.model_dir is not None:
            base_dir = self.model_dir / construct
        else:
            base_dir = self._download_from_huggingface(construct)

        model_path = base_dir / f'{self.model_type}_model.pkl'
        model_type = self.model_type

        if not model_path.exists():
            fallback_type = 'prototype' if self.model_type == 'contrastive' else 'contrastive'
            fallback_path = base_dir / f'{fallback_type}_model.pkl'
            if fallback_path.exists():
                print(f"  {self.model_type} model not found for {construct}, falling back to {fallback_type}")
                model_path = fallback_path
                model_type = fallback_type
            else:
                source = str(base_dir) if self.model_dir is not None else self.hf_repo_map.get(construct, '?')
                raise FileNotFoundError(
                    f"Model not found for construct '{construct}' at {source}.\n"
                    f"Expected: <root>/<construct>/{self.model_type}_model.pkl (local) "
                    f"or {self.model_type}_model.pkl at the repo root (Hugging Face)."
                )

        if model_type == 'contrastive':
            self.models[construct] = ContrastiveModel.load(str(model_path), self.device)
        elif model_type == 'prototype':
            self.models[construct] = PrototypeModel.load(str(model_path), self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Loaded {self.model_type} model for {construct}")
        
        # Override knn_alpha if specified (otherwise use model's stored value)
        if self.knn_alpha is not None and hasattr(self.models[construct], 'knn_alpha'):
            stored_alpha = getattr(self.models[construct], 'knn_alpha', None)
            self.models[construct].knn_alpha = self.knn_alpha
            if stored_alpha != self.knn_alpha:
                print(f"  k-NN alpha overridden: {stored_alpha} → {self.knn_alpha}")

        # Override class_balance if specified (otherwise use model's stored value)
        if self.class_balance is not None and hasattr(self.models[construct], 'class_balance'):
            stored_cb = getattr(self.models[construct], 'class_balance', False)
            self.models[construct].class_balance = self.class_balance
            if stored_cb != self.class_balance:
                print(f"  Class balance overridden: {stored_cb} → {self.class_balance}")

        return self.models[construct]
    
    def harmonize(
        self,
        items_df: pd.DataFrame,
        construct: str,
        item_id_col: str = 'item_id',
        item_text_col: str = 'item_text',
        questionnaire_col: str = 'questionnaire'
    ) -> List[HarmonizationResult]:
        """
        Harmonize questionnaire items to dimensions.
        
        Parameters
        ----------
        items_df : pd.DataFrame
            DataFrame with item information
        construct : str
            Construct name (e.g., 'depression', 'anxiety')
        item_id_col : str
            Column name for item IDs
        item_text_col : str
            Column name for item text
        questionnaire_col : str
            Column name for questionnaire name
        
        Returns
        -------
        List[HarmonizationResult]
            Results for each item
        """
        # Filter to items belonging to this construct if a construct column exists
        if 'construct' in items_df.columns:
            items_df = items_df[items_df['construct'] == construct].copy()
            print(f"Filtered to {len(items_df)} items for construct '{construct}'")

        results = []
        new_items = []
        new_item_indices = []

        # First pass: check inventory for known items
        for idx, row in items_df.iterrows():
            item_id = row[item_id_col]
            item_text = row[item_text_col]
            questionnaire = row.get(questionnaire_col, 'Unknown')
            
            # Check inventory
            lookup_result = self.inventory.lookup(item_id, construct)
            
            if lookup_result is not None:
                # Known item - return stored assignment
                results.append(HarmonizationResult(
                    item_id=item_id,
                    item_text=item_text,
                    construct=construct,
                    questionnaire=questionnaire,
                    dimension=int(lookup_result['dimension']),
                    dimension_label=lookup_result['dimension_label'],
                    confidence=float(lookup_result['confidence']),
                    confidence_type=lookup_result.get('confidence_type', 'expert_agreement'),
                    source='lookup_table',
                    date_harmonized=datetime.now().isoformat()
                ))
            else:
                # New item - need to predict
                new_items.append({
                    'idx': idx,
                    'item_id': item_id,
                    'item_text': item_text,
                    'questionnaire': questionnaire
                })
                new_item_indices.append(len(results))
                results.append(None)  # Placeholder
        
        # Second pass: predict for new items
        if new_items:
            print(f"Predicting {len(new_items)} new items...")
            
            # Load model
            model = self._load_model(construct)
            
            # Predict using raw text (model handles encoding internally)
            item_texts = [item['item_text'] for item in new_items]
            predictions = model.predict(item_texts)
            probabilities = model.predict_proba(item_texts)
            
            # Get dimension labels
            dim_labels = DIMENSION_LABELS.get(construct, {})
            
            # Fill in results
            for i, item in enumerate(new_items):
                pred_dim = int(predictions[i])
                pred_label = dim_labels.get(pred_dim, f"Dimension {pred_dim}")
                
                # Confidence from probability distribution
                dim_idx = model.dimensions.index(pred_dim) if pred_dim in model.dimensions else 0
                confidence = float(probabilities[i, dim_idx])
                
                # Create probability distribution dict
                prob_dist = {
                    int(dim): float(probabilities[i, j])
                    for j, dim in enumerate(model.dimensions)
                }
                
                result = HarmonizationResult(
                    item_id=item['item_id'],
                    item_text=item['item_text'],
                    construct=construct,
                    questionnaire=item['questionnaire'],
                    dimension=int(pred_dim),
                    dimension_label=pred_label,
                    confidence=confidence,
                    confidence_type='model_confidence',
                    source='model_prediction',
                    model_version=self.model_version,
                    date_harmonized=datetime.now().isoformat(),
                    probability_distribution=prob_dist
                )
                
                results[new_item_indices[i]] = result
                
                # Auto-update inventory
                if self.auto_update:
                    self.inventory.add_item(
                        item_id=item['item_id'],
                        item_text=item['item_text'],
                        construct=construct,
                        questionnaire=item['questionnaire'],
                        dimension=int(pred_dim),
                        dimension_label=pred_label,
                        confidence=confidence,
                        source='model_prediction',
                        confidence_type='model_confidence',
                        model_version=self.model_version
                    )
            
            # Save updated inventory
            if self.auto_update:
                self.inventory.save()
                print(f"Inventory updated with {len(new_items)} new items")
        
        return results
    
    def harmonize_single(
        self,
        item_id: str,
        item_text: str,
        construct: str,
        questionnaire: str = 'Unknown'
    ) -> HarmonizationResult:
        """Harmonize a single item."""
        df = pd.DataFrame([{
            'item_id': item_id,
            'item_text': item_text,
            'questionnaire': questionnaire
        }])
        
        results = self.harmonize(df, construct)
        return results[0]
    
    def get_inventory_summary(self) -> pd.DataFrame:
        """Get summary of current inventory."""
        return self.inventory.summary()
    
    def export_results(
        self,
        results: List[HarmonizationResult],
        output_path: str
    ):
        """Export harmonization results to CSV."""
        rows = [asdict(r) for r in results]
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")


# =============================================================================
# Utility Functions
# =============================================================================

def _survey_majority_dim(rating_dist: str):
    """Return the dimension with the highest proportion in a rating distribution string."""
    dims = {}
    for part in str(rating_dist).split(';'):
        part = part.strip()
        if ':' in part:
            try:
                dim_part, pct_part = part.split(':')
                dim = int(dim_part.replace('Dim', '').strip())
                pct = float(pct_part.replace('%', '').strip())
                dims[dim] = pct
            except (ValueError, AttributeError):
                continue
    return max(dims, key=dims.get) if dims else None


def create_initial_inventory(
    assignments_path: str,
    item_assignments_path: str,
    output_path: str
) -> HarmonizedInventory:
    """
    Create initial inventory from expert assignments.
    
    Parameters
    ----------
    assignments_path : str
        Path to final_assignments_corrected.csv
    item_assignments_path : str
        Path to item_assignments.csv (with rating distributions)
    output_path : str
        Path for output inventory CSV
    
    Returns
    -------
    HarmonizedInventory
        Initialized inventory
    """
    # Load data
    df_final = pd.read_csv(assignments_path)
    df_expert = pd.read_csv(item_assignments_path)
    
    # Create inventory
    inventory = HarmonizedInventory()
    
    # Add each item
    for _, row in df_final.iterrows():
        item_id = row['Item ID']
        construct = row['Construct']
        
        # Get expert info
        expert_row = df_expert[
            (df_expert['Item ID'] == item_id) & 
            (df_expert['Construct'] == construct)
        ]
        
        if len(expert_row) > 0:
            expert_row = expert_row.iloc[0]
            p_i = expert_row.get('P_i', 0)
            rating_dist = expert_row.get('Rating Distribution', '')
        else:
            p_i = row.get('Expert Confidence (P_i)', 0)
            rating_dist = ''

        # Determine source
        was_corrected = row.get('Was Corrected', False)
        source = 'expert_reviewed' if was_corrected else 'expert_consensus'

        # For corrected items where the survey majority voted for a different dimension
        # than the final expert-corrected one, the original P_i reflects the wrong
        # dimension. Override confidence to 1.0 so the correction is authoritative.
        final_dim = int(row['Final Dimension'])
        if was_corrected and rating_dist:
            survey_top = _survey_majority_dim(rating_dist)
            if survey_top is not None and survey_top != final_dim:
                p_i = 1.0

        inventory.add_item(
            item_id=item_id,
            item_text=row.get('Question Text', ''),
            construct=construct,
            questionnaire=row.get('Questionnaire', ''),
            dimension=final_dim,
            dimension_label=row.get('Final Dimension Label', ''),
            confidence=float(p_i) if not pd.isna(p_i) else 0.0,
            source=source,
            confidence_type='expert_agreement',
            expert_agreement=float(p_i) if not pd.isna(p_i) else None,
            rating_distribution=rating_dist
        )
    
    # Save
    inventory.save(output_path)
    print(f"Created inventory with {len(inventory)} items at {output_path}")
    
    return inventory


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Harmonize questionnaire items")
    subparsers = parser.add_subparsers(dest='command')
    
    # Create inventory command
    create_parser = subparsers.add_parser('create-inventory', help='Create initial inventory')
    create_parser.add_argument('--assignments', required=True, help='Path to final_assignments_corrected.csv')
    create_parser.add_argument('--item-assignments', required=True, help='Path to item_assignments.csv')
    create_parser.add_argument('--output', required=True, help='Output inventory path')
    
    # Harmonize command
    harm_parser = subparsers.add_parser('harmonize', help='Harmonize items')
    harm_parser.add_argument('--items', required=True, help='Path to items CSV')
    harm_parser.add_argument('--construct', required=True, help='Construct name')
    harm_parser.add_argument('--inventory', required=True, help='Path to inventory')
    harm_parser.add_argument('--model-dir', default=None,
                            help='Local directory containing trained models '
                                 '(<model-dir>/<construct>/<model-type>_model.pkl). '
                                 'If omitted, the model is downloaded and cached from '
                                 'Hugging Face instead (see --hf-repo / HF_MODEL_REPOS).')
    harm_parser.add_argument('--hf-repo', type=str, default=None,
                            help='Hugging Face repo id to download the model from '
                                 '(default: derived from --construct via the built-in '
                                 'HarmoniCA collection mapping, HF_MODEL_REPOS). '
                                 'Only used when --model-dir is not given.')
    harm_parser.add_argument('--output', required=True, help='Output path')
    harm_parser.add_argument('--model-type', default='contrastive', choices=['contrastive', 'prototype'])
    harm_parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    harm_parser.add_argument('--no-auto-update', action='store_true')
    harm_parser.add_argument('--knn-alpha', type=float, default=None,
                            help='Override k-NN mixing weight (1.0=pure prototype, 0.0=pure k-NN, '
                                 'default: use value stored in model)')
    harm_parser.add_argument('--class-balance', action='store_true', default=None,
                            help='Divide dimension scores by their training frequency before '
                                 'renormalizing, compensating for unequal class sizes. '
                                 'Recomputed from the current inventory at prediction time. '
                                 'Default: use value stored in model.')
    harm_parser.add_argument('--dimension-config', type=str, default=None,
                            help='Path to a dimension_descriptions.yaml defining the '
                                 f'constructs/dimensions to harmonize (default: {DEFAULT_DIMENSION_CONFIG}). '
                                 'Point this at your own copy to run on different constructs '
                                 'without editing this script.')

    args = parser.parse_args()

    if getattr(args, 'dimension_config', None):
        DIMENSION_LABELS = load_dimension_labels(args.dimension_config)

    if args.command == 'create-inventory':
        create_initial_inventory(
            args.assignments,
            args.item_assignments,
            args.output
        )
    
    elif args.command == 'harmonize':
        hf_repo_map = None
        if args.hf_repo:
            hf_repo_map = dict(HF_MODEL_REPOS)
            hf_repo_map[args.construct] = args.hf_repo

        harmonizer = Harmonizer(
            inventory_path=args.inventory,
            model_dir=args.model_dir,
            hf_repo_map=hf_repo_map,
            auto_update=not args.no_auto_update,
            model_type=args.model_type,
            device=args.device,
            knn_alpha=args.knn_alpha,
            class_balance=args.class_balance if args.class_balance else None,
        )
        
        items_df = pd.read_csv(args.items)
        results = harmonizer.harmonize(items_df, args.construct)
        harmonizer.export_results(results, args.output)