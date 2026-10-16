"""
Semantic Similarity Analysis and Integration of Mappings 
=========================================================================

ENIGMA-PD Questionnaire Harmonization - Steps 4 & 5

Step 4: Semantic Similarity Analysis
- Compute embeddings for questionnaire items using multiple sentence transformer models
- Compute embeddings for dimension descriptions (the same information given to experts)
- Assign items to dimensions based on maximum cosine similarity to dimension descriptions
- This directly parallels the expert task: items are assigned to the dimension whose
  description they most closely match semantically

Step 5: Integration of Mappings
- Calculate agreement metrics (Cohen's kappa, dimension-level accuracy) for model selection
- Select best-performing embedding model based on agreement with expert assignments
- Compute a simple model confidence (top1-top2 similarity margin) for reporting purposes
- Generate final item-to-dimension assignments based on EXPERT decisions

Final Assignment Framework:
---------------------------
1. FINAL DIMENSION ASSIGNMENT
   - Always use expert assignment (including "Does not fit" = -1)
   - Expert decision is never overwritten by model

2. MODEL SCORES (for reporting only)
   - Model dimension: argmax(similarity_k)
   - Model confidence: margin between top1 and top2 similarity scores

3. EXPERT CONFIDENCE
   - P_i = Σ(E_ik²) - probability that two random experts agree

4. CORRECTIONS WORKFLOW
   - Supports loading corrected assignments for disagreement items
   - Creates final_assignments_corrected.csv for fine-tuning

"""

import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import json

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent.parent
DEFAULT_DIMENSION_CONFIG = PROJECT_DIR / "input" / "shared" / "dimension_descriptions.yaml"

# Optional imports - will check availability
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


# =============================================================================
# Configuration: Embedding Models to Test
# =============================================================================

# Models from the registered report 
EMBEDDING_MODELS = {
    # High-performance general models
    'bge-large': 'BAAI/bge-large-en-v1.5',
    'e5-large': 'intfloat/e5-large-v2',
    'instructor': 'hkunlp/instructor-large',
    'gtr-t5': 'sentence-transformers/gtr-t5-large',
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',
    'minilm': 'sentence-transformers/all-MiniLM-L12-v2',
    'simcse': 'princeton-nlp/sup-simcse-roberta-large',
    'sbert-similarity': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
    
    # Clinical/biomedical models
    'clinicalbert': 'medicalai/ClinicalBERT',
    'bio-clinicalbert': 'emilyalsentzer/Bio_ClinicalBERT',
    'biobert': 'dmis-lab/biobert-v1.1',
    'clinicalSentenceTransformer': 'Shobhank-iiitdwd/Clinical_sentence_transformers_mpnet_base_v2',
    
    # Mental health specific models
    'mentalroberta': 'mental/mental-roberta-base',
    'harmony': 'harmonydata/mental_health_harmonisation_1',
}

# Lighter subset for faster testing
EMBEDDING_MODELS_LITE = {
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',
    'minilm': 'sentence-transformers/all-MiniLM-L12-v2',
    'sbert-similarity': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
}


# =============================================================================
# Dimension Descriptions (same as provided to experts in the survey)
# =============================================================================

def load_dimension_config(path: Union[str, Path] = DEFAULT_DIMENSION_CONFIG) -> Tuple[Dict, Dict]:
    """
    Load dimension descriptions and labels from a YAML config file.

    dimension_descriptions provides the "anchor point" text (label +
    description) embedded and compared against item embeddings in Step 4
    (theory-primed assignment). dimension_labels provides the labels used
    for Step 5 outputs, including the -1 "Does not fit" catch-all every
    construct has.

    To adapt this pipeline to your own constructs, edit a copy of
    input/shared/dimension_descriptions.yaml (see that file for
    the expected format) and pass its path here / via --dimension-config --
    no need to touch this script.

    Returns
    -------
    dimension_descriptions : Dict[str, Dict[int, Dict]]
        construct -> dimension_id -> {label, description, keywords}
    dimension_labels : Dict[str, Dict[int, str]]
        construct -> dimension_id -> label
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
    return config['dimension_descriptions'], config['dimension_labels']


DIMENSION_DESCRIPTIONS, DEFAULT_DIMENSION_LABELS = load_dimension_config()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EmbeddingResult:
    """Results from embedding a set of items."""
    model_name: str
    model_path: str
    embeddings: np.ndarray  # Shape: (n_items, embedding_dim)
    item_ids: List[str]
    construct: str
    embedding_dim: int


@dataclass
class DimensionEmbeddingResult:
    """Results from embedding dimension descriptions."""
    model_name: str
    construct: str
    dimension_embeddings: np.ndarray  # Shape: (n_dimensions, embedding_dim)
    dimension_ids: List[int]
    dimension_labels: List[str]


@dataclass
class SimilarityResult:
    """Similarity matrix between items and dimension descriptions."""
    model_name: str
    construct: str
    item_to_dimension_similarity: np.ndarray  # Shape: (n_items, n_dimensions)
    item_ids: List[str]
    dimension_ids: List[int]
    dimension_labels: List[str]


@dataclass
class TheoryPrimedAssignment:
    """Model-based assignment using theory-primed similarity."""
    model_name: str
    construct: str
    item_ids: List[str]
    assigned_dimensions: np.ndarray  # Dimension ID for each item
    similarity_scores: np.ndarray  # Max similarity score for each item
    all_similarities: np.ndarray  # Full similarity matrix (items x dimensions)
    dimension_ids: List[int]
    dimension_labels: List[str]


@dataclass
class ModelAgreementMetrics:
    """Agreement metrics between expert and model-based assignments."""
    model_name: str
    construct: str
    cohens_kappa: float
    accuracy: float
    dimension_accuracies: Dict[int, float]  # Per-dimension accuracy
    n_items: int
    n_correct: int
    expert_assignments: List[int]
    model_assignments: List[int]
    confusion_matrix: Optional[np.ndarray] = None


@dataclass
class FinalAssignment:
    """Final item-to-dimension assignment (expert decision is always authoritative)."""
    item_id: str
    construct: str
    questionnaire: str
    question_text: str
    # Dimension assignments
    expert_dimension: int  # argmax(E_ik)
    expert_dimension_label: str
    model_dimension: int  # argmax(similarity)
    model_dimension_label: str
    final_dimension: int  # Final assigned dimension (always == expert_dimension)
    final_dimension_label: str
    # Confidence scores
    expert_confidence: float  # P_i = Σ(E_ik²)
    model_confidence: float  # margin between top1 and top2 similarity scores
    # Agreement
    expert_model_agree: bool
    # Raw data for reference
    expert_proportions: Dict[int, float]  # E_ik for all dimensions
    raw_similarities: Dict[int, float]  # Raw cosine similarities


@dataclass
class IntegrationResults:
    """Complete results from the integration step."""
    construct: str
    best_model: str
    model_metrics: Dict[str, ModelAgreementMetrics]
    final_assignments: List[FinalAssignment]
    n_disagree: int


# =============================================================================
# Step 4: Embedding and Theory-Primed Assignment Functions
# =============================================================================

def load_items_for_embedding(filepath: Union[str, Path], 
                             construct: Optional[str] = None) -> pd.DataFrame:
    """
    Load questionnaire items for embedding.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the items CSV file
    construct : str, optional
        Filter to specific construct
    
    Returns
    -------
    pd.DataFrame
        DataFrame with item information
    """
    df = pd.read_csv(filepath)
    
    if construct:
        df = df[df['construct'] == construct].copy()
    
    return df


def compute_embeddings(texts: List[str], 
                      model_name: str,
                      model_path: str,
                      instruction: Optional[str] = None) -> np.ndarray:
    """
    Compute embeddings for a list of texts using a sentence transformer model.
    
    Parameters
    ----------
    texts : List[str]
        List of text strings to embed
    model_name : str
        Short name of the model
    model_path : str
        HuggingFace model path
    instruction : str, optional
        Instruction prefix for instructor-style models
    
    Returns
    -------
    np.ndarray
        Embeddings matrix of shape (n_texts, embedding_dim)
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
    
    print(f"  Loading model: {model_name} ({model_path})")
    
    try:
        model = SentenceTransformer(model_path)
    except Exception as e:
        warnings.warn(f"Failed to load model {model_name}: {e}")
        return None
    
    # Handle instructor-style models
    if 'instructor' in model_name.lower() and instruction:
        texts = [[instruction, text] for text in texts]
    
    print(f"  Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    return embeddings


def get_dimension_descriptions_for_embedding(construct: str) -> Tuple[List[int], List[str], List[str]]:
    """
    Get dimension descriptions formatted for embedding.
    
    Parameters
    ----------
    construct : str
        Name of the construct
    
    Returns
    -------
    dimension_ids : List[int]
        Numeric dimension IDs
    dimension_labels : List[str]
        Human-readable labels
    dimension_texts : List[str]
        Full text descriptions for embedding
    """
    if construct not in DIMENSION_DESCRIPTIONS:
        raise ValueError(f"Unknown construct: {construct}. Available: {list(DIMENSION_DESCRIPTIONS.keys())}")
    
    dims = DIMENSION_DESCRIPTIONS[construct]
    
    dimension_ids = []
    dimension_labels = []
    dimension_texts = []
    
    for dim_id in sorted(dims.keys()):
        dim_info = dims[dim_id]
        dimension_ids.append(dim_id)
        dimension_labels.append(dim_info['label'])
        # Combine label and description for richer embedding
        dimension_texts.append(f"{dim_info['label']}: {dim_info['description']}")
    
    return dimension_ids, dimension_labels, dimension_texts


def compute_theory_primed_assignment(
    item_embeddings: np.ndarray,
    dimension_embeddings: np.ndarray,
    item_ids: List[str],
    dimension_ids: List[int],
    dimension_labels: List[str],
    model_name: str,
    construct: str
) -> TheoryPrimedAssignment:
    """
    Assign items to dimensions based on similarity to dimension descriptions.
    
    This is the core of the theory-primed approach: each item is assigned to
    the dimension whose description embedding it is most similar to.
    
    Parameters
    ----------
    item_embeddings : np.ndarray
        Embeddings for items (n_items, embedding_dim)
    dimension_embeddings : np.ndarray
        Embeddings for dimension descriptions (n_dimensions, embedding_dim)
    item_ids : List[str]
        Item identifiers
    dimension_ids : List[int]
        Dimension numeric IDs
    dimension_labels : List[str]
        Dimension labels
    model_name : str
        Name of the embedding model
    construct : str
        Construct name
    
    Returns
    -------
    TheoryPrimedAssignment
        Assignment results including all similarity scores
    """
    # Normalize embeddings for cosine similarity
    item_emb_norm = normalize(item_embeddings, axis=1)
    dim_emb_norm = normalize(dimension_embeddings, axis=1)
    
    # Compute similarity matrix (items x dimensions)
    similarity_matrix = cosine_similarity(item_emb_norm, dim_emb_norm)
    
    # Assign each item to highest-similarity dimension
    best_dim_indices = np.argmax(similarity_matrix, axis=1)
    assigned_dimensions = np.array([dimension_ids[i] for i in best_dim_indices])
    similarity_scores = np.max(similarity_matrix, axis=1)
    
    return TheoryPrimedAssignment(
        model_name=model_name,
        construct=construct,
        item_ids=item_ids,
        assigned_dimensions=assigned_dimensions,
        similarity_scores=similarity_scores,
        all_similarities=similarity_matrix,
        dimension_ids=dimension_ids,
        dimension_labels=dimension_labels
    )


def embed_and_assign_construct(
    df: pd.DataFrame,
    construct: str,
    models: Dict[str, str]
) -> Dict[str, Dict]:
    """
    Compute embeddings and theory-primed assignments for a construct using multiple models.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with items (filtered to construct)
    construct : str
        Name of the construct
    models : Dict[str, str]
        Dictionary mapping model names to HuggingFace paths
    
    Returns
    -------
    Dict[str, Dict]
        Results for each model containing embeddings, similarities, and assignments
    """
    df_construct = df[df['construct'] == construct].copy()
    # question_text is stripped from the public items file (copyright); embedding
    # empty strings makes the resulting similarities meaningless but lets the
    # pipeline run end-to-end on data that has real item wording.
    if 'question_text' in df_construct.columns:
        texts = df_construct['question_text'].fillna('').tolist()
    else:
        texts = [''] * len(df_construct)
    item_ids = df_construct['question_number'].tolist()
    
    # Get dimension descriptions
    dimension_ids, dimension_labels, dimension_texts = get_dimension_descriptions_for_embedding(construct)
    
    print(f"\n  Construct: {construct}")
    print(f"  Items: {len(texts)}")
    print(f"  Dimensions: {len(dimension_ids)}")
    for d_id, d_label in zip(dimension_ids, dimension_labels):
        print(f"    {d_id}: {d_label}")
    
    results = {}
    
    for model_name, model_path in models.items():
        print(f"\n  Processing model: {model_name}")
        
        try:
            # Set instruction for instructor-style models
            instruction = None
            if 'instructor' in model_name.lower():
                instruction = "Represent this mental health questionnaire item for semantic similarity:"
            
            # Compute item embeddings
            item_embeddings = compute_embeddings(texts, model_name, model_path, instruction)
            
            if item_embeddings is None:
                continue
            
            # Compute dimension description embeddings
            dim_instruction = None
            if 'instructor' in model_name.lower():
                dim_instruction = "Represent this mental health symptom dimension description:"
            
            # Need to reload model for dimension embeddings (or reuse)
            print(f"  Computing dimension description embeddings...")
            dimension_embeddings = compute_embeddings(dimension_texts, model_name, model_path, dim_instruction)
            
            if dimension_embeddings is None:
                continue
            
            # Compute theory-primed assignment
            assignment = compute_theory_primed_assignment(
                item_embeddings, dimension_embeddings,
                item_ids, dimension_ids, dimension_labels,
                model_name, construct
            )
            
            # Store results
            results[model_name] = {
                'item_embeddings': EmbeddingResult(
                    model_name=model_name,
                    model_path=model_path,
                    embeddings=item_embeddings,
                    item_ids=item_ids,
                    construct=construct,
                    embedding_dim=item_embeddings.shape[1]
                ),
                'dimension_embeddings': DimensionEmbeddingResult(
                    model_name=model_name,
                    construct=construct,
                    dimension_embeddings=dimension_embeddings,
                    dimension_ids=dimension_ids,
                    dimension_labels=dimension_labels
                ),
                'similarity': SimilarityResult(
                    model_name=model_name,
                    construct=construct,
                    item_to_dimension_similarity=assignment.all_similarities,
                    item_ids=item_ids,
                    dimension_ids=dimension_ids,
                    dimension_labels=dimension_labels
                ),
                'assignment': assignment
            }
            
            # Print summary
            print(f"  Assignments per dimension:")
            for d_id, d_label in zip(dimension_ids, dimension_labels):
                n_assigned = np.sum(assignment.assigned_dimensions == d_id)
                print(f"    {d_label}: {n_assigned} items")
            
        except Exception as e:
            warnings.warn(f"Error processing model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


# =============================================================================
# Step 5: Agreement Metrics and Model Selection
# =============================================================================

def get_expert_assignments(item_assignments_df: pd.DataFrame, 
                          construct: str) -> Dict[str, int]:
    """
    Get expert-assigned dimensions from the expert mappings output.
    
    Parameters
    ----------
    item_assignments_df : pd.DataFrame
        DataFrame from item_assignments.csv (output of expert_mappings_analysis.py)
    construct : str
        Name of the construct
    
    Returns
    -------
    Dict[str, int]
        Mapping from item_id to assigned dimension
    """
    df_construct = item_assignments_df[item_assignments_df['Construct'] == construct]
    return dict(zip(df_construct['Item ID'], df_construct['Assigned Dimension']))


def evaluate_model_agreement(
    assignment: TheoryPrimedAssignment,
    expert_assignments: Dict[str, int]
) -> ModelAgreementMetrics:
    """
    Evaluate agreement between model-based and expert assignments.
    
    Parameters
    ----------
    assignment : TheoryPrimedAssignment
        Theory-primed assignment results
    expert_assignments : Dict[str, int]
        Expert-assigned dimensions
    
    Returns
    -------
    ModelAgreementMetrics
        Agreement metrics for this model
    """
    expert_labels = []
    model_labels = []
    
    for i, item_id in enumerate(assignment.item_ids):
        if item_id in expert_assignments:
            expert_labels.append(expert_assignments[item_id])
            model_labels.append(assignment.assigned_dimensions[i])
    
    if len(expert_labels) < 2:
        return ModelAgreementMetrics(
            model_name=assignment.model_name,
            construct=assignment.construct,
            cohens_kappa=np.nan,
            accuracy=np.nan,
            dimension_accuracies={},
            n_items=0,
            n_correct=0,
            expert_assignments=[],
            model_assignments=[]
        )
    
    # Calculate Cohen's kappa
    kappa = cohen_kappa_score(expert_labels, model_labels)
    
    # Calculate accuracy
    accuracy = accuracy_score(expert_labels, model_labels)
    n_correct = sum(e == m for e, m in zip(expert_labels, model_labels))
    
    # Calculate per-dimension accuracy
    dimension_accuracies = {}
    for dim in assignment.dimension_ids:
        dim_mask = [e == dim for e in expert_labels]
        if sum(dim_mask) > 0:
            dim_correct = sum(e == m for e, m, mask in zip(expert_labels, model_labels, dim_mask) if mask)
            dimension_accuracies[dim] = dim_correct / sum(dim_mask)
        else:
            dimension_accuracies[dim] = np.nan
    
    return ModelAgreementMetrics(
        model_name=assignment.model_name,
        construct=assignment.construct,
        cohens_kappa=kappa,
        accuracy=accuracy,
        dimension_accuracies=dimension_accuracies,
        n_items=len(expert_labels),
        n_correct=n_correct,
        expert_assignments=expert_labels,
        model_assignments=model_labels
    )


def select_best_model(
    model_metrics: Dict[str, ModelAgreementMetrics],
    min_kappa_threshold: float = 0.40,
    kappa_tolerance: float = 0.05
) -> Tuple[str, ModelAgreementMetrics]:
    """
    Select the best-performing embedding model based on agreement metrics.
    
    Selection criteria (from registered report):
    1. Primary: Highest Cohen's kappa (minimum threshold κ ≥ 0.40)
    2. Secondary: If multiple models within 0.05 of highest κ, select highest accuracy
    
    Parameters
    ----------
    model_metrics : Dict[str, ModelAgreementMetrics]
        Agreement metrics for each model
    min_kappa_threshold : float
        Minimum acceptable kappa (default: 0.40)
    kappa_tolerance : float
        Tolerance for considering models equivalent (default: 0.05)
    
    Returns
    -------
    best_model : str
        Name of the best model
    best_metrics : ModelAgreementMetrics
        Metrics for the best model
    """
    # Filter models meeting minimum threshold
    valid_models = {
        name: metrics for name, metrics in model_metrics.items()
        if not np.isnan(metrics.cohens_kappa) and metrics.cohens_kappa >= min_kappa_threshold
    }
    
    if not valid_models:
        warnings.warn(f"No models achieved κ ≥ {min_kappa_threshold}. "
                     "Proceeding with best available model.")
        # Return the model with highest kappa anyway
        best_name = max(model_metrics, key=lambda x: model_metrics[x].cohens_kappa 
                       if not np.isnan(model_metrics[x].cohens_kappa) else -1)
        return best_name, model_metrics[best_name]
    
    # Find highest kappa
    max_kappa = max(m.cohens_kappa for m in valid_models.values())
    
    # Find models within tolerance of highest
    top_models = {
        name: metrics for name, metrics in valid_models.items()
        if max_kappa - metrics.cohens_kappa <= kappa_tolerance
    }
    
    # If tie, use accuracy as tiebreaker
    if len(top_models) > 1:
        best_name = max(top_models, key=lambda x: top_models[x].accuracy)
    else:
        best_name = list(top_models.keys())[0]
    
    return best_name, valid_models[best_name]


# =============================================================================
# Step 5: Final Assignment
# =============================================================================

def compute_expert_confidence(rating_distribution: str) -> Tuple[float, Dict[int, float]]:
    """
    Compute expert confidence P_i from rating distribution.
    
    P_i = Σ(E_ik²) where E_ik is the proportion of experts who chose dimension k.
    This is the probability that two randomly selected experts agree.
    
    Parameters
    ----------
    rating_distribution : str
        Rating distribution string, e.g., "Dim 1: 70%; Dim 2: 30%"
    
    Returns
    -------
    Tuple[float, Dict[int, float]]
        - P_i (expert confidence/concentration)
        - E_ik dictionary (proportions for each dimension)
    """
    E_ik = {}
    
    if pd.isna(rating_distribution) or not rating_distribution:
        return 0.0, E_ik
    
    for part in str(rating_distribution).split(';'):
        part = part.strip()
        if ':' in part:
            try:
                dim_part, pct_part = part.split(':')
                dim = int(dim_part.replace('Dim', '').strip())
                pct = float(pct_part.replace('%', '').strip()) / 100
                E_ik[dim] = pct
            except (ValueError, AttributeError):
                continue
    
    # P_i = sum of squared proportions
    P_i = sum(p**2 for p in E_ik.values())
    
    return P_i, E_ik


def generate_final_assignments(
    df_items: pd.DataFrame,
    item_assignments_df: pd.DataFrame,
    assignment: TheoryPrimedAssignment,
    construct: str,
    dimension_labels: Dict[int, str]
) -> List[FinalAssignment]:
    """
    Generate final item-to-dimension assignments.

    Framework:
    - Expert: argmax(E_ik)
    - Model: argmax(similarity_k), reported alongside for comparison
    - Final: Always the expert assignment (experts take precedence, including
      "Does not fit" = -1); the model decision never overwrites it.

    Parameters
    ----------
    df_items : pd.DataFrame
        Original items dataframe
    item_assignments_df : pd.DataFrame
        Expert assignments from step 3
    assignment : TheoryPrimedAssignment
        Best model's assignment results
    construct : str
        Construct name
    dimension_labels : Dict[int, str]
        Dimension labels

    Returns
    -------
    List[FinalAssignment]
        Final assignments for all items
    """
    df_construct = df_items[df_items['construct'] == construct]
    df_expert = item_assignments_df[item_assignments_df['Construct'] == construct]

    # Build lookup dictionaries from expert data
    expert_dimensions = dict(zip(df_expert['Item ID'], df_expert['Assigned Dimension']))
    expert_p_i = dict(zip(df_expert['Item ID'], df_expert['P_i']))
    expert_distributions = {}

    if 'Rating Distribution' in df_expert.columns:
        for _, row in df_expert.iterrows():
            _, E_ik = compute_expert_confidence(row.get('Rating Distribution', ''))
            expert_distributions[row['Item ID']] = E_ik

    final_assignments = []

    for _, row in df_construct.iterrows():
        item_id = row['question_number']

        # Find item index in assignment
        try:
            item_idx = assignment.item_ids.index(item_id)
        except ValueError:
            continue

        # Get raw similarities
        similarities = assignment.all_similarities[item_idx]
        raw_sims = {dim_id: float(similarities[i])
                    for i, dim_id in enumerate(assignment.dimension_ids)}

        # --- EXPERT SIDE ---
        expert_dim = expert_dimensions.get(item_id, -1)
        expert_dim_label = dimension_labels.get(expert_dim, "Unknown")
        expert_conf = expert_p_i.get(item_id, 0.0)
        if pd.isna(expert_conf):
            expert_conf = 0.0
        expert_props = expert_distributions.get(item_id, {})

        # --- MODEL SIDE ---
        # Dimension assignment: argmax(similarity)
        model_dim_idx = np.argmax(similarities)
        model_dim = assignment.dimension_ids[model_dim_idx]
        model_dim_label = dimension_labels.get(model_dim, "Unknown")

        # Confidence: margin between top1 and top2 similarity scores
        sorted_sims = np.sort(similarities)[::-1]
        model_conf = float(sorted_sims[0] - sorted_sims[1]) if len(sorted_sims) > 1 else float(sorted_sims[0])

        # --- FINAL ASSIGNMENT ---
        # ALWAYS use expert assignment, including "Does not fit" (-1)
        # Expert decision is never overwritten by model
        final_dim = expert_dim
        final_dim_label = dimension_labels.get(final_dim, "Does not fit" if final_dim == -1 else "Unknown")

        # Agreement check (only meaningful when expert_dim != -1)
        expert_model_agree = (expert_dim == model_dim) if expert_dim != -1 else False

        final_assignments.append(FinalAssignment(
            item_id=item_id,
            construct=construct,
            questionnaire=row['questionnaire_name'],
            question_text=row.get('question_text', ''),  # stripped from public item files (copyright)
            expert_dimension=expert_dim,
            expert_dimension_label=expert_dim_label,
            model_dimension=model_dim,
            model_dimension_label=model_dim_label,
            final_dimension=final_dim,
            final_dimension_label=final_dim_label,
            expert_confidence=float(expert_conf),
            model_confidence=model_conf,
            expert_model_agree=expert_model_agree,
            expert_proportions=expert_props,
            raw_similarities=raw_sims
        ))

    return final_assignments


# =============================================================================
# Output Functions
# =============================================================================

def create_model_comparison_table(all_metrics: Dict[str, Dict[str, ModelAgreementMetrics]]) -> pd.DataFrame:
    """Create a table comparing all models across constructs."""
    rows = []
    
    for construct, model_metrics in all_metrics.items():
        for model_name, metrics in model_metrics.items():
            rows.append({
                'Construct': construct,
                'Model': model_name,
                "Cohen's κ": round(metrics.cohens_kappa, 3) if not np.isnan(metrics.cohens_kappa) else 'N/A',
                'Accuracy': round(metrics.accuracy, 3) if not np.isnan(metrics.accuracy) else 'N/A',
                'N Items': metrics.n_items,
                'N Correct': metrics.n_correct
            })
    
    return pd.DataFrame(rows)


def create_dimension_accuracy_table(
    all_metrics: Dict[str, Dict[str, ModelAgreementMetrics]],
    dimension_labels: Dict[str, Dict[int, str]]
) -> pd.DataFrame:
    """
    Create a table with dimension-level accuracy for each model.
    
    Returns a DataFrame with columns:
    - Construct
    - Model  
    - Dimension (numeric)
    - Dimension Label
    - Accuracy
    """
    rows = []
    
    for construct, model_metrics in all_metrics.items():
        labels = dimension_labels.get(construct, {})
        
        for model_name, metrics in model_metrics.items():
            for dim, accuracy in metrics.dimension_accuracies.items():
                rows.append({
                    'Construct': construct,
                    'Model': model_name,
                    'Dimension': dim,
                    'Dimension Label': labels.get(dim, f'Dimension {dim}'),
                    'Accuracy': round(accuracy, 3) if not np.isnan(accuracy) else 'N/A'
                })
    
    return pd.DataFrame(rows)


def create_final_assignments_table(all_assignments: Dict[str, List[FinalAssignment]]) -> pd.DataFrame:
    """Create a table of all final assignments."""
    rows = []
    
    for construct, assignments in all_assignments.items():
        for a in assignments:
            rows.append({
                'Construct': construct,
                'Questionnaire': a.questionnaire,
                'Item ID': a.item_id,
                'Question Text': a.question_text[:100] + '...' if len(a.question_text) > 100 else a.question_text,
                'Expert Dimension': a.expert_dimension,
                'Expert Dimension Label': a.expert_dimension_label,
                'Model Dimension': a.model_dimension,
                'Model Dimension Label': a.model_dimension_label,
                'Final Dimension': a.final_dimension,
                'Final Dimension Label': a.final_dimension_label,
                'Expert Confidence (P_i)': round(a.expert_confidence, 3),
                'Model Confidence (margin)': round(a.model_confidence, 3),
                'Expert-Model Agree': a.expert_model_agree
            })
    
    return pd.DataFrame(rows)


# =============================================================================
# Corrections Workflow Functions
# =============================================================================

def load_corrections(corrections_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load corrections for disagreement items.
    
    Expected columns in corrections file:
    - Item ID: The item identifier (e.g., 'BDI_01')
    - Construct: The construct name
    - Corrected Dimension: The final corrected dimension assignment
    
    Optional columns:
    - Corrected Dimension Label: Label for the dimension
    - Notes: Any notes about the correction
    
    Parameters
    ----------
    corrections_path : Union[str, Path]
        Path to the corrections CSV file
    
    Returns
    -------
    pd.DataFrame
        Corrections dataframe with at minimum Item ID, Construct, and Corrected Dimension
    """
    corrections_df = pd.read_csv(corrections_path)
    
    # Validate required columns
    required_cols = ['Item ID', 'Construct', 'Corrected Dimension']
    missing_cols = [col for col in required_cols if col not in corrections_df.columns]
    if missing_cols:
        raise ValueError(f"Corrections file missing required columns: {missing_cols}")
    
    print(f"Loaded {len(corrections_df)} corrections from {corrections_path}")
    
    # Show summary by construct
    for construct in corrections_df['Construct'].unique():
        n_corrections = len(corrections_df[corrections_df['Construct'] == construct])
        print(f"  {construct}: {n_corrections} corrections")
    
    return corrections_df


def apply_corrections_to_assignments(
    assignments_df: pd.DataFrame,
    corrections_df: pd.DataFrame,
    dimension_labels_by_construct: Dict[str, Dict[int, str]]
) -> pd.DataFrame:
    """
    Apply corrections to final assignments, creating a corrected version for fine-tuning.
    
    Parameters
    ----------
    assignments_df : pd.DataFrame
        Original final assignments table (from create_final_assignments_table)
    corrections_df : pd.DataFrame
        Corrections dataframe with Item ID, Construct, Corrected Dimension
    dimension_labels_by_construct : Dict[str, Dict[int, str]]
        Dictionary mapping construct -> {dimension_id: dimension_label}
    
    Returns
    -------
    pd.DataFrame
        Corrected assignments table with additional columns tracking corrections
    """
    corrected_df = assignments_df.copy()
    
    # Add columns to track corrections
    corrected_df['Original Dimension'] = corrected_df['Final Dimension']
    corrected_df['Original Dimension Label'] = corrected_df['Final Dimension Label']
    corrected_df['Was Corrected'] = False
    corrected_df['Correction Source'] = 'original'
    
    # Create lookup for corrections
    corrections_lookup = {}
    for _, row in corrections_df.iterrows():
        key = (row['Item ID'], row['Construct'])
        corrections_lookup[key] = row['Corrected Dimension']
    
    # Apply corrections
    n_corrected = 0
    for idx, row in corrected_df.iterrows():
        key = (row['Item ID'], row['Construct'])
        if key in corrections_lookup:
            new_dim = corrections_lookup[key]
            construct = row['Construct']
            
            # Update the dimension
            corrected_df.at[idx, 'Final Dimension'] = new_dim
            
            # Update the label
            if construct in dimension_labels_by_construct:
                dim_labels = dimension_labels_by_construct[construct]
                new_label = dim_labels.get(new_dim, "Does not fit" if new_dim == -1 else "Unknown")
            else:
                new_label = f"Dimension {new_dim}" if new_dim != -1 else "Does not fit"
            corrected_df.at[idx, 'Final Dimension Label'] = new_label
            
            # Mark as corrected
            corrected_df.at[idx, 'Was Corrected'] = True
            corrected_df.at[idx, 'Correction Source'] = 'expert_review'
            n_corrected += 1
    
    print(f"\nApplied {n_corrected} corrections to assignments")
    
    # Summary of changes
    if n_corrected > 0:
        changed = corrected_df[corrected_df['Was Corrected']]
        print("\nCorrected items:")
        for _, row in changed.iterrows():
            print(f"  {row['Item ID']} ({row['Construct']}): "
                  f"Dim {row['Original Dimension']} → Dim {row['Final Dimension']}")
    
    return corrected_df


def create_finetuning_dataset(
    corrected_assignments_df: pd.DataFrame,
    item_assignments_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Create the final dataset for fine-tuning the model.
    
    This combines:
    1. Final corrected dimension assignments
    2. Expert confidence (P_i) from original ratings
    3. Expert proportion distribution (E_ik) for soft labels
    
    Parameters
    ----------
    corrected_assignments_df : pd.DataFrame
        Corrected final assignments
    item_assignments_df : pd.DataFrame
        Original item assignments with Rating Distribution and P_i
    output_path : Optional[Union[str, Path]]
        If provided, save the dataset to this path
    
    Returns
    -------
    pd.DataFrame
        Dataset ready for fine-tuning with columns:
        - Item ID, Construct, Question Text
        - Final Dimension (corrected)
        - Expert Proportions (E_ik for soft labels)
        - P_i (expert agreement)
        - Source (original/expert_review)
    """
    # Get the expert rating distributions
    expert_data = {}
    for _, row in item_assignments_df.iterrows():
        item_id = row['Item ID']
        construct = row['Construct']
        key = (item_id, construct)
        
        # Parse rating distribution
        p_i, e_ik = compute_expert_confidence(row.get('Rating Distribution', ''))
        
        expert_data[key] = {
            'P_i': row.get('P_i', p_i),
            'E_ik': e_ik,
            'Rating Distribution': row.get('Rating Distribution', '')
        }
    
    # Build fine-tuning dataset
    rows = []
    for _, row in corrected_assignments_df.iterrows():
        key = (row['Item ID'], row['Construct'])
        expert_info = expert_data.get(key, {'P_i': 0, 'E_ik': {}, 'Rating Distribution': ''})

        final_dim = row['Final Dimension']
        was_corrected = row.get('Was Corrected', False)

        p_i = expert_info['P_i']
        e_ik = expert_info['E_ik']

        # For corrected items where the survey majority voted for a different dimension
        # than the final expert-corrected one, the original P_i/E_ik point the wrong way.
        # Override them so the soft label is authoritative for the correct dimension.
        if was_corrected and e_ik:
            survey_top = max(e_ik, key=e_ik.get)
            if survey_top != final_dim:
                p_i = 1.0
                e_ik = {final_dim: 1.0}

        rows.append({
            'item_id': row['Item ID'],
            'construct': row['Construct'],
            'questionnaire': row['Questionnaire'],
            'question_text': row.get('Question Text', ''),
            'final_dimension': final_dim,
            'final_dimension_label': row['Final Dimension Label'],
            'expert_confidence_p_i': p_i,
            'expert_proportions_e_ik': json.dumps(e_ik),
            'rating_distribution': expert_info['Rating Distribution'],
            'source': row.get('Correction Source', 'original'),
            'was_corrected': was_corrected
        })
    
    finetuning_df = pd.DataFrame(rows)
    
    # Summary
    print(f"\nFine-tuning dataset created with {len(finetuning_df)} items")
    print(f"  Original assignments: {(~finetuning_df['was_corrected']).sum()}")
    print(f"  Corrected assignments: {finetuning_df['was_corrected'].sum()}")
    print(f"  Items with 'Does not fit' (-1): {(finetuning_df['final_dimension'] == -1).sum()}")
    
    if output_path:
        finetuning_df.to_csv(output_path, index=False)
        print(f"\nSaved fine-tuning dataset to {output_path}")
    
    return finetuning_df


def create_corrections_template(
    disagreement_items_path: Union[str, Path],
    output_path: Union[str, Path],
    dimension_labels_by_construct: Dict[str, Dict[int, str]]
) -> pd.DataFrame:
    """
    Create a template CSV for entering corrections for disagreement items.
    
    This is a helper function to create a template that you can fill in
    with the final corrected ratings from expert review.
    
    Parameters
    ----------
    disagreement_items_path : Union[str, Path]
        Path to the expert_disagreement_items.csv file
    output_path : Union[str, Path]
        Path where to save the template
    dimension_labels_by_construct : Dict[str, Dict[int, str]]
        Dictionary mapping construct -> {dimension_id: dimension_label}
    
    Returns
    -------
    pd.DataFrame
        Template dataframe
    """
    disagreement_df = pd.read_csv(disagreement_items_path)
    
    rows = []
    for _, row in disagreement_df.iterrows():
        construct = row.get('Construct', '')
        dim_labels = dimension_labels_by_construct.get(construct, {})
        
        # Create string showing available dimensions
        dim_options = ", ".join([f"{k}: {v}" for k, v in sorted(dim_labels.items())])
        
        rows.append({
            'Item ID': row.get('Item ID', ''),
            'Construct': construct,
            'Questionnaire': row.get('Questionnaire', ''),
            'Question Text': row.get('Question Text', ''),
            'Original Rating Distribution': row.get('Rating Distribution', ''),
            'Original Assigned Dimension': row.get('Assigned Dimension', ''),
            'Corrected Dimension': '',  # To be filled in
            'Dimension Options': dim_options,
            'Notes': ''  # Optional notes
        })
    
    template_df = pd.DataFrame(rows)
    template_df.to_csv(output_path, index=False)
    
    print(f"Created corrections template with {len(template_df)} items")
    print(f"Saved to: {output_path}")
    print("\nInstructions:")
    print("1. Open the CSV file")
    print("2. For each item, enter the corrected dimension number in 'Corrected Dimension'")
    print("3. Use -1 for 'Does not fit'")
    print("4. Add any notes in the 'Notes' column")
    print("5. Save and use with load_corrections() and apply_corrections_to_assignments()")
    
    return template_df


def create_similarity_matrix_table(
    results: Dict[str, Dict[str, Dict]],
    construct: str,
    model_name: str
) -> pd.DataFrame:
    """
    Create a table showing item-to-dimension similarity scores.
    
    This is useful for understanding why items were assigned to specific dimensions.
    """
    if construct not in results or model_name not in results[construct]:
        return pd.DataFrame()
    
    similarity_result = results[construct][model_name]['similarity']
    assignment = results[construct][model_name]['assignment']
    
    rows = []
    for i, item_id in enumerate(similarity_result.item_ids):
        row = {'Item ID': item_id}
        for j, dim_label in enumerate(similarity_result.dimension_labels):
            row[dim_label] = round(similarity_result.item_to_dimension_similarity[i, j], 3)
        row['Assigned To'] = assignment.dimension_labels[
            assignment.dimension_ids.index(assignment.assigned_dimensions[i])
        ]
        rows.append(row)
    
    return pd.DataFrame(rows)


def save_embeddings_and_similarities(
    results: Dict[str, Dict[str, Dict]],
    output_dir: Path
):
    """Save embedding matrices and similarity matrices to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for construct, model_results in results.items():
        construct_dir = output_dir / construct
        construct_dir.mkdir(exist_ok=True)
        
        for model_name, data in model_results.items():
            # Save item-dimension similarity matrix
            sim_result = data['similarity']
            sim_df = pd.DataFrame(
                sim_result.item_to_dimension_similarity,
                index=sim_result.item_ids,
                columns=sim_result.dimension_labels
            )
            sim_df.to_csv(construct_dir / f'{model_name}_item_dimension_similarity.csv')
            
            # Save item embeddings
            emb_result = data['item_embeddings']
            emb_df = pd.DataFrame(
                emb_result.embeddings,
                index=emb_result.item_ids
            )
            emb_df.to_csv(construct_dir / f'{model_name}_item_embeddings.csv')
            
            # Save dimension embeddings
            dim_emb_result = data['dimension_embeddings']
            dim_emb_df = pd.DataFrame(
                dim_emb_result.dimension_embeddings,
                index=dim_emb_result.dimension_labels
            )
            dim_emb_df.to_csv(construct_dir / f'{model_name}_dimension_embeddings.csv')


# =============================================================================
# Main Pipeline Functions
# =============================================================================

def run_step4_analysis(
    items_filepath: Union[str, Path],
    models: Optional[Dict[str, str]] = None,
    constructs: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Dict[str, Dict]]:
    """
    Run Step 4: Theory-Primed Semantic Similarity Analysis.
    
    Parameters
    ----------
    items_filepath : str or Path
        Path to items CSV file
    models : Dict[str, str], optional
        Models to test (default: EMBEDDING_MODELS_LITE)
    constructs : List[str], optional
        Constructs to analyze (default: all)
    output_dir : str or Path, optional
        Directory to save outputs
    
    Returns
    -------
    Dict[str, Dict[str, Dict]]
        Results organized by construct, then model
    """
    if models is None:
        models = EMBEDDING_MODELS_LITE
    
    df_items = pd.read_csv(items_filepath)
    
    if constructs is None:
        constructs = df_items['construct'].unique().tolist()
    
    all_results = {}
    
    for construct in constructs:
        print(f"\n{'='*60}")
        print(f"Processing construct: {construct}")
        print('='*60)
        
        results = embed_and_assign_construct(df_items, construct, models)
        all_results[construct] = results
    
    # Save outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_embeddings_and_similarities(all_results, output_dir / 'embeddings_and_similarities')
        print(f"\nStep 4 outputs saved to {output_dir}/")
    
    return all_results


def run_step5_analysis(
    items_filepath: Union[str, Path],
    expert_assignments_filepath: Union[str, Path],
    step4_results: Dict[str, Dict[str, Dict]],
    dimension_labels: Dict[str, Dict[int, str]],
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, IntegrationResults]:
    """
    Run Step 5: Integration of Mappings.

    Framework:
    - Expert: argmax(E_ik), confidence P_i = Σ(E_ik²)
    - Model: argmax(similarity_k), confidence = top1-top2 similarity margin
    - Final: Always the expert assignment (experts take precedence); model
      scores are reported alongside for agreement/validation purposes only.

    Parameters
    ----------
    items_filepath : str or Path
        Path to items CSV file
    expert_assignments_filepath : str or Path
        Path to item_assignments.csv from Step 3
    step4_results : Dict
        Results from Step 4
    dimension_labels : Dict
        Dimension labels for each construct
    output_dir : str or Path, optional
        Directory to save outputs

    Returns
    -------
    Dict[str, IntegrationResults]
        Integration results for each construct
    """
    df_items = pd.read_csv(items_filepath)
    df_expert = pd.read_csv(expert_assignments_filepath)

    all_metrics = {}
    best_models = {}
    all_assignments = {}
    integration_results = {}

    # =================================================================
    # First pass: Evaluate models and select best for each construct
    # =================================================================
    for construct, model_results in step4_results.items():
        print(f"\n{'-'*40}")
        print(f"Evaluating models for: {construct}")
        print('-'*40)

        # Get expert assignments
        expert_assignments = get_expert_assignments(df_expert, construct)

        # Evaluate each model
        construct_metrics = {}
        for model_name, data in model_results.items():
            assignment = data['assignment']

            metrics = evaluate_model_agreement(assignment, expert_assignments)
            construct_metrics[model_name] = metrics

            print(f"  {model_name}: κ = {metrics.cohens_kappa:.3f}, "
                  f"accuracy = {metrics.accuracy:.3f}")

        all_metrics[construct] = construct_metrics

        # Select best model
        best_model, best_metrics = select_best_model(construct_metrics)
        best_models[construct] = best_model
        print(f"\n  Selected model: {best_model}")

    # =================================================================
    # Second pass: Generate final assignments
    # =================================================================
    print(f"\n{'='*60}")
    print("GENERATING FINAL ASSIGNMENTS")
    print("="*60)

    for construct, model_results in step4_results.items():
        print(f"\n{'-'*40}")
        print(f"Processing: {construct}")
        print('-'*40)

        best_model = best_models[construct]
        best_assignment = model_results[best_model]['assignment']
        labels = dimension_labels.get(construct, {})

        final_assignments = generate_final_assignments(
            df_items, df_expert, best_assignment, construct, labels
        )
        all_assignments[construct] = final_assignments

        # Compute summary statistics
        n_items = len(final_assignments)
        n_agree = sum(1 for a in final_assignments if a.expert_model_agree)

        print(f"  Items: {n_items}")
        print(f"  Expert-Model agreement: {n_agree}/{n_items} ({100*n_agree/n_items:.1f}%)")

        # Compile results
        integration_results[construct] = IntegrationResults(
            construct=construct,
            best_model=best_model,
            model_metrics=all_metrics[construct],
            final_assignments=final_assignments,
            n_disagree=n_items - n_agree
        )

    # =================================================================
    # Print summary report
    # =================================================================
    print(f"\n{'='*60}")
    print("INTEGRATION SUMMARY REPORT")
    print("="*60)

    print(f"\n{'Construct':<20} {'Best Model':<25} {'Agreement':>15}")
    print("-"*65)

    for construct in step4_results.keys():
        best_model = best_models[construct]
        assignments = all_assignments[construct]
        n_agree = sum(1 for a in assignments if a.expert_model_agree)
        n_total = len(assignments)

        print(f"{construct:<20} {best_model:<25} "
              f"{n_agree}/{n_total:>6} ({100*n_agree/n_total:>4.1f}%)")

    # =================================================================
    # Save outputs
    # =================================================================
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Model comparison
        model_df = create_model_comparison_table(all_metrics)
        model_df.to_csv(output_dir / 'model_comparison.csv', index=False)

        # Dimension-level accuracy table
        accuracy_df = create_dimension_accuracy_table(all_metrics, dimension_labels)
        accuracy_df.to_csv(output_dir / 'dimension_accuracy_detailed.csv', index=False)

        # Final assignments
        assign_df = create_final_assignments_table(all_assignments)
        assign_df.to_csv(output_dir / 'final_assignments_combined.csv', index=False)

        # Similarity matrices for best models
        for construct, model_name in best_models.items():
            sim_df = create_similarity_matrix_table(step4_results, construct, model_name)
            sim_df.to_csv(output_dir / f'{construct}_similarity_scores.csv', index=False)

        print(f"\nResults saved to {output_dir}/")

    return integration_results


# =============================================================================
# Main Entry Point
# =============================================================================

def main(
    items_filepath: Union[str, Path],
    expert_assignments_filepath: Union[str, Path],
    output_dir: Union[str, Path] = './output_steps4_5',
    models: Optional[Dict[str, str]] = None,
    constructs: Optional[List[str]] = None,
    corrections_filepath: Optional[Union[str, Path]] = None,
    create_corrections_template_only: bool = False,
    disagreement_items_filepath: Optional[Union[str, Path]] = None,
    dimension_config_path: Optional[Union[str, Path]] = None
) -> Tuple[Dict, Dict]:
    """
    Main function to run Steps 4 and 5 of the analysis.

    Parameters
    ----------
    items_filepath : str or Path
        Path to original items CSV (items-and-ratings.csv)
    expert_assignments_filepath : str or Path
        Path to item_assignments.csv from Step 3
    output_dir : str or Path
        Directory for outputs
    models : Dict[str, str], optional
        Models to test
    constructs : List[str], optional
        Constructs to analyze
    corrections_filepath : str or Path, optional
        Path to corrections CSV file for disagreement items.
        If provided, corrections will be applied and final_assignments_corrected.csv created.
    create_corrections_template_only : bool
        If True, only create a corrections template and exit
    disagreement_items_filepath : str or Path, optional
        Path to expert_disagreement_items.csv (needed for template creation)
    dimension_config_path : str or Path, optional
        Path to a dimension_descriptions.yaml (default: DEFAULT_DIMENSION_CONFIG).
        Point this at your own copy to run the pipeline on different
        constructs/dimensions without editing this script.

    Returns
    -------
    step4_results : Dict
        Embeddings and assignment results
    step5_results : Dict
        Integration results
    """
    global DIMENSION_DESCRIPTIONS
    if dimension_config_path is not None:
        DIMENSION_DESCRIPTIONS, dimension_labels = load_dimension_config(dimension_config_path)
    else:
        dimension_labels = DEFAULT_DIMENSION_LABELS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If only creating corrections template, do that and exit
    if create_corrections_template_only:
        if not disagreement_items_filepath:
            raise ValueError("--disagreement-items required when creating corrections template")
        template_path = output_dir / "corrections_template.csv"
        create_corrections_template(
            disagreement_items_filepath,
            template_path,
            dimension_labels
        )
        return {}, {}
    
    print("="*80)
    print("ENIGMA-PD Questionnaire Harmonization")
    print("Step 4: Theory-Primed Semantic Similarity Analysis")
    print("Step 5: Integration of Mappings")
    print("="*80)
    print("\nThis analysis uses THEORY-PRIMED assignment:")
    print("  - Items are assigned to dimensions based on semantic similarity")
    print("    to dimension descriptions (the same info given to experts)")
    print("  - FINAL ASSIGNMENTS ARE ALWAYS BASED ON EXPERT DECISIONS")
    print("  - Model scores are computed for reporting purposes only")
    print("  - Expert 'Does not fit' (-1) decisions are preserved")
    
    # Step 4
    print("\n" + "="*80)
    print("STEP 4: THEORY-PRIMED SEMANTIC SIMILARITY ANALYSIS")
    print("="*80)
    
    step4_results = run_step4_analysis(
        items_filepath,
        models=models,
        constructs=constructs,
        output_dir=output_dir
    )
    
    # Step 5
    print("\n" + "="*80)
    print("STEP 5: INTEGRATION OF MAPPINGS")
    print("="*80)
    
    step5_results = run_step5_analysis(
        items_filepath,
        expert_assignments_filepath,
        step4_results,
        dimension_labels,
        output_dir=output_dir
    )
    
    # Apply corrections if provided
    if corrections_filepath:
        print("\n" + "="*80)
        print("APPLYING CORRECTIONS TO FINAL ASSIGNMENTS")
        print("="*80)
        
        # Load the final assignments table
        final_assignments_path = output_dir / "final_assignments_combined.csv"
        if final_assignments_path.exists():
            assignments_df = pd.read_csv(final_assignments_path)
        else:
            # Create from step5_results
            all_assignments = {
                construct: result.final_assignments 
                for construct, result in step5_results.items()
            }
            assignments_df = create_final_assignments_table(all_assignments)
        
        # Load and apply corrections
        corrections_df = load_corrections(corrections_filepath)
        corrected_df = apply_corrections_to_assignments(
            assignments_df, 
            corrections_df,
            dimension_labels
        )
        
        # Save corrected assignments
        corrected_path = output_dir / "final_assignments_corrected.csv"
        corrected_df.to_csv(corrected_path, index=False)
        print(f"\nSaved corrected assignments to {corrected_path}")
        
        # Create fine-tuning dataset
        item_assignments_df = pd.read_csv(expert_assignments_filepath)
        finetuning_path = output_dir / "finetuning_dataset.csv"
        finetuning_df = create_finetuning_dataset(
            corrected_df,
            item_assignments_df,
            output_path=finetuning_path
        )
        
        # Add to results
        step5_results['corrected_assignments'] = corrected_df
        step5_results['finetuning_dataset'] = finetuning_df
    
    return step4_results, step5_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Steps 4 & 5: Theory-Primed Semantic Similarity and Integration Analysis"
    )
    parser.add_argument(
        "items_filepath",
        type=str,
        help="Path to items CSV file (items-and-ratings.csv)"
    )
    parser.add_argument(
        "expert_assignments_filepath",
        type=str,
        help="Path to item_assignments.csv from Step 3"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./output_steps4_5",
        help="Output directory"
    )
    parser.add_argument(
        "--use-all-models",
        action="store_true",
        help="Use all models (slower) instead of lite subset"
    )
    parser.add_argument(
        "--constructs",
        type=str,
        nargs="+",
        help="Specific constructs to analyze"
    )
    parser.add_argument(
        "--corrections",
        type=str,
        help="Path to corrections CSV file for disagreement items"
    )
    parser.add_argument(
        "--create-corrections-template",
        action="store_true",
        help="Create a template for entering corrections and exit"
    )
    parser.add_argument(
        "--disagreement-items",
        type=str,
        help="Path to expert_disagreement_items.csv (needed for template creation)"
    )
    parser.add_argument(
        "--dimension-config",
        type=str,
        default=None,
        help="Path to a dimension_descriptions.yaml defining the constructs/dimensions "
             f"to assign items to (default: {DEFAULT_DIMENSION_CONFIG}). Point this at "
             "your own copy to run on different constructs without editing this script."
    )

    args = parser.parse_args()

    models = EMBEDDING_MODELS if args.use_all_models else EMBEDDING_MODELS_LITE

    step4_results, step5_results = main(
        items_filepath=args.items_filepath,
        expert_assignments_filepath=args.expert_assignments_filepath,
        output_dir=args.output_dir,
        models=models,
        constructs=args.constructs,
        corrections_filepath=args.corrections,
        create_corrections_template_only=args.create_corrections_template,
        disagreement_items_filepath=args.disagreement_items,
        dimension_config_path=args.dimension_config
    )