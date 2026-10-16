"""
Descriptive and Exploratory Analysis for ENIGMA-PD Expert Survey Data
======================================================================

Analyses included:
1. Basic descriptive statistics
2. Rater-level analysis (outlier detection, systematic disagreement)
3. Item-level analysis (difficulty, discrimination)
4. Construct-level comparisons
5. Agreement pattern analysis
6. Visualization of results

Special value handling:
- -9: Missing data (excluded from analysis)
- -1: "None/Does not fit" - valid rating
"""

import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
from pathlib import Path
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent.parent
DEFAULT_DIMENSION_CONFIG = PROJECT_DIR / "input" / "shared" / "dimension_descriptions.yaml"


# =============================================================================
# Configuration: Dimension Definitions
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

# Number of valid dimensions per construct (excludes -1 "Does not fit"),
# derived from DIMENSION_LABELS so it can never drift out of sync with it.
N_DIMENSIONS = {
    construct: sum(1 for dim_id in labels if dim_id != -1)
    for construct, labels in DIMENSION_LABELS.items()
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RaterProfile:
    """Profile of a single rater's behavior."""
    rater_id: str
    n_ratings_total: int
    n_ratings_per_construct: Dict[str, int]
    completion_rate: float
    none_usage_rate: float  # How often they use "None/Does not fit"
    dimension_preferences: Dict[str, Dict[int, float]]  # Per construct
    agreement_with_majority: float
    mean_pairwise_agreement: float
    is_potential_outlier: bool
    outlier_reasons: List[str]


@dataclass
class ItemProfile:
    """Profile of a single item's characteristics."""
    item_id: str
    construct: str
    questionnaire: str
    question_text: str
    n_valid_ratings: int
    modal_dimension: int
    modal_proportion: float
    entropy: float  # Higher = more disagreement
    max_pairwise_disagreement: float
    dimension_distribution: Dict[int, float]
    is_problematic: bool
    problematic_reasons: List[str]


@dataclass 
class RaterPairAnalysis:
    """Analysis of agreement between a pair of raters."""
    rater_1: str
    rater_2: str
    n_common_items: int
    agreement_rate: float
    cohens_kappa: float
    disagreement_pattern: Dict[Tuple[int, int], int]  # (dim1, dim2): count


@dataclass
class ConstructProfile:
    """Profile of a construct's rating patterns."""
    construct: str
    n_items: int
    n_raters: int
    mean_raters_per_item: float
    dimension_distribution: Dict[int, float]
    mean_item_entropy: float
    mean_agreement_with_majority: float
    hardest_items: List[str]
    easiest_items: List[str]


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_data(filepath: Union[str, Path], missing_value: int = -9) -> pd.DataFrame:
    """Load and preprocess the expert ratings data."""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    
    # Identify rater columns
    rater_cols = [col for col in df.columns if col.startswith('rater_')]
    
    # Replace missing values with NaN
    for col in rater_cols:
        df[col] = df[col].replace(missing_value, np.nan)
    
    print(f"Loaded {len(df)} items with {len(rater_cols)} raters")
    print(f"Constructs: {df['construct'].unique().tolist()}")
    
    return df


def get_rater_columns(df: pd.DataFrame) -> List[str]:
    """Get list of rater column names."""
    return [col for col in df.columns if col.startswith('rater_')]


# =============================================================================
# 1. Basic Descriptive Statistics
# =============================================================================

def compute_basic_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic descriptive statistics for the dataset."""
    rater_cols = get_rater_columns(df)
    
    results = []
    
    for construct in df['construct'].unique():
        construct_df = df[df['construct'] == construct]
        ratings = construct_df[rater_cols].values
        
        # Count valid ratings
        valid_ratings = ~np.isnan(ratings)
        n_valid = valid_ratings.sum()
        
        # Flatten valid ratings
        flat_ratings = ratings[valid_ratings].astype(int)
        
        # Count dimension usage
        unique, counts = np.unique(flat_ratings, return_counts=True)
        dim_counts = dict(zip(unique, counts))
        
        results.append({
            'Construct': construct,
            'N Items': len(construct_df),
            'N Raters (max)': len(rater_cols),
            'Total Valid Ratings': n_valid,
            'Mean Ratings per Item': n_valid / len(construct_df),
            'Mean Ratings per Rater': n_valid / len(rater_cols),
            'None Usage (%)': 100 * dim_counts.get(-1, 0) / n_valid if n_valid > 0 else 0,
            'N Questionnaires': construct_df['questionnaire_name'].nunique()
        })
    
    return pd.DataFrame(results)


def compute_response_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute distribution of responses across dimensions per construct."""
    rater_cols = get_rater_columns(df)
    
    results = []
    
    for construct in df['construct'].unique():
        construct_df = df[df['construct'] == construct]
        ratings = construct_df[rater_cols].values
        
        # Get valid ratings
        valid_mask = ~np.isnan(ratings)
        flat_ratings = ratings[valid_mask].astype(int)
        
        # Count dimensions
        unique, counts = np.unique(flat_ratings, return_counts=True)
        total = counts.sum()
        
        for dim, count in zip(unique, counts):
            dim_label = DIMENSION_LABELS.get(construct, {}).get(dim, f"Dimension {dim}")
            results.append({
                'Construct': construct,
                'Dimension': dim,
                'Dimension Label': dim_label,
                'Count': count,
                'Percentage': 100 * count / total if total > 0 else 0
            })
    
    return pd.DataFrame(results)


# =============================================================================
# 2. Rater-Level Analysis
# =============================================================================

def compute_rater_completion(df: pd.DataFrame) -> pd.DataFrame:
    """Compute completion rates for each rater."""
    rater_cols = get_rater_columns(df)
    
    results = []
    
    for rater in rater_cols:
        for construct in df['construct'].unique():
            construct_df = df[df['construct'] == construct]
            ratings = construct_df[rater].values
            
            n_items = len(construct_df)
            n_completed = (~np.isnan(ratings)).sum()
            
            results.append({
                'Rater': rater,
                'Construct': construct,
                'N Items': n_items,
                'N Completed': n_completed,
                'Completion Rate (%)': 100 * n_completed / n_items if n_items > 0 else 0
            })
    
    return pd.DataFrame(results)


def compute_rater_majority_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute how often each rater agrees with the majority vote.
    
    The majority is defined as the modal (most frequent) dimension for each item.
    """
    rater_cols = get_rater_columns(df)
    
    results = []
    
    for construct in df['construct'].unique():
        construct_df = df[df['construct'] == construct].copy()
        ratings = construct_df[rater_cols].values
        
        # Compute majority (mode) for each item
        majorities = []
        for i in range(ratings.shape[0]):
            row = ratings[i, :]
            valid = row[~np.isnan(row)].astype(int)
            if len(valid) > 0:
                counts = Counter(valid)
                majority = counts.most_common(1)[0][0]
            else:
                majority = np.nan
            majorities.append(majority)
        
        majorities = np.array(majorities)
        
        # Compute agreement for each rater
        for j, rater in enumerate(rater_cols):
            rater_ratings = ratings[:, j]
            valid_mask = ~np.isnan(rater_ratings) & ~np.isnan(majorities)
            
            if valid_mask.sum() > 0:
                agreements = rater_ratings[valid_mask] == majorities[valid_mask]
                agreement_rate = agreements.sum() / valid_mask.sum()
            else:
                agreement_rate = np.nan
            
            results.append({
                'Rater': rater,
                'Construct': construct,
                'N Items Rated': valid_mask.sum(),
                'Agreement with Majority (%)': 100 * agreement_rate if not np.isnan(agreement_rate) else np.nan
            })
    
    return pd.DataFrame(results)


def compute_rater_none_usage(df: pd.DataFrame, none_value: int = -1) -> pd.DataFrame:
    """Compute how often each rater uses the 'None/Does not fit' category."""
    rater_cols = get_rater_columns(df)
    
    results = []
    
    for construct in df['construct'].unique():
        construct_df = df[df['construct'] == construct]
        ratings = construct_df[rater_cols].values
        
        for j, rater in enumerate(rater_cols):
            rater_ratings = ratings[:, j]
            valid = rater_ratings[~np.isnan(rater_ratings)]
            
            if len(valid) > 0:
                none_count = (valid == none_value).sum()
                none_rate = none_count / len(valid)
            else:
                none_count = 0
                none_rate = np.nan
            
            results.append({
                'Rater': rater,
                'Construct': construct,
                'N Valid Ratings': len(valid),
                'N None Responses': none_count,
                'None Usage Rate (%)': 100 * none_rate if not np.isnan(none_rate) else np.nan
            })
    
    return pd.DataFrame(results)


def compute_pairwise_rater_agreement(df: pd.DataFrame, construct: str) -> pd.DataFrame:
    """
    Compute pairwise agreement between all raters for a specific construct.
    
    Returns a matrix of agreement rates.
    """
    construct_df = df[df['construct'] == construct]
    rater_cols = get_rater_columns(df)
    ratings = construct_df[rater_cols].values
    
    n_raters = len(rater_cols)
    agreement_matrix = np.full((n_raters, n_raters), np.nan)
    
    for i in range(n_raters):
        for j in range(i, n_raters):
            if i == j:
                agreement_matrix[i, j] = 1.0
            else:
                r1 = ratings[:, i]
                r2 = ratings[:, j]
                
                # Both must have rated the item
                valid_mask = ~np.isnan(r1) & ~np.isnan(r2)
                
                if valid_mask.sum() >= 5:  # Minimum items for meaningful comparison
                    agreements = r1[valid_mask] == r2[valid_mask]
                    agreement_rate = agreements.sum() / valid_mask.sum()
                    agreement_matrix[i, j] = agreement_rate
                    agreement_matrix[j, i] = agreement_rate
    
    return pd.DataFrame(agreement_matrix, index=rater_cols, columns=rater_cols)


def identify_rater_outliers(df: pd.DataFrame, 
                           agreement_threshold: float = 0.5,
                           none_threshold: float = 0.3) -> pd.DataFrame:
    """
    Identify raters who may be outliers based on:
    1. Low agreement with majority (< agreement_threshold)
    2. Excessive use of "None" category (> none_threshold)
    3. Low mean pairwise agreement with other raters
    """
    rater_cols = get_rater_columns(df)
    
    # Get agreement with majority
    agreement_df = compute_rater_majority_agreement(df)
    agreement_summary = agreement_df.groupby('Rater')['Agreement with Majority (%)'].mean()
    
    # Get None usage
    none_df = compute_rater_none_usage(df)
    none_summary = none_df.groupby('Rater')['None Usage Rate (%)'].mean()
    
    # Compute mean pairwise agreement per rater
    pairwise_agreements = {}
    for construct in df['construct'].unique():
        pair_matrix = compute_pairwise_rater_agreement(df, construct)
        for rater in rater_cols:
            if rater not in pairwise_agreements:
                pairwise_agreements[rater] = []
            # Mean agreement with all other raters
            rater_agreements = pair_matrix.loc[rater, :].drop(rater)
            valid_agreements = rater_agreements.dropna()
            if len(valid_agreements) > 0:
                pairwise_agreements[rater].append(valid_agreements.mean())
    
    results = []
    for rater in rater_cols:
        reasons = []
        
        mean_agreement = agreement_summary.get(rater, np.nan)
        mean_none = none_summary.get(rater, np.nan)
        
        if len(pairwise_agreements.get(rater, [])) > 0:
            mean_pairwise = np.mean(pairwise_agreements[rater])
        else:
            mean_pairwise = np.nan
        
        # Check outlier criteria
        if not np.isnan(mean_agreement) and mean_agreement < agreement_threshold * 100:
            reasons.append(f"Low majority agreement ({mean_agreement:.1f}%)")
        
        if not np.isnan(mean_none) and mean_none > none_threshold * 100:
            reasons.append(f"High None usage ({mean_none:.1f}%)")
        
        if not np.isnan(mean_pairwise) and mean_pairwise < agreement_threshold:
            reasons.append(f"Low pairwise agreement ({mean_pairwise:.1%})")
        
        is_outlier = len(reasons) > 0
        
        results.append({
            'Rater': rater,
            'Mean Agreement with Majority (%)': mean_agreement,
            'Mean None Usage (%)': mean_none,
            'Mean Pairwise Agreement': mean_pairwise,
            'Is Potential Outlier': is_outlier,
            'Outlier Reasons': '; '.join(reasons) if reasons else 'None'
        })
    
    return pd.DataFrame(results)


def compute_rater_dimension_preferences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze if certain raters show systematic preferences for specific dimensions.
    
    Compares each rater's dimension usage to the overall distribution.
    """
    rater_cols = get_rater_columns(df)
    
    results = []
    
    for construct in df['construct'].unique():
        construct_df = df[df['construct'] == construct]
        ratings = construct_df[rater_cols].values
        
        # Overall distribution
        all_valid = ratings[~np.isnan(ratings)].astype(int)
        overall_counts = Counter(all_valid)
        overall_total = sum(overall_counts.values())
        overall_props = {k: v/overall_total for k, v in overall_counts.items()}
        
        # Per-rater distribution
        for j, rater in enumerate(rater_cols):
            rater_ratings = ratings[:, j]
            valid = rater_ratings[~np.isnan(rater_ratings)].astype(int)
            
            if len(valid) < 5:
                continue
            
            rater_counts = Counter(valid)
            rater_total = sum(rater_counts.values())
            
            # Compare to overall
            for dim in set(overall_counts.keys()) | set(rater_counts.keys()):
                rater_prop = rater_counts.get(dim, 0) / rater_total
                overall_prop = overall_props.get(dim, 0)
                deviation = rater_prop - overall_prop
                
                dim_label = DIMENSION_LABELS.get(construct, {}).get(dim, f"Dim {dim}")
                
                results.append({
                    'Rater': rater,
                    'Construct': construct,
                    'Dimension': dim,
                    'Dimension Label': dim_label,
                    'Rater Usage (%)': 100 * rater_prop,
                    'Overall Usage (%)': 100 * overall_prop,
                    'Deviation (pp)': 100 * deviation  # Percentage points
                })
    
    return pd.DataFrame(results)


def find_systematic_disagreers(df: pd.DataFrame, min_deviation: float = 15.0) -> pd.DataFrame:
    """
    Find raters who systematically deviate from the group on specific dimensions.
    
    Parameters
    ----------
    min_deviation : float
        Minimum deviation in percentage points to flag
    """
    pref_df = compute_rater_dimension_preferences(df)
    
    # Find large deviations
    flagged = pref_df[abs(pref_df['Deviation (pp)']) >= min_deviation].copy()
    flagged = flagged.sort_values('Deviation (pp)', key=abs, ascending=False)
    
    return flagged


# =============================================================================
# 3. Item-Level Analysis
# =============================================================================

def compute_item_entropy(ratings: np.ndarray) -> float:
    """
    Compute Shannon entropy of ratings for an item.
    Higher entropy = more disagreement.
    """
    valid = ratings[~np.isnan(ratings)].astype(int)
    if len(valid) == 0:
        return np.nan
    
    counts = Counter(valid)
    total = sum(counts.values())
    probs = np.array([c/total for c in counts.values()])
    
    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    return entropy


def compute_item_profiles(df: pd.DataFrame) -> List[ItemProfile]:
    """Compute detailed profiles for each item."""
    rater_cols = get_rater_columns(df)
    
    profiles = []
    
    for _, row in df.iterrows():
        ratings = row[rater_cols].values.astype(float)
        valid = ratings[~np.isnan(ratings)].astype(int)
        
        if len(valid) == 0:
            continue
        
        # Dimension distribution
        counts = Counter(valid)
        total = sum(counts.values())
        distribution = {k: v/total for k, v in counts.items()}
        
        # Modal dimension
        modal_dim = counts.most_common(1)[0][0]
        modal_prop = counts.most_common(1)[0][1] / total
        
        # Entropy
        entropy = compute_item_entropy(ratings)
        
        # Check for problems
        problems = []
        if modal_prop < 0.5:
            problems.append(f"Low modal agreement ({modal_prop:.0%})")
        if entropy > 1.5:
            problems.append(f"High entropy ({entropy:.2f})")
        if len(counts) > 3 and row['construct'] != 'impulse_control':
            problems.append(f"Spread across {len(counts)} dimensions")
        
        profiles.append(ItemProfile(
            item_id=row['question_number'],
            construct=row['construct'],
            questionnaire=row['questionnaire_name'],
            question_text=row.get('question_text', ''),  # stripped from public item files (copyright)
            n_valid_ratings=len(valid),
            modal_dimension=modal_dim,
            modal_proportion=modal_prop,
            entropy=entropy,
            max_pairwise_disagreement=0,  # Placeholder
            dimension_distribution=distribution,
            is_problematic=len(problems) > 0,
            problematic_reasons=problems
        ))
    
    return profiles


def compute_item_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 'difficulty' of items based on agreement level.
    
    Difficulty = 1 - modal_proportion
    (Easier items have higher agreement)
    """
    profiles = compute_item_profiles(df)
    
    results = []
    for p in profiles:
        results.append({
            'Item ID': p.item_id,
            'Construct': p.construct,
            'Questionnaire': p.questionnaire,
            'Question Text': p.question_text[:80] + '...' if len(p.question_text) > 80 else p.question_text,
            'Modal Dimension': p.modal_dimension,
            'Modal Dimension Label': DIMENSION_LABELS.get(p.construct, {}).get(p.modal_dimension, f"Dim {p.modal_dimension}"),
            'Modal Proportion': p.modal_proportion,
            'Difficulty (1-modal)': 1 - p.modal_proportion,
            'Entropy': p.entropy,
            'N Raters': p.n_valid_ratings,
            'Is Problematic': p.is_problematic,
            'Problem Reasons': '; '.join(p.problematic_reasons) if p.problematic_reasons else 'None'
        })
    
    return pd.DataFrame(results).sort_values('Difficulty (1-modal)', ascending=False)


def find_confusing_dimension_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find pairs of dimensions that are frequently confused.
    
    For each item with low agreement, identify which dimensions
    are competing.
    """
    profiles = compute_item_profiles(df)
    
    confusion_counts = {}
    
    for p in profiles:
        if p.modal_proportion < 0.7:  # Item has some confusion
            # Get top 2 dimensions
            sorted_dims = sorted(p.dimension_distribution.items(), 
                                key=lambda x: x[1], reverse=True)
            
            if len(sorted_dims) >= 2:
                dim1, prop1 = sorted_dims[0]
                dim2, prop2 = sorted_dims[1]
                
                if prop2 >= 0.15:  # Second dimension has meaningful support
                    key = (p.construct, min(dim1, dim2), max(dim1, dim2))
                    if key not in confusion_counts:
                        confusion_counts[key] = {
                            'count': 0, 
                            'items': [],
                            'dim1_label': DIMENSION_LABELS.get(p.construct, {}).get(min(dim1, dim2)),
                            'dim2_label': DIMENSION_LABELS.get(p.construct, {}).get(max(dim1, dim2))
                        }
                    confusion_counts[key]['count'] += 1
                    confusion_counts[key]['items'].append(p.item_id)
    
    results = []
    for (construct, dim1, dim2), info in confusion_counts.items():
        results.append({
            'Construct': construct,
            'Dimension 1': dim1,
            'Dimension 1 Label': info['dim1_label'],
            'Dimension 2': dim2,
            'Dimension 2 Label': info['dim2_label'],
            'N Items Confused': info['count'],
            'Example Items': ', '.join(info['items'][:5])
        })
    
    return pd.DataFrame(results).sort_values('N Items Confused', ascending=False)


# =============================================================================
# 4. Construct-Level Comparisons
# =============================================================================

def compare_constructs(df: pd.DataFrame) -> pd.DataFrame:
    """Compare rating patterns across constructs."""
    rater_cols = get_rater_columns(df)
    
    results = []
    
    for construct in df['construct'].unique():
        construct_df = df[df['construct'] == construct]
        ratings = construct_df[rater_cols].values
        
        valid_mask = ~np.isnan(ratings)
        
        # Items per dimension
        flat_ratings = ratings[valid_mask].astype(int)
        counts = Counter(flat_ratings)
        
        # Compute item-level metrics
        item_entropies = []
        item_modal_props = []
        
        for i in range(ratings.shape[0]):
            row = ratings[i, :]
            valid = row[~np.isnan(row)].astype(int)
            if len(valid) > 0:
                item_entropies.append(compute_item_entropy(row))
                mode_count = Counter(valid).most_common(1)[0][1]
                item_modal_props.append(mode_count / len(valid))
        
        results.append({
            'Construct': construct,
            'N Items': len(construct_df),
            'N Dimensions': N_DIMENSIONS[construct],
            'Mean Item Entropy': np.nanmean(item_entropies),
            'SD Item Entropy': np.nanstd(item_entropies),
            'Mean Modal Proportion': np.mean(item_modal_props),
            'SD Modal Proportion': np.std(item_modal_props),
            'None Usage (%)': 100 * counts.get(-1, 0) / sum(counts.values()),
            'Mean Raters per Item': valid_mask.sum() / len(construct_df)
        })
    
    return pd.DataFrame(results)


# =============================================================================
# 5. Agreement Pattern Analysis
# =============================================================================

def analyze_agreement_by_questionnaire(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze if agreement differs by questionnaire within constructs.
    
    Some questionnaires may have clearer items than others.
    """
    rater_cols = get_rater_columns(df)
    
    results = []
    
    for construct in df['construct'].unique():
        construct_df = df[df['construct'] == construct]
        
        for questionnaire in construct_df['questionnaire_name'].unique():
            q_df = construct_df[construct_df['questionnaire_name'] == questionnaire]
            ratings = q_df[rater_cols].values
            
            # Compute metrics
            item_modal_props = []
            item_entropies = []
            
            for i in range(ratings.shape[0]):
                row = ratings[i, :]
                valid = row[~np.isnan(row)].astype(int)
                if len(valid) >= 2:
                    mode_count = Counter(valid).most_common(1)[0][1]
                    item_modal_props.append(mode_count / len(valid))
                    item_entropies.append(compute_item_entropy(row))
            
            if len(item_modal_props) > 0:
                results.append({
                    'Construct': construct,
                    'Questionnaire': questionnaire,
                    'N Items': len(q_df),
                    'Mean Modal Proportion': np.mean(item_modal_props),
                    'SD Modal Proportion': np.std(item_modal_props),
                    'Mean Entropy': np.nanmean(item_entropies),
                    'Proportion High Agreement': sum(1 for p in item_modal_props if p >= 0.7) / len(item_modal_props)
                })
    
    return pd.DataFrame(results).sort_values(['Construct', 'Mean Modal Proportion'], 
                                              ascending=[True, False])


def analyze_rater_clusters(df: pd.DataFrame, construct: str) -> Dict:
    """
    Cluster raters based on their agreement patterns.
    
    This can reveal if there are "schools of thought" among raters.
    """
    agreement_matrix = compute_pairwise_rater_agreement(df, construct)
    
    # Handle NaN by filling with median agreement
    median_agreement = agreement_matrix.median().median()
    filled_matrix = agreement_matrix.fillna(median_agreement)
    
    # Convert agreement to distance
    distance_matrix = 1 - filled_matrix.values
    
    # Hierarchical clustering
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')
    
    # Cut tree to get 2-4 clusters
    clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')
    
    return {
        'agreement_matrix': agreement_matrix,
        'clusters': dict(zip(agreement_matrix.index, clusters)),
        'linkage_matrix': linkage_matrix,
        'n_clusters': len(set(clusters))
    }

# =============================================================================
# 6. Visualization Functions
# =============================================================================

def plot_rater_agreement_heatmap(df: pd.DataFrame, construct: str, 
                                 output_path: Optional[Path] = None):
    """Plot heatmap of pairwise rater agreement."""
    agreement_matrix = compute_pairwise_rater_agreement(df, construct)
    
    # Only include raters who rated this construct
    valid_raters = agreement_matrix.columns[agreement_matrix.notna().any()]
    agreement_matrix = agreement_matrix.loc[valid_raters, valid_raters]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    mask = np.triu(np.ones_like(agreement_matrix, dtype=bool), k=1)
    
    sns.heatmap(agreement_matrix, 
                mask=mask,
                annot=False, 
                cmap='RdYlGn',
                vmin=0.3, vmax=1.0,
                square=True,
                ax=ax)
    
    ax.set_title(f'Pairwise Rater Agreement - {construct.title()}', fontsize=14)
    ax.set_xlabel('Rater')
    ax.set_ylabel('Rater')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_item_difficulty_distribution(item_df: pd.DataFrame,
                                      output_path: Optional[Path] = None):
    """Plot distribution of item difficulty across constructs."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    constructs = item_df['Construct'].unique()
    
    for i, construct in enumerate(constructs):
        subset = item_df[item_df['Construct'] == construct]
        positions = np.random.normal(i, 0.1, len(subset))
        ax.scatter(positions, subset['Difficulty (1-modal)'], alpha=0.6, 
                  label=construct.title(), s=50)
    
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
               label='Difficulty threshold (0.5)')
    ax.set_xticks(range(len(constructs)))
    ax.set_xticklabels([c.title() for c in constructs], rotation=45, ha='right')
    ax.set_ylabel('Item Difficulty (1 - Modal Proportion)')
    ax.set_title('Item Difficulty Distribution by Construct')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_rater_dimension_preferences(pref_df: pd.DataFrame, construct: str,
                                    output_path: Optional[Path] = None):
    """Plot rater-specific dimension preferences compared to overall."""
    subset = pref_df[pref_df['Construct'] == construct].copy()
    
    if len(subset) == 0:
        print(f"No preference data for {construct}")
        return
    
    # Pivot for heatmap
    pivot = subset.pivot(index='Rater', columns='Dimension Label', values='Deviation (pp)')
    
    # Sort raters by total absolute deviation
    rater_order = pivot.abs().sum(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[rater_order]
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot) * 0.3)))
    
    sns.heatmap(pivot, cmap='RdBu_r', center=0, 
                annot=True, fmt='.0f', 
                cbar_kws={'label': 'Deviation from Overall (pp)'},
                ax=ax)
    
    ax.set_title(f'Rater Dimension Preferences - {construct.title()}\n'
                f'(Positive = rater uses dimension more than average)', fontsize=12)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Rater')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_full_analysis(filepath: Union[str, Path],
                     output_dir: Optional[Union[str, Path]] = None,
                     missing_value: int = -9,
                     generate_plots: bool = True) -> Dict:
    """
    Run the complete descriptive and exploratory analysis.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file with expert ratings
    output_dir : str or Path, optional
        Directory to save output files and plots
    missing_value : int
        Value indicating missing data (default: -9)
    generate_plots : bool
        Whether to generate visualization plots
    
    Returns
    -------
    Dict containing all analysis results
    """
    print("="*80)
    print("EXPERT SURVEY DESCRIPTIVE AND EXPLORATORY ANALYSIS")
    print("="*80)
    
    # Setup output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    df = load_data(filepath, missing_value)
    
    # Basic descriptives
    print("\n2. Computing basic descriptive statistics...")
    basic_stats = compute_basic_descriptives(df)
    response_dist = compute_response_distribution(df)
    
    print("\n   Summary by construct:")
    print(basic_stats.to_string(index=False))
    
    # Rater analysis
    print("\n3. Analyzing rater behavior...")
    rater_completion = compute_rater_completion(df)
    rater_agreement = compute_rater_majority_agreement(df)
    rater_none = compute_rater_none_usage(df)
    rater_outliers = identify_rater_outliers(df)
    rater_preferences = compute_rater_dimension_preferences(df)
    systematic_disagreers = find_systematic_disagreers(df)
    
    n_outliers = rater_outliers['Is Potential Outlier'].sum()
    print(f"   Found {n_outliers} potential outlier raters")
    
    if n_outliers > 0:
        print("\n   Potential outliers:")
        outlier_subset = rater_outliers[rater_outliers['Is Potential Outlier']]
        for _, row in outlier_subset.iterrows():
            print(f"   - {row['Rater']}: {row['Outlier Reasons']}")
    
    # Item analysis
    print("\n4. Analyzing item characteristics...")
    item_difficulty = compute_item_difficulty(df)
    confusion_pairs = find_confusing_dimension_pairs(df)
    
    n_problematic = item_difficulty['Is Problematic'].sum()
    print(f"   Found {n_problematic} problematic items (low agreement)")
    
    print("\n   Most difficult items (highest disagreement):")
    print(item_difficulty.head(10)[['Item ID', 'Construct', 'Modal Proportion', 
                                    'Difficulty (1-modal)', 'Problem Reasons']].to_string(index=False))
    
    # Construct comparison
    print("\n5. Comparing constructs...")
    construct_comparison = compare_constructs(df)
    questionnaire_comparison = analyze_agreement_by_questionnaire(df)
    
    print("\n   Construct summary:")
    print(construct_comparison.to_string(index=False))
    
    # Dimension confusion analysis
    print("\n6. Analyzing dimension confusions...")
    if len(confusion_pairs) > 0:
        print("\n   Most frequently confused dimension pairs:")
        print(confusion_pairs.head(10).to_string(index=False))
    else:
        print("   No significant dimension confusions found.")
    
    # Save results
    if output_dir:
        print(f"\n7. Saving results to {output_dir}...")
        
        basic_stats.to_csv(output_dir / 'basic_descriptives.csv', index=False)
        response_dist.to_csv(output_dir / 'response_distribution.csv', index=False)
        rater_completion.to_csv(output_dir / 'rater_completion.csv', index=False)
        rater_agreement.to_csv(output_dir / 'rater_majority_agreement.csv', index=False)
        rater_none.to_csv(output_dir / 'rater_none_usage.csv', index=False)
        rater_outliers.to_csv(output_dir / 'rater_outliers.csv', index=False)
        rater_preferences.to_csv(output_dir / 'rater_dimension_preferences.csv', index=False)
        systematic_disagreers.to_csv(output_dir / 'systematic_disagreers.csv', index=False)
        item_difficulty.to_csv(output_dir / 'item_difficulty.csv', index=False)
        confusion_pairs.to_csv(output_dir / 'dimension_confusions.csv', index=False)
        construct_comparison.to_csv(output_dir / 'construct_comparison.csv', index=False)
        questionnaire_comparison.to_csv(output_dir / 'questionnaire_comparison.csv', index=False)
        
        # Pairwise agreement matrices
        for construct in df['construct'].unique():
            pairwise = compute_pairwise_rater_agreement(df, construct)
            pairwise.to_csv(output_dir / f'pairwise_agreement_{construct}.csv')
        
        print("   CSV files saved.")
    
    # Generate plots
    if generate_plots and output_dir:
        print("\n8. Generating visualizations...")
        
        # Item difficulty
        plot_item_difficulty_distribution(item_difficulty,
                                         output_path=plots_dir / 'item_difficulty_distribution.png')
        
        # Per-construct plots
        for construct in df['construct'].unique():
            # Agreement heatmaps
            plot_rater_agreement_heatmap(df, construct,
                                        output_path=plots_dir / f'agreement_heatmap_{construct}.png')
            
            # Dimension preferences
            plot_rater_dimension_preferences(rater_preferences, construct,
                                            output_path=plots_dir / f'dimension_preferences_{construct}.png')
        
        print(f"   Plots saved to {plots_dir}/")
    
    # Compile results
    results = {
        'basic_stats': basic_stats,
        'response_distribution': response_dist,
        'rater_completion': rater_completion,
        'rater_agreement': rater_agreement,
        'rater_none_usage': rater_none,
        'rater_outliers': rater_outliers,
        'rater_preferences': rater_preferences,
        'systematic_disagreers': systematic_disagreers,
        'item_difficulty': item_difficulty,
        'confusion_pairs': confusion_pairs,
        'construct_comparison': construct_comparison,
        'questionnaire_comparison': questionnaire_comparison
    }
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return results


# =============================================================================
# Command Line Interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Descriptive and Exploratory Analysis for ENIGMA-PD Expert Survey"
    )
    parser.add_argument("filepath", type=str, 
                       help="Path to the CSV file with expert ratings")
    parser.add_argument("--output-dir", "-o", type=str, default="./descriptive_output",
                       help="Directory to save output files (default: ./descriptive_output)")
    parser.add_argument("--missing-value", type=int, default=-9,
                       help="Value indicating missing data (default: -9)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip plot generation")
    parser.add_argument("--dimension-config", type=str, default=None,
                       help="Path to a dimension_descriptions.yaml defining the "
                            f"constructs/dimensions to analyze (default: {DEFAULT_DIMENSION_CONFIG}). "
                            "Point this at your own copy to run on different constructs "
                            "without editing this script.")

    args = parser.parse_args()

    if args.dimension_config:
        DIMENSION_LABELS = load_dimension_labels(args.dimension_config)
        N_DIMENSIONS = {
            construct: sum(1 for dim_id in labels if dim_id != -1)
            for construct, labels in DIMENSION_LABELS.items()
        }

    results = run_full_analysis(
        filepath=args.filepath,
        output_dir=args.output_dir,
        missing_value=args.missing_value,
        generate_plots=not args.no_plots
    )