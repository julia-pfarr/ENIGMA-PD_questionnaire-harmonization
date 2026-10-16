"""
Expert Mappings Analysis for ENIGMA-PD Questionnaire Harmonization
==================================================================

This script implements Analysis Plan Step 3: Expert Mappings

For each item i, we calculate:
- P_i: The extent to which raters agree (proportion of agreeing rater pairs)
- Item assignment to the dimension with highest P_i

For overall reliability:
- P_j: Proportion of all assignments to the j-th category
- Fleiss' kappa (κ_F): Overall inter-rater reliability

Special value handling:
- -9: Missing data (NaN) - rater did not provide a rating
- -1: "Does not fit" - valid rating indicating item doesn't fit any dimension

References:
- Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters.
  Psychological Bulletin, 76, 378-382.
- Landis, J. R. & Koch, G. G. (1977). The measurement of observer agreement
  for categorical data. Biometrics, 33, 159-174.
"""

import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
import warnings
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent.parent
DEFAULT_DIMENSION_CONFIG = PROJECT_DIR / "input" / "shared" / "dimension_descriptions.yaml"


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class FleissKappaResult:
    """Results from Fleiss' kappa calculation."""
    kappa: float
    se: float
    z: float
    p_value: float
    ci_lower: float
    ci_upper: float
    P_bar: float  # Mean of P_i values (observed agreement)
    P_e: float    # Expected agreement by chance
    n_items: int
    n_raters: int
    n_categories: int
    interpretation: str
    
    def __repr__(self):
        return (f"Fleiss' κ = {self.kappa:.3f} (95% CI: [{self.ci_lower:.3f}, {self.ci_upper:.3f}])\n"
                f"SE = {self.se:.3f}, z = {self.z:.3f}, p = {self.p_value:.4f}\n"
                f"Observed agreement (P̄) = {self.P_bar:.3f}, Expected agreement (Pₑ) = {self.P_e:.3f}\n"
                f"N items = {self.n_items}, N raters = {self.n_raters}, N categories = {self.n_categories}\n"
                f"Interpretation: {self.interpretation}")


@dataclass
class ItemAssignment:
    """Assignment results for a single item."""
    item_id: str
    construct: str
    questionnaire: str
    question_text: str
    assigned_dimension: int
    assigned_dimension_label: Optional[str]
    P_i: float
    dimension_counts: Dict[int, int]
    dimension_proportions: Dict[int, float]
    n_valid_raters: int
    confidence: str
    is_ambiguous: bool
    competing_dimensions: List[int]


@dataclass
class ConstructResults:
    """Results for a single construct."""
    construct: str
    fleiss_kappa: FleissKappaResult
    item_assignments: List[ItemAssignment]
    dimension_labels: Dict[int, str]
    n_items: int
    n_raters: int
    items_per_dimension: Dict[int, int]
    mean_P_i: float
    median_P_i: float
    items_with_low_agreement: List[str]


# =============================================================================
# Dimension Definitions (from Table 2 of the registered report)
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
# Core Statistical Functions
# =============================================================================

def calculate_P_i(ratings_row: np.ndarray, n_raters: int) -> float:
    """
    Calculate P_i for a single item - the extent to which raters agree.
    
    P_i = (1 / (n * (n-1))) * sum_j(n_ij * (n_ij - 1))
    
    where n_ij is the number of raters who assigned item i to category j.
    """
    if n_raters < 2:
        return np.nan
    
    sum_term = np.sum(ratings_row * (ratings_row - 1))
    P_i = sum_term / (n_raters * (n_raters - 1))
    
    return P_i


def calculate_P_j(ratings_matrix: np.ndarray, n_raters_per_item: np.ndarray) -> np.ndarray:
    """
    Calculate P_j for each category - proportion of all assignments to category j.
    """
    category_totals = ratings_matrix.sum(axis=0)
    total_ratings = n_raters_per_item.sum()
    
    if total_ratings == 0:
        return np.zeros(ratings_matrix.shape[1])
    
    P_j = category_totals / total_ratings
    return P_j


def fleiss_kappa(ratings_matrix: np.ndarray, 
                n_raters_per_item: Optional[np.ndarray] = None,
                confidence_level: float = 0.95) -> FleissKappaResult:
    """
    Calculate Fleiss' kappa for multiple raters and nominal categories.
    
    κ = (P̄ - P_e) / (1 - P_e)
    """
    n_items, n_categories = ratings_matrix.shape
    
    if n_raters_per_item is None:
        n_raters_per_item = ratings_matrix.sum(axis=1)
    
    # Check for items with fewer than 2 raters
    valid_items_mask = n_raters_per_item >= 2
    if not valid_items_mask.all():
        warnings.warn(f"{(~valid_items_mask).sum()} items have fewer than 2 raters "
                     "and will be excluded from kappa calculation.")
        ratings_matrix = ratings_matrix[valid_items_mask]
        n_raters_per_item = n_raters_per_item[valid_items_mask]
        n_items = ratings_matrix.shape[0]
    
    if n_items == 0:
        raise ValueError("No items with at least 2 raters")
    
    # Calculate P_i for each item
    P_i_values = np.array([
        calculate_P_i(ratings_matrix[i], int(n_raters_per_item[i]))
        for i in range(n_items)
    ])
    
    P_bar = np.nanmean(P_i_values)
    P_j_values = calculate_P_j(ratings_matrix, n_raters_per_item)
    P_e = np.sum(P_j_values ** 2)
    
    if abs(1 - P_e) < 1e-10:
        kappa = 1.0 if P_bar > 0.999 else 0.0
    else:
        kappa = (P_bar - P_e) / (1 - P_e)
    
    # Calculate standard error
    n_bar = np.mean(n_raters_per_item)
    sum_pj_cubed = np.sum(P_j_values ** 3)
    
    if abs(1 - P_e) < 1e-10:
        se = 0.0
    else:
        var_kappa = (2 / (n_items * n_bar * (n_bar - 1))) * \
                   ((P_e - (2 * n_bar - 3) * P_e**2 + 2 * (n_bar - 2) * sum_pj_cubed) / 
                    (1 - P_e)**2)
        se = np.sqrt(max(0, var_kappa))
    
    if se > 0:
        z = kappa / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        z = np.inf if kappa > 0 else -np.inf if kappa < 0 else 0
        p_value = 0.0 if kappa != 0 else 1.0
    
    z_crit = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    ci_lower = kappa - z_crit * se
    ci_upper = kappa + z_crit * se
    
    # Interpretation (Landis & Koch, 1977)
    if kappa < 0:
        interpretation = "Poor (less than chance agreement)"
    elif kappa < 0.20:
        interpretation = "Slight agreement"
    elif kappa < 0.40:
        interpretation = "Fair agreement"
    elif kappa < 0.60:
        interpretation = "Moderate agreement"
    elif kappa < 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"
    
    return FleissKappaResult(
        kappa=kappa, se=se, z=z, p_value=p_value,
        ci_lower=ci_lower, ci_upper=ci_upper,
        P_bar=P_bar, P_e=P_e, n_items=n_items,
        n_raters=int(np.median(n_raters_per_item)),
        n_categories=n_categories, interpretation=interpretation
    )


# =============================================================================
# Data Processing Functions
# =============================================================================

def load_expert_ratings(filepath: Union[str, Path], 
                       missing_value: int = -9) -> pd.DataFrame:
    """Load expert ratings from CSV file."""
    df = pd.read_csv(filepath)
    rater_cols = [col for col in df.columns if col.startswith('rater_')]
    
    for col in rater_cols:
        df[col] = df[col].replace(missing_value, np.nan)
    
    return df


def create_ratings_matrix(df: pd.DataFrame, 
                         construct: str,
                         does_not_fit_value: int = -1) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    """Create a ratings matrix for a specific construct."""
    df_construct = df[df['construct'] == construct].copy()
    
    if len(df_construct) == 0:
        raise ValueError(f"No items found for construct: {construct}")
    
    rater_cols = [col for col in df.columns if col.startswith('rater_')]
    
    # Convert to float to handle NaN properly
    all_ratings = df_construct[rater_cols].values.astype(float).flatten()
    all_ratings = all_ratings[~np.isnan(all_ratings)]
    categories = sorted(set(int(r) for r in all_ratings))
    
    if does_not_fit_value in categories:
        categories = [c for c in categories if c != does_not_fit_value] + [does_not_fit_value]
    
    n_items = len(df_construct)
    n_categories = len(categories)
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    ratings_matrix = np.zeros((n_items, n_categories), dtype=int)
    n_raters_per_item = np.zeros(n_items, dtype=int)
    item_ids = df_construct['question_number'].tolist()
    
    for i, (_, row) in enumerate(df_construct.iterrows()):
        ratings = row[rater_cols].values.astype(float)
        valid_ratings = ratings[~np.isnan(ratings)]
        n_raters_per_item[i] = len(valid_ratings)
        
        for rating in valid_ratings:
            cat_idx = cat_to_idx[int(rating)]
            ratings_matrix[i, cat_idx] += 1
    
    return ratings_matrix, n_raters_per_item, item_ids, categories


def assign_items_to_dimensions(df: pd.DataFrame,
                               construct: str,
                               ratings_matrix: np.ndarray,
                               n_raters_per_item: np.ndarray,
                               item_ids: List[str],
                               categories: List[int],
                               dimension_labels: Optional[Dict[int, str]] = None,
                               ambiguity_threshold: float = 0.10) -> List[ItemAssignment]:
    """Assign each item to the dimension with highest expert agreement."""
    df_construct = df[df['construct'] == construct].copy()
    assignments = []
    
    for i, (_, row) in enumerate(df_construct.iterrows()):
        counts = ratings_matrix[i]
        n_valid = n_raters_per_item[i]
        
        if n_valid > 0:
            proportions = counts / n_valid
        else:
            proportions = np.zeros_like(counts, dtype=float)
        
        dim_counts = {categories[j]: int(counts[j]) for j in range(len(categories))}
        dim_proportions = {categories[j]: float(proportions[j]) for j in range(len(categories))}
        
        max_idx = np.argmax(counts)
        assigned_dim = categories[max_idx]
        max_proportion = proportions[max_idx]
        
        P_i = calculate_P_i(counts, n_valid) if n_valid >= 2 else np.nan
        
        sorted_proportions = sorted(proportions, reverse=True)
        competing_dims = []
        is_ambiguous = False
        
        if len(sorted_proportions) > 1:
            if sorted_proportions[0] - sorted_proportions[1] <= ambiguity_threshold:
                is_ambiguous = True
                threshold = sorted_proportions[0] - ambiguity_threshold
                for j, prop in enumerate(proportions):
                    if prop >= threshold:
                        competing_dims.append(categories[j])
        
        if n_valid < 2:
            confidence = "insufficient_data"
        elif max_proportion >= 0.80:
            confidence = "high"
        elif max_proportion >= 0.60:
            confidence = "moderate"
        elif max_proportion >= 0.40:
            confidence = "low"
        else:
            confidence = "very_low"
        
        dim_label = None
        if dimension_labels and assigned_dim in dimension_labels:
            dim_label = dimension_labels[assigned_dim]
        
        assignments.append(ItemAssignment(
            item_id=item_ids[i],
            construct=construct,
            questionnaire=row['questionnaire_name'],
            question_text=row.get('question_text', ''),  # stripped from public item files (copyright)
            assigned_dimension=assigned_dim,
            assigned_dimension_label=dim_label,
            P_i=P_i,
            dimension_counts=dim_counts,
            dimension_proportions=dim_proportions,
            n_valid_raters=n_valid,
            confidence=confidence,
            is_ambiguous=is_ambiguous,
            competing_dimensions=competing_dims
        ))
    
    return assignments


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_construct(df: pd.DataFrame, 
                     construct: str,
                     dimension_labels: Optional[Dict[int, str]] = None,
                     does_not_fit_value: int = -1,
                     confidence_level: float = 0.95) -> ConstructResults:
    """Perform complete expert mapping analysis for a single construct."""
    ratings_matrix, n_raters_per_item, item_ids, categories = create_ratings_matrix(
        df, construct, does_not_fit_value
    )
    
    kappa_result = fleiss_kappa(ratings_matrix, n_raters_per_item, confidence_level)
    
    assignments = assign_items_to_dimensions(
        df, construct, ratings_matrix, n_raters_per_item,
        item_ids, categories, dimension_labels
    )
    
    P_i_values = [a.P_i for a in assignments if not np.isnan(a.P_i)]
    mean_P_i = np.mean(P_i_values) if P_i_values else np.nan
    median_P_i = np.median(P_i_values) if P_i_values else np.nan
    
    items_per_dimension = {}
    for a in assignments:
        dim = a.assigned_dimension
        items_per_dimension[dim] = items_per_dimension.get(dim, 0) + 1
    
    low_agreement_items = [a.item_id for a in assignments if a.P_i < 0.40]
    
    return ConstructResults(
        construct=construct,
        fleiss_kappa=kappa_result,
        item_assignments=assignments,
        dimension_labels=dimension_labels or {},
        n_items=len(assignments),
        n_raters=kappa_result.n_raters,
        items_per_dimension=items_per_dimension,
        mean_P_i=mean_P_i,
        median_P_i=median_P_i,
        items_with_low_agreement=low_agreement_items
    )


def analyze_all_constructs(df: pd.DataFrame,
                          dimension_labels: Optional[Dict[str, Dict[int, str]]] = None,
                          does_not_fit_value: int = -1,
                          confidence_level: float = 0.95) -> Dict[str, ConstructResults]:
    """Analyze all constructs in the dataset."""
    constructs = df['construct'].unique()
    results = {}
    
    for construct in constructs:
        labels = None
        if dimension_labels and construct in dimension_labels:
            labels = dimension_labels[construct]
        
        print(f"\nAnalyzing construct: {construct}")
        results[construct] = analyze_construct(
            df, construct, labels, does_not_fit_value, confidence_level
        )
    
    return results


# =============================================================================
# Output and Reporting Functions
# =============================================================================

def create_summary_table(results: Dict[str, ConstructResults]) -> pd.DataFrame:
    """Create a summary table of Fleiss' kappa results for all constructs."""
    rows = []
    for construct, res in results.items():
        k = res.fleiss_kappa
        rows.append({
            'Construct': construct,
            'N Items': res.n_items,
            'N Raters': res.n_raters,
            'N Categories': k.n_categories,
            'Fleiss κ': round(k.kappa, 3),
            '95% CI Lower': round(k.ci_lower, 3),
            '95% CI Upper': round(k.ci_upper, 3),
            'SE': round(k.se, 3),
            'p-value': f"{k.p_value:.4f}" if k.p_value >= 0.0001 else "<0.0001",
            'P̄ (Observed)': round(k.P_bar, 3),
            'Pₑ (Expected)': round(k.P_e, 3),
            'Interpretation': k.interpretation,
            'Mean P_i': round(res.mean_P_i, 3),
            'Items Low Agreement': len(res.items_with_low_agreement)
        })
    
    return pd.DataFrame(rows)


def create_item_assignments_table(results: Dict[str, ConstructResults]) -> pd.DataFrame:
    """Create a detailed table of all item assignments."""
    rows = []
    for construct, res in results.items():
        for a in res.item_assignments:
            prop_str = "; ".join([
                f"Dim {k}: {v:.0%}" for k, v in sorted(a.dimension_proportions.items())
                if v > 0
            ])
            
            rows.append({
                'Construct': construct,
                'Questionnaire': a.questionnaire,
                'Item ID': a.item_id,
                'Question Text': a.question_text[:100] + "..." if len(a.question_text) > 100 else a.question_text,
                'Assigned Dimension': a.assigned_dimension,
                'Dimension Label': a.assigned_dimension_label or "",
                'P_i': round(a.P_i, 3) if not np.isnan(a.P_i) else "N/A",
                'N Valid Raters': a.n_valid_raters,
                'Confidence': a.confidence,
                'Is Ambiguous': a.is_ambiguous,
                'Competing Dimensions': str(a.competing_dimensions) if a.competing_dimensions else "",
                'Rating Distribution': prop_str
            })
    
    return pd.DataFrame(rows)


def create_dimension_distribution_table(results: Dict[str, ConstructResults]) -> pd.DataFrame:
    """Create a table showing the distribution of items across dimensions."""
    rows = []
    for construct, res in results.items():
        for dim, count in sorted(res.items_per_dimension.items()):
            label = res.dimension_labels.get(dim, f"Dimension {dim}")
            rows.append({
                'Construct': construct,
                'Dimension': dim,
                'Dimension Label': label,
                'N Items': count,
                'Percentage': round(100 * count / res.n_items, 1)
            })
    
    return pd.DataFrame(rows)


def print_detailed_report(results: Dict[str, ConstructResults]):
    """Print a detailed report of the analysis results."""
    print("\n" + "="*80)
    print("EXPERT MAPPINGS ANALYSIS - DETAILED REPORT")
    print("="*80)
    
    for construct, res in results.items():
        print(f"\n{'='*80}")
        print(f"CONSTRUCT: {construct.upper()}")
        print("="*80)
        
        print(f"\n{res.fleiss_kappa}")
        
        print(f"\n--- Items per Dimension ---")
        for dim, count in sorted(res.items_per_dimension.items()):
            label = res.dimension_labels.get(dim, f"Dimension {dim}")
            pct = 100 * count / res.n_items
            print(f"  {label}: {count} items ({pct:.1f}%)")
        
        if res.items_with_low_agreement:
            print(f"\n--- Items with Low Agreement (P_i < 0.40) ---")
            for item_id in res.items_with_low_agreement:
                item = next(a for a in res.item_assignments if a.item_id == item_id)
                print(f"  {item_id}: P_i = {item.P_i:.3f}, assigned to {item.assigned_dimension_label or item.assigned_dimension}")
        
        ambiguous = [a for a in res.item_assignments if a.is_ambiguous]
        if ambiguous:
            print(f"\n--- Ambiguous Items (competing dimensions within 10%) ---")
            for a in ambiguous:
                print(f"  {a.item_id}: Competing dimensions: {a.competing_dimensions}")


def save_results(results: Dict[str, ConstructResults], 
                output_dir: Union[str, Path] = "."):
    """Save all results to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_df = create_summary_table(results)
    summary_df.to_csv(output_dir / "fleiss_kappa_summary.csv", index=False)
    
    items_df = create_item_assignments_table(results)
    items_df.to_csv(output_dir / "item_assignments.csv", index=False)
    
    dist_df = create_dimension_distribution_table(results)
    dist_df.to_csv(output_dir / "dimension_distribution.csv", index=False)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"  - fleiss_kappa_summary.csv")
    print(f"  - item_assignments.csv")
    print(f"  - dimension_distribution.csv")


# =============================================================================
# Main Entry Point
# =============================================================================

def main(filepath: Union[str, Path],
         output_dir: Optional[Union[str, Path]] = None,
         missing_value: int = -9,
         does_not_fit_value: int = -1,
         confidence_level: float = 0.95,
         print_report: bool = True,
         save_outputs: bool = True,
         dimension_config_path: Optional[Union[str, Path]] = None) -> Dict[str, ConstructResults]:
    """
    Main function to run the complete expert mappings analysis.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file with expert ratings
    output_dir : str or Path, optional
        Directory to save output files
    missing_value : int
        Value indicating missing data (default: -9)
    does_not_fit_value : int
        Value indicating "does not fit" (default: -1)
    confidence_level : float
        Confidence level for CIs (default: 0.95)
    print_report : bool
        Whether to print detailed report (default: True)
    save_outputs : bool
        Whether to save results to CSV files (default: True)
    dimension_config_path : str or Path, optional
        Path to a dimension_descriptions.yaml (default: DEFAULT_DIMENSION_CONFIG).
        Point this at your own copy to run on different constructs/dimensions
        without editing this script.

    Returns
    -------
    Dict[str, ConstructResults]
        Results for each construct
    """
    dimension_labels = (load_dimension_labels(dimension_config_path)
                        if dimension_config_path is not None else DIMENSION_LABELS)

    print(f"Loading data from: {filepath}")
    df = load_expert_ratings(filepath, missing_value)

    rater_cols = [col for col in df.columns if col.startswith('rater_')]
    print(f"Found {len(rater_cols)} raters: {rater_cols}")
    print(f"Total items: {len(df)}")
    print(f"Constructs: {df['construct'].unique().tolist()}")

    results = analyze_all_constructs(
        df,
        dimension_labels=dimension_labels,
        does_not_fit_value=does_not_fit_value,
        confidence_level=confidence_level
    )
    
    if print_report:
        print_detailed_report(results)
    
    if save_outputs and output_dir:
        save_results(results, output_dir)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Expert Mappings Analysis for ENIGMA-PD Questionnaire Harmonization"
    )
    parser.add_argument("filepath", type=str, help="Path to the CSV file with expert ratings")
    parser.add_argument("--output-dir", "-o", type=str, default="./output",
                       help="Directory to save output files (default: ./output)")
    parser.add_argument("--missing-value", type=int, default=-9,
                       help="Value indicating missing data (default: -9)")
    parser.add_argument("--does-not-fit-value", type=int, default=-1,
                       help="Value indicating 'does not fit' (default: -1)")
    parser.add_argument("--confidence-level", type=float, default=0.95,
                       help="Confidence level for CIs (default: 0.95)")
    parser.add_argument("--no-print", action="store_true",
                       help="Suppress detailed report printing")
    parser.add_argument("--no-save", action="store_true",
                       help="Do not save output files")
    parser.add_argument("--dimension-config", type=str, default=None,
                       help="Path to a dimension_descriptions.yaml defining the "
                            f"constructs/dimensions to analyze (default: {DEFAULT_DIMENSION_CONFIG}). "
                            "Point this at your own copy to run on different constructs "
                            "without editing this script.")

    args = parser.parse_args()

    results = main(
        filepath=args.filepath,
        output_dir=args.output_dir,
        missing_value=args.missing_value,
        does_not_fit_value=args.does_not_fit_value,
        confidence_level=args.confidence_level,
        print_report=not args.no_print,
        save_outputs=not args.no_save,
        dimension_config_path=args.dimension_config
    )