"""
============================================================================
MULTI-KAPPA SIMULATION
Sample Size Determination for Expert Rater Survey
Array-compatible version for HPC cluster processing
============================================================================

NOTE ON RUNTIME: analyze_precision_bootstrap() alone runs
10 n_raters values x 1000 iterations x 500 bootstrap replicates x
len(kappa_range) kappa values through calculate_fleiss_kappa_from_ratings(),
which loops item-by-item in pure Python -- tens of millions of calls per
construct. This was written to be run per-construct on an HPC array job,
not interactively; expect a single `--construct` run to take from several
hours up to multiple days on a laptop.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path
import warnings
import argparse
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

survey_structure = {
    'Depression': {
        'dimensions': ['Mood_Affective', 'Cognitive_SelfPerception', 
                      'Somatic_Vegetative', 'Activity_Interest', 'Anxiety_Distress', 'None'],
        'n_items': 73
    },
    'Anxiety': {
        'dimensions': ['Somatic', 'Cognitive', 'None'],
        'n_items': 83
    },
    'Psychosis': {
        'dimensions': ['Hallucinations', 'Delusions', 'None'],
        'n_items': 37
    },
    'Apathy': {
        'dimensions': ['Cognitive', 'Behavioral', 'Affective', 'None'],
        'n_items': 89
    },
    'ICD': {
        'dimensions': ['Gambling', 'Hypersexuality', 'Buying', 
                      'Eating', 'Punding', 'DDS', 'None'],
        'n_items': 27
    },
    'Sleep': {
        'dimensions': ['Daytime_Sleepiness', 'Nocturnal_Disturbances', 
                      'REM_Behavior', 'Sleep_Breathing', 'None'],
        'n_items': 100
    }
}

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_fleiss_kappa_from_ratings(ratings, n_categories):
    """Calculate Fleiss' kappa from raw ratings array."""
    n_items, n_raters = ratings.shape
    P_values = []
    all_ratings = ratings.flatten()
    category_props = np.array([np.mean(all_ratings == k) for k in range(n_categories)])
    P_e = np.sum(category_props ** 2)
    
    for i in range(n_items):
        item_ratings = ratings[i, :]
        counts = np.array([np.sum(item_ratings == k) for k in range(n_categories)])
        sum_squares = np.sum(counts ** 2)
        P_i = (sum_squares - n_raters) / (n_raters * (n_raters - 1))
        P_values.append(P_i)
    
    P_bar = np.mean(P_values)
    
    if P_e >= 1.0:
        return 1.0 if P_bar >= 1.0 else 0.0
    
    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa

def generate_dimensional_distribution(n_items, n_categories, balance='random'):
    """
    Generate item-to-dimension assignments with various balance levels.
    
    Parameters:
    -----------
    n_items : int
        Total number of items
    n_categories : int
        Number of dimensions
    balance : str or array
        - 'even': Distribute items evenly (balanced)
        - 'random': Random distribution using Dirichlet
        - 'imbalanced': Deliberately create imbalance
        - array: Explicit proportions (must sum to 1.0)
    
    Returns:
    --------
    true_dimensions : array
        Assignment of each item to a dimension
    distribution : dict
        Items per dimension for debugging
    """
    if isinstance(balance, (list, np.ndarray)):
        # Explicit proportions provided
        proportions = np.array(balance)
        proportions = proportions / proportions.sum()  # Normalize
    elif balance == 'even':
        # Even distribution
        proportions = np.ones(n_categories) / n_categories
    elif balance == 'imbalanced':
        # Deliberately imbalanced: one large dimension, others smaller
        proportions = np.array([0.5] + [0.5/(n_categories-1)]*(n_categories-1))
    elif balance == 'random':
        # Random distribution using Dirichlet
        # Alpha = 2 creates moderate variability 
        proportions = np.random.dirichlet([2.0] * n_categories)
    else:
        raise ValueError(f"Unknown balance type: {balance}")
    
    # Convert proportions to item counts
    item_counts = np.round(proportions * n_items).astype(int)
    
    # Adjust for rounding errors
    diff = n_items - item_counts.sum()
    if diff > 0:
        # Add missing items to dimensions with fewest items
        for _ in range(diff):
            item_counts[np.argmin(item_counts)] += 1
    elif diff < 0:
        # Remove excess items from dimensions with most items
        for _ in range(-diff):
            item_counts[np.argmax(item_counts)] -= 1
    
    # Create assignment array
    true_dimensions = []
    for dim in range(n_categories):
        true_dimensions.extend([dim] * item_counts[dim])
    
    true_dimensions = np.array(true_dimensions)
    np.random.shuffle(true_dimensions)  # Randomize order
    
    distribution = {f'dim_{i}': count for i, count in enumerate(item_counts)}
    
    return true_dimensions, distribution


def simulate_expert_ratings_direct(n_raters, n_items, n_categories, target_kappa=0.50, 
                                   dimension_balance='random'):
    """
    Simulate expert ratings for dimensional classification.
    
    Each item has a TRUE dimension assignment (ground truth).
    Experts attempt to classify items into dimensions with agreement level = target_kappa.
    
    Dimensional distributions (even, random, imbalanced) to model uncertainty 
    about true item distribution across dimensions.
    
    Parameters:
    -----------
    n_raters : int
        Number of expert raters
    n_items : int
        Number of items to classify
    n_categories : int
        Number of dimensions/categories
    target_kappa : float
        Target inter-rater agreement (Fleiss' kappa)
    dimension_balance : str or array
        How items are distributed across dimensions:
        - 'even': Balanced distribution
        - 'random': Random distribution (DEFAULT - models uncertainty)
        - 'imbalanced': Deliberately imbalanced
        - array: Explicit proportions
    
    Returns:
    --------
    ratings : array (n_items × n_raters)
        Expert dimensional classifications
    """
    # Calculate required agreement probability
    P_e = 1.0 / n_categories
    required_P = target_kappa * (1 - P_e) + P_e
    agree_prob = np.sqrt(required_P) if required_P >= 0 else 0.0
    agree_prob = np.clip(agree_prob, 1.0/n_categories, 1.0)
    
    # Generate dimensional assignments
    true_dimensions, _ = generate_dimensional_distribution(n_items, n_categories, dimension_balance)
    
    # Initialize ratings matrix
    ratings = np.zeros((n_items, n_raters), dtype=int)
    
    # Generate expert ratings
    # For each item, establish a "consensus" classification
    for i in range(n_items):
        true_dim = true_dimensions[i]
        
        # The "consensus" dimension is usually (but not always) the true dimension
        # This models that experts usually agree on clear items, but may consensually 
        # misclassify ambiguous items
        if np.random.random() < 0.8:  # 80% of items: consensus = true dimension
            consensus_dim = true_dim
        else:  # 20% of items: consensus is another dimension (ambiguous items)
            other_dims = [d for d in range(n_categories) if d != true_dim]
            consensus_dim = np.random.choice(other_dims) if other_dims else true_dim
        
        # Generate rater classifications based on consensus
        for r in range(n_raters):
            if np.random.random() < agree_prob:
                # Agree with consensus
                ratings[i, r] = consensus_dim
            else:
                # Disagree - choose randomly from other dimensions
                other_dims = [d for d in range(n_categories) if d != consensus_dim]
                ratings[i, r] = np.random.choice(other_dims) if other_dims else consensus_dim
    
    return ratings

def analyze_stability(construct_info, true_kappa, max_raters, n_iterations):
    """Analyze stability across increasing number of raters."""
    n_items = construct_info['n_items']
    n_categories = len(construct_info['dimensions'])
    results = []
    
    for iteration in tqdm(range(n_iterations), desc=f"    κ={true_kappa:.2f} stability", leave=False):
        ratings = simulate_expert_ratings_direct(max_raters, n_items, n_categories, true_kappa)
        kappas_cumulative = []
        
        for n in range(3, max_raters + 1):
            ratings_subset = ratings[:, :n]
            kappa = calculate_fleiss_kappa_from_ratings(ratings_subset, n_categories)
            kappas_cumulative.append(kappa)
        
        for idx, n in enumerate(range(3, max_raters + 1)):
            kappa = kappas_cumulative[idx]
            delta_kappa = abs(kappa - kappas_cumulative[idx - 1]) if idx > 0 else np.nan
            delta_from_final = abs(kappa - kappas_cumulative[-1])
            
            results.append({
                'iteration': iteration,
                'n_raters': n,
                'kappa': kappa,
                'delta_kappa': delta_kappa,
                'delta_from_final': delta_from_final
            })
    
    results_df = pd.DataFrame(results)
    stability_summary = results_df.groupby('n_raters').agg({
        'kappa': ['mean', 'std'],
        'delta_kappa': ['mean', 'std'],
        'delta_from_final': ['mean', 'std']
    }).reset_index()
    
    stability_summary.columns = [
        'n_raters', 'mean_kappa', 'sd_kappa',
        'mean_delta', 'sd_delta',
        'mean_delta_from_final', 'sd_delta_from_final'
    ]
    
    return stability_summary

def analyze_precision_bootstrap(construct_info, n_raters_range, true_kappa, n_iterations, n_bootstrap):
    """Analyze precision with bootstrap confidence intervals."""
    n_items = construct_info['n_items']
    n_categories = len(construct_info['dimensions'])
    results = []
    
    for n_raters in tqdm(n_raters_range, desc=f"    κ={true_kappa:.2f} precision", leave=False):
        ci_widths = []
        
        for iteration in range(n_iterations):
            ratings = simulate_expert_ratings_direct(n_raters, n_items, n_categories, true_kappa)
            kappas_boot = []
            
            for _ in range(n_bootstrap):
                boot_indices = np.random.choice(n_items, size=n_items, replace=True)
                ratings_boot = ratings[boot_indices, :]
                kappa_boot = calculate_fleiss_kappa_from_ratings(ratings_boot, n_categories)
                kappas_boot.append(kappa_boot)
            
            ci_lower = np.percentile(kappas_boot, 2.5)
            ci_upper = np.percentile(kappas_boot, 97.5)
            ci_widths.append(ci_upper - ci_lower)
        
        results.append({
            'n_raters': n_raters,
            'mean_ci_width': np.mean(ci_widths),
            'sd_ci_width': np.std(ci_widths),
            'median_ci_width': np.median(ci_widths)
        })
    
    return pd.DataFrame(results)

def analyze_replication_variability(construct_info, n_raters_range, true_kappa, n_iterations):
    """Analyze replication variability across independent samples."""
    n_items = construct_info['n_items']
    n_categories = len(construct_info['dimensions'])
    results = []
    
    for n_raters in tqdm(n_raters_range, desc=f"    κ={true_kappa:.2f} replication", leave=False):
        kappas_all = []
        
        for iteration in range(n_iterations):
            ratings = simulate_expert_ratings_direct(n_raters, n_items, n_categories, true_kappa)
            kappa = calculate_fleiss_kappa_from_ratings(ratings, n_categories)
            kappas_all.append(kappa)
        
        mean_kappa = np.mean(kappas_all)
        sd_across_samples = np.std(kappas_all)
        cv = sd_across_samples / mean_kappa if mean_kappa > 0 else np.nan
        
        ci_lower_95 = np.percentile(kappas_all, 2.5)
        ci_upper_95 = np.percentile(kappas_all, 97.5)
        range_95 = ci_upper_95 - ci_lower_95
        
        ci_lower_90 = np.percentile(kappas_all, 5)
        ci_upper_90 = np.percentile(kappas_all, 95)
        range_90 = ci_upper_90 - ci_lower_90
        
        results.append({
            'n_raters': n_raters,
            'mean_kappa': mean_kappa,
            'sd_across_samples': sd_across_samples,
            'cv': cv,
            'range_95': range_95,
            'range_90': range_90
        })
    
    return pd.DataFrame(results)

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_stability_comparison(all_stability_df, construct_name, save_path):
    """Create comparison plots for stability across kappa values."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        '0.40': '#e74c3c',
        '0.50': '#3498db', 
        '0.60': '#2ecc71',
        '0.70': '#9b59b6'
    }
    
    # Kappa estimates
    ax = axes[0, 0]
    for true_kappa in sorted(all_stability_df['true_kappa'].unique()):
        data = all_stability_df[all_stability_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['mean_kappa'],
               marker='o', linewidth=2, markersize=4,
               label=f'True κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('Mean Kappa Estimate')
    ax.set_title('Kappa Estimates by True Agreement Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mean change
    ax = axes[0, 1]
    for true_kappa in sorted(all_stability_df['true_kappa'].unique()):
        data = all_stability_df[all_stability_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['mean_delta'],
               marker='o', linewidth=2, markersize=4,
               label=f'κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.axhline(y=0.01, linestyle='--', color='black', alpha=0.5, label='Stable (< 0.01)')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('Mean Change in κ')
    ax.set_title('Estimate Stability Across Agreement Levels')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # SD of estimates
    ax = axes[1, 0]
    for true_kappa in sorted(all_stability_df['true_kappa'].unique()):
        data = all_stability_df[all_stability_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['sd_kappa'],
               marker='o', linewidth=2, markersize=4,
               label=f'κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('SD of Kappa Estimates')
    ax.set_title('Precision Across Agreement Levels')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Convergence
    ax = axes[1, 1]
    for true_kappa in sorted(all_stability_df['true_kappa'].unique()):
        data = all_stability_df[all_stability_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['mean_delta_from_final'],
               marker='o', linewidth=2, markersize=4,
               label=f'κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.axhline(y=0.02, linestyle='--', color='black', alpha=0.5, label='Within 0.02')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('Distance from Final Estimate')
    ax.set_title('Convergence Across Agreement Levels')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.suptitle(f'Stability Analysis Comparison: {construct_name}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

def plot_precision_comparison(all_precision_df, construct_name, save_path):
    """Create comparison plot for precision across kappa values."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {
        '0.40': '#e74c3c',
        '0.50': '#3498db',
        '0.60': '#2ecc71', 
        '0.70': '#9b59b6'
    }
    
    for true_kappa in sorted(all_precision_df['true_kappa'].unique()):
        data = all_precision_df[all_precision_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['mean_ci_width'],
               marker='o', linewidth=2.5, markersize=6,
               label=f'True κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.axhline(y=0.10, linestyle='--', color='#e74c3c', linewidth=2, 
              alpha=0.7, label='Target (±0.05)')
    ax.axhline(y=0.15, linestyle='--', color='#f39c12', linewidth=2,
              alpha=0.7, label='Acceptable (±0.075)')
    
    ax.set_xlabel('Number of Raters', fontsize=12)
    ax.set_ylabel('95% CI Width', fontsize=12)
    ax.set_title(f'Precision Comparison: {construct_name}\nBootstrap CIs Across Agreement Levels',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

def plot_replication_comparison(all_replication_df, construct_name, save_path):
    """Create comparison plots for replication variability across kappa values."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        '0.40': '#e74c3c',
        '0.50': '#3498db',
        '0.60': '#2ecc71',
        '0.70': '#9b59b6'
    }
    
    # SD across samples
    ax = axes[0, 0]
    for true_kappa in sorted(all_replication_df['true_kappa'].unique()):
        data = all_replication_df[all_replication_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['sd_across_samples'],
               marker='o', linewidth=2, markersize=4,
               label=f'κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.axhline(y=0.05, linestyle='--', color='black', alpha=0.5, label='Target (≤0.05)')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('SD Across Samples')
    ax.set_title('Between-Sample Variability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 95% range
    ax = axes[0, 1]
    for true_kappa in sorted(all_replication_df['true_kappa'].unique()):
        data = all_replication_df[all_replication_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['range_95'],
               marker='o', linewidth=2, markersize=4,
               label=f'κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.axhline(y=0.10, linestyle='--', color='black', alpha=0.5, label='Target (≤0.10)')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('95% Range')
    ax.set_title('Replication Range (95%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 90% range
    ax = axes[1, 0]
    for true_kappa in sorted(all_replication_df['true_kappa'].unique()):
        data = all_replication_df[all_replication_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['range_90'],
               marker='o', linewidth=2, markersize=4,
               label=f'κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.axhline(y=0.08, linestyle='--', color='black', alpha=0.5, label='Target (≤0.08)')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('90% Range')
    ax.set_title('Replication Range (90%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # CV
    ax = axes[1, 1]
    for true_kappa in sorted(all_replication_df['true_kappa'].unique()):
        data = all_replication_df[all_replication_df['true_kappa'] == true_kappa]
        ax.plot(data['n_raters'], data['cv'],
               marker='o', linewidth=2, markersize=4,
               label=f'κ = {true_kappa:.2f}',
               color=colors[f'{true_kappa:.2f}'])
    
    ax.axhline(y=0.10, linestyle='--', color='black', alpha=0.5, label='Target (≤0.10)')
    ax.set_xlabel('Number of Raters')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Relative Variability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.suptitle(f'Replication Variability Comparison: {construct_name}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_multi_kappa_analyses(construct_name, true_kappa_range=None, random_seed=None):
    """
    Run analyses across multiple true kappa values.
    """
    if true_kappa_range is None:
        true_kappa_range = [0.40, 0.50, 0.60, 0.70]
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    print("\n" + "="*70)
    print(f"MULTI-KAPPA ANALYSES: {construct_name}")
    print(f"Testing κ = {true_kappa_range}")
    print(f"Random seed: {random_seed}")
    print("="*70 + "\n")
    
    construct_info = survey_structure[construct_name]
    output_dir = Path(f'analysis_multikappa_{construct_name.lower()}')
    output_dir.mkdir(exist_ok=True)
    
    all_stability_data = []
    all_precision_data = []
    all_replication_data = []
    all_recommendations = []
    
    # Run analyses for each kappa value
    for true_kappa in true_kappa_range:
        print(f"\n  Analyzing κ = {true_kappa:.2f}...")
        
        # Stability
        stability_df = analyze_stability(
            construct_info=construct_info,
            true_kappa=true_kappa,
            max_raters=50,
            n_iterations=1000
        )
        stability_df['true_kappa'] = true_kappa
        all_stability_data.append(stability_df)
        
        # Precision
        precision_df = analyze_precision_bootstrap(
            construct_info=construct_info,
            n_raters_range=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            true_kappa=true_kappa,
            n_iterations=1000,
            n_bootstrap=500
        )
        precision_df['true_kappa'] = true_kappa
        all_precision_data.append(precision_df)
        
        # Replication (increased to 500 iterations)
        replication_df = analyze_replication_variability(
            construct_info=construct_info,
            n_raters_range=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            true_kappa=true_kappa,
            n_iterations=1000  
        )
        replication_df['true_kappa'] = true_kappa
        all_replication_data.append(replication_df)
        
        # Find thresholds for this kappa
        stable_rows = stability_df[stability_df['mean_delta'] < 0.01]
        stable_n = stable_rows['n_raters'].min() if len(stable_rows) > 0 else 30
        
        precise_rows = precision_df[precision_df['mean_ci_width'] <= 0.10]
        precise_n = precise_rows['n_raters'].min() if len(precise_rows) > 0 else 30
        
        range_rows = replication_df[replication_df['range_95'] <= 0.10]
        range_n = range_rows['n_raters'].min() if len(range_rows) > 0 else 30
        
        overall_n = max(stable_n, precise_n, range_n)
        
        all_recommendations.append({
            'construct': construct_name,
            'true_kappa': true_kappa,
            'stability_n': stable_n,
            'precision_n': precise_n,
            'replication_range_n': range_n,
            'overall_recommendation': overall_n,
            'conservative': overall_n + 3
        })
        
        print(f"    Stability: {stable_n}, Precision: {precise_n}, Range: {range_n} → Rec: {overall_n}")
    
    # Combine all data
    all_stability_df = pd.concat(all_stability_data, ignore_index=True)
    all_precision_df = pd.concat(all_precision_data, ignore_index=True)
    all_replication_df = pd.concat(all_replication_data, ignore_index=True)
    recommendations_df = pd.DataFrame(all_recommendations)
    
    # Save combined data
    all_stability_df.to_csv(output_dir / 'stability_all_kappas.csv', index=False)
    all_precision_df.to_csv(output_dir / 'precision_all_kappas.csv', index=False)
    all_replication_df.to_csv(output_dir / 'replication_all_kappas.csv', index=False)
    recommendations_df.to_csv(output_dir / 'recommendations_by_kappa.csv', index=False)
    
    # Create comparison plots
    print("\n  Creating comparison plots...")
    plot_stability_comparison(all_stability_df, construct_name,
                             save_path=output_dir / 'stability_comparison.png')
    plot_precision_comparison(all_precision_df, construct_name,
                              save_path=output_dir / 'precision_comparison.png')
    plot_replication_comparison(all_replication_df, construct_name,
                               save_path=output_dir / 'replication_comparison.png')
    
    # Print summary
    print("\n" + "="*70)
    print(f"SUMMARY: {construct_name} Across Agreement Levels")
    print("="*70)
    print("\nRecommended sample sizes by true kappa:")
    print(recommendations_df[['true_kappa', 'stability_n', 'precision_n', 
                             'replication_range_n', 'overall_recommendation', 'conservative']])
    
    print(f"\n✓ Results saved to: {output_dir}/")
    
    return recommendations_df

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Parse command-line arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description='Multi-Kappa Analysis for Expert Rater Survey Sample Size Determination'
    )
    parser.add_argument(
        '--construct',
        type=str,
        required=True,
        choices=list(survey_structure.keys()),
        help='Neuropsychiatric construct to analyze'
    )
    parser.add_argument(
        '--kappa-range',
        nargs='+',
        type=float,
        default=[0.40, 0.50, 0.60, 0.70],
        help='True kappa values to test (default: 0.40 0.50 0.60 0.70)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("MULTI-KAPPA SIMULATION: HPC Array Job")
    print("ENIGMA-PD Harmonization Study")
    print("="*70)
    
    recommendations_df = run_multi_kappa_analyses(
        construct_name=args.construct,
        true_kappa_range=args.kappa_range,
        random_seed=args.seed
    )
    
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()