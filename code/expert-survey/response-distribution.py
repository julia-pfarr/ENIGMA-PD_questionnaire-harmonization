"""
Response Distribution and Questionnaire Comparison
=======================================================================

This script provides:
1. Visualization of response distributions across dimensions per construct
2. Questionnaire-level agreement analysis and ranking
3. Identification of questionnaires with most disagreement

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Union
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# Questionnaire Comparison Analysis (Revised)
# =============================================================================

def calculate_confidence_interval(mean: float, n: int, confidence: float = 0.95) -> tuple:
    """
    Calculate confidence interval for a proportion/mean.
    
    Uses Wilson score interval for proportions, which handles edge cases better.
    """
    if n == 0:
        return (np.nan, np.nan)
    
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    # Wilson score interval (better for proportions)
    denominator = 1 + z**2 / n
    center = (mean + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((mean * (1 - mean) + z**2 / (4 * n)) / n) / denominator
    
    return (max(0, center - margin), min(1, center + margin))


def analyze_questionnaire_difficulty_revised(questionnaire_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze questionnaire difficulty with confidence intervals and multiple metrics.
    
    Key improvements:
    - Adds 95% CI for mean agreement (wider for smaller questionnaires)
    - Reports both mean agreement and % items with high agreement
    - Categorizes reliability of the estimate based on N items
    """
    analysis = questionnaire_df.copy()
    
    # Calculate confidence intervals
    cis = analysis.apply(
        lambda row: calculate_confidence_interval(row['Mean Modal Proportion'], row['N Items']),
        axis=1
    )
    analysis['CI Lower'] = [ci[0] for ci in cis]
    analysis['CI Upper'] = [ci[1] for ci in cis]
    analysis['CI Width'] = analysis['CI Upper'] - analysis['CI Lower']
    
    # Reliability of estimate based on N items
    def estimate_reliability(n):
        if n >= 15:
            return 'High (N≥15)'
        elif n >= 8:
            return 'Moderate (8≤N<15)'
        elif n >= 4:
            return 'Low (4≤N<8)'
        else:
            return 'Very Low (N<4)'
    
    analysis['Estimate Reliability'] = analysis['N Items'].apply(estimate_reliability)
    
    # Difficulty score (inverse of agreement)
    analysis['Difficulty Score'] = 1 - analysis['Mean Modal Proportion']
    
    # Rank within construct
    analysis['Rank (within construct)'] = analysis.groupby('Construct')['Mean Modal Proportion'].rank(ascending=True).astype(int)
    
    # Categorize difficulty (considering CI)
    def categorize_difficulty_with_uncertainty(row):
        """Categorize considering the confidence interval"""
        mean = row['Mean Modal Proportion']
        ci_lower = row['CI Lower']
        
        # Use lower bound of CI for conservative categorization
        if ci_lower >= 0.85:
            return 'Easy (High Agreement)'
        elif ci_lower >= 0.70:
            return 'Likely Moderate'
        elif mean >= 0.70:
            return 'Possibly Moderate (wide CI)'
        else:
            return 'Difficult (Low Agreement)'
    
    analysis['Difficulty Category'] = analysis.apply(categorize_difficulty_with_uncertainty, axis=1)
    
    # Sort by difficulty
    analysis = analysis.sort_values('Mean Modal Proportion', ascending=True)
    
    return analysis


def plot_questionnaire_agreement_revised(questionnaire_df: pd.DataFrame,
                                         output_path: Optional[Path] = None):
    """
    Create a bar plot with error bars showing agreement levels for each questionnaire.
    
    Key improvements:
    - Shows confidence intervals as error bars
    - Point size indicates N items
    - Color indicates difficulty category
    """
    analysis = analyze_questionnaire_difficulty_revised(questionnaire_df)
    
    constructs = analysis['Construct'].unique()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, construct in enumerate(constructs):
        ax = axes[idx]
        subset = analysis[analysis['Construct'] == construct].sort_values('Mean Modal Proportion')
        
        y_positions = range(len(subset))
        
        # Color by difficulty (using CI lower bound for conservative estimate)
        colors = []
        for _, row in subset.iterrows():
            ci_lower = row['CI Lower']
            if ci_lower >= 0.85:
                colors.append('#2ecc71')  # Green - easy
            elif ci_lower >= 0.70:
                colors.append('#f39c12')  # Orange - moderate
            else:
                colors.append('#e74c3c')  # Red - difficult
        
        # Plot bars
        bars = ax.barh(y_positions, subset['Mean Modal Proportion'], color=colors, alpha=0.7)
        
        # Add error bars (confidence intervals)
        ax.errorbar(
            subset['Mean Modal Proportion'], 
            y_positions,
            xerr=[subset['Mean Modal Proportion'] - subset['CI Lower'],
                  subset['CI Upper'] - subset['Mean Modal Proportion']],
            fmt='none', 
            color='black', 
            capsize=3,
            capthick=1,
            linewidth=1
        )
        
        # Add N items annotation
        for i, (_, row) in enumerate(subset.iterrows()):
            # Show N items in parentheses
            ax.text(0.02, i, f"n={row['N Items']}", va='center', fontsize=8, 
                   color='white', fontweight='bold')
            # Show value at end of bar
            ax.text(row['Mean Modal Proportion'] + 0.02, i,
                   f"{row['Mean Modal Proportion']:.0%}", va='center', fontsize=9)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(subset['Questionnaire'], fontsize=9)
        
        # Add reference lines
        ax.axvline(x=0.70, color='orange', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=0.85, color='green', linestyle='--', alpha=0.7, linewidth=1)
        
        ax.set_xlabel('Mean Modal Proportion (Agreement)')
        ax.set_xlim(0, 1.05)
        ax.set_title(f'{construct.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(constructs), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Expert Agreement by Questionnaire\n'
                 '(Error bars = 95% CI; n = number of items; wider CI = less reliable estimate)', 
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def create_questionnaire_summary_revised(questionnaire_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table with confidence intervals and reliability indicators.
    """
    analysis = analyze_questionnaire_difficulty_revised(questionnaire_df)
    
    # Select and format columns
    summary = analysis[[
        'Construct', 'Questionnaire', 'N Items', 
        'Mean Modal Proportion', 'CI Lower', 'CI Upper',
        'SD Modal Proportion', 'Mean Entropy', 
        'Proportion High Agreement', 'Estimate Reliability',
        'Difficulty Category'
    ]].copy()
    
    # Create formatted CI column
    summary['95% CI'] = summary.apply(
        lambda row: f"[{row['CI Lower']:.2f}, {row['CI Upper']:.2f}]", axis=1
    )
    
    # Round numeric columns
    summary['Mean Modal Proportion'] = summary['Mean Modal Proportion'].round(3)
    summary['SD Modal Proportion'] = summary['SD Modal Proportion'].round(3)
    summary['Mean Entropy'] = summary['Mean Entropy'].round(3)
    summary['Proportion High Agreement'] = (summary['Proportion High Agreement'] * 100).round(1)
    
    summary = summary.rename(columns={
        'Mean Modal Proportion': 'Mean Agreement',
        'Proportion High Agreement': '% Items High Agreement'
    })
    
    # Reorder columns
    summary = summary[[
        'Construct', 'Questionnaire', 'N Items', 'Mean Agreement', '95% CI',
        'SD Modal Proportion', 'Mean Entropy', '% Items High Agreement',
        'Estimate Reliability', 'Difficulty Category'
    ]]
    
    return summary


def print_questionnaire_summary_revised(questionnaire_df: pd.DataFrame):
    """
    Print a readable summary with appropriate caveats for small questionnaires.
    """
    analysis = analyze_questionnaire_difficulty_revised(questionnaire_df)
    
    print("\n" + "="*90)
    print("QUESTIONNAIRE DIFFICULTY ANALYSIS (with uncertainty estimates)")
    print("="*90)
    print("\nNote: Questionnaires with few items have wider confidence intervals.")
    print("Interpret low agreement with caution when N items is small.\n")
    
    # Group by reliability of estimate
    for reliability in ['High (N≥15)', 'Moderate (8≤N<15)', 'Low (4≤N<8)', 'Very Low (N<4)']:
        subset = analysis[analysis['Estimate Reliability'] == reliability]
        if len(subset) > 0:
            print(f"\n--- Estimate Reliability: {reliability} ---")
            for _, row in subset.sort_values('Mean Modal Proportion').iterrows():
                ci_str = f"[{row['CI Lower']:.0%}, {row['CI Upper']:.0%}]"
                print(f"  {row['Questionnaire']:12} ({row['Construct']:15}): "
                      f"{row['Mean Modal Proportion']:.0%} {ci_str:15} "
                      f"N={row['N Items']:2} items, "
                      f"{row['Proportion High Agreement']*100:.0f}% items high agreement")
    
    # Summary by construct (item-weighted)
    print("\n" + "-"*90)
    print("CONSTRUCT-LEVEL SUMMARY (weighted by number of items):")
    print("-"*90)
    
    for construct in analysis['Construct'].unique():
        subset = analysis[analysis['Construct'] == construct]
        
        # Calculate item-weighted mean
        total_items = subset['N Items'].sum()
        weighted_mean = (subset['Mean Modal Proportion'] * subset['N Items']).sum() / total_items
        
        # Simple mean for comparison
        simple_mean = subset['Mean Modal Proportion'].mean()
        
        print(f"\n{construct.upper()}:")
        print(f"  Total items: {total_items}")
        print(f"  Item-weighted mean agreement: {weighted_mean:.1%}")
        print(f"  Simple mean across questionnaires: {simple_mean:.1%}")
        print(f"  Range: {subset['Mean Modal Proportion'].min():.1%} - {subset['Mean Modal Proportion'].max():.1%}")


def plot_item_level_summary(questionnaire_df: pd.DataFrame,
                            output_path: Optional[Path] = None):
    """
    Focus on item-level statistics rather than questionnaire-level.
    
    Shows total items per construct and proportion meeting different agreement thresholds.
    """
    # Aggregate at construct level (weighted by N items)
    construct_stats = []
    
    for construct in questionnaire_df['Construct'].unique():
        subset = questionnaire_df[questionnaire_df['Construct'] == construct]
        
        total_items = subset['N Items'].sum()
        
        # Calculate total items at each agreement level
        # Using Proportion High Agreement which indicates % of items with modal prop >= 0.70
        weighted_high_agreement = (subset['Proportion High Agreement'] * subset['N Items']).sum() / total_items
        
        # Item-weighted mean agreement
        weighted_mean = (subset['Mean Modal Proportion'] * subset['N Items']).sum() / total_items
        
        construct_stats.append({
            'Construct': construct,
            'Total Items': total_items,
            'Item-Weighted Mean Agreement': weighted_mean,
            '% Items High Agreement': weighted_high_agreement * 100
        })
    
    stats_df = pd.DataFrame(construct_stats)
    stats_df = stats_df.sort_values('Item-Weighted Mean Agreement')
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Item-weighted mean agreement
    ax1 = axes[0]
    colors = ['#2ecc71' if x >= 0.85 else '#f39c12' if x >= 0.70 else '#e74c3c' 
              for x in stats_df['Item-Weighted Mean Agreement']]
    bars = ax1.barh(stats_df['Construct'], stats_df['Item-Weighted Mean Agreement'], color=colors)
    ax1.axvline(x=0.70, color='orange', linestyle='--', alpha=0.7)
    ax1.axvline(x=0.85, color='green', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Item-Weighted Mean Agreement')
    ax1.set_title('Agreement by Construct\n(weighted by number of items)', fontweight='bold')
    ax1.set_xlim(0, 1)
    
    # Add labels
    for bar, val, n in zip(bars, stats_df['Item-Weighted Mean Agreement'], stats_df['Total Items']):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.0%} (n={n})', va='center', fontsize=10)
    
    # Right: % items with high agreement
    ax2 = axes[1]
    bars2 = ax2.barh(stats_df['Construct'], stats_df['% Items High Agreement'], 
                     color=sns.color_palette("Blues_d", len(stats_df)))
    ax2.set_xlabel('% of Items with High Agreement (≥70%)')
    ax2.set_title('Proportion of Items with High Agreement', fontweight='bold')
    ax2.set_xlim(0, 100)
    
    for bar, val in zip(bars2, stats_df['% Items High Agreement']):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}%', va='center', fontsize=10)
    
    plt.suptitle('Construct-Level Summary (Item-Weighted)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    return stats_df


# =============================================================================
# Main Function
# =============================================================================

def run_analyses_revised(response_dist_path: Union[str, Path],
                         questionnaire_path: Union[str, Path],
                         output_dir: Optional[Union[str, Path]] = None):
    """
    Run revised analyses with improved questionnaire comparison.
    """
    print("="*80)
    print("RESPONSE DISTRIBUTION & QUESTIONNAIRE COMPARISON (REVISED)")
    print("="*80)
    
    # Setup output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    response_dist_df = pd.read_csv(response_dist_path)
    questionnaire_df = pd.read_csv(questionnaire_path)
    
    print(f"  Response distribution: {len(response_dist_df)} rows")
    print(f"  Questionnaire comparison: {len(questionnaire_df)} questionnaires")
    
    # Response Distribution Analysis
    print("\n" + "-"*40)
    print("RESPONSE DISTRIBUTION ANALYSIS")
    print("-"*40)
    
    # Questionnaire Analysis (Revised)
    print("\n" + "-"*40)
    print("QUESTIONNAIRE COMPARISON ANALYSIS (REVISED)")
    print("-"*40)
    
    # Print summary with uncertainty
    print_questionnaire_summary_revised(questionnaire_df)
    
    # Create summary table
    summary = create_questionnaire_summary_revised(questionnaire_df)
    
    # Item-level summary
    print("\n" + "-"*40)
    print("ITEM-LEVEL SUMMARY BY CONSTRUCT")
    print("-"*40)
    item_stats = plot_item_level_summary(questionnaire_df, 
                                         plots_dir / 'construct_item_summary.png' if output_dir else None)
    print(item_stats.to_string(index=False))
    
    # Save outputs
    if output_dir:
        summary.to_csv(output_dir / 'questionnaire_summary_with_ci.csv', index=False)
        item_stats.to_csv(output_dir / 'construct_item_summary.csv', index=False)
        
        # Generate plots
        plot_questionnaire_agreement_revised(questionnaire_df,
                                            plots_dir / 'questionnaire_agreement_with_ci.png')
        print(f"\nResults saved to {output_dir}/")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return {
        'response_distribution': response_dist_df,
        'questionnaire_summary': summary,
        'item_level_summary': item_stats
    }


# =============================================================================
# Command Line Interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Revised response distribution and questionnaire comparison analysis"
    )
    parser.add_argument("--response-dist", type=str, required=True,
                       help="Path to response_distribution.csv")
    parser.add_argument("--questionnaire", type=str, required=True,
                       help="Path to questionnaire_comparison.csv")
    parser.add_argument("--output-dir", "-o", type=str, default="./revised_output",
                       help="Directory to save outputs")
    
    args = parser.parse_args()
    
    results = run_analyses_revised(
        response_dist_path=args.response_dist,
        questionnaire_path=args.questionnaire,
        output_dir=args.output_dir
    )