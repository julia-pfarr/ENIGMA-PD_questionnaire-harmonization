"""
Power/Precision Analysis for H1:
Testing questionnaire equivalence in harmonized C-scores

This script simulates ANCOVA analyses to determine:
1. Power to detect questionnaire effects of various sizes (η²)
2. Precision of η² estimates given expected sample sizes
3. Power to correctly conclude η² < .01 when true effect is negligible

Based on ENIGMA-PD questionnaire harmonization project.
Following methodology from Hussong et al. (2021).

Author: Generated for ENIGMA-PD registered report
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# EXPECTED DATA STRUCTURE FROM ENIGMA-PD
# =============================================================================

# Expected sample sizes per questionnaire by construct
# Based on consortium data availability
# Note: Sample sizes per questionnaire are estimates; actual distribution
# will depend on site-level questionnaire usage patterns

CONSTRUCT_QUESTIONNAIRES = {
    'depression': {
        'expected_total_n': 1500,
        'n_questionnaires': 8,
        'n_items': 73,
        'questionnaires': {
            'BDI': 250,
            'GDS': 250,
            'HADS': 200,
            'HAMD': 150,
            'NMSQuest': 150,
            'NMSS': 150,
            'NPI': 200,
            'MDS-UPDRS_I': 150
        }
    },
    'anxiety': {
        'expected_total_n': 1500,
        'n_questionnaires': 8,
        'n_items': 83,
        'questionnaires': {
            'HADS': 250,
            'HAMA': 200,
            'BAI': 200,
            'PAS': 150,
            'STAI': 200,
            'NMSQuest': 150,
            'NPI': 200,
            'MDS-UPDRS_I': 150
        }
    },
    'psychosis': {
        'expected_total_n': 800,
        'n_questionnaires': 8,
        'n_items': 37,
        'questionnaires': {
            'SCOPA-PC': 100,
            'SAPS-PD': 80,
            'PPRS': 80,
            'UM-PDHQ': 80,
            'NMSQuest': 120,
            'NMSS': 120,
            'NPI': 120,
            'MDS-UPDRS_I': 100
        }
    },
    'apathy': {
        'expected_total_n': 800,
        'n_questionnaires': 7,
        'n_items': 89,
        'questionnaires': {
            'SAS': 120,
            'AES': 120,
            'AMI': 100,
            'LARS': 150,
            'NMSQuest': 100,
            'NPI': 120,
            'MDS-UPDRS_I': 90
        }
    },
    'sleep': {
        'expected_total_n': 1500,
        'n_questionnaires': 13,
        'n_items': 100,
        'questionnaires': {
            'ESS': 150,
            'PSQI': 120,
            'RBD1Q': 100,
            'RBDSQ': 120,
            'ISI': 100,
            'KSS': 80,
            'NMSQuest': 150,
            'NMSS': 150,
            'PDSS': 120,
            'PDSS-2': 120,
            'SCOPA-SLEEP': 100,
            'MDS-UPDRS_I': 100,
            'NPI': 90
        }
    },
    'impulse_control': {
        'expected_total_n': 1500,
        'n_questionnaires': 1,
        'n_items': 27,
        'questionnaires': {
            'QUIP-RS': 1500  # Only one questionnaire - no H1 test possible
        }
    }
}


def simulate_ancova_data(
    n_per_questionnaire: dict,
    true_questionnaire_eta_sq: float = 0.0,
    covariate_r_squared: float = 0.15,
    n_covariates: int = 5,
    seed: int = None
) -> pd.DataFrame:
    """
    Simulate data for ANCOVA testing questionnaire effects on C-scores.
    
    Parameters
    ----------
    n_per_questionnaire : dict
        Sample size per questionnaire, e.g., {'BDI': 400, 'GDS': 350}
    true_questionnaire_eta_sq : float
        True partial eta-squared for questionnaire effect (0 = no effect)
    covariate_r_squared : float
        Variance explained by covariates (age, sex, disease duration, etc.)
    n_covariates : int
        Number of covariates
    seed : int, optional
        Random seed
    
    Returns
    -------
    pd.DataFrame
        Simulated data with C-score, questionnaire, and covariates
    """
    if seed is not None:
        np.random.seed(seed)
    
    questionnaire_names = list(n_per_questionnaire.keys())
    n_questionnaires = len(questionnaire_names)
    total_n = sum(n_per_questionnaire.values())
    
    # Generate questionnaire membership
    questionnaire = []
    for q_name, n in n_per_questionnaire.items():
        questionnaire.extend([q_name] * n)
    questionnaire = np.array(questionnaire)
    
    # Generate covariates (standardized)
    covariates = np.random.normal(0, 1, (total_n, n_covariates))
    
    # Generate covariate effects on C-score
    # Total R² from covariates = covariate_r_squared
    covariate_betas = np.random.uniform(0.1, 0.3, n_covariates)
    covariate_betas = covariate_betas / np.sqrt(np.sum(covariate_betas**2))  # Normalize
    covariate_betas = covariate_betas * np.sqrt(covariate_r_squared)
    
    covariate_effect = covariates @ covariate_betas
    
    # Generate questionnaire effects
    # Partial eta² = SS_questionnaire / (SS_questionnaire + SS_error)
    # We need to calibrate questionnaire effects to achieve target eta²
    
    if true_questionnaire_eta_sq > 0 and n_questionnaires > 1:
        # Calculate required between-group variance
        # eta² = sigma²_between / (sigma²_between + sigma²_error)
        # Solving: sigma²_between = eta² * sigma²_error / (1 - eta²)
        
        residual_var = 1 - covariate_r_squared  # Variance after covariates
        error_var = residual_var * (1 - true_questionnaire_eta_sq)
        between_var = residual_var * true_questionnaire_eta_sq
        
        # Generate questionnaire means
        questionnaire_means = np.random.normal(0, np.sqrt(between_var), n_questionnaires)
        questionnaire_means = questionnaire_means - np.mean(questionnaire_means)  # Center
        
        questionnaire_effect = np.zeros(total_n)
        for i, q_name in enumerate(questionnaire_names):
            mask = questionnaire == q_name
            questionnaire_effect[mask] = questionnaire_means[i]
    else:
        questionnaire_effect = np.zeros(total_n)
        error_var = 1 - covariate_r_squared
    
    # Generate error
    error = np.random.normal(0, np.sqrt(error_var), total_n)
    
    # Generate C-score
    c_score = covariate_effect + questionnaire_effect + error
    
    # Create DataFrame
    df = pd.DataFrame({
        'c_score': c_score,
        'questionnaire': questionnaire
    })
    
    for i in range(n_covariates):
        df[f'covariate_{i+1}'] = covariates[:, i]
    
    return df


def run_ancova(df: pd.DataFrame, n_covariates: int = 5) -> dict:
    """
    Run ANCOVA and return questionnaire effect statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with c_score, questionnaire, and covariates
    n_covariates : int
        Number of covariates
    
    Returns
    -------
    dict
        F-statistic, p-value, partial eta-squared, etc.
    """
    from scipy import stats
    
    # Get unique questionnaires
    questionnaires = df['questionnaire'].unique()
    n_questionnaires = len(questionnaires)
    
    if n_questionnaires == 1:
        return {
            'f_stat': np.nan,
            'p_value': np.nan,
            'partial_eta_sq': 0.0,
            'n_questionnaires': 1,
            'total_n': len(df),
            'questionnaire_significant': False
        }
    
    # Create design matrix
    # Covariates
    covariate_cols = [f'covariate_{i+1}' for i in range(n_covariates)]
    X_cov = df[covariate_cols].values
    
    # Questionnaire dummy codes (reference = first questionnaire)
    questionnaire_dummies = pd.get_dummies(df['questionnaire'], drop_first=True)
    X_quest = questionnaire_dummies.values
    
    # Full model: covariates + questionnaire
    X_full = np.column_stack([np.ones(len(df)), X_cov, X_quest])
    
    # Reduced model: covariates only
    X_reduced = np.column_stack([np.ones(len(df)), X_cov])
    
    y = df['c_score'].values
    
    # Fit full model
    beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
    y_pred_full = X_full @ beta_full
    ss_residual_full = np.sum((y - y_pred_full)**2)
    
    # Fit reduced model
    beta_reduced = np.linalg.lstsq(X_reduced, y, rcond=None)[0]
    y_pred_reduced = X_reduced @ beta_reduced
    ss_residual_reduced = np.sum((y - y_pred_reduced)**2)
    
    # Calculate F-statistic for questionnaire effect
    ss_questionnaire = ss_residual_reduced - ss_residual_full
    df_questionnaire = n_questionnaires - 1
    df_error = len(df) - X_full.shape[1]
    
    ms_questionnaire = ss_questionnaire / df_questionnaire
    ms_error = ss_residual_full / df_error
    
    f_stat = ms_questionnaire / ms_error
    p_value = 1 - stats.f.cdf(f_stat, df_questionnaire, df_error)
    
    # Partial eta-squared
    partial_eta_sq = ss_questionnaire / (ss_questionnaire + ss_residual_full)
    
    return {
        'f_stat': f_stat,
        'p_value': p_value,
        'partial_eta_sq': partial_eta_sq,
        'ss_questionnaire': ss_questionnaire,
        'ss_error': ss_residual_full,
        'df_questionnaire': df_questionnaire,
        'df_error': df_error,
        'n_questionnaires': n_questionnaires,
        'total_n': len(df),
        'questionnaire_significant': p_value < 0.05
    }


def run_single_simulation(
    n_per_questionnaire: dict,
    true_eta_sq: float,
    covariate_r_squared: float = 0.15,
    n_covariates: int = 5,
    seed: int = None
) -> dict:
    """
    Run a single simulation: generate data, run ANCOVA, evaluate.
    
    Parameters
    ----------
    n_per_questionnaire : dict
        Sample sizes per questionnaire
    true_eta_sq : float
        True partial eta-squared for questionnaire effect
    covariate_r_squared : float
        R² for covariates
    n_covariates : int
        Number of covariates
    seed : int, optional
        Random seed
    
    Returns
    -------
    dict
        Results including ANCOVA statistics and decision metrics
    """
    # Simulate data
    df = simulate_ancova_data(
        n_per_questionnaire=n_per_questionnaire,
        true_questionnaire_eta_sq=true_eta_sq,
        covariate_r_squared=covariate_r_squared,
        n_covariates=n_covariates,
        seed=seed
    )
    
    # Run ANCOVA
    results = run_ancova(df, n_covariates=n_covariates)
    
    # Add simulation parameters
    results['true_eta_sq'] = true_eta_sq
    results['covariate_r_sq'] = covariate_r_squared
    
    # Decision metrics
    # Correct decision when true_eta_sq = 0: conclude eta_sq < .01 (non-significant)
    # Correct decision when true_eta_sq >= .01: detect significant effect
    
    results['eta_sq_below_threshold'] = results['partial_eta_sq'] < 0.01
    results['correct_decision_h0'] = (true_eta_sq < 0.005) and (not results['questionnaire_significant'])
    results['correct_decision_h1'] = (true_eta_sq >= 0.01) and results['questionnaire_significant']
    
    return results


def run_power_analysis(
    construct: str = None,
    n_per_questionnaire: dict = None,
    true_eta_sq_values: list = None,
    n_simulations: int = 1000,
    save_results: bool = True,
    output_file: str = None
) -> pd.DataFrame:
    """
    Run full power analysis for H1.
    
    Parameters
    ----------
    construct : str, optional
        Construct name (uses predefined structure if provided)
    n_per_questionnaire : dict, optional
        Custom sample sizes per questionnaire
    true_eta_sq_values : list, optional
        Values of true eta² to test
    n_simulations : int
        Number of simulations per condition
    save_results : bool
        Whether to save to CSV
    output_file : str, optional
        Output filename
    
    Returns
    -------
    pd.DataFrame
        Results from all simulations
    """
    if true_eta_sq_values is None:
        true_eta_sq_values = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
    
    if construct is not None and construct in CONSTRUCT_QUESTIONNAIRES:
        n_per_questionnaire = CONSTRUCT_QUESTIONNAIRES[construct]['questionnaires']
        if output_file is None:
            output_file = f'h1_power_analysis_{construct}.csv'
    elif n_per_questionnaire is None:
        # Default example
        n_per_questionnaire = {'Q1': 400, 'Q2': 350, 'Q3': 300, 'Q4': 200, 'Q5': 150}
        if output_file is None:
            output_file = 'h1_power_analysis_default.csv'
    
    if output_file is None:
        output_file = 'h1_power_analysis.csv'
    
    all_results = []
    
    print("="*70)
    print(f"H1 POWER ANALYSIS: Questionnaire Equivalence")
    if construct:
        print(f"Construct: {construct}")
    print(f"Total N: {sum(n_per_questionnaire.values())}")
    print(f"Questionnaires: {len(n_per_questionnaire)}")
    print(f"N per questionnaire: {n_per_questionnaire}")
    print("="*70)
    
    for true_eta_sq in true_eta_sq_values:
        print(f"\nTrue η² = {true_eta_sq}")
        
        for sim in range(n_simulations):
            if (sim + 1) % 200 == 0:
                print(f"  Simulation {sim + 1}/{n_simulations}")
            
            results = run_single_simulation(
                n_per_questionnaire=n_per_questionnaire,
                true_eta_sq=true_eta_sq,
                seed=sim
            )
            results['simulation'] = sim
            results['construct'] = construct if construct else 'custom'
            all_results.append(results)
    
    results_df = pd.DataFrame(all_results)
    
    if save_results:
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    
    return results_df


def summarize_power_analysis(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize power analysis results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Output from run_power_analysis
    
    Returns
    -------
    pd.DataFrame
        Summary statistics
    """
    summary = results_df.groupby(['construct', 'true_eta_sq']).agg({
        'partial_eta_sq': ['mean', 'std', 'median'],
        'p_value': ['mean', 'median'],
        'questionnaire_significant': 'mean',  # Power / Type I error
        'eta_sq_below_threshold': 'mean',
        'total_n': 'first',
        'n_questionnaires': 'first'
    }).round(4)
    
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
    summary = summary.reset_index()
    
    # Interpretation
    summary['interpretation'] = summary.apply(
        lambda x: 'Type I error rate' if x['true_eta_sq'] == 0 
        else ('Power at threshold' if x['true_eta_sq'] == 0.01 
              else ('Power to detect' if x['true_eta_sq'] > 0.01 
                    else 'Near-zero effect')),
        axis=1
    )
    
    return summary


def print_power_recommendations(summary_df: pd.DataFrame):
    """
    Print recommendations based on power analysis.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Output from summarize_power_analysis
    """
    print("\n" + "="*70)
    print("H1 POWER ANALYSIS RECOMMENDATIONS")
    print("="*70)
    
    for construct in summary_df['construct'].unique():
        construct_data = summary_df[summary_df['construct'] == construct]
        
        print(f"\n{construct.upper()}")
        print(f"  N = {construct_data['total_n_first'].iloc[0]}, "
              f"Questionnaires = {construct_data['n_questionnaires_first'].iloc[0]}")
        print("-" * 50)
        
        for _, row in construct_data.iterrows():
            eta_sq = row['true_eta_sq']
            power = row['questionnaire_significant_mean']
            mean_est = row['partial_eta_sq_mean']
            
            if eta_sq == 0:
                print(f"  True η² = {eta_sq:.3f}: Type I error = {power:.1%}")
            elif eta_sq < 0.01:
                print(f"  True η² = {eta_sq:.3f}: P(significant) = {power:.1%} "
                      f"(should be low)")
            else:
                print(f"  True η² = {eta_sq:.3f}: Power = {power:.1%}")
            print(f"    Mean estimated η² = {mean_est:.4f}")
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
For your registered report, you want to demonstrate:

1. LOW Type I error (when true η² = 0):
   - P(significant) should be ≈ 5% (nominal alpha)
   
2. HIGH probability of correctly concluding η² < .01 when true η² ≈ 0:
   - This is your "power" to confirm successful harmonization
   
3. ADEQUATE power to detect meaningful effects (η² ≥ .02):
   - Power ≥ 80% indicates you can detect problematic questionnaire effects

Key finding for manuscript:
- If power to detect η² = .02 is > 80%, you can confidently interpret 
  non-significant results as evidence of successful harmonization.
""")


def run_full_h1_power_analysis():
    """
    Run power analysis for all constructs and summarize.
    """
    all_results = []
    
    for construct in CONSTRUCT_QUESTIONNAIRES.keys():
        if construct == 'impulse_control':
            print(f"\nSkipping {construct} (only 1 questionnaire)")
            continue
        
        results = run_power_analysis(
            construct=construct,
            n_simulations=500,
            save_results=False
        )
        all_results.append(results)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv('h1_power_analysis_all_constructs.csv', index=False)
    
    summary = summarize_power_analysis(combined_results)
    summary.to_csv('h1_power_analysis_summary.csv', index=False)
    
    print_power_recommendations(summary)
    
    return combined_results, summary


# =============================================================================
# QUICK ANALYSIS (for demonstration)
# =============================================================================

def quick_power_calculation(
    total_n: int,
    n_questionnaires: int,
    eta_sq: float,
    alpha: float = 0.05
) -> float:
    """
    Analytical power calculation for ANCOVA questionnaire effect.
    
    Uses non-central F distribution.
    
    Parameters
    ----------
    total_n : int
        Total sample size
    n_questionnaires : int
        Number of questionnaires (groups)
    eta_sq : float
        Expected partial eta-squared
    alpha : float
        Significance level
    
    Returns
    -------
    float
        Statistical power
    """
    # Degrees of freedom
    df1 = n_questionnaires - 1  # Numerator (questionnaire effect)
    df2 = total_n - n_questionnaires - 5  # Denominator (residual, adjusting for ~5 covariates)
    
    # Non-centrality parameter
    # lambda = (eta_sq / (1 - eta_sq)) * df2
    if eta_sq >= 1:
        return 1.0
    
    ncp = (eta_sq / (1 - eta_sq)) * (total_n - 1)
    
    # Critical F value
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    
    # Power = P(F > f_crit | H1)
    power = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)
    
    return power


def quick_power_table(save_results: bool = True, output_file: str = 'h1_analytical_power_table.csv'):
    """
    Generate quick power table for all constructs (analytical approximation --
    deterministic, no simulation). This is the table reported as
    "Analytical power estimates" in the manuscript's Supplementary Material S3.
    """
    print("\n" + "="*70)
    print("QUICK POWER TABLE (Analytical Approximation)")
    print("="*70)
    print(f"\n{'Construct':<18} {'N':>6} {'Q':>4} {'Items':>6} {'η²=.01':>10} {'η²=.02':>10} {'η²=.05':>10}")
    print("-"*70)

    rows = []
    for construct, info in CONSTRUCT_QUESTIONNAIRES.items():
        total_n = info['expected_total_n']
        n_q = info['n_questionnaires']
        n_items = info['n_items']

        if n_q == 1:
            print(f"{construct:<18} {total_n:>6} {n_q:>4} {n_items:>6} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            rows.append({'construct': construct, 'expected_n': total_n, 'n_questionnaires': n_q,
                        'n_items': n_items, 'power_eta_01': None, 'power_eta_02': None, 'power_eta_05': None})
            continue

        power_01 = quick_power_calculation(total_n, n_q, 0.01)
        power_02 = quick_power_calculation(total_n, n_q, 0.02)
        power_05 = quick_power_calculation(total_n, n_q, 0.05)

        print(f"{construct:<18} {total_n:>6} {n_q:>4} {n_items:>6} {power_01:>10.1%} {power_02:>10.1%} {power_05:>10.1%}")
        rows.append({'construct': construct, 'expected_n': total_n, 'n_questionnaires': n_q,
                    'n_items': n_items, 'power_eta_01': power_01, 'power_eta_02': power_02, 'power_eta_05': power_05})

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
- η² = .01: Threshold for "negligible" effect (your success criterion)
- η² = .02: Small effect that would indicate harmonization concerns
- η² = .05: Moderate effect indicating clear harmonization failure

Power > 80% to detect η² = .02 means you can confidently interpret
non-significant results as supporting successful harmonization.
""")

    table_df = pd.DataFrame(rows)
    if save_results:
        table_df.to_csv(output_file, index=False)
        print(f"Saved analytical power table to {output_file}")
    return table_df


if __name__ == "__main__":
    print("="*70)
    print("H1 POWER ANALYSIS")
    print("Questionnaire Equivalence in Harmonized C-Scores")
    print("="*70)

    # Quick analytical power table
    quick_power_table()

    # Simulation-based analysis for every construct (more accurate than the
    # analytical approximation above). Writes h1_power_<construct>.csv for
    # each -- run from within output/power-and-precision-analyses/power-analyses_H1/
    # to reproduce the committed output there.
    for construct in CONSTRUCT_QUESTIONNAIRES:
        print(f"\n\nRunning simulation-based analysis for {construct.upper()} construct...")

        results = run_power_analysis(
            construct=construct,
            n_simulations=500,
            save_results=True,
            output_file=f'h1_power_{construct}.csv'
        )

        summary = summarize_power_analysis(results)
        print("\nSUMMARY:")
        print(summary.to_string(index=False))

        print_power_recommendations(summary)