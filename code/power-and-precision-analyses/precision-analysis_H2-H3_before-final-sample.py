"""
Simulation-based power/precision analysis for H2 and H3:
- H2: Internal consistency (McDonald's omega)
- H3: Confirmatory factor analysis fit indices

Based on ENIGMA-PD questionnaire harmonization project data structure.

"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Check for optional dependencies
try:
    import semopy
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False
    print("semopy not installed. Install with: pip install semopy --break-system-packages")

try:
    from factor_analyzer import FactorAnalyzer
    FA_AVAILABLE = True
except ImportError:
    FA_AVAILABLE = False
    print("factor_analyzer not installed. Install with: pip install factor_analyzer --break-system-packages")


# =============================================================================
# DATA STRUCTURE FROM ENIGMA-PD
# =============================================================================

CONSTRUCT_STRUCTURE = {
    'depression': {
        'expected_n': 1500,
        'dimensions': {
            'mood_and_affective': 21,
            'activity_and_interest': 19,
            'cognitive_and_self_perception': 18,
            'somatic_and_vegetative': 11,
            'anxiety_and_distress': 4
        }
    },
    'anxiety': {
        'expected_n': 1500,
        'dimensions': {
            'cognitive_anxiety': 56,
            'somatic_anxiety': 28
        }
    },
    'psychosis': {
        'expected_n': 800,
        'dimensions': {
            'hallucinations': 21,
            'delusions': 17
        }
    },
    'apathy': {
        'expected_n': 800,
        'dimensions': {
            'behavioral_apathy': 45,
            'affective_apathy': 25,
            'cognitive_apathy': 20
        }
    },
    'sleep': {
        'expected_n': 1500,
        'dimensions': {
            'nocturnal_disturbances': 53,
            'daytime_sleepiness': 28,
            'rem_sleep_behavior': 17,
            'sleep_disordered_breathing': 3
        }
    },
    'impulse_control': {
        'expected_n': 1500,
        'dimensions': {
            'punding_hobbyism': 8,
            'compulsive_buying': 4,
            'compulsive_eating': 4,
            'dopamine_dysregulation': 4,
            'hypersexuality': 4,
            'pathological_gambling': 4
        }
    }
}


# =============================================================================
# H2: McDONALD'S OMEGA SIMULATION
# =============================================================================

def simulate_factor_data(n_subjects, n_items, factor_loading=0.7, 
                         error_var=None, seed=None):
    """
    Simulate data from a single-factor model.
    
    Parameters
    ----------
    n_subjects : int
        Number of participants
    n_items : int
        Number of items
    factor_loading : float
        Standardized factor loading for all items (default 0.7)
    error_var : float, optional
        Error variance. If None, calculated to give standardized loadings.
    seed : int, optional
        Random seed
    
    Returns
    -------
    np.ndarray
        Simulated item responses (n_subjects x n_items)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Factor scores
    factor = np.random.normal(0, 1, n_subjects)
    
    # Error variance (to achieve standardized loadings)
    if error_var is None:
        error_var = 1 - factor_loading**2
    
    # Generate items
    items = np.zeros((n_subjects, n_items))
    for i in range(n_items):
        error = np.random.normal(0, np.sqrt(error_var), n_subjects)
        items[:, i] = factor_loading * factor + error
    
    return items


def calculate_omega_hierarchical(data, return_details=False):
    """
    Calculate McDonald's Omega using a one-factor model fit via
    FactorAnalyzer(method="ml"), matching the production H3 script
    (hypothesis-testing.py):

        omega = (sum of loadings)^2 /
                [(sum of loadings)^2 + sum of unique variances]

    NOTE: an earlier version of this function estimated "loadings" as the
    first principal-component eigenvector of the correlation matrix scaled
    by sqrt(eigenvalue), and divided by var_total = n_items + 2*sum(r_ij).
    For equal-loading single-factor data these two quantities are
    mathematically (near-)identical, so that version always returned
    omega ~= 1 regardless of the true reliability -- confirmed numerically
    (true loadings 0.3/0.5/0.7/0.9 all produced omega >= 0.996). This
    version was validated to recover the analytic omega within simulation
    noise across true loadings 0.3-0.9 at N=2000 and N=20000.

    Parameters
    ----------
    data : np.ndarray
        Item response data (n_subjects x n_items)
    return_details : bool
        Whether to return additional details

    Returns
    -------
    float or dict
        Omega value, or dict with details if return_details=True
    """
    from factor_analyzer import FactorAnalyzer

    n_items = data.shape[1]
    corr_matrix = np.corrcoef(data, rowvar=False)
    off_diag = corr_matrix[np.triu_indices(n_items, k=1)]
    mean_r = np.mean(off_diag)
    alpha = (n_items * mean_r) / (1 + (n_items - 1) * mean_r)

    loadings = None
    try:
        fa = FactorAnalyzer(n_factors=1, rotation=None, method="ml")
        fa.fit(data)
        loadings = fa.loadings_[:, 0]
        uniquenesses = 1 - loadings ** 2
        sum_loadings = np.sum(loadings)
        omega = (sum_loadings ** 2) / (sum_loadings ** 2 + np.sum(uniquenesses))
        omega = float(np.clip(omega, 0, 1))
    except Exception:
        omega = alpha  # Fall back to alpha as approximation

    if return_details:
        return {
            'omega': omega,
            'alpha': alpha,
            'mean_r': mean_r,
            'n_items': n_items,
            'loadings': loadings
        }

    return omega


def bootstrap_omega_ci(data, n_bootstrap=1000, ci_level=0.95, seed=None):
    """
    Calculate bootstrap confidence interval for omega.
    
    Parameters
    ----------
    data : np.ndarray
        Item response data
    n_bootstrap : int
        Number of bootstrap samples
    ci_level : float
        Confidence level (default 0.95)
    seed : int, optional
        Random seed
    
    Returns
    -------
    dict
        Omega estimate and confidence interval
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_subjects = data.shape[0]
    omega_boot = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n_subjects, size=n_subjects, replace=True)
        boot_data = data[idx, :]
        
        try:
            omega = calculate_omega_hierarchical(boot_data)
            omega_boot.append(omega)
        except Exception:
            continue
    
    omega_boot = np.array(omega_boot)
    
    # Calculate CI
    alpha = 1 - ci_level
    ci_lower = np.percentile(omega_boot, 100 * alpha / 2)
    ci_upper = np.percentile(omega_boot, 100 * (1 - alpha / 2))
    
    return {
        'omega': np.mean(omega_boot),
        'omega_se': np.std(omega_boot),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower
    }


def run_omega_simulation(n_subjects, n_items, true_omega, n_simulations=100,
                         n_bootstrap=500, seed=None):
    """
    Run simulation to assess precision of omega estimates.
    
    Parameters
    ----------
    n_subjects : int
        Sample size
    n_items : int
        Number of items
    true_omega : float
        True population omega (determines factor loadings)
    n_simulations : int
        Number of simulation replications
    n_bootstrap : int
        Bootstrap samples per simulation
    seed : int, optional
        Random seed
    
    Returns
    -------
    dict
        Summary of simulation results
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Approximate factor loading from true omega
    # omega ≈ (n * lambda)^2 / (n * lambda^2 + n * (1-lambda^2))
    # For equal loadings: omega = n*lambda^2 / (1 + (n-1)*lambda^2)
    # Solving for lambda given omega and n:
    # lambda = sqrt(omega / (n - omega*(n-1)))
    
    denom = n_items - true_omega * (n_items - 1)
    if denom > 0:
        loading = np.sqrt(true_omega / denom)
        loading = np.clip(loading, 0.1, 0.95)
    else:
        loading = 0.7
    
    results = []
    
    for sim in range(n_simulations):
        # Simulate data
        data = simulate_factor_data(n_subjects, n_items, 
                                   factor_loading=loading, 
                                   seed=seed + sim if seed else None)
        
        # Calculate omega with bootstrap CI
        omega_result = bootstrap_omega_ci(data, n_bootstrap=n_bootstrap)
        omega_result['simulation'] = sim
        omega_result['true_omega'] = true_omega
        omega_result['n_subjects'] = n_subjects
        omega_result['n_items'] = n_items
        
        # Decision: correctly classify as adequate (>= 0.70)?
        omega_result['omega_above_threshold'] = omega_result['omega'] >= 0.70
        omega_result['ci_lower_above_threshold'] = omega_result['ci_lower'] >= 0.70
        
        results.append(omega_result)
    
    return pd.DataFrame(results)


def run_h2_precision_analysis(n_simulations=100, save_results=True,
                               output_file='h2_omega_precision_results.csv',
                               n_bootstrap=500):
    """
    Run full precision analysis for H2 across all constructs/dimensions.
    
    Parameters
    ----------
    n_simulations : int
        Number of simulations per scenario
    save_results : bool
        Whether to save to CSV
    output_file : str
        Output filename
    
    Returns
    -------
    pd.DataFrame
        Results from all simulations
    """
    all_results = []
    
    # Test scenarios based on actual ENIGMA-PD structure
    scenarios = []
    
    for construct, info in CONSTRUCT_STRUCTURE.items():
        for dim_name, n_items in info['dimensions'].items():
            scenarios.append({
                'construct': construct,
                'dimension': dim_name,
                'n_subjects': info['expected_n'],
                'n_items': n_items,
                'true_omega': 0.75  # Assume adequate reliability
            })
    
    # Also test at threshold
    for scenario in scenarios.copy():
        threshold_scenario = scenario.copy()
        threshold_scenario['true_omega'] = 0.70
        scenarios.append(threshold_scenario)
    
    print("="*70)
    print("H2 PRECISION ANALYSIS: McDonald's Omega")
    print("="*70)
    
    for scenario in scenarios:
        print(f"\n{scenario['construct']} - {scenario['dimension']}")
        print(f"  N={scenario['n_subjects']}, items={scenario['n_items']}, "
              f"true_ω={scenario['true_omega']}")
        
        results = run_omega_simulation(
            n_subjects=scenario['n_subjects'],
            n_items=scenario['n_items'],
            true_omega=scenario['true_omega'],
            n_simulations=n_simulations,
            n_bootstrap=n_bootstrap
        )
        
        results['construct'] = scenario['construct']
        results['dimension'] = scenario['dimension']
        
        all_results.append(results)
        
        # Quick summary
        mean_omega = results['omega'].mean()
        mean_ci_width = results['ci_width'].mean()
        pct_above = results['omega_above_threshold'].mean() * 100
        
        print(f"  Mean ω = {mean_omega:.3f}, CI width = {mean_ci_width:.3f}, "
              f"{pct_above:.0f}% above 0.70")
    
    results_df = pd.concat(all_results, ignore_index=True)
    
    if save_results:
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    
    return results_df


def summarize_h2_results(results_df):
    """
    Summarize H2 precision analysis results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Output from run_h2_precision_analysis
    
    Returns
    -------
    pd.DataFrame
        Summary by construct/dimension
    """
    summary = results_df.groupby(['construct', 'dimension', 'true_omega', 
                                   'n_subjects', 'n_items']).agg({
        'omega': ['mean', 'std'],
        'ci_width': ['mean', 'std'],
        'omega_above_threshold': 'mean',
        'ci_lower_above_threshold': 'mean'
    }).round(3)
    
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
    summary = summary.reset_index()
    
    # Calculate bias
    summary['bias'] = summary['omega_mean'] - summary['true_omega']
    
    # Power to correctly conclude omega >= 0.70 when true omega = 0.70
    summary['power'] = summary.apply(
        lambda x: x['omega_above_threshold_mean'] if x['true_omega'] >= 0.70 else np.nan,
        axis=1
    )
    
    return summary


# =============================================================================
# H3: CFA FIT INDICES SIMULATION
# =============================================================================

def simulate_cfa_data(n_subjects, factor_structure, factor_loadings=0.7,
                      factor_correlations=0.3, seed=None):
    """
    Simulate data from a CFA model.
    
    Parameters
    ----------
    n_subjects : int
        Number of participants
    factor_structure : dict
        Dict mapping factor names to number of items, e.g., {'F1': 5, 'F2': 4}
    factor_loadings : float or dict
        Factor loading(s). If float, same for all. If dict, by factor.
    factor_correlations : float
        Correlation between factors (same for all pairs)
    seed : int, optional
        Random seed
    
    Returns
    -------
    pd.DataFrame
        Simulated item responses
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_factors = len(factor_structure)
    factor_names = list(factor_structure.keys())
    
    # Create factor correlation matrix
    factor_corr = np.eye(n_factors)
    factor_corr[factor_corr == 0] = factor_correlations
    
    # Cholesky decomposition for correlated factors
    L = np.linalg.cholesky(factor_corr)
    
    # Generate factor scores
    uncorrelated_factors = np.random.normal(0, 1, (n_subjects, n_factors))
    factor_scores = uncorrelated_factors @ L.T
    
    # Generate items
    all_items = {}
    item_idx = 0
    
    for f_idx, (factor_name, n_items) in enumerate(factor_structure.items()):
        # Get loading for this factor
        if isinstance(factor_loadings, dict):
            loading = factor_loadings.get(factor_name, 0.7)
        else:
            loading = factor_loadings
        
        error_var = 1 - loading**2
        
        for i in range(n_items):
            item_name = f"item_{item_idx}"
            error = np.random.normal(0, np.sqrt(error_var), n_subjects)
            all_items[item_name] = loading * factor_scores[:, f_idx] + error
            item_idx += 1
    
    return pd.DataFrame(all_items), factor_structure


def calculate_cfa_fit_indices(data, factor_structure):
    """
    Fit CFA model and calculate fit indices.
    
    Parameters
    ----------
    data : pd.DataFrame
        Item response data
    factor_structure : dict
        Factor structure specification
    
    Returns
    -------
    dict
        Fit indices (RMSEA, CFI, SRMR, chi2/df)
    """
    if not SEMOPY_AVAILABLE:
        # Return approximate fit indices based on correlations
        return calculate_approximate_fit_indices(data, factor_structure)

    # Build model syntax for semopy
    model_syntax = ""
    item_idx = 0

    for factor_name, n_items in factor_structure.items():
        items = [f"item_{i}" for i in range(item_idx, item_idx + n_items)]
        model_syntax += f"{factor_name} =~ " + " + ".join(items) + "\n"
        item_idx += n_items

    try:
        model = semopy.Model(model_syntax)
        model.fit(data)

        # semopy.calc_stats() returns a 1-row DataFrame indexed 'Value' with
        # stat names as COLUMNS (not as the index). The previous version of
        # this function checked `'chi2' in stats.index` / `stats.loc['RMSEA', ...]`,
        # which is always False/absent for this shape, so it silently returned
        # NaN for every fit index regardless of whether semopy was installed
        # or fit successfully -- confirmed by inspecting stats.index (== ['Value'])
        # and stats.columns (contains 'chi2', 'DoF', 'RMSEA', 'CFI', ...).
        stats = semopy.calc_stats(model)
        chi2 = float(stats['chi2'].iloc[0])
        df = float(stats['DoF'].iloc[0])
        rmsea = float(stats['RMSEA'].iloc[0])
        cfi = float(stats['CFI'].iloc[0])

        # semopy.calc_stats() does not provide SRMR directly, so compute it
        # from the model-implied vs. observed correlation matrices.
        #
        # NOTE: an earlier version used np.cov(data, rowvar=False) here, which
        # is in the *data column* order. model.calc_sigma() is NOT in that
        # order -- it follows semopy's internal variable ordering (confirmed:
        # for an 84-item two-factor model, model.vars['indicator'] returned
        # ['item_62', 'item_45', 'item_14', ...], a shuffled order unrelated
        # to the data columns). Comparing the two matrices element-by-element
        # therefore compared mismatched item pairs, and the resulting "SRMR"
        # was garbage (~0.12-0.24 even for a correctly-specified model,
        # confirmed on the real ENIGMA-PD construct sizes, while RMSEA/CFI --
        # computed internally by semopy using its own consistent ordering --
        # looked correct). model.mx_cov is semopy's own internal observed-
        # covariance attribute and is guaranteed to share calc_sigma()'s
        # ordering (both come from the same fitted model instance), so use
        # that instead of independently recomputing the covariance from data.
        sigma = model.calc_sigma()[0]
        obs_cov = model.mx_cov
        d_obs = np.sqrt(np.diag(obs_cov))
        d_impl = np.sqrt(np.diag(sigma))
        obs_corr = obs_cov / np.outer(d_obs, d_obs)
        impl_corr = sigma / np.outer(d_impl, d_impl)
        p = obs_corr.shape[0]
        resid = obs_corr - impl_corr
        srmr = float(np.sqrt(np.mean(resid[np.triu_indices(p, k=1)] ** 2)))

        return {
            'chi2': chi2, 'df': df, 'chi2_df': chi2 / df if df > 0 else np.nan,
            'rmsea': rmsea, 'cfi': cfi, 'srmr': srmr,
            'converged': True, 'method': 'semopy'
        }
    except Exception as e:
        result = calculate_approximate_fit_indices(data, factor_structure)
        result['method'] = 'approximate_fallback'
        result['error'] = str(e)
        return result


def calculate_approximate_fit_indices(data, factor_structure):
    """
    Calculate approximate fit indices without full SEM.
    
    Uses analytical formulas based on correlation residuals.
    Good for simulation purposes when semopy unavailable.
    
    Parameters
    ----------
    data : pd.DataFrame
        Item response data
    factor_structure : dict
        Factor structure
    
    Returns
    -------
    dict
        Approximate fit indices
    """
    n = len(data)
    p = data.shape[1]
    n_factors = len(factor_structure)
    
    # Observed correlation matrix
    R = data.corr().values
    
    # Fit single-factor models per block and get residuals
    residuals = R.copy()
    item_idx = 0
    
    for factor_name, n_items in factor_structure.items():
        idx_range = range(item_idx, item_idx + n_items)
        
        # Within-factor: get average correlation
        block = R[np.ix_(list(idx_range), list(idx_range))]
        off_diag = block[np.triu_indices(n_items, k=1)]
        avg_r = np.mean(off_diag) if len(off_diag) > 0 else 0
        
        # Implied correlations within factor
        for i in idx_range:
            for j in idx_range:
                if i != j:
                    residuals[i, j] -= avg_r
        
        item_idx += n_items
    
    # SRMR: sqrt of mean squared residuals
    resid_off_diag = residuals[np.triu_indices(p, k=1)]
    srmr = np.sqrt(np.mean(resid_off_diag**2))
    
    # Degrees of freedom
    n_params = sum(factor_structure.values()) + n_factors * (n_factors + 1) // 2
    df = p * (p + 1) // 2 - n_params
    df = max(1, df)
    
    # Chi-square approximation
    chi2 = n * np.sum(resid_off_diag**2)
    
    # RMSEA approximation
    rmsea_num = max(0, chi2 - df)
    rmsea = np.sqrt(rmsea_num / (df * n)) if df > 0 else 0
    
    # CFI approximation (compare to null model)
    null_chi2 = n * np.sum(R[np.triu_indices(p, k=1)]**2)
    null_df = p * (p - 1) // 2
    
    if null_chi2 > null_df:
        cfi = 1 - max(0, chi2 - df) / (null_chi2 - null_df)
        cfi = np.clip(cfi, 0, 1)
    else:
        cfi = 1.0
    
    return {
        'chi2': chi2,
        'df': df,
        'chi2_df': chi2 / df if df > 0 else np.nan,
        'rmsea': rmsea,
        'cfi': cfi,
        'srmr': srmr,
        'converged': True,
        'method': 'approximate'
    }


def run_cfa_simulation(n_subjects, factor_structure, true_model='correct',
                       n_simulations=100, seed=None):
    """
    Run CFA simulation to assess fit index behavior.
    
    Parameters
    ----------
    n_subjects : int
        Sample size
    factor_structure : dict
        Factor structure
    true_model : str
        'correct' for correctly specified model, 'misspecified' for test
    n_simulations : int
        Number of replications
    seed : int, optional
        Random seed
    
    Returns
    -------
    pd.DataFrame
        Fit indices from all simulations
    """
    results = []
    
    for sim in range(n_simulations):
        # Simulate data
        data, _ = simulate_cfa_data(
            n_subjects=n_subjects,
            factor_structure=factor_structure,
            factor_loadings=0.7,
            factor_correlations=0.3,
            seed=seed + sim if seed else None
        )
        
        # Calculate fit indices.
        # NOTE: this previously called calculate_approximate_fit_indices()
        # directly, bypassing calculate_cfa_fit_indices() (and therefore
        # semopy) entirely even when semopy was installed and working --
        # confirmed by grep showing this was the only call site feeding
        # run_h3_precision_analysis(). Now routes through the
        # semopy-aware function, matching the intent of SEMOPY_AVAILABLE.
        fit = calculate_cfa_fit_indices(data, factor_structure)
        fit['simulation'] = sim
        fit['n_subjects'] = n_subjects
        fit['n_factors'] = len(factor_structure)
        fit['n_items'] = sum(factor_structure.values())
        fit['model_type'] = true_model
        
        # Check against thresholds
        fit['rmsea_ok'] = fit['rmsea'] <= 0.08 if not np.isnan(fit['rmsea']) else False
        fit['cfi_ok'] = fit['cfi'] >= 0.90 if not np.isnan(fit['cfi']) else False
        fit['srmr_ok'] = fit['srmr'] <= 0.08 if not np.isnan(fit['srmr']) else False
        fit['chi2_df_ok'] = fit.get('chi2_df', np.inf) <= 3
        fit['all_ok'] = fit['rmsea_ok'] and fit['cfi_ok'] and fit['srmr_ok']
        
        results.append(fit)
    
    return pd.DataFrame(results)


def run_h3_precision_analysis(n_simulations=100, save_results=True,
                               output_file='h3_cfa_precision_results.csv'):
    """
    Run full precision analysis for H3 across all constructs.
    
    Parameters
    ----------
    n_simulations : int
        Number of simulations per scenario
    save_results : bool
        Whether to save to CSV
    output_file : str
        Output filename
    
    Returns
    -------
    pd.DataFrame
        Results from all simulations
    """
    all_results = []
    
    print("="*70)
    print("H3 PRECISION ANALYSIS: CFA Fit Indices")
    print("="*70)
    
    for construct, info in CONSTRUCT_STRUCTURE.items():
        print(f"\n{construct.upper()}")
        print(f"  N={info['expected_n']}, factors={len(info['dimensions'])}, "
              f"items={sum(info['dimensions'].values())}")
        
        # Convert dimension names to simple factor names for CFA
        factor_structure = {f"F{i}": n_items 
                          for i, (_, n_items) in enumerate(info['dimensions'].items())}
        
        results = run_cfa_simulation(
            n_subjects=info['expected_n'],
            factor_structure=factor_structure,
            n_simulations=n_simulations
        )
        
        results['construct'] = construct
        
        all_results.append(results)
        
        # Quick summary
        print(f"  RMSEA: {results['rmsea'].mean():.3f} ({results['rmsea_ok'].mean()*100:.0f}% ≤ 0.08)")
        print(f"  CFI: {results['cfi'].mean():.3f} ({results['cfi_ok'].mean()*100:.0f}% ≥ 0.90)")
        print(f"  SRMR: {results['srmr'].mean():.3f} ({results['srmr_ok'].mean()*100:.0f}% ≤ 0.08)")
        print(f"  All criteria met: {results['all_ok'].mean()*100:.0f}%")
    
    results_df = pd.concat(all_results, ignore_index=True)
    
    if save_results:
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    
    return results_df


def summarize_h3_results(results_df):
    """
    Summarize H3 CFA precision analysis results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Output from run_h3_precision_analysis
    
    Returns
    -------
    pd.DataFrame
        Summary by construct
    """
    summary = results_df.groupby(['construct', 'n_subjects', 'n_factors', 'n_items']).agg({
        'rmsea': ['mean', 'std'],
        'cfi': ['mean', 'std'],
        'srmr': ['mean', 'std'],
        'rmsea_ok': 'mean',
        'cfi_ok': 'mean',
        'srmr_ok': 'mean',
        'all_ok': 'mean'
    }).round(3)
    
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns]
    summary = summary.reset_index()
    
    return summary


# =============================================================================
# MAIN: RUN ALL ANALYSES
# =============================================================================

def print_h2_recommendations(summary_df):
    """Print recommendations for H2."""
    print("\n" + "="*70)
    print("H2 RECOMMENDATIONS: McDonald's Omega")
    print("="*70)
    
    # Find problematic dimensions
    threshold_results = summary_df[summary_df['true_omega'] == 0.70]
    
    print("\n1. PRECISION BY NUMBER OF ITEMS")
    print("-" * 50)
    for _, row in threshold_results.iterrows():
        print(f"   {row['construct']}/{row['dimension']}: "
              f"{row['n_items']} items, CI width = {row['ci_width_mean']:.3f}")
    
    print("\n2. POTENTIAL ISSUES (CI width > 0.10 or low power)")
    print("-" * 50)
    issues = threshold_results[
        (threshold_results['ci_width_mean'] > 0.10) | 
        (threshold_results['power'] < 0.80)
    ]
    if len(issues) > 0:
        for _, row in issues.iterrows():
            print(f"   ⚠ {row['construct']}/{row['dimension']}: "
                  f"{row['n_items']} items, power = {row['power']:.0%}")
    else:
        print("   None identified")
    
    print("\n3. MINIMUM RECOMMENDATIONS")
    print("-" * 50)
    print("   - Dimensions with < 4 items may have unstable omega estimates")
    print("   - Consider reporting alpha alongside omega for small dimensions")


def print_h3_recommendations(summary_df):
    """Print recommendations for H3."""
    print("\n" + "="*70)
    print("H3 RECOMMENDATIONS: CFA Fit Indices")
    print("="*70)
    
    print("\n1. FIT INDEX PERFORMANCE BY CONSTRUCT")
    print("-" * 50)
    for _, row in summary_df.iterrows():
        print(f"   {row['construct']}: {row['all_ok_mean']*100:.0f}% meet all criteria")
        print(f"      N={row['n_subjects']}, {row['n_factors']} factors, {row['n_items']} items")
    
    print("\n2. N:PARAMETER RATIOS")
    print("-" * 50)
    for _, row in summary_df.iterrows():
        # Approximate number of parameters
        n_params = row['n_items'] + row['n_factors'] * (row['n_factors'] + 1) // 2
        ratio = row['n_subjects'] / n_params
        status = "✓" if ratio >= 10 else "⚠"
        print(f"   {status} {row['construct']}: N:param ratio = {ratio:.1f}:1")
    
    print("\n3. MINIMUM RECOMMENDATIONS")
    print("-" * 50)
    print("   - N:parameter ratio should be ≥ 10:1 (ideally 20:1)")
    print("   - All constructs meet minimum N = 200 recommendation")


if __name__ == "__main__":
    # NOTE: n_simulations=50 here does not exactly reproduce the committed
    # output/power-and-precision-analyses/precision-analysis_H2-H3/ CSVs --
    # those were generated with n_simulations=20 (H2, 880 rows / 44
    # scenarios) and 30 (H3, 180 rows / 6 scenarios), and the accompanying
    # *_summary_FIXED.csv files there reflect a manual correction that
    # predates this script's current summarize_h2/h3_results() and isn't
    # reproducible from this code as-is. Re-running this script will
    # therefore differ numerically (and in row count) from the committed
    # results, though the methodology is unchanged.
    print("="*70)
    print("POWER/PRECISION ANALYSIS FOR H2 AND H3")
    print("ENIGMA-PD Questionnaire Harmonization")
    print("="*70)

    # Run H2 analysis
    print("\n\nRunning H2 analysis (this may take a few minutes)...")
    h2_results = run_h2_precision_analysis(n_simulations=50)
    h2_summary = summarize_h2_results(h2_results)
    print_h2_recommendations(h2_summary)

    # Run H3 analysis
    print("\n\nRunning H3 analysis (this may take a few minutes)...")
    h3_results = run_h3_precision_analysis(n_simulations=50)
    h3_summary = summarize_h3_results(h3_results)
    print_h3_recommendations(h3_summary)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nResults saved to:")
    print("  - h2_omega_precision_results.csv")
    print("  - h3_cfa_precision_results.csv")