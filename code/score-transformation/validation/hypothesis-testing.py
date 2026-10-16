"""
Validation Analyses v2 — Tetrachoric Omega for Binary Items
============================================================

Identical to validation_analyses.py except H3 (McDonald's omega):
  - Detects binary-response dimensions (all items 0/1, e.g. GDS, RBDSQ)
  - Uses tetrachoric correlations instead of Pearson for those dimensions
  - Pearson-based omega (v1 method) is used for all other dimensions

Rationale: Pearson correlations between binary items are attenuated by
endorsement rates (maximum possible correlation is limited when base rates
differ), causing factor loadings — and therefore omega — to be systematically
underestimated. Tetrachoric correlations recover the latent continuous
correlation and give unbiased omega estimates for binary scales.

H1 and H2 are identical to v1; outputs are saved with _v2 suffix to
avoid overwriting the original results.

Usage:
------
python hypothesis-testing.py

Outputs (in output/validation/):
  h1_omnibus_v2.csv
  h1_pairwise_v2.csv
  h1a_omnibus_eta2_v2.png
  h1b_*_pairwise_v2.png
  h1c_*_distributions_v2.png
  h2_cfa_fit_v2.csv
  h2_cfa_fit_v2.png
  h3_omega_v2.csv             ← key difference: tetrachoric for binary dims
  h3_omega_v2.png
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import patsy
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import norm, multivariate_normal
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from factor_analyzer import FactorAnalyzer
import semopy

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Paths (overridden in main() when --binary is passed)
# =============================================================================

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent.parent.parent
INPUT_DIR = PROJECT_DIR / "input" / "score-transformation"
OUTPUT_DIR = PROJECT_DIR / "output" / "score-transformation" / "hypotheses-testing"
SCORE_DIR = PROJECT_DIR / "output" / "score-transformation"

MASTER_TABLE = INPUT_DIR / "master-table_subject-data.csv"
ITEM_MAPPING = INPUT_DIR / "item_dimension_mapping.csv"
DIMENSION_SCORES = SCORE_DIR / "dimension_scores.csv"

MISSING_SENTINEL = -9
MIN_N_PER_QUESTIONNAIRE = 30
MIN_N_QUESTIONNAIRES = 2
MIN_N_CFA = 100
MIN_N_OMEGA = 30
SITE_TOP_N = 50

COVARIATES = ["age", "sex", "group", "disease_duration", "HY_stage", "LEDD_total", "MDS_UPDRS_III_total_score"]
REDUCED_COVARIATES = ["age", "sex", "group"]  # language added as C(language) like site_group
C_SCORE_COL = "av_cdf_score"


def assign_site_group(site_series: pd.Series) -> pd.Series:
    counts = site_series.value_counts()
    major = counts[counts >= SITE_TOP_N].index
    return site_series.apply(lambda s: s if s in major else "other")

# Suffix appended to all output filenames (empty for base run; set per mode below)
V2_SUFFIX = ""

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# H1 / H2 helpers (identical to v1)
# =============================================================================

def partial_eta_squared(anova_table: pd.DataFrame, factor: str) -> float:
    if factor not in anova_table.index or "Residual" not in anova_table.index:
        return np.nan
    ss_effect = anova_table.loc[factor, "sum_sq"]
    ss_resid = anova_table.loc["Residual", "sum_sq"]
    if ss_effect + ss_resid == 0:
        return np.nan
    return ss_effect / (ss_effect + ss_resid)


def pairwise_emm_differences(df, group_col, score_col, covariates, cat_covariates=None):
    """
    Estimated marginal means (EMMs) and pairwise differences for `group_col`
    (typically "questionnaire"), adjusted for numeric `covariates` and
    categorical `cat_covariates` (e.g. site_group, language).

    NOTE: an earlier version of this function only accepted numeric
    `covariates` and was called from run_h1() WITHOUT site_group/language,
    even though the omnibus ANCOVA it is reported alongside always includes
    those terms when available. This meant the pairwise results reflected a
    less-adjusted model than the omnibus F-test/eta^2. Empirically verified
    on real data (depression dim1, BDI vs GDS, which are strongly confounded
    with site): the estimated EMM difference shifted from 0.080 (no site
    adjustment) to 0.089 (with site adjustment matching the omnibus model).
    run_h1() now passes whatever categorical terms actually ended up in the
    fitted omnibus formula for that dimension (see cat_covs_used below).

    EMMs are computed by g-computation: for each level of group_col, predict
    every observed row's score with group_col counterfactually set to that
    level (holding each row's own covariate values, including categorical
    ones, fixed) and average. This is required once categorical covariates
    are present (there is no single "mean site" to plug into `.predict()`)
    and is the standard definition of an EMM under covariate adjustment.

    The SE of each pairwise difference is obtained via the delta method,
    using the model's parameter covariance matrix and the (row-averaged)
    contrast vector between the two group levels' design rows -- the exact
    variance of a linear combination of OLS coefficients -- rather than the
    previous version's naive sqrt(mse_resid*(1/n_a+1/n_b)) two-independent-
    samples formula, which ignores covariate-induced correlation between the
    compared groups' coefficient estimates and is only exact in a fully
    balanced/orthogonal design.
    """
    cat_covariates = cat_covariates or []
    groups = df[group_col].unique()
    if len(groups) < 2:
        return pd.DataFrame()

    avail_covs = [c for c in covariates if c in df.columns and df[c].notna().any()]
    avail_cat = [c for c in cat_covariates if c in df.columns
                 and df[c].notna().any() and df[c].nunique() > 1]

    formula_parts = [f"C({group_col})"] + avail_covs + [f"C({c})" for c in avail_cat]
    formula = f"{score_col} ~ {' + '.join(formula_parts)}"
    needed = [score_col, group_col] + avail_covs + avail_cat
    fit_df = df.dropna(subset=needed)
    if fit_df[group_col].nunique() < 2:
        return pd.DataFrame()

    try:
        model = ols(formula, data=fit_df).fit()
    except Exception:
        return pd.DataFrame()

    design_info = model.model.data.design_info
    cov_params = model.cov_params().values
    params = model.params.values
    df_resid = model.df_resid
    t_crit_90 = stats.t.ppf(0.95, df_resid)

    design_rows = {}
    emms = {}
    for g in groups:
        tmp = fit_df.copy()
        tmp[group_col] = g
        try:
            dm = patsy.dmatrix(design_info, tmp, return_type="dataframe").values
        except Exception:
            return pd.DataFrame()
        design_rows[g] = dm.mean(axis=0)
        emms[g] = float(design_rows[g] @ params)

    pairs = []
    for i, ga in enumerate(groups):
        for gb in groups[i + 1:]:
            n_a = (fit_df[group_col] == ga).sum()
            n_b = (fit_df[group_col] == gb).sum()
            diff = emms[ga] - emms[gb]
            contrast = design_rows[ga] - design_rows[gb]
            var_diff = float(contrast @ cov_params @ contrast.T)
            se = np.sqrt(var_diff) if var_diff > 0 else np.nan
            t_stat = diff / se if (se and not np.isnan(se) and se > 0) else np.nan
            p_val = 2 * stats.t.sf(abs(t_stat), df_resid) if not np.isnan(t_stat) else np.nan
            pairs.append({
                "group_a": ga, "group_b": gb,
                "emm_a": emms[ga], "emm_b": emms[gb],
                "diff": diff, "se": se,
                "ci_lo_90": diff - t_crit_90 * se if not np.isnan(se) else np.nan,
                "ci_hi_90": diff + t_crit_90 * se if not np.isnan(se) else np.nan,
                "p_value": p_val,
                "n_a": int(n_a), "n_b": int(n_b),
            })
    return pd.DataFrame(pairs)


# =============================================================================
# H3 NEW: tetrachoric correlation helpers
# =============================================================================

def is_binary(data: pd.DataFrame) -> bool:
    """Return True if all non-NaN values across all columns are in {0, 1}."""
    vals = data.values.flatten()
    vals = vals[~np.isnan(vals.astype(float))]
    if len(vals) == 0:
        return False
    return bool(np.all((vals == 0) | (vals == 1)))


def tetrachoric_corr_pair(x: np.ndarray, y: np.ndarray) -> float:
    """
    Estimate the tetrachoric correlation between two binary vectors.

    Assumes both variables are indicators of underlying continuous normal
    latent traits. Finds the latent correlation ρ that makes the bivariate
    normal model consistent with the observed 2×2 table.

    Parameters
    ----------
    x, y : 1-D arrays of 0/1 values (NaN values are dropped pairwise).

    Returns
    -------
    float : estimated tetrachoric correlation, or NaN if estimation fails.
    """
    valid = ~(np.isnan(x.astype(float)) | np.isnan(y.astype(float)))
    x, y = x[valid].astype(float), y[valid].astype(float)
    if len(x) < 5:
        return np.nan

    p1 = np.mean(x)   # P(X = 1)
    p2 = np.mean(y)   # P(Y = 1)

    if p1 <= 0 or p1 >= 1 or p2 <= 0 or p2 >= 1:
        return np.nan  # no variation in one variable

    # Standard normal thresholds: P(X = 0) = Φ(t_x)
    t_x = norm.ppf(1 - p1)
    t_y = norm.ppf(1 - p2)

    p11_obs = np.mean((x == 1) & (y == 1))

    def implied_p11(rho: float) -> float:
        """P(X=1, Y=1) under bivariate normal model with correlation rho."""
        phi2 = multivariate_normal.cdf([t_x, t_y], mean=[0, 0],
                                        cov=[[1, rho], [rho, 1]])
        return 1.0 - norm.cdf(t_x) - norm.cdf(t_y) + phi2

    def residual(rho: float) -> float:
        return implied_p11(rho) - p11_obs

    lo_r, hi_r = residual(-0.9999), residual(0.9999)
    if lo_r * hi_r > 0:
        # No sign change: clamp to the boundary with the smaller residual
        return float(-0.9999 if abs(lo_r) < abs(hi_r) else 0.9999)

    try:
        return float(brentq(residual, -0.9999, 0.9999, xtol=1e-6, maxiter=100))
    except Exception:
        return np.nan


def tetrachoric_corr_matrix(data: pd.DataFrame) -> np.ndarray:
    """
    Compute a pairwise tetrachoric correlation matrix for a binary item dataframe.

    Falls back to 0 for pairs where estimation fails.
    Applies a nearest-PSD correction if the resulting matrix is not positive
    semi-definite (can happen when estimated correlations are inconsistent).
    """
    cols = data.columns.tolist()
    p = len(cols)
    R = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            r = tetrachoric_corr_pair(
                data[cols[i]].values,
                data[cols[j]].values,
            )
            R[i, j] = R[j, i] = 0.0 if np.isnan(r) else r

    # Nearest positive semi-definite correction (Higham 2002 approximation)
    eigvals = np.linalg.eigvalsh(R)
    if eigvals.min() < 1e-8:
        R += np.eye(p) * (abs(eigvals.min()) + 1e-6)
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)  # rescale back to correlation form

    return R


def omega_from_corr_matrix(R: np.ndarray) -> float:
    """
    Compute McDonald's omega total from a correlation matrix using
    iterative principal axis factoring (one factor).

    ω = (Σλᵢ)² / ((Σλᵢ)² + Σδᵢ)
    where λᵢ are loadings on the first factor and δᵢ = 1 − h²ᵢ.

    This mirrors the formula used by FactorAnalyzer in v1 but accepts a
    pre-computed correlation matrix — necessary for tetrachoric input.
    """
    p = R.shape[0]
    if p < 3:
        return np.nan

    # Initial communality estimates: squared multiple correlations
    try:
        R_inv = np.linalg.pinv(R)
        h2 = np.clip(1.0 - 1.0 / np.diag(R_inv), 0.001, 0.999)
    except Exception:
        h2 = np.full(p, 0.5)

    loadings = np.zeros(p)
    for _ in range(1000):
        R_red = R.copy()
        np.fill_diagonal(R_red, h2)
        eigvals, eigvecs = np.linalg.eigh(R_red)
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        lam = np.sqrt(max(float(eigvals[0]), 0.0)) * eigvecs[:, 0]
        h2_new = np.clip(lam ** 2, 0.0, 1.0)

        if np.max(np.abs(h2_new - h2)) < 1e-8:
            loadings = lam
            h2 = h2_new
            break
        h2 = h2_new
        loadings = lam

    unique_var = 1.0 - h2
    sum_lam = float(np.sum(loadings))
    denom = sum_lam ** 2 + float(np.sum(unique_var))
    if denom <= 0:
        return np.nan
    return float(sum_lam ** 2 / denom)


# =============================================================================
# H3: omega computation (v2)
# =============================================================================

def compute_omega_v2(item_data: pd.DataFrame) -> dict:
    """
    Compute McDonald's omega, choosing the correlation type based on item scale:

    Binary items (0/1 only)  → tetrachoric correlation matrix → omega_from_corr_matrix
    Polytomous items          → Pearson-based FactorAnalyzer   (same as v1)
    """
    n, p = item_data.shape
    if p < 3 or n < 10:
        return {"omega": np.nan, "n_items": p, "n_subjects": n, "corr_type": "none"}

    if is_binary(item_data):
        R = tetrachoric_corr_matrix(item_data)
        omega = omega_from_corr_matrix(R)
        return {"omega": omega, "n_items": p, "n_subjects": n, "corr_type": "tetrachoric"}

    try:
        fa = FactorAnalyzer(n_factors=1, rotation=None, method="ml")
        fa.fit(item_data)
        loadings = fa.loadings_[:, 0]
        unique_vars = 1 - fa.get_communalities()
        sum_lam = np.sum(loadings)
        omega = sum_lam ** 2 / (sum_lam ** 2 + np.sum(unique_vars))
        return {"omega": float(omega), "n_items": p, "n_subjects": n, "corr_type": "pearson"}
    except Exception as e:
        return {"omega": np.nan, "n_items": p, "n_subjects": n,
                "corr_type": "pearson", "error": str(e)}


def bootstrap_omega_v2(
    item_data: pd.DataFrame,
    n_bootstrap: int = 1000,
    n_bootstrap_binary: int = 500,
    random_state: int = 42,
) -> tuple[float, float, float, str]:
    """
    Compute omega with 95% bootstrap CI.

    Returns (omega, ci_lo, ci_hi, corr_type).
    Fewer bootstrap resamples are used for binary dimensions because
    computing the tetrachoric matrix is more expensive than Pearson.
    """
    rng = np.random.default_rng(random_state)
    result = compute_omega_v2(item_data)
    omega = result.get("omega", np.nan)
    corr_type = result.get("corr_type", "unknown")

    if np.isnan(omega):
        return np.nan, np.nan, np.nan, corr_type

    n_boot = n_bootstrap_binary if corr_type == "tetrachoric" else n_bootstrap
    boot_omegas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(item_data), size=len(item_data))
        boot_data = item_data.iloc[idx].reset_index(drop=True)
        boot_res = compute_omega_v2(boot_data)
        bo = boot_res.get("omega", np.nan)
        if not np.isnan(bo):
            boot_omegas.append(bo)

    if len(boot_omegas) < 10:
        return omega, np.nan, np.nan, corr_type

    ci_lo = float(np.percentile(boot_omegas, 2.5))
    ci_hi = float(np.percentile(boot_omegas, 97.5))
    return omega, ci_lo, ci_hi, corr_type


def run_h3_v2(
    master_df: pd.DataFrame,
    item_mapping: pd.DataFrame,
    n_bootstrap: int = 1000,
    n_bootstrap_binary: int = 500,
) -> pd.DataFrame:
    """
    H3 with tetrachoric corrections for binary dimensions.
    """
    print("\n" + "=" * 70)
    print("H3: INTERNAL CONSISTENCY (McDONALD'S OMEGA — v2 with tetrachoric)")
    print("=" * 70)

    rows = []
    groups = item_mapping.groupby(
        ["questionnaire", "construct", "dimension", "dimension_label"]
    )

    for (questionnaire, construct, dimension, dim_label), group in groups:
        items = [i for i in group["item_id"].tolist() if i in master_df.columns]
        if len(items) < 3:
            continue

        df_dim = master_df[items].replace(MISSING_SENTINEL, np.nan).dropna()
        if len(df_dim) < MIN_N_OMEGA:
            continue

        binary = is_binary(df_dim)
        n_boot_label = f"n_boot={n_bootstrap_binary}" if binary else f"n_boot={n_bootstrap}"
        print(f"  {questionnaire:8s} / {construct:12s} / dim {dimension} "
              f"({dim_label[:28]:<28}): "
              f"n={len(df_dim)}, k={len(items)}, "
              f"{'BINARY → tetrachoric' if binary else 'polytomous → pearson'} "
              f"[{n_boot_label}]",
              end=" ... ", flush=True)

        omega, ci_lo, ci_hi, corr_type = bootstrap_omega_v2(
            df_dim,
            n_bootstrap=n_bootstrap,
            n_bootstrap_binary=n_bootstrap_binary,
            random_state=42,
        )
        adequate = omega >= 0.70 if not np.isnan(omega) else False
        print(f"ω = {omega:.3f}" if not np.isnan(omega) else "failed")

        rows.append({
            "questionnaire": questionnaire,
            "construct": construct,
            "dimension": dimension,
            "dimension_label": dim_label,
            "n_items": len(items),
            "n_participants": len(df_dim),
            "correlation_type": corr_type,
            "omega": omega,
            "ci_lo_95": ci_lo,
            "ci_hi_95": ci_hi,
            "adequate": adequate,
        })

    omega_df = pd.DataFrame(rows)
    if omega_df.empty:
        print("  No combinations met minimum requirements.")
        return omega_df

    adequate_n = omega_df["adequate"].sum()
    print(f"\nOmega results: {adequate_n}/{len(omega_df)} dimensions with ω ≥ .70")
    tet_mask = omega_df["correlation_type"] == "tetrachoric"
    print(f"Tetrachoric correlations used for {tet_mask.sum()} binary dimension(s):")
    if tet_mask.any():
        for _, r in omega_df[tet_mask].iterrows():
            print(f"  {r['questionnaire']} dim {r['dimension']} "
                  f"({r['dimension_label']}): ω = {r['omega']:.3f}")
    return omega_df


def plot_h3_v2(omega_df: pd.DataFrame):
    """Plot omega with CI bars, colour-coded by correlation type."""
    if omega_df.empty:
        return

    omega_df = omega_df.copy()
    omega_df["label"] = (omega_df["questionnaire"] + "\n"
                         + omega_df["construct"] + " dim "
                         + omega_df["dimension"].astype(str))

    colors = []
    for _, r in omega_df.iterrows():
        if not r["adequate"]:
            colors.append("salmon")
        elif r["correlation_type"] == "tetrachoric":
            colors.append("steelblue")
        else:
            colors.append("cornflowerblue")

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(omega_df)), 5))
    x = np.arange(len(omega_df))

    ax.bar(x, omega_df["omega"], color=colors, alpha=0.85, width=0.6)

    valid_ci = omega_df[omega_df["ci_lo_95"].notna()]
    if not valid_ci.empty:
        xi = np.arange(len(omega_df))[omega_df["ci_lo_95"].notna().values]
        lo_err = valid_ci["omega"] - valid_ci["ci_lo_95"]
        hi_err = valid_ci["ci_hi_95"] - valid_ci["omega"]
        ax.errorbar(xi, valid_ci["omega"], yerr=[lo_err, hi_err],
                    fmt="none", color="black", capsize=3, elinewidth=1)

    ax.axhline(0.70, color="red", linestyle="--", linewidth=1.5,
               label="Criterion: ω = .70")
    ax.set_xticks(x)
    ax.set_xticklabels(omega_df["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("McDonald's ω (95% bootstrap CI)")
    ax.set_ylim(0, 1.05)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.85, label="Adequate (tetrachoric)"),
        Patch(facecolor="cornflowerblue", alpha=0.85, label="Adequate (Pearson)"),
        Patch(facecolor="salmon", alpha=0.85, label="Below criterion (ω < .70)"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="Criterion: ω = .70"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax.set_title(
        "H3 (v2): Internal Consistency — McDonald's Omega\n"
        "(tetrachoric correlations for binary items; Pearson for polytomous items)"
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"h3_omega{V2_SUFFIX}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved h3_omega{V2_SUFFIX}.png")


# =============================================================================
# H1 (identical to v1, different output filenames)
# =============================================================================

def run_h1(scores_df: pd.DataFrame, covariates=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if covariates is None:
        covariates = COVARIATES
    print("\n" + "=" * 70)
    print("H1: CROSS-QUESTIONNAIRE MEASUREMENT EQUIVALENCE")
    print("=" * 70)

    omnibus_rows = []
    interaction_rows = []
    pairwise_rows = []

    for (construct, dimension, dim_label), group in scores_df.groupby(
        ["construct", "dimension", "dimension_label"]
    ):
        group = group.dropna(subset=[C_SCORE_COL])
        q_counts = group["questionnaire"].value_counts()
        q_keep = q_counts[q_counts >= MIN_N_PER_QUESTIONNAIRE].index.tolist()
        if len(q_keep) < MIN_N_QUESTIONNAIRES:
            continue

        df_sub = group[group["questionnaire"].isin(q_keep)].copy()
        avail_covs = [c for c in covariates if c in df_sub.columns
                      and df_sub[c].notna().mean() > 0.5
                      and df_sub[c].nunique() > 1]   # drop constant covariates (e.g. group when single-group run)
        cov_str = " + ".join(avail_covs) if avail_covs else ""
        q_term = "C(questionnaire)"
        df_fit = df_sub.dropna(subset=[C_SCORE_COL, "questionnaire"] + avail_covs)

        # Re-check q_keep after covariate dropna: questionnaires whose participants
        # all lack a required covariate are silently absent from df_fit otherwise.
        q_counts_fit = df_fit["questionnaire"].value_counts()
        q_keep = q_counts_fit[q_counts_fit >= MIN_N_PER_QUESTIONNAIRE].index.tolist()
        df_fit = df_fit[df_fit["questionnaire"].isin(q_keep)]
        if len(q_keep) < MIN_N_QUESTIONNAIRES:
            continue

        if len(df_fit) < 20 or df_fit["questionnaire"].nunique() < 2:
            continue

        # Categorical covariates (language, site_group): added to formula directly,
        # not to avail_covs, because EMM prediction averages only numeric covariates.
        # language: replace sentinel strings ("-9") with NaN before use.
        if "language" in df_fit.columns:
            df_fit["language"] = df_fit["language"].replace({"-9": pd.NA, -9: pd.NA})
        lang_str  = (" + C(language)"  if "language"   in df_fit.columns
                     and df_fit["language"].notna().any()
                     and df_fit["language"].nunique() > 1 else "")
        site_str  = (" + C(site_group)" if "site_group" in df_fit.columns
                     and df_fit["site_group"].nunique() > 1 else "")
        cat_str   = lang_str + site_str
        # Tracks which categorical terms actually end up in the fitted omnibus
        # model, so pairwise_emm_differences() below can be given the same
        # adjustment set (see its docstring for why this matters).
        cat_covs_used = (["language"] if lang_str else []) + (["site_group"] if site_str else [])
        formula_base = (f"{C_SCORE_COL} ~ {q_term} + {cov_str}{cat_str}"
                        if cov_str else f"{C_SCORE_COL} ~ {q_term}{cat_str}")

        def _fit_anova(formula, data, q_factor):
            model = ols(formula, data=data).fit()
            anova_table = anova_lm(model, typ=2)
            if q_factor not in anova_table.index:
                return None, None
            return model, anova_table

        try:
            model_base, anova_table = _fit_anova(formula_base, df_fit, q_term)
            if model_base is None:
                continue
        except Exception:
            if not cat_str:
                continue
            # Retry without categorical terms (questionnaire collinearity)
            formula_base = (f"{C_SCORE_COL} ~ {q_term} + {cov_str}"
                            if cov_str else f"{C_SCORE_COL} ~ {q_term}")
            cat_covs_used = []
            try:
                model_base, anova_table = _fit_anova(formula_base, df_fit, q_term)
                if model_base is None:
                    continue
            except Exception as e:
                warnings.warn(f"H1 failed for {construct} dim {dimension}: {e}")
                continue

        q_factor = "C(questionnaire)"
        f_stat = anova_table.loc[q_factor, "F"]
        df_num = anova_table.loc[q_factor, "df"]
        df_den = anova_table.loc["Residual", "df"]
        p_val = anova_table.loc[q_factor, "PR(>F)"]
        eta2p = partial_eta_squared(anova_table, q_factor)

        omnibus_rows.append({
            "construct": construct, "dimension": dimension,
            "dimension_label": dim_label,
            "questionnaires": "; ".join(sorted(q_keep)),
            "n_total": len(df_fit), "n_questionnaires": len(q_keep),
            "F": f_stat, "df_num": int(df_num), "df_den": int(df_den),
            "p_value": p_val, "partial_eta2": eta2p,
            "site_adjusted": bool(site_str),
            "language_adjusted": bool(lang_str),
        })

        for cov in avail_covs:
            try:
                model_int = ols(f"{formula_base} + {q_term}:{cov}", data=df_fit).fit()
                anova_int = anova_lm(model_int, typ=2)
                int_term = f"C(questionnaire):{cov}"
                if int_term in anova_int.index:
                    interaction_rows.append({
                        "construct": construct, "dimension": dimension,
                        "dimension_label": dim_label, "covariate": cov,
                        "F": anova_int.loc[int_term, "F"],
                        "df_num": int(anova_int.loc[int_term, "df"]),
                        "df_den": int(anova_int.loc["Residual", "df"]),
                        "p_value": anova_int.loc[int_term, "PR(>F)"],
                    })
            except Exception:
                pass
            continue

        pw = pairwise_emm_differences(df_fit, "questionnaire", C_SCORE_COL, avail_covs,
                                       cat_covariates=cat_covs_used)
        if not pw.empty:
            pw["construct"] = construct
            pw["dimension"] = dimension
            pw["dimension_label"] = dim_label
            pairwise_rows.append(pw)

    omnibus_df = pd.DataFrame(omnibus_rows)
    pairwise_df = pd.concat(pairwise_rows, ignore_index=True) if pairwise_rows else pd.DataFrame()

    if not omnibus_df.empty:
        _, p_adj, _, _ = multipletests(omnibus_df["p_value"].fillna(1), method="fdr_bh")
        omnibus_df["p_value_fdr"] = p_adj

    if interaction_rows:
        int_df = pd.DataFrame(interaction_rows)
        _, p_adj, _, _ = multipletests(int_df["p_value"].fillna(1), method="fdr_bh")
        int_df["p_value_fdr"] = p_adj
        omnibus_df = omnibus_df.merge(
            int_df.groupby(["construct", "dimension"])[["p_value_fdr"]].min()
            .rename(columns={"p_value_fdr": "interaction_min_p_fdr"}),
            on=["construct", "dimension"], how="left"
        )

    print(f"\nOmnibus: {len(omnibus_df)} combinations tested, "
          f"{(omnibus_df.get('p_value_fdr', pd.Series()) < 0.05).sum()} significant (FDR < 0.05)")
    return omnibus_df, pairwise_df


def plot_h1(omnibus_df, scores_df):
    if omnibus_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, 0.6 * len(omnibus_df))))
    pivot = omnibus_df.pivot_table(
        index="dimension_label", columns="construct", values="partial_eta2"
    )
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=0, vmax=0.1, ax=ax, linewidths=0.5)
    ax.set_title("H1a: Partial η² — questionnaire effect (v2)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"h1a_omnibus_eta2{V2_SUFFIX}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for construct in omnibus_df["construct"].unique():
        omb_c = omnibus_df[omnibus_df["construct"] == construct]
        dims = omb_c["dimension_label"].unique()
        fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4), squeeze=False)
        for ax, dim_label in zip(axes[0], dims):
            sub = scores_df[
                (scores_df["construct"] == construct)
                & (scores_df["dimension_label"] == dim_label)
            ].dropna(subset=[C_SCORE_COL])
            q_means = (sub.groupby("questionnaire")[C_SCORE_COL]
                       .agg(["mean", "sem", "count"]).reset_index())
            q_means = q_means[q_means["count"] >= MIN_N_PER_QUESTIONNAIRE]
            if q_means.empty:
                continue
            ax.errorbar(q_means["questionnaire"], q_means["mean"],
                        yerr=1.645 * q_means["sem"], fmt="o", capsize=5)
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
            ax.set_title(dim_label, fontsize=10)
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle(f"H1b: {construct.upper()} (v2)", y=1.02)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"h1b_{construct}_pairwise{V2_SUFFIX}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4), squeeze=False)
        for ax, dim_label in zip(axes[0], dims):
            sub = scores_df[
                (scores_df["construct"] == construct)
                & (scores_df["dimension_label"] == dim_label)
            ].dropna(subset=[C_SCORE_COL])
            q_counts = sub["questionnaire"].value_counts()
            q_keep = q_counts[q_counts >= MIN_N_PER_QUESTIONNAIRE].index.tolist()
            sub = sub[sub["questionnaire"].isin(q_keep)]
            if sub.empty:
                continue
            sns.violinplot(data=sub, x="questionnaire", y=C_SCORE_COL,
                           inner="quartile", ax=ax, cut=0)
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
            ax.set_title(dim_label, fontsize=10)
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle(f"H1c: C-score distributions — {construct.upper()} (v2)", y=1.02)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"h1c_{construct}_distributions{V2_SUFFIX}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved H1 figures (*{V2_SUFFIX}.png)")


# =============================================================================
# H2 (identical to v1, different output filenames)
# =============================================================================

def build_cfa_model(items_by_dim):
    lines = [f"F{dim} =~ {' + '.join(items)}"
             for dim, items in items_by_dim.items()]
    return "\n".join(lines)


def cfa_fit_indices(model, n):
    try:
        stats_df = semopy.calc_stats(model)
        row = stats_df.iloc[0]
        chi2 = float(row.get("chi2", np.nan))
        dof = float(row.get("DoF", np.nan))
        cfi = float(row.get("CFI", np.nan))
        tli = float(row.get("TLI", np.nan))
        rmsea = float(row.get("RMSEA", np.nan))
        try:
            obs_cov = model.mx_cov
            impl_cov = model.calc_sigma()[0]
            d = np.sqrt(np.diag(obs_cov))
            obs_cor = obs_cov / np.outer(d, d)
            d2 = np.sqrt(np.diag(impl_cov))
            impl_cor = impl_cov / np.outer(d2, d2)
            residuals = obs_cor - impl_cor
            p = obs_cor.shape[0]
            srmr = float(np.sqrt(2 * np.sum(np.tril(residuals ** 2)) / (p * (p + 1))))
        except Exception:
            srmr = np.nan
        return {"rmsea": rmsea, "cfi": cfi, "tli": tli, "srmr": srmr, "chi2": chi2,
                "df_model": dof, "chi2_df_ratio": chi2 / dof if dof > 0 else np.nan}
    except Exception as e:
        return {"error": str(e)}


def run_h2(master_df, item_mapping):
    print("\n" + "=" * 70)
    print("H2: CONFIRMATORY FACTOR ANALYSIS")
    print("=" * 70)
    rows = []
    for (questionnaire, construct), q_map in item_mapping.groupby(["questionnaire", "construct"]):
        items_by_dim = {}
        for (dim_num, dim_label), dm in q_map.groupby(["dimension", "dimension_label"]):
            items = [i for i in dm["item_id"].tolist() if i in master_df.columns]
            if len(items) >= 2:
                items_by_dim[dim_num] = items
        if not items_by_dim:
            continue
        all_items = [i for items in items_by_dim.values() for i in items]
        df_q = master_df[all_items].replace(MISSING_SENTINEL, np.nan)
        df_cfa = df_q[df_q.notna().all(axis=1)]
        if len(df_cfa) < MIN_N_CFA:
            continue
        # semopy parses hyphens as subtraction; rename to underscores for the fit only
        rename = {c: c.replace("-", "_") for c in df_cfa.columns if "-" in c}
        df_cfa_fit = df_cfa.rename(columns=rename)
        items_by_dim_safe = {
            dim: [rename.get(i, i) for i in items]
            for dim, items in items_by_dim.items()
        }
        model_spec = build_cfa_model(items_by_dim_safe)
        try:
            model = semopy.Model(model_spec)
            model.fit(df_cfa_fit)
            fit = cfa_fit_indices(model, len(df_cfa_fit))
        except Exception as e:
            fit = {"error": str(e)}
        adequate = (fit.get("rmsea", np.nan) <= 0.08
                    and fit.get("cfi", np.nan) >= 0.90
                    and fit.get("tli", np.nan) >= 0.90
                    and fit.get("srmr", np.nan) <= 0.08)
        rows.append({
            "questionnaire": questionnaire, "construct": construct,
            "n_dimensions": len(items_by_dim), "n_items": len(all_items),
            "n_participants": len(df_cfa),
            "rmsea": fit.get("rmsea", np.nan), "cfi": fit.get("cfi", np.nan),
            "tli": fit.get("tli", np.nan), "srmr": fit.get("srmr", np.nan),
            "chi2": fit.get("chi2", np.nan), "df_model": fit.get("df_model", np.nan),
            "adequate_fit": adequate, "error": fit.get("error", ""),
        })
    cfa_df = pd.DataFrame(rows)
    if not cfa_df.empty:
        print(f"\n{(cfa_df['adequate_fit']).sum()}/{len(cfa_df)} models with adequate fit")
    return cfa_df


def plot_h2(cfa_df):
    if cfa_df.empty:
        return
    metrics = ["rmsea", "cfi", "tli", "srmr"]
    thresholds = {"rmsea": 0.08, "cfi": 0.90, "tli": 0.90, "srmr": 0.08}
    cfa_df = cfa_df.copy()
    cfa_df["model"] = cfa_df["questionnaire"] + " × " + cfa_df["construct"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), max(4, 0.5 * len(cfa_df))))
    for ax, metric in zip(axes, metrics):
        data = cfa_df.set_index("model")[[metric]]
        cmap = "RdYlGn_r" if metric in ("rmsea", "srmr") else "RdYlGn"
        sns.heatmap(data, annot=True, fmt=".2f", cmap=cmap, ax=ax, linewidths=0.5,
                    vmax=0.2 if metric == "rmsea" else 0.2 if metric == "srmr" else 1.0)
        ax.set_title(f"{metric.upper()}\n(cutoff: {thresholds[metric]})")
    fig.suptitle("H2: CFA Fit Indices (v2)\nRMSEA≤.08 | CFI≥.90 | TLI≥.90 | SRMR≤.08", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"h2_cfa_fit{V2_SUFFIX}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved h2_cfa_fit{V2_SUFFIX}.png")


# =============================================================================
# Main
# =============================================================================

def main():
    global OUTPUT_DIR, SCORE_DIR, DIMENSION_SCORES, MASTER_TABLE, ITEM_MAPPING, V2_SUFFIX, C_SCORE_COL

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", action="store_true",
                        help="Use binary-format pipeline outputs")
    parser.add_argument("--group", type=int, choices=[0, 1], default=None,
                        help="Restrict all analyses to a single group: 0=HC, 1=PD patients")
    parser.add_argument("--reduced-covariates", action="store_true",
                        help="H1 with language+age+sex+group only (covariates available for all datasets)")
    parser.add_argument("--score-dir", type=Path, default=None,
                        help="Directory containing dimension_scores.csv (default: "
                             "output/score-transformation/). Point this at a scratch "
                             "directory when validating against run_pipeline.py's "
                             "--output-dir, so real committed results are never overwritten.")
    parser.add_argument("--master-table", type=Path, default=None,
                        help="Path to the master table used for the H2 CFA "
                             "(default: input/score-transformation/master-table_subject-data.csv)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write results (default: "
                             "output/score-transformation/hypotheses-testing/)")
    args = parser.parse_args()

    if args.binary:
        binary_dir = PROJECT_DIR / "output" / "score-transformation-binary"
        SCORE_DIR = binary_dir
        DIMENSION_SCORES = binary_dir / "dimension_scores.csv"
        MASTER_TABLE = binary_dir / "master-table_binary.csv"
        ITEM_MAPPING = binary_dir / "item_dimension_mapping_binary.csv"
        OUTPUT_DIR = PROJECT_DIR / "output" / "score-transformation" / "hypotheses-testing-binary"
        V2_SUFFIX = "_binary"

    # Explicit overrides win over --binary and the defaults, so a test/synthetic
    # run can be pointed at scratch input/output locations independently.
    if args.score_dir is not None:
        SCORE_DIR = args.score_dir
        DIMENSION_SCORES = args.score_dir / "dimension_scores.csv"
    if args.master_table is not None:
        MASTER_TABLE = args.master_table
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ENIGMA-PD VALIDATION ANALYSES v2")
    print("(tetrachoric omega for binary-response dimensions)")
    if args.binary:
        print("MODE: BINARY")
    print("=" * 70)
    print(f"\nScore dir:    {SCORE_DIR}")
    print(f"Master table: {MASTER_TABLE}")
    print(f"Output dir:   {OUTPUT_DIR}")

    if not DIMENSION_SCORES.exists():
        raise FileNotFoundError(
            f"dimension_scores.csv not found at {DIMENSION_SCORES}.\n"
            "Run run_pipeline.py first (see --output-dir if you pointed "
            "--score-dir at a non-default location)."
        )

    print("\nLoading dimension scores...")
    scores_df = pd.read_csv(DIMENSION_SCORES)
    scores_df["site_group"] = assign_site_group(scores_df["site"])
    print(f"  {len(scores_df)} rows, {scores_df['participant_id'].nunique()} participants")

    print("Loading master table...")
    master_df = pd.read_csv(MASTER_TABLE)
    print(f"  {len(master_df)} participants, {len(master_df.columns)} columns")

    h1_covariates = COVARIATES
    if args.reduced_covariates:
        h1_covariates = REDUCED_COVARIATES
        V2_SUFFIX = "_reduced-covariates"
        print("\nREDUCED COVARIATES MODE: age + sex + group (+ language & site as categorical)")

    if args.group is not None:
        group_label = "pd" if args.group == 1 else "hc"
        V2_SUFFIX = f"_{group_label}{V2_SUFFIX}"
        scores_df = scores_df[scores_df["group"] == args.group].copy()
        master_df = master_df[master_df["group"] == args.group].copy()
        print(f"\nFiltered to group={args.group} ({group_label.upper()}): "
              f"{scores_df['participant_id'].nunique()} participants in scores, "
              f"{len(master_df)} in master table")

    item_mapping = pd.read_csv(ITEM_MAPPING)

    # H1 (identical to v1)
    omnibus_df, pairwise_df = run_h1(scores_df, covariates=h1_covariates)
    omnibus_df.to_csv(OUTPUT_DIR / f"h1_omnibus{V2_SUFFIX}.csv", index=False)
    if not pairwise_df.empty:
        pairwise_df.to_csv(OUTPUT_DIR / f"h1_pairwise{V2_SUFFIX}.csv", index=False)
    print(f"\nSaved h1_omnibus{V2_SUFFIX}.csv, h1_pairwise{V2_SUFFIX}.csv")
    plot_h1(omnibus_df, scores_df)

    # H2 (identical to v1)
    cfa_df = run_h2(master_df, item_mapping)
    if not cfa_df.empty:
        cfa_df.to_csv(OUTPUT_DIR / f"h2_cfa_fit{V2_SUFFIX}.csv", index=False)
        print(f"Saved h2_cfa_fit{V2_SUFFIX}.csv")
        plot_h2(cfa_df)

    # H3 (v2: tetrachoric for binary dimensions)
    omega_df = run_h3_v2(master_df, item_mapping,
                         n_bootstrap=1000, n_bootstrap_binary=500)
    if not omega_df.empty:
        omega_df.to_csv(OUTPUT_DIR / f"h3_omega{V2_SUFFIX}.csv", index=False)
        print(f"\nSaved h3_omega{V2_SUFFIX}.csv")
        plot_h3_v2(omega_df)

    print("\n" + "=" * 70)
    print("VALIDATION ANALYSES v2 COMPLETE")
    print(f"Outputs saved to {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
