"""
Exploratory Validation Analyses — ENIGMA-PD
=============================================

Five secondary/exploratory validation checks for the score-transformation
pipeline, distinct from the core H1-H3 hypothesis tests in
`hypothesis-testing.py`:

  1. Convergent / discriminant validity — do same-dimension scores correlate
     across questionnaires (convergent) more than different-dimension scores
     on the same construct (discriminant)?
  2. Cross-questionnaire within-person prediction — does one questionnaire's
     dimension score predict another's, after covariate adjustment?
  3. Language effects — do CDF-transformed scores differ by language, and how
     does that compare in size to site effects?
  4. Nomological network validation — do different questionnaires on the same
     dimension show consistent PD-vs-HC group effects?
  5. Rank preservation — does the CDF transform preserve participant rankings
     relative to the raw total questionnaire score?

Usage:
------
python exploratory-validation-analyses.py                        # run all five
python exploratory-validation-analyses.py --binary               # cross-questionnaire
                                                                    prediction & nomological
                                                                    validation use binary-format
                                                                    pipeline outputs
python exploratory-validation-analyses.py --reduced-covariates   # cross-questionnaire prediction:
                                                                    age+sex+group+language+site only
python exploratory-validation-analyses.py --out-dir <path>       # language-effects table output dir
python exploratory-validation-analyses.py \
    --score-dir /tmp/synthetic-test-output \
    --output-dir /tmp/synthetic-test-output/validation       # read + write from a scratch
                                                                 location (e.g. testing against
                                                                 run_pipeline.py's --output-dir),
                                                                 so real committed results are
                                                                 never overwritten

Outputs:
  output/score-transformation/convergent-discriminant-validity/
  output/score-transformation/cross-questionnaire-prediction/
  output/results_figures_tables/Table2_language_effects.xlsx  (override with --out-dir)
  output/score-transformation/nomological-validation/
  output/score-transformation/rank-preservation/
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations
from pathlib import Path
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

warnings.filterwarnings("ignore")

# =============================================================================
# Shared paths
# =============================================================================

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent.parent.parent
# SCORE_DIR / DIMENSION_SCORES / BASE_OUTPUT_DIR are read fresh (as globals) by each
# run_*() function, so `main()`'s --score-dir / --output-dir can redirect all five
# analyses at once -- e.g. to validate against a synthetic/test run without ever
# touching the committed real results in output/score-transformation/.
SCORE_DIR = PROJECT_DIR / "output" / "score-transformation"
DIMENSION_SCORES = SCORE_DIR / "dimension_scores.csv"
BASE_OUTPUT_DIR = PROJECT_DIR / "output" / "score-transformation"


def assign_site_group(site_series: pd.Series, site_top_n: int = 50) -> pd.Series:
    """Sites with < site_top_n participants get pooled into 'other'."""
    counts = site_series.value_counts()
    major = counts[counts >= site_top_n].index
    return site_series.apply(lambda s: s if s in major else "other")


# =============================================================================
# 1. Convergent / Discriminant Validity
# =============================================================================
#
# Tests semantic harmonization in ENIGMA-PD participants who completed more
# than one questionnaire on the same construct/dimension.
#
#   Convergent validity  — correlation of proportion scores between the two
#                          questionnaires on the SAME dimension.
#   Discriminant validity — correlation between the two questionnaires on
#                          DIFFERENT dimensions (requires pairs that overlap
#                          on ≥ 2 dimensions, e.g. HADS × MDS-UPDRS_I which
#                          both cover depression AND anxiety).
#
# Uses proportion_score (before CDF harmonisation) so the test reflects the
# semantic mapping itself rather than the downstream score alignment.
#
# Output (in output/score-transformation/convergent-discriminant-validity/):
#   convergent_discriminant_enigma.csv   — all pairwise correlations
#   convergent_discriminant_heatmap_<construct>.png
#   convergent_discriminant_summary.png  — scatter: convergent vs discriminant r

CONV_SCORE_COL = "proportion_score"
CONV_MIN_N = 50   # minimum overlapping participants for a valid correlation


def _cdv_parse_col(c):
    q, dk = c.split("|", 1)
    parts = dk.rsplit("_dim", 1)
    return q, parts[0], int(parts[1])   # questionnaire, construct, dimension


def run_convergent_discriminant_validity():
    CONV_OUTPUT_DIR = BASE_OUTPUT_DIR / "convergent-discriminant-validity"
    CONV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CONVERGENT / DISCRIMINANT VALIDITY — ENIGMA-PD")
    print("=" * 70)

    scores = pd.read_csv(DIMENSION_SCORES)
    scores = scores[scores[CONV_SCORE_COL].notna()].copy()
    scores["dim_key"] = scores["construct"] + "_dim" + scores["dimension"].astype(str)

    print(f"\nTotal rows: {len(scores)}")
    print(f"Participants: {scores['participant_id'].nunique()}")
    print(f"Score column: {CONV_SCORE_COL}\n")

    # Pivot to wide: one row per participant, columns = (questionnaire, dim_key)
    # Keep one row per participant × questionnaire × dim_key
    # (drop duplicate rows if any — e.g. multiple sites)
    dedup = (scores.groupby(["participant_id", "questionnaire", "dim_key"])[CONV_SCORE_COL]
             .mean().reset_index())

    wide = dedup.pivot_table(
        index="participant_id",
        columns=["questionnaire", "dim_key"],
        values=CONV_SCORE_COL,
    )
    wide.columns = [f"{q}|{d}" for q, d in wide.columns]

    print(f"Wide matrix: {wide.shape[0]} participants × {wide.shape[1]} Q×dim columns\n")

    # ── Compute pairwise correlations ───────────────────────────────────────
    cols = wide.columns.tolist()
    rows = []

    col_meta = {c: _cdv_parse_col(c) for c in cols}

    # All questionnaire pairs that share ≥1 dimension key
    qs_all = sorted({m[0] for m in col_meta.values()})

    for q1, q2 in combinations(qs_all, 2):
        q1_cols = [c for c in cols if col_meta[c][0] == q1]
        q2_cols = [c for c in cols if col_meta[c][0] == q2]

        for c1 in q1_cols:
            _, construct1, dim1 = col_meta[c1]
            for c2 in q2_cols:
                _, construct2, dim2 = col_meta[c2]

                pair = wide[[c1, c2]].dropna()
                n = len(pair)
                if n < CONV_MIN_N:
                    continue

                r = pair[c1].corr(pair[c2])

                same_construct = construct1 == construct2
                same_dimension = same_construct and dim1 == dim2

                validity_type = (
                    "convergent"   if same_dimension  else
                    "discriminant" if same_construct  else
                    "cross-construct"
                )

                rows.append({
                    "q1": q1, "q2": q2,
                    "construct1": construct1, "dim1": dim1,
                    "construct2": construct2, "dim2": dim2,
                    "dim_label1": scores.loc[
                        scores["questionnaire"].eq(q1) & scores["dim_key"].eq(
                            f"{construct1}_dim{dim1}"), "dimension_label"
                    ].iloc[0] if len(scores.loc[
                        scores["questionnaire"].eq(q1) & scores["dim_key"].eq(
                            f"{construct1}_dim{dim1}")]) > 0 else "",
                    "dim_label2": scores.loc[
                        scores["questionnaire"].eq(q2) & scores["dim_key"].eq(
                            f"{construct2}_dim{dim2}"), "dimension_label"
                    ].iloc[0] if len(scores.loc[
                        scores["questionnaire"].eq(q2) & scores["dim_key"].eq(
                            f"{construct2}_dim{dim2}")]) > 0 else "",
                    "n": n,
                    "r": round(r, 3),
                    "validity_type": validity_type,
                })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        print("No overlapping pairs found above MIN_N threshold.")
        return

    result_df.to_csv(CONV_OUTPUT_DIR / "convergent_discriminant_enigma.csv", index=False)
    print(f"Saved convergent_discriminant_enigma.csv ({len(result_df)} pairs)\n")

    # ── Print summary by validity type ──────────────────────────────────────
    for vtype in ["convergent", "discriminant", "cross-construct"]:
        sub = result_df[result_df["validity_type"] == vtype]
        if sub.empty:
            continue
        print(f"{vtype.upper()} ({len(sub)} pairs, n range {sub['n'].min()}–{sub['n'].max()}):")
        print(f"  r: mean={sub['r'].mean():.3f}  median={sub['r'].median():.3f}  "
              f"range=[{sub['r'].min():.3f}, {sub['r'].max():.3f}]")
        if vtype == "convergent":
            print("  Top pairs:")
            for _, row in sub.nlargest(5, "n").iterrows():
                print(f"    {row['q1']} × {row['q2']}  {row['construct1']} dim{row['dim1']}"
                      f"  r={row['r']:.3f}  n={row['n']}")
        print()

    # ── Per-construct heatmap for questionnaire pairs with multi-dim overlap ─
    for construct in sorted(result_df["construct1"].unique()):
        sub = result_df[
            (result_df["construct1"] == construct) &
            (result_df["construct2"] == construct)
        ].copy()
        if sub.empty:
            continue

        # Find pairs that appear for ≥2 dimensions → proper MTMM comparison
        pair_dims = sub.groupby(["q1", "q2"])["dim1"].nunique()
        mtmm_pairs = pair_dims[pair_dims >= 2].index.tolist()

        if not mtmm_pairs:
            # Just plot convergent correlations as bar chart
            conv = sub[sub["validity_type"] == "convergent"].copy()
            if conv.empty:
                continue
            conv["label"] = conv["q1"] + " × " + conv["q2"] + "\n" + \
                            conv["dim_label1"].str[:25]
            conv = conv.sort_values("r", ascending=True)

            fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(conv) + 1)))
            bars = ax.barh(range(len(conv)), conv["r"], color="#4A90D9", alpha=0.8)
            ax.set_yticks(range(len(conv)))
            ax.set_yticklabels(conv["label"], fontsize=8)
            ax.set_xlabel("Pearson r (proportion score)", fontsize=9)
            ax.set_title(f"Convergent validity — {construct.upper()}\n"
                         f"(participants completing both questionnaires, min n={CONV_MIN_N})",
                         fontsize=9)
            ax.axvline(0, color="black", lw=0.5)
            ax.set_xlim(0, 1.05)
            for i, (_, row) in enumerate(conv.iterrows()):
                ax.text(row["r"] + 0.01, i, f"n={row['n']}", va="center", fontsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            fig.savefig(CONV_OUTPUT_DIR / f"convergent_discriminant_heatmap_{construct}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved convergent_discriminant_heatmap_{construct}.png")
            continue

        # Build MTMM-style matrix for pairs with multi-dimension overlap
        # Rows/cols: questionnaire × dimension labels
        all_qdims = sorted(set(
            [(row.q1, row.construct1, row.dim1, row.dim_label1) for _, row in sub.iterrows()] +
            [(row.q2, row.construct2, row.dim2, row.dim_label2) for _, row in sub.iterrows()]
        ))
        # De-duplicate by (q, dim)
        seen = set()
        qdims = []
        for item in all_qdims:
            key = (item[0], item[2])
            if key not in seen:
                seen.add(key)
                qdims.append(item)
        qdims = sorted(qdims, key=lambda x: (x[0], x[2]))  # sort by questionnaire then dim

        n_qdims = len(qdims)
        mat = np.full((n_qdims, n_qdims), np.nan)
        nmat = np.full((n_qdims, n_qdims), np.nan)

        qdim_index = {(q, d): i for i, (q, _, d, _) in enumerate(qdims)}
        np.fill_diagonal(mat, 1.0)

        for _, row in sub.iterrows():
            i = qdim_index.get((row.q1, row.dim1))
            j = qdim_index.get((row.q2, row.dim2))
            if i is not None and j is not None:
                mat[i, j] = row.r
                mat[j, i] = row.r
                nmat[i, j] = row.n
                nmat[j, i] = row.n

        labels = [f"{q}\ndim{d}" for q, _, d, _ in qdims]

        fig, ax = plt.subplots(figsize=(2 + 0.7 * n_qdims, 2 + 0.7 * n_qdims))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

        # Annotate cells
        for i in range(n_qdims):
            for j in range(n_qdims):
                if not np.isnan(mat[i, j]):
                    n_ij = nmat[i, j]
                    txt = f"{mat[i,j]:.2f}"
                    if not np.isnan(n_ij):
                        txt += f"\nn={int(n_ij)}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                            color="white" if abs(mat[i, j]) > 0.5 else "black")

        # Draw questionnaire group borders
        qs_order = [q for q, _, _, _ in qdims]
        prev_q = qs_order[0]
        for idx in range(1, n_qdims):
            if qs_order[idx] != prev_q:
                ax.axhline(idx - 0.5, color="black", lw=1.5)
                ax.axvline(idx - 0.5, color="black", lw=1.5)
                prev_q = qs_order[idx]

        ax.set_xticks(range(n_qdims))
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(n_qdims))
        ax.set_yticklabels(labels, fontsize=8)
        plt.colorbar(im, ax=ax, label="Pearson r", fraction=0.03, pad=0.04)
        ax.set_title(
            f"Convergent / discriminant validity — {construct.upper()}\n"
            f"Diagonal blocks = same questionnaire (convergent expected high); "
            f"off-diagonal = discriminant",
            fontsize=9
        )
        plt.tight_layout()
        fig.savefig(CONV_OUTPUT_DIR / f"convergent_discriminant_heatmap_{construct}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved convergent_discriminant_heatmap_{construct}.png")

    # ── Summary scatter: convergent vs discriminant r per questionnaire pair ─
    pair_summary = []
    for (q1, q2), grp in result_df.groupby(["q1", "q2"]):
        conv = grp[grp["validity_type"] == "convergent"]["r"].tolist()
        disc = grp[grp["validity_type"] == "discriminant"]["r"].tolist()
        if not conv or not disc:
            continue
        pair_summary.append({
            "pair": f"{q1}\n× {q2}",
            "construct": grp.iloc[0]["construct1"],
            "mean_convergent":   np.mean(conv),
            "mean_discriminant": np.mean(disc),
            "n_conv":  len(conv),
            "n_disc":  len(disc),
            "mean_n":  grp["n"].mean(),
        })

    if pair_summary:
        ps_df = pd.DataFrame(pair_summary)
        constructs = sorted(ps_df["construct"].unique())
        pal = dict(zip(constructs, sns.color_palette("tab10", len(constructs))))

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="y = x")
        for _, row in ps_df.iterrows():
            color = pal.get(row["construct"], "gray")
            ax.scatter(row["mean_convergent"], row["mean_discriminant"],
                       s=80, color=color, zorder=4, alpha=0.85)
            ax.annotate(row["pair"], (row["mean_convergent"], row["mean_discriminant"]),
                        fontsize=6.5, xytext=(4, 4), textcoords="offset points", alpha=0.8)

        legend_handles = [mpatches.Patch(color=pal[c], label=c) for c in constructs]
        ax.legend(handles=legend_handles, fontsize=8, title="Construct")
        ax.set_xlabel("Mean convergent r (same dimension)", fontsize=9)
        ax.set_ylabel("Mean discriminant r (different dimension, same construct)", fontsize=9)
        ax.set_title("Convergent vs discriminant validity — ENIGMA-PD\n"
                     "Points below diagonal: questionnaires are more similar within dimension\n"
                     "than across dimensions (good discriminant validity)",
                     fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(CONV_OUTPUT_DIR / "convergent_discriminant_summary.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved convergent_discriminant_summary.png")

    print(f"\n{'='*70}")
    print(f"Done. Outputs → {CONV_OUTPUT_DIR}")
    print(f"{'='*70}")


# =============================================================================
# 2. Cross-Questionnaire Within-Person Prediction
# =============================================================================
#
# For every pair of questionnaires measuring the same construct where the same
# participants completed both instruments, tests whether one questionnaire's
# dimension score predicts the other's — after residualising both scores on
# age, sex, group (PD/HC), and site.
#
# If the cross-prediction is positive and substantial (high r, large ΔR²),
# the two questionnaires capture the same individual variation on that
# dimension, validating the item-dimension mapping. A near-zero or negative
# cross-prediction signals that the mapping is problematic.
#
# Only pairs/dimensions with n ≥ MIN_N overlapping participants are included.
#
# Outputs (in output/score-transformation/cross-questionnaire-prediction/):
#   cross_prediction_results.csv  — per pair × dimension statistics
#   cross_prediction_{construct}.png  — scatter plots (residualised scores)

CROSS_PRED_CONSTRUCTS = ["depression", "anxiety", "sleep"]
CROSS_PRED_MIN_N = 30
CROSS_PRED_SITE_TOP_N = 50
CROSS_PRED_SCORE_COL = "av_cdf_score"
CROSS_PRED_EXTRA_COVARIATES = ["disease_duration", "HY_stage", "LEDD_total", "MDS_UPDRS_III_total_score"]


def find_overlapping_pairs(df: pd.DataFrame) -> list[dict]:
    """Find all questionnaire pairs with ≥ MIN_N shared participants and ≥1 shared dimension."""
    pairs = []
    for construct, cdf in df.groupby("construct"):
        if construct not in CROSS_PRED_CONSTRUCTS:
            continue
        qs = sorted(cdf["questionnaire"].unique())
        for q1, q2 in combinations(qs, 2):
            pids1 = set(cdf[cdf["questionnaire"] == q1]["participant_id"])
            pids2 = set(cdf[cdf["questionnaire"] == q2]["participant_id"])
            overlap = pids1 & pids2
            if len(overlap) < CROSS_PRED_MIN_N:
                continue
            dims1 = set(cdf[cdf["questionnaire"] == q1]["dimension"])
            dims2 = set(cdf[cdf["questionnaire"] == q2]["dimension"])
            shared = sorted(dims1 & dims2)
            if shared:
                pairs.append({
                    "construct":   construct,
                    "q1": q1, "q2": q2,
                    "n_overlap":   len(overlap),
                    "shared_dims": shared,
                })
    return pairs


def cross_predict(df: pd.DataFrame, construct: str, q1: str, q2: str,
                  dim: int, extra_covariates=None,
                  add_language: bool = False) -> dict | None:
    """
    Regress q2's score on q1's score (+ covariates) for overlapping participants.
    Also runs the reverse direction. Returns stats for both directions.
    """
    if extra_covariates is None:
        extra_covariates = CROSS_PRED_EXTRA_COVARIATES
    cdf = df[df["construct"] == construct]
    extra_cols = [c for c in extra_covariates if c in df.columns]
    lang_cols = ["language"] if add_language and "language" in df.columns else []
    base_cols = (["participant_id", CROSS_PRED_SCORE_COL, "age", "sex", "group", "site_group"]
                  + lang_cols + extra_cols)

    # Keep only rows where participant answered this specific dimension in q1
    d1_dim = cdf[(cdf["questionnaire"] == q1) & (cdf["dimension"] == dim)][
        base_cols
    ].rename(columns={CROSS_PRED_SCORE_COL: "s1"})
    d2 = cdf[(cdf["questionnaire"] == q2) & (cdf["dimension"] == dim)][
        ["participant_id", CROSS_PRED_SCORE_COL]
    ].rename(columns={CROSS_PRED_SCORE_COL: "s2"})

    merged = d1_dim.merge(d2, on="participant_id").dropna(
        subset=["s1", "s2", "age", "sex", "group"]
    )
    if len(merged) < CROSS_PRED_MIN_N:
        return None

    dim_label = cdf[cdf["dimension"] == dim]["dimension_label"].iloc[0]

    # Build covariate formula: required terms + site + language + extra numeric covariates
    cov_terms = ["age", "C(sex)", "C(group)"]
    if merged["site_group"].nunique() > 1:
        cov_terms.append("C(site_group)")
    if add_language and "language" in merged.columns:
        merged["language"] = merged["language"].replace({"-9": np.nan, -9: np.nan})
        if merged["language"].notna().any() and merged["language"].nunique() > 1:
            cov_terms.append("C(language)")
    for ec in extra_cols:
        col_vals = pd.to_numeric(merged[ec], errors="coerce")
        if col_vals.notna().sum() >= CROSS_PRED_MIN_N and col_vals.nunique() > 1:
            merged[ec] = col_vals
            cov_terms.append(ec)
        else:
            merged = merged.drop(columns=[ec])
    cov = " + ".join(cov_terms)

    def _fit(y_col, x_col):
        m0 = ols(f"{y_col} ~ {cov}", data=merged).fit()
        m1 = ols(f"{y_col} ~ {x_col} + {cov}", data=merged).fit()
        coef = m1.params[x_col]
        ci = m1.conf_int().loc[x_col]
        pval = m1.pvalues[x_col]
        dr2 = m1.rsquared - m0.rsquared
        # Residual correlation
        r1 = ols(f"s1 ~ {cov}", data=merged).fit().resid
        r2 = ols(f"s2 ~ {cov}", data=merged).fit().resid
        r, _ = stats.pearsonr(r1, r2)
        return dict(coef=coef, ci_lo=ci[0], ci_hi=ci[1],
                    p=pval, delta_r2=dr2, r=r,
                    resid_x=r1 if x_col == "s1" else r2,
                    resid_y=r2 if x_col == "s1" else r1)

    fwd = _fit("s2", "s1")   # q1 → q2
    rev = _fit("s1", "s2")   # q2 → q1

    return {
        "construct":      construct,
        "dimension":      dim,
        "dimension_label": dim_label,
        "q1": q1, "q2": q2,
        "n":              len(merged),
        # Forward: q1 predicts q2
        "r":              round(fwd["r"],   3),
        "beta_fwd":       round(fwd["coef"], 3),
        "ci_lo_fwd":      round(fwd["ci_lo"], 3),
        "ci_hi_fwd":      round(fwd["ci_hi"], 3),
        "p_fwd":          round(fwd["p"],   4),
        "delta_r2_fwd":   round(fwd["delta_r2"], 3),
        # Reverse: q2 predicts q1
        "beta_rev":       round(rev["coef"], 3),
        "ci_lo_rev":      round(rev["ci_lo"], 3),
        "ci_hi_rev":      round(rev["ci_hi"], 3),
        "p_rev":          round(rev["p"],   4),
        "delta_r2_rev":   round(rev["delta_r2"], 3),
        # Residuals for plotting
        "_resid_q1": fwd["resid_x"],
        "_resid_q2": fwd["resid_y"],
        "_merged":   merged,
    }


def plot_cross_prediction_construct(results: list[dict], construct: str,
                                    output_dir: Path, cov_label: str = ""):
    panels = [r for r in results if r["construct"] == construct]
    if not panels:
        return

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.8), squeeze=False)

    for ax, res in zip(axes[0], panels):
        r1 = res["_resid_q1"]
        r2 = res["_resid_q2"]
        r = res["r"]

        color = "#E74C3C" if r < 0 else "#2980B9"
        ax.scatter(r1, r2, alpha=0.2, s=8, color=color)

        xr = np.linspace(r1.min(), r1.max(), 100)
        sl, ic, *_ = stats.linregress(r1, r2)
        line_color = "#E74C3C" if r < 0 else "#C0392B"
        ax.plot(xr, sl * xr + ic, color=line_color, linewidth=1.8)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")

        ax.set_xlabel(f'{res["q1"]} (residualised)', fontsize=8)
        ax.set_ylabel(f'{res["q2"]} (residualised)', fontsize=8)
        ax.set_title(
            f'Dim {res["dimension"]}: {res["dimension_label"][:30]}\n'
            f'r = {r:.3f},  β = {res["beta_fwd"]:+.3f} '
            f'[{res["ci_lo_fwd"]:+.3f}, {res["ci_hi_fwd"]:+.3f}]\n'
            f'ΔR² = {res["delta_r2_fwd"]:.3f},  n = {res["n"]}',
            fontsize=8,
        )

    cov_note = cov_label or "age, sex, group, site, disease duration, HY stage, LEDD, UPDRS-III"
    fig.suptitle(
        f"{construct.capitalize()}: within-person cross-questionnaire prediction\n"
        f"(scores residualised on {cov_note})",
        fontsize=10, y=1.03,
    )
    plt.tight_layout()
    fname = output_dir / f"cross_prediction_{construct}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved cross_prediction_{construct}.png")


def run_cross_questionnaire_prediction(binary: bool = False, reduced_covariates: bool = False):
    score_dir = SCORE_DIR
    dimension_scores = DIMENSION_SCORES
    output_dir = BASE_OUTPUT_DIR / "cross-questionnaire-prediction"

    if binary:
        binary_dir = PROJECT_DIR / "output" / "score-transformation-binary"
        score_dir = binary_dir
        dimension_scores = binary_dir / "dimension_scores.csv"
        output_dir = BASE_OUTPUT_DIR / "cross_questionnaire_prediction-binary"

    if reduced_covariates:
        output_dir = BASE_OUTPUT_DIR / "cross_questionnaire_prediction-reduced-covariates"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CROSS-QUESTIONNAIRE WITHIN-PERSON PREDICTION")
    if binary:
        print("MODE: BINARY")
    if reduced_covariates:
        print("COVARIATES: age + sex + group + language + site (reduced)")
    print("=" * 70)

    df = pd.read_csv(dimension_scores)
    df["site_group"] = assign_site_group(df["site"], site_top_n=CROSS_PRED_SITE_TOP_N)
    df["group"] = pd.to_numeric(df["group"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex"] = pd.to_numeric(df["sex"], errors="coerce")

    extra_covariates = [] if reduced_covariates else CROSS_PRED_EXTRA_COVARIATES
    add_language = reduced_covariates

    cov_label = ("age, sex, group, language, site"
                 if reduced_covariates
                 else "age, sex, group, site, disease duration, HY stage, LEDD, UPDRS-III")

    print("\nFinding overlapping questionnaire pairs ...")
    pairs = find_overlapping_pairs(df)
    print(f"  {len(pairs)} pairs with shared participants and dimensions:")
    for p in pairs:
        print(f"    {p['construct']:12s}  {p['q1']:15s} × {p['q2']:15s}"
              f"  n={p['n_overlap']}  dims={p['shared_dims']}")

    print("\nRunning cross-predictions ...")
    all_results = []
    for p in pairs:
        for dim in p["shared_dims"]:
            res = cross_predict(df, p["construct"], p["q1"], p["q2"], dim,
                                extra_covariates=extra_covariates,
                                add_language=add_language)
            if res is not None:
                all_results.append(res)

    if not all_results:
        print("  No results computed.")
        return

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS  (forward direction: q1 → q2)")
    print("=" * 70)
    for construct in CROSS_PRED_CONSTRUCTS:
        c_res = [r for r in all_results if r["construct"] == construct]
        if not c_res:
            continue
        print(f"\n{construct.upper()}")
        for r in c_res:
            flag = "  *** NEGATIVE ***" if r["r"] < 0 else ""
            print(f"  {r['q1']:15s} × {r['q2']:15s}  "
                  f"dim{r['dimension']} ({r['dimension_label'][:28]})\n"
                  f"    r={r['r']:+.3f}  β={r['beta_fwd']:+.3f} "
                  f"[{r['ci_lo_fwd']:+.3f},{r['ci_hi_fwd']:+.3f}]  "
                  f"ΔR²={r['delta_r2_fwd']:.3f}  p={r['p_fwd']:.4f}"
                  f"  n={r['n']}{flag}")

    # Save CSV (drop plotting artefacts)
    csv_cols = [k for k in all_results[0] if not k.startswith("_")]
    pd.DataFrame([{k: r[k] for k in csv_cols} for r in all_results]).to_csv(
        output_dir / "cross_prediction_results.csv", index=False
    )
    print(f"\n  Saved cross_prediction_results.csv ({len(all_results)} rows)")

    print("\nGenerating figures ...")
    for construct in CROSS_PRED_CONSTRUCTS:
        plot_cross_prediction_construct(all_results, construct, output_dir, cov_label=cov_label)

    print(f"\n{'='*70}")
    print(f"Done. Outputs saved to {output_dir}")
    print(f"{'='*70}")


# =============================================================================
# 3. Language Effects
# =============================================================================
#
# For each questionnaire (with ≥2 languages and ≥20 participants per language),
# tests whether the CDF-transformed dimension scores differ between language
# groups.
#
# Per questionnaire × averaged across dimensions:
#   - One-way ANOVA: score ~ C(language)  → F, p, η²(language)
#   - Site-only model: score ~ C(site)    → η²(site)
#   - Incremental R² when site added to language model → % var site within language
#   - Site:language ratio = η²(site) / η²(language)
#
# Output:
#   output/results_figures_tables/Table2_language_effects.xlsx (override with --out-dir)

LANG_SCORE_COL = "av_cdf_score"
LANG_MIN_N_LANG = 20   # minimum participants per language group


def fmt_p(p):
    if pd.isna(p):
        return ""
    if p < .001:
        return "<.001"
    return f"{p:.3f}"


def analyse_questionnaire_language(q_df: pd.DataFrame) -> dict | None:
    q_df = q_df.copy()

    # Mean score per participant across all dimensions
    participant_scores = (
        q_df.groupby(["participant_id", "language", "site"])[LANG_SCORE_COL]
        .mean()
        .reset_index()
        .rename(columns={LANG_SCORE_COL: "score"})
    )
    participant_scores = participant_scores.dropna(subset=["score", "language"])

    # Drop sentinel language values
    participant_scores = participant_scores[
        ~participant_scores["language"].isin(["-9", -9, ""])
    ]

    # Only keep languages with sufficient n
    lang_counts = participant_scores["language"].value_counts()
    valid_langs = lang_counts[lang_counts >= LANG_MIN_N_LANG].index
    participant_scores = participant_scores[
        participant_scores["language"].isin(valid_langs)
    ]

    if participant_scores["language"].nunique() < 2:
        return None

    n = len(participant_scores)
    languages = sorted(participant_scores["language"].unique())
    n_langs = len(languages)

    # ── Language-only model ─────────────────────────────────────────────────
    m_lang = ols("score ~ C(language)", data=participant_scores).fit()
    anova_lang = anova_lm(m_lang, typ=1)
    ss_lang = anova_lang.loc["C(language)", "sum_sq"]
    ss_resid = anova_lang.loc["Residual", "sum_sq"]
    ss_total = ss_lang + ss_resid
    F_lang = anova_lang.loc["C(language)", "F"]
    p_lang = anova_lang.loc["C(language)", "PR(>F)"]
    eta2_lang = ss_lang / ss_total

    # ── Site-only model ─────────────────────────────────────────────────────
    if participant_scores["site"].nunique() > 1:
        m_site = ols("score ~ C(site)", data=participant_scores).fit()
        eta2_site = m_site.rsquared
    else:
        eta2_site = eta2_lang   # only one site → same as language

    # ── Language + site model (incremental site variance) ──────────────────
    if participant_scores["site"].nunique() > 1:
        m_lang_site = ols("score ~ C(language) + C(site)",
                          data=participant_scores).fit()
        pct_site_within_lang = (m_lang_site.rsquared - m_lang.rsquared) * 100
    else:
        pct_site_within_lang = 0.0

    ratio = eta2_site / eta2_lang if eta2_lang > 0 else np.nan

    return {
        "Languages":                   ", ".join(languages),
        "N languages":                 n_langs,
        "N sites":                     participant_scores["site"].nunique(),
        "N":                           n,
        "F":                           round(float(F_lang), 2),
        "p":                           fmt_p(float(p_lang)),
        "η² (language)":               round(eta2_lang, 3),
        "% var language":              round(eta2_lang * 100, 1),
        "% var site within language":  round(pct_site_within_lang, 1),
        "η² (site)":                   round(eta2_site, 3),
        "Site:language ratio":         round(ratio, 1),
    }


def run_language_effects(out_dir: Path | None = None):
    out_dir = out_dir or (PROJECT_DIR / "output" / "results_figures_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LANGUAGE EFFECTS ANALYSIS")
    print("=" * 70)

    scores = pd.read_csv(SCORE_DIR / "dimension_scores.csv")
    scores = scores[scores[LANG_SCORE_COL].notna()]
    scores["language"] = scores["language"].replace({"-9": np.nan, -9: np.nan})

    rows = []
    for questionnaire, q_df in scores.groupby("questionnaire"):
        result = analyse_questionnaire_language(q_df)
        if result is None:
            print(f"  {questionnaire:15s}: skipped (< 2 languages with n≥{LANG_MIN_N_LANG})")
            continue
        result["Questionnaire"] = questionnaire
        rows.append(result)
        print(f"  {questionnaire:15s}: F={result['F']:.2f}  p={result['p']}  "
              f"η²_lang={result['η² (language)']:.3f}  "
              f"η²_site={result['η² (site)']:.3f}  "
              f"ratio={result['Site:language ratio']:.1f}")

    if not rows:
        print("No results.")
        return

    col_order = ["Questionnaire", "Languages", "N languages", "N sites", "N", "F", "p",
                 "η² (language)", "% var language", "% var site within language",
                 "η² (site)", "Site:language ratio"]
    result_df = pd.DataFrame(rows)[col_order].sort_values("Questionnaire").reset_index(drop=True)

    out_path = out_dir / "Table2_language_effects.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="Sheet1", index=False)
        ws = writer.sheets["Sheet1"]
        for col_cells in ws.columns:
            max_len = max(
                (len(str(c.value)) for c in col_cells if c.value is not None),
                default=8,
            )
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 50)

    print(f"\nSaved → {out_path}")
    print(f"\n{result_df.to_string(index=False)}")


# =============================================================================
# 4. Nomological Network Validation
# =============================================================================
#
# Tests whether dimension scores from different questionnaires show consistent
# associations with external validators, providing evidence that items
# assigned to the same dimension across questionnaires measure the same
# construct.
#
# Approach:
#   For each (construct × dimension × questionnaire):
#     OLS: proportion_score ~ C(group) + age + C(sex) + C(site_group)
#     → extract group coefficient (PD effect), SE, 95% CI, p-value
#
#   If BDI dim1 and GDS dim1 both show a similar PD-vs-HC effect after
#   controlling for age, sex, and site, that is independent evidence of
#   cross-questionnaire construct validity for the dimension mapping.
#
#   Heterogeneity test (Cochran's Q) across questionnaires per dimension:
#   significant Q = questionnaires differ in how they capture the group effect
#   → flags potential mapping problems.
#
# Outputs (in output/score-transformation/nomological-validation/):
#   nomological_coefficients.csv   — per questionnaire × dimension coefficients
#   nomological_forest_{construct}.png  — forest plots (group effect per dimension)
#   nomological_heatmap.png        — group coefficient heatmap

NOMO_CONSTRUCTS = ["depression", "anxiety", "sleep"]
NOMO_MIN_N = 30          # min participants per questionnaire × dimension
NOMO_MIN_N_GROUP = 10    # min participants per group level
NOMO_SITE_TOP_N = 50     # sites with ≥ 50 participants get own dummy
NOMO_ALPHA = 0.05
NOMO_SCORE_COL = "av_cdf_score"


def run_nomological(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (construct × dimension × questionnaire):
      OLS: proportion_score ~ C(group) + age + C(sex) + C(site_group)
    Returns long-format dataframe with group coefficient + stats per combo.
    """
    df = df.copy()
    df["site_group"] = assign_site_group(df["site"], site_top_n=NOMO_SITE_TOP_N)
    df["group"] = pd.to_numeric(df["group"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex"] = pd.to_numeric(df["sex"], errors="coerce")

    rows = []

    for (construct, dimension, questionnaire), grp in df.groupby(
        ["construct", "dimension", "questionnaire"]
    ):
        if construct not in NOMO_CONSTRUCTS:
            continue

        dim_label = grp["dimension_label"].iloc[0] if "dimension_label" in grp.columns else ""

        sub = grp[[NOMO_SCORE_COL, "group", "age", "sex", "site_group"]].dropna(
            subset=[NOMO_SCORE_COL, "group", "age", "sex"]
        )
        if len(sub) < NOMO_MIN_N:
            continue

        # Require both group levels present with enough data
        gc = sub["group"].value_counts()
        if not ({0.0, 1.0} <= set(gc.index)):
            continue
        if gc.min() < NOMO_MIN_N_GROUP:
            continue

        # Determine formula: add site_group only if it has >1 level
        if sub["site_group"].nunique() > 1:
            formula = (f"{NOMO_SCORE_COL} ~ C(group) + age + C(sex) + C(site_group)")
        else:
            formula = f"{NOMO_SCORE_COL} ~ C(group) + age + C(sex)"

        try:
            model = ols(formula, data=sub).fit()
        except Exception as e:
            print(f"    WARN: {questionnaire} {construct} dim{dimension}: {e}")
            continue

        # Group coefficient: group=1 (PD) vs group=0 (HC, reference)
        coef_name = "C(group)[T.1.0]"
        if coef_name not in model.params:
            continue

        coef = model.params[coef_name]
        se = model.bse[coef_name]
        ci_lo = model.conf_int().loc[coef_name, 0]
        ci_hi = model.conf_int().loc[coef_name, 1]
        pval = model.pvalues[coef_name]
        t_val = model.tvalues[coef_name]

        rows.append({
            "construct":       construct,
            "dimension":       dimension,
            "dimension_label": dim_label,
            "questionnaire":   questionnaire,
            "n":               len(sub),
            "n_hc":            int((sub["group"] == 0).sum()),
            "n_pd":            int((sub["group"] == 1).sum()),
            "group_coef":      round(coef, 4),
            "se":              round(se, 4),
            "ci_lo_95":        round(ci_lo, 4),
            "ci_hi_95":        round(ci_hi, 4),
            "t":               round(t_val, 3),
            "p_value":         round(pval, 4),
            "significant":     pval < NOMO_ALPHA,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Cochran's Q heterogeneity test per (construct × dimension)
    q_rows = []
    for (construct, dimension), grp in result.groupby(["construct", "dimension"]):
        if len(grp) < 2:
            q_rows.append({"construct": construct, "dimension": dimension,
                           "cochran_Q": np.nan, "Q_df": np.nan, "Q_p": np.nan,
                           "I2": np.nan, "heterogeneous": False})
            continue
        w = 1 / grp["se"].values ** 2           # inverse-variance weights
        b = grp["group_coef"].values
        bw = np.average(b, weights=w)
        Q = np.sum(w * (b - bw) ** 2)
        dof = len(grp) - 1
        p_Q = 1 - stats.chi2.cdf(Q, dof)
        I2 = max(0, (Q - dof) / Q) if Q > 0 else 0
        q_rows.append({
            "construct": construct, "dimension": dimension,
            "cochran_Q": round(Q, 3), "Q_df": dof,
            "Q_p": round(p_Q, 4), "I2": round(I2, 3),
            "heterogeneous": p_Q < NOMO_ALPHA,
        })

    q_df = pd.DataFrame(q_rows)
    result = result.merge(q_df, on=["construct", "dimension"], how="left")

    return result.sort_values(["construct", "dimension", "questionnaire"]).reset_index(drop=True)


def plot_nomological_forest(result: pd.DataFrame, output_dir: Path):
    """
    One figure per construct: forest plots of group coefficient (PD vs HC)
    per questionnaire per dimension.
    """
    for construct in NOMO_CONSTRUCTS:
        c_df = result[result["construct"] == construct]
        if c_df.empty:
            continue

        dimensions = sorted(c_df["dimension"].unique())
        n_dims = len(dimensions)

        fig, axes = plt.subplots(1, n_dims, figsize=(4.5 * n_dims, 5), squeeze=False)

        for ax, dim in zip(axes[0], dimensions):
            d_df = c_df[c_df["dimension"] == dim].sort_values("questionnaire")
            dim_label = d_df["dimension_label"].iloc[0]

            # Heterogeneity annotation
            Q_p = d_df["Q_p"].iloc[0] if "Q_p" in d_df.columns else np.nan
            I2 = d_df["I2"].iloc[0] if "I2" in d_df.columns else np.nan
            het_str = ""
            if not np.isnan(Q_p):
                het_str = f"Q p={Q_p:.3f}, I²={I2:.2f}"

            y = np.arange(len(d_df))
            sig = d_df["significant"].values

            for i, (_, row) in enumerate(d_df.iterrows()):
                color = "#E74C3C" if sig[i] else "#95A5A6"
                ax.errorbar(
                    row["group_coef"], i,
                    xerr=[[row["group_coef"] - row["ci_lo_95"]],
                          [row["ci_hi_95"] - row["group_coef"]]],
                    fmt="o", color=color, capsize=3,
                    markersize=6, linewidth=1.2,
                )
                # n annotation: fixed position in axes coordinates (right margin)
                ax.text(
                    1.02, i, f'n={row["n"]}',
                    transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7, color="#555",
                )

            ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_yticks(y)
            ax.set_yticklabels(d_df["questionnaire"], fontsize=9)
            ax.set_xlabel(f"PD − HC  ({NOMO_SCORE_COL})", fontsize=8)
            ax.set_title(
                f"Dim {dim}: {dim_label}\n{het_str}",
                fontsize=8, fontweight="bold"
            )

        sig_patch = mpatches.Patch(color="#E74C3C", label=f"p < {NOMO_ALPHA}")
        ns_patch = mpatches.Patch(color="#95A5A6", label=f"p ≥ {NOMO_ALPHA}")
        fig.legend(handles=[sig_patch, ns_patch], loc="upper center",
                   ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(
            f"{construct.capitalize()} — Group effect (PD vs HC) on dimension score\n"
            f"covariate-adjusted (age, sex, site)   95% CI",
            fontsize=9, y=1.06,
        )
        plt.tight_layout()
        fname = output_dir / f"nomological_forest_{construct}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved nomological_forest_{construct}.png")


def plot_nomological_heatmap(result: pd.DataFrame, output_dir: Path):
    """
    Heatmap: group coefficient per questionnaire × (construct_dim).
    """
    pivot = result.pivot_table(
        index="questionnaire",
        columns=result.apply(
            lambda r: f"{r['construct']} d{r['dimension']}\n{r['dimension_label'][:18]}",
            axis=1
        ),
        values="group_coef",
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, 1.3 * pivot.shape[1]), max(4, 0.6 * pivot.shape[0])))
    vmax = result["group_coef"].abs().max()
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Group coefficient (PD − HC)")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(val) > vmax * 0.6 else "black")

    # Mark heterogeneous dimensions
    het = result.groupby("dimension")["heterogeneous"].first()
    for j, col in enumerate(pivot.columns):
        # extract dimension number from column name
        try:
            dim_num = int(col.split("d")[1].split("\n")[0])
            if het.get(dim_num, False):
                ax.add_patch(
                    plt.Rectangle((j - 0.5, -0.5), 1, pivot.shape[0],
                                  fill=False, edgecolor="gold", linewidth=2)
                )
        except Exception:
            pass

    ax.set_title(
        "Nomological validation: PD vs HC group effect per questionnaire × dimension\n"
        "(gold border = significant heterogeneity across questionnaires, Q p < .05)",
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "nomological_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved nomological_heatmap.png")


def print_nomological_summary(result: pd.DataFrame):
    print("\n" + "=" * 70)
    print("NOMOLOGICAL VALIDATION SUMMARY")
    print("=" * 70)

    for construct in NOMO_CONSTRUCTS:
        c_df = result[result["construct"] == construct]
        if c_df.empty:
            continue
        print(f"\n{construct.upper()}")
        for dim, d_df in c_df.groupby("dimension"):
            label = d_df["dimension_label"].iloc[0]
            het = d_df["heterogeneous"].iloc[0] if "heterogeneous" in d_df.columns else False
            het_str = "  *** HETEROGENEOUS ***" if het else ""
            print(f"  Dim {dim} ({label}){het_str}")
            for _, row in d_df.iterrows():
                sig_str = "*" if row["significant"] else " "
                print(f"    {sig_str} {row['questionnaire']:15s}  "
                      f"β={row['group_coef']:+.3f}  "
                      f"[{row['ci_lo_95']:+.3f}, {row['ci_hi_95']:+.3f}]  "
                      f"p={row['p_value']:.3f}  n={row['n']}")


def run_nomological_validation(binary: bool = False):
    score_dir = SCORE_DIR
    dimension_scores = DIMENSION_SCORES
    output_dir = BASE_OUTPUT_DIR / "nomological-validation"

    if binary:
        binary_dir = PROJECT_DIR / "output" / "score-transformation-binary"
        score_dir = binary_dir
        dimension_scores = binary_dir / "dimension_scores.csv"
        output_dir = BASE_OUTPUT_DIR / "nomological_validation-binary"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NOMOLOGICAL NETWORK VALIDATION")
    if binary:
        print("MODE: BINARY")
    print("=" * 70)

    if not dimension_scores.exists():
        raise FileNotFoundError(
            f"dimension_scores.csv not found at {dimension_scores}.\n"
            "Run run_pipeline.py first."
        )

    print("\nLoading dimension scores ...")
    df = pd.read_csv(dimension_scores)
    print(f"  {len(df)} rows, {df['participant_id'].nunique()} participants")

    # Report data availability
    n_group_avail = df["group"].notna().sum()
    print(f"  group available: {n_group_avail}/{len(df)} rows "
          f"({n_group_avail/len(df)*100:.0f}%)")
    print(f"  age available:   {df['age'].notna().sum()}/{len(df)} rows")
    print(f"  sex available:   {df['sex'].notna().sum()}/{len(df)} rows")

    print("\nRunning nomological regression per questionnaire × dimension ...")
    result = run_nomological(df)

    if result.empty:
        print("  No results computed (insufficient data with group/age/sex).")
        return

    result.to_csv(output_dir / "nomological_coefficients.csv", index=False)
    print(f"  Saved nomological_coefficients.csv  ({len(result)} rows)")

    print_nomological_summary(result)

    print("\nGenerating figures ...")
    plot_nomological_forest(result, output_dir)
    plot_nomological_heatmap(result, output_dir)

    print(f"\n{'='*70}")
    print(f"Done. Outputs saved to {output_dir}")
    print(f"{'='*70}")


# =============================================================================
# 5. Rank Preservation
# =============================================================================
#
# Tests whether the CDF-transformed dimension scores preserve participant
# rankings relative to their raw full-questionnaire score.
#
# For each questionnaire × construct:
#   - "Full questionnaire rank"  : rank on mean proportion_score across ALL
#     dimensions (raw, pre-transformation aggregate — the natural total score)
#   - "Dimension rank (CDF)"     : rank on av_cdf_score per dimension
#
# If the transformation is well-behaved, high/low scorers on the full
# questionnaire should also rank high/low on each individual dimension after
# transformation.
#
# Outputs (in output/score-transformation/rank-preservation/):
#   rank_preservation.csv                        — Spearman ρ + top/bottom concordance
#   rank_preservation_heatmap.png                — ρ heatmap (questionnaire × dimension)
#   rank_preservation_scatter_{questionnaire}.png — per-questionnaire scatter plots

RANK_SCORE_COL = "av_cdf_score"
RANK_TOP_FRAC = 0.25    # top/bottom quartile for concordance test
RANK_MIN_N = 20


def concordance(a_ids: set, b_ids: set, k: int) -> float:
    return len(a_ids & b_ids) / k if k > 0 else np.nan


def analyse_questionnaire_rank(q_scores: pd.DataFrame, questionnaire: str,
                               construct: str) -> list[dict]:
    """Return one row per dimension for a given questionnaire × construct."""
    total = (q_scores.groupby("participant_id")["proportion_score"]
             .mean().rename("total_score").reset_index())

    rows = []
    for (dim, dim_label), d_df in q_scores.groupby(["dimension", "dimension_label"]):
        merged = total.merge(
            d_df[["participant_id", RANK_SCORE_COL]].rename(columns={RANK_SCORE_COL: "dim_score"}),
            on="participant_id"
        ).dropna(subset=["total_score", "dim_score"])

        if len(merged) < RANK_MIN_N:
            continue

        merged = merged.copy()
        merged["rank_total"] = merged["total_score"].rank(method="average", ascending=True)
        merged["rank_dim"] = merged["dim_score"].rank(method="average", ascending=True)
        n = len(merged)
        k = max(1, round(n * RANK_TOP_FRAC))

        rho, pval = stats.spearmanr(merged["total_score"], merged["dim_score"])

        bottom_total = set(merged.nsmallest(k, "total_score")["participant_id"])
        bottom_dim = set(merged.nsmallest(k, "dim_score")["participant_id"])
        top_total = set(merged.nlargest(k, "total_score")["participant_id"])
        top_dim = set(merged.nlargest(k, "dim_score")["participant_id"])

        rows.append({
            "questionnaire":        questionnaire,
            "construct":            construct,
            "dimension":            dim,
            "dimension_label":      dim_label,
            "n":                    n,
            "spearman_rho":         round(float(rho), 3),
            "p_value":              round(float(pval), 4),
            "bottom25_concordance": round(concordance(bottom_total, bottom_dim, k), 3),
            "top25_concordance":    round(concordance(top_total, top_dim, k), 3),
            "_merged":              merged,
        })
    return rows


def plot_rank_scatter(results: list[dict], questionnaire: str, construct: str, output_dir: Path):
    dims = [r for r in results
            if r["questionnaire"] == questionnaire and r["construct"] == construct]
    if not dims:
        return
    n_dims = len(dims)
    fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 4.2), squeeze=False)
    for ax, res in zip(axes[0], dims):
        m = res["_merged"]
        ax.scatter(m["rank_total"], m["rank_dim"], alpha=0.35, s=12, color="#1565C0")
        lo = min(m["rank_total"].min(), m["rank_dim"].min())
        hi = max(m["rank_total"].max(), m["rank_dim"].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, alpha=0.4, label="identity")
        ax.set_xlabel("Rank — full questionnaire (raw)", fontsize=8)
        ax.set_ylabel("Rank — dimension (av_cdf_score)", fontsize=8)
        label = res["dimension_label"][:28]
        ax.set_title(
            f"Dim {res['dimension']}: {label}\n"
            f"ρ = {res['spearman_rho']:.3f}  (p = {res['p_value']:.3f})\n"
            f"Bottom-25% overlap: {res['bottom25_concordance']:.0%}  "
            f"Top-25% overlap: {res['top25_concordance']:.0%}",
            fontsize=8,
        )
    fig.suptitle(
        f"{questionnaire} ({construct}) — rank preservation after CDF transform\n"
        f"x = participant rank by raw total questionnaire score  |  "
        f"y = participant rank by CDF-transformed dimension score",
        fontsize=9, y=1.04,
    )
    plt.tight_layout()
    fname = output_dir / f"rank_preservation_scatter_{questionnaire}_{construct}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname.name}")


def plot_rank_heatmap(result_df: pd.DataFrame, output_dir: Path):
    df = result_df.copy()
    df["q_label"] = df["questionnaire"] + " × " + df["construct"]
    pivot_rho = df.pivot_table("spearman_rho", index="q_label", columns="dimension_label", aggfunc="mean")
    pivot_bot = df.pivot_table("bottom25_concordance", index="q_label", columns="dimension_label", aggfunc="mean")
    pivot_top = df.pivot_table("top25_concordance", index="q_label", columns="dimension_label", aggfunc="mean")

    col_order = pivot_rho.median(skipna=True).sort_values(ascending=False).index

    n_rows = len(pivot_rho)
    fig, axes = plt.subplots(1, 3, figsize=(7 * 3, max(4, 0.55 * n_rows)))

    panels = [
        (pivot_rho, "Spearman ρ\n(full questionnaire vs dimension CDF rank)", "RdYlGn", 0, 1),
        (pivot_bot, "Bottom-25% concordance\n(fraction in shared lowest quartile)", "RdYlGn", 0, 1),
        (pivot_top, "Top-25% concordance\n(fraction in shared highest quartile)", "RdYlGn", 0, 1),
    ]
    for ax, (piv, title, cmap, vmin, vmax) in zip(axes, panels):
        sns.heatmap(piv.reindex(columns=col_order), annot=True, fmt=".2f", cmap=cmap,
                    vmin=vmin, vmax=vmax, ax=ax, linewidths=0.5,
                    annot_kws={"size": 7})
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=7, rotation=35)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(
        "Rank preservation — ENIGMA-PD cohort\n"
        "Does the CDF-transformed dimension score preserve the raw questionnaire ranking?",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    fname = output_dir / "rank_preservation_heatmap.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname.name}")


def run_rank_preservation():
    output_dir = BASE_OUTPUT_DIR / "rank-preservation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RANK PRESERVATION — ENIGMA-PD COHORT")
    print("=" * 70)

    scores = pd.read_csv(SCORE_DIR / "dimension_scores.csv")
    scores = scores[scores[RANK_SCORE_COL].notna() & scores["proportion_score"].notna()]

    all_results: list[dict] = []

    for (questionnaire, construct), q_df in scores.groupby(["questionnaire", "construct"]):
        rows = analyse_questionnaire_rank(q_df, questionnaire, construct)
        all_results.extend(rows)
        for r in rows:
            print(f"  {questionnaire:15s} × {construct:10s}  "
                  f"dim {r['dimension']} ({r['dimension_label'][:28]:28s}): "
                  f"ρ={r['spearman_rho']:+.3f}  "
                  f"bottom-25%={r['bottom25_concordance']:.0%}  "
                  f"top-25%={r['top25_concordance']:.0%}  "
                  f"(n={r['n']})")

    if not all_results:
        print("No results — check that dimension_scores.csv exists.")
        return

    csv_cols = [k for k in all_results[0] if not k.startswith("_")]
    result_df = pd.DataFrame([{k: r[k] for k in csv_cols} for r in all_results])
    result_df.to_csv(output_dir / "rank_preservation.csv", index=False)
    print(f"\nSaved rank_preservation.csv ({len(result_df)} rows)")

    print("\nGenerating figures...")
    plot_rank_heatmap(result_df, output_dir)
    for (q, c) in result_df[["questionnaire", "construct"]].drop_duplicates().values:
        plot_rank_scatter(all_results, q, c, output_dir)

    print(f"\n{'='*70}")
    print(f"Done. Outputs saved to {output_dir}")
    print(f"{'='*70}")


# =============================================================================
# Main
# =============================================================================

def main():
    global SCORE_DIR, DIMENSION_SCORES, BASE_OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--binary", action="store_true",
                        help="Use binary-format pipeline outputs "
                             "(cross-questionnaire prediction & nomological validation only)")
    parser.add_argument("--reduced-covariates", action="store_true",
                        help="Cross-questionnaire prediction: use only age + sex + group "
                             "(+ language & site) as covariates")
    parser.add_argument("--score-dir", type=Path, default=None,
                        help="Directory containing dimension_scores.csv (default: "
                             "output/score-transformation/). Point this at a scratch "
                             "directory when validating against run_pipeline.py's "
                             "--output-dir, so real committed results are never overwritten.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Base directory for all five analyses' output subfolders "
                             "(default: output/score-transformation/). Each analysis still "
                             "writes its own named subfolder underneath it.")
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_DIR / "output" / "results_figures_tables",
                        help="Language effects: directory to write Table2_language_effects.xlsx to")
    args = parser.parse_args()

    if args.score_dir is not None:
        SCORE_DIR = args.score_dir
        DIMENSION_SCORES = args.score_dir / "dimension_scores.csv"
    if args.output_dir is not None:
        BASE_OUTPUT_DIR = args.output_dir

    print("#" * 70)
    print("# EXPLORATORY VALIDATION ANALYSES")
    print("#" * 70)

    print("\n[1/5] Convergent / discriminant validity")
    run_convergent_discriminant_validity()

    print("\n[2/5] Cross-questionnaire within-person prediction")
    run_cross_questionnaire_prediction(binary=args.binary, reduced_covariates=args.reduced_covariates)

    print("\n[3/5] Language effects")
    run_language_effects(out_dir=args.out_dir)

    print("\n[4/5] Nomological network validation")
    run_nomological_validation(binary=args.binary)

    print("\n[5/5] Rank preservation")
    run_rank_preservation()

    print("\n" + "#" * 70)
    print("# ALL EXPLORATORY VALIDATION ANALYSES COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()
