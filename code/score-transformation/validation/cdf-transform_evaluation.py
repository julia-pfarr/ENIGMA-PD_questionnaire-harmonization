"""
CDF Transform Evaluation
========================
Compares av_cdf_score (averaged-CDF transform) against all other score types to:

  1. Quantify how much individual scores changed (correlations, MAD)
  2. Show within-questionnaire rank preservation (scatter plots)
  3. Show distributional alignment across questionnaires before/after (KDE)
  4. Cross-questionnaire KS test: distributions more similar after transform?
  5. Verify PD vs HC clinical signal is preserved (Cohen's d)
  6. Descriptive stats per questionnaire (mean, SD, skewness) before/after

All outputs saved to output/score-transformation/cdf-transform_evaluation/
(override with --score-dir / --output-dir)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from pathlib import Path
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_SCORE_DIR = PROJECT_DIR / "output" / "score-transformation"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output" / "score-transformation" / "cdf-transform_evaluation"

SCORES_CSV  = DEFAULT_SCORE_DIR / "dimension_scores.csv"
OUTPUT_DIR  = DEFAULT_OUTPUT_DIR

TARGET = "av_cdf_score"
SCORE_LABELS = {
    "raw_score":                            "Raw dimension score",
    "proportion_score":                     "Proportion",
    "q_cdf_score":                          "Per-Q C-score",
    "av_cdf_score":                         "Averaged CDF",
}
COMPARE_COLS = [c for c in SCORE_LABELS if c != TARGET]
MIN_N = 30


# =============================================================================
# Helpers
# =============================================================================

def cohen_d(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    sd = np.sqrt(((n1 - 1) * a.std() ** 2 + (n2 - 1) * b.std() ** 2) / (n1 + n2 - 2))
    return np.nan if sd == 0 else (a.mean() - b.mean()) / sd


def ks_stat(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return stats.ks_2samp(a, b).statistic


def dim_label_short(s, maxlen=28):
    return s if len(s) <= maxlen else s[:maxlen - 1] + "…"


# =============================================================================
# 1. Correlation table
# =============================================================================

def run_correlations(df, available):
    print("\n=== 1. CORRELATIONS WITH CDF TRANSFORM ===")
    rows = []
    for (construct, dim), g in df.groupby(["construct", "dimension"]):
        dim_label = g["dimension_label"].iloc[0]
        for sc in available:
            both = g[[sc, TARGET]].dropna()
            if len(both) < MIN_N:
                continue
            rp  = both[sc].corr(both[TARGET])
            rs  = both[sc].corr(both[TARGET], method="spearman")
            mad = (both[sc] - both[TARGET]).abs().mean()
            rows.append({
                "construct": construct, "dimension": dim,
                "dimension_label": dim_label,
                "score_type": SCORE_LABELS[sc],
                "n": len(both),
                "pearson_r": round(rp, 4),
                "spearman_r": round(rs, 4),
                "MAD": round(mad, 4),
            })
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(OUTPUT_DIR / "score_correlations.csv", index=False)
    print(corr_df.to_string(index=False))

    # Fig 1a: one heatmap per construct (dimension × score type)
    for construct in sorted(corr_df["construct"].unique()):
        c_df = corr_df[corr_df["construct"] == construct]
        piv = c_df.pivot_table(
            index="dimension_label", columns="score_type", values="pearson_r",
        )
        piv.columns = [c.replace(" ", "\n") for c in piv.columns]
        fig, ax = plt.subplots(figsize=(max(6, len(piv.columns) * 2.0),
                                        max(3, len(piv) * 0.65 + 1.2)))
        sns.heatmap(piv, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1,
                    ax=ax, linewidths=0.5, cbar_kws={"label": "Pearson r with CDF transform"})
        ax.set_title(f"Pearson r with CDF-transform score — {construct.upper()}\n"
                     f"(pooled across questionnaires)", fontsize=10)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"fig1a_correlation_heatmap_{construct}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved fig1a_correlation_heatmap_{construct}.png")

    # Per-questionnaire data
    rows_q = []
    for (construct, dim, q), g in df.groupby(["construct", "dimension", "questionnaire"]):
        dim_label = g["dimension_label"].iloc[0]
        for sc in available:
            both = g[[sc, TARGET]].dropna()
            if len(both) < MIN_N:
                continue
            rows_q.append({
                "questionnaire": q, "construct": construct,
                "dimension": dim, "dimension_label": dim_label,
                "score_type": SCORE_LABELS[sc],
                "pearson_r": round(both[sc].corr(both[TARGET]), 4),
            })
    corr_q_df = pd.DataFrame(rows_q)
    corr_q_df.to_csv(OUTPUT_DIR / "score_correlations_per_questionnaire.csv", index=False)

    # Fig 1b: one heatmap per construct, questionnaire as section heading
    for construct in sorted(corr_q_df["construct"].unique()):
        c_df = corr_q_df[corr_q_df["construct"] == construct]
        piv = c_df.pivot_table(
            index=["questionnaire", "dimension_label"],
            columns="score_type", values="pearson_r",
        ).sort_index()
        n_score_cols = len(piv.columns)

        # NaN spacer rows carry questionnaire name as text annotation (not as tick label)
        header_qs, new_data, ytick_labels = [], [], []
        prev_q = None
        for (q, d), row in piv.iterrows():
            if q != prev_q:
                header_qs.append((len(new_data), q))
                new_data.append([np.nan] * n_score_cols)
                ytick_labels.append("")
                prev_q = q
            new_data.append(row.values.tolist())
            short_d = d if len(d) <= 32 else d[:31] + "…"
            ytick_labels.append(short_d)

        n_rows = len(new_data)
        plot_df = pd.DataFrame(new_data, columns=piv.columns)

        annot = [
            [""] * n_score_cols if i in {hp for hp, _ in header_qs}
            else [f"{v:.3f}" if not np.isnan(v) else "" for v in row]
            for i, row in enumerate(new_data)
        ]

        fig, ax = plt.subplots(figsize=(max(7, n_score_cols * 2.2),
                                        max(4, n_rows * 0.52 + 1.2)))
        sns.heatmap(plot_df, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1,
                    ax=ax, linewidths=0.3,
                    cbar_kws={"label": "Pearson r with CDF transform"},
                    mask=plot_df.isna())

        # Separator lines and bold questionnaire name annotations in spacer rows
        from matplotlib.transforms import blended_transform_factory
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        for hp, q_name in header_qs:
            if hp > 0:
                ax.axhline(hp, color="black", lw=1.2, zorder=3)
            ax.text(0.0, hp + 0.5, q_name, transform=trans,
                    ha="right", va="center", fontweight="bold",
                    fontsize=9.5, clip_on=False)

        data_positions = [i + 0.5 for i in range(n_rows) if i not in {hp for hp, _ in header_qs}]
        data_labels    = [ytick_labels[i] for i in range(n_rows) if i not in {hp for hp, _ in header_qs}]
        ax.set_yticks(data_positions)
        ax.set_yticklabels(data_labels, fontsize=8, rotation=0)

        ax.set_xticklabels([c.replace(" ", "\n") for c in piv.columns], fontsize=8, rotation=0)
        ax.set_title(f"Pearson r with CDF-transform score — {construct.upper()}\n"
                     f"(per questionnaire)", fontsize=10)
        ax.set_ylabel("")
        ax.set_xlabel("")
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"fig1b_correlation_heatmap_per_questionnaire_{construct}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved fig1b_correlation_heatmap_per_questionnaire_{construct}.png")
    return corr_df


# =============================================================================
# 2. Distribution alignment — KDE per questionnaire, before/after comparison
# =============================================================================

def plot_distributions(df):
    print("\n=== 2. DISTRIBUTION ALIGNMENT (KDE) ===")
    from scipy.stats import gaussian_kde
    from matplotlib.lines import Line2D

    score_cols  = ["proportion_score", TARGET]
    col_titles  = ["Proportion score\n(before harmonisation)",
                   "Averaged CDF\n(per-questionnaire KDE on shared scale)",
                   "Averaged CDF\n(participants by questionnaire)"]

    for construct in df["construct"].unique():
        c_df  = df[df["construct"] == construct]
        dims  = sorted(c_df["dimension"].unique())
        qs    = sorted(c_df["questionnaire"].unique())
        pal   = sns.color_palette("tab10", len(qs))
        q_col = {q: pal[i] for i, q in enumerate(qs)}

        fig, axes = plt.subplots(len(dims), 3,
                                  figsize=(13, 3 * len(dims) + 0.8),
                                  squeeze=False)

        plotted_all = []
        rng = np.random.default_rng(42)

        for ri, dim in enumerate(dims):
            dim_label = c_df[c_df["dimension"] == dim]["dimension_label"].iloc[0]

            # Columns 0–1: per-questionnaire KDEs
            for ci, (sc, title) in enumerate(zip(score_cols, col_titles)):
                ax = axes[ri][ci]
                for q in qs:
                    vals = c_df[(c_df["dimension"] == dim) &
                                (c_df["questionnaire"] == q)][sc].dropna()
                    if len(vals) < MIN_N:
                        continue
                    vals.plot.kde(ax=ax, color=q_col[q], label=q, linewidth=1.8)
                    if q not in plotted_all:
                        plotted_all.append(q)
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(bottom=0)
                ax.set_xlabel("Score" if ri == len(dims) - 1 else "")
                ax.set_ylabel(dim_label_short(dim_label) if ci == 0 else "", fontsize=8)
                if ri == 0:
                    ax.set_title(title, fontsize=9)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            # Column 2: combined KDE + per-questionnaire scatter on the curve
            ax4 = axes[ri][2]
            all_vals = c_df[c_df["dimension"] == dim][TARGET].dropna()
            if len(all_vals) >= MIN_N:
                kde_func = gaussian_kde(all_vals)
                x_grid   = np.linspace(-0.05, 1.05, 300)
                kde_line = kde_func(x_grid)
                ax4.plot(x_grid, kde_line, color="black", linewidth=1.5, zorder=2)
                ax4.fill_between(x_grid, kde_line, alpha=0.08, color="black", zorder=1)

                y_scale = kde_line.max()
                for q in qs:
                    q_vals = c_df[(c_df["dimension"] == dim) &
                                  (c_df["questionnaire"] == q)][TARGET].dropna()
                    if len(q_vals) < 1:
                        continue
                    y_on_kde = kde_func(q_vals.values)
                    jitter   = rng.uniform(-y_scale * 0.10, y_scale * 0.10,
                                           size=len(q_vals))
                    ax4.scatter(q_vals.values, y_on_kde + jitter,
                                color=q_col[q], s=2, alpha=0.4, zorder=3)

            ax4.set_xlim(-0.05, 1.05)
            ax4.set_ylim(bottom=0)
            ax4.set_xlabel("Score" if ri == len(dims) - 1 else "")
            if ri == 0:
                ax4.set_title(col_titles[2], fontsize=9)
            ax4.spines["top"].set_visible(False)
            ax4.spines["right"].set_visible(False)

        if plotted_all:
            handles = [Line2D([0], [0], color=q_col[q], linewidth=1.8, label=q)
                       for q in plotted_all]
            fig.legend(handles=handles, fontsize=7, loc="center left",
                       bbox_to_anchor=(1.01, 0.5), title="Questionnaire")

        fig.suptitle(f"Distribution alignment across questionnaires — {construct.upper()}",
                     fontsize=11, y=1.01)
        plt.tight_layout()
        plt.subplots_adjust(right=0.87)
        fig.savefig(OUTPUT_DIR / f"fig2_distributions_{construct}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved fig2_distributions_{construct}.png")


# =============================================================================
# 3. KS test: cross-questionnaire distributional similarity before vs after
# =============================================================================

def run_ks_test(df):
    print("\n=== 3. KS TEST: DISTRIBUTIONAL SIMILARITY ===")
    ks_cols = ["proportion_score", TARGET]
    ks_labels = {
        "proportion_score":  "Proportion",
        TARGET:              "CDF transform",
    }
    rows = []
    for (construct, dim), g in df.groupby(["construct", "dimension"]):
        dim_label = g["dimension_label"].iloc[0]
        qs = [q for q in g["questionnaire"].unique()
              if g[g["questionnaire"] == q]["proportion_score"].notna().sum() >= MIN_N]
        for q1, q2 in combinations(sorted(qs), 2):
            row = {"construct": construct, "dimension": dim,
                   "dimension_label": dim_label, "q1": q1, "q2": q2}
            for sc in ks_cols:
                v1 = g[g["questionnaire"] == q1][sc].dropna().values
                v2 = g[g["questionnaire"] == q2][sc].dropna().values
                row[f"ks_{ks_labels[sc]}"] = round(ks_stat(v1, v2), 4)
            rows.append(row)

    ks_df = pd.DataFrame(rows)
    ks_df["ks_reduction_pct"] = (
        100 * (ks_df["ks_Proportion"] - ks_df["ks_CDF transform"])
        / ks_df["ks_Proportion"].replace(0, np.nan)
    ).round(1)
    ks_df.to_csv(OUTPUT_DIR / "ks_distributional_similarity.csv", index=False)
    print(ks_df[["construct", "dimension_label", "q1", "q2",
                 "ks_Proportion", "ks_CDF transform",
                 "ks_reduction_pct"]].to_string(index=False))

    # Figure: dumbbell plot showing KS before/after per pair
    ks_df["pair_label"] = (ks_df["q1"] + " vs " + ks_df["q2"]
                           + "\n(" + ks_df["dimension_label"].apply(dim_label_short) + ")")
    ks_sorted = ks_df.sort_values("ks_Proportion", ascending=True).reset_index(drop=True)
    y = np.arange(len(ks_sorted))

    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(ks_sorted) + 1.5)))
    ax.scatter(ks_sorted["ks_Proportion"], y, color="#e74c3c", s=45,
               zorder=4, label="Proportion score")
    ax.scatter(ks_sorted["ks_CDF transform"], y, color="#27ae60", s=45,
               zorder=4, marker="D", label="CDF transform")
    for i, row in ks_sorted.iterrows():
        ax.plot([row["ks_Proportion"], row["ks_CDF transform"]], [i, i],
                color="gray", lw=0.8, zorder=2, alpha=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(ks_sorted["pair_label"], fontsize=7)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("KS statistic  (lower = more similar distributions between questionnaires)", fontsize=9)
    ax.set_title("Cross-questionnaire distributional similarity\nbefore and after CDF transform", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_ks_before_after.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_ks_before_after.png")
    return ks_df


# =============================================================================
# 4. PD vs HC effect preservation
# =============================================================================

def run_effect_preservation(df):
    print("\n=== 4. PD vs HC EFFECT PRESERVATION (Cohen's d) ===")
    df = df.copy()
    df["group"] = pd.to_numeric(df["group"], errors="coerce")

    rows = []
    for (construct, dim, q), g in df.groupby(["construct", "dimension", "questionnaire"]):
        dim_label = g["dimension_label"].iloc[0]
        pd_g = g[g["group"] == 1]
        hc_g = g[g["group"] == 0]
        if len(pd_g) < MIN_N or len(hc_g) < MIN_N:
            continue
        for sc, label in [("proportion_score", "Proportion"), (TARGET, "CDF transform")]:
            pd_vals = pd_g[sc].dropna()
            hc_vals = hc_g[sc].dropna()
            if len(pd_vals) < MIN_N or len(hc_vals) < MIN_N:
                continue
            rows.append({
                "construct": construct, "dimension": dim,
                "dimension_label": dim_label, "questionnaire": q,
                "score_type": label,
                "cohen_d": round(cohen_d(pd_vals, hc_vals), 3),
                "n_pd": len(pd_vals), "n_hc": len(hc_vals),
            })

    eff_df = pd.DataFrame(rows)
    eff_df.to_csv(OUTPUT_DIR / "pd_hc_effect_preservation.csv", index=False)
    print(eff_df.to_string(index=False))

    wide = eff_df.pivot_table(
        index=["construct", "dimension_label", "questionnaire"],
        columns="score_type", values="cohen_d",
    ).reset_index()

    if "Proportion" not in wide.columns or "CDF transform" not in wide.columns:
        print("  Not enough data for scatter plot.")
        return eff_df

    fig, ax = plt.subplots(figsize=(6, 6))
    constructs = wide["construct"].unique()
    pal = sns.color_palette("tab10", len(constructs))
    for ci, construct in enumerate(constructs):
        sub = wide[wide["construct"] == construct]
        ax.scatter(sub["Proportion"].abs(), sub["CDF transform"].abs(),
                   label=construct, s=55, alpha=0.8, color=pal[ci], zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["questionnaire"],
                        (abs(row["Proportion"]), abs(row["CDF transform"])),
                        fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")

    lim = max(wide[["Proportion", "CDF transform"]].abs().max().max() * 1.15, 0.5)
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6, label="y = x (no change)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("|Cohen's d| — Proportion score", fontsize=9)
    ax.set_ylabel("|Cohen's d| — CDF transform", fontsize=9)
    ax.set_title("PD vs HC effect size before vs after CDF transform\n"
                 "Points on the diagonal = clinical signal perfectly preserved", fontsize=9)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_effect_preservation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_effect_preservation.png")
    return eff_df


# =============================================================================
# 5. Within-questionnaire scatter: proportion vs CDF transform
# =============================================================================

def plot_within_q_scatter(df):
    print("\n=== 5. WITHIN-QUESTIONNAIRE SCATTER (rank preservation) ===")
    for construct in df["construct"].unique():
        c_df = df[df["construct"] == construct]
        dims = sorted(c_df["dimension"].unique())
        qs   = sorted(c_df["questionnaire"].unique())

        fig, axes = plt.subplots(len(dims), len(qs),
                                  figsize=(2.8 * len(qs), 2.8 * len(dims)),
                                  squeeze=False)

        for ri, dim in enumerate(dims):
            dim_label = c_df[c_df["dimension"] == dim]["dimension_label"].iloc[0]
            for ci, q in enumerate(qs):
                ax = axes[ri][ci]
                sub = c_df[(c_df["dimension"] == dim) &
                           (c_df["questionnaire"] == q)][
                    ["proportion_score", TARGET]].dropna()

                if len(sub) < MIN_N:
                    ax.axis("off")
                    if ri == 0:
                        ax.set_title(q, fontsize=8)
                    continue

                rs = sub["proportion_score"].corr(sub[TARGET], method="spearman")
                ax.scatter(sub["proportion_score"], sub[TARGET],
                           s=3, alpha=0.25, color="#3498db", rasterized=True)
                ax.plot([0, 1], [0, 1], "r--", lw=0.8, alpha=0.4)
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(-0.02, 1.02)
                ax.set_aspect("equal")
                ax.set_title(q if ri == 0 else "", fontsize=8)
                ax.set_ylabel(dim_label_short(dim_label, 20) if ci == 0 else "", fontsize=7)
                ax.set_xlabel("Proportion" if ri == len(dims) - 1 else "", fontsize=7)
                ax.text(0.04, 0.92, f"ρ={rs:.3f}", transform=ax.transAxes,
                        fontsize=6, color="#2c3e50")
                ax.tick_params(labelsize=5)

        fig.suptitle(f"Within-questionnaire: proportion score vs CDF transform — {construct.upper()}\n"
                     f"(monotone curve = ranks preserved; ρ ≈ 1 in all panels)",
                     fontsize=10, y=1.01)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"fig5_within_q_scatter_{construct}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved fig5_within_q_scatter_{construct}.png")


# =============================================================================
# 6. Descriptive stats per questionnaire: mean, SD, skewness before/after
# =============================================================================

def run_descriptive_stats(df):
    print("\n=== 6. DESCRIPTIVE STATS PER QUESTIONNAIRE ===")
    score_pairs = [
        ("proportion_score", "Proportion"),
        (TARGET,             "CDF transform"),
    ]
    rows = []
    for (construct, dim, q), g in df.groupby(["construct", "dimension", "questionnaire"]):
        dim_label = g["dimension_label"].iloc[0]
        for sc, label in score_pairs:
            vals = g[sc].dropna()
            if len(vals) < MIN_N:
                continue
            rows.append({
                "construct": construct, "dimension": dim,
                "dimension_label": dim_label, "questionnaire": q,
                "score_type": label, "n": len(vals),
                "mean": round(vals.mean(), 4),
                "sd":   round(vals.std(),  4),
                "skewness": round(float(stats.skew(vals)), 4),
                "p10": round(np.percentile(vals, 10), 4),
                "p90": round(np.percentile(vals, 90), 4),
            })
    desc_df = pd.DataFrame(rows)
    desc_df.to_csv(OUTPUT_DIR / "descriptive_stats_per_questionnaire.csv", index=False)
    print(desc_df.to_string(index=False))

    # Figure: mean and SD per questionnaire, before vs after, per construct × dim
    constructs = df["construct"].unique()
    for construct in constructs:
        c_desc = desc_df[desc_df["construct"] == construct]
        dims = sorted(c_desc["dimension"].unique())
        fig, axes = plt.subplots(len(dims), 2,
                                  figsize=(9, 2.5 * len(dims) + 0.8),
                                  squeeze=False)
        for ri, dim in enumerate(dims):
            dim_label = c_desc[c_desc["dimension"] == dim]["dimension_label"].iloc[0]
            sub = c_desc[c_desc["dimension"] == dim]
            qs  = sorted(sub["questionnaire"].unique())
            x   = np.arange(len(qs))
            w   = 0.35

            for ci, stat in enumerate(["mean", "sd"]):
                ax = axes[ri][ci]
                before = [sub[(sub["questionnaire"] == q) &
                              (sub["score_type"] == "Proportion")][stat].values
                          for q in qs]
                after  = [sub[(sub["questionnaire"] == q) &
                              (sub["score_type"] == "CDF transform")][stat].values
                          for q in qs]
                before = [v[0] if len(v) > 0 else np.nan for v in before]
                after  = [v[0] if len(v) > 0 else np.nan for v in after]

                ax.bar(x - w / 2, before, width=w, label="Proportion", color="#e74c3c", alpha=0.7)
                ax.bar(x + w / 2, after,  width=w, label="CDF transform", color="#27ae60", alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels(qs, rotation=30, ha="right", fontsize=8)
                ax.set_ylabel(stat.upper(), fontsize=8)
                ax.set_title(f"{dim_label_short(dim_label)} — {stat}" if ri == 0
                             else dim_label_short(dim_label), fontsize=8)
                if ri == 0 and ci == 0:
                    ax.legend(fontsize=7)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

        fig.suptitle(f"Mean and SD per questionnaire before/after CDF transform — {construct.upper()}",
                     fontsize=10, y=1.01)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / f"fig6_descriptive_stats_{construct}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved fig6_descriptive_stats_{construct}.png")
    return desc_df


# =============================================================================
# Main
# =============================================================================

def main():
    global SCORES_CSV, OUTPUT_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", type=Path, default=None,
                        help="Directory containing dimension_scores.csv (default: "
                             f"{DEFAULT_SCORE_DIR}). Point this at a scratch directory "
                             "when validating against run_pipeline.py's --output-dir, "
                             "so real committed results are never overwritten.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"Where to write results (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    if args.score_dir is not None:
        SCORES_CSV = args.score_dir / "dimension_scores.csv"
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CDF TRANSFORM EVALUATION")
    print("=" * 70)
    print(f"\nScores CSV: {SCORES_CSV}")
    print(f"Output dir: {OUTPUT_DIR}")

    df = pd.read_csv(SCORES_CSV)
    available = [c for c in COMPARE_COLS if c in df.columns]
    print(f"\nScore columns available for comparison: {available}")
    print(f"Target (CDF transform): {TARGET}")
    print(f"Rows in scores file: {len(df)}")

    run_correlations(df, available)
    plot_distributions(df)
    run_ks_test(df)
    run_effect_preservation(df)
    plot_within_q_scatter(df)
    run_descriptive_stats(df)

    print(f"\n{'=' * 70}")
    print(f"All outputs saved to {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
