"""
Evaluate round-2 fine-tuned models on testing_round-2.csv.

Models:    output/fine-tuning/models-final/
Test file: input/model-testing/testing_round-2.csv
Output:    output/fine-tuning/model-comparison/testing_models-final_round-2/

Constructs:
  depression      → contrastive_model.pkl
  anxiety         → contrastive_model.pkl
  psychosis       → contrastive_model.pkl
  sleep           → contrastive_model.pkl
  impulse_control → prototype_model.pkl
  apathy          → skipped (no test items in testing_round-2.csv)
"""

import sys
import warnings
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

SCRIPT_DIR   = Path(__file__).resolve().parent
FT_CODE_DIR  = SCRIPT_DIR.parent / 'semantic-learning_harmonize'
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_DIMENSION_CONFIG = PROJECT_ROOT / 'input' / 'shared' / 'dimension_descriptions.yaml'

sys.path.insert(0, str(FT_CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import model_comparison_validation as mcv

MODELS_DIR = PROJECT_ROOT / 'output' / 'fine-tuning' / 'models-final'
TEST_FILE  = PROJECT_ROOT / 'input'  / 'model-testing' / 'testing_round-2.csv'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'fine-tuning' / 'model-comparison' / 'testing_models-final_round-2'

CONSTRUCTS = ['depression', 'anxiety', 'psychosis', 'sleep', 'impulse_control']


def build_label_to_dim(path: Path = DEFAULT_DIMENSION_CONFIG) -> dict:
    """
    Case-insensitive label -> dim_id, built from the shared dimension config.
    "does not fit" variants are handled separately in map_label().

    Includes labels from BOTH dimension_labels and dimension_descriptions
    (lowercased) since round-2 free-text output echoes either wording
    depending on how a given item was scored, plus a couple of known
    typos/abbreviations actually seen in the round-2 test file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dimension config not found at {path} -- see "
            "input/shared/dimension_descriptions.yaml for the expected format."
        )
    with open(path) as f:
        config = yaml.safe_load(f)

    mapping = {}
    for construct in CONSTRUCTS:
        entries = {}
        for dim_id, label in config['dimension_labels'].get(construct, {}).items():
            if dim_id != -1:
                entries[label.lower()] = dim_id
        for dim_id, info in config['dimension_descriptions'].get(construct, {}).items():
            entries[info['label'].lower()] = dim_id
        mapping[construct] = entries

    # Known typos/abbreviations seen in the round-2 test file's free text,
    # not present in either canonical label set.
    mapping['sleep']['daytime sleepiness and altertness'] = 1  # typo in test file
    mapping['impulse_control']['dds'] = 6  # abbreviation for Dopamine Dysregulation Syndrome

    return mapping


LABEL_TO_DIM = build_label_to_dim()


def map_label(label: str, construct: str) -> int:
    """Map an expert_label string to a dimension integer (-1 for 'does not fit')."""
    label_lower = label.strip().lower()
    if label_lower in ('does not fit', 'does_not_fit', 'does not fit.'):
        return -1
    mapping = LABEL_TO_DIM.get(construct, {})
    if label_lower in mapping:
        return mapping[label_lower]
    for key, val in mapping.items():
        if key in label_lower or label_lower in key:
            return val
    raise ValueError(
        f"Cannot map expert_label '{label}' for construct '{construct}'. "
        f"Known: {list(mapping.keys())} + 'does not fit'"
    )


def bootstrap_kappa_ci(y1, y2, n_bootstrap=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y1)
    kappas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            k = cohen_kappa_score(y1[idx], y2[idx])
            kappas.append(k)
        except ValueError:
            continue
    if not kappas:
        return np.nan, np.nan
    alpha = (1 - ci) / 2
    return float(np.percentile(kappas, 100 * alpha)), float(np.percentile(kappas, 100 * (1 - alpha)))


def interpret_kappa(kappa):
    if np.isnan(kappa):  return 'N/A'
    if kappa < 0:        return 'Poor'
    if kappa < 0.20:     return 'Slight'
    if kappa < 0.40:     return 'Fair'
    if kappa < 0.60:     return 'Moderate'
    if kappa < 0.80:     return 'Substantial'
    return 'Almost perfect'


def compute_kappa_stats(expert_dims, model_dims, n_bootstrap=1000):
    y_exp = np.array(expert_dims)
    y_mod = np.array(model_dims)
    accuracy = float(np.mean(y_exp == y_mod))
    try:
        kappa = float(cohen_kappa_score(y_exp, y_mod))
    except ValueError:
        kappa = np.nan
    ci_lo, ci_hi = bootstrap_kappa_ci(y_exp, y_mod, n_bootstrap)
    return {
        'n_items':        len(y_exp),
        'accuracy':       round(accuracy, 3),
        'cohen_kappa':    round(kappa, 3) if not np.isnan(kappa) else np.nan,
        'ci_lower':       round(ci_lo,  3) if not np.isnan(ci_lo)  else np.nan,
        'ci_upper':       round(ci_hi,  3) if not np.isnan(ci_hi)  else np.nan,
        'interpretation': interpret_kappa(kappa),
    }


def plot_results(kappa_rows, merged_df, constructs, output_dir):
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    df_k = pd.DataFrame(kappa_rows)

    # Kappa bar chart
    df_per   = df_k[df_k['construct'] != 'OVERALL'].sort_values('cohen_kappa')
    df_over  = df_k[df_k['construct'] == 'OVERALL']
    df_plot  = pd.concat([df_per, df_over]).reset_index(drop=True)

    kappas   = df_plot['cohen_kappa'].tolist()
    ci_lower = df_plot['ci_lower'].tolist()
    ci_upper = df_plot['ci_upper'].tolist()
    labels   = [c.replace('_', ' ').title() for c in df_plot['construct']]

    def _color(k):
        if np.isnan(k): return '#cccccc'
        if k >= 0.80:   return '#27ae60'
        if k >= 0.60:   return '#2ecc71'
        if k >= 0.40:   return '#f39c12'
        return '#e74c3c'

    colors = [_color(k) for k in kappas]
    errors = [
        [k - lo for k, lo in zip(kappas, ci_lower)],
        [hi - k  for k, hi in zip(kappas, ci_upper)],
    ]

    fig, ax = plt.subplots(figsize=(12, max(4, len(df_plot) * 0.75)))
    bars = ax.barh(range(len(labels)), kappas, xerr=errors, color=colors,
                   capsize=5, edgecolor='white')
    for bar, val, hi in zip(bars, kappas, ci_upper):
        if not np.isnan(val):
            ax.text(hi + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=10)
    ax.axvline(0.40, color='red',    linestyle='--', alpha=0.7, label='Moderate (0.40)')
    ax.axvline(0.60, color='orange', linestyle='--', alpha=0.7, label='Substantial (0.60)')
    ax.axvline(0.80, color='green',  linestyle='--', alpha=0.7, label='Almost perfect (0.80)')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's κ")
    ax.set_xlim(0, 1.05)
    ax.set_title("Round-2 Models vs Expert (testing_round-2.csv)\nCohen's κ with 95% CI")
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    fig.savefig(figures_dir / 'kappa_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved kappa_summary.png")

    # Confusion matrices
    for construct in constructs:
        df_c = merged_df[merged_df['construct'] == construct]
        if df_c.empty:
            continue
        conf = pd.crosstab(
            df_c['expert_dimension'],
            df_c['model_dimension'],
            rownames=['Expert'],
            colnames=['Model'],
        )
        nr, nc = conf.shape
        fig, ax = plt.subplots(figsize=(max(4, nc * 1.2), max(3, nr * 0.9)))
        sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', ax=ax,
                    linewidths=0.5, linecolor='#cccccc', cbar=False)
        ax.set_title(f'{construct.capitalize()} — Confusion Matrix\n(Expert rows × Model columns)')
        plt.tight_layout()
        fig.savefig(figures_dir / f'confusion_{construct}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved confusion_{construct}.png")

    # Accuracy bar chart
    df_acc = df_k[df_k['construct'] != 'OVERALL'].copy()
    fig, ax = plt.subplots(figsize=(7, max(3, len(df_acc) * 0.7)))
    ax.barh(df_acc['construct'], df_acc['accuracy'], color='#4575b4', edgecolor='white')
    ax.axvline(0.5, color='#d73027', linewidth=1, linestyle='--')
    for _, row in df_acc.iterrows():
        ax.text(row['accuracy'] + 0.01, row['construct'],
                f"{row['accuracy']:.1%}", va='center', fontsize=9)
    ax.set_xlabel('Accuracy')
    ax.set_xlim(0, 1.15)
    ax.set_title('Round-2 Models: Accuracy per Construct')
    plt.tight_layout()
    fig.savefig(figures_dir / 'accuracy_by_construct.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved accuracy_by_construct.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ROUND-2 MODEL EVALUATION")
    print(f"Models:    {MODELS_DIR}")
    print(f"Test file: {TEST_FILE}")
    print(f"Output:    {OUTPUT_DIR}")
    print("=" * 70)

    test_df = pd.read_csv(TEST_FILE)
    print(f"\nLoaded {len(test_df)} test items")
    print(f"Constructs in test file: {test_df['construct'].value_counts().to_dict()}")

    all_rows   = []
    kappa_rows = []

    for construct in CONSTRUCTS:
        items = test_df[test_df['construct'] == construct].copy().reset_index(drop=True)
        if items.empty:
            print(f"\n[SKIP] {construct}: no test items")
            continue

        construct_dir = MODELS_DIR / construct
        has_model = (construct_dir / 'contrastive_model.pkl').exists() or \
                    (construct_dir / 'prototype_model.pkl').exists()
        if not has_model:
            print(f"\n[SKIP] {construct}: no model found in {construct_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"  {construct.upper()}  ({len(items)} test items)")
        print('='*60)

        # Map expert labels → dim IDs
        try:
            items['expert_dimension'] = items['expert_label'].apply(
                lambda lbl: map_label(lbl, construct)
            )
        except ValueError as e:
            print(f"  ERROR mapping labels: {e}")
            continue

        # item_text is stripped from the public testing_round-2.csv (copyright);
        # falling back to '' lets the pipeline run, though predictions become
        # meaningless without real item wording.
        if 'item_text' not in items.columns:
            items['item_text'] = ''
        item_texts = items['item_text'].fillna('').tolist()
        try:
            preds, probs, model_dims, model_labels, _ = mcv.predict_finetuned_model(
                item_texts, construct, MODELS_DIR
            )
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            print(f"  Skipping {construct} — check encoder files for corruption/incomplete transfer.")
            continue

        dim_label_map = dict(zip(model_dims, model_labels))
        dim_label_map[-1] = 'Does not fit'

        for i, (_, row) in enumerate(items.iterrows()):
            pred_dim   = int(preds[i])
            pred_label = dim_label_map.get(pred_dim, f'Dim {pred_dim}')
            exp_dim    = int(row['expert_dimension'])
            exp_label  = dim_label_map.get(exp_dim, f'Dim {exp_dim}')

            pred_idx   = list(model_dims).index(pred_dim) if pred_dim in model_dims else -1
            confidence = float(probs[i, pred_idx]) if pred_idx >= 0 else np.nan

            prob_dist = {int(model_dims[j]): round(float(probs[i, j]), 6)
                         for j in range(len(model_dims))}

            all_rows.append({
                'construct':              construct,
                'item_id':                row['item_id'],
                'questionnaire':          row['questionnaire'],
                'item_text':              row['item_text'],
                'expert_label_original':  row['expert_label'],
                'expert_dimension':       exp_dim,
                'expert_dimension_label': exp_label,
                'model_dimension':        pred_dim,
                'model_dimension_label':  pred_label,
                'correct':                exp_dim == pred_dim,
                'model_confidence':       round(confidence, 4),
                'probability_distribution': str(prob_dist),
            })

        expert_dims = items['expert_dimension'].tolist()
        model_preds = [int(p) for p in preds]
        stats = compute_kappa_stats(expert_dims, model_preds)
        stats['construct'] = construct
        kappa_rows.append(stats)

        print(f"  Accuracy: {stats['accuracy']:.1%}")
        print(f"  Cohen's κ: {stats['cohen_kappa']:.3f} "
              f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}] "
              f"— {stats['interpretation']}")

        # Per-dimension breakdown
        construct_rows = [r for r in all_rows if r['construct'] == construct]
        df_c = pd.DataFrame(construct_rows)
        print("  Per-dimension accuracy:")
        for dim_id in sorted(dim_label_map.keys()):
            subset = df_c[df_c['expert_dimension'] == dim_id]
            if len(subset) > 0:
                acc = subset['correct'].mean()
                print(f"    Dim {dim_id:3d} ({dim_label_map[dim_id]:<38s}): "
                      f"{int(subset['correct'].sum())}/{len(subset)} = {acc:.1%}")

    if not kappa_rows:
        print("\nNo constructs evaluated.")
        return

    merged_df = pd.DataFrame(all_rows)

    overall = compute_kappa_stats(
        merged_df['expert_dimension'].tolist(),
        merged_df['model_dimension'].tolist(),
    )
    overall['construct'] = 'OVERALL'
    kappa_rows.append(overall)

    print(f"\n{'='*60}")
    print(f"OVERALL  ({len(merged_df)} items, {len(kappa_rows)-1} constructs)")
    print(f"  Accuracy: {overall['accuracy']:.1%}")
    print(f"  Cohen's κ: {overall['cohen_kappa']:.3f} "
          f"[{overall['ci_lower']:.3f}, {overall['ci_upper']:.3f}] "
          f"— {overall['interpretation']}")
    print('='*60)

    # Save outputs
    kappa_df = pd.DataFrame(kappa_rows)[[
        'construct', 'n_items', 'accuracy', 'cohen_kappa',
        'ci_lower', 'ci_upper', 'interpretation'
    ]]
    kappa_df.to_csv(OUTPUT_DIR / 'kappa_round2_models_vs_experts.csv', index=False)
    print(f"\nSaved kappa summary → kappa_round2_models_vs_experts.csv")

    merged_df.to_csv(OUTPUT_DIR / 'predictions_all.csv', index=False)
    print(f"Saved per-item predictions → predictions_all.csv")

    for construct in merged_df['construct'].unique():
        df_c = merged_df[merged_df['construct'] == construct]
        df_c.to_csv(OUTPUT_DIR / f'predictions_{construct}.csv', index=False)
        print(f"Saved → predictions_{construct}.csv")

    print("\nGenerating figures...")
    evaluated = [r['construct'] for r in kappa_rows if r['construct'] != 'OVERALL']
    plot_results(kappa_rows, merged_df, evaluated, OUTPUT_DIR)

    print("\nDone!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension-config', type=str, default=None,
                        help='Path to a dimension_descriptions.yaml defining the '
                             f'constructs/dimensions to evaluate (default: {DEFAULT_DIMENSION_CONFIG}). '
                             'Point this at your own copy to run on different constructs '
                             'without editing this script.')
    args = parser.parse_args()

    if args.dimension_config:
        LABEL_TO_DIM = build_label_to_dim(args.dimension_config)

    main()
