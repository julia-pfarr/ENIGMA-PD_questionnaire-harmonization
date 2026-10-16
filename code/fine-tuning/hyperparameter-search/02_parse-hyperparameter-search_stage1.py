"""
Parse Stage 1 HP Search Results and Prepare Stage 2
=====================================================

Reads LOGO-CV results from stage 1 (8 HP configs × 5 constructs),
finds the best epochs/lr/margin per construct, and writes a config
file for stage 2 (boost × percentile search on top of the winner).

Usage:
    python parse_stage1_results.py \
        --results-dir hp_search_hn/stage1 \
        --output stage2_configs.csv

Then submit stage 2:
    sbatch submit_hp_stage2.sh
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys


def find_cv_results(results_dir: Path) -> pd.DataFrame:
    """
    Scan stage 1 output directory for CV result files.

    Expected structure:
        results_dir/
        ├── depression/
        │   ├── e3_lr1e-5_m0.2/
        │   │   └── contrastive_cv_results.csv
        │   ├── e3_lr1e-5_m0.3/
        │   │   └── contrastive_cv_results.csv
        │   └── ...
        ├── anxiety/
        └── ...
    """
    rows = []

    for construct_dir in sorted(results_dir.iterdir()):
        if not construct_dir.is_dir():
            continue
        construct = construct_dir.name

        for config_dir in sorted(construct_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            # Parse config from directory name: e3_lr1e-5_m0.2
            config_name = config_dir.name
            parts = config_name.split('_')

            try:
                epochs = int(parts[0].replace('e', ''))
                lr = float(parts[1].replace('lr', ''))
                margin = float(parts[2].replace('m', ''))
            except (IndexError, ValueError):
                print(f"  Warning: couldn't parse config from '{config_name}', skipping")
                continue

            # Find CV results file
            cv_file = config_dir / construct / 'contrastive_cv_results.csv'
            if not cv_file.exists():
                # Try without nested construct dir
                cv_file = config_dir / 'contrastive_cv_results.csv'
            if not cv_file.exists():
                # Try glob
                candidates = list(config_dir.rglob('*cv_results.csv'))
                if candidates:
                    cv_file = candidates[0]
                else:
                    print(f"  Warning: no CV results found in {config_dir}")
                    continue

            # Read CV results
            try:
                cv_df = pd.read_csv(cv_file)

                # Extract mean accuracy (adapt column names to your output)
                if 'accuracy' in cv_df.columns:
                    mean_acc = cv_df['accuracy'].mean()
                    std_acc = cv_df['accuracy'].std()
                elif 'Accuracy' in cv_df.columns:
                    mean_acc = cv_df['Accuracy'].mean()
                    std_acc = cv_df['Accuracy'].std()
                else:
                    # Try to find any accuracy-like column
                    acc_cols = [c for c in cv_df.columns if 'acc' in c.lower()]
                    if acc_cols:
                        mean_acc = cv_df[acc_cols[0]].mean()
                        std_acc = cv_df[acc_cols[0]].std()
                    else:
                        print(f"  Warning: no accuracy column in {cv_file}")
                        continue

                # F1 if available
                f1_cols = [c for c in cv_df.columns if 'f1' in c.lower()]
                mean_f1 = cv_df[f1_cols[0]].mean() if f1_cols else np.nan

                rows.append({
                    'construct': construct,
                    'epochs': epochs,
                    'learning_rate': lr,
                    'margin': margin,
                    'mean_accuracy': round(mean_acc, 4),
                    'std_accuracy': round(std_acc, 4),
                    'mean_f1': round(mean_f1, 4) if not np.isnan(mean_f1) else np.nan,
                    'n_folds': len(cv_df),
                    'config_dir': str(config_dir),
                })

            except Exception as e:
                print(f"  Warning: error reading {cv_file}: {e}")
                continue

    return pd.DataFrame(rows)


def select_best_per_construct(results_df: pd.DataFrame) -> pd.DataFrame:
    """Select best HP config per construct by mean accuracy."""
    best_rows = []

    for construct in results_df['construct'].unique():
        df_c = results_df[results_df['construct'] == construct].copy()
        df_c = df_c.sort_values('mean_accuracy', ascending=False)

        best = df_c.iloc[0]
        best_rows.append(best)

        print(f"\n{construct.upper()}")
        print(f"  Best: epochs={int(best['epochs'])}, lr={best['learning_rate']}, "
              f"margin={best['margin']} → acc={best['mean_accuracy']:.4f} ± {best['std_accuracy']:.4f}")
        print(f"  All configs:")
        for _, row in df_c.iterrows():
            marker = " ← best" if row['mean_accuracy'] == best['mean_accuracy'] else ""
            print(f"    e={int(row['epochs'])}, lr={row['learning_rate']}, "
                  f"m={row['margin']}: {row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f}{marker}")

    return pd.DataFrame(best_rows)


def generate_stage2_configs(best_df: pd.DataFrame, output_path: str):
    """
    Generate stage 2 config file: best epochs/lr/margin × boost × percentile.

    Stage 2 search grid:
        boost:      [1.0, 2.0, 3.0, 5.0]
        percentile: [50, 75, 90]

    boost=1.0 is effectively "no hard negative mining" (control condition).
    """
    boost_values = [1.0, 2.0, 3.0, 5.0]
    percentile_values = [50, 75, 90]

    configs = []
    config_id = 0

    for _, row in best_df.iterrows():
        construct = row['construct']
        epochs = int(row['epochs'])
        lr = row['learning_rate']
        margin = row['margin']

        for boost in boost_values:
            for pct in percentile_values:
                configs.append({
                    'config_id': config_id,
                    'construct': construct,
                    'epochs': epochs,
                    'learning_rate': lr,
                    'margin': margin,
                    'hard_negative_boost': boost,
                    'hard_negative_percentile': pct,
                })
                config_id += 1

    configs_df = pd.DataFrame(configs)
    configs_df.to_csv(output_path, index=False)

    n_constructs = len(best_df)
    n_per_construct = len(boost_values) * len(percentile_values)
    total = len(configs_df)

    print(f"\nStage 2 configs written to {output_path}")
    print(f"  {n_constructs} constructs × {n_per_construct} boost/percentile combos = {total} jobs")
    print(f"  boost values: {boost_values}")
    print(f"  percentile values: {percentile_values}")
    print(f"\n  Update submit_hp_stage2.sh with: #SBATCH --array=0-{total - 1}")

    return configs_df


def main():
    parser = argparse.ArgumentParser(
        description="Parse stage 1 HP results and generate stage 2 configs"
    )
    parser.add_argument('--results-dir', required=True,
                        help='Stage 1 results directory (hp_search_hn/stage1)')
    parser.add_argument('--output', default='stage2_configs.csv',
                        help='Output config CSV for stage 2')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} not found")
        sys.exit(1)

    print("=" * 70)
    print("PARSING STAGE 1 RESULTS")
    print("=" * 70)

    # Find and parse all CV results
    results_df = find_cv_results(results_dir)

    if len(results_df) == 0:
        print("No results found! Check directory structure.")
        sys.exit(1)

    print(f"\nFound {len(results_df)} results across "
          f"{results_df['construct'].nunique()} constructs")

    # Save full results table
    full_path = Path(args.output).parent / 'stage1_all_results.csv'
    results_df.to_csv(full_path, index=False)
    print(f"Full results saved to {full_path}")

    # Select best per construct
    print(f"\n{'=' * 70}")
    print("BEST CONFIG PER CONSTRUCT")
    print("=" * 70)

    best_df = select_best_per_construct(results_df)

    # Generate stage 2 configs
    print(f"\n{'=' * 70}")
    print("GENERATING STAGE 2 CONFIGS")
    print("=" * 70)

    generate_stage2_configs(best_df, args.output)


if __name__ == "__main__":
    main()