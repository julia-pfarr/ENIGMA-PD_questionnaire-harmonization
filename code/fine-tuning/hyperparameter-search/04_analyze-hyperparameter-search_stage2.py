"""
Analyze Stage 2 HP Search Results
===================================

Parses LOGO-CV results from stage 2 (boost × percentile search)
and reports the best configuration per construct.

Usage:
    python analyze_stage2_results.py --results-dir hp_search_hn/stage2
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def find_cv_results(results_dir: Path) -> pd.DataFrame:
    """Scan stage 2 output directories for CV results."""
    rows = []

    for construct_dir in sorted(results_dir.iterdir()):
        if not construct_dir.is_dir():
            continue
        construct = construct_dir.name

        for config_dir in sorted(construct_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            # Parse: boost3.0_pct75
            config_name = config_dir.name
            try:
                parts = config_name.split('_')
                boost = float(parts[0].replace('boost', ''))
                pct = float(parts[1].replace('pct', ''))
            except (IndexError, ValueError):
                print(f"  Warning: couldn't parse '{config_name}', skipping")
                continue

            # Find CV results
            candidates = list(config_dir.rglob('*cv_results.csv'))
            if not candidates:
                print(f"  Warning: no CV results in {config_dir}")
                continue

            cv_file = candidates[0]

            try:
                cv_df = pd.read_csv(cv_file)

                # Find accuracy column
                acc_col = None
                for col in ['accuracy', 'Accuracy', 'accuracy_no_minus1']:
                    if col in cv_df.columns:
                        acc_col = col
                        break

                if acc_col is None:
                    acc_cols = [c for c in cv_df.columns if 'acc' in c.lower()]
                    acc_col = acc_cols[0] if acc_cols else None

                if acc_col is None:
                    print(f"  Warning: no accuracy column in {cv_file}")
                    continue

                mean_acc = cv_df[acc_col].mean()
                std_acc = cv_df[acc_col].std()

                # F1 if available
                f1_cols = [c for c in cv_df.columns if 'f1' in c.lower()]
                mean_f1 = cv_df[f1_cols[0]].mean() if f1_cols else np.nan

                # Kappa if available
                kappa_cols = [c for c in cv_df.columns if 'kappa' in c.lower()]
                mean_kappa = cv_df[kappa_cols[0]].mean() if kappa_cols else np.nan

                rows.append({
                    'construct': construct,
                    'boost': boost,
                    'percentile': pct,
                    'mean_accuracy': round(mean_acc, 4),
                    'std_accuracy': round(std_acc, 4),
                    'mean_f1': round(mean_f1, 4) if not np.isnan(mean_f1) else np.nan,
                    'mean_kappa': round(mean_kappa, 4) if not np.isnan(mean_kappa) else np.nan,
                    'n_folds': len(cv_df),
                })

            except Exception as e:
                print(f"  Warning: error reading {cv_file}: {e}")
                continue

    return pd.DataFrame(rows)


def print_results(df: pd.DataFrame):
    """Print results with best config per construct."""

    print("\n" + "=" * 80)
    print("STAGE 2 RESULTS: HARD NEGATIVE MINING HYPERPARAMETER SEARCH")
    print("=" * 80)

    for construct in sorted(df['construct'].unique()):
        df_c = df[df['construct'] == construct].sort_values('mean_accuracy', ascending=False)

        best = df_c.iloc[0]
        control = df_c[df_c['boost'] == 1.0]
        control_acc = control['mean_accuracy'].values[0] if len(control) > 0 else np.nan

        print(f"\n{'─' * 80}")
        print(f"  {construct.upper()}")
        print(f"{'─' * 80}")
        print(f"  Best:    boost={best['boost']}, pct={best['percentile']:.0f} "
              f"→ acc={best['mean_accuracy']:.4f} ± {best['std_accuracy']:.4f}")
        if not np.isnan(control_acc):
            diff = best['mean_accuracy'] - control_acc
            print(f"  Control: boost=1.0 (no HN mining) "
                  f"→ acc={control_acc:.4f} "
                  f"({'+'if diff >= 0 else ''}{diff:.4f} from HN mining)")

        print(f"\n  {'Boost':>6}  {'Pct':>5}  {'Accuracy':>10}  {'± Std':>8}  {'F1':>8}  {'Kappa':>8}")
        print(f"  {'─'*52}")
        for _, row in df_c.iterrows():
            marker = " ← best" if (row['boost'] == best['boost'] and
                                    row['percentile'] == best['percentile']) else ""
            marker = " ← control" if row['boost'] == 1.0 and marker == "" else marker
            f1_str = f"{row['mean_f1']:.4f}" if not np.isnan(row['mean_f1']) else "   n/a"
            kp_str = f"{row['mean_kappa']:.4f}" if not np.isnan(row['mean_kappa']) else "   n/a"
            print(f"  {row['boost']:>6.1f}  {row['percentile']:>5.0f}  "
                  f"{row['mean_accuracy']:>10.4f}  {row['std_accuracy']:>8.4f}  "
                  f"{f1_str}  {kp_str}{marker}")

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY: BEST CONFIG PER CONSTRUCT")
    print(f"{'=' * 80}")
    print(f"\n  {'Construct':<20} {'Boost':>6} {'Pct':>5} {'Accuracy':>10} {'vs Control':>11}")
    print(f"  {'─'*56}")

    for construct in sorted(df['construct'].unique()):
        df_c = df[df['construct'] == construct].sort_values('mean_accuracy', ascending=False)
        best = df_c.iloc[0]
        control = df_c[df_c['boost'] == 1.0]
        control_acc = control['mean_accuracy'].values[0] if len(control) > 0 else np.nan
        diff = best['mean_accuracy'] - control_acc if not np.isnan(control_acc) else np.nan
        diff_str = f"{'+'if diff >= 0 else ''}{diff:.4f}" if not np.isnan(diff) else "n/a"

        print(f"  {construct:<20} {best['boost']:>6.1f} {best['percentile']:>5.0f} "
              f"{best['mean_accuracy']:>10.4f} {diff_str:>11}")


def main():
    parser = argparse.ArgumentParser(description="Analyze stage 2 HP search results")
    parser.add_argument('--results-dir', required=True,
                        help='Stage 2 results directory (hp_search_hn/stage2)')
    parser.add_argument('--output', default=None,
                        help='Save full results table to CSV')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} not found")
        sys.exit(1)

    df = find_cv_results(results_dir)

    if len(df) == 0:
        print("No results found! Check directory structure.")
        sys.exit(1)

    print(f"Found {len(df)} results across {df['construct'].nunique()} constructs")

    print_results(df)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()