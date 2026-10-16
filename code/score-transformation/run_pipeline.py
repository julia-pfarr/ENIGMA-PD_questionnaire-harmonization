"""
Run Score Transformation Pipeline on ENIGMA-PD master table.

Adapts the wide-format master table (one row per participant, all questionnaire
items as columns, -9 = missing) into the format expected by ScoreTransformer,
then runs the full pipeline:

  1. Raw dimension scores (sum of item responses, with imputation)
  2. Proportion scores (raw / max_dimension_score)
  3. Per-questionnaire C-scores (ECDF within questionnaire × dimension)
  4. Averaged-CDF C-scores (av_cdf_score; per-questionnaire ECDFs averaged with
     equal weight per questionnaire, used in the downstream analyses)

Prerequisites:
  Run create_item_dimension_mapping.py first to generate item_dimension_mapping.csv.

Output (in --output-dir, default output/score-transformation/):
  dimension_scores.csv
  dimension_scores_wide.csv
  cdf_norms_enigma-pd.json

Usage:
  python3 run_pipeline.py                                    # real data, real output dir
  python3 run_pipeline.py \\
      --master-table ../../input/score-transformation/synthetic-data_master-table_subject-data.csv \\
      --output-dir /tmp/synthetic-test-output                # synthetic data, isolated output dir
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from score_transformation import ScoreTransformer

PROJECT_DIR = CODE_DIR.parent.parent
INPUT_DIR = PROJECT_DIR / "input" / "score-transformation"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output" / "score-transformation"

DEFAULT_MASTER_TABLE = INPUT_DIR / "master-table_subject-data.csv"
ITEM_MAPPING = INPUT_DIR / "item_dimension_mapping.csv"
MAX_SCORES = INPUT_DIR / "max_dimension_scores.csv"

MISSING_SENTINEL = -9
META_COLS = [
    "participant_id", "site", "language", "age", "sex", "group",
    "LEDD_total", "disease_duration", "HY_stage", "MDS_UPDRS_III_total_score",
]


def reshape_to_long(master_df: pd.DataFrame, item_mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Produce one row per (participant, questionnaire) pair.

    A participant is included for a given questionnaire if they have at least
    one non-NaN response to any item of that questionnaire that appears in
    the item dimension mapping (i.e., items contributing to a scored dimension).
    """
    parts = []
    for questionnaire, q_df in item_mapping.groupby("questionnaire"):
        q_items = q_df["item_id"].unique().tolist()
        present = [i for i in q_items if i in master_df.columns]
        if not present:
            continue

        has_any = master_df[present].notna().any(axis=1)
        if not has_any.any():
            continue

        keep_cols = [c for c in META_COLS if c in master_df.columns] + present
        subset = master_df.loc[has_any, keep_cols].copy()
        subset["questionnaire"] = questionnaire
        parts.append(subset)

    if not parts:
        raise ValueError("No participant-questionnaire combinations found in master table.")

    long_df = pd.concat(parts, ignore_index=True)
    print(f"  {len(long_df)} (participant × questionnaire) rows | "
          f"{long_df['questionnaire'].nunique()} questionnaires: "
          f"{sorted(long_df['questionnaire'].unique())}")
    return long_df


def main(master_table: Path = DEFAULT_MASTER_TABLE, output_dir: Path = DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    norms_out = output_dir / "cdf_norms_enigma-pd.json"

    print("=" * 70)
    print("ENIGMA-PD SCORE TRANSFORMATION PIPELINE")
    print("=" * 70)
    print(f"\nMaster table: {master_table}")
    print(f"Output dir:   {output_dir}")

    if not ITEM_MAPPING.exists():
        raise FileNotFoundError(
            f"item_dimension_mapping.csv not found at {ITEM_MAPPING}.\n"
            "Run create_item_dimension_mapping.py first."
        )
    if not Path(master_table).exists():
        raise FileNotFoundError(
            f"Master table not found at {master_table}.\n"
            "See README 'Data' section — either supply your own "
            "master-table_subject-data.csv, or run "
            "generate_synthetic_master_table.py and pass its output via --master-table."
        )

    # Load and preprocess master table (-9 → NaN)
    print(f"\nLoading master table...")
    master_df = pd.read_csv(master_table)
    master_df = master_df.replace(MISSING_SENTINEL, np.nan)
    print(f"  {len(master_df)} participants, {len(master_df.columns)} columns")
    print(f"  Sites: {dict(master_df['site'].value_counts())}")

    item_mapping = pd.read_csv(ITEM_MAPPING)
    print(f"  Item mapping: {len(item_mapping)} items across "
          f"{item_mapping['questionnaire'].nunique()} questionnaire(s)")

    # Reshape to long format (one row per participant × questionnaire)
    print("\nReshaping master table to participant × questionnaire format...")
    long_df = reshape_to_long(master_df, item_mapping)

    # Initialize transformer
    transformer = ScoreTransformer(
        min_items_high=4,
        min_items_moderate=2,
        missing_threshold=0.5,
    )
    transformer.load_mapping(str(ITEM_MAPPING), str(MAX_SCORES))

    # Step 1–2: Raw scores and proportion scores
    print("\nComputing raw and proportion scores...")
    scores_df = transformer.compute_scores(long_df, "participant_id", "questionnaire")

    # Step 3: Per-questionnaire CDF scores (q_cdf_score)
    print("\nComputing per-questionnaire CDF scores (q_cdf_score)...")
    scores_df = transformer.compute_q_cdf_scores(scores_df, "questionnaire")

    # Step 4: Averaged-CDF scores (av_cdf_score) — default for all analyses
    print("\nComputing averaged-CDF scores (av_cdf_score)...")
    scores_df = transformer.compute_av_cdf_scores(scores_df, "questionnaire")

    # Print summary, export results and norms
    transformer._print_summary(scores_df, "questionnaire")
    transformer.export(scores_df, str(output_dir))
    transformer.save_norms(str(norms_out))

    print(f"\nDone. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master-table", type=Path, default=DEFAULT_MASTER_TABLE,
                        help="Path to master-table_subject-data.csv "
                             f"(default: {DEFAULT_MASTER_TABLE}). Point this at "
                             "synthetic-data_master-table_subject-data.csv (see "
                             "generate_synthetic_master_table.py) to test the pipeline "
                             "without real data.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}). "
                             "Use a scratch directory when running on synthetic/test "
                             "data so you never overwrite the committed real results.")
    args = parser.parse_args()
    main(master_table=args.master_table, output_dir=args.output_dir)
