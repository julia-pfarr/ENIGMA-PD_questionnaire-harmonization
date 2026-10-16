"""
Generate a Synthetic Master Table for the Score-Transformation Pipeline
=========================================================================

Real ENIGMA-PD participant-level data cannot be redistributed here. Without it, 
reviewers can't run `run_pipeline.py` or the validation scripts at all. 
This script creates a FAKE `master-table_subject-data.csv` with the exact 
schema real data must have, so anyone can run the full pipeline end-to-end 
and see that the code behaves as described in the manuscript.

IMPORTANT: The generated values are entirely synthetic. Running the pipeline
on them will NOT reproduce the manuscript's reported numbers — only the real
ENIGMA-PD cohort can do that. Use this to verify the pipeline runs and
produces sensibly-structured output, not to check specific statistics.

Generative model
-----------------
- Participant metadata (site, language, age, sex, group, disease_duration,
  HY_stage, LEDD_total, MDS_UPDRS_III_total_score) drawn from simple
  plausible distributions; clinical covariates are PD-only (HC rows get the
  missing sentinel, matching real data conventions).
- One latent severity trait per participant x (construct, dimension), shared
  across every questionnaire that measures it. This is what gives the
  synthetic data realistic convergent validity (the same dimension measured
  by two different questionnaires correlates) and discriminant validity
  (different dimensions don't). PD participants get a higher mean trait
  (PD_TRAIT_SHIFT) than HC, so downstream group-effect analyses have
  something non-trivial to detect.
- Each item's response is that latent trait (+ small item-specific noise)
  mapped onto the item's own scored response options, taken from
  `questionnaire_response-options.csv`'s `scoring` column (sorted ascending
  so the mapping direction is always "higher latent -> higher score",
  independent of how any given item happens to be reverse-coded) — so scale
  ranges match the real instruments exactly. A handful of items (e.g. MADRS,
  PHQ-9, IDS-CR, GHQ-28) aren't listed in that file; for those, a standard
  0..max_item_score integer scale is used instead (matches how those
  instruments are actually scored).
- A small fraction of item responses are set to the missing sentinel (-9) to
  exercise the pipeline's missing-data handling.

The output filename is deliberately prefixed `synthetic-data_` (never plain
`master-table_subject-data.csv`) so it's never mistaken for real data and
never silently picked up by a script expecting the real file. To run the
pipeline on it, pass it explicitly via --master-table (see run_pipeline.py
and hypothesis-testing.py's own --master-table/--score-dir/--output-dir
flags) together with a scratch --output-dir, so you never overwrite the
real, committed results in output/score-transformation/.

Usage:
    python3 code/score-transformation/generate_synthetic_master_table.py
    python3 code/score-transformation/generate_synthetic_master_table.py --n-participants 500 --seed 1

Output:
    input/score-transformation/synthetic-data_master-table_subject-data.csv
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent.parent
INPUT_DIR = PROJECT_DIR / "input" / "score-transformation"

ITEM_MAPPING = INPUT_DIR / "item_dimension_mapping.csv"
RESPONSE_OPTIONS = INPUT_DIR / "questionnaire_response-options.csv"
OUTPUT_PATH = INPUT_DIR / "synthetic-data_master-table_subject-data.csv"

MISSING_SENTINEL = -9

# Site sizes are deliberately uneven: four sites >=50 participants (each gets
# its own dummy in the pipeline's site-consolidation logic) and one small
# site <50 (gets folded into "other") -- so both code paths get exercised.
SITES = [("SITE_A", 150), ("SITE_B", 100), ("SITE_C", 90), ("SITE_D", 60), ("SITE_E", 20)]
LANGUAGES = [("en", 0.60), ("de", 0.28), ("fr", 0.12)]
PD_FRACTION = 0.60
ITEM_MISSING_RATE = 0.03
PD_TRAIT_SHIFT = 0.6   # SD units; PD participants score this much higher on average


def load_scoring_options() -> dict:
    """item_id -> sorted list of possible numeric scored values."""
    resp = pd.read_csv(RESPONSE_OPTIONS)
    scoring = {}
    for _, row in resp.iterrows():
        try:
            values = ast.literal_eval(row["scoring"])
        except (ValueError, SyntaxError):
            continue
        scoring[row["question_number"]] = sorted(float(v) for v in values)
    return scoring


def latent_to_response(latent: np.ndarray, sorted_options: list,
                       rng: np.random.Generator) -> np.ndarray:
    """Map a continuous latent trait to one of an item's discrete scored options
    (ascending order, so higher latent always means a higher/more symptomatic score)."""
    noisy = latent + rng.normal(0, 0.4, size=latent.shape)
    percentile = norm.cdf(noisy)
    idx = np.clip((percentile * len(sorted_options)).astype(int), 0, len(sorted_options) - 1)
    return np.asarray(sorted_options)[idx]


def build_participant_metadata(n: int, rng: np.random.Generator) -> pd.DataFrame:
    site_names, site_counts = zip(*SITES)
    site_counts = np.array(site_counts)
    site_counts = np.round(site_counts / site_counts.sum() * n).astype(int)
    site_counts[-1] += n - site_counts.sum()  # fix rounding drift
    sites = np.repeat(site_names, site_counts)
    rng.shuffle(sites)

    lang_names, lang_probs = zip(*LANGUAGES)
    languages = rng.choice(lang_names, size=n, p=lang_probs)

    group = (rng.random(n) < PD_FRACTION).astype(int)
    age = np.clip(rng.normal(62, 9, size=n), 30, 90).round(1)
    sex = rng.integers(0, 2, size=n)

    disease_duration = np.where(
        group == 1, np.clip(rng.normal(6, 4, size=n), 0, None).round(1), MISSING_SENTINEL
    )
    hy_stage = np.where(
        group == 1, rng.integers(1, 6, size=n), MISSING_SENTINEL
    )
    ledd = np.where(
        group == 1, np.clip(rng.normal(600, 300, size=n), 0, None).round(0), MISSING_SENTINEL
    )
    updrs = np.where(
        group == 1, np.clip(rng.normal(28, 12, size=n), 0, None).round(0), MISSING_SENTINEL
    )

    return pd.DataFrame({
        "participant_id": [f"SYN{i:04d}" for i in range(1, n + 1)],
        "site": sites,
        "language": languages,
        "age": age,
        "sex": sex,
        "group": group,
        "disease_duration": disease_duration,
        "HY_stage": hy_stage,
        "LEDD_total": ledd,
        "MDS_UPDRS_III_total_score": updrs,
    })


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-participants", type=int, default=420,
                        help="Number of synthetic participants (default: 420, "
                             "large enough to clear every downstream sample-size threshold)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help=f"Output path (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    if not ITEM_MAPPING.exists() or not RESPONSE_OPTIONS.exists():
        raise FileNotFoundError(
            "item_dimension_mapping.csv / questionnaire_response-options.csv not found "
            f"in {INPUT_DIR}. These ship with the repo; run this script from a full checkout."
        )

    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("GENERATING SYNTHETIC MASTER TABLE")
    print("=" * 70)
    print("\nThis creates FAKE data for pipeline testing only -- it does not")
    print("reproduce the manuscript's reported results. See script docstring.\n")

    item_mapping = pd.read_csv(ITEM_MAPPING)
    scoring = load_scoring_options()

    meta_df = build_participant_metadata(args.n_participants, rng)
    n = len(meta_df)
    print(f"Participants: {n}")
    print(f"  Sites: {meta_df['site'].value_counts().to_dict()}")
    print(f"  Languages: {meta_df['language'].value_counts().to_dict()}")
    print(f"  Group (0=HC, 1=PD): {meta_df['group'].value_counts().to_dict()}")

    # One latent trait per participant x (construct, dimension), shared across
    # every questionnaire that measures it.
    dims = item_mapping[["construct", "dimension"]].drop_duplicates()
    latent = {}
    for _, row in dims.iterrows():
        key = (row["construct"], row["dimension"])
        base = rng.normal(0, 1, size=n)
        shift = np.where(meta_df["group"].values == 1, PD_TRAIT_SHIFT, 0.0)
        latent[key] = base + shift

    print(f"\nGenerating item responses for {item_mapping['item_id'].nunique()} items "
          f"across {item_mapping['questionnaire'].nunique()} questionnaires ...")

    item_cols = {}
    n_fallback = 0
    n_skipped = 0
    for _, row in item_mapping.iterrows():
        item_id = row["item_id"]
        options = scoring.get(item_id)
        if not options:
            # Not every item in item_dimension_mapping.csv has a matching row in
            # questionnaire_response-options.csv (e.g. MADRS, PHQ-9, IDS-CR,
            # GHQ-28). Fall back to a standard 0..max_item_score integer scale,
            # which matches how those instruments are actually scored.
            max_score = row.get("max_item_score")
            if pd.notna(max_score) and max_score >= 0:
                options = [float(v) for v in range(int(max_score) + 1)]
                n_fallback += 1
            else:
                n_skipped += 1
                continue
        key = (row["construct"], row["dimension"])
        values = latent_to_response(latent[key], options, rng).astype(float)
        missing_mask = rng.random(n) < ITEM_MISSING_RATE
        values[missing_mask] = MISSING_SENTINEL
        item_cols[item_id] = values

    if n_fallback:
        print(f"  {n_fallback} item(s) not found in questionnaire_response-options.csv; "
              f"used a 0..max_item_score fallback scale instead.")
    if n_skipped:
        print(f"  WARNING: {n_skipped} item(s) had no usable scoring info and "
              f"were skipped (will read as entirely missing).")

    items_df = pd.DataFrame(item_cols)
    master_df = pd.concat([meta_df, items_df], axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(args.output, index=False)

    print(f"\n{len(master_df.columns)} columns "
          f"({len(meta_df.columns)} metadata + {len(item_cols)} items), {len(master_df)} rows")
    print(f"Saved synthetic master table to {args.output}")
    print("\nNext (use a scratch --output-dir so you never overwrite the real, "
          "committed results):")
    print(f"  python3 code/score-transformation/run_pipeline.py \\")
    print(f"      --master-table {args.output} \\")
    print(f"      --output-dir /tmp/synthetic-test-output")


if __name__ == "__main__":
    main()
