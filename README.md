# ENIGMA-PD - Questionnaire Harmonization Project

This repository holds all code and other necessary files to reproduce analysis
reported in the manuscript [add when submitted]. 

Harmonizing psychiatric questionnaires across the ENIGMA-PD consortium. Many
sites measure the same constructs — depression, anxiety, apathy, psychosis,
impulse control, sleep — with different instruments (BDI, GDS, HADS, ESS,
RBDSQ, ...), different items, and different response scales. This project maps
individual items onto shared, construct-specific dimensions and converts raw
responses into scores that are directly comparable across questionnaires and
sites.

The pipeline has three main stages:

1. **Item-dimension mapping** — assign every item, across every questionnaire,
   to a construct dimension (e.g. BDI item 9 → depression / mood & affective
   symptoms). Initial mappings came from an expert survey; a fine-tuned
   sentence-embedding model now automates this for new questionnaires.
2. **Score transformation** — convert raw item responses into comparable
   dimension scores using empirical-CDF-based percentile ranks ("C-scores"),
   within each questionnaire × dimension and averaged across questionnaires.
3. **Validation** — test whether the harmonized scores behave as they should:
   do different questionnaires measuring the same dimension agree (H1), does
   the item-dimension structure hold up under confirmatory factor analysis
   (H2), are dimensions internally consistent (H3), and do scores relate to
   clinical group membership as expected?

## Repository layout

```
code/
  score-transformation/   mapping + C-score pipeline + validation (H1-H3, intercept correction, ...)
  fine-tuning/            training and evaluation of the item-to-dimension embedding models
  expert-survey/          analysis of the expert item-mapping surveys (Fleiss' kappa, agreement)
  similarity-analysis/    base (non-fine-tuned) embedding model comparison
  power-and-precision-analyses/  pre-registered sample-size justification (power/precision simulations)
input/                    item mappings, response options, expert-survey ratings, model test sets
output/                   aggregate results: fit statistics, figures, model comparison summaries
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

This repository ships aggregate and item-level results, but **not raw
participant-level data**. To run the score-transformation pipeline on your own
cohort, supply a `master-table_subject-data.csv` at
`input/score-transformation/` with one row per participant, one column per
questionnaire item (missing = `-9`), plus demographic columns
(`participant_id`, `site`, `age`, `sex`, `group`, ...). See
`input/score-transformation/item_dimension_mapping.csv` for the item columns
the pipeline looks for.

Item wording (`item_text`/`question_text`) has been stripped from every file
in this repo, since most of the questionnaires involved are copyrighted
clinical instruments. Items are identified by `item_id` throughout; obtain the
instruments themselves from their publishers if you need the literal wording.

### Testing the pipeline without real data

Since real participant data can't be included, `run_pipeline.py` and the
validation scripts can't be run out of the box. To verify that the
*code* works — without access to the real cohort — generate a fake dataset:

```bash
python3 code/score-transformation/generate_synthetic_master_table.py
```

This writes `input/score-transformation/synthetic-data_master-table_subject-data.csv`
— randomly generated values matching the real schema (see the script's
docstring for the generative model). The `synthetic-data_` prefix is
deliberate: it's never picked up by anything automatically, and is never
confusable with `master-table_subject-data.csv` (the real filename the
pipeline expects by default).

**Running the pipeline on this file will NOT reproduce the numbers reported
in the manuscript** — only the real, non-public PPMI/ENIGMA-PD cohort
produces those. Its only purpose is to confirm the pipeline runs end-to-end
and produces sensibly-structured output. All results already committed in
this repo's `output/` directory come from the real cohort, not from this
synthetic data.

To avoid ever overwriting those real, committed results, point every script
at scratch input/output paths explicitly — `run_pipeline.py`,
`hypothesis-testing.py`, and `exploratory-validation-analyses.py` all accept
`--master-table` / `--score-dir` / `--output-dir` overrides for exactly this:

```bash
python3 code/score-transformation/run_pipeline.py \
    --master-table input/score-transformation/synthetic-data_master-table_subject-data.csv \
    --output-dir /tmp/synthetic-test-output

python3 code/score-transformation/validation/hypothesis-testing.py \
    --master-table input/score-transformation/synthetic-data_master-table_subject-data.csv \
    --score-dir /tmp/synthetic-test-output \
    --output-dir /tmp/synthetic-test-output/hypotheses-testing

python3 code/score-transformation/validation/exploratory-validation-analyses.py \
    --score-dir /tmp/synthetic-test-output \
    --output-dir /tmp/synthetic-test-output/validation \
    --out-dir /tmp/synthetic-test-output/results_figures_tables
```

## Trained models

The fine-tuned item-to-dimension embedding models (one per construct) are
hosted on Hugging Face: https://huggingface.co/collections/julia-pfarr/harmonica.
`harmonizer.py` downloads and caches them automatically — just omit
`--model-dir` when running `harmonize` (uses `huggingface_hub`'s own cache, so
each model is only downloaded once). Point `--hf-repo <repo id>` at a
different repo to use your own fine-tuned model for that construct instead
of the default HarmoniCA one, or pass `--model-dir <path>` to use a local
copy instead of downloading at all.

## Running the pipeline

Tries the whole thing on synthetic data (safe to
run repeatedly — writes only to `./quickstart-output/`, never touches the
real `output/` tree):

```bash
./code/score-transformation/run_all.sh
```

Once you have the real (non-public) cohort data in place (see "Data" above),
`./code/score-transformation/run_all.sh --real` runs the same steps against it, writing to the real
`output/score-transformation/` — this is what reproduces the manuscript's
results. Under the hood it just runs each step below with its own defaults;
run them individually if you want to inspect one step at a time or pass
different flags (`--score-dir`, `--output-dir`, `--dimension-config`, etc. —
see each script's `--help`):

```bash
# Step 0: item-dimension mapping (only needed when the harmonizer output changes)
python3 code/score-transformation/create_item_dimension_mapping.py \
    --harmonizer-output output/fine-tuning/harmonized_inventory.csv \
    --response-options input/score-transformation/questionnaire_response-options.csv \
    --output input/score-transformation

# Step 1: score transformation (needs your own master table, see "Data" above —
# or "Testing the pipeline without real data" to try it with synthetic data first)
python3 code/score-transformation/run_pipeline.py

# Step 2: validation (H1-H3 + downstream checks)
python3 code/score-transformation/validation/hypothesis-testing.py
python3 code/score-transformation/validation/exploratory-validation-analyses.py
```

Every script resolves paths relative to its own location (`code/<...>/script.py`
→ repo root is two or more `.parent`s up), so the pipeline runs unmodified as
long as the repo's directory layout is preserved — no local path editing
needed.

## License

CC BY-NC 4.0  — see [LICENSE](LICENSE).
