# LLM Persona Prediction from Summarized Survey Profiles

This repository contains a public version of the LLM-based persona prediction pipeline used for experiments on survey respondents.

The public release is intentionally summary-only:

- raw independent variables were removed from the published datasets
- each row keeps only the target variables and one `independent_summary` column
- the codebase includes only the LLM workflow

## Repository layout

- `data/train_with_independent_summary.csv`: public training split with 800 rows
- `data/test_with_independent_summary.csv`: public test split with 200 rows
- `simulate_personas.py`: runs one LLM configuration
- `run_experiments.py`: runs the full experiment grid and writes an aggregated CSV
- `pipeline_utils.py`: shared helpers
- `outputs_from_paper.zip`: archived outputs used in the paper

## Dataset schema

Both CSV files contain exactly 8 columns:

1. `Region`
2. `Pol`
3. `Obrazovanje`
4. `Radni status`
5. `Nacionalnost`
6. `Mesto stanovanja`
7. `Starost kategorija`
8. `independent_summary`

The first 7 columns are the target variables to predict. The `independent_summary` column is the only LLM input retained in the public version.

## How the public pipeline works

For each respondent in the test set, the pipeline:

1. uses `independent_summary` as the prompt input
2. finds similar training respondents by token overlap over summary text
3. builds target-value distributions from the top-N most similar training rows
4. optionally includes few-shot examples from those similar rows
5. asks the LLM to predict all target variables in one JSON response

The current public code does not include:

- the traditional baseline pipeline
- correlation-rule prompts
- raw independent-variable matching

## Requirements

- Python 3.10+
- an OpenAI-compatible API key

Install dependencies:

```bash
python -m pip install openai pandas python-dotenv
```

## Environment variables

Create a `.env` file in the repo root with only:

```env
OPENAI_API_KEY=your_api_key_here
```

Notes:

- `OPENAI_API_KEY` is the only environment variable used by the public setup
- model names, reasoning settings, temperatures, and optional base URLs are configured in `run_experiments.py`
- if you run `simulate_personas.py` directly, those same settings should be passed as CLI arguments

## Run the full experiment grid

This runs all model and prompt configurations defined in `run_experiments.py` and writes results to `outputs/`.

```bash
python run_experiments.py
```

Run only the first `N` test rows:

```bash
python run_experiments.py --test-n 20
```

Write the aggregated run summary to a custom CSV path:

```bash
python run_experiments.py --results-csv outputs/my_results.csv
```

## Run one configuration directly

Example:

```bash
python simulate_personas.py \
  --model-name gpt-5.4-mini \
  --train-path data/train_with_independent_summary.csv \
  --test-path data/test_with_independent_summary.csv \
  --target-columns Region Pol Obrazovanje "Radni status" Nacionalnost "Mesto stanovanja" "Starost kategorija" \
  --summary-column independent_summary \
  --similar-top-n 20 \
  --expanded-top-n false \
  --return-reason true \
  --use-contrast-distribution false \
  --test-n all \
  --reasoning-effort high \
  --temperature none
```

On PowerShell, the same command may be easier to run on one line:

```powershell
python simulate_personas.py --model-name gpt-5.4-mini --train-path data/train_with_independent_summary.csv --test-path data/test_with_independent_summary.csv --target-columns Region Pol Obrazovanje "Radni status" Nacionalnost "Mesto stanovanja" "Starost kategorija" --summary-column independent_summary --similar-top-n 20 --expanded-top-n false --return-reason true --use-contrast-distribution false --test-n all --reasoning-effort high --temperature none
```

## Outputs

Each run creates a directory under `outputs/` named after the model and prompt settings. Typical files are:

- `results.csv`: one row per respondent-question prediction
- `accuracy_per_respondent.csv`: mean accuracy per respondent
- `accuracy_per_question.csv`: mean accuracy per target variable
- `summary.json`: metadata for the run, including overall accuracy and average prompt size

When using `run_experiments.py`, an additional aggregated CSV is created in `outputs/`, usually named like:

```text
outputs/experiment_results_YYYYMMDD_HHMMSS.csv
```

## Reproducibility notes

- the scripts use a fixed seed of `42`
- all target variables are predicted in a single LLM call per respondent
- the similarity method recorded in `summary.json` is `token_overlap_jaccard`
- published datasets contain only target columns and the summary column

## Limitations of the public release

This repository is designed for the LLM experiments only. It is not a drop-in reproduction of the original private workspace because the following were removed before publication:

- raw independent-variable columns
- ignore-column metadata
- traditional ML baselines
- correlation-based prompt rules

## Citation / artifact note

If you are using the public release to inspect previously generated results, see `outputs_from_paper.zip`.
