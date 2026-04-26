import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd

from pipeline_utils import (
    config_value_to_tag,
    load_csv_clean,
    parse_optional_int_arg,
    result_suffix,
)


TARGET_COLUMNS = [
    "Region",
    "Pol",
    "Obrazovanje",
    "Radni status",
    "Nacionalnost",
    "Mesto stanovanja",
    "Starost kategorija",
]
SUMMARY_COLUMN = "independent_summary"
DEFAULT_DATA_DIR = Path("data")
DEFAULT_PERSONA_TRAIN_PATH = DEFAULT_DATA_DIR / "train_with_independent_summary.csv"
DEFAULT_PERSONA_TEST_PATH = DEFAULT_DATA_DIR / "test_with_independent_summary.csv"

SIMILAR_TOP_N_VALUES = [0, 10, 20, 25, 30]
EXPANDED_TOP_N_VALUES = [False]
RETURN_REASON_VALUES = [True, False]
USE_CONTRAST_DISTRIBUTION_VALUES = [False]
DEFAULT_TEST_N = None


LLM_REQUEST_CONFIGS = {
    "gpt-4.1-mini": {"reasoning_effort": None, "temperature": 0},
    "gpt-4o-mini": {"reasoning_effort": None, "temperature": 0},
    "gpt-5-mini": {"reasoning_effort": "high", "temperature": None},
    "gpt-5.4-mini": {"reasoning_effort": "high", "temperature": None},
}


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-n", type=parse_optional_int_arg, default=DEFAULT_TEST_N)
    parser.add_argument("--results-csv", default=None)
    return parser.parse_args()


def default_results_csv(repo_root):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return repo_root / "outputs" / f"experiment_results_{stamp}.csv"


def output_paths_for_personas(
    repo_root,
    model_name,
    model_base_url,
    similar_top_n,
    expanded_top_n,
    return_reason,
    use_contrast_distribution,
    reasoning_effort,
    temperature,
    test_n,
    total_test_rows,
):
    run_dir_name = (
        f"model-{config_value_to_tag(model_name)}_"
        f"temp-{config_value_to_tag(temperature)}_"
        f"reasoning-{config_value_to_tag(reasoning_effort)}_"
        f"base-url-{config_value_to_tag(model_base_url)}_"
        f"batch-targets-true_distribution-topn-{similar_top_n}_"
        f"expanded-topn-{config_value_to_tag(expanded_top_n)}_"
        f"return-reason-{config_value_to_tag(return_reason)}_"
        f"contrast-{config_value_to_tag(use_contrast_distribution)}"
    )
    output_dir = repo_root / "outputs" / run_dir_name
    suffix = result_suffix(test_n, total_test_rows)
    return {
        "summary_path": output_dir / f"summary{suffix}.json",
        "per_question_path": output_dir / f"accuracy_per_question{suffix}.csv",
        "results_path": output_dir / f"results{suffix}.csv",
    }


def build_persona_runs(repo_root, test_n, total_test_rows, persona_train_path, persona_test_path):
    script_path = repo_root / "simulate_personas.py"
    runs = []

    for model_name, similar_top_n, expanded_top_n, return_reason, use_contrast_distribution in product(
        LLM_REQUEST_CONFIGS,
        SIMILAR_TOP_N_VALUES,
        EXPANDED_TOP_N_VALUES,
        RETURN_REASON_VALUES,
        USE_CONTRAST_DISTRIBUTION_VALUES,
    ):
        request_config = LLM_REQUEST_CONFIGS[model_name]
        model_base_url = request_config.get("base_url")

        runs.append(
            {
                "name": (
                    f"{model_name}_topn-{similar_top_n}_"
                    f"expanded-topn-{config_value_to_tag(expanded_top_n)}_"
                    f"return-reason-{config_value_to_tag(return_reason)}_"
                    f"contrast-{config_value_to_tag(use_contrast_distribution)}"
                ),
                "model_name": model_name,
                "model_base_url": model_base_url,
                "reasoning_effort": request_config["reasoning_effort"],
                "temperature": request_config["temperature"],
                "similar_top_n": similar_top_n,
                "expanded_top_n": expanded_top_n,
                "return_reason": return_reason,
                "use_contrast_distribution": use_contrast_distribution,
                "test_n": test_n,
                "command": [
                    sys.executable,
                    str(script_path),
                    "--model-name",
                    model_name,
                    "--base-url",
                    "none" if model_base_url is None else str(model_base_url),
                    "--train-path",
                    str(persona_train_path),
                    "--test-path",
                    str(persona_test_path),
                    "--target-columns",
                    *TARGET_COLUMNS,
                    "--summary-column",
                    SUMMARY_COLUMN,
                    "--similar-top-n",
                    str(similar_top_n),
                    "--expanded-top-n",
                    str(expanded_top_n).lower(),
                    "--return-reason",
                    str(return_reason).lower(),
                    "--use-contrast-distribution",
                    str(use_contrast_distribution).lower(),
                    "--test-n",
                    "all" if test_n is None else str(test_n),
                    "--reasoning-effort",
                    "none" if request_config["reasoning_effort"] is None else request_config["reasoning_effort"],
                    "--temperature",
                    "none" if request_config["temperature"] is None else str(request_config["temperature"]),
                ],
                **output_paths_for_personas(
                    repo_root,
                    model_name,
                    model_base_url,
                    similar_top_n,
                    expanded_top_n,
                    return_reason,
                    use_contrast_distribution,
                    request_config["reasoning_effort"],
                    request_config["temperature"],
                    test_n,
                    total_test_rows,
                ),
            }
        )

    return runs


def build_run_plan(repo_root, test_n, total_test_rows, persona_train_path, persona_test_path):
    return build_persona_runs(
        repo_root,
        test_n,
        total_test_rows,
        persona_train_path,
        persona_test_path,
    )


def question_accuracy_columns(target_questions):
    return [f"accuracy__{question}" for question in target_questions]


def csv_fieldnames(target_questions):
    return [
        "run_index",
        "started_at",
        "finished_at",
        "status",
        "name",
        "model_name",
        "model_base_url",
        "reasoning_effort",
        "temperature",
        "similar_top_n",
        "expanded_top_n",
        "return_reason",
        "use_contrast_distribution",
        "test_n",
        "duration_seconds",
        "wall_clock_duration_seconds",
        "avg_prompt_chars",
        "overall_accuracy",
        "num_comparable",
        "summary_path",
        "per_question_path",
        "results_path",
        "error",
        *question_accuracy_columns(target_questions),
    ]


def append_result_row(csv_path, row, target_questions):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = csv_fieldnames(target_questions)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def load_metrics(summary_path, per_question_path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_question_df = pd.read_csv(per_question_path) if per_question_path.exists() else pd.DataFrame()
    accuracies = {}
    if not per_question_df.empty:
        for _, row in per_question_df.iterrows():
            accuracies[str(row["question"])] = float(row["accuracy"])
    return summary, accuracies


def build_result_row(run_index, run, started_at, finished_at, duration_seconds, status, error):
    return {
        "run_index": run_index,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "name": run["name"],
        "model_name": run["model_name"],
        "model_base_url": run.get("model_base_url"),
        "reasoning_effort": run["reasoning_effort"],
        "temperature": run["temperature"],
        "similar_top_n": run["similar_top_n"],
        "expanded_top_n": run.get("expanded_top_n"),
        "return_reason": run.get("return_reason"),
        "use_contrast_distribution": run.get("use_contrast_distribution"),
        "test_n": run["test_n"],
        "duration_seconds": None,
        "wall_clock_duration_seconds": round(duration_seconds, 3),
        "avg_prompt_chars": None,
        "overall_accuracy": None,
        "num_comparable": None,
        "summary_path": str(run["summary_path"]),
        "per_question_path": str(run["per_question_path"]),
        "results_path": str(run["results_path"]),
        "error": error,
    }


def build_success_row(run_index, run, started_at, finished_at, duration_seconds, summary, accuracies, target_questions):
    row = build_result_row(run_index, run, started_at, finished_at, duration_seconds, "ok", "")
    avg_respondent_duration_seconds = summary.get("avg_respondent_duration_seconds")
    if avg_respondent_duration_seconds is None:
        num_respondents = summary.get("num_respondents", summary.get("num_test_rows"))
        if num_respondents:
            avg_respondent_duration_seconds = duration_seconds / num_respondents

    row["duration_seconds"] = None if avg_respondent_duration_seconds is None else round(avg_respondent_duration_seconds, 3)
    row["avg_prompt_chars"] = summary.get("avg_prompt_chars")
    row["overall_accuracy"] = summary.get("overall_accuracy")
    row["num_comparable"] = summary.get("num_comparable")
    for question in target_questions:
        row[f"accuracy__{question}"] = accuracies.get(question)
    return row


def build_failure_row(run_index, run, started_at, finished_at, duration_seconds, error, target_questions):
    row = build_result_row(run_index, run, started_at, finished_at, duration_seconds, "failed", error)
    for question in target_questions:
        row[f"accuracy__{question}"] = None
    return row


def main():
    args = parse_cli_args()
    repo_root = Path(__file__).resolve().parent
    persona_train_path = repo_root / DEFAULT_PERSONA_TRAIN_PATH
    persona_test_path = repo_root / DEFAULT_PERSONA_TEST_PATH

    test_df = load_csv_clean(persona_test_path)
    expected_columns = TARGET_COLUMNS + [SUMMARY_COLUMN]
    if list(test_df.columns) != expected_columns:
        raise ValueError(
            "Test dataset mora da sadrzi samo target kolone i summary kolonu "
            f"u ovom redosledu: {expected_columns}"
        )

    target_questions = list(TARGET_COLUMNS)
    total_test_rows = len(test_df)
    results_csv = Path(args.results_csv) if args.results_csv else default_results_csv(repo_root)

    runs = build_run_plan(
        repo_root,
        args.test_n,
        total_test_rows,
        persona_train_path,
        persona_test_path,
    )

    print(f"Zbirni CSV: {results_csv}")
    print(f"Persona train summary: {persona_train_path}")
    print(f"Persona test summary: {persona_test_path}")

    for run_index, run in enumerate(runs, start=1):
        started_at = datetime.now().isoformat(timespec="seconds")
        started = time.perf_counter()
        print(f"[{run_index}/{len(runs)}] {run['name']}")

        try:
            subprocess.run(run["command"], check=True, cwd=repo_root)
            if not run["summary_path"].exists():
                raise FileNotFoundError(f"Nedostaje summary fajl: {run['summary_path']}")
            if not run["per_question_path"].exists():
                raise FileNotFoundError(f"Nedostaje per-question fajl: {run['per_question_path']}")

            finished_at = datetime.now().isoformat(timespec="seconds")
            duration_seconds = time.perf_counter() - started
            summary, accuracies = load_metrics(run["summary_path"], run["per_question_path"])
            row = build_success_row(
                run_index,
                run,
                started_at,
                finished_at,
                duration_seconds,
                summary,
                accuracies,
                target_questions,
            )
            append_result_row(results_csv, row, target_questions)
            print(
                f"  -> upisano u zbirni CSV | overall_accuracy={summary.get('overall_accuracy')} "
                f"| avg/respondent={row['duration_seconds']}s | wall={row['wall_clock_duration_seconds']}s"
            )
        except Exception as exc:
            finished_at = datetime.now().isoformat(timespec="seconds")
            duration_seconds = time.perf_counter() - started
            row = build_failure_row(
                run_index,
                run,
                started_at,
                finished_at,
                duration_seconds,
                str(exc),
                target_questions,
            )
            append_result_row(results_csv, row, target_questions)
            print(f"  -> run neuspesan, upisan failure red | greska={exc}")


if __name__ == "__main__":
    main()
