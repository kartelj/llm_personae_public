import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from pipeline_utils import (
    config_value_to_tag,
    extract_chat_completion_text,
    extract_first_json_object,
    load_csv_clean,
    normalize_value,
    parse_optional_float_arg,
    parse_optional_int_arg,
    parse_optional_text_arg,
    result_suffix,
    tokenize_text,
)


SEED = 42
RETURN_REASON = False
MAX_WORKERS_RESPONDENTS = 1
EXPANDED_TOP_N = False
MODEL_BASE_URL = None


def parse_bool_arg(value):
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Vrednost mora biti true ili false.")


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--target-columns", nargs="+", required=True)
    parser.add_argument("--summary-column", required=True)
    parser.add_argument("--similar-top-n", type=int, required=True)
    parser.add_argument("--expanded-top-n", type=parse_bool_arg, default=False)
    parser.add_argument("--return-reason", type=parse_bool_arg, default=False)
    parser.add_argument("--use-contrast-distribution", type=parse_bool_arg, required=True)
    parser.add_argument("--test-n", type=parse_optional_int_arg, required=True)
    parser.add_argument("--reasoning-effort", type=parse_optional_text_arg, required=True)
    parser.add_argument("--temperature", type=parse_optional_float_arg, required=True)
    parser.add_argument("--base-url", default="auto")

    args = parser.parse_args()
    if args.similar_top_n < 0:
        raise argparse.ArgumentTypeError("--similar-top-n mora biti nenegativan ceo broj.")
    return args


def resolve_base_url(cli_value):
    raw = "" if cli_value is None else str(cli_value).strip()
    if not raw or raw.lower() in {"auto", "__use_env__"}:
        env_model_base_url = parse_optional_text_arg(os.getenv("MODEL_BASE_URL") or "none")
        if env_model_base_url is not None:
            return env_model_base_url
        return parse_optional_text_arg(os.getenv("OPENAI_BASE_URL") or "none")
    if raw.lower() in {"none", "null"}:
        return None
    return raw


def unique_non_null_values(values):
    uniq = []
    seen = set()
    for value in values:
        normalized = normalize_value(value)
        if normalized is None:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(normalized)
    return uniq


def build_options(df, question_cols):
    return {col: unique_non_null_values(df[col].tolist()) for col in question_cols}


def format_indexed_option(index, option):
    return f"[{index}] {option}"


def build_input_summary_from_row(row, summary_column):
    if summary_column not in row.index:
        raise ValueError(f"Nedostaje kolona {summary_column}.")

    summary_text = normalize_value(row[summary_column])
    if summary_text is not None:
        return summary_text
    raise ValueError(f"Prazna vrednost u koloni {summary_column}")


def validate_summary_dataset(df, dataset_name, summary_column):
    if summary_column not in df.columns:
        raise ValueError(f"{dataset_name} ne sadrzi kolonu {summary_column}.")

    missing_count = sum(1 for value in df[summary_column].tolist() if normalize_value(value) is None)
    if missing_count > 0:
        raise ValueError(f"{dataset_name} sadrzi {missing_count} redova bez vrednosti u koloni {summary_column}.")


def build_summary_token_sets(df, summary_column):
    token_sets = []
    for row_idx in range(len(df)):
        summary_text = build_input_summary_from_row(df.iloc[row_idx], summary_column)
        token_sets.append(set(tokenize_text(summary_text)))
    return token_sets


def score_summary_similarity(input_tokens, candidate_tokens):
    shared_tokens = len(input_tokens & candidate_tokens)
    union_size = len(input_tokens | candidate_tokens)
    jaccard = shared_tokens / union_size if union_size else 0.0
    return shared_tokens, jaccard


def score_train_rows_by_summary_similarity(train_summary_tokens, input_summary_tokens):
    scored = []

    for train_idx, candidate_tokens in enumerate(train_summary_tokens):
        shared_tokens, jaccard = score_summary_similarity(input_summary_tokens, candidate_tokens)
        scored.append(
            {
                "train_index": train_idx,
                "shared_tokens": shared_tokens,
                "jaccard": jaccard,
            }
        )

    scored.sort(key=lambda item: (-item["shared_tokens"], -item["jaccard"], item["train_index"]))
    return scored


def request_kwargs_for_prompt(prompt):
    request_kwargs = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "seed": SEED,
    }
    if REASONING_EFFORT is not None:
        request_kwargs["reasoning_effort"] = REASONING_EFFORT
    if TEMPERATURE is not None:
        request_kwargs["temperature"] = TEMPERATURE
    return request_kwargs


def find_option_index_for_label(answer_label, options):
    if answer_label is None:
        return None
    raw = str(answer_label).strip()
    if not raw:
        return None
    for idx, option in enumerate(options, start=1):
        if raw == option:
            return idx
    raw_fold = raw.casefold()
    for idx, option in enumerate(options, start=1):
        if raw_fold == option.casefold():
            return idx
    return None


def parse_option_index(value, options):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        option_index = value
    else:
        raw = str(value).strip()
        if not raw.isdigit():
            return None
        option_index = int(raw)
    return option_index if 1 <= option_index <= len(options) else None


def select_top_similar_row_indices(scored_rows):
    if SIMILAR_TOP_N <= 0:
        return []
    return [item["train_index"] for item in scored_rows[:SIMILAR_TOP_N]]


def select_bottom_dissimilar_row_indices(scored_rows, excluded_row_indices=None):
    if SIMILAR_TOP_N <= 0:
        return []

    excluded = set(excluded_row_indices or [])
    candidates = [item for item in scored_rows if item["train_index"] not in excluded]
    candidates.sort(key=lambda item: (item["shared_tokens"], item["jaccard"], item["train_index"]))
    return [item["train_index"] for item in candidates[:SIMILAR_TOP_N]]


def compute_target_distributions(df, row_indices, target_cols, options_map):
    distributions = {}
    index_iterable = range(len(df)) if row_indices is None else row_indices

    for target in target_cols:
        options = options_map.get(target, [])
        option_positions = {option: idx for idx, option in enumerate(options, start=1)}
        counts = {option: 0 for option in options}
        valid_count = 0

        for row_index in index_iterable:
            row = df.iloc[row_index]
            answer_raw = normalize_value(row[target])
            option_index = find_option_index_for_label(answer_raw, options)
            if option_index is None:
                continue
            answer_label = options[option_index - 1]
            counts[answer_label] += 1
            valid_count += 1

        items = []
        if valid_count > 0:
            for option in options:
                count = counts[option]
                if count <= 0:
                    continue
                items.append(
                    {
                        "option_index": option_positions[option],
                        "answer_label": option,
                        "count": count,
                        "share": count / valid_count,
                    }
                )
            items.sort(key=lambda item: (-item["count"], item["option_index"]))

        distributions[target] = {
            "num_valid": valid_count,
            "items": items,
        }

    return distributions


def format_distribution_items(distribution):
    items = distribution.get("items", [])
    if not items:
        return "nema validnih podataka"

    def format_share(value):
        if 0 < value < 0.005:
            return "<0.01"
        return f"{value:.2f}"

    return ", ".join(
        f"{format_indexed_option(item['option_index'], item['answer_label'])} {format_share(item['share'])}"
        for item in items
    )


def format_target_answer_for_few_shot(row, target_col, options):
    if not options:
        return "nema definisanih opcija"

    answer_raw = normalize_value(row[target_col])
    option_index = find_option_index_for_label(answer_raw, options)
    if option_index is None:
        return "nema validnog odgovora"
    return format_indexed_option(option_index, options[option_index - 1])


def make_expanded_top_n_block(train_df, row_indices, summary_column, target_cols, options_map):
    if not row_indices:
        return ""

    examples = []
    for example_no, row_idx in enumerate(row_indices, start=1):
        row = train_df.iloc[row_idx]
        summary_text = build_input_summary_from_row(row, summary_column)
        answers_text = "\n".join(
            f"- {target}: {format_target_answer_for_few_shot(row, target, options_map.get(target, []))}"
            for target in target_cols
        )
        examples.append(
            f"Primer {example_no}:\n"
            f"Sazetak: {summary_text}\n"
            "Ciljne promenljive:\n"
            f"{answers_text}"
        )

    return (
        f"Few-shot primeri medju {len(row_indices)} najslicnijih ispitanika "
        "(summary + tacni odgovori na ciljne promenljive):\n\n"
        + "\n\n".join(examples)
    )


def make_distribution_block(
    local_distributions,
    global_distributions,
    contrast_distributions,
    target_cols,
    local_count,
    contrast_count,
    expanded_top_n,
    expanded_top_n_block,
):
    sections = []

    if local_count > 0:
        if expanded_top_n and expanded_top_n_block:
            sections.append(expanded_top_n_block)
        else:
            local_lines = [
                f"- {target} (n={local_distributions[target]['num_valid']}): {format_distribution_items(local_distributions[target])}"
                for target in target_cols
            ]
            sections.append(
                f"Raspodele medju {local_count} najslicnijih ispitanika:\n" + "\n".join(local_lines)
            )

    global_lines = [
        f"- {target} (n={global_distributions[target]['num_valid']}): {format_distribution_items(global_distributions[target])}"
        for target in target_cols
    ]
    sections.append("Bazne raspodele u celom train skupu:\n" + "\n".join(global_lines))

    if contrast_count > 0:
        contrast_lines = [
            f"- {target} (n={contrast_distributions[target]['num_valid']}): {format_distribution_items(contrast_distributions[target])}"
            for target in target_cols
        ]
        sections.append(
            f"Kontrastne raspodele medju {contrast_count} najmanje slicnih ispitanika:\n"
            + "\n".join(contrast_lines)
        )

    return "\n\n".join(sections)


def make_batch_prompt(
    input_summary,
    target_cols,
    options_map,
    local_distributions,
    global_distributions,
    contrast_distributions,
    local_count,
    contrast_count,
    expanded_top_n,
    expanded_top_n_block,
    return_reason,
):
    distribution_text = make_distribution_block(
        local_distributions,
        global_distributions,
        contrast_distributions,
        target_cols,
        local_count,
        contrast_count,
        expanded_top_n=expanded_top_n,
        expanded_top_n_block=expanded_top_n_block,
    )

    targets_text = "\n\n".join(
        f"{idx}. {question}\n"
        f"   Ponudjene opcije:\n"
        f"{chr(10).join(f'  {format_indexed_option(option_idx, opt)}' for option_idx, opt in enumerate(options_map.get(question, []), start=1))}"
        for idx, question in enumerate(target_cols, start=1)
    )

    if return_reason:
        json_shape = (
            '{"answers":[{"question":"<ime pitanja>","option_index":<redni broj opcije>,'
            '"reason":"<kratko objasnjenje>"}]}'
        )
        if expanded_top_n and local_count > 0:
            reason_rule = (
                'Polje "reason" neka bude jedna kratka recenica zasnovana na ulaznom sazetku, '
                "few-shot primerima i raspodelama."
            )
        else:
            reason_rule = (
                'Polje "reason" neka bude jedna kratka recenica zasnovana na ulaznom sazetku i raspodelama.'
            )
    else:
        json_shape = '{"answers":[{"question":"<ime pitanja>","option_index":<redni broj opcije>}]}'
        reason_rule = None

    if expanded_top_n and local_count > 0:
        rule_lines = [
            "- Few-shot primeri medju najslicnijima su glavni signal kada su dostupni.",
            "- Bazne raspodele u celom train skupu tretiraj kao prosecni prior.",
        ]
    else:
        rule_lines = [
            "- Koristi raspodele medju slicnim ispitanicima kao glavni signal kada su dostupne.",
            "- Bazne raspodele u celom train skupu tretiraj kao prosecni prior.",
        ]

    if contrast_count > 0:
        rule_lines.extend(
            [
                "- Raspodele medju najmanje slicnim ispitanicima koristi samo kao kontrastni signal, ne kao glavni izvor odluke.",
                "- Opcija je posebno uverljiva ako je cesca medju slicnima nego u celom train skupu i redja medju najmanje slicnima.",
                "- Ako je opcija jaka medju najmanje slicnima, a slaba medju slicnima, tretiraj to kao negativan signal.",
            ]
        )

    if expanded_top_n and local_count > 0:
        rule_lines.append(
            "- Ako je signal iz few-shot primera neodlucan ili su opcije bliske, koristi ulazni sazetak ispitanika i baznu raspodelu kao dopunski signal."
        )
    else:
        rule_lines.append(
            "- Ako je lokalna raspodela neodlucna ili su opcije bliske, koristi ulazni sazetak ispitanika i baznu raspodelu kao dopunski signal."
        )

    rule_lines.append(
        "- Ako nisi siguran, izaberi najverovatniju opciju na osnovu kombinacije sazetka i raspodela."
    )
    if return_reason:
        rule_lines.append(f"- {reason_rule}")
    else:
        rule_lines.append('- Ne vracaj polje "reason".')
    rule_lines.extend(
        [
            f'- Vrati odgovor ISKLJUCIVO kao JSON objekat oblika: {json_shape}',
            '- Za SVAKO zadato pitanje vrati tacno jedan unos u nizu "answers".',
            '- U polju "question" koristi tacno naziv pitanja, na primer "Region", bez prefiksa "Pitanje:" i bez rednog broja.',
            '- U polju "option_index" vrati TACNO redni broj jedne ponudjene opcije za to pitanje.',
            '- Ne vracaj tekst opcije u JSON-u, samo redni broj kroz "option_index".',
            "- Ne izmisljaj nova pitanja ni nove opcije.",
            "- Bez markdown-a, bez dodatnog teksta van JSON-a.",
        ]
    )
    rules_text = "\n".join(rule_lines)

    return f"""Na osnovu sazetka jednog ispitanika izvedenog iz njegovih odgovora na anketna pitanja,
predvidi odgovore za vise ciljnih promenljivih odjednom.

Ulazni sazetak ispitanika:
{input_summary}

{distribution_text}

Ciljne promenljive i njihove opcije:
{targets_text}

Pravila:
{rules_text}
"""


def normalize_question_key(text):
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\ufeff", "")
    s = s.replace("\u200b", "")
    s = s.replace("\u200c", "")
    s = s.replace("\u200d", "")
    s = " ".join(s.split())
    return s.casefold().strip()


def compute_result_tables(results_df):
    valid_cmp = results_df.dropna(subset=["is_match"])
    overall_accuracy = valid_cmp["is_match"].mean() if not valid_cmp.empty else None
    per_respondent = (
        valid_cmp.groupby("respondent_index", as_index=False)["is_match"]
        .mean()
        .rename(columns={"is_match": "accuracy"})
    )
    per_question = (
        valid_cmp.groupby("question", as_index=False)["is_match"]
        .mean()
        .rename(columns={"is_match": "accuracy"})
    )
    return valid_cmp, overall_accuracy, per_respondent, per_question


def build_summary(
    results_df,
    valid_cmp,
    overall_accuracy,
    avg_respondent_duration_seconds,
    avg_prompt_chars,
    target_cols,
    summary_column,
    train_path,
    test_path,
    output_dir,
    base_url,
    expanded_top_n,
    return_reason,
):
    contrast_enabled = USE_CONTRAST_DISTRIBUTION and SIMILAR_TOP_N > 0
    prompt_input_mode = (
        "summary_with_expanded_topn_fewshots_and_distributions"
        if expanded_top_n
        else "summary_with_distributions"
    )
    return {
        "num_respondents": int(results_df["respondent_index"].nunique()) if not results_df.empty else 0,
        "num_questions": int(len(target_cols)),
        "num_predictions": int(len(results_df)),
        "num_comparable": int(len(valid_cmp)),
        "overall_accuracy": None if overall_accuracy is None else float(overall_accuracy),
        "target_columns": list(target_cols),
        "model_name": MODEL_NAME,
        "base_url": base_url,
        "reasoning_effort": REASONING_EFFORT,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "ask_all_targets_in_one_call": True,
        "prompt_input_mode": prompt_input_mode,
        "prompt_summary_column": summary_column,
        "summary_similarity_method": "token_overlap_jaccard",
        "local_distribution_enabled": SIMILAR_TOP_N > 0,
        "similar_top_n": SIMILAR_TOP_N,
        "expanded_top_n": bool(expanded_top_n),
        "return_reason": bool(return_reason),
        "contrast_distribution_enabled": contrast_enabled,
        "contrast_bottom_n": SIMILAR_TOP_N if contrast_enabled else None,
        "distribution_reference_scope": "full_train",
        "contrast_distribution_scope": "bottom_n_dissimilar" if contrast_enabled else None,
        "max_workers_respondents": MAX_WORKERS_RESPONDENTS,
        "requested_test_n": TEST_N,
        "avg_respondent_duration_seconds": (
            None if avg_respondent_duration_seconds is None else float(avg_respondent_duration_seconds)
        ),
        "avg_prompt_chars": None if avg_prompt_chars is None else float(avg_prompt_chars),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "num_target_columns": int(len(target_cols)),
        "output_dir": str(output_dir),
    }


def ask_all_questions_once(
    client,
    input_summary,
    target_cols,
    options_map,
    respondent_idx,
    local_distributions,
    global_distributions,
    contrast_distributions,
    local_count,
    contrast_count,
    expanded_top_n,
    expanded_top_n_block,
    return_reason,
    row,
):
    prompt = make_batch_prompt(
        input_summary,
        target_cols,
        options_map,
        local_distributions=local_distributions,
        global_distributions=global_distributions,
        contrast_distributions=contrast_distributions,
        local_count=local_count,
        contrast_count=contrast_count,
        expanded_top_n=expanded_top_n,
        expanded_top_n_block=expanded_top_n_block,
        return_reason=return_reason,
    )
    prompt_char_count = len(prompt)
    question_lookup = {normalize_question_key(q): q for q in target_cols}
    answers_by_question = {}
    llm_raw = None
    api_error = None
    request_started = time.perf_counter()

    try:
        request_kwargs = request_kwargs_for_prompt(prompt)
        resp = client.chat.completions.create(**request_kwargs)
        request_duration_seconds = time.perf_counter() - request_started
        llm_raw = extract_chat_completion_text(resp)
        parsed = extract_first_json_object(llm_raw)

        if isinstance(parsed, dict):
            raw_answers = parsed.get("answers")
            if isinstance(raw_answers, list):
                for item in raw_answers:
                    if not isinstance(item, dict):
                        continue
                    q_raw = str(item.get("question", "")).strip()
                    if not q_raw:
                        continue
                    q_key = question_lookup.get(normalize_question_key(q_raw))
                    if q_key is None:
                        continue
                    options = options_map.get(q_key, [])
                    parsed_option_index = parse_option_index(item.get("option_index"), options)
                    parsed_answer = options[parsed_option_index - 1] if parsed_option_index is not None else None
                    parsed_reason = None
                    if return_reason:
                        reason_value = item.get("reason")
                        if reason_value is not None:
                            parsed_reason = str(reason_value).strip()
                    answers_by_question[q_key] = (parsed_answer, parsed_reason)
        else:
            llm_raw = llm_raw or ""
    except Exception as exc:
        print(exc)
        request_duration_seconds = time.perf_counter() - request_started
        api_error = f"API_ERROR: {exc}"

    out = []
    for question in target_cols:
        real_answer = normalize_value(row[question])
        if api_error is not None:
            llm_answer = None
            llm_reason = None
            llm_raw_for_question = api_error
        else:
            answer_tuple = answers_by_question.get(question)
            if answer_tuple is None:
                llm_answer = None
                llm_reason = None
            else:
                llm_answer, llm_reason = answer_tuple
            llm_raw_for_question = llm_raw

        is_match = None
        if real_answer is not None and llm_answer is not None:
            is_match = int(real_answer.casefold() == llm_answer.casefold())

        out.append(
            {
                "respondent_index": respondent_idx,
                "question": question,
                "real_answer": real_answer,
                "llm_answer": llm_answer,
                "llm_reason": llm_reason,
                "llm_raw": llm_raw_for_question,
                "is_match": is_match,
            }
        )

    return out, request_duration_seconds, prompt_char_count


def process_one_respondent(
    api_key,
    base_url,
    row,
    respondent_idx,
    summary_column,
    expanded_top_n,
    return_reason,
    use_contrast_distribution,
    target_cols,
    options_map,
    train_df,
    train_summary_tokens,
    global_distributions,
):
    input_summary = build_input_summary_from_row(row, summary_column)
    input_summary_tokens = set(tokenize_text(input_summary))
    batch_target_cols = [q for q in target_cols if options_map.get(q, [])]

    scored_rows = score_train_rows_by_summary_similarity(
        train_summary_tokens=train_summary_tokens,
        input_summary_tokens=input_summary_tokens,
    )
    local_row_indices = select_top_similar_row_indices(scored_rows)
    contrast_row_indices = (
        select_bottom_dissimilar_row_indices(scored_rows, excluded_row_indices=local_row_indices)
        if use_contrast_distribution
        else []
    )

    local_distributions = compute_target_distributions(
        df=train_df,
        row_indices=local_row_indices,
        target_cols=batch_target_cols,
        options_map=options_map,
    )
    expanded_top_n_block = (
        make_expanded_top_n_block(
            train_df=train_df,
            row_indices=local_row_indices,
            summary_column=summary_column,
            target_cols=batch_target_cols,
            options_map=options_map,
        )
        if expanded_top_n
        else ""
    )
    contrast_distributions = compute_target_distributions(
        df=train_df,
        row_indices=contrast_row_indices,
        target_cols=batch_target_cols,
        options_map=options_map,
    )

    client_kwargs = {"api_key": api_key}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    batch_results, request_duration_seconds, prompt_char_count = ask_all_questions_once(
        client=client,
        input_summary=input_summary,
        target_cols=batch_target_cols,
        options_map=options_map,
        respondent_idx=respondent_idx,
        local_distributions=local_distributions,
        global_distributions=global_distributions,
        contrast_distributions=contrast_distributions,
        local_count=len(local_row_indices),
        contrast_count=len(contrast_row_indices),
        expanded_top_n=expanded_top_n,
        expanded_top_n_block=expanded_top_n_block,
        return_reason=return_reason,
        row=row,
    )
    return {
        "results": batch_results,
        "request_duration_seconds": request_duration_seconds,
        "prompt_char_count": prompt_char_count,
    }


def main():
    global MODEL_NAME, REASONING_EFFORT, TEMPERATURE, SIMILAR_TOP_N, TEST_N, USE_CONTRAST_DISTRIBUTION, EXPANDED_TOP_N, RETURN_REASON, MODEL_BASE_URL

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_cli_args()
    MODEL_NAME = args.model_name
    REASONING_EFFORT = args.reasoning_effort
    TEMPERATURE = args.temperature
    SIMILAR_TOP_N = args.similar_top_n
    EXPANDED_TOP_N = args.expanded_top_n
    RETURN_REASON = args.return_reason
    USE_CONTRAST_DISTRIBUTION = args.use_contrast_distribution
    TEST_N = args.test_n
    target_cols = list(args.target_columns)
    summary_column = args.summary_column

    if summary_column in target_cols:
        raise ValueError("Summary kolona ne sme da bude medju target kolonama.")

    load_dotenv()
    os.environ.pop("SSLKEYLOGFILE", None)

    MODEL_BASE_URL = resolve_base_url(args.base_url)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MODEL_API_KEY")
    if not api_key and MODEL_BASE_URL is not None:
        api_key = "not-needed"
    if not api_key:
        raise ValueError("OPENAI_API_KEY nije pronadjen u .env fajlu (ili MODEL_API_KEY).")

    train_path = Path(args.train_path)
    input_path = Path(args.test_path)
    run_dir_name = (
        f"model-{config_value_to_tag(MODEL_NAME)}_"
        f"temp-{config_value_to_tag(TEMPERATURE)}_"
        f"reasoning-{config_value_to_tag(REASONING_EFFORT)}_"
        f"base-url-{config_value_to_tag(MODEL_BASE_URL)}_"
        f"batch-targets-true_distribution-topn-{SIMILAR_TOP_N}_"
        f"expanded-topn-{config_value_to_tag(EXPANDED_TOP_N)}_"
        f"return-reason-{config_value_to_tag(RETURN_REASON)}_"
        f"contrast-{config_value_to_tag(USE_CONTRAST_DISTRIBUTION)}"
    )
    output_dir = Path("outputs") / run_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv_clean(input_path)
    train_df = load_csv_clean(train_path)
    validate_summary_dataset(train_df, "train dataset", summary_column)
    validate_summary_dataset(df, "test dataset", summary_column)

    expected_columns = target_cols + [summary_column]
    if list(train_df.columns) != expected_columns:
        raise ValueError(
            "Train dataset mora da sadrzi samo target kolone i summary kolonu "
            f"u ovom redosledu: {expected_columns}"
        )
    if list(df.columns) != expected_columns:
        raise ValueError(
            "Test dataset mora da sadrzi samo target kolone i summary kolonu "
            f"u ovom redosledu: {expected_columns}"
        )

    options_map = build_options(train_df, target_cols)
    global_distributions = compute_target_distributions(
        df=train_df,
        row_indices=None,
        target_cols=target_cols,
        options_map=options_map,
    )
    train_summary_tokens = build_summary_token_sets(train_df, summary_column)

    print(f"Train ulaz: {train_path}")
    print(f"Test ulaz: {input_path}")
    print(f"Prompt koristi summary kolonu: {summary_column}")
    print("Slicnost izmedju ispitanika: token overlap nad summary kolonom")
    print(f"Model base URL: {MODEL_BASE_URL}")
    print(f"Lokalne raspodele koriste top N slicnih: {SIMILAR_TOP_N}")
    print(f"Expanded few-shot prikaz top N slicnih: {EXPANDED_TOP_N}")
    print(f"Vraca reason u JSON-u: {RETURN_REASON}")
    if USE_CONTRAST_DISTRIBUTION:
        print(f"Kontrastne raspodele koriste bottom N najmanje slicnih: {SIMILAR_TOP_N}")
    else:
        print("Kontrastne raspodele: iskljucene.")

    results = []
    respondent_durations = []
    prompt_char_counts = []
    total_respondents = len(df) if TEST_N is None else min(TEST_N, len(df))
    futures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_RESPONDENTS) as executor:
        for respondent_idx in range(total_respondents):
            row = df.iloc[respondent_idx]
            futures.append(
                executor.submit(
                    process_one_respondent,
                    api_key,
                    MODEL_BASE_URL,
                    row,
                    respondent_idx,
                    summary_column,
                    EXPANDED_TOP_N,
                    RETURN_REASON,
                    USE_CONTRAST_DISTRIBUTION,
                    target_cols,
                    options_map,
                    train_df,
                    train_summary_tokens,
                    global_distributions,
                )
            )

        for future in as_completed(futures):
            respondent_output = future.result()
            results.extend(respondent_output["results"])
            respondent_durations.append(respondent_output["request_duration_seconds"])
            prompt_char_counts.append(respondent_output["prompt_char_count"])
            print(".", end="", flush=True)

    print()

    results_df = pd.DataFrame(results)
    valid_cmp, overall_accuracy, per_respondent, per_question = compute_result_tables(results_df)
    avg_respondent_duration_seconds = (
        sum(respondent_durations) / len(respondent_durations) if respondent_durations else None
    )
    avg_prompt_chars = sum(prompt_char_counts) / len(prompt_char_counts) if prompt_char_counts else None

    summary = build_summary(
        results_df,
        valid_cmp,
        overall_accuracy,
        avg_respondent_duration_seconds,
        avg_prompt_chars,
        target_cols,
        summary_column,
        train_path,
        input_path,
        output_dir,
        base_url=MODEL_BASE_URL,
        expanded_top_n=EXPANDED_TOP_N,
        return_reason=RETURN_REASON,
    )

    suffix = result_suffix(TEST_N, len(df))

    results_path = output_dir / f"results{suffix}.csv"
    per_resp_path = output_dir / f"accuracy_per_respondent{suffix}.csv"
    per_question_path = output_dir / f"accuracy_per_question{suffix}.csv"
    summary_path = output_dir / f"summary{suffix}.json"

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    per_respondent.to_csv(per_resp_path, index=False, encoding="utf-8-sig")
    per_question.to_csv(per_question_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Ukupna tacnost: {summary['overall_accuracy']}")
    print(f"Prosecno vreme po ispitaniku (LLM poziv): {summary['avg_respondent_duration_seconds']}")
    print(f"Prosecna velicina prompta (karakteri): {summary['avg_prompt_chars']}")

    print("Tacnost po svakom pitanju:")
    for _, row in per_question.iterrows():
        print(f"- {row['question']}: {row['accuracy']}")


if __name__ == "__main__":
    main()
