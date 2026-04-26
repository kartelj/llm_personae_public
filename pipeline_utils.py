import argparse
import json
import re

import pandas as pd


TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def normalize_value(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def parse_optional_int_arg(value):
    text = str(value).strip().lower()
    if text in {"none", "all", "full"}:
        return None
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("Vrednost mora biti pozitivan ceo broj ili 'all'.")
    return out


def parse_optional_float_arg(value):
    text = str(value).strip().lower()
    if text in {"none", "auto"}:
        return None
    out = float(value)
    return int(out) if out.is_integer() else out


def parse_optional_text_arg(value):
    text = str(value).strip()
    if text.lower() in {"none", "auto"}:
        return None
    return text


def load_csv_clean(path):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def result_suffix(top_n, total_rows):
    return "" if top_n is None or top_n >= total_rows else f"_first{top_n}"


def config_value_to_tag(value):
    if value is None:
        return "none"
    text = str(value).strip().lower()
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|', ' ']:
        text = text.replace(char, "_")
    return text


def tokenize_text(text):
    normalized = normalize_value(text)
    if normalized is None:
        return []
    return TOKEN_PATTERN.findall(normalized.casefold())


def extract_first_json_object(text):
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except Exception:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(stripped[start : end + 1])
    except Exception:
        return None


def extract_chat_completion_text(response):
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts).strip()
    return "" if content is None else str(content).strip()
