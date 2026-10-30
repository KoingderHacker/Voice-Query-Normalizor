"""Controlled benchmark, reproducible CSV reports, and stemming error candidates."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from .lemmatizer_comparison import compare_lemmatizers
from .pipeline import benchmark_tokenizers
from .preprocessing import clean_transcript
from .stemmer_comparison import compare_stemmers
from .utils import write_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "benchmark.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

# Manually curated contrastive pairs. Relatedness labels are human-curated;
# whether a method actually conflates/splits a pair is measured at run time.
CURATED_PAIRS = (
    {"word_a": "university", "word_b": "universal", "relation": "unrelated", "basis": "manual"},
    {"word_a": "goose", "word_b": "geese", "relation": "related_irregular_forms", "basis": "manual"},
    {"word_a": "mouse", "word_b": "mice", "relation": "related_irregular_forms", "basis": "manual"},
)


def load_dataset(path: str | Path = DATASET_PATH) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows or any(not row.get("transcript", "").strip() for row in rows):
        raise ValueError(f"Evaluation dataset is empty or contains a blank transcript: {path}")
    return rows


def analyze_stemming_errors(rows: list[dict], stems: dict[str, dict]) -> list[dict]:
    records: list[dict] = []
    observed_words = {token for row in rows for token in row["cleaning"].tokens}
    for method, result in stems.items():
        stemmer = result["stemmer"]
        stem_groups: dict[str, set[str]] = defaultdict(set)
        for word in observed_words:
            stem_groups[stemmer.stem(word)].add(word)
        for stem, words in stem_groups.items():
            if len(words) < 2:
                continue
            for word_a, word_b in _pairs(words):
                pair = _curated_pair(word_a, word_b)
                records.append({
                    "method": method,
                    "error_type": "over-stemming_candidate",
                    "word_a": word_a,
                    "word_b": word_b,
                    "shared_stem": stem,
                    "basis": pair["basis"] if pair else "automatic_collision_candidate",
                    "manual_relation": pair["relation"] if pair else "unreviewed",
                    "manually_verified_error": bool(pair and pair["relation"] == "unrelated"),
                })
        for pair in CURATED_PAIRS:
            if pair["relation"] != "related_irregular_forms":
                continue
            if pair["word_a"] not in observed_words or pair["word_b"] not in observed_words:
                continue
            stem_a, stem_b = stemmer.stem(pair["word_a"]), stemmer.stem(pair["word_b"])
            if stem_a != stem_b:
                records.append({
                    "method": method,
                    "error_type": "under-stemming_candidate",
                    "word_a": pair["word_a"],
                    "word_b": pair["word_b"],
                    "shared_stem": "",
                    "basis": pair["basis"],
                    "manual_relation": pair["relation"],
                    "manually_verified_error": False,
                    "distinct_stems": f"{stem_a}|{stem_b}",
                })
    return records


def _pairs(words: set[str]):
    ordered = sorted(words)
    for index, word_a in enumerate(ordered):
        for word_b in ordered[index + 1:]:
            yield word_a, word_b


def _curated_pair(word_a: str, word_b: str):
    return next(
        (pair for pair in CURATED_PAIRS if {pair["word_a"], pair["word_b"]} == {word_a, word_b}),
        None,
    )


def run_evaluation(dataset_path: str | Path = DATASET_PATH, output_dir: str | Path = REPORT_DIR) -> dict:
    dataset = load_dataset(dataset_path)
    output_dir = Path(output_dir)
    records: list[dict] = []
    all_clean_tokens: list[str] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_fillers = 0
    total_stop_words = 0
    total_noise = 0
    total_protected = 0
    total_time = 0.0
    regex_tokenizer_seconds = 0.0
    nltk_tokenizer_seconds = 0.0
    tokenizer_equivalent = True

    for row in dataset:
        result = clean_transcript(row["transcript"])
        tokenizer = benchmark_tokenizers(result.normalized_text)
        input_count = len(result.tokens_before_cleaning)
        output_count = len(result.tokens)
        total_input_tokens += input_count
        total_output_tokens += output_count
        total_fillers += len(result.fillers_removed)
        total_stop_words += len(result.stop_words_removed)
        total_noise += len(result.noise_tokens_removed)
        total_protected += len(result.protected_keywords_preserved)
        total_time += result.timings["total_preprocessing"]
        all_clean_tokens.extend(result.tokens)
        regex_tokenizer_seconds += tokenizer["regex_execution_time_seconds"]
        nltk_tokenizer_seconds += tokenizer["nltk_execution_time_seconds"]
        tokenizer_equivalent &= tokenizer["equivalent"]
        records.append({
            "id": row["id"],
            "transcript": row["transcript"],
            "cleaned_transcript": result.normalized_text,
            "tokens_before_cleaning": input_count,
            "tokens_after_cleaning": output_count,
            "fillers_removed": " ".join(result.fillers_removed),
            "stop_words_removed": " ".join(result.stop_words_removed),
            "noise_tokens_removed": " ".join(result.noise_tokens_removed),
            "repeated_tokens_removed": " ".join(result.repeated_tokens_removed),
            "protected_keywords_preserved": " ".join(result.protected_keywords_preserved),
            "preprocessing_time_seconds": f"{result.timings['total_preprocessing']:.9f}",
            "regex_tokenizer_time_seconds": f"{tokenizer['regex_execution_time_seconds']:.9f}",
            "nltk_tokenizer_time_seconds": f"{tokenizer['nltk_execution_time_seconds']:.9f}",
            "tokenizer_outputs_equivalent": tokenizer["equivalent"],
        })

    stemmer_results = compare_stemmers(all_clean_tokens)
    # Keep stemmer instances in-memory for candidate error analysis.
    from nltk.stem import LancasterStemmer, PorterStemmer
    stemmer_results["porter"]["stemmer"] = PorterStemmer()
    stemmer_results["lancaster"]["stemmer"] = LancasterStemmer()
    error_rows = analyze_stemming_errors(
        [{"cleaning": clean_transcript(row["transcript"])} for row in dataset],
        stemmer_results,
    )
    lemmatizer_results = compare_lemmatizers(all_clean_tokens)

    input_vocabulary = set(all_clean_tokens)
    comparison_rows: list[dict] = []
    for method, result in [*stemmer_results.items(), *lemmatizer_results.items()]:
        if not result.get("available", True):
            comparison_rows.append({
                "method": method,
                "family": "stemming" if method in stemmer_results else "lemmatization",
                "available": False,
                "error": result["error"],
                "original_vocabulary_size": len(input_vocabulary),
                "resulting_vocabulary_size": "",
                "vocabulary_reduction": "",
                "vocabulary_reduction_percent": "",
                "vocabulary_retention_percent": "",
                "unique_transformations": "",
                "execution_time_seconds": "",
                "semantic_preservation_observation": "Not measured: required resource unavailable.",
            })
            continue
        comparison_rows.append({
            "method": method,
            "family": "stemming" if method in stemmer_results else "lemmatization",
            "available": True,
            "error": "",
            "original_vocabulary_size": result["original_vocabulary_size"],
            "resulting_vocabulary_size": result["resulting_vocabulary_size"],
            "vocabulary_reduction": result["vocabulary_reduction"],
            "vocabulary_reduction_percent": f"{result['vocabulary_reduction_percent']:.6f}",
            "vocabulary_retention_percent": f"{result['vocabulary_retention_percent']:.6f}",
            "unique_transformations": result["unique_transformations"],
            "execution_time_seconds": f"{result['execution_time_seconds']:.9f}",
            "semantic_preservation_observation": (
                "Morphological output only; no semantic accuracy score is claimed."
                if method in lemmatizer_results else
                "Stem output may not be a valid word or preserve lexical meaning."
            ),
        })

    reduction_percent = (
        100.0 * (total_input_tokens - total_output_tokens) / total_input_tokens
        if total_input_tokens else 0.0
    )
    summary_rows = [
        {"metric": "total_input_transcripts", "value": len(dataset), "interpretation": "Controlled evaluation dataset; not representative of all assistants."},
        {"metric": "tokens_before_cleaning", "value": total_input_tokens, "interpretation": "Regex-tokenized after Unicode/contraction/noise normalization and before filler/stop-word cleanup."},
        {"metric": "tokens_after_cleaning", "value": total_output_tokens, "interpretation": "Total tokens in the final cleaned transcripts."},
        {"metric": "token_reduction_percent", "value": f"{reduction_percent:.6f}", "interpretation": "100 * (before - after) / before."},
        {"metric": "filler_words_removed", "value": total_fillers, "interpretation": "Configured filler words/phrases removed."},
        {"metric": "stop_words_removed", "value": total_stop_words, "interpretation": "Configured general/domain stop words removed."},
        {"metric": "noise_tokens_removed", "value": total_noise, "interpretation": "Recognized bracketed ASR annotation tokens removed."},
        {"metric": "protected_keywords_preserved", "value": total_protected, "interpretation": "Occurrences of protected command/negation tokens retained after cleaning."},
        {"metric": "preprocessing_execution_time_seconds", "value": f"{total_time:.9f}", "interpretation": "Sum of measured preprocessing stage durations across input rows; excludes reports and morphology."},
        {"metric": "average_preprocessing_execution_time_seconds", "value": f"{total_time / len(dataset):.9f}", "interpretation": "Total measured preprocessing time divided by transcript count."},
        {"metric": "regex_tokenizer_execution_time_seconds", "value": f"{regex_tokenizer_seconds:.9f}", "interpretation": "Measured tokenization time across cleaned dataset rows."},
        {"metric": "nltk_tokenizer_execution_time_seconds", "value": f"{nltk_tokenizer_seconds:.9f}", "interpretation": "NLTK word_tokenize with preserve_line=True; first-call import/init cost is included."},
        {"metric": "tokenizer_outputs_equivalent", "value": tokenizer_equivalent, "interpretation": "Whether token sequences matched on every cleaned controlled example."},
        {"metric": "over_stemming_candidates", "value": sum(row["error_type"] == "over-stemming_candidate" for row in error_rows), "interpretation": "Automatic same-stem collisions; manually verified unrelated pairs are separately flagged."},
        {"metric": "under_stemming_candidates", "value": sum(row["error_type"] == "under-stemming_candidate" for row in error_rows), "interpretation": "Curated related irregular forms left with distinct stems; candidates, not an accuracy score."},
    ]

    write_csv(output_dir / "preprocessing_report.csv", list(records[0]), records)
    write_csv(
        output_dir / "stemming_lemmatization_comparison.csv",
        list(comparison_rows[0]),
        comparison_rows,
    )
    error_fields = ["method", "error_type", "word_a", "word_b", "shared_stem", "basis", "manual_relation", "manually_verified_error", "distinct_stems"]
    write_csv(output_dir / "error_analysis.csv", error_fields, error_rows)
    write_csv(output_dir / "evaluation_summary.csv", ["metric", "value", "interpretation"], summary_rows)

    return {
        "transcript_count": len(dataset),
        "input_token_count": total_input_tokens,
        "output_token_count": total_output_tokens,
        "token_reduction_percent": reduction_percent,
        "preprocessing_time_seconds": total_time,
        "average_preprocessing_time_seconds": total_time / len(dataset),
        "stemmer_results": stemmer_results,
        "lemmatizer_results": lemmatizer_results,
        "comparison_rows": comparison_rows,
        "error_rows": error_rows,
        "summary_rows": summary_rows,
        "tokenizer_equivalent": tokenizer_equivalent,
        "reports_directory": str(output_dir),
    }
