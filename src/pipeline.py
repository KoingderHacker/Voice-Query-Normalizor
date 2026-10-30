"""Composition of cleaning, tokenization, and comparative morphology branches."""

from __future__ import annotations

import time

from .lemmatizer_comparison import compare_lemmatizers
from .preprocessing import clean_transcript, nltk_tokenize, regex_tokenize
from .stemmer_comparison import compare_stemmers


def run_pipeline(text: str, *, include_morphology: bool = True) -> dict:
    cleaning = clean_transcript(text)
    regex_tokens = list(cleaning.tokens)
    if not regex_tokens:
        return {
            "cleaning": cleaning,
            "tokens": regex_tokens,
            "nltk_tokens": (),
            "tokenizer_benchmark": None,
            "stemmers": {},
            "lemmatizers": {},
            "analysis_performed": False,
        }
    if not include_morphology:
        return {
            "cleaning": cleaning,
            "tokens": regex_tokens,
            "nltk_tokens": (),
            "tokenizer_benchmark": None,
            "stemmers": {},
            "lemmatizers": {},
            "analysis_performed": False,
        }

    tokenizer_benchmark = benchmark_tokenizers(cleaning.normalized_text)
    nltk_tokens = tokenizer_benchmark["nltk_tokens"]
    # Regex is the production tokenizer; compare the alternate implementation
    # without changing the canonical cleaned transcript.
    stems = compare_stemmers(regex_tokens)
    lemmas = compare_lemmatizers(regex_tokens)
    return {
        "cleaning": cleaning,
        "tokens": regex_tokens,
        "nltk_tokens": nltk_tokens,
        "tokenizer_benchmark": tokenizer_benchmark,
        "stemmers": stems,
        "lemmatizers": lemmas,
        "analysis_performed": True,
    }


def benchmark_tokenizers(text: str) -> dict:
    started = time.perf_counter()
    regex_tokens = tuple(regex_tokenize(text))
    regex_elapsed = time.perf_counter() - started
    started = time.perf_counter()
    nltk_tokens = nltk_tokenize(text)
    nltk_elapsed = time.perf_counter() - started
    return {
        "regex_tokens": regex_tokens,
        "nltk_tokens": nltk_tokens,
        "regex_execution_time_seconds": regex_elapsed,
        "nltk_execution_time_seconds": nltk_elapsed,
        "equivalent": regex_tokens == nltk_tokens,
    }
