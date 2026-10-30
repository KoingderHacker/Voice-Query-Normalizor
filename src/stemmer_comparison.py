"""Measured comparisons of NLTK Porter and Lancaster stemmers."""

from __future__ import annotations

import time
from collections.abc import Iterable

from nltk.stem import LancasterStemmer, PorterStemmer

from .utils import vocabulary_metrics


def compare_stemmers(tokens: Iterable[str]) -> dict[str, dict]:
    words = tuple(tokens)
    original_vocabulary = set(words)
    methods = {
        "porter": PorterStemmer(),
        "lancaster": LancasterStemmer(),
    }
    results: dict[str, dict] = {}
    for name, stemmer in methods.items():
        started = time.perf_counter()
        transformed = tuple(stemmer.stem(word) for word in words)
        elapsed = time.perf_counter() - started
        vocabulary = set(transformed)
        metrics = vocabulary_metrics(original_vocabulary, vocabulary)
        transformations = {word: stemmer.stem(word) for word in original_vocabulary if stemmer.stem(word) != word}
        results[name] = {
            "tokens": transformed,
            "vocabulary": vocabulary,
            **metrics,
            "unique_transformations": len(transformations),
            "transformations": transformations,
            "execution_time_seconds": elapsed,
        }
    return results
