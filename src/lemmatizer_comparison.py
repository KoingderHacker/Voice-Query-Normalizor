"""Measured WordNet and optional spaCy English lemmatization."""

from __future__ import annotations

import time
from collections.abc import Iterable
from functools import lru_cache

from .preprocessing import is_semantic_numeric_token
from .utils import vocabulary_metrics

SPACY_MODEL_NAME = "en_core_web_sm"


class LemmatizerUnavailable(RuntimeError):
    """Raised when an optional NLP resource is not installed."""


@lru_cache(maxsize=1)
def _wordnet_lemmatizer():
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet

    try:
        wordnet.ensure_loaded()
    except LookupError as error:
        raise LemmatizerUnavailable(
            "NLTK WordNet data is missing. Install it with "
            "`python -m nltk.downloader wordnet omw-1.4`."
        ) from error
    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def _spacy_pipeline():
    try:
        import spacy
    except ImportError as error:
        raise LemmatizerUnavailable(
            "spaCy is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from error
    try:
        return spacy.load(SPACY_MODEL_NAME, disable=["parser", "ner"])
    except OSError as error:
        raise LemmatizerUnavailable(
            f"spaCy model {SPACY_MODEL_NAME!r} is missing. Install it with "
            "`python -m spacy download en_core_web_sm`."
        ) from error


def _wordnet_pos_tag(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Use POS-aware lemmatization when the NLTK tagger resource is available."""
    from nltk import pos_tag

    try:
        tagged = pos_tag(list(tokens))
    except LookupError:
        tagged = [(token, _fallback_penn_tag(token)) for token in tokens]

    def map_pos(tag: str) -> str:
        if tag.startswith("J"):
            return "a"
        if tag.startswith("V"):
            return "v"
        if tag.startswith("R"):
            return "r"
        return "n"

    return tuple(map_pos(tag) for _, tag in tagged)


def _fallback_penn_tag(token: str) -> str:
    """Approximate common inflectional POS when the optional tagger data is absent."""
    if len(token) > 4 and token.endswith(("ing", "ed")):
        return "VBG" if token.endswith("ing") else "VBD"
    if len(token) > 4 and token.endswith(("ous", "ful", "able", "ive", "al")):
        return "JJ"
    return "NN"


def compare_lemmatizers(tokens: Iterable[str]) -> dict[str, dict]:
    words = tuple(tokens)
    original_vocabulary = set(words)
    results: dict[str, dict] = {}

    try:
        lemmatizer = _wordnet_lemmatizer()
        from nltk import pos_tag

        try:
            pos_tag([])
        except LookupError:
            pass
        started = time.perf_counter()
        pos_tags = _wordnet_pos_tag(words)
        transformed = tuple(
            lemmatizer.lemmatize(token, pos) for token, pos in zip(words, pos_tags)
        )
        elapsed = time.perf_counter() - started
        results["wordnet"] = _result(words, transformed, elapsed)
    except LemmatizerUnavailable as error:
        results["wordnet"] = _unavailable_result(len(original_vocabulary), str(error))

    try:
        nlp = _spacy_pipeline()
        started = time.perf_counter()
        from spacy.symbols import ORTH

        for token in words:
            if is_semantic_numeric_token(token):
                nlp.tokenizer.add_special_case(token, [{ORTH: token}])
        doc = nlp(" ".join(words))
        transformed = tuple(token.lemma_.lower() for token in doc if not token.is_space)
        elapsed = time.perf_counter() - started
        results["spacy"] = _result(words, transformed, elapsed)
    except LemmatizerUnavailable as error:
        results["spacy"] = _unavailable_result(len(original_vocabulary), str(error))

    for result in results.values():
        result.setdefault("available", True)
        result.setdefault("original_vocabulary_size", len(original_vocabulary))
    return results


def _unavailable_result(original_vocabulary_size: int, error: str) -> dict:
    """Preserve the morphology result schema when an optional resource is absent."""
    return {
        "available": False,
        "error": error,
        "tokens": (),
        "vocabulary": set(),
        "original_vocabulary_size": original_vocabulary_size,
        "resulting_vocabulary_size": None,
        "vocabulary_reduction": None,
        "vocabulary_reduction_percent": None,
        "vocabulary_retention_percent": None,
        "unique_transformations": None,
        "transformations": {},
        "execution_time_seconds": None,
    }


def _result(words: tuple[str, ...], transformed: tuple[str, ...], elapsed: float) -> dict:
    original_vocabulary = set(words)
    vocabulary = set(transformed)
    metrics = vocabulary_metrics(original_vocabulary, vocabulary)
    transformations: dict[str, str] = {}
    for word, lemma in zip(words, transformed):
        if word != lemma:
            transformations.setdefault(word, lemma)
    return {
        "available": True,
        "tokens": transformed,
        "vocabulary": vocabulary,
        **metrics,
        "unique_transformations": len(transformations),
        "transformations": transformations,
        "execution_time_seconds": elapsed,
    }
