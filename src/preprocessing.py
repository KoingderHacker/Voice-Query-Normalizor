"""Deterministic, conservative normalization for voice-assistant transcripts."""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from . import config

_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_PUNCTUATION_RE = re.compile(r"([!?.,;:])\1+")
_ASR_ARTIFACT_ALTERNATION = "|".join(
    re.escape(word) for word in sorted(config.ASR_ARTIFACTS, key=len, reverse=True)
)
_ASR_TAG_RE = re.compile(
    r"(?<!\w)[\[<(]\s*(" + _ASR_ARTIFACT_ALTERNATION + r")\s*[\])>](?!\w)",
    re.I,
)
_CONTRACTION_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(key) for key in sorted(config.CONTRACTIONS, key=len, reverse=True)) + r")(?!\w)",
    re.I,
)
_NUMERIC_TOKEN_PATTERN = r"[+-]?\d+(?::\d{1,2})?(?:\.\d+)?%?"
_NUMERIC_TOKEN_RE = re.compile(rf"^{_NUMERIC_TOKEN_PATTERN}$")
_TOKEN_RE = re.compile(
    rf"{_NUMERIC_TOKEN_PATTERN}|[^\W\d_]+(?:'[^\W\d_]+)?",
    re.UNICODE | re.IGNORECASE,
)
_NONLINGUISTIC_RE = re.compile(r"[^\w\s':.%+-]", re.UNICODE)
_REPEATED_LETTER_NOISE_RE = re.compile(
    r"(?<!\w)(?:u+h{2,}|u+m{2,}|h{3,}m+|e+r{2,})(?!\w)", re.I
)


@dataclass(frozen=True)
class CleaningResult:
    """Observable result and counters from each independently timed stage."""

    raw_text: str
    normalized_text: str
    tokens: tuple[str, ...]
    tokens_before_cleaning: tuple[str, ...]
    fillers_removed: tuple[str, ...]
    stop_words_removed: tuple[str, ...]
    noise_tokens_removed: tuple[str, ...]
    repeated_tokens_removed: tuple[str, ...]
    protected_keywords_preserved: tuple[str, ...]
    timings: dict[str, float]

    @property
    def removed_words(self) -> tuple[str, ...]:
        return self.fillers_removed + self.stop_words_removed + self.noise_tokens_removed + self.repeated_tokens_removed


def validate_input(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Transcript must be a string.")
    if not text.strip():
        raise ValueError("Transcript must not be empty.")
    return text


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\u2019", "'").replace("\u2018", "'")


def normalize_contractions(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return config.CONTRACTIONS[match.group(0).lower()]

    return _CONTRACTION_RE.sub(replace, text)


def remove_regex_noise(text: str) -> tuple[str, tuple[str, ...]]:
    """Remove known annotation tags and non-linguistic symbols, retaining digits."""
    removed = tuple(match.group(1).lower() for match in _ASR_TAG_RE.finditer(text))
    text = _ASR_TAG_RE.sub(" ", text)
    text = _REPEATED_PUNCTUATION_RE.sub(r"\1", text)
    text = _NONLINGUISTIC_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text, removed


def regex_tokenize(text: str) -> list[str]:
    """Tokenize Unicode words and preserve integer, decimal, time, and percent values."""
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def is_semantic_numeric_token(token: str) -> bool:
    """Recognize numeric values/identifiers that duplicate cleanup must retain."""
    return _NUMERIC_TOKEN_RE.fullmatch(token) is not None


@lru_cache(maxsize=4096)
def nltk_tokenize(text: str) -> tuple[str, ...]:
    """NLTK tokenization without requiring a downloaded Punkt sentence model."""
    from nltk import word_tokenize

    return tuple(token.lower() for token in word_tokenize(text, preserve_line=True))


def clean_transcript(
    raw_text: str,
    *,
    filler_words: Iterable[str] | None = None,
    filler_phrases: Iterable[str] | None = None,
    general_stop_words: Iterable[str] | None = None,
    domain_stop_words: Iterable[str] | None = None,
    protected_words: Iterable[str] | None = None,
) -> CleaningResult:
    """Normalize a transcript while giving command and negation words priority."""
    validate_input(raw_text)
    timings: dict[str, float] = {}

    started = time.perf_counter()
    text = normalize_unicode(raw_text).lower()
    timings["unicode_and_casing"] = time.perf_counter() - started

    started = time.perf_counter()
    text = normalize_contractions(text)
    timings["contraction_normalization"] = time.perf_counter() - started

    started = time.perf_counter()
    text, noise_removed = remove_regex_noise(text)
    timings["regex_noise_removal"] = time.perf_counter() - started

    started = time.perf_counter()
    raw_tokens = tuple(regex_tokenize(text))
    timings["tokenization"] = time.perf_counter() - started

    fillers = {
        word.lower() for word in
        (config.VOICE_FILLER_WORDS if filler_words is None else filler_words)
    }
    phrases = tuple(
        phrase.lower() for phrase in
        (config.VOICE_FILLER_PHRASES if filler_phrases is None else filler_phrases)
    )
    started = time.perf_counter()
    phrase_tokens = [
        (phrase.lower(), tuple(regex_tokenize(phrase.lower())))
        for phrase in sorted(phrases, key=len, reverse=True)
    ]
    filtered: list[str] = []
    filler_removed: list[str] = []
    token_index = 0
    while token_index < len(raw_tokens):
        matched_phrase = next(
            (
                (label, words)
                for label, words in phrase_tokens
                if words and raw_tokens[token_index:token_index + len(words)] == words
            ),
            None,
        )
        if matched_phrase:
            label, words = matched_phrase
            filler_removed.append(label)
            token_index += len(words)
            continue
        token = raw_tokens[token_index]
        if token in fillers or _REPEATED_LETTER_NOISE_RE.fullmatch(token):
            filler_removed.append(token)
        else:
            filtered.append(token)
        token_index += 1
    timings["filler_removal"] = time.perf_counter() - started

    protected = {
        word.lower() for word in
        (config.PROTECTED_COMMAND_WORDS if protected_words is None else protected_words)
    }
    started = time.perf_counter()
    repeated_removed: list[str] = []
    deduplicated: list[str] = []
    for token in filtered:
        if (
            deduplicated
            and token == deduplicated[-1]
            and not is_semantic_numeric_token(token)
            and token not in {"not", "no", "never"}
            and token not in protected
        ):
            repeated_removed.append(token)
        else:
            deduplicated.append(token)
    timings["repeated_word_cleanup"] = time.perf_counter() - started

    stop_words = {
        word.lower() for word in
        (config.GENERAL_STOP_WORDS if general_stop_words is None else general_stop_words)
    }
    stop_words.update(
        word.lower() for word in
        (config.DOMAIN_STOP_WORDS if domain_stop_words is None else domain_stop_words)
    )
    started = time.perf_counter()
    stop_removed: list[str] = []
    final_tokens: list[str] = []
    for token in deduplicated:
        if token in stop_words and token not in protected:
            stop_removed.append(token)
        else:
            final_tokens.append(token)
    timings["stop_word_filtering"] = time.perf_counter() - started

    timings["total_preprocessing"] = sum(timings.values())
    preserved = tuple(token for token in final_tokens if token in protected)
    return CleaningResult(
        raw_text=raw_text,
        normalized_text=" ".join(final_tokens),
        tokens=tuple(final_tokens),
        tokens_before_cleaning=raw_tokens,
        fillers_removed=tuple(filler_removed),
        stop_words_removed=tuple(stop_removed),
        noise_tokens_removed=noise_removed,
        repeated_tokens_removed=tuple(repeated_removed),
        protected_keywords_preserved=preserved,
        timings=timings,
    )
