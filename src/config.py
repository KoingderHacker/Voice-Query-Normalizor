"""Configurable lexical rules for conservative voice-command normalization."""

from __future__ import annotations

# Only low-risk conversational fillers are enabled by default. Add/remove entries
# here to adapt the cleaner without changing its processing logic.
VOICE_FILLER_WORDS = {
    "uh",
    "um",
    "hmm",
    "erm",
    "ah",
    "er",
}
# No multiword phrase is removed by default: expressions such as "you know"
# may be meaningful questions. Add a phrase only for a reviewed domain.
VOICE_FILLER_PHRASES: set[str] = set()

# Intentionally small: function words that rarely carry command intent.
GENERAL_STOP_WORDS = {
    "a",
    "an",
    "the",
}

# Politeness / addressing language can be removed without suppressing commands.
DOMAIN_STOP_WORDS = {
    "please",
    "kindly",
}

# Never delete these words as stop words. This list is explicit and configurable;
# repeated instances may still be deduplicated by the separate repetition rule.
PROTECTED_COMMAND_WORDS = {
    "on",
    "off",
    "open",
    "close",
    "start",
    "stop",
    "increase",
    "decrease",
    "next",
    "previous",
    "play",
    "pause",
    "cancel",
    "delete",
    "enable",
    "disable",
    "set",
    "call",
    "send",
    "show",
    "search",
    "turn",
    "not",
    "no",
    "never",
    "cannot",
}

CONTRACTIONS = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
}

# These are known transcription annotations, not spoken words. The pattern is
# deliberately limited to bracketed/angle-bracketed ASR artifacts.
ASR_ARTIFACTS = {
    "noise",
    "unk",
    "unknown",
    "inaudible",
    "unintelligible",
    "static",
}
