"""Lightweight Streamlit demonstration for Problem 1."""

import streamlit as st

from src.evaluation import run_evaluation
from src.pipeline import run_pipeline
from src.transcription import (
    DEFAULT_MODEL_SIZE,
    MODEL_SIZES,
    SUPPORTED_AUDIO_EXTENSIONS,
    TranscriptionUnavailable,
    transcribe_audio_bytes,
)


st.set_page_config(page_title="Voice Command Normalizer", page_icon=":material/graphic_eq:", layout="centered")
st.title("Voice assistant query normalizer")
st.caption("Problem 1 · deterministic transcript cleaning and morphology comparison")
st.session_state.setdefault("normalization_result", None)
st.session_state.setdefault("evaluation_result", None)
st.session_state.setdefault("transcription_result", None)


def _numeric_display_value(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return int(numeric) if numeric.is_integer() else numeric


input_mode = st.radio("Input type", ["Text", "Audio file", "Microphone"], horizontal=True)

text = ""
audio_source = None  # (bytes, filename) when an audio input is selected
if input_mode == "Text":
    text = st.text_area(
        "Input transcript",
        "Uh PLEASE turn OFF the living room light!!!",
        height=120,
    )
else:
    if input_mode == "Audio file":
        uploaded = st.file_uploader(
            "Upload speech audio",
            type=[extension.lstrip(".") for extension in SUPPORTED_AUDIO_EXTENSIONS],
        )
        if uploaded is not None:
            st.audio(uploaded)
            audio_source = (uploaded.getvalue(), uploaded.name)
    else:
        recorded = st.audio_input("Record a voice command")
        if recorded is not None:
            audio_source = (recorded.getvalue(), "recording.wav")
    column_model, column_language = st.columns(2)
    model_size = column_model.selectbox(
        "Speech model",
        MODEL_SIZES,
        index=MODEL_SIZES.index(DEFAULT_MODEL_SIZE),
        help="Larger models are more accurate but slower. Each is downloaded once on first use.",
    )
    language_code = column_language.text_input(
        "Spoken language code",
        "en",
        help="ISO code such as en, hi, gu. Enter auto to detect the language.",
    ).strip().lower()

include_morphology = st.checkbox(
    "Include Porter, Lancaster, WordNet, and spaCy analysis (loads spaCy model)",
    value=False,
)

if st.button("Normalize transcript", type="primary"):
    st.session_state.transcription_result = None
    st.session_state.normalization_result = None
    try:
        transcript = text
        if input_mode != "Text":
            if audio_source is None:
                raise ValueError("Provide an audio input first.")
            with st.spinner("Transcribing audio (the first run downloads the speech model)..."):
                transcription = transcribe_audio_bytes(
                    audio_source[0],
                    audio_source[1],
                    model_size=model_size,
                    language=None if language_code in ("", "auto") else language_code,
                )
            st.session_state.transcription_result = transcription
            transcript = transcription.text
            if not transcript:
                raise ValueError("No speech was detected in the audio.")
        st.session_state.normalization_result = run_pipeline(
            transcript,
            include_morphology=include_morphology,
        )
    except TranscriptionUnavailable as error:
        st.error(f"Speech recognition unavailable: {error}")
    except (TypeError, ValueError) as error:
        st.error(str(error))

transcription = st.session_state.transcription_result
if transcription is not None:
    st.subheader("Recognized transcript")
    st.code(transcription.text or "(no speech detected)")
    st.caption(
        f"Whisper {transcription.model_size} · language {transcription.language} "
        f"({transcription.language_probability:.2f}) · audio {transcription.audio_duration_seconds:.2f}s · "
        f"transcription {transcription.transcription_time_seconds:.2f}s"
    )

result = st.session_state.normalization_result
if result is not None:
    cleaning = result["cleaning"]
    if cleaning.normalized_text:
        st.subheader("Normalized command")
        st.code(cleaning.normalized_text)
    else:
        st.warning("Input contains no meaningful linguistic tokens after preprocessing.")
    st.write("**Tokens:**", list(result["tokens"]))
    st.write("**Removed words/artifacts:**", list(cleaning.removed_words))
    st.write("**Preserved command keywords:**", list(cleaning.protected_keywords_preserved))
    st.write(f"**Preprocessing time:** {cleaning.timings['total_preprocessing']:.9f} seconds")

    if result["analysis_performed"]:
        comparison = []
        for method, output in [*result["stemmers"].items(), *result["lemmatizers"].items()]:
            if output.get("available", True):
                comparison.append({
                    "Method": method.title(),
                    "Output": " ".join(output["tokens"]),
                    "Vocabulary Before": output["original_vocabulary_size"],
                    "Vocabulary After": output["resulting_vocabulary_size"],
                    "Reduction": output["vocabulary_reduction"],
                    "Reduction %": round(output["vocabulary_reduction_percent"], 4),
                    "Retention %": round(output["vocabulary_retention_percent"], 4),
                    "Execution Time (s)": output["execution_time_seconds"],
                })
            else:
                comparison.append({"Method": method.title(), "Output": output["error"]})
        st.subheader("Morphology comparison")
        st.dataframe(comparison, hide_index=True)
        if result["tokenizer_benchmark"] is not None:
            benchmark = result["tokenizer_benchmark"]
            st.caption(
                "Tokenizer check: "
                f"regex {benchmark['regex_execution_time_seconds']:.9f}s; "
                f"NLTK {benchmark['nltk_execution_time_seconds']:.9f}s; "
                f"equivalent: {benchmark['equivalent']}."
            )
    elif cleaning.normalized_text:
        st.info("Morphology was not run. Enable the comparison checkbox and normalize again.")

st.divider()
st.subheader("Controlled evaluation")
st.caption(
    "Run the checked-in controlled benchmark. Stem collisions and curated "
    "related-form splits are reported as candidates, not general accuracy scores."
)
if st.button("Run controlled evaluation"):
    try:
        with st.spinner("Running the controlled NLP evaluation..."):
            st.session_state.evaluation_result = run_evaluation()
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Evaluation failed: {error}")
        st.session_state.evaluation_result = None

evaluation = st.session_state.evaluation_result
if evaluation is not None:
    st.markdown("**Morphology comparison**")
    method_names = {
        "porter": "Porter",
        "lancaster": "Lancaster",
        "wordnet": "WordNet",
        "spacy": "spaCy",
    }
    morphology_rows = []
    for row in evaluation["comparison_rows"]:
        morphology_rows.append({
            "Method": method_names.get(row["method"], row["method"]),
            "Vocabulary Before": _numeric_display_value(row["original_vocabulary_size"]),
            "Vocabulary After": _numeric_display_value(row["resulting_vocabulary_size"]),
            "Reduction": _numeric_display_value(row["vocabulary_reduction"]),
            "Reduction %": _numeric_display_value(row["vocabulary_reduction_percent"]),
            "Retention %": _numeric_display_value(row["vocabulary_retention_percent"]),
            "Execution Time (s)": _numeric_display_value(row["execution_time_seconds"]),
        })
    st.dataframe(morphology_rows, hide_index=True)

    over_rows = [
        row for row in evaluation["error_rows"]
        if row["error_type"] == "over-stemming_candidate"
    ]
    under_rows = [
        row for row in evaluation["error_rows"]
        if row["error_type"] == "under-stemming_candidate"
    ]
    st.markdown("**Stemming error analysis**")
    st.caption(
        "Over-stemming rows are same-stem collision candidates; only curated "
        "unrelated pairs are manually marked as verified. Under-stemming rows "
        "are curated related forms with different stems and are candidates, not "
        "validated semantic errors."
    )
    st.write(f"Over-stemming candidates: {len(over_rows)}")
    st.dataframe(over_rows, hide_index=True)
    st.write(f"Under-stemming candidates: {len(under_rows)}")
    st.dataframe(under_rows, hide_index=True)

    summary_by_name = {row["metric"]: row["value"] for row in evaluation["summary_rows"]}
    aggregate_names = (
        "total_input_transcripts",
        "tokens_before_cleaning",
        "tokens_after_cleaning",
        "token_reduction_percent",
        "filler_words_removed",
        "stop_words_removed",
        "protected_keywords_preserved",
    )
    aggregate_labels = {
        "total_input_transcripts": "Transcripts processed",
        "tokens_before_cleaning": "Tokens before cleaning",
        "tokens_after_cleaning": "Tokens after cleaning",
        "token_reduction_percent": "Token reduction (%)",
        "filler_words_removed": "Fillers removed",
        "stop_words_removed": "Stop words removed",
        "protected_keywords_preserved": "Protected keywords preserved",
    }
    st.markdown("**Aggregate evaluation**")
    st.dataframe(
        [
            {
                "Metric": aggregate_labels[name],
                "Measured value": _numeric_display_value(summary_by_name[name]),
            }
            for name in aggregate_names
        ],
        hide_index=True,
    )
    st.caption(f"CSV reports generated in: {evaluation['reports_directory']}")
