"""CLI for Problem 1: voice-assistant query normalization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.evaluation import run_evaluation
from src.lemmatizer_comparison import LemmatizerUnavailable
from src.pipeline import run_pipeline
from src.transcription import DEFAULT_MODEL_SIZE, MODEL_SIZES, TranscriptionUnavailable, transcribe_audio


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize and compare NLP processing of voice commands.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Transcript text to normalize.")
    source.add_argument("--file", type=Path, help="UTF-8 text file; one transcript per non-empty line.")
    source.add_argument(
        "--audio",
        type=Path,
        nargs="+",
        help="One or more audio files (wav, mp3, m4a, flac, ogg, ...) to transcribe and normalize.",
    )
    source.add_argument("--evaluate", action="store_true", help="Run the controlled dataset and generate CSV reports.")
    parser.add_argument(
        "--audio-model",
        choices=MODEL_SIZES,
        default=DEFAULT_MODEL_SIZE,
        help="Whisper model size used with --audio (default: %(default)s). Larger is slower but more accurate.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Spoken language code for --audio (default: en). Use 'auto' to detect it.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Also run the stemmer and lemmatizer comparisons for --text or --file.",
    )
    return parser


def _display_result(text: str, *, include_morphology: bool = False) -> None:
    result = run_pipeline(text, include_morphology=include_morphology)
    cleaned = result["cleaning"]
    print(f"RAW:\n{text}")
    if cleaned.normalized_text:
        print(f"\nCLEANED:\n{cleaned.normalized_text}")
    else:
        print("\nCLEANED:\nInput contains no meaningful linguistic tokens after preprocessing.")
    print(f"\nTOKENS:\n{list(result['tokens'])}")
    if result["analysis_performed"]:
        for key, label in (("porter", "PORTER"), ("lancaster", "LANCASTER")):
            values = result["stemmers"][key]
            print(f"\n{label}:\n{list(values['tokens'])}")
        for key, label in (("wordnet", "WORDNET"), ("spacy", "SPACY")):
            values = result["lemmatizers"][key]
            print(f"\n{label}:\n{list(values['tokens']) if values.get('available') else 'UNAVAILABLE: ' + values['error']}")
        print("\nVOCABULARY COMPARISON:")
        for key, values in (
            *result["stemmers"].items(),
            *result["lemmatizers"].items(),
        ):
            if values.get("available", True):
                print(
                    f"  {key}: {values['original_vocabulary_size']} -> "
                    f"{values['resulting_vocabulary_size']}; "
                    f"reduction {values['vocabulary_reduction']} "
                    f"({values['vocabulary_reduction_percent']:.2f}%); "
                    f"retention {values['vocabulary_retention_percent']:.2f}%; "
                    f"{values['execution_time_seconds']:.9f}s"
                )
            else:
                print(f"  {key}: unavailable ({values['error']})")
    elif cleaned.normalized_text and include_morphology:
        print("\nMORPHOLOGY:\nUnavailable; see the resource installation instructions.")
    elif cleaned.normalized_text:
        print("\nMORPHOLOGY:\nNot run (pass --analyze to include the comparative NLP branches).")
    print(f"\nREMOVED:\n{list(cleaned.removed_words)}")
    print(f"\nPRESERVED COMMAND KEYWORDS:\n{list(cleaned.protected_keywords_preserved)}")
    if result["tokenizer_benchmark"] is not None:
        print(f"\nTOKENIZER COMPARISON:\nregex={result['tokenizer_benchmark']['regex_execution_time_seconds']:.9f}s, "
              f"nltk={result['tokenizer_benchmark']['nltk_execution_time_seconds']:.9f}s, "
              f"equivalent={result['tokenizer_benchmark']['equivalent']}")
    print(f"\nEXECUTION TIME:\n{cleaned.timings['total_preprocessing']:.9f}s preprocessing")
    print("STAGE TIMES:")
    for stage, elapsed in cleaned.timings.items():
        print(f"  {stage}: {elapsed:.9f}s")


def main() -> int:
    args = _parser().parse_args()
    if args.evaluate:
        report = run_evaluation()
        print(f"Controlled transcripts: {report['transcript_count']}")
        print(f"Preprocessing time: {report['preprocessing_time_seconds']:.9f}s total; "
              f"{report['average_preprocessing_time_seconds']:.9f}s/transcript")
        print(f"Token reduction: {report['token_reduction_percent']:.2f}%")
        for method, values in [*report["stemmer_results"].items(), *report["lemmatizer_results"].items()]:
            if values.get("available", True):
                print(f"{method}: vocab {values['original_vocabulary_size']} -> {values['resulting_vocabulary_size']}; "
                      f"reduction {values['vocabulary_reduction_percent']:.2f}%; "
                      f"retention {values['vocabulary_retention_percent']:.2f}%; "
                      f"time {values['execution_time_seconds']:.9f}s")
            else:
                print(f"{method}: unavailable ({values['error']})")
        print(f"CSV reports: {report['reports_directory']}")
        return 0

    if args.audio is not None:
        language = None if args.language.lower() == "auto" else args.language
        for index, audio_path in enumerate(args.audio, start=1):
            if len(args.audio) > 1:
                print(f"\n=== AUDIO {index}: {audio_path} ===")
            try:
                transcription = transcribe_audio(audio_path, model_size=args.audio_model, language=language)
            except (FileNotFoundError, ValueError) as error:
                print(f"Invalid audio input: {error}", file=sys.stderr)
                return 2
            except TranscriptionUnavailable as error:
                print(f"Speech recognition unavailable: {error}", file=sys.stderr)
                return 3
            print(
                f"ASR ({transcription.model_size}): language={transcription.language} "
                f"({transcription.language_probability:.2f}), audio={transcription.audio_duration_seconds:.2f}s, "
                f"transcription time={transcription.transcription_time_seconds:.2f}s"
            )
            if not transcription.text:
                print("No speech was detected in this audio file.")
                continue
            try:
                _display_result(transcription.text, include_morphology=args.analyze)
            except (TypeError, ValueError) as error:
                print(f"Invalid transcript: {error}", file=sys.stderr)
                return 2
        return 0

    if args.text is not None:
        try:
            _display_result(args.text, include_morphology=args.analyze)
        except (TypeError, ValueError) as error:
            print(f"Invalid transcript: {error}", file=sys.stderr)
            return 2
    else:
        if not args.file.is_file():
            print(f"Input file not found: {args.file}", file=sys.stderr)
            return 2
        lines = [line.strip() for line in args.file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            print(f"No non-empty transcripts found in: {args.file}", file=sys.stderr)
            return 2
        for index, line in enumerate(lines, start=1):
            if len(lines) > 1:
                print(f"\n=== TRANSCRIPT {index} ===")
            _display_result(line, include_morphology=args.analyze)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
