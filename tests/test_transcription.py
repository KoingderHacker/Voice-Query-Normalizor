import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src import transcription
from src.pipeline import run_pipeline
from src.transcription import (
    TranscriptionUnavailable,
    transcribe_audio,
    transcribe_audio_bytes,
    validate_audio_path,
)


class FakeModel:
    """Stands in for faster_whisper.WhisperModel so tests need no model download."""

    def transcribe(self, path, **kwargs):
        segments = [SimpleNamespace(text=" Uh please"), SimpleNamespace(text="turn OFF the light!!! ")]
        info = SimpleNamespace(language="en", language_probability=0.98, duration=2.5)
        return iter(segments), info


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.audio = Path(self.directory.name) / "command.wav"
        self.audio.write_bytes(b"RIFF")

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            validate_audio_path(Path(self.directory.name) / "missing.wav")

    def test_unsupported_extension(self):
        text_file = Path(self.directory.name) / "notes.txt"
        text_file.write_text("hello")
        with self.assertRaises(ValueError):
            validate_audio_path(text_file)

    def test_transcribe_joins_segments(self):
        with mock.patch.object(transcription, "_load_model", return_value=FakeModel()):
            result = transcribe_audio(self.audio)
        self.assertEqual(result.text, "Uh please turn OFF the light!!!")
        self.assertEqual(result.language, "en")
        self.assertAlmostEqual(result.audio_duration_seconds, 2.5)

    def test_transcript_feeds_existing_pipeline(self):
        with mock.patch.object(transcription, "_load_model", return_value=FakeModel()):
            result = transcribe_audio(self.audio)
        cleaned = run_pipeline(result.text, include_morphology=False)["cleaning"]
        self.assertEqual(cleaned.normalized_text, "turn off light")

    def test_bytes_input(self):
        with mock.patch.object(transcription, "_load_model", return_value=FakeModel()):
            result = transcribe_audio_bytes(b"RIFF", "recording.wav")
        self.assertTrue(result.text)
        with self.assertRaises(ValueError):
            transcribe_audio_bytes(b"", "recording.wav")
        with self.assertRaises(ValueError):
            transcribe_audio_bytes(b"data", "recording.txt")

    def test_missing_dependency_is_reported(self):
        transcription._MODEL_CACHE.clear()
        with mock.patch.dict("sys.modules", {"faster_whisper": None}):
            with self.assertRaises(TranscriptionUnavailable):
                transcribe_audio(self.audio)


if __name__ == "__main__":
    unittest.main()
