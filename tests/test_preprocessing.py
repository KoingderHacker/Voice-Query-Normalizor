import unittest
import csv
import tempfile
from pathlib import Path

from nltk.stem import LancasterStemmer, PorterStemmer

from src.evaluation import run_evaluation
from src.lemmatizer_comparison import LemmatizerUnavailable, compare_lemmatizers
from src.config import PROTECTED_COMMAND_WORDS
from src.pipeline import run_pipeline
from src.preprocessing import clean_transcript, is_semantic_numeric_token, regex_tokenize
from src.stemmer_comparison import compare_stemmers
from src.utils import vocabulary_metrics


class PreprocessingTests(unittest.TestCase):
    def test_filler_and_politeness_removal(self):
        result = clean_transcript("Uh please set an alarm")
        self.assertEqual(result.normalized_text, "set alarm")
        self.assertEqual(result.fillers_removed, ("uh",))
        self.assertEqual(result.stop_words_removed, ("please", "an"))

    def test_casing_and_punctuation_cleanup(self):
        result = clean_transcript("Uh PLEASE turn OFF the living room light!!!")
        self.assertEqual(result.normalized_text, "turn off living room light")

    def test_contractions_preserve_negation(self):
        result = clean_transcript("don't turn off the fan")
        self.assertEqual(result.normalized_text, "do not turn off fan")
        self.assertIn("not", result.tokens)

    def test_common_contractions_expand(self):
        self.assertEqual(clean_transcript("I'm sure it's ready").normalized_text, "i am sure it is ready")
        self.assertEqual(clean_transcript("you're here").normalized_text, "you are here")

    def test_protected_command_words_survive_stop_word_filter(self):
        result = clean_transcript("please play the next song")
        self.assertEqual(result.normalized_text, "play next song")
        self.assertEqual(result.protected_keywords_preserved, ("play", "next"))

    def test_protected_words_override_configured_stop_words(self):
        result = clean_transcript(
            "turn off play next",
            general_stop_words={"off", "play", "next"},
        )
        self.assertEqual(result.normalized_text, "turn off play next")

    def test_numbers_and_clock_times_are_preserved(self):
        result = clean_transcript("set temperature to 22 at 7:30")
        self.assertIn("22", result.tokens)
        self.assertIn("7:30", result.tokens)

    def test_repeated_words_are_collapsed_but_negation_and_numbers_remain(self):
        result = clean_transcript("room room turn turn off no no 7 7")
        self.assertEqual(result.normalized_text, "room turn turn off no no 7 7")
        self.assertEqual(result.repeated_tokens_removed, ("room",))

    def test_asr_noise_markers_and_symbols_are_removed(self):
        result = clean_transcript("play [noise] song @#$!!! uhhh ummm")
        self.assertEqual(result.normalized_text, "play song")
        self.assertEqual(result.noise_tokens_removed, ("noise",))
        self.assertEqual(result.fillers_removed, ("uhhh", "ummm"))
        self.assertIn("uhhh", result.tokens_before_cleaning)

    def test_custom_filler_configuration(self):
        result = clean_transcript("well play music", filler_words={"well"})
        self.assertEqual(result.normalized_text, "play music")
        self.assertEqual(result.fillers_removed, ("well",))

    def test_configured_filler_phrase(self):
        result = clean_transcript("well you know play music", filler_phrases={"you know"})
        self.assertEqual(result.normalized_text, "well play music")
        self.assertEqual(result.fillers_removed, ("you know",))

    def test_you_know_is_not_removed_by_default(self):
        result = clean_transcript("do you know the time?")
        self.assertEqual(result.normalized_text, "do you know time")
        command_result = clean_transcript("uh you know please turn off the light")
        self.assertEqual(command_result.normalized_text, "you know turn off light")

    def test_repeated_protected_command_words_are_preserved(self):
        self.assertEqual(clean_transcript("stop stop the music").normalized_text, "stop stop music")
        self.assertEqual(clean_transcript("play play the song").normalized_text, "play play song")

    def test_repeated_fillers_are_removed_without_losing_command(self):
        result = clean_transcript("uh uh uh please play music")
        self.assertEqual(result.normalized_text, "play music")
        self.assertEqual(result.fillers_removed, ("uh", "uh", "uh"))

    def test_unicode_words_and_names_are_preserved(self):
        result = clean_transcript("café naïve résumé München open the door")
        for word in ("café", "naïve", "résumé", "münchen"):
            self.assertIn(word, result.tokens)
        self.assertEqual(
            tuple(regex_tokenize(result.normalized_text)),
            result.tokens,
        )

    def test_unknown_words_are_not_silently_removed(self):
        result = clean_transcript("flibbertigibbet xyzzy")
        self.assertEqual(result.normalized_text, "flibbertigibbet xyzzy")

    def test_numeric_time_decimal_percentage_and_identifier_tokens(self):
        text = "set temperature to 22.5 percent and alarm for 7:30 room 204 at 50%"
        result = clean_transcript(text)
        for value in ("22.5", "7:30", "204", "50%"):
            self.assertIn(value, result.tokens)
            self.assertTrue(is_semantic_numeric_token(value))
        self.assertEqual(clean_transcript("set 22.5 22.5").normalized_text, "set 22.5 22.5")
        self.assertEqual(clean_transcript("set 7:30 7:30").normalized_text, "set 7:30 7:30")

    def test_all_protected_commands_survive_stop_word_configuration(self):
        result = clean_transcript(
            "open close start stop play pause next previous increase decrease "
            "cancel delete enable disable set call send show search",
            general_stop_words=PROTECTED_COMMAND_WORDS,
        )
        self.assertEqual(
            result.tokens,
            tuple("open close start stop play pause next previous increase decrease "
                  "cancel delete enable disable set call send show search".split()),
        )

    def test_empty_normalized_result_is_reported_without_morphology(self):
        result = run_pipeline("!!!???", include_morphology=True)
        self.assertEqual(result["cleaning"].normalized_text, "")
        self.assertFalse(result["analysis_performed"])
        self.assertEqual(result["stemmers"], {})
        self.assertEqual(result["lemmatizers"], {})

    def test_normalization_can_skip_optional_morphology(self):
        result = run_pipeline("open café", include_morphology=False)
        self.assertEqual(result["cleaning"].normalized_text, "open café")
        self.assertFalse(result["analysis_performed"])
        self.assertEqual(result["stemmers"], {})

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            clean_transcript("  ")
        with self.assertRaises(TypeError):
            clean_transcript(None)

    def test_regex_tokenizer_preserves_numbers(self):
        self.assertEqual(regex_tokenize("alarm 7:30 at 22.5"), ["alarm", "7:30", "at", "22.5"])

    def test_stemmer_comparison_metrics(self):
        result = compare_stemmers(["connect", "connected", "connecting"])
        for method in ("porter", "lancaster"):
            self.assertLessEqual(result[method]["resulting_vocabulary_size"], 3)
            self.assertGreaterEqual(result[method]["vocabulary_retention_percent"], 0)
            self.assertIn("execution_time_seconds", result[method])
        self.assertEqual(PorterStemmer().stem("connected"), result["porter"]["tokens"][1])
        self.assertEqual(LancasterStemmer().stem("connected"), result["lancaster"]["tokens"][1])

    def test_spacy_lemmatization(self):
        result = compare_lemmatizers(["cars", "running"])["spacy"]
        if result["available"]:
            self.assertEqual(result["tokens"], ("car", "run"))
        else:
            self.assertIn("en_core_web_sm", result["error"])

    def test_numeric_tokens_remain_aligned_in_lemmatizer_comparison(self):
        result = compare_lemmatizers(["22.5", "7:30", "50%"])["spacy"]
        if result["available"]:
            self.assertEqual(result["tokens"], ("22.5", "7:30", "50%"))

    def test_vocabulary_reduction_and_retention_complement(self):
        results = compare_stemmers(["cars", "car", "dogs"])
        for result in results.values():
            self.assertAlmostEqual(
                result["vocabulary_reduction_percent"] + result["vocabulary_retention_percent"],
                100.0,
            )
        for result in compare_lemmatizers(["cars", "car", "dogs"]).values():
            if result.get("available", True):
                self.assertAlmostEqual(
                    result["vocabulary_reduction_percent"] + result["vocabulary_retention_percent"],
                    100.0,
                )

    def test_vocabulary_reduction_metric_exact_100_to_80(self):
        metrics = vocabulary_metrics(
            {f"word-{index}" for index in range(100)},
            {f"word-{index}" for index in range(80)},
        )
        self.assertEqual(metrics["original_vocabulary_size"], 100)
        self.assertEqual(metrics["resulting_vocabulary_size"], 80)
        self.assertEqual(metrics["vocabulary_reduction"], 20)
        self.assertAlmostEqual(metrics["vocabulary_reduction_percent"], 20.0)
        self.assertAlmostEqual(metrics["vocabulary_retention_percent"], 80.0)
        self.assertAlmostEqual(
            metrics["vocabulary_reduction_percent"] + metrics["vocabulary_retention_percent"],
            100.0,
        )

    def test_empty_vocabulary_metrics_do_not_divide_by_zero(self):
        metrics = vocabulary_metrics(set(), set())
        self.assertEqual(metrics["vocabulary_reduction"], 0)
        self.assertEqual(metrics["vocabulary_reduction_percent"], 0.0)
        self.assertEqual(metrics["vocabulary_retention_percent"], 0.0)

    def test_every_morphology_method_returns_the_complete_metric_schema(self):
        required = {
            "original_vocabulary_size",
            "resulting_vocabulary_size",
            "vocabulary_reduction",
            "vocabulary_reduction_percent",
            "vocabulary_retention_percent",
            "unique_transformations",
            "execution_time_seconds",
        }
        methods = {
            **compare_stemmers(["connect", "connected", "dogs"]),
            **compare_lemmatizers(["connect", "connected", "dogs"]),
        }
        self.assertEqual(set(methods), {"porter", "lancaster", "wordnet", "spacy"})
        for method, result in methods.items():
            with self.subTest(method=method):
                self.assertTrue(required.issubset(result))
                if result.get("available", True):
                    self.assertEqual(
                        result["vocabulary_reduction"],
                        result["original_vocabulary_size"] - result["resulting_vocabulary_size"],
                    )
                    self.assertAlmostEqual(
                        result["vocabulary_reduction_percent"],
                        100 * result["vocabulary_reduction"] / result["original_vocabulary_size"],
                    )
                    self.assertAlmostEqual(
                        result["vocabulary_retention_percent"],
                        100 * result["resulting_vocabulary_size"] / result["original_vocabulary_size"],
                    )

    def test_evaluation_generates_complete_comparison_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluation = run_evaluation(output_dir=temp_dir)
            csv_path = Path(temp_dir) / "stemming_lemmatization_comparison.csv"
            with csv_path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(
                {row["method"] for row in rows},
                {"porter", "lancaster", "wordnet", "spacy"},
            )
            for row in rows:
                with self.subTest(method=row["method"]):
                    before = int(row["original_vocabulary_size"])
                    after = int(row["resulting_vocabulary_size"])
                    reduction = int(row["vocabulary_reduction"])
                    self.assertEqual(reduction, before - after)
                    self.assertAlmostEqual(
                        float(row["vocabulary_reduction_percent"]),
                        100 * reduction / before if before else 0.0,
                        places=5,
                    )
                    self.assertAlmostEqual(
                        float(row["vocabulary_retention_percent"]),
                        100 * after / before if before else 0.0,
                        places=5,
                    )
            self.assertEqual(evaluation["transcript_count"], 30)

    def test_wordnet_lemmatization(self):
        result = compare_lemmatizers(["cars", "running"])
        wordnet = result["wordnet"]
        if wordnet["available"]:
            self.assertEqual(wordnet["tokens"], ("car", "run"))
        else:
            self.assertIn("nltk.downloader wordnet", wordnet["error"])


if __name__ == "__main__":
    unittest.main()
