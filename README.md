# Voice Assistant Query Normalizer and Text Cleaner

**Scope:** MT-AIML mini-project Problem 1 only. This project does not implement Problems 2–20. Input can be typed text or speech audio; audio is first converted to a transcript and then handled by the same text pipeline.

## Problem statement and objective

Speech-to-text transcripts commonly contain filler speech, casing and punctuation variation, contractions, repeated words, and ASR annotations. This lightweight Python pipeline produces a normalized transcript for downstream voice-command processing while retaining command intent, negation, and numbers. It compares two stemmers with two lemmatizers and generates measurements from an included controlled dataset. It makes no claim of real-world assistant accuracy.

## Architecture

```text
main.py / app.py
       │
       ├── src/transcription.py       optional audio to transcript (faster-whisper)
       ├── src/preprocessing.py       validation, Unicode, contractions, regex, token cleanup
       ├── src/stemmer_comparison.py  Porter and Lancaster measurements
       ├── src/lemmatizer_comparison.py WordNet and optional spaCy measurements
       ├── src/pipeline.py            composition and tokenizer comparison
       ├── src/evaluation.py          controlled benchmark, metrics, and CSV reports
       ├── src/config.py              configurable lexical rules
       └── src/utils.py               deterministic CSV output helper
```

`data/raw_transcripts/sample.txt` contains demo inputs. `data/evaluation/benchmark.csv` is a controlled, manually assembled set of 30 command-like examples covering fillers, casing, punctuation, contractions, negation, numbers, ASR tags, repeated commands, Unicode terms, and curated morphological pairs. The dataset is for reproducible demonstrations, not a representative sample of real users.

## Preprocessing pipeline

1. Validate non-empty string input.
2. Normalize Unicode with NFKC, standardize curly apostrophes, and lowercase.
3. Expand deterministic contractions (for example, `don't` → `do not`, `can't` → `cannot`, `it's` → `it is`). Ambiguous possessives such as `John's` are not expanded.
4. Remove configured bracketed ASR tags (`[noise]`, `[unk]`, `[inaudible]`, etc.), repeated punctuation, obvious prolonged filler spellings, and non-linguistic symbols. Numeric digits, signed values, decimals, percentages, and clock-time colons are retained.
5. Remove configured fillers and explicitly configured filler phrases, then collapse adjacent repeated ordinary words. Repeated protected command words, negations (`no no`, `not not`), and semantic numeric/time tokens are deliberately retained.
6. Tokenize with a compiled regex. The regex tokenizer is timed against NLTK `word_tokenize(preserve_line=True)` on the controlled inputs.
7. Remove only a small configured set of general and domain stop words. Protected command words take priority over stop-word removal.
8. Return the cleaned words as the canonical transcript; stemming and lemmatization are comparison branches and do not alter that canonical output.

### Regex noise handling

Noise removal is intentionally narrow. Bracketed ASR annotations are recognized as artifacts; arbitrary words such as `noise` are not removed when they appear as ordinary speech. Punctuation and symbols are excluded from the normalized token stream, while numbers remain available to identify alarms, rooms, temperatures, and song numbers. Prolonged filler spellings such as `uhhh` and `ummm` are recognized during filler filtering and counted as removed fillers rather than silently discarded as noise.

### Tokenization and stop-word strategy

The production tokenizer uses a deterministic, precompiled regular expression and preserves tokens such as `22`, `22.5`, and `7:30`. NLTK's tokenizer is benchmarked as an alternative. The report records both measured elapsed times and whether the resulting token sequences match; the first NLTK call may include import/setup overhead. The cleaner also reports time separately for Unicode/casing, contraction expansion, regex noise handling, tokenization, filler filtering, repeated-word cleanup, and stop-word filtering.

`GENERAL_STOP_WORDS`, `DOMAIN_STOP_WORDS`, `VOICE_FILLER_WORDS`, `VOICE_FILLER_PHRASES`, and `PROTECTED_COMMAND_WORDS` are separate sets in `src/config.py`. Defaults remove only articles and politeness markers among ordinary words. They do not blindly apply NLTK's English stop list. `on`, `off`, `open`, `close`, `start`, `stop`, `play`, `pause`, `next`, `previous`, `increase`, `decrease`, `cancel`, `delete`, `enable`, `disable`, `set`, `call`, `send`, `show`, `search`, and negation terms are protected from stop-word removal. Duplicate protected commands are also preserved by repeated-word cleanup.

### Semantic-safety rules

- **Command words:** protected keywords are not deleted by stop-word filtering; the same protected vocabulary is excluded from duplicate collapse.
- **Negation:** contractions such as `don't` expand to `do not`. `not`, `no`, and `never` are protected; repeated negation tokens are not collapsed. For example, `don't turn off the fan` becomes `do not turn off fan`.
- **Numbers:** integers, signed values, decimals, percentages, clock times, and numeric identifiers are treated as semantic numeric tokens. For example, `22`, `22.5`, `7:30`, `-2`, `50%`, and room `204` survive cleanup and duplicate filtering.
- **Fillers:** only the configured single-word low-risk fillers are removed by default. Conversational phrases are not assumed to be noise; in particular, `you know` is not removed by default because it can be part of a meaningful question. Additional phrases can be configured for a reviewed domain.
- **Repeated words:** only adjacent repeated ordinary words are collapsed. Repeated protected commands and meaningful number/time tokens are retained. Repetition can still express emphasis for ordinary words, so this remains a configurable heuristic rather than a universal speech rule.
- **Unicode:** the word tokenizer uses Unicode-aware letter matching, preserving words and names such as `café`, `naïve`, `résumé`, and `München`.
- **Empty output:** punctuation-only or otherwise tokenless inputs are explicitly reported as having no meaningful linguistic tokens. The pipeline does not run stemming or lemmatization on an empty normalized command.

## Stemming and lemmatization

### Porter and Lancaster

Both use NLTK's deterministic English stemmers. For a token sequence with distinct normalized input vocabulary \(V\) and resulting distinct stems \(S\):

- Vocabulary retention (%) = \(100 \times |S| / |V|\)
- Vocabulary reduction = \(|V| - |S|\)
- Vocabulary reduction (%) = \(100 \times (|V| - |S|) / |V|\)
- Unique transformations = number of distinct input word types changed by the method
- Execution time is measured with `time.perf_counter()` around the transformation pass.

Retention and reduction use the same unique normalized input vocabulary \(V\) and unique transformed vocabulary \(S\), so the two percentages sum to 100% when \(V\) is non-empty. These values measure vocabulary compression, **not correctness**. Stems can be non-words, and a high reduction is not evidence of better semantics. Both reduction count and percentage are calculated from the measured method output and written to the comparison CSV, CLI, and UI.

### WordNet and spaCy

WordNet uses NLTK's `WordNetLemmatizer`, with Penn Treebank POS tags mapped to WordNet POS when the NLTK tagger data is available. If that tagger is not installed, common inflectional endings receive a small deterministic POS heuristic and other tokens use noun POS. WordNet corpus data is required.

spaCy uses the small English `en_core_web_sm` model with parser and named-entity components disabled. spaCy is initialized once per process. If the package or model is unavailable, the CLI and reports mark spaCy unavailable and show the exact installation instruction; they do not substitute invented output. A small spaCy model is used instead of a transformer.

Lemmatizers are judged here by their output vocabulary size, retention, transformations, and measured execution time. Timings are for tagging/transformation or model inference after required resources have been initialized; one-time resource/model loading is excluded. No semantic-accuracy score is calculated because the benchmark has no gold-standard lemma annotations.

## Evaluation methodology and error analysis

`python main.py --evaluate` runs every row of the controlled CSV and writes reports from actual execution. `preprocessing_report.csv` has row-level cleaned text, removals, protected words, timings, and tokenizer comparison. `stemming_lemmatization_comparison.csv` contains method-level measured vocabulary and timing data. `evaluation_summary.csv` provides aggregate cleaning counts and timing. `error_analysis.csv` separates automatic stem-collision candidates from pairs with manually curated relation labels.

An over-stemming candidate is a pair of distinct observed words with the same stem. A pair is marked as a **manually verified unrelated pair** only when it matches a curated pair in `src/evaluation.py` (for example, `university` / `universal`) and the current stemmer actually collapses it. Other collisions remain unreviewed candidates, not asserted errors.

Under-stemming candidates are curated related irregular forms that have different stems (for example, `goose` / `geese`). Their relationship is manually curated, but they are labeled candidates rather than stemmer accuracy errors because conflating irregular forms is not necessarily the intended purpose of a stemmer. The report includes only pairs observed in the cleaned dataset. No error count is fabricated.

## Reports and technical documentation

`python main.py --evaluate` generates these current-run reports in `reports/`:

- `preprocessing_report.csv` — transcript-level normalized text, removals, protected terms, and timing.
- `stemming_lemmatization_comparison.csv` — measured vocabulary sizes, reductions, retention, transformation counts, and method times.
- `error_analysis.csv` — actual stem collisions and curated under-stemming candidates.
- `evaluation_summary.csv` — aggregate cleaning and timing statistics.

The submission report is `docs/Architecture_and_Methodology.pdf`. To regenerate it from the latest evaluation CSVs and actual example-pipeline outputs, run:

```powershell
python docs/generate_documentation.py
```

This report generator uses ReportLab (included in `requirements.txt`) and leaves student/institution fields as placeholders.

Cleaning counts distinguish:

- **Before-cleaning tokens:** regex tokens after Unicode/casing, contraction expansion, and regex noise removal, before filler, repeat, and stop-word filtering.
- **After-cleaning tokens:** final normalized transcript tokens.
- **Token reduction (%):** \(100 \times (N_\text{before} - N_\text{after}) / N_\text{before}\).
- **Protected keywords preserved:** occurrences of protected words remaining after the cleaner.
- **Preprocessing time:** sum of per-stage measured durations, excluding optional morphology and report writing.

All timings are machine-, runtime-, and first-call-dependent. They are measured values, not performance guarantees.

## Results

Run `python main.py --evaluate` to regenerate all reports from the checked-in dataset. The CSV reports and the UI evaluation section are the source of the latest measured values; no benchmark timing is treated as a permanent or machine-independent result. Run the evaluation again after changing code, data, or dependencies. Missing WordNet or spaCy resources are reported as unavailable rather than filled with estimates. Vocabulary statistics depend on the benchmark vocabulary and should not be generalized to other transcripts. The controlled dataset includes regressions for meaningful “you know” queries, repeated protected commands, decimals, times, percentages, and Unicode words.

Execution times vary with machine, Python/dependency versions, initialization, and system load. The evaluation CSV records measured preprocessing, tokenization, stemming, and lemmatization times for each run. Ordinary CLI normalization skips optional morphology for lower startup cost; pass `--analyze` to calculate per-query stemmer and lemmatizer outputs. Controlled evaluation always executes all available comparisons. In Streamlit, enable the morphology checkbox for single-query analysis or use **Run controlled evaluation** to display method metrics, candidate errors, and aggregate cleaning counts directly from the evaluation backend.

**Example evaluation snapshot** (values read from the checked-in reports; rerun the command above for current measurements): the 30 controlled transcripts contained 152 tokens before cleanup and 120 after (21.052632% token reduction); 6 fillers, 24 stop words, and 1 ASR artifact were removed, and 37 protected-keyword occurrences were retained.

| Method | Unique vocabulary before → after | Vocabulary reduction | Reduction % | Retention % |
|---|---:|---:|---:|---:|
| Porter | 71 → 70 | 1 | 1.408451% | 98.591549% |
| Lancaster | 71 → 69 | 2 | 2.816901% | 97.183099% |
| WordNet | 71 → 70 | 1 | 1.408451% | 98.591549% |
| spaCy | 71 → 71 | 0 | 0.000000% | 100.000000% |

The error report contains three over-stemming candidates, including one manually curated unrelated pair that both methods conflate, and four under-stemming candidates from two curated related irregular pairs. NLTK's tokenizer differs from the regex tokenizer for the percent example because NLTK returns the percent sign as a separate token; the regex tokenizer intentionally preserves values such as `50%` as one semantic token. These controlled-set observations are not general accuracy claims.

## Limitations

- The controlled benchmark is small and authored for coverage, not a statistically representative dataset.
- Filler selection, politeness removal, and ASR artifact patterns are configurable heuristics; domain-specific review remains important.
- Contraction expansion is deterministic and cannot disambiguate every colloquial or dialectal use.
- Adjacent repetition cleanup can change emphasis; repeated protected commands, negation, and semantic numeric/time tokens are exempt, but repeated unprotected ordinary words are still collapsed.
- A protected-word set reduces accidental command loss but cannot guarantee semantic preservation for every command or language.
- POS-aware WordNet output depends on optional NLTK corpora/tagger data; spaCy output depends on the small model.
- Runtime measurements vary by host, dependency version, and warm/cold caches.
- The evaluation records that NLTK and the production regex tokenizer may differ on percentage tokens such as `50%`; the regex tokenizer deliberately treats the numeric percentage as one token.

## Audio input

Audio is supported as an optional front end. `src/transcription.py` uses faster-whisper to turn speech into a raw transcript, which is passed unchanged to the Problem 1 text pipeline. Cleaning, tokenization, stemming, and lemmatization are identical for typed and spoken input, and the benchmark and reports are unaffected.

- Supported formats: wav, mp3, m4a, flac, ogg, opus, webm, mp4, aac (decoding is bundled with faster-whisper, so ffmpeg is not required).
- Model sizes: tiny, base (default), small, medium, large-v3. Larger models are more accurate and slower. The chosen model is downloaded once on first use, so the first run needs an internet connection.
- Voice activity detection is enabled to avoid inventing words in silent gaps.
- Whisper adds its own casing and punctuation. The existing cleaner already removes these, so no extra step is needed.
- The transcript is shown before normalization so recognition errors can be told apart from cleaning behavior. Recognition quality depends on microphone, noise, accent, and model size; no accuracy figure is claimed.
- If faster-whisper or the model is missing, the CLI and app report it clearly. Text input keeps working without them.

## Future scope

Possible later work: streaming recognition, word-error-rate measurement against reference transcripts, and domain vocabulary prompts for the speech model.

## Installation and commands

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4 averaged_perceptron_tagger_eng
python -m spacy download en_core_web_sm
```

`requirements.txt` pins the verified Python libraries: NLTK 3.9.4, spaCy 3.8.16, Streamlit 1.59.2, and faster-whisper 1.2.1 (audio input only). The verified English model is `en_core_web_sm` 3.8.0. The model is distributed separately from the pinned Python packages, so `python -m spacy download en_core_web_sm` is required; `requirements.txt` alone does not install it.

Single transcript CLI:

```powershell
python main.py --text "Uh please turn OFF the living room light!!!"
```

Run the optional morphology branches on one transcript:

```powershell
python main.py --text "Uh please turn OFF the living room light!!!" --analyze
```

Audio file (one or more files; the transcript is printed, then normalized):

```powershell
python main.py --audio command.wav
python main.py --audio a.mp3 b.wav --audio-model small --analyze
python main.py --audio hindi.wav --language hi
python main.py --audio unknown.wav --language auto
```

Batch file (one transcript per non-empty line):

```powershell
python main.py --file data/raw_transcripts/sample.txt
```

Add `--analyze` to either text or file mode to run the optional morphology comparison. Without this flag, CLI normalization does not initialize WordNet tagging or the spaCy model. Punctuation-only input produces an explicit “no meaningful linguistic tokens” message and skips morphology.

Controlled evaluation and report generation:

```powershell
python main.py --evaluate
```

Run unit tests:

```powershell
python -m unittest discover -s tests -v
```

Launch the optional Streamlit demonstration:

```powershell
python -m streamlit run app.py
```

The app has three input types: Text, Audio file (upload), and Microphone (browser recording).
