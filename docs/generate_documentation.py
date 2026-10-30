"""Build the Problem 1 technical report from the current generated reports."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ASR_ARTIFACTS,
    DOMAIN_STOP_WORDS,
    GENERAL_STOP_WORDS,
    PROTECTED_COMMAND_WORDS,
    VOICE_FILLER_PHRASES,
    VOICE_FILLER_WORDS,
)
from src.evaluation import CURATED_PAIRS, DATASET_PATH, PROJECT_ROOT
from src.pipeline import run_pipeline

REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "Architecture_and_Methodology.pdf"

INK = colors.HexColor("#172B3A")
BLUE = colors.HexColor("#1E5D78")
TEAL = colors.HexColor("#DDEEF0")
PALE = colors.HexColor("#F2F6F8")
RULE = colors.HexColor("#C7D4DC")
MUTED = colors.HexColor("#506574")
WHITE = colors.white


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def parse_summary(rows: list[dict[str, str]]) -> dict[str, str]:
    return {row["metric"]: row["value"] for row in rows}


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=27,
        leading=33,
        textColor=INK,
        alignment=TA_CENTER,
        spaceAfter=16,
    ))
    styles.add(ParagraphStyle(
        name="Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=14,
        leading=20,
        textColor=BLUE,
        alignment=TA_CENTER,
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="Section",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=INK,
        spaceBefore=8,
        spaceAfter=9,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="Subsection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=BLUE,
        spaceBefore=8,
        spaceAfter=5,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="BodyReport",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=12.4,
        textColor=INK,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="SmallReport",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=INK,
        spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="TableHead",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9,
        textColor=WHITE,
    ))
    styles.add(ParagraphStyle(
        name="TableCell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.1,
        leading=9,
        textColor=INK,
        splitLongWords=1,
    ))
    styles.add(ParagraphStyle(
        name="CodeBlock",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7.3,
        leading=10,
        textColor=INK,
        backColor=PALE,
        borderColor=RULE,
        borderWidth=0.5,
        borderPadding=7,
        spaceBefore=4,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="CoverField",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=21,
        textColor=INK,
        leftIndent=100,
    ))
    styles.add(ParagraphStyle(
        name="Callout",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=BLUE,
        backColor=TEAL,
        borderColor=BLUE,
        borderWidth=0.6,
        borderPadding=8,
        spaceBefore=5,
        spaceAfter=9,
    ))
    return styles


def p(text: str, style, *, escape: bool = False) -> Paragraph:
    return Paragraph(text, style)


def make_table(rows, widths, *, header=True, font_size=7.2, repeat_rows=1):
    table = Table(rows, colWidths=widths, repeatRows=repeat_rows if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ])
        if len(rows) > 1:
            commands.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]))
    table.setStyle(TableStyle(commands))
    return table


def header_row(labels, styles):
    return [Paragraph(label, styles["TableHead"]) for label in labels]


def cell(value, styles):
    return Paragraph(str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["TableCell"])


def architecture_diagram(styles):
    stages = [
        "Raw speech-to-text<br/>transcript",
        "Validation",
        "Unicode +<br/>lowercase",
        "Contraction<br/>expansion",
        "Regex noise<br/>removal",
        "Regex<br/>tokenization",
        "Filler<br/>handling",
        "Protected keyword<br/>detection",
        "Stop-word<br/>filtering",
        "Normalized<br/>query",
    ]
    widths = [750 / len(stages)] * len(stages)
    stage_cells = [
        Paragraph(stage, ParagraphStyle(
            "ArchitectureCell",
            parent=styles["TableCell"],
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            fontSize=7.1,
            leading=9,
            textColor=INK,
        ))
        for stage in stages
    ]
    rows = [stage_cells, [Paragraph("&gt;", styles["Subtitle"]) for _ in stages]]
    table = Table(rows, colWidths=widths, rowHeights=[43, 17])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("BOX", (0, 0), (-1, 0), 0.8, BLUE),
        ("INNERGRID", (0, 0), (-1, 0), 0.4, RULE),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TEXTCOLOR", (0, 1), (-1, 1), BLUE),
        ("FONTSIZE", (0, 1), (-1, 1), 12),
    ]))
    branch = Table(
        [[
            Paragraph("<b>Stemming</b><br/>Porter · Lancaster", styles["TableCell"]),
            Paragraph("<b>Lemmatization</b><br/>WordNet · spaCy", styles["TableCell"]),
        ]],
        colWidths=[3.75 * inch, 3.75 * inch],
        hAlign="CENTER",
    )
    branch.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.7, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, RULE),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return [table, Spacer(1, 4), Paragraph("Morphology is an analysis branch: it does not replace or mutate the normalized command.", styles["SmallReport"]), Spacer(1, 4), branch]


def build_story(comparison, summary, errors, preprocessing, styles):
    story = []
    body = styles["BodyReport"]
    small = styles["SmallReport"]
    section = styles["Section"]
    subsection = styles["Subsection"]
    method_results = {row["method"]: row for row in comparison}
    porter = method_results["porter"]
    lancaster = method_results["lancaster"]
    wordnet = method_results["wordnet"]
    spacy = method_results["spacy"]

    story.extend([
        Spacer(1, 0.55 * inch),
        Paragraph("Voice Assistant Query<br/>Normalizer and Text Cleaner", styles["ReportTitle"]),
        Paragraph("MT-AIML Mini Project — Problem 1", styles["Subtitle"]),
        HRFlowable(width="56%", thickness=2, color=BLUE, hAlign="CENTER", spaceBefore=8, spaceAfter=24),
        Paragraph("<b>Student Name:</b> __________________________________________", styles["CoverField"]),
        Paragraph("<b>Enrollment Number:</b> _____________________________________", styles["CoverField"]),
        Paragraph("<b>Course:</b> MT-AIML", styles["CoverField"]),
        Paragraph("<b>Semester:</b> ______________________________________________", styles["CoverField"]),
        Paragraph("<b>Academic Year:</b> _________________________________________", styles["CoverField"]),
        Paragraph("<b>Institution:</b> _____________________________________________", styles["CoverField"]),
        Paragraph("<b>Submission Date:</b> ______________________________________", styles["CoverField"]),
        Spacer(1, 0.4 * inch),
        Paragraph("Technical Documentation: Architecture, Methodology, and Controlled Evaluation", styles["Subtitle"]),
        PageBreak(),
    ])

    story += [
        Paragraph("1. Problem Statement", section),
        Paragraph(
            "Voice assistants commonly convert spoken requests into text before downstream "
            "command processing. Such raw speech-to-text transcripts may contain spoken fillers, "
            "acoustic or ASR annotation noise, contractions, casing variation, repeated words, "
            "punctuation, and function words that add little command value. Problem 1 addresses "
            "the deterministic NLP preprocessing of that transcript, including tokenization, "
            "stop-word handling, stemming/lemmatization comparison, and command-term preservation.",
            body,
        ),
        Paragraph("<b>Input:</b> a raw speech-to-text transcript supplied as text.", styles["Callout"]),
        Paragraph("<b>Output:</b> normalized text intended for downstream voice-command processing. "
                  "It is not a recognized audio signal and is not an execution/intent-classification result.", styles["Callout"]),
        Paragraph("2. Objective", section),
        Paragraph(
            "The project cleans raw voice-assistant transcripts; normalizes casing and configured "
            "contractions; removes selected fillers, ASR tags, and stop words; preserves protected "
            "command and negation terms; compares Porter and Lancaster stemming with WordNet and "
            "spaCy lemmatization; quantifies unique-vocabulary changes; analyzes stemming-error "
            "candidates; and supplies a runnable CLI, evaluation workflow, and demonstration UI.",
            body,
        ),
        Paragraph("3. Scope", section),
        Paragraph(
            "The system starts with a transcript string or UTF-8 text file. The current Problem 1 "
            "implementation performs text preprocessing and controlled NLP comparison only. It "
            "does <b>not</b> record audio, decode audio files, perform speech recognition, or use "
            "Whisper, microphone input, or cloud speech services. A speech-to-text component is "
            "outside the current scope; audio integration is listed only as optional future work.",
            body,
        ),
        Paragraph("4. System Architecture", section),
        *architecture_diagram(styles),
        Spacer(1, 6),
        Paragraph(
            "Implementation modules: <font name='Courier'>main.py</font> (CLI), "
            "<font name='Courier'>app.py</font> (Streamlit demo), "
            "<font name='Courier'>src/preprocessing.py</font> (cleaner), "
            "<font name='Courier'>src/pipeline.py</font> (fast normalization/optional morphology), "
            "<font name='Courier'>src/stemmer_comparison.py</font>, "
            "<font name='Courier'>src/lemmatizer_comparison.py</font>, and "
            "<font name='Courier'>src/evaluation.py</font> (controlled dataset and reports).",
            small,
        ),
        PageBreak(),
    ]

    story += [
        Paragraph("5. Preprocessing Methodology", section),
        Paragraph(
            "Each preprocessing stage is deterministic and timed separately in "
            "<font name='Courier'>clean_transcript()</font>. Its normalized output is the "
            "canonical query. Comparative stemming and lemmatization do not alter this output.",
            body,
        ),
        Paragraph("5.1 Unicode and casing normalization", subsection),
        Paragraph(
            "Input is normalized with Python's Unicode NFKC normalization; curly apostrophe "
            "characters are mapped to the straight apostrophe used by the contraction rules. "
            "Text is lowercased. The compiled word-token expression uses Unicode-aware letter "
            "matching so words such as café, naïve, résumé, and München remain intact.",
            body,
        ),
        Paragraph("5.2 Contraction handling", subsection),
        Paragraph(
            "A deterministic dictionary in <font name='Courier'>src/config.py</font> expands "
            "configured English contractions using a compiled case-insensitive regex. For example, "
            "<font name='Courier'>don't</font> becomes <font name='Courier'>do not</font>. "
            "Keeping the negator matters because removing “not” could reverse the intended command. "
            "Ambiguous possessives are intentionally not treated as contractions by a general rule.",
            body,
        ),
        Paragraph("5.3 Regex-based noise removal", subsection),
        Paragraph(
            "The cleaner removes configured bracketed/angle-bracketed ASR annotations from "
            f"<font name='Courier'>{', '.join(sorted(ASR_ARTIFACTS))}</font>, collapses repeated "
            "runs of selected punctuation, replaces unsupported non-linguistic symbols with spaces, "
            "and normalizes whitespace. These operations do not indiscriminately delete digits. "
            "The token pattern separately recognizes signed numeric values, decimals, clock times, "
            "and percentages. Arbitrary unbracketed words are not assumed to be ASR noise.",
            body,
        ),
        Paragraph("5.4 Filler handling", subsection),
        Paragraph(
            "Default single-token filler vocabulary is "
            f"<font name='Courier'>{', '.join(sorted(VOICE_FILLER_WORDS))}</font>. Prolonged "
            "forms matching the explicit filler pattern (for example, “uhhh” and “ummm”) are "
            "counted as removed fillers. The default phrase list is empty; “you know” is not "
            "removed unconditionally because it can be meaningful. Filler words and phrases can "
            "be configured without rewriting the cleaning pipeline.",
            body,
        ),
        Paragraph("5.5 Tokenization", subsection),
        Paragraph(
            "The production tokenizer is a precompiled Unicode-aware Python regular expression. "
            "It emits lowercased word tokens and preserves numeric tokens such as 22, 22.5, 7:30, "
            "50%, and 204 as applicable. An NLTK "
            "<font name='Courier'>word_tokenize(..., preserve_line=True)</font> alternative is "
            "timed in the evaluation. These tokenizers can differ on percent signs: the production "
            "regex retains 50% as one semantic numeric token, while NLTK may split it into 50 and %.",
            body,
        ),
        Paragraph("5.6 Stop words and command vocabulary", subsection),
        Paragraph(
            "The general stop set is "
            f"<font name='Courier'>{', '.join(sorted(GENERAL_STOP_WORDS))}</font>; the domain "
            "politeness set is "
            f"<font name='Courier'>{', '.join(sorted(DOMAIN_STOP_WORDS))}</font>. These are "
            "intentionally small and are not the NLTK English stop list. Fillers are a separate "
            "category and are counted separately. Stop-word removal runs only after protected "
            "command words have been identified.",
            body,
        ),
        Paragraph("5.7 Protected command keywords", subsection),
        Paragraph(
            "Configured protected vocabulary is "
            f"<font name='Courier'>{', '.join(sorted(PROTECTED_COMMAND_WORDS))}</font>. Words "
            "such as on/off, open/close, play/pause, next/previous, increase/decrease, and "
            "enable/disable can encode the requested action or state. The cleaner exempts these "
            "tokens from stop-word removal and adjacent-duplicate collapse. Negation words are "
            "protected as well. This is a safety heuristic, not a proof of semantic correctness for "
            "every possible utterance.",
            body,
        ),
        Paragraph("5.8 Repeated words and empty output", subsection),
        Paragraph(
            "Adjacent repeated ordinary words are collapsed, while repeated protected commands, "
            "negations, and recognized numeric/time values are retained. Punctuation-only input "
            "can yield no linguistic tokens; the pipeline reports an empty normalized result and "
            "skips optional morphology rather than presenting an empty command as usable output.",
            body,
        ),
        PageBreak(),
    ]

    story += [
        Paragraph("6. Stemming", section),
        Paragraph(
            "Stemming applies rule-based suffix transformations to reduce related surface forms "
            "to a common stem. A stem can be truncated or not be a dictionary lemma. Here both "
            "stemmer branches receive the same normalized token sequence from the evaluation corpus.",
            body,
        ),
        Paragraph("6.1 Porter Stemmer", subsection),
        Paragraph(
            "The implementation uses NLTK's <font name='Courier'>PorterStemmer</font>. On this "
            f"controlled vocabulary it changed {porter['unique_transformations']} distinct word "
            f"types and reduced the unique vocabulary by {porter['vocabulary_reduction']} term(s). "
            "Stemming is deterministic and lightweight, but can produce "
            "non-lemmas and conflations; this single benchmark does not establish general accuracy "
            "or superiority.",
            body,
        ),
        Paragraph("6.2 Lancaster Stemmer", subsection),
        Paragraph(
            "The implementation uses NLTK's <font name='Courier'>LancasterStemmer</font>, an "
            f"iterative rule-based stemmer. On this run it changed "
            f"{lancaster['unique_transformations']} distinct types and reduced the unique "
            f"vocabulary by {lancaster['vocabulary_reduction']} term(s). Its transformations can be more aggressive on "
            "some words; a larger reduction does not mean better semantic preservation. Neither "
            "stemmer is universally preferable based on these measurements.",
            body,
        ),
        Paragraph("7. Lemmatization", section),
        Paragraph(
            "Lemmatization maps an inflected form toward a lexical lemma, using lexical and/or "
            "linguistic information rather than only trimming suffixes. It is still a model/tool "
            "output and is not a semantic command parser.",
            body,
        ),
        Paragraph("7.1 WordNet Lemmatization", subsection),
        Paragraph(
            "The WordNet branch uses NLTK's <font name='Courier'>WordNetLemmatizer</font>. "
            "Penn Treebank POS tags are mapped to WordNet POS when the tagger resource is "
            "available; otherwise the implementation uses its documented lightweight suffix "
            f"heuristic and noun fallback. In the current controlled run, its unique vocabulary "
            f"reduced by {wordnet['vocabulary_reduction']} term(s). The example table shows its "
            "measured output for the temperature command; this is morphological output, not "
            "command execution.",
            body,
        ),
        Paragraph("7.2 spaCy Lemmatization", subsection),
        Paragraph(
            "The spaCy branch loads <font name='Courier'>en_core_web_sm</font> once per process, "
            "disabling parser and named-entity components, then processes the normalized tokens. "
            f"The evaluation run retained {spacy['resulting_vocabulary_size']} unique terms for "
            "this particular benchmark. "
            "This reflects this dataset and model output only; it does not establish that spaCy "
            "always retains vocabulary or is semantically more accurate.",
            body,
        ),
        Paragraph("8. Vocabulary Metrics", section),
        Paragraph(
            "<b>Original vocabulary</b> is the number of unique normalized terms before the "
            "morphology transformation. <b>Resulting vocabulary</b> is the number of unique terms "
            "after the transformation. Duplicate token occurrences do not increase either size.",
            body,
        ),
        Paragraph(
            "<b>Vocabulary reduction</b> = Original Vocabulary − Resulting Vocabulary<br/>"
            "<b>Vocabulary reduction (%)</b> = ((Original Vocabulary − Resulting Vocabulary) / "
            "Original Vocabulary) × 100<br/>"
            "<b>Vocabulary retention (%)</b> = (Resulting Vocabulary / Original Vocabulary) × 100",
            styles["Callout"],
        ),
        Paragraph(
            "Reduction and retention are complementary for non-empty input vocabularies. They "
            "quantify unique-type compression, not semantic accuracy, command correctness, or "
            "general NLP quality. For an empty original vocabulary, the implementation reports "
            "0.0% for both rates to avoid division by zero and does not imply a meaningful ratio.",
            body,
        ),
        PageBreak(),
    ]

    comparison_rows = [header_row([
        "Method", "Original<br/>Vocabulary", "Resulting<br/>Vocabulary",
        "Reduction", "Reduction %", "Retention %", "Execution Time (s)",
    ], styles)]
    for row in comparison:
        comparison_rows.append([
            cell(row["method"].title(), styles),
            cell(row["original_vocabulary_size"], styles),
            cell(row["resulting_vocabulary_size"], styles),
            cell(row["vocabulary_reduction"], styles),
            cell(f'{float(row["vocabulary_reduction_percent"]):.6f}%', styles),
            cell(f'{float(row["vocabulary_retention_percent"]):.6f}%', styles),
            cell(f'{float(row["execution_time_seconds"]):.9f}', styles),
        ])

    story += [
        Paragraph("9. Comparative Analysis — Latest Generated Evaluation", section),
        Paragraph(
            "The following values are read directly from "
            "<font name='Courier'>reports/stemming_lemmatization_comparison.csv</font> "
            "after the latest evaluation run used to generate this document. Execution times "
            "are observed timings from this environment and are not fixed performance guarantees.",
            body,
        ),
        make_table(comparison_rows, [1.15*inch, 1.22*inch, 1.25*inch, 0.9*inch, 1.0*inch, 1.0*inch, 1.25*inch]),
        Spacer(1, 10),
        Paragraph("10. Stemming Error Analysis", section),
        Paragraph("10.1 Over-stemming", subsection),
        Paragraph(
            "Over-stemming occurs when distinct words that should remain distinct are mapped to "
            "the same stem. The evaluation detects candidates by grouping distinct normalized "
            "benchmark word types by their actual stem. A collision is marked as a manually "
            "verified unrelated pair only if it matches the manually curated contrast "
            "university/universal and the method actually conflates that pair.",
            body,
        ),
    ]
    over_rows = [row for row in errors if row["error_type"] == "over-stemming_candidate"]
    if over_rows:
        error_table = [header_row(["Method", "Word A", "Word B", "Shared Stem", "Evidence", "Manual Relation", "Verified Error?"], styles)]
        for row in over_rows:
            error_table.append([
                cell(row["method"].title(), styles),
                cell(row["word_a"], styles),
                cell(row["word_b"], styles),
                cell(row["shared_stem"], styles),
                cell(row["basis"], styles),
                cell(row["manual_relation"], styles),
                cell(row["manually_verified_error"], styles),
            ])
        story.append(make_table(error_table, [1.0*inch, 1.1*inch, 1.1*inch, 1.0*inch, 1.7*inch, 1.45*inch, 1.0*inch]))
    story.extend([
        Paragraph(
            "The “university” / “universal” pair is a curated unrelated contrast; the report "
            "records a verified conflation only when observed for that stemmer. Automatic "
            "collisions such as “time” / “timer” remain unreviewed candidates, not confirmed "
            "semantic errors. Candidate counts depend on benchmark vocabulary.",
            small,
        ),
        Paragraph("10.2 Under-stemming", subsection),
        Paragraph(
            "Under-stemming candidates arise when a manually curated related pair remains "
            "different after stemming. The current candidate set uses the irregular pairs "
            "goose/geese and mouse/mice. Their actual stems and method are read from "
            "<font name='Courier'>reports/error_analysis.csv</font>. Different stems are "
            "reported as candidates; they are not automatically labeled errors because the "
            "intended grouping depends on the application.",
            body,
        ),
    ])
    under_rows = [row for row in errors if row["error_type"] == "under-stemming_candidate"]
    under_table = [header_row(["Method", "Word A", "Word B", "Distinct Stems", "Curated Relation", "Status"], styles)]
    for row in under_rows:
        under_table.append([
            cell(row["method"].title(), styles),
            cell(row["word_a"], styles),
            cell(row["word_b"], styles),
            cell(row.get("distinct_stems", ""), styles),
            cell(row["manual_relation"], styles),
            cell("Candidate", styles),
        ])
    if len(under_table) > 1:
        story.append(make_table(under_table, [1.0*inch, 1.05*inch, 1.05*inch, 1.5*inch, 1.8*inch, 1.0*inch]))
    story.append(PageBreak())

    examples = [
        "Uh PLEASE turn OFF the living room light!!!",
        "don't turn off the fan",
        "set temperature to 22.5 degrees",
        "set an alarm for 7:30",
        "open the café door in München",
        "do you know the time?",
    ]
    story += [
        Paragraph("11. Example Transformations", section),
        Paragraph(
            "Each row below is generated by executing the current pipeline with optional "
            "morphology enabled. Output is shown as token sequences for the four comparison "
            "methods; unavailable resources would be labelled rather than synthesized.",
            body,
        ),
    ]
    example_rows = [header_row([
        "Raw Transcript", "Normalized Transcript", "Porter", "Lancaster", "WordNet", "spaCy",
    ], styles)]
    for text in examples:
        result = run_pipeline(text, include_morphology=True)
        cleaning = result["cleaning"]
        def morph_tokens(group, name):
            output = group[name]
            return " ".join(output["tokens"]) if output.get("available", True) else "Unavailable"
        example_rows.append([
            cell(text, styles),
            cell(cleaning.normalized_text, styles),
            cell(" ".join(result["stemmers"]["porter"]["tokens"]), styles),
            cell(" ".join(result["stemmers"]["lancaster"]["tokens"]), styles),
            cell(morph_tokens(result["lemmatizers"], "wordnet"), styles),
            cell(morph_tokens(result["lemmatizers"], "spacy"), styles),
        ])
    story += [
        make_table(example_rows, [1.75*inch, 1.65*inch, 1.28*inch, 1.28*inch, 1.28*inch, 1.28*inch]),
        Spacer(1, 7),
        Paragraph("12. Semantic Preservation", section),
        Paragraph(
            "Regression tests and pipeline runs cover turning a light on/off, preserving "
            "“do not” after contraction expansion, retaining command verbs, numbers including "
            "22.5, and the clock token 7:30. The default cleaner removes articles and configured "
            "politeness terms but does not remove on/off, protected command verbs, or negation. "
            "Unicode examples preserve café and München in the normalized output. These are "
            "specific tested cases, not a guarantee of universal semantic preservation.",
            body,
        ),
        Paragraph(
            "Command keywords encode action and polarity; numbers, times, identifiers, and names "
            "can identify the object or parameter. Stemming and lemmatization are shown as "
            "analysis outputs only and must not be substituted for the canonical normalized "
            "command without application-specific review.",
            body,
        ),
        PageBreak(),
    ]

    dataset_rows = load_csv(DATASET_PATH)
    story += [
        Paragraph("13. Evaluation Dataset", section),
        Paragraph(
            f"The controlled, manually assembled dataset is "
            f"<font name='Courier'>data/evaluation/benchmark.csv</font> and contains "
            f"<b>{len(dataset_rows)} transcripts</b>. It covers command forms, fillers, casing, "
            "punctuation, contraction/negation, repeated terms, integer/decimal/time/percentage "
            "values, ASR tags, Unicode vocabulary, and curated morphology pairs. It is a "
            "reproducible coverage fixture, not a representative sample of Alexa/Siri or all "
            "real-world voice interactions.",
            body,
        ),
        Paragraph("14. Performance and Cleaning Statistics", section),
    ]
    statistic_rows = [header_row(["Measured item", "Latest report value", "Definition"], styles)]
    stats = [
        ("Input transcripts", "total_input_transcripts", "Number of controlled dataset rows."),
        ("Tokens before cleaning", "tokens_before_cleaning", "After Unicode/contraction/noise processing; before filler/repeat/stop filtering."),
        ("Tokens after cleaning", "tokens_after_cleaning", "Final normalized tokens across all rows."),
        ("Token reduction (%)", "token_reduction_percent", "100 × (before − after) / before."),
        ("Fillers removed", "filler_words_removed", "Configured filler occurrences."),
        ("Stop words removed", "stop_words_removed", "General/domain stop-word occurrences."),
        ("ASR noise tags removed", "noise_tokens_removed", "Recognized bracketed artifact count."),
        ("Protected terms retained", "protected_keywords_preserved", "Protected keyword occurrences in cleaned output."),
        ("Preprocessing time (s)", "preprocessing_execution_time_seconds", "Sum of measured cleaner stage timings; morphology/reporting excluded."),
        ("Mean preprocessing time (s/transcript)", "average_preprocessing_execution_time_seconds", "Total stage time / transcript count."),
        ("Regex tokenization time (s)", "regex_tokenizer_execution_time_seconds", "Aggregate production-tokenizer timing."),
        ("NLTK tokenization time (s)", "nltk_tokenizer_execution_time_seconds", "Aggregate comparison timing; first-call setup may be included."),
        ("Tokenizer outputs equivalent", "tokenizer_outputs_equivalent", "Whether all token sequences matched."),
    ]
    for label, key, definition in stats:
        value = summary.get(key, "not reported")
        if key.endswith("_seconds") and value != "not reported":
            value = f"{float(value):.9f}"
        statistic_rows.append([cell(label, styles), cell(value, styles), cell(definition, styles)])
    story += [
        make_table(statistic_rows, [2.1*inch, 1.5*inch, 5.4*inch]),
        Spacer(1, 8),
        Paragraph(
            "Morphology timings cover the transformation passes after WordNet resources and the "
            "spaCy model are loaded. Overall process startup and model loading are separate costs. "
            "All timings depend on CPU, Python, dependency versions, initialization state, and "
            "system load; they are measurements of this run, not guarantees.",
            small,
        ),
        Paragraph("15. Limitations", section),
        Paragraph(
            "The controlled benchmark is small and curated. Regex, contraction, filler, stop-word, "
            "and duplicate rules are deterministic heuristics and may not fit every domain or "
            "dialect. Repeated ordinary words can sometimes encode emphasis. A protected vocabulary "
            "reduces specific risks but cannot guarantee intent preservation in arbitrary text. "
            "Stems can be malformed or conflate meanings; WordNet POS tagging depends on NLTK "
            "resources; spaCy lemmatization depends on its separately installed English model. "
            "No gold-standard semantic annotations are provided, so the report does not claim "
            "semantic accuracy. Upstream ASR errors and audio handling are out of scope.",
            body,
        ),
        Paragraph("16. Conclusion", section),
        Paragraph(
            "The implementation provides a runnable deterministic transcript cleaner, keeps "
            "command, negation, and tested numeric forms in the normalized output, and generates "
            "reproducible controlled comparisons for Porter, Lancaster, WordNet, and spaCy. "
            "Evaluation measures vocabulary size changes and timed transformation passes; it "
            "does not establish real-world assistant accuracy or perfect semantic safety.",
            body,
        ),
        Paragraph("17. Future Scope", section),
        Paragraph(
            "Potential follow-on work includes broader manually annotated text benchmarks, "
            "domain-specific review of filler and stop-word policies, multilingual evaluation, "
            "and explicit downstream command-semantic tests. Optional audio integration would "
            "be a separate upstream extension, not part of this submission:",
            body,
        ),
        Paragraph("Audio (future extension) &gt; Speech-to-Text (future component) &gt; Problem 1 text preprocessing", styles["Callout"]),
        Paragraph("18. Reproducibility and Execution", section),
        Paragraph(
            "Install Python dependencies with <font name='Courier'>python -m pip install -r "
            "requirements.txt</font>; install NLTK WordNet/tagger data with "
            "<font name='Courier'>python -m nltk.downloader wordnet omw-1.4 "
            "averaged_perceptron_tagger_eng</font>; and install spaCy's model with "
            "<font name='Courier'>python -m spacy download en_core_web_sm</font>. Run a single "
            "transcript with <font name='Courier'>python main.py --text \"...\"</font>, an input "
            "file with <font name='Courier'>python main.py --file "
            "data/raw_transcripts/sample.txt</font>, the controlled benchmark with "
            "<font name='Courier'>python main.py --evaluate</font>, tests with "
            "<font name='Courier'>python -m unittest discover -s tests -v</font>, and the UI "
            "with <font name='Courier'>python -m streamlit run app.py</font>.",
            body,
        ),
    ]
    return story


def add_page_chrome(canvas, document):
    canvas.saveState()
    width, height = landscape(letter)
    if document.page > 1:
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(document.leftMargin, height - 0.42 * inch, width - document.rightMargin, height - 0.42 * inch)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(document.leftMargin, height - 0.32 * inch, "MT-AIML · Problem 1 · Technical Methodology")
        canvas.drawRightString(width - document.rightMargin, 0.28 * inch, f"Page {document.page}")
    canvas.restoreState()


def main():
    comparison = load_csv(REPORTS_DIR / "stemming_lemmatization_comparison.csv")
    errors = load_csv(REPORTS_DIR / "error_analysis.csv")
    summary = parse_summary(load_csv(REPORTS_DIR / "evaluation_summary.csv"))
    preprocessing = load_csv(REPORTS_DIR / "preprocessing_report.csv")
    styles = make_styles()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=landscape(letter),
        rightMargin=0.42 * inch,
        leftMargin=0.42 * inch,
        topMargin=0.58 * inch,
        bottomMargin=0.48 * inch,
        title="Voice Assistant Query Normalizer and Text Cleaner — Architecture and Methodology",
        author="MT-AIML Mini Project",
        subject="Problem 1 technical documentation",
    )
    doc.build(
        build_story(comparison, summary, errors, preprocessing, styles),
        onFirstPage=add_page_chrome,
        onLaterPages=add_page_chrome,
    )
    print(f"Generated {OUTPUT_PATH}")
    print(f"Size: {OUTPUT_PATH.stat().st_size} bytes")


if __name__ == "__main__":
    main()
