"""Generate spoken test commands, transcribe them, and check the normalized output."""

from pathlib import Path

import pyttsx3

from src.pipeline import run_pipeline
from src.transcription import transcribe_audio

CASES = [
    ("Uh please turn off the living room light", "turn off living room light"),
    ("Don't turn off the fan", "do not turn off fan"),
    ("Set an alarm for 7:30", "set alarm for 7:30"),
    ("Please play the next song", "play next song"),
    ("Set the temperature to 22 degrees", "set temperature to 22 degrees"),
]

out_dir = Path("data/test_audio")
out_dir.mkdir(parents=True, exist_ok=True)

engine = pyttsx3.init()
engine.setProperty("rate", 160)
for index, (spoken, _) in enumerate(CASES, start=1):
    engine.save_to_file(spoken, str(out_dir / f"case{index}.wav"))
engine.runAndWait()

passed = 0
for index, (spoken, expected) in enumerate(CASES, start=1):
    transcription = transcribe_audio(out_dir / f"case{index}.wav", model_size="base")
    cleaned = run_pipeline(transcription.text, include_morphology=False)["cleaning"].normalized_text
    ok = cleaned == expected
    passed += ok
    print(f"[{'PASS' if ok else 'FAIL'}] case {index}")
    print(f"  spoken:     {spoken}")
    print(f"  recognized: {transcription.text}")
    print(f"  normalized: {cleaned}")
    print(f"  expected:   {expected}")
print(f"\n{passed}/{len(CASES)} passed")