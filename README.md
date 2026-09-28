# Meeting Audio Analysis System (Vietnamese)

Production-oriented pipeline that turns a Vietnamese meeting recording into an
analyzed, speaker-attributed transcript.

Pipeline: audio → ASR (`faster-whisper`) → speaker diarization (`pyannote.audio`)
→ per-speaker gender → per-utterance emotion → structured transcript.

## Output contract

Each utterance renders as:

```
[<Gender> <SpeakerLetter> - <Emotion>]: "<text>"
```

Example:

```
[Nam A - Vui vẻ]: "Em làm phần này xong rồi nhé."
[Nữ B - Nóng giận]: "Tại sao tiến độ lại chậm như vậy?"
```

Labels:
- Gender: `Nam` (male) / `Nữ` (female)
- Speaker: `A`, `B`, `C`, … (detected, never hardcoded)
- Emotion: `Vui vẻ` / `Nóng giận` / `Bình thường`

Every run emits **both** a human-readable `.txt` and a machine-readable `.json`
array of `{start, end, speaker, gender, emotion, text}`, produced from the same
in-memory result.

## Status

**Phase 0 — scaffolding + environment verification.** No pipeline logic yet.

## Stack (locked)

| Component | Choice |
|---|---|
| ASR | `faster-whisper` (`language="vi"`, `int8_float16`) |
| Diarization | `pyannote.audio` `speaker-diarization-3.1` |
| Gender | per-speaker audio classifier + F0 cross-check |
| Emotion | `wonrax/phobert-base-vietnamese-sentiment` (text) |
| GPU | NVIDIA RTX 3050 Laptop, **4 GB VRAM** |

## Setup (coming in later phases)

Full install, HF token, and run instructions will be documented once the
pipeline is implemented.
