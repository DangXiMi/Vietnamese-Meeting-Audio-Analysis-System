# Vietnamese Meeting Audio Analysis System

An end-to-end speech analytics pipeline that converts a Vietnamese meeting
recording into a **speaker-attributed, gender-labelled, emotion-annotated
transcript**, delivered as both a human-readable text file and a
machine-readable JSON array.

Built for **fully local inference** on consumer hardware (NVIDIA RTX 3050 Laptop,
4 GB VRAM) — no cloud APIs, no data leaves the machine.

---

## Table of contents

- [What it does](#what-it-does)
- [Output contract](#output-contract)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Command-line reference](#command-line-reference)
- [Web interface](#web-interface)
- [Project layout](#project-layout)
- [Testing](#testing)
- [Measured performance](#measured-performance)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)

---

## What it does

| Stage | Description |
|---|---|
| **1. Normalize** | Probes the input and converts any container/codec to one canonical 16 kHz mono WAV, so every downstream stage shares an identical time base. |
| **2. Diarize** | Determines *who spoke when* using pyannote.audio. |
| **3. Transcribe** | Produces Vietnamese speech-to-text with **word-level** timestamps using faster-whisper. |
| **4. Align** | Merges word timings onto speaker turns to attribute every utterance to a speaker. |
| **5. Gender** | Classifies gender **once per speaker** from that speaker's pooled audio, cross-checked against median pitch (F0). |
| **6. Emotion** | Classifies emotion **per utterance** from Vietnamese text. |
| **7. Emit** | Writes a `.txt` transcript and a matching `.json` array from a single in-memory result. |

Every model-backed stage sits behind a protocol, so any component can be
replaced (larger model, cloud API, custom diarizer) without touching the rest of
the system.

---

## Output contract

Each utterance renders as:

```
[<Gender> <SpeakerLetter> - <Emotion>]: "<text>"
```

**`<name>.txt`**

```
[Nam A - Vui vẻ]: "Chào cô"
[Nữ B - Bình thường]: "Chào anh Cho hỏi, anh tên là gì?"
[Nam A - Bình thường]: "Tôi tên là Mike Cô tên là gì?"
```

**`<name>.json`**

```json
[
  {
    "start": 0.4,
    "end": 2.31,
    "speaker": "A",
    "gender": "Nam",
    "emotion": "Vui vẻ",
    "text": "Chào cô"
  }
]
```

**Label set**

| Field | Allowed values |
|---|---|
| `gender` | `Nam` (male) · `Nữ` (female) |
| `speaker` | `A`, `B`, `C`, … — **detected at runtime, never hardcoded** |
| `emotion` | `Vui vẻ` (happy) · `Nóng giận` (angry) · `Bình thường` (neutral) |

Both artifacts are produced from the **same in-memory result**, so they always
agree on segment count, order, and content. The `.txt` is never parsed back to
build the `.json`.

---

## Quick start

### Prerequisites

| Requirement | Notes |
|---|---|
| Python | 3.11 |
| FFmpeg | **Required system dependency** — see step 1 |
| NVIDIA GPU | Optional but strongly recommended (CPU fallback is automatic) |
| Hugging Face account | Required — the pyannote models are gated |

### 1. Install FFmpeg

```powershell
# Windows
winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
```

Restart your shell afterwards so `PATH` is refreshed, then verify:

```powershell
ffmpeg -version
ffprobe -version
```

For Docker/Linux, install FFmpeg from your distribution or base image instead.
FFmpeg is never bundled into this repository.

### 2. Accept the gated model licences

Log in to Hugging Face and accept the terms on **both** pages:

- <https://huggingface.co/pyannote/speaker-diarization-community-1>
- <https://huggingface.co/pyannote/segmentation-3.0>

> Accepting only one is the classic cause of a confusing `403 GatedRepoError`.

### 3. Configure credentials

```powershell
Copy-Item .env.example .env
```

Set your token in `.env` (the file is gitignored):

```ini
HF_TOKEN=hf_...
```

Both `HF_TOKEN` and `hf_token` are accepted.

### 4. Create a Python Virtual Environment
#### 1. Create the virtual environment
```bash
python -m venv .venv
```

#### 2. Activate the virtual environment

PowerShell:
```bash
.\.venv\Scripts\Activate.ps1
```
Command Prompt (CMD):
```bash
.venv\Scripts\activate
```

#### 3. Verify the environment
```bash
python --version
pip --version
```

### 5. Install Python dependencies

Install PyTorch **first**, choosing the build that matches your hardware. The
project runs on CPU as well as GPU, but diarization is far slower without CUDA:

```powershell
# NVIDIA GPU — match the CUDA index to your driver
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# ...or CPU only
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Then the rest:

```powershell
pip install -r requirements.txt
```

> **`constraints.txt` is machine-specific.** It pins the exact CUDA build of the
> development machine (`torch==2.11.0+cu126`), which is **not on PyPI**, and
> exists only to stop pip replacing an already-working CUDA install. Do **not**
> pass it on a fresh machine.

### 6. Warm the model cache

The first run downloads several GB and can look like a hang. Pre-fetch instead:

```powershell
python scripts/prefetch_models.py
```

### 7. Run

```powershell
$env:PYTHONPATH = "src"
python -m meeting_analysis.cli "data/raw/meeting.mp4" -o data/artifacts
```

Outputs land in `data/artifacts/`:

```
<name>.txt          the transcript
<name>.json         the segment array
<name>.16k.wav      the normalized audio
```

---

## Reproducing on another machine

The system runs fully offline once set up, but it is **not** a
`pip install && streamlit run` experience. Four things must be in place first:

| Requirement | Why it is needed | Where |
|---|---|---|
| FFmpeg on `PATH` | decoding any input container/codec | step 1 — system install, never bundled |
| PyTorch (CUDA or CPU) | all model inference | step 5 — install **before** `requirements.txt` |
| A Hugging Face token | the pyannote models are **gated** | steps 2–3 |
| Licence acceptance on **both** gated repos | a token alone is not enough | step 2 |

### The one unavoidable gate

Access to the gated models is granted **per Hugging Face account**, so every new
user must accept the licences themselves. No shared token can work around it —
and accepting only `speaker-diarization-3.1` is *not* sufficient, because
pyannote.audio 4.x resolves that pipeline through
`speaker-diarization-community-1`. See step 2.

### Does it need a GPU?

No. `DEVICE=cpu` works and the pipeline degrades automatically, but diarization
is many times slower. A 7-minute recording takes roughly 4 minutes on the
reference RTX 3050.

### Sample data

No recording is committed. Place your own in `data/raw/` — any format FFmpeg
can decode. All runtime directories (`data/`, `models_cache/`) are created
automatically on first use.

### Self-check before reporting a problem

```powershell
ffmpeg -version                      # FFmpeg present
python -m pytest tests -q            # 40 tests, no GPU or downloads needed
python scripts/prefetch_models.py    # confirms token + licence access
```

`prefetch_models.py` is the fastest way to verify the gated-model setup: if it
reports all repos available, diarization will run.

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │            Entry points                  │
                    │   cli.py  (CLI)     app.py  (Streamlit)  │
                    └───────────────────┬──────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────┐
                    │           pipeline.py                    │
                    │  orchestration · per-stage timing ·      │
                    │  VRAM discipline · artifact emission      │
                    └───────────────────┬──────────────────────┘
                                        │
   ┌──────────┬──────────┬──────────────┼──────────────┬──────────┐
   ▼          ▼          ▼              ▼              ▼          ▼
audio.py   diarization  asr.py      alignment.py   gender.py  emotion.py
normalize  .py          faster-     word↔speaker   wav2vec2   PhoBERT
16k mono   pyannote     whisper     merge          + F0       + pyvi
   │          │          │              │              │          │
   └──────────┴──────────┴──────────────┴──────────────┴──────────┘
                                        │
                    ┌───────────────────▼──────────────────────┐
                    │  interfaces.py — provider protocols      │
                    │  models.py — Pydantic domain contract    │
                    └──────────────────────────────────────────┘
```

**Design principles**

- **Modular monolith** — one deployable unit, clean internal boundaries.
- **Providers behind interfaces** — `Transcriber`, `Diarizer`,
  `GenderClassifier`, `EmotionAnalyzer`.
- **Library-independent domain models** — the output schema imports no ML
  framework.
- **Sequential GPU stages** — ASR and diarization never reside in VRAM
  simultaneously.

---

## Configuration

Settings resolve from environment variables (via `.env`) with sane defaults.

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token for gated models (**required**) |
| `DEVICE` | `cuda` | `cuda` or `cpu`; falls back automatically if CUDA is unavailable |
| `HF_HOME` | `./models_cache` | Model cache location (redirected into the project automatically) |
| `ASR_MODEL` | `medium` | faster-whisper size: `small` · `medium` · `large-v3` |
| `ASR_COMPUTE_TYPE` | `int8_float16` | CTranslate2 precision |
| `ASR_LANGUAGE` | `vi` | Forced language code; empty = auto-detect |

---

## Command-line reference

```
python -m meeting_analysis.cli <audio> [-o OUT] [--asr-model SIZE]
                                     [--language LANG] [--device DEV]
                                     [--num-speakers N] [-v]
```

| Flag | Description |
|---|---|
| `audio` | Path to the meeting recording (any FFmpeg-supported format) |
| `-o`, `--out` | Output directory (default `data/artifacts`) |
| `--asr-model` | `small` · `medium` · `large-v3` (default `medium`) |
| `--language` | Language code, or `auto` to detect (default `vi`) |
| `--device` | `cuda` or `cpu` |
| `--num-speakers` | Pin the speaker count when diarization over/under-splits |
| `-v`, `--verbose` | Debug logging |

> **On `--language`:** the default `vi` is deliberate — short Vietnamese turns
> frequently misdetect as English or Chinese. However, for **mixed
> Vietnamese/English** recordings, forcing one language mangles the other into
> nonsense syllables. Use `--language auto` (or `--language en`) in that case.

---

## Web interface

```powershell
streamlit run app.py
```

Opens at **<http://localhost:8501>**.

Features:

- Upload any audio/video format, with inline playback
- Sidebar controls for ASR model, language, device, and speaker count
- Metric cards: speakers / utterances / duration
- Per-speaker gender table with confidence, median F0, and decision source
- Emotion distribution chart
- Transcript rendered in the required format
- Expandable structured JSON and alignment-flags panel
- Download buttons for both `.txt` and `.json`

The UI calls the same `run_pipeline` + `emit` functions as the CLI, so its
downloads are byte-identical to the command-line artifacts. Analysis is
synchronous — expect roughly 5 minutes for a 7-minute recording, dominated by
diarization.

For a safe local-only bind:

```powershell
streamlit run app.py --server.address 127.0.0.1
```

---

## Project layout

```
.
├── app.py                        Streamlit demo UI
├── pyproject.toml                Packaging, dependencies, tool config
├── requirements.txt              Runtime dependency list
├── constraints.txt               Pins the pre-installed CUDA torch build
├── .env.example                  Credential/config template
├── scripts/
│   └── prefetch_models.py        Warm the model cache before first run
├── src/meeting_analysis/
│   ├── __init__.py               Configures the HF cache on import
│   ├── config.py                 Settings and .env loading
│   ├── models.py                 Pydantic output contract
│   ├── interfaces.py             Provider protocols (swap seams)
│   ├── audio.py                  ffprobe/ffmpeg probing + normalization
│   ├── asr.py                    faster-whisper transcriber
│   ├── diarization.py            pyannote diarizer
│   ├── alignment.py              Word-timestamp ↔ speaker merge
│   ├── gender.py                 Per-speaker gender + F0 cross-check
│   ├── emotion.py                Per-utterance Vietnamese emotion
│   ├── pipeline.py               Stage orchestration and emission
│   ├── cli.py                    Command-line entry point
│   ├── evaluation.py             WER / CER / DER and label-accuracy metrics
│   ├── evaluate.py               Evaluation CLI
│   └── utils/
│       ├── gpu.py                CUDA DLL registration, VRAM hygiene
│       └── logging.py            UTF-8-safe logging setup
├── tests/
│   ├── test_models.py            Output-contract tests
│   ├── test_alignment.py         Alignment edge-case tests
│   └── test_evaluation.py        Metric known-answer tests
└── docs/
    ├── ARCHITECTURE.md           Design and data flow
    ├── IMPLEMENTATION_NOTES.md   Engineering record
    └── SUPERVISOR_GUIDE.md       Presentation guide
```

Runtime directories (`data/`, `models_cache/`) are gitignored. Place input
recordings in `data/raw/`.

---

## Testing

```powershell
python -m pytest tests -q
```

The suite covers the output contract (exact line format, label validity,
TXT/JSON agreement, diacritic preservation), the alignment edge cases
(multi-speaker overlap, diarization gaps, leading gaps, silent turns, segment
merging), and the evaluation metrics themselves against hand-computed known
answers. It requires **no** model downloads and no GPU.

---

## Benchmarking against labelled data

Accuracy is measured with standard metrics rather than inspected by eye.

### 1. Create a labelling template

```powershell
$env:PYTHONPATH = "src"
python -m meeting_analysis.evaluate --prediction data/artifacts/out.json `
                                    --write-template data/artifacts/reference.json
```

This pre-fills timings, speakers, and labels from a prediction, so labelling
means **correcting** fields rather than transcribing from scratch.

### 2. Correct the reference by hand

The reference uses the **same JSON shape** as the pipeline output:

```json
[
  {
    "start": 0.4,
    "end": 2.31,
    "speaker": "A",
    "gender": "Nam",
    "emotion": "Vui vẻ",
    "text": "Chào cô"
  }
]
```

### 3. Score

```powershell
python -m meeting_analysis.evaluate --reference data/artifacts/reference.json `
                                    --prediction data/artifacts/out.json `
                                    --report eval.md --json-out eval.json
```

### Metrics reported

| Metric | Meaning |
|---|---|
| **WER** | Syllable-level word error rate (Vietnamese separates syllables with spaces), broken down into substitutions / deletions / insertions |
| **CER** | Character-level error rate |
| **DER** | Diarization error rate, decomposed into missed speech, false alarm, and speaker confusion |
| **Speaker attribution accuracy** | Correct speaker on matched segments, after an **optimal** speaker mapping |
| **Gender / emotion accuracy** | Label agreement on matched segments, plus an emotion confusion matrix |


> **Status.** The harness is implemented and unit-tested against hand-computed
> known answers. **No accuracy figures are quoted yet**, because no
> human-labelled Vietnamese reference has been produced. Publishing numbers
> derived from the pipeline's own output would be circular and meaningless.

---

## Measured performance

Reference machine: RTX 3050 Laptop (4 GB VRAM), Python 3.11, faster-whisper
`medium` at `int8_float16`.

| Recording | Audio | Diarization | ASR | Gender | Emotion | Total |
|---|---|---|---|---|---|---|
| Vietnamese dialogue | 7 min 09 s | ~3 min | 80 s | 1.5 s | 2 s | **~5 min** |
| Bilingual interview | 15 min 36 s | ~1 min 45 s | 85 s | 2 s | 13 s | **~4.5 min** |

Peak VRAM stays under ~0.1 GB between stages because each stage unloads before
the next loads.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Library cublas64_12.dll is not found` | CTranslate2 cannot see torch's bundled CUDA DLLs. Handled automatically by `utils/gpu.py::register_cuda_dlls`; ensure `torch` is the CUDA build. |
| `GET was unable to find an engine to execute this computation` | cuDNN's heuristic algorithm selection found no convolution engine for the wespeaker embedding ResNet. Handled automatically: the diarizer enables `cudnn.benchmark` and falls back to disabling cuDNN if needed. |
| Run fails after an earlier failed run | A previous run died before releasing its models. Stages now unload in a `finally` block, and the pipeline refuses to start below ~1.2 GB free VRAM with an actionable message. Close stale `python.exe` processes if it persists. |
| `403 GatedRepoError` from pyannote | You have not accepted **both** model licences (see step 2), or accepted them under a different account than the token belongs to. |
| `PermissionError` writing `~/.cache/huggingface` | The default cache is unwritable. The project redirects `HF_HOME` to `./models_cache` automatically; do not override it to an unwritable path. |
| `UnicodeEncodeError` / `UnicodeDecodeError` | Console or subprocess using cp1252. The CLI reconfigures stdout to UTF-8; for ad-hoc scripts set `PYTHONIOENCODING=utf-8`. |
| `ffmpeg: command not found` | Install FFmpeg (step 1) and **restart the shell** — a running process keeps its old `PATH`. |
| First run appears to hang | It is downloading several GB of model weights. Run `scripts/prefetch_models.py` to do this up front. |
| Unicode filename processing fails | Ensure the project runs from a path the current user can read; Vietnamese filenames are handled correctly by the UTF-8 fixes above. |

---

## Known limitations

- **Emotion labels are the least reliable output.** PhoBERT is a Vietnamese
  sentiment model; on very short or code-switched utterances its predictions are
  noisy. Treat emotion as indicative, not authoritative.
- **No chunking for very long recordings.** A single pass is used; multi-hour
  meetings will need VAD-bounded chunking to bound memory.
- **Speaker count is inferred.** Overlapping speech may cause speakers to be
  merged or split; use `--num-speakers` to pin it when known.
- **Gender is binary** (`Nam` / `Nữ`), matching the required output format.
- **Word-level timing drift** inside a single Whisper segment is possible; the
  alignment layer compensates by keeping each ASR segment together.
- **Diarization quality depends on audio quality**, microphone count, and
  overlap. Clean, single-microphone or well-separated audio performs best.

## Future Improvement

-  **Labelled reference corpus** — hand-label 2–3 minutes of audio and publish
      real WER / DER figures using `python -m meeting_analysis.evaluate`
-  **FastAPI service** — REST endpoints for upload and analysis
-  **SQLite job history** — persisted job/meeting metadata behind a
      repository interface (PostgreSQL-ready)
-  **Chunked processing** — VAD-bounded windows for multi-hour meetings
-  **Cloud provider adapters** — implement the existing protocols for a
      managed ASR/diarization backend
-  **Diarization fallback** — speaker-embedding + clustering implementation
      behind `Diarizer`

---

## Acknowledgements

This project builds on the work of the open-source speech community:

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · CTranslate2
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [PhoBERT Vietnamese sentiment](https://huggingface.co/wonrax/phobert-base-vietnamese-sentiment)
- [pyvi](https://github.com/trungtv/pyvi) Vietnamese word segmentation
- [Streamlit](https://streamlit.io/)
- [FFmpeg](https://ffmpeg.org/)

## Licence

MIT
