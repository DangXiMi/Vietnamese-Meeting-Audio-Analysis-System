# Meeting Audio Analysis System (Vietnamese)

End-to-end pipeline that turns a Vietnamese meeting recording into a
speaker-attributed, emotion-labeled transcript.

```
audio → normalize (16 kHz mono) → diarize → transcribe → align → gender → emotion → TXT + JSON
```

## Output contract

Each utterance renders as:

```
[<Gender> <SpeakerLetter> - <Emotion>]: "<text>"
```

Every run emits **both** artifacts from one in-memory result:

**`.txt`**
```
[Nam A - Vui vẻ]: "Em làm phần này xong rồi nhé."
[Nữ B - Nóng giận]: "Tại sao tiến độ lại chậm như vậy?"
```

**`.json`**
```json
[
  {
    "start": 0.031,
    "end": 2.845,
    "speaker": "A",
    "gender": "Nam",
    "emotion": "Vui vẻ",
    "text": "Em làm phần này xong rồi nhé."
  }
]
```

Labels: gender `Nam` / `Nữ`; speaker `A`, `B`, `C`… (**detected, never
hardcoded**); emotion `Vui vẻ` / `Nóng giận` / `Bình thường`.

## Stack

| Stage | Implementation |
|---|---|
| ASR | `faster-whisper` (CTranslate2), `language="vi"` forced, `int8_float16` |
| Diarization | `pyannote.audio` `speaker-diarization-3.1` |
| Alignment | word-level timestamps merged onto diarization turns |
| Gender | `wav2vec2` audio classifier **per speaker** (pooled audio) + median-F0 cross-check |
| Emotion | `wonrax/phobert-base-vietnamese-sentiment` (PhoBERT), per utterance |

**Hardware target:** NVIDIA RTX 3050 Laptop, **4 GB VRAM**. ASR and diarization
never sit on the GPU at the same time — each stage unloads and releases VRAM
before the next loads.

## Architecture

Every model-backed stage sits behind a protocol in `interfaces.py`, so a larger
model, a cloud API, or a custom diarization fallback can replace an
implementation without touching alignment, orchestration, or the CLI.

| Module | Responsibility |
|---|---|
| `audio.py` | ffprobe/ffmpeg probing + normalization to one canonical 16 kHz mono WAV |
| `asr.py` | `FasterWhisperTranscriber` → segments with word-level timestamps |
| `diarization.py` | `PyannoteDiarizer` → speaker turns |
| `alignment.py` | merge word timings onto turns; handles split/gap/no-text cases |
| `gender.py` | per-speaker gender from pooled audio + F0 cross-check |
| `emotion.py` | per-utterance Vietnamese emotion (PhoBERT) |
| `pipeline.py` | stage orchestration + artifact emission |
| `cli.py` | command-line entry point |
| `config.py` | `.env` + settings; redirects the HF cache into `models_cache/` |

## Setup

### 1. FFmpeg (required system dependency)

```powershell
winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
```

Restart your shell afterwards so `PATH` is refreshed, then verify:

```powershell
ffmpeg -version
ffprobe -version
```

### 2. Hugging Face token (required — the pyannote models are gated)

1. Create a token: https://huggingface.co/settings/tokens
2. Accept the licence terms for **both**:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

   Accepting only the pipeline yields a confusing 403.
3. Create `.env` from the template and fill it in (`.env` is gitignored):

```powershell
Copy-Item .env.example .env
```

```ini
HF_TOKEN=hf_...
```

Either casing (`HF_TOKEN` / `hf_token`) is accepted.

### 3. Python dependencies

`torch`/`torchvision` are CUDA builds matched to this machine and are pinned in
`constraints.txt` so pip cannot replace them:

```powershell
pip install -r requirements.txt
pip install -c constraints.txt faster-whisper pyannote.audio librosa soundfile
```

### 4. Warm the model cache

First execution downloads several GB and looks like a hang. Pre-fetch instead:

```powershell
python scripts/prefetch_models.py
```

## Usage

```powershell
python -m meeting_analysis.cli "data/raw/<meeting>.mp4" -o data/artifacts
```

| Flag | Purpose |
|---|---|
| `--asr-model medium\|large-v3` | ASR size (default `medium`) |
| `--device cuda\|cpu` | force a device |
| `--num-speakers N` | pin the speaker count when diarization over/under-splits |
| `-v` | debug logging |

Artifacts land in the output directory as `<name>.txt`, `<name>.json`, plus the
normalized `<name>.16k.wav`.

## Windows notes

- **UTF-8.** Vietnamese diacritics break through a cp1252 console. The CLI
  reconfigures stdout to UTF-8 and every artifact is written with
  `encoding="utf-8"`. When running ad-hoc scripts, set `PYTHONIOENCODING=utf-8`.
- **Model cache.** The default `~/.cache/huggingface` may be unwritable, so the
  cache is redirected to `models_cache/` inside the project automatically.

## Demo UI (Streamlit)

```powershell
streamlit run app.py
```

Opens at **http://localhost:8501**. Upload a recording, click **Analyze**, then
read the transcript in the app and download the `.txt` / `.json`.

The UI calls the same `run_pipeline` + `emit` functions as the CLI, so its
downloads are byte-identical to the command-line artifacts. It also shows the
per-speaker gender table (with median F0), the emotion distribution, and any
alignment flags.

Analysis is **synchronous**: a 7-minute recording takes roughly 5 minutes on the
RTX 3050, dominated by diarization.

## Status

Working end-to-end: CLI and Streamlit UI producing the required TXT + JSON.
Not yet implemented: FastAPI service, SQLite job history.
