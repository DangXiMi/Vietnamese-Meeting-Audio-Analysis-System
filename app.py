"""Streamlit demo UI for the Vietnamese meeting audio analysis system.

Run with::

    streamlit run app.py

The UI is a thin shell over the same library entry point the CLI uses, so the
TXT and JSON it offers for download are byte-identical to `python -m
meeting_analysis.cli`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import meeting_analysis  # noqa: E402,F401  (configures the HF cache on import)
from meeting_analysis.config import Settings  # noqa: E402
from meeting_analysis.pipeline import emit, run_pipeline  # noqa: E402
from meeting_analysis.utils.logging import configure_logging  # noqa: E402
from meeting_analysis.utils.gpu import free_cuda, free_vram_gb  # noqa: E402

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"

AUDIO_TYPES = ["wav", "mp3", "m4a", "mp4", "flac", "aac", "ogg", "webm", "mov", "mkv"]

st.set_page_config(
    page_title="Vietnamese Meeting Audio Analysis",
    page_icon="🎙️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def bootstrap() -> bool:
    """One-time setup: logging, UTF-8 stdout, runtime directories."""
    configure_logging()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return True


def save_upload(uploaded) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / Path(uploaded.name).name
    target.write_bytes(uploaded.getbuffer())
    return target


def render_results(result) -> None:
    st.subheader("Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Detected speakers", result.num_speakers)
    col2.metric("Utterances", len(result.utterances))
    col3.metric("Duration", f"{result.metadata.get('duration_seconds', 0):.0f} s")

    # --- speaker summary (raw diarization labels -> A, B, C ...) -------------
    letters = result.metadata.get("speaker_labels", {})
    detail = result.metadata.get("speaker_gender_detail", {})
    if detail:
        st.markdown("**Speakers**")
        rows = []
        for raw, info in detail.items():
            f0 = info.get("f0_median")
            rows.append(
                {
                    "Speaker": letters.get(raw, raw),
                    "Gender": info.get("gender"),
                    "Confidence": round(float(info.get("confidence", 0.0)), 2),
                    "Median F0 (Hz)": round(f0, 1) if f0 else None,
                    "Source": info.get("source"),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)

    # --- emotion distribution ------------------------------------------------
    counts: dict[str, int] = {}
    for utterance in result.utterances:
        key = utterance.emotion.value
        counts[key] = counts.get(key, 0) + 1
    if counts:
        st.markdown("**Emotion distribution**")
        st.bar_chart(counts, horizontal=True)

    # --- transcript ----------------------------------------------------------
    st.markdown("**Transcript**")
    st.caption('Format: [<Gender> <Speaker> - <Emotion>]: "<text>"')
    st.code(result.to_text(), language=None)

    with st.expander("Structured JSON"):
        st.json(result.to_json())

    if any(u.notes for u in result.utterances):
        with st.expander("Alignment notes (flagged cases)"):
            for index, utterance in enumerate(result.utterances):
                if utterance.notes:
                    st.write(f"`#{index}` {utterance.speaker}: {'; '.join(utterance.notes)}")


def main() -> None:
    bootstrap()

    st.title("🎙️ Vietnamese Meeting Audio Analysis")
    st.caption(
        "Audio → diarization → ASR → per-speaker gender → per-utterance emotion → transcript"
    )

    with st.sidebar:
        st.header("Settings")
        asr_model = st.selectbox(
            "ASR model",
            ["medium", "large-v3", "small"],
            index=0,
            help="large-v3 is more accurate but downloads ~3 GB on first use.",
        )
        language = st.selectbox(
            "Language",
            ["vi", "auto", "en"],
            index=0,
            help="'auto' suits mixed Vietnamese/English recordings. Forcing a single "
            "language mangles the other one into nonsense syllables.",
        )
        device = st.selectbox("Device", ["cuda", "cpu"], index=0)
        forced = st.number_input(
            "Force speaker count (0 = auto-detect)", min_value=0, max_value=12, value=0
        )
        st.divider()
        free_gb = free_vram_gb()
        if free_gb is not None:
            st.caption(f"Free GPU memory: **{free_gb:.2f} GB**")
            if free_gb < 1.2:
                st.warning(
                    "Low GPU memory. Another process may still hold the GPU; "
                    "release it below or close the stale `python.exe`."
                )
            if st.button("Release cached GPU memory"):
                free_cuda()
                st.success("Released. The figure above refreshes on the next interaction.")
        st.caption(
            "Requires **FFmpeg** on PATH and an **HF token** in `.env` with the "
            "pyannote licences accepted. See README."
        )

    uploaded = st.file_uploader("Upload a meeting recording", type=AUDIO_TYPES)
    if uploaded is None:
        st.info("Upload an audio or video file to begin.")
        return

    st.audio(uploaded)

    if st.button("Analyze", type="primary"):
        source = save_upload(uploaded)
        settings = Settings()
        settings.asr_model = asr_model
        settings.asr_language = None if language == "auto" else language
        settings.device = device
        settings.num_speakers = int(forced) or None

        try:
            with st.status("Running pipeline…", expanded=True) as status:
                st.write("1/2 Diarizing speakers and transcribing (this can take minutes)…")
                result = run_pipeline(source, ARTIFACT_DIR, settings)
                st.write("2/2 Writing TXT and JSON artifacts…")
                txt_path, json_path = emit(result, ARTIFACT_DIR, source.stem)
                status.update(label="Analysis complete", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
            st.exception(exc)
            return

        st.session_state["result"] = result
        st.session_state["txt_path"] = str(txt_path)
        st.session_state["json_path"] = str(json_path)

    result = st.session_state.get("result")
    if result is None:
        return

    render_results(result)

    txt_path = Path(st.session_state["txt_path"])
    json_path = Path(st.session_state["json_path"])
    left, right = st.columns(2)
    left.download_button(
        "⬇️ Download transcript (.txt)",
        txt_path.read_bytes(),
        file_name=txt_path.name,
        mime="text/plain",
        width="stretch",
    )
    right.download_button(
        "⬇️ Download segments (.json)",
        json_path.read_bytes(),
        file_name=json_path.name,
        mime="application/json",
        width="stretch",
    )


if __name__ == "__main__":
    main()
