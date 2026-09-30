"""Command-line evaluation: score a prediction against a labelled reference.

Usage::

    # Score a run
    python -m meeting_analysis.evaluate --reference ref.json --prediction out.json

    # Also write a Markdown report and a machine-readable summary
    python -m meeting_analysis.evaluate --reference ref.json --prediction out.json \
        --report eval.md --json-out eval.json

    # Create a labelling template from a prediction (pre-fills timings/speakers)
    python -m meeting_analysis.evaluate --prediction out.json --write-template ref.json

The reference file uses the same JSON shape as the pipeline output, so labelling
is a matter of correcting the text and the gender/emotion labels in a copy of a
result file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .evaluation import evaluate, labelling_template
from .utils.logging import configure_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meeting_analysis.evaluate",
        description="Score pipeline output against a labelled reference.",
    )
    parser.add_argument("--prediction", type=Path, required=True, help="pipeline output JSON")
    parser.add_argument("--reference", type=Path, help="labelled reference JSON")
    parser.add_argument("--report", type=Path, help="write a Markdown report here")
    parser.add_argument("--json-out", type=Path, help="write the metrics as JSON here")
    parser.add_argument(
        "--write-template",
        type=Path,
        help="write a labelling template built from the prediction, then exit",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    if not args.prediction.exists():
        print(f"error: prediction not found: {args.prediction}", file=sys.stderr)
        return 2
    prediction = _read_json(args.prediction)

    # --- template mode -------------------------------------------------------
    if args.write_template:
        template = labelling_template(prediction)
        args.write_template.parent.mkdir(parents=True, exist_ok=True)
        args.write_template.write_text(
            json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Labelling template written to: {args.write_template}")
        print(
            f"  {len(template)} segments. Correct the text and gender/emotion labels, "
            "then score with --reference."
        )
        return 0

    if not args.reference:
        print("error: --reference is required unless --write-template is used", file=sys.stderr)
        return 2
    if not args.reference.exists():
        print(f"error: reference not found: {args.reference}", file=sys.stderr)
        return 2

    report = evaluate(_read_json(args.reference), prediction)

    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)
    print(f"Segments      : {report.n_reference} reference / {report.n_hypothesis} predicted "
          f"/ {report.matched} matched")
    print(f"Speakers      : {len(report.reference_speakers)} reference / "
          f"{len(report.hypothesis_speakers)} predicted")
    print("-" * 70)
    print(f"WER (syllable): {report.wer.rate:7.2%}   "
          f"(S={report.wer.substitutions} D={report.wer.deletions} I={report.wer.insertions})")
    print(f"CER (character): {report.cer.rate:6.2%}")
    print(f"DER           : {report.der.der:7.2%}   "
          f"(miss={report.der.missed_seconds:.1f}s "
          f"fa={report.der.false_alarm_seconds:.1f}s "
          f"conf={report.der.confusion_seconds:.1f}s)")
    print(f"Speaker attrib.: {report.speaker_accuracy:7.2%}")
    if report.gender_accuracy is not None:
        print(f"Gender        : {report.gender_accuracy:7.2%}")
    if report.emotion_accuracy is not None:
        print(f"Emotion       : {report.emotion_accuracy:7.2%}")
    print(f"Speaker map   : {report.der.mapping}")
    print("=" * 70)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report.to_markdown(), encoding="utf-8")
        print(f"Markdown report -> {args.report}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"JSON metrics    -> {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
