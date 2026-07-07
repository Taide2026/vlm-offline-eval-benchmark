"""Show one video's prompt and per-model results across an intelli-bench sweep.

Usage:

```bash
uv run scripts/show_video_result.py fall_from_bed_156.mp4 intelli-sweep-new-scores-300
```

Prints the VLM prompt used in the sweep and a markdown table with each
model's response, LLM-judge score, and NLI score for that video.
"""
from __future__ import annotations

import sys
from pathlib import Path

from analyze_intelligence import format_table, iter_runs


def _cell(value) -> str:
    """Render a value as a single markdown table cell."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}" if value == int(value) else f"{value:.3f}"
    return " ".join(str(value).split()).replace("|", "\\|")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 1
    video, root = sys.argv[1], Path(sys.argv[2])

    prompt = None
    rows = []
    for config, predictions in iter_runs(root):
        prompt = config["prompt"]
        record = next((p for p in predictions if Path(p["video"]).name == video), None)
        if record is None:
            continue
        scores = record.get("scores") or {}
        rows.append(
            (config["model_id"], record.get("response"), scores.get("llm_judge"), scores.get("nli"))
        )

    if not rows:
        print(f"no results for {video} under {root}", file=sys.stderr)
        return 1

    print(f"prompt: {prompt}")
    print()
    table_rows = [[_cell(v) for v in row] for row in rows]
    print(format_table(["model", "response", "llm_judge", "nli"], table_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
