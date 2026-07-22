"""Aggregate an intelli-bench sweep into an analyze.md report.

Usage:

```bash
uv run scripts/analyze_intelligence.py intelli-sweep-new-scores-300
```

The argument is an output root from `uv run intelli-bench`, containing one
subdirectory per run (`config.json`, `intelligence.log`, `predictions.jsonl`,
`summary.json`). Writes `<dir>/analyze.md`.

To add a new analysis, write a function taking `runs` and returning markdown,
and decorate it with `@section("Your heading")`. Sections render into
`analyze.md` in registration order.
"""
from __future__ import annotations

import inspect
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, NamedTuple

SCORES = range(5)  # llm_judge is an integer 0..4


class Run(NamedTuple):
    """One intelli-bench run: its config and its parsed predictions."""

    config: dict
    predictions: list[dict]

    @property
    def model(self) -> str:
        return self.config["model_id"]


def load_runs(root: Path) -> list[Run]:
    """Load every intelli-bench run under a sweep output root.

    Args:
        root: Sweep directory containing per-run subdirectories.

    Returns:
        One ``Run`` per subdirectory, in sorted run-directory order.
    """
    runs = []
    for cfg_path in sorted(root.glob("*/config.json")):
        with (cfg_path.parent / "predictions.jsonl").open() as f:
            predictions = [json.loads(line) for line in f]
        runs.append(Run(json.loads(cfg_path.read_text()), predictions))
    return runs


def judge_scores(predictions: list[dict]) -> list[int]:
    """Extract the integer llm_judge score of every scored prediction."""
    return [
        int(score)
        for p in predictions
        if (score := (p.get("scores") or {}).get("llm_judge")) is not None
    ]


def format_table(header: list[str], rows: list[list[str]]) -> str:
    """Render a column-aligned markdown table.

    Args:
        header: Column names.
        rows: Table rows; each row must have one cell per header column.

    Returns:
        The table as a multi-line string with each column padded to the
        width of its longest cell.
    """
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows)]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    separator = ["-" * w for w in widths]
    return "\n".join(line(cells) for cells in [header, separator, *rows])


# --- section registry -------------------------------------------------------

SECTIONS: list[tuple[str, Callable[[list[Run]], str]]] = []


def section(heading: str):
    """Register a function as an analyze.md section under ``heading``.

    The function's docstring is rendered as the section's intro paragraph, so
    write it for whoever reads the report, not just for whoever reads the code.
    """

    def register(fn: Callable[[list[Run]], str]):
        SECTIONS.append((heading, fn))
        return fn

    return register


# --- sections ---------------------------------------------------------------


@section("LLM judge score distribution")
def score_distribution(runs: list[Run]) -> str:
    """How many predictions each model earned at each judge score.

    One row per model; columns `0`-`4` are the LLM judge's verdict on that
    prediction, where 0 means the response missed what the video showed and 4
    means it matched. Cells are prediction counts, so each row sums to the
    number of scored predictions in that run.
    """
    rows = [
        [run.model, *(str(Counter(judge_scores(run.predictions)).get(s, 0)) for s in SCORES)]
        for run in runs
    ]
    return format_table(["model", *(str(s) for s in SCORES)], rows)


def _mean_by_label(run: Run) -> dict[str, float]:
    """Mean llm_judge score per label for one run."""
    by_label: dict[str, list[int]] = defaultdict(list)
    for p in run.predictions:
        if (score := (p.get("scores") or {}).get("llm_judge")) is not None:
            by_label[p["label"]].append(int(score))
    return {label: sum(v) / len(v) for label, v in by_label.items()}


def _heat(mean: float) -> str:
    """Render a mean score 0..4 as ``2.75 ███░`` for at-a-glance scanning."""
    filled = round(mean)
    return f"{mean:.2f} " + "█" * filled + "░" * (len(SCORES) - 1 - filled)


@section("Per-label mean judge score")
def label_heatmap(runs: list[Run]) -> str:
    """Which kinds of video each model handles well, and which it fails.

    One row per label (the video category), one column per model. Each cell is
    that model's **mean** judge score on that label, from 0.00 to 4.00 — higher
    is better — followed by a bar rounding the mean to whole blocks (`████` =
    4, `░░░░` = 0) so a weak row or column stands out without reading numbers.

    `n` is how many videos of that label the sweep scored per run. Small `n`
    means a noisy mean: a single video swings an `n = 4` row by 1.00. Where a
    run scored no videos of a label at all, the cell shows `-`.
    """
    means = {run.model: _mean_by_label(run) for run in runs}
    # n = videos per label in a single run (runs may differ; take the largest)
    counts: Counter[str] = Counter()
    for run in runs:
        counts |= Counter(p["label"] for p in run.predictions)

    rows = []
    for label, n in counts.most_common():
        cells = [
            _heat(m) if (m := means[run.model].get(label)) is not None else "-"
            for run in runs
        ]
        rows.append([label, str(n), *cells])
    return format_table(["label", "n", *(run.model for run in runs)], rows)


@section("Accident vs hard-negative")
def category_split(runs: list[Run]) -> str:
    """The same scores rolled up into the two groups that matter.

    Labels prefixed `hn_` are **hard negatives**: everyday floor-level motion
    (sitting down, yoga, tying shoes) that looks like a fall but is not one. A
    model should describe those as normal. Everything else is a real accident
    it should report.

    A model that scores well on accidents but poorly on hard negatives is
    over-reporting — it calls a fall on anything near the floor. The reverse
    means it is missing real accidents. Same 0.00-4.00 mean and bar as above.
    """

    def group_mean(run: Run, hard_negative: bool) -> str:
        scores = [
            int(score)
            for p in run.predictions
            if p["label"].startswith("hn_") is hard_negative
            and (score := (p.get("scores") or {}).get("llm_judge")) is not None
        ]
        return _heat(sum(scores) / len(scores)) if scores else "-"

    rows = [
        ["accident (fall_*, post_fall_*)", *(group_mean(r, False) for r in runs)],
        ["hard negative (hn_*)", *(group_mean(r, True) for r in runs)],
    ]
    return format_table(["category", *(run.model for run in runs)], rows)


# --- entry point ------------------------------------------------------------


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 1
    root = Path(sys.argv[1])

    runs = load_runs(root)
    if not runs:
        print(f"no runs found under {root}", file=sys.stderr)
        return 1

    lines = ["# Intelligence benchmark analysis"]
    for heading, fn in SECTIONS:
        lines += ["", f"## {heading}", "", inspect.getdoc(fn), "", fn(runs)]

    out = root / "analyze.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
