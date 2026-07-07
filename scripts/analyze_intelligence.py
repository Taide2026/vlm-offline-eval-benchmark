"""Aggregate an intelli-bench sweep into an analyze.md report.

Usage:

```bash
uv run scripts/analyze_intelligence.py intelli-sweep-new-scores-300
```

The argument is an output root from `uv run intelli-bench`, containing one
subdirectory per run (`config.json`, `intelligence.log`, `predictions.jsonl`,
`summary.json`). Writes `<dir>/analyze.md` with a per-model LLM-judge score
distribution table.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def iter_runs(root: Path):
    """Yield each intelli-bench run under a sweep output root.

    Args:
        root: Sweep directory containing per-run subdirectories.

    Yields:
        Tuples of ``(config, predictions)`` where ``config`` is the parsed
        ``config.json`` dict and ``predictions`` is the list of parsed
        ``predictions.jsonl`` records, in sorted run-directory order.
    """
    for cfg_path in sorted(root.glob("*/config.json")):
        config = json.loads(cfg_path.read_text())
        with (cfg_path.parent / "predictions.jsonl").open() as f:
            predictions = [json.loads(line) for line in f]
        yield config, predictions


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


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 1
    root = Path(sys.argv[1])

    rows = []
    for config, predictions in iter_runs(root):
        counts = Counter(
            int(score)
            for p in predictions
            if (score := (p.get("scores") or {}).get("llm_judge")) is not None
        )
        rows.append([config["model_id"], *(str(counts.get(s, 0)) for s in range(5))])

    lines = [
        "# Intelligence benchmark analysis",
        "",
        "## LLM judge score distribution",
        "",
        format_table(["model", "0", "1", "2", "3", "4"], rows),
    ]
    out = root / "analyze.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
