"""Self-check for analyze_intelligence.py. Run: uv run scripts/test_analyze_intelligence.py"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from analyze_intelligence import SECTIONS, load_runs, main


def write_run(root: Path, model: str, records: list[tuple[str, float | None]]) -> None:
    d = root / model.replace("/", "_")
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_id": model}))
    with (d / "predictions.jsonl").open("w") as f:
        for label, score in records:
            f.write(json.dumps({"label": label, "scores": {"llm_judge": score}}) + "\n")


def demo() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_run(root, "a/one", [("fall_general", 4.0), ("hn_lie_down_floor", 0.0)])
        # second run misses a label and has one unscored prediction
        write_run(root, "b/two", [("fall_general", 2.0), ("fall_general", None)])

        runs = load_runs(root)
        assert [r.model for r in runs] == ["a/one", "b/two"], runs

        assert main.__module__  # entry point importable
        report = {heading: fn(runs) for heading, fn in SECTIONS}

        dist = report["LLM judge score distribution"]
        assert "| a/one | 1 | 0 | 0 | 0 | 1 |" in dist, dist
        assert "| b/two | 0 | 0 | 1 | 0 | 0 |" in dist, dist  # None ignored

        heat = report["Per-label mean judge score"]
        assert "4.00 ████" in heat and "0.00 ░░░░" in heat, heat
        assert "-" in heat.split("hn_lie_down_floor")[1].splitlines()[0], heat  # missing label

        split = report["Accident vs hard-negative"]
        assert "hard negative (hn_*)" in split and "| -" in split, split  # b/two has no hn_*

    print("ok")


if __name__ == "__main__":
    demo()
