"""Filter metadata.csv by human_filter.csv score column -> custom_metadata.csv.

Usage:
    uv run filter_metadata.py [score]

    score: value of human_filter.csv's score column to keep (default: 2).
           Rows with an empty score are always skipped.

Example:
    uv run filter_metadata.py 2

Reads metadata.csv and human_filter.csv from the current directory (run from
the repo root), matches rows on (label, filename), and writes the kept
metadata rows to custom_metadata.csv plus per-label counts to stdout.
The input files are not modified.

Required CSV columns:
    human_filter.csv: label, filename, score
    metadata.csv:     label, source_filename (all columns are copied to output)
"""
import argparse
import csv
from collections import Counter

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("score", nargs="?", default="2",
                    help="score value to keep (default: 2)")
SCORE = parser.parse_args().score

with open("human_filter.csv", newline="") as f:
    keep = {
        (r["label"], r["filename"])
        for r in csv.DictReader(f)
        if r["score"].strip() == SCORE  # empty score rows skipped
    }

with open("metadata.csv", newline="") as fin, open("custom_metadata.csv", "w", newline="") as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
    writer.writeheader()
    counts = Counter()
    for row in reader:
        if (row["label"], row["source_filename"]) in keep:
            writer.writerow(row)
            counts[row["label"]] += 1

for label, n in sorted(counts.items()):
    print(f"{n} videos for label {label}")
print(f"kept {sum(counts.values())}/{len(keep)} matched rows -> custom_metadata.csv")
