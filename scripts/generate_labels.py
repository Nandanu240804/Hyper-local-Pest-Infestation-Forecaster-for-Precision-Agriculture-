# scripts/generate_labels.py
import csv
import json
from pathlib import Path

ROOT = Path.cwd()
CSV = ROOT / "ip102_v1.1" / "train.csv"
OUT = ROOT / "frontend" / "labels.json"

if not CSV.exists():
    print("train.csv not found at", CSV)
    raise SystemExit(1)

labels = []
with open(CSV, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader, None)
    for row in reader:
        if len(row) < 2: 
            continue
        labels.append(row[1].strip())

# If labels look numeric (all digits) then we can't get names from CSV.
all_digits = all(l.isdigit() for l in labels if l != "")
if all_digits:
    # fallback: make generic mapping class_0 ... class_101
    print("CSV labels appear numeric. Generating generic class names class_0..class_{n}.")
    unique_count = max(int(x) for x in set(labels)) + 1 if labels else 102
    mapping = {str(i): f"class_{i}" for i in range(unique_count)}
else:
    # non-numeric -> assume labels are human names already
    unique = []
    seen = set()
    for name in labels:
        if name not in seen:
            seen.add(name); unique.append(name)
    # map index (0..N-1) -> label name (preserve order seen)
    mapping = {str(i): unique[i] for i in range(len(unique))}
    # pad if fewer than 102
    for i in range(len(unique), 102):
        mapping[str(i)] = f"class_{i}"

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as of:
    json.dump(mapping, of, indent=2, ensure_ascii=False)

print("Wrote", OUT)
