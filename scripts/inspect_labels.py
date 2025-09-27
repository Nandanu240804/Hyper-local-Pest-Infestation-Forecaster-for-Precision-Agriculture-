# scripts/inspect_labels.py
import csv, os, json
from pathlib import Path

ROOT = Path.cwd()
CSV = ROOT / "ip102_v1.1" / "train.csv"
TXT = ROOT / "ip102_v1.1" / "train.txt"
OUT = ROOT / "frontend" / "labels.json"

def show_head(path, n=10):
    print(f"\n--- head of {path} ---")
    try:
        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= n: break
                print(line.rstrip())
    except Exception as e:
        print("  (error reading)", e)

def unique_csv_labels(path, limit=100):
    vals = []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    vals.append(row[1].strip())
                if len(vals) >= limit:
                    break
    except Exception as e:
        return None, str(e)
    return list(dict.fromkeys(vals)), None

print("Running label inspection in:", ROOT)
print("Checking files...")

if CSV.exists():
    show_head(CSV, n=5)
    vals, err = unique_csv_labels(CSV, limit=200)
    if vals is None:
        print("Error reading CSV labels:", err)
    else:
        print(f"\nFirst {min(len(vals),20)} unique label values from train.csv (total shown up to 200):")
        for i, v in enumerate(vals[:50]):
            print(f"  {i:3d}: {v}")
        if all(v.isdigit() for v in vals if v != ""):
            print("\nObservation: train.csv labels look numeric (integers).")
        else:
            print("\nObservation: train.csv labels look like names (non-numeric).")
else:
    print("train.csv not found at", CSV)

if TXT.exists():
    show_head(TXT, n=10)
else:
    print("train.txt not found at", TXT)

# check for other candidate files
cands = ["labels.txt", "classes.txt", "class_names.txt", "synset.txt"]
for c in cands:
    p = ROOT / "ip102_v1.1" / c
    if p.exists():
        show_head(p, n=10)

print("\nDone. If train.csv contains numeric labels, you need a separate mapping file (id -> pest name).")
print("If you want me to generate a labels.json for you, paste the first ~100 unique label values shown above, or provide a class-names file.")
