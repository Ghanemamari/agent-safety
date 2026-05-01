import os

raw_dir = "data/raw"
for f in os.listdir(raw_dir):
    if f.endswith(".jsonl"):
        path = os.path.join(raw_dir, f)
        with open(path, "r", encoding="utf-8") as file:
            count = sum(1 for _ in file)
        print(f"{f}: {count}")
