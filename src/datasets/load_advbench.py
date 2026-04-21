"""
AdvBench Dataset Loader  (Zou et al., 2023)
GitHub: https://github.com/llm-attacks/llm-attacks

AdvBench provides 500 harmful behaviors + jailbreak strings.
We use it as an OOD (out-of-distribution) test set only — never for training.

Usage:
    python -m src.datasets.load_advbench --out data/raw/advbench.jsonl --download
"""

import argparse
import csv
import json
import os
import urllib.request
from pathlib import Path
from typing import List, Dict

# ─── Public URLs ──────────────────────────────────────────────────────────────
ADVBENCH_URLS = {
    "harmful_behaviors": (
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
        "data/advbench/harmful_behaviors.csv"
    ),
    "harmful_strings": (
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
        "data/advbench/harmful_strings.csv"
    ),
}

CACHE_DIR = Path("data/cache/advbench")


def _download_csv(url: str, cache_path: Path) -> List[Dict]:
    if cache_path.exists():
        print(f"  [cache] {cache_path}")
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [download] {url}")
        urllib.request.urlretrieve(url, cache_path)

    rows = []
    with open(cache_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def load_advbench(
    download: bool = True,
    subset: str = "behaviors",  # "behaviors" | "strings" | "both"
    benign_source: str = "alpaca",  # source for benign counterparts
    benign_jsonl: str = None,       # optional path to a benign prompts JSONL
) -> List[Dict]:
    """
    Load AdvBench and return standard rows.

    ⚠️  AdvBench contains only harmful prompts — we pair each harmful prompt
    with a benign prompt from a neutral source (Alpaca instructions by default).
    This creates approximate pairs (not as tight as your synthetic pairs, which
    is intentional for OOD testing).

    Args:
        download: Whether to download files from GitHub.
        subset: Which AdvBench subset to use.
        benign_source: Label for benign source in metadata.
        benign_jsonl: Optional path to a JSONL file with benign prompts.

    Returns:
        List of standard dicts with keys: id, prompt, label, pair_id, source
    """
    rows_harmful = []

    if subset in ("behaviors", "both"):
        url = ADVBENCH_URLS["harmful_behaviors"]
        cache = CACHE_DIR / "harmful_behaviors.csv"
        if download or cache.exists():
            entries = _download_csv(url, cache)
            for e in entries:
                rows_harmful.append({
                    "text": e.get("goal", e.get("behavior", "")),
                    "subtype": "behavior",
                })

    if subset in ("strings", "both"):
        url = ADVBENCH_URLS["harmful_strings"]
        cache = CACHE_DIR / "harmful_strings.csv"
        if download or cache.exists():
            entries = _download_csv(url, cache)
            for e in entries:
                rows_harmful.append({
                    "text": e.get("target", e.get("string", "")),
                    "subtype": "string",
                })

    # ── Load benign prompts ────────────────────────────────────────────────────
    benign_prompts = _get_benign_prompts(len(rows_harmful), benign_jsonl)

    # ── Build paired output rows ───────────────────────────────────────────────
    output = []
    for i, harmful in enumerate(rows_harmful):
        pair_id = f"advbench_{i:04d}"
        benign_text = benign_prompts[i % len(benign_prompts)] if benign_prompts else (
            "Please summarize the following text in one paragraph."
        )

        output.append({
            "id": f"{pair_id}_injected",
            "prompt": harmful["text"],
            "label": "injected",
            "pair_id": pair_id,
            "source": "advbench",
            "subtype": harmful["subtype"],
        })
        output.append({
            "id": f"{pair_id}_benign",
            "prompt": benign_text,
            "label": "benign",
            "pair_id": pair_id,
            "source": benign_source,
            "subtype": "benign_counterpart",
        })

    return output


def _get_benign_prompts(n: int, benign_jsonl: str = None) -> List[str]:
    """
    Return a list of benign prompts.
    Uses a provided JSONL or falls back to built-in neutral instructions.
    """
    if benign_jsonl and os.path.exists(benign_jsonl):
        with open(benign_jsonl, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        # Support both existing project format and Alpaca format
        prompts = [
            r.get("prompt", r.get("instruction", r.get("text", "")))
            for r in rows if r.get("label", "benign") == "benign"
        ]
        if prompts:
            return prompts

    # Built-in minimal benign fallback (neutral, task-oriented instructions)
    return [
        "What is the capital of France?",
        "Summarize the main themes of Romeo and Juliet.",
        "Write a poem about a sunset over the ocean.",
        "Explain how photosynthesis works in simple terms.",
        "What are three tips for better time management?",
        "Describe the water cycle.",
        "Translate 'hello' into five different languages.",
        "List the planets in our solar system in order from the sun.",
        "What is the difference between a simile and a metaphor?",
        "How does a refrigerator work?",
    ] * (n // 10 + 1)


def save_jsonl(rows: List[Dict], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    injected = sum(1 for r in rows if r["label"] == "injected")
    print(f"Saved {len(rows)} rows ({injected} injected, {len(rows)-injected} benign) -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Convert AdvBench to project JSONL format")
    ap.add_argument("--out", default="data/raw/advbench.jsonl")
    ap.add_argument("--download", action="store_true", help="Download from GitHub")
    ap.add_argument("--subset", choices=["behaviors", "strings", "both"], default="behaviors")
    ap.add_argument("--benign_jsonl", default=None,
                    help="Optional path to your existing benign prompts JSONL for pairing")
    ap.add_argument("--no_cache", action="store_true")
    args = ap.parse_args()

    if args.no_cache and CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)

    rows = load_advbench(
        download=args.download or not (CACHE_DIR / "harmful_behaviors.csv").exists(),
        subset=args.subset,
        benign_jsonl=args.benign_jsonl,
    )

    if rows:
        save_jsonl(rows, args.out)
    else:
        print("No data loaded. Use --download to fetch from GitHub.")


if __name__ == "__main__":
    main()
