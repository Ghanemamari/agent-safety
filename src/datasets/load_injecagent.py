"""
InjecAgent Dataset Loader  (Zhan et al., ACL 2024)
GitHub: https://github.com/uiuc-kang-lab/InjecAgent

Converts InjecAgent test cases into the project's standard JSONL format:
    {"id": str, "prompt": str, "label": "benign"|"injected", "pair_id": str, "source": "injecagent"}

Usage:
    python -m src.datasets.load_injecagent --out data/raw/injecagent.jsonl
    python -m src.datasets.load_injecagent --out data/raw/injecagent.jsonl --download
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import List, Dict

# ─── URLs des fichiers JSON publics d'InjecAgent ────────────────────────────
INJECAGENT_URLS = {
    "direct_harm": (
        "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_dh_base.json"
    ),
    "data_exfil": (
        "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_ds_base.json"
    ),
}

CACHE_DIR = Path("data/cache/injecagent")


def _download_json(url: str, cache_path: Path) -> List[Dict]:
    """Download and cache a JSON file from a URL."""
    if cache_path.exists():
        print(f"  [cache] {cache_path}")
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [download] {url}")
        urllib.request.urlretrieve(url, cache_path)
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_prompt(entry: Dict) -> str:
    """
    Reconstruct a prompt from an InjecAgent test case.
    The dataset provides: User Instruction + Tool Response (containing injection).
    We format it as the agent would see it at inference time.
    """
    user_instr = entry.get("User Instruction", "").strip()
    # Tool response is where the injection is embedded
    tool_response = entry.get("Injected Tool Response", entry.get("Tool Response", "")).strip()
    attack_instr = entry.get("Injected Instruction", "").strip()

    if tool_response:
        prompt = (
            f"User instruction: {user_instr}\n\n"
            f"Tool response: {tool_response}"
        )
    else:
        prompt = f"User instruction: {user_instr}"

    return prompt


def load_injecagent(download: bool = True) -> List[Dict]:
    """
    Load InjecAgent and return a list of standard dicts.
    Each injected case is paired with a benign version (without injection).
    Returns rows with keys: id, prompt, label, pair_id, source, attack_type
    """
    rows = []
    pair_counter = 0

    for attack_type, url in INJECAGENT_URLS.items():
        cache_path = CACHE_DIR / f"{attack_type}.json"

        if download:
            entries = _download_json(url, cache_path)
        elif cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        else:
            print(f"  [skip] {attack_type} - not cached. Use --download to fetch.")
            continue

        for i, entry in enumerate(entries):
            pair_id = f"injecagent_{attack_type}_{pair_counter}"

            # ── Injected version ────────────────────────────────────────────
            injected_prompt = _build_prompt(entry)
            rows.append({
                "id": f"{pair_id}_injected",
                "prompt": injected_prompt,
                "label": "injected",
                "pair_id": pair_id,
                "source": "injecagent",
                "attack_type": attack_type,
            })

            # ── Benign version (same user instruction, no injected tool resp) ─
            user_instr = entry.get("User Instruction", "").strip()
            benign_tool_resp = entry.get("Tool Response", "").strip()
            if benign_tool_resp:
                benign_prompt = (
                    f"User instruction: {user_instr}\n\n"
                    f"Tool response: {benign_tool_resp}"
                )
            else:
                benign_prompt = f"User instruction: {user_instr}"

            rows.append({
                "id": f"{pair_id}_benign",
                "prompt": benign_prompt,
                "label": "benign",
                "pair_id": pair_id,
                "source": "injecagent",
                "attack_type": attack_type,
            })

            pair_counter += 1

    return rows


def save_jsonl(rows: List[Dict], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} rows -> {out_path}")
    injected = sum(1 for r in rows if r["label"] == "injected")
    benign = sum(1 for r in rows if r["label"] == "benign")
    print(f"  injected={injected}  benign={benign}")


def main():
    ap = argparse.ArgumentParser(description="Convert InjecAgent to project JSONL format")
    ap.add_argument("--out", default="data/raw/injecagent.jsonl", help="Output JSONL path")
    ap.add_argument("--download", action="store_true", help="Download dataset from GitHub")
    ap.add_argument("--no_cache", action="store_true", help="Force re-download even if cached")
    args = ap.parse_args()

    if args.no_cache:
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)

    rows = load_injecagent(download=args.download or not (CACHE_DIR / "direct_harm.json").exists())
    if rows:
        save_jsonl(rows, args.out)
    else:
        print("No rows loaded. Run with --download to fetch the dataset.")


if __name__ == "__main__":
    main()
