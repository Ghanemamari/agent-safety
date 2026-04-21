"""
AgentDojo Dataset Loader  (Debenedetti et al., NeurIPS 2024)
GitHub: https://github.com/ethz-spylab/agentdojo

Converts AgentDojo injection test cases into the project's standard JSONL format.

Usage:
    pip install agentdojo   (or install from source)
    python -m src.datasets.load_agentdojo --out data/raw/agentdojo.jsonl
"""

import argparse
import json
import os
from typing import List, Dict


# ─── AgentDojo tasks format (simplified extraction) ──────────────────────────
# AgentDojo organizes tasks by "suite" (workspace, banking, slack, travel).
# Each suite has user tasks and injection test cases.
# We extract the injected prompt the agent would process + a benign baseline.

AGENTDOJO_SUITES = [
    "workspace",
    "banking",
    "slack",
    "travel",
]


def _try_import_agentdojo():
    """Try importing agentdojo package; return None if not installed."""
    try:
        import agentdojo
        return agentdojo
    except ImportError:
        return None


import urllib.request
import urllib.error

def load_agentdojo_via_package() -> List[Dict]:
    """Fetch AgentDojo using the HF Datasets Server API (avoids pyarrow requirement)."""
    rows = []
    print("Fetching AgentDojo data from HF datasets server API...")
    url = "https://datasets-server.huggingface.co/rows?dataset=ffuuugor%2Fagentdojo-dump&config=default&split=train&offset=0&length=100"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
    except urllib.error.URLError as e:
        print(f"Error touching HF Datasets server: {e}")
        return []
        
    pair_counter = 0
    # Expected HF rows have `row` field containing columns
    for item in data.get("rows", []):
        r = item.get("row", {})
        suite_name = r.get("suite", "unknown")
        
        # We try to extract injection or prompt
        injected_instruction = r.get("injection", None) or r.get("prompt", None) or str(r)
        benign_instruction = r.get("user_task", f"Complete the task in {suite_name} environment.")
        
        pair_id = f"agentdojo_{suite_name}_{pair_counter}"
        
        # Injected row
        rows.append({
            "id": f"{pair_id}_injected",
            "prompt": f"[AgentDojo:{suite_name}] {injected_instruction}",
            "label": "injected",
            "pair_id": pair_id,
            "source": "agentdojo",
            "suite": suite_name,
            "task_id": str(pair_counter),
        })

        # Benign row
        rows.append({
            "id": f"{pair_id}_benign",
            "prompt": f"[AgentDojo:{suite_name}] {benign_instruction}",
            "label": "benign",
            "pair_id": pair_id,
            "source": "agentdojo",
            "suite": suite_name,
            "task_id": str(pair_counter),
        })
        
        pair_counter += 1

    return rows


def load_agentdojo_from_json(json_dir: str) -> List[Dict]:
    """
    Fallback: load AgentDojo from raw JSON files if the package is not installed.
    Expects files like: {json_dir}/{suite}_injection_tasks.json
    
    Each JSON has format:
    [{"task_id": "...", "injection": "...", "goal": "..."}, ...]
    """
    rows = []
    pair_counter = 0

    for suite in AGENTDOJO_SUITES:
        json_path = os.path.join(json_dir, f"{suite}_injection_tasks.json")
        if not os.path.exists(json_path):
            print(f"  [skip] {json_path} not found")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)

        for task in tasks:
            pair_id = f"agentdojo_{suite}_{pair_counter}"

            injected_text = task.get("injection", task.get("goal", ""))
            benign_text = task.get("user_task", task.get("benign", ""))

            rows.append({
                "id": f"{pair_id}_injected",
                "prompt": injected_text,
                "label": "injected",
                "pair_id": pair_id,
                "source": "agentdojo",
                "suite": suite,
                "task_id": str(task.get("task_id", pair_counter)),
            })

            rows.append({
                "id": f"{pair_id}_benign",
                "prompt": benign_text,
                "label": "benign",
                "pair_id": pair_id,
                "source": "agentdojo",
                "suite": suite,
                "task_id": str(task.get("task_id", pair_counter)),
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
    print(f"  injected={injected}  benign={len(rows)-injected}")


def main():
    ap = argparse.ArgumentParser(description="Convert AgentDojo to project JSONL format")
    ap.add_argument("--out", default="data/raw/agentdojo.jsonl", help="Output JSONL path")
    ap.add_argument("--json_dir", default=None,
                    help="Directory with raw AgentDojo JSON files (fallback if package not installed)")
    args = ap.parse_args()

    # Try package import first
    rows = load_agentdojo_via_package()

    if not rows and args.json_dir:
        print("Package import failed, trying JSON fallback...")
        rows = load_agentdojo_from_json(args.json_dir)

    if not rows:
        print(
            "No data loaded. Options:\n"
            "  1. Install agentdojo: pip install agentdojo\n"
            "  2. Provide raw JSONs: --json_dir path/to/jsons"
        )
        return

    save_jsonl(rows, args.out)


if __name__ == "__main__":
    main()
