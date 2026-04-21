"""
LLM-LAT Benign Dataset Loader
=============================
Downloads the LLM-LAT/benign-dataset from Hugging Face using the Datasets Server API.
This bypasses the need for the `datasets` library or `pyarrow` which fail to install due to DLL restrictions.

Usage:
    python -m src.datasets.load_llmlat --out data/raw/llm_lat_benign.jsonl --download
"""

import argparse
import json
import os
import urllib.request
import time
from typing import List, Dict

API_BASE = "https://datasets-server.huggingface.co/rows?dataset=LLM-LAT/benign-dataset&config=default&split=train"

def load_llmlat_benign(num_samples: int = 2500, download: bool = True, out_path: str = None) -> List[Dict]:
    rows = []
    
    if out_path and os.path.exists(out_path) and not download:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    print(f"Downloading ~{num_samples} benign prompts from LLM-LAT/benign-dataset...")
    
    limit = 100
    offset = 0
    while len(rows) < num_samples:
        url = f"{API_BASE}&offset={offset}&length={limit}"
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            if 'error' in data:
                print(f"API Error: {data['error']}")
                break
                
            batch = data.get('rows', [])
            if not batch:
                break
                
            for item in batch:
                prompt_text = item['row']['prompt']
                rows.append({
                    "id": f"llmlat_benign_{len(rows)}",
                    "prompt": prompt_text,
                    "label": "benign",
                    "pair_id": f"llmlat_{len(rows)}",
                    "source": "llm-lat",
                    "attack_type": "none"
                })
                
                if len(rows) >= num_samples:
                    break
                    
            print(f"  Fetched {len(rows)}/{num_samples}...")
            offset += limit
            time.sleep(0.1)  # rate limit
            
        except Exception as e:
            print(f"Failed to fetch batch at offset {offset}: {e}")
            break

    return rows

def save_jsonl(rows: List[Dict], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} benign rows -> {out_path}")

def main():
    ap = argparse.ArgumentParser("LLM-LAT Benign loader")
    ap.add_argument("--out", default="data/raw/llmlat_benign.jsonl")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--num_samples", type=int, default=2500)
    args = ap.parse_args()
    
    rows = load_llmlat_benign(args.num_samples, download=args.download, out_path=args.out)
    if rows:
        save_jsonl(rows, args.out)

if __name__ == "__main__":
    main()
