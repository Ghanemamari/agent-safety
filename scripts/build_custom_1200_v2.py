import os
import json
import random
import urllib.request

random.seed(42)

def read_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows

def fetch_agentdojo(limit=200):
    rows = []
    print("Fetching AgentDojo data from HF datasets server API...")
    pair_counter = 0
    for offset in range(0, limit, 100):
        length = min(100, limit - offset)
        url = f"https://datasets-server.huggingface.co/rows?dataset=ffuuugor%2Fagentdojo-dump&config=default&split=train&offset={offset}&length={length}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
        except Exception as e:
            print(f"Error touching HF Datasets server at offset {offset}: {e}")
            break
            
        for item in data.get("rows", []):
            r = item.get("row", {})
            suite_name = r.get("suite", "unknown")
            injected_instruction = r.get("injection", None) or r.get("prompt", None) or str(r)
            pair_id = f"agentdojo_extended_{suite_name}_{pair_counter}"
            
            rows.append({
                "id": f"{pair_id}_injected",
                "prompt": f"[AgentDojo:{suite_name}] {injected_instruction}",
                "label": "injected",
                "pair_id": pair_id,
                "source": "agentdojo",
                "suite": suite_name,
                "task_id": str(pair_counter),
            })
            pair_counter += 1

    return rows

def main():
    out_path = "data/raw/custom_1200_v2.jsonl"
    final_rows = []

    # 1. LLMLat (600 benign)
    print("Loading LLMLat benign...")
    llmlat_rows = read_jsonl("data/raw/llmlat_benign.jsonl")
    benign_rows = [r for r in llmlat_rows if r.get("label") in ["benign", "safe", 0, "0"]]
    selected_benign = random.sample(benign_rows, min(600, len(benign_rows)))
    final_rows.extend(selected_benign)
    print(f"  -> Added {len(selected_benign)} benign samples.")

    # 2. InjecAgent (200 malicious)
    print("Loading InjecAgent...")
    injec_rows = read_jsonl("data/raw/injecagent.jsonl")
    injec_mal = [r for r in injec_rows if r.get("label") in ["injected", "malicious", "unsafe", 1, "1"]]
    selected_injec = random.sample(injec_mal, min(200, len(injec_mal)))
    final_rows.extend(selected_injec)
    print(f"  -> Added {len(selected_injec)} malicious samples.")

    # 3. AdvBench (200 malicious)
    print("Loading AdvBench...")
    adv_rows = read_jsonl("data/raw/advbench.jsonl")
    adv_mal = [r for r in adv_rows if r.get("label") in ["injected", "malicious", "unsafe", 1, "1"]]
    selected_adv = random.sample(adv_mal, min(200, len(adv_mal)))
    final_rows.extend(selected_adv)
    print(f"  -> Added {len(selected_adv)} malicious samples.")

    # 4. AgentDojo (200 malicious)
    dojo_mal = fetch_agentdojo(limit=200)
    selected_dojo = random.sample(dojo_mal, min(200, len(dojo_mal)))
    final_rows.extend(selected_dojo)
    print(f"  -> Added {len(selected_dojo)} malicious samples.")

    random.shuffle(final_rows)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(final_rows)} rows to {out_path}.")

if __name__ == "__main__":
    main()
