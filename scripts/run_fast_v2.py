import json
import numpy as np
from src.eval.run_experiments import stratified_split_70_30, label_to_int
from src.baselines.tfidf import run_tfidf_numpy
from src.baselines.semantic_mlp import run_semantic_mlp

def main():
    jsonl_path = "data/raw/custom_1200_v2.jsonl"
    texts = []
    y_txt = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                texts.append(d["prompt"])
                y_txt.append(label_to_int(d["label"]))
    y_txt = np.array(y_txt, dtype=np.float32)

    tr_idx, te_idx = stratified_split_70_30(y_txt)
    t_tr = [texts[i] for i in tr_idx]
    t_te = [texts[i] for i in te_idx]
    y_tr = y_txt[tr_idx]
    y_te = y_txt[te_idx]

    print("\n--- TF-IDF + LR ---")
    m = run_tfidf_numpy(t_tr, y_tr, t_te, y_te)
    print(f"AUROC: {m['auroc']:.4f} | F1: {m['f1']:.4f} | Acc: {m['accuracy']:.4f}")

if __name__ == "__main__":
    main()
