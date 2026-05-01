import argparse
import json
import os
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line.strip()))
    return rows

def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} ablated rows to {path}")

def get_top_features(texts, labels, top_n=50):
    vec = TfidfVectorizer(ngram_range=(1, 1), lowercase=True, max_features=10000)
    X = vec.fit_transform(texts)
    y = np.array([1 if str(l).lower() in ["1", "true", "unsafe", "malicious", "injected"] else 0 for l in labels])
    
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)
    
    feature_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]
    
    # Sort by absolute coefficient value
    top_indices = np.argsort(np.abs(coefs))[::-1][:top_n]
    top_words = feature_names[top_indices]
    
    print(f"Top {top_n} predictive words (by absolute logistic regression coefficient):")
    for i, idx in enumerate(top_indices[:10]):
        print(f"  {i+1}. '{feature_names[idx]}' (coef: {coefs[idx]:.4f})")
    
    return list(top_words)

def ablate_text(text, words_to_remove):
    # Case insensitive removal, word boundary preserved
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, words_to_remove)) + r')\b', re.IGNORECASE)
    return pattern.sub("[MASK]", text)

def run_ablation(input_paths, top_n=50):
    all_texts = []
    all_labels = []
    dataset_rows = {}
    
    for path in input_paths:
        rows = load_jsonl(path)
        dataset_rows[path] = rows
        all_texts.extend([r["prompt"] for r in rows])
        all_labels.extend([r["label"] for r in rows])
        
    top_words = get_top_features(all_texts, all_labels, top_n=top_n)
    
    for path, rows in dataset_rows.items():
        ablated_rows = []
        for r in rows:
            new_r = r.copy()
            new_r["prompt"] = ablate_text(r["prompt"], top_words)
            ablated_rows.append(new_r)
            
        base, ext = os.path.splitext(path)
        out_path = f"{base}_ablated{ext}"
        save_jsonl(ablated_rows, out_path)
        
    print("\n--- Next Steps ---")
    print("Run extraction on the ablated files to get new hidden states:")
    print("  python -m src.extract.extract_activations --model meta-llama/Llama-2-7b-chat-hf --input data/raw/injecagent_ablated.jsonl --out data/processed/injecagent_ablated.npz")
    print("Then evaluate the probes on these new NPZ files to check for performance collapse.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TF-IDF Baseline Artifact Ablation")
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to JSONL files to analyze and ablate (e.g. injecagent.jsonl advbench.jsonl)")
    parser.add_argument("--top_n", type=int, default=50, help="Number of top words to redact")
    args = parser.parse_args()
    
    run_ablation(args.inputs, top_n=args.top_n)
