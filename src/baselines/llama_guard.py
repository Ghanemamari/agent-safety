"""
Llama Guard 3 Baseline  (Meta, 2024)
HuggingFace: meta-llama/Llama-Guard-3-8B

Llama Guard 3 is an LLM fine-tuned for safety classification.
This module wraps it as a drop-in baseline that produces AUROC scores
in the same format as our linear probe, enabling direct comparison.

Usage:
    python -m src.baselines.llama_guard \
        --input data/raw/injecagent.jsonl \
        --out data/processed/injecagent_llama_guard_metrics.json

Note: Requires Llama Guard 3 weights access.
      Free access: meta-llama/Llama-Guard-3-1B  (smaller, no gating)
      Gated access: meta-llama/Llama-Guard-3-8B (requires HuggingFace approval)
"""

import argparse
import json
import os
from typing import List, Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.eval.metrics import compute_metrics
from src.utils.io import read_jsonl, write_json

# ─── Default model — use 1B version (ungated) for reproducibility ────────────
DEFAULT_MODEL = "meta-llama/Llama-Guard-3-1B"

# Safety token IDs in Llama Guard vocabulary
SAFE_TOKEN = "safe"
UNSAFE_TOKEN = "unsafe"


class LlamaGuardClassifier:
    """
    Wraps Llama Guard 3 as a binary safe/unsafe classifier.
    Returns a scalar score in [0, 1] representing P(unsafe).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, load_in_4bit: bool = False):
        print(f"Loading Llama Guard: {model_name}")
        kwargs = {"device_map": "auto", "trust_remote_code": True}
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
        else:
            kwargs["torch_dtype"] = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

        # Get token IDs for "safe" and "unsafe"
        self.safe_id = self.tokenizer.convert_tokens_to_ids(SAFE_TOKEN)
        self.unsafe_id = self.tokenizer.convert_tokens_to_ids(UNSAFE_TOKEN)

        if self.safe_id == self.tokenizer.unk_token_id:
            # Fallback: find via encode
            self.safe_id = self.tokenizer.encode(SAFE_TOKEN, add_special_tokens=False)[0]
            self.unsafe_id = self.tokenizer.encode(UNSAFE_TOKEN, add_special_tokens=False)[0]

    def _build_llama_guard_prompt(self, user_text: str) -> str:
        """
        Format prompt according to Llama Guard's expected input format.
        Uses the conversation template: [INST] User: {text} [/INST]
        """
        # Llama Guard 3 conversation format
        conversation = [
            {"role": "user", "content": user_text}
        ]
        try:
            # Use chat template if available
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback manual format
            prompt = f"[INST] {user_text} [/INST]"
        return prompt

    @torch.no_grad()
    def score(self, text: str, max_length: int = 512) -> float:
        """
        Returns P(unsafe) ∈ [0, 1] for a given text.
        Higher = more likely to be unsafe/injected.
        """
        prompt = self._build_llama_guard_prompt(text)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token logits

        # Get logits for safe and unsafe tokens
        log_safe = logits[self.safe_id].float().item()
        log_unsafe = logits[self.unsafe_id].float().item()

        # Softmax over just these two tokens → P(unsafe)
        max_logit = max(log_safe, log_unsafe)
        exp_safe = (log_safe - max_logit)
        exp_unsafe = (log_unsafe - max_logit)
        import math
        p_unsafe = math.exp(exp_unsafe) / (math.exp(exp_safe) + math.exp(exp_unsafe))

        return p_unsafe

    def score_batch(self, texts: List[str], max_length: int = 512) -> List[float]:
        return [self.score(t, max_length) for t in tqdm(texts, desc="LlamaGuard scoring")]


def evaluate_llama_guard(
    input_jsonl: str,
    model_name: str = DEFAULT_MODEL,
    out_json: str = None,
    load_in_4bit: bool = False,
    max_length: int = 512,
) -> Dict:
    """
    Run Llama Guard on a JSONL file and compute classification metrics.
    Returns metrics dict in the same format as compute_metrics().
    """
    from src.extract.extract_activations import normalize_label

    rows = read_jsonl(input_jsonl)
    texts = [r["prompt"] for r in rows]
    y_true = np.array([normalize_label(r["label"]) for r in rows])

    classifier = LlamaGuardClassifier(model_name=model_name, load_in_4bit=load_in_4bit)
    scores = classifier.score_batch(texts, max_length=max_length)
    probs = np.array(scores)

    metrics = compute_metrics(y_true, probs)
    metrics.update({
        "baseline": "llama_guard",
        "model_name": model_name,
        "n_samples": len(y_true),
        "input": input_jsonl,
    })

    print(f"\nLlama Guard Results on {os.path.basename(input_jsonl)}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        write_json(out_json, metrics)
        print(f"\nSaved metrics → {out_json}")

    return metrics


def main():
    ap = argparse.ArgumentParser(description="Llama Guard 3 baseline evaluation")
    ap.add_argument("--input", required=True, help="Path to prompts JSONL")
    ap.add_argument("--out", default=None, help="Path to output metrics JSON")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Llama Guard model name/path")
    ap.add_argument("--load4bit", action="store_true", help="Load model in 4-bit quantization")
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    evaluate_llama_guard(
        input_jsonl=args.input,
        model_name=args.model,
        out_json=args.out,
        load_in_4bit=args.load4bit,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
