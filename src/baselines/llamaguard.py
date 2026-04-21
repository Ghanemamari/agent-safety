"""
LlamaGuard 3 Baseline
=====================
Uses Meta's LlamaGuard-3 to predict whether a prompt is safe or unsafe.
Requires `huggingface-cli login` to be executed if gated.
"""

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.eval.metrics import compute_metrics

@torch.no_grad()
def run_llamaguard(texts_test, y_test, model_id="meta-llama/Llama-Guard-3-8B"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu")
    except Exception as e:
        print(f"  [warning] Failed to load LlamaGuard ({e}). Skipping. Provide HF Token!")
        return {"auroc": np.nan, "f1": np.nan, "accuracy": np.nan}
        
    scores = []
    device = model.device
    
    for t in tqdm(texts_test, desc="LlamaGuard"):
        chat = [
            {"role": "user", "content": t}
        ]
        input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]
        
        output = model.generate(input_ids=input_ids, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
        generated_tokens = output[:, prompt_len:]
        response = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip().lower()
        
        # LlamaGuard responses start with "safe" or "unsafe"
        is_unsafe = 1.0 if "unsafe" in response else 0.0
        scores.append(is_unsafe)
        
    test_scores = np.array(scores)
    metrics = compute_metrics(y_test, test_scores, threshold=0.5)
    
    # cleanup
    del model
    del tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return metrics
