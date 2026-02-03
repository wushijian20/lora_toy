"""
Domain-specific evaluation for existing LoRA adapters.
Uses a domain config file to map domain -> eval file.
"""

import os
import math
import json
import argparse
from typing import Dict, List

import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

POSSIBLE_TEXT_COLS = ["text", "Text", "content", "Content", "prompt", "review", "body", "message"]

DEFAULT_DOMAIN_PROMPTS = {
    "poetry": [
        "Write a short poem about winter.",
        "Compose a haiku about the moon.",
    ],
    "dialogue": [
        "User: I feel anxious. Assistant:",
        "Doctor: Tell me about your symptoms. Patient:",
    ],
    "finance": [
        "The quarterly earnings report shows",
        "Market volatility increased as",
    ],
    "literature": [
        "In the quiet of the library,",
        "The hero returned after years away,",
    ],
    "casual": [
        "Hey, what are you up to today?",
        "That movie was actually pretty good because",
    ],
}


def load_texts(path: str | None, max_rows: int, fallback: List[str]) -> List[str]:
    if not path:
        return fallback

    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines[:max_rows] if max_rows else lines

    if path.endswith(".csv"):
        df = pd.read_csv(path)
        text_col = next((c for c in POSSIBLE_TEXT_COLS if c in df.columns), None)
        if not text_col:
            raise ValueError(f"No text column found in CSV. Columns: {df.columns.tolist()}")
        texts = df[text_col].dropna().astype(str).tolist()
        return texts[:max_rows] if max_rows else texts

    if path.endswith(".jsonl"):
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = pd.read_json(line, typ="series")
                text_col = next((c for c in POSSIBLE_TEXT_COLS if c in obj.index), None)
                if text_col:
                    texts.append(str(obj[text_col]))
                if max_rows and len(texts) >= max_rows:
                    break
        if not texts:
            raise ValueError("No text found in JSONL file.")
        return texts

    raise ValueError("Unsupported file type. Use .txt, .csv, or .jsonl")


def get_adapters(adapters_dir: str) -> List[str]:
    if not os.path.isdir(adapters_dir):
        raise FileNotFoundError(f"Adapters directory not found: {adapters_dir}")
    return [
        os.path.join(adapters_dir, d)
        for d in os.listdir(adapters_dir)
        if os.path.isdir(os.path.join(adapters_dir, d))
    ]


def compute_loss(model, tokenizer, texts: List[str], device: str, max_length: int, batch_size: int) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

        batch_tokens = attention_mask.sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    return total_loss / max(1, total_tokens)


def load_domain_config(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Domain config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Domain-specific evaluation for LoRA adapters")
    parser.add_argument("--adapters-dir", default="./adapters", help="Path to adapters folder")
    parser.add_argument("--base-model", default="gpt2", help="Base model id")
    parser.add_argument("--domain-config", default="./config/domains_eval.json", help="Domain config JSON")
    parser.add_argument("--max-rows", type=int, default=200, help="Max rows from eval files")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", default=None, help="cuda or cpu (auto if not set)")
    parser.add_argument("--output", default="./outputs/metrics/domain_eval.csv", help="CSV output path")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    adapters = get_adapters(args.adapters_dir)
    config = load_domain_config(args.domain_config)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    rows = []

    for adapter_path in sorted(adapters):
        adapter_name = os.path.basename(adapter_path)
        print(f"Evaluating adapter: {adapter_name}")

        base_model = GPT2LMHeadModel.from_pretrained(args.base_model)
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.to(device)

        for domain, domain_cfg in config.get("domains", {}).items():
            eval_file = domain_cfg.get("eval_file")
            fallback = DEFAULT_DOMAIN_PROMPTS.get(domain, DEFAULT_DOMAIN_PROMPTS["literature"])
            texts = load_texts(eval_file, args.max_rows, fallback)

            avg_loss = compute_loss(model, tokenizer, texts, device, args.max_length, args.batch_size)
            ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

            rows.append({
                "adapter": adapter_name,
                "domain": domain,
                "avg_loss": avg_loss,
                "perplexity": ppl,
                "eval_samples": len(texts),
                "max_length": args.max_length,
            })

        del model
        del base_model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values(["domain", "perplexity"])
    df.to_csv(args.output, index=False)

    print(f"\n✅ Saved domain results to {args.output}")
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
