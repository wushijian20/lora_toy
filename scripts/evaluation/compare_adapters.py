"""
Compare existing LoRA adapters on a shared evaluation set.
Outputs a CSV with loss/perplexity per adapter.
"""

import os
import math
import argparse
from typing import List

import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

DEFAULT_PROMPTS = [
    "Once upon a time, in a small village,",
    "The quick brown fox jumps over the lazy dog.",
    "In a shocking turn of events, the stock market",
    "Doctor: How long have you had these symptoms?",
    "To be, or not to be, that is the question:",
    "Breaking news: scientists have discovered",
    "Write a short poem about the ocean.",
    "User: Can you explain quantum computing?",
    "The recipe calls for two cups of flour and",
    "The mystery deepened as the detective found",
]

POSSIBLE_TEXT_COLS = ["text", "Text", "content", "Content", "prompt", "review", "body", "message"]


def load_eval_texts(path: str | None, max_rows: int) -> List[str]:
    if not path:
        return DEFAULT_PROMPTS

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

        # Approximate token count
        batch_tokens = attention_mask.sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    return total_loss / max(1, total_tokens)


def main():
    parser = argparse.ArgumentParser(description="Compare LoRA adapters on a shared eval set")
    parser.add_argument("--adapters-dir", default="./adapters", help="Path to adapters folder")
    parser.add_argument("--base-model", default="gpt2", help="Base model id")
    parser.add_argument("--eval-file", default=None, help="Optional eval file (.txt/.csv/.jsonl)")
    parser.add_argument("--max-rows", type=int, default=200, help="Max rows from eval file")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", default=None, help="cuda or cpu (auto if not set)")
    parser.add_argument("--output", default="./outputs/metrics/adapter_eval.csv", help="CSV output path")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    texts = load_eval_texts(args.eval_file, args.max_rows)
    adapters = get_adapters(args.adapters_dir)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    results = []

    for adapter_path in sorted(adapters):
        adapter_name = os.path.basename(adapter_path)
        print(f"Evaluating: {adapter_name}")

        base_model = GPT2LMHeadModel.from_pretrained(args.base_model)
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.to(device)

        avg_loss = compute_loss(model, tokenizer, texts, device, args.max_length, args.batch_size)
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

        results.append({
            "adapter": adapter_name,
            "avg_loss": avg_loss,
            "perplexity": ppl,
            "eval_samples": len(texts),
            "max_length": args.max_length,
        })

        del model
        del base_model
        torch.cuda.empty_cache()

    df = pd.DataFrame(results).sort_values("perplexity")
    df.to_csv(args.output, index=False)

    print(f"\n✅ Saved results to {args.output}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
