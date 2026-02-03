"""
Generate side-by-side samples from existing LoRA adapters.
Outputs a CSV for easy manual comparison.
"""

import os
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


def load_prompts(path: str | None, max_rows: int) -> List[str]:
    if not path:
        return DEFAULT_PROMPTS

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")

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

    raise ValueError("Unsupported file type. Use .txt or .csv")


def get_adapters(adapters_dir: str) -> List[str]:
    if not os.path.isdir(adapters_dir):
        raise FileNotFoundError(f"Adapters directory not found: {adapters_dir}")
    return [
        os.path.join(adapters_dir, d)
        for d in os.listdir(adapters_dir)
        if os.path.isdir(os.path.join(adapters_dir, d))
    ]


def generate_for_adapter(
    adapter_path: str,
    prompts: List[str],
    tokenizer: GPT2Tokenizer,
    base_model_id: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> List[dict]:
    torch.manual_seed(seed)

    base_model = GPT2LMHeadModel.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model.eval()

    results = []

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "generated": text})

    del model
    del base_model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate samples for LoRA adapters")
    parser.add_argument("--adapters-dir", default="./adapters", help="Path to adapters folder")
    parser.add_argument("--base-model", default="gpt2", help="Base model id")
    parser.add_argument("--prompts", default=None, help="Optional prompt file (.txt/.csv)")
    parser.add_argument("--max-rows", type=int, default=20, help="Max prompts to use")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default=None, help="cuda or cpu (auto if not set)")
    parser.add_argument("--output", default="./outputs/samples/adapter_generations.csv", help="CSV output path")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prompts = load_prompts(args.prompts, args.max_rows)
    adapters = get_adapters(args.adapters_dir)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    rows = []

    for adapter_path in sorted(adapters):
        adapter_name = os.path.basename(adapter_path)
        print(f"Generating: {adapter_name}")

        samples = generate_for_adapter(
            adapter_path,
            prompts,
            tokenizer,
            args.base_model,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.seed,
        )

        for sample in samples:
            rows.append({
                "adapter": adapter_name,
                "prompt": sample["prompt"],
                "generated": sample["generated"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    print(f"\n✅ Saved samples to {args.output}")


if __name__ == "__main__":
    main()
