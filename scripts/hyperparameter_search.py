"""
Hyperparameter search: grid search wrapper for any train script.

Usage:
    uv run python scripts/hyperparameter_search.py \
        --model cv.simplecnn.train \
        --config cv/simplecnn/config.yaml \
        --param lr:0.0001,0.001,0.01 \
        --param batch_size:64,128 \
        --epochs 10

Output: results/hyperparameter_search.csv
"""

import argparse
import csv
import itertools
import os
import re
import sys
import io
import time
import importlib
import yaml

from utils.config import save_config


def parse_param(param_str):
    key, vals = param_str.split(":", 1)
    values = []
    for v in vals.split(","):
        v = v.strip()
        for cast in (int, float):
            try:
                values.append(cast(v))
                break
            except (ValueError, TypeError):
                continue
        else:
            values.append(v)
    return key, values


def extract_metrics(text):
    """Extract final loss and accuracy from training output."""
    loss = re.findall(r"Loss:\s*([\d.]+)", text)
    acc = re.findall(r"Acc:\s*([\d.]+)%", text)
    return (float(loss[-1]) if loss else None,
            float(acc[-1]) if acc else None)


def main():
    parser = argparse.ArgumentParser(description="Grid search hyperparameter optimization")
    parser.add_argument("--model", required=True, help="e.g. cv.simplecnn.train")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--param", action="append", required=True, help="key:val1,val2,...")
    parser.add_argument("--epochs", type=int, default=10, help="Override num_epochs")
    args = parser.parse_args()

    param_keys, param_grids = [], []
    for p in args.param:
        key, values = parse_param(p)
        param_keys.append(key)
        param_grids.append(values)

    total = 1
    for g in param_grids:
        total *= len(g)

    print(f"Hyperparameter Grid Search")
    print(f"  Model:   {args.model}")
    print(f"  Config:  {args.config}")
    print(f"  Params:  {param_keys} = {param_grids}")
    print(f"  Combos:  {total}")
    print(f"  Epochs:  {args.epochs}")
    print()

    results_file = "results/hyperparameter_search.csv"
    os.makedirs("results", exist_ok=True)

    # Save original config for restoration at the end.
    with open(args.config) as f:
        original_config = yaml.safe_load(f)

    module_path, func_name = args.model.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    train_fn = getattr(mod, func_name)

    with open(results_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model"] + param_keys + ["final_loss", "final_acc", "total_time_s"])

        for combo_idx, values in enumerate(itertools.product(*param_grids), 1):
            # Load and modify config.
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            for k, v in zip(param_keys, values):
                cfg[k] = v
            cfg["num_epochs"] = args.epochs
            cfg["use_wandb"] = False
            save_config(cfg, args.config)

            combo_str = ", ".join(f"{k}={v}" for k, v in zip(param_keys, values))
            print(f"[{combo_idx}/{total}] {combo_str}", flush=True)

            # Run training with stdout capture.
            t0 = time.time()
            captured = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured
            try:
                train_fn()
            finally:
                sys.stdout = old_stdout

            elapsed = time.time() - t0
            output = captured.getvalue()
            print(output, end="")

            final_loss, final_acc = extract_metrics(output)
            print(f"  → Loss: {final_loss}, Acc: {final_acc}%, Time: {elapsed:.0f}s", flush=True)

            writer.writerow([args.model] + list(values) +
                            [f"{final_loss:.4f}" if final_loss else "",
                             f"{final_acc:.2f}" if final_acc else "",
                             f"{elapsed:.0f}"])
            csvfile.flush()

    # Restore original config.
    save_config(original_config, args.config)
    print(f"\nDone. Results saved to {results_file}")
    print(f"View with: column -t -s, {results_file}")


if __name__ == "__main__":
    main()
