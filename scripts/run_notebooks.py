#!/usr/bin/env python3
"""Execute every notebook headlessly and collect per-notebook results.

Each notebook is run with the kernel's working directory set to the notebook's
own folder, which is what JupyterLab does. That is the condition the bootstrap
cell has to survive, so passing runs here mean the notebook also opens and runs
in the UI.

A per-cell timeout interrupts a long training cell instead of failing the run,
so heavy notebooks still yield partial output (epoch lines) that we can report.

    python scripts/run_notebooks.py                       # run everything
    python scripts/run_notebooks.py --only 'ml/basics/*'  # subset
    python scripts/run_notebooks.py --resume              # skip already-ok
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", ".git", "__pycache__", ".ipynb_checkpoints", "runs", "node_modules"}

# Cheapest first, so a long batch still produces the widest coverage.
ORDER = [
    "ml/basics/knn.ipynb",
    "ml/basics/perceptron.ipynb",
    "ml/basics/decision_tree.ipynb",
    "ml/basics/naive_bayes.ipynb",
    "ml/basics/linear_regression.ipynb",
    "ml/basics/logistic_regression.ipynb",
    "ml/basics/pca.ipynb",
    "ml/basics/k_means.ipynb",
    "ml/basics/svm.ipynb",
    "ml/basics/tsne.ipynb",
    "rl/dqn/dqn.ipynb",
    "graph/gcn/gcn.ipynb",
    "nlp/word2vec/word2vec.ipynb",
    "cv/simplecnn/cnn.ipynb",
    "gen/dcgan/dcgan.ipynb",
    "nlp/lstm/lstm.ipynb",
    "nlp/lora/lora.ipynb",
    "cv/mobilenet/mobilenet.ipynb",
    "nlp/bert/bert.ipynb",
    "nlp/gpt/gpt.ipynb",
    "cv/unet/unet.ipynb",
    "cv/vit/vit.ipynb",
    "cv/resnet18/resnet18.ipynb",
    "cv/yolo/yolo.ipynb",
    "cv/resnet34/resnet34.ipynb",
    "selfsup/simclr/simclr.ipynb",
]

# Lines worth surfacing in the report: epoch progress, accuracy, loss, scores.
INTERESTING = re.compile(
    r"(epoch|acc|accuracy|loss|fid|auc|precision|recall|r2|r²|silhouette|"
    r"nmi|ari|psnr|perplexity|reward|iou|map\b|params|parameters)",
    re.IGNORECASE,
)


def discover():
    found = []
    for path in REPO_ROOT.rglob("*.ipynb"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        found.append(path.relative_to(REPO_ROOT).as_posix())
    ordered = [p for p in ORDER if p in found]
    return ordered + sorted(p for p in found if p not in ordered)


def stream_text(outputs):
    parts = []
    for out in outputs:
        if out.get("output_type") == "stream":
            text = out.get("text", "")
            parts.append(text if isinstance(text, str) else "".join(text))
        elif out.get("output_type") in ("execute_result", "display_data"):
            data = out.get("data", {}).get("text/plain", "")
            parts.append(data if isinstance(data, str) else "".join(data))
        elif out.get("output_type") == "error":
            parts.append(f"__ERROR__ {out.get('ename')}: {out.get('evalue')}")
    return "".join(parts)


def summarize(nb, elapsed, timeout):
    """Turn an executed notebook into a compact result record."""
    errors, interesting, epochs = [], [], []
    timed_out = False
    executed = 0

    for idx, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            executed += 1
        text = stream_text(cell.get("outputs", []))
        for line in text.splitlines():
            line = line.rstrip()
            if not line.strip():
                continue
            if "__ERROR__" in line:
                errors.append(f"cell {idx}: {line.replace('__ERROR__ ', '')}")
            if INTERESTING.search(line) and len(line) < 200:
                interesting.append(line)
            m = re.search(r"[Ee]poch\s*\[?\s*(\d+)\s*/\s*(\d+)", line)
            if m:
                epochs.append(f"{m.group(1)}/{m.group(2)}")
        # Per-cell timing lives in metadata when record_timing is on.
        dur = cell.get("metadata", {}).get("execution", {}).get("duration")
        if dur is not None and timeout and dur >= timeout * 0.98:
            timed_out = True

    if errors and not timed_out:
        status = "error"
    elif timed_out:
        status = "partial"
    elif errors:
        status = "partial"
    else:
        status = "ok"

    return {
        "status": status,
        "elapsed_s": round(elapsed, 1),
        "cells_executed": executed,
        "errors": errors[:5],
        "epochs_progress": epochs[-1] if epochs else None,
        "epochs_count": len(epochs),
        "highlights": interesting[-6:],
    }


def run_one(rel, timeout, out_root):
    path = REPO_ROOT / rel
    started = time.time()
    nb = nbf.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=True,
        interrupt_on_timeout=True,
        record_timing=True,
        resources={"metadata": {"path": str(path.parent)}},
    )
    try:
        client.execute()
    except CellExecutionError as exc:  # pragma: no cover - defensive
        elapsed = time.time() - started
        return {"status": "error", "elapsed_s": round(elapsed, 1),
                "errors": [str(exc)[:400]], "highlights": [], "epochs_progress": None,
                "epochs_count": 0, "cells_executed": 0}

    elapsed = time.time() - started
    record = summarize(nb, elapsed, timeout)
    dest = out_root / "executed" / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, dest)
    return record


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=1200, help="per-cell timeout in seconds")
    ap.add_argument("--budget", type=int, default=0, help="stop after this many seconds (0 = no cap)")
    ap.add_argument("--only", default=None, help="glob-like substring filter on the path")
    ap.add_argument("--resume", action="store_true", help="skip notebooks already recorded as ok")
    ap.add_argument("--outdir", default="runs/notebooks")
    args = ap.parse_args(argv)

    out_root = REPO_ROOT / args.outdir
    out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / "summary.json"

    results = {}
    if results_path.exists():
        results = json.loads(results_path.read_text())

    targets = discover()
    if args.only:
        targets = [t for t in targets if args.only in t]
    if args.resume:
        targets = [t for t in targets if results.get(t, {}).get("status") != "ok"]

    started = time.time()
    print(f"executing {len(targets)} notebooks (timeout={args.timeout}s/cell)\n", flush=True)

    for i, rel in enumerate(targets, 1):
        spent = time.time() - started
        if args.budget and spent > args.budget:
            print(f"budget of {args.budget}s exhausted after {i-1} notebooks", flush=True)
            break
        print(f"[{i}/{len(targets)}] {rel} ...", flush=True)
        record = run_one(rel, args.timeout, out_root)
        record["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        results[rel] = record
        results_path.write_text(json.dumps(results, indent=2) + "\n")
        tail = record["highlights"][-1] if record["highlights"] else ""
        print(f"    {record['status']:8s} {record['elapsed_s']:7.1f}s  {tail[:100]}", flush=True)

    write_markdown(results, out_root / "summary.md")
    counts = {}
    for rec in results.values():
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    print(f"\ndone: {counts}")
    return 0


def write_markdown(results, dest):
    order = {"ok": 0, "partial": 1, "error": 2}
    lines = [
        "# Notebook execution report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "| Notebook | Status | Seconds | Epoch reached | Last highlight |",
        "|---|---|---|---|---|",
    ]
    for rel in sorted(results, key=lambda r: (order.get(results[r]["status"], 9), r)):
        rec = results[rel]
        hl = (rec["highlights"][-1] if rec["highlights"] else "").replace("|", "\\|")[:90]
        lines.append(
            f"| `{rel}` | {rec['status']} | {rec['elapsed_s']:.0f} | "
            f"{rec.get('epochs_progress') or '-'} | {hl} |"
        )
    lines.append("")
    dest.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
