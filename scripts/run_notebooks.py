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

Filters (``--only`` / ``--skip``) accept a glob (``ml/basics/*``) or a plain
substring (``ml/basics``), and may be repeated. ``--fail-on`` turns the report
into a gate: it exits 1 when a run reaches the given status, which is what CI
uses.
"""

import argparse
import fnmatch
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

# Cheapest first, so a long batch still produces the widest coverage. Seconds
# are the measured CPU time on a GB10 box (see runs/notebooks/summary.json).
ORDER = [
    "ml/basics/decision_tree.ipynb",        # 5
    "graph/gcn/gcn.ipynb",                  # 5
    "ml/basics/perceptron.ipynb",           # 6
    "ml/basics/pca.ipynb",                  # 7
    "ml/basics/svm.ipynb",                  # 7
    "ml/basics/dbscan.ipynb",               # 10
    "ml/basics/naive_bayes.ipynb",          # 10
    "ml/basics/knn.ipynb",                  # 10
    "ml/basics/k_means.ipynb",              # 12
    "rl/dqn/dqn.ipynb",                     # 14
    "gen/autoencoder/autoencoder.ipynb",    # 14
    "ml/basics/linear_regression.ipynb",    # 16
    "selfsup/deep_cluster/dec.ipynb",       # 19
    "nlp/lora/lora.ipynb",                  # 26
    "nlp/bert/bert.ipynb",                  # 27
    "ml/basics/gmm.ipynb",                  # 28
    "ml/basics/logistic_regression.ipynb",  # 50
    "nlp/lstm/lstm.ipynb",                  # 52
    "nlp/gpt/gpt.ipynb",                    # 69
    "ml/mlp/mlp.ipynb",                     # 77
    "gen/dcgan/dcgan.ipynb",                # 84
    "cv/simplecnn/cnn.ipynb",               # 102
    "nlp/word2vec/word2vec.ipynb",          # 133
    "cv/resnet18/resnet18.ipynb",           # 208
    "cv/mobilenet/mobilenet.ipynb",         # 252
    "selfsup/mae/mae.ipynb",                # 310
    "cv/vit/vit.ipynb",                     # 322
    "nlp/seq2seq/seq2seq.ipynb",            # 339
    "ml/basics/tsne.ipynb",                 # 471
    "gen/vae/vae.ipynb",                    # 614
    "selfsup/simclr/simclr.ipynb",          # 674
    "cv/yolo/yolo.ipynb",                   # 724
    "selfsup/byol/byol.ipynb",              # 866
    "cv/unet/unet.ipynb",                   # 1882
    "gen/ddpm/ddpm.ipynb",                  # 2187
    "cv/resnet34/resnet34.ipynb",           # 3119
    "cv/resnet50/resnet50.ipynb",           # 3292
]

STATUS_ORDER = {"ok": 0, "partial": 1, "error": 2}

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


def matches(rel, patterns):
    """True when `rel` matches any pattern as a glob or as a plain substring."""
    return any(fnmatch.fnmatch(rel, pattern) or pattern in rel for pattern in patterns)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=1200, help="per-cell timeout in seconds")
    ap.add_argument("--budget", type=int, default=0, help="stop after this many seconds (0 = no cap)")
    ap.add_argument("--only", action="append", default=[], metavar="PATTERN",
                    help="keep notebooks matching this glob or substring (repeatable)")
    ap.add_argument("--skip", action="append", default=[], metavar="PATTERN",
                    help="drop notebooks matching this glob or substring (repeatable)")
    ap.add_argument("--fail-on", choices=("none", "partial", "error"), default="none",
                    help="exit 1 when any executed notebook reaches this status (default: none)")
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
        targets = [t for t in targets if matches(t, args.only)]
    if args.skip:
        targets = [t for t in targets if not matches(t, args.skip)]
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
        # Print the errors too: the first one is the cause, the rest are cascade,
        # and a CI log that only shows the status sends the reader to the notebook.
        for err in record["errors"]:
            print(f"      ! {err[:200]}", flush=True)

    write_markdown(results, out_root / "summary.md")
    counts = {}
    for rec in results.values():
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    print(f"\ndone: {counts}")

    if args.fail_on != "none":
        threshold = STATUS_ORDER[args.fail_on]
        failing = [
            rel for rel in targets
            if rel in results and STATUS_ORDER.get(results[rel]["status"], 9) >= threshold
        ]
        if failing:
            print(f"\nfail-on={args.fail_on}: {len(failing)} notebook(s) below par", file=sys.stderr)
            for rel in failing:
                print(f"  {rel}: {results[rel]['status']}", file=sys.stderr)
            return 1
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
