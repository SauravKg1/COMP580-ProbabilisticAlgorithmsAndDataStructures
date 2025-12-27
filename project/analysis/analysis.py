#!/usr/bin/env python3

"""
Compare Hokusai and Ring Buffer CMS across multiple width/depth settings on the
same dataset. Produces per-run metrics (JSON), a combined TSV summary, and
comparison plots under analysis/graphs/combined.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is on sys.path when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure matplotlib has a writable config/cache directory.
MPLCONFIGDIR = ROOT / "analysis" / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

from cms_hakusai.hokusai import run_experiment as run_hokusai
from cms_ring_buffer.ring_buffer import run_experiment as run_ring_buffer
from cms_burst_sketch.burst_sketch import run_experiment as run_burst
from cms_ada_sketch.ada_sketch import run_experiment as run_ada
from cms_burst_sketch.burst_sketch import (
    build_true_bucket_counts as burst_build_true_buckets,
    true_count_in_bucket_range as burst_true_count_range,
    load_events as load_events_burst,
)
from cms_ada_sketch.ada_sketch import (
    compute_true_decayed_counts_at_T as ada_true_at_T,
    AdaSketchCMS,
)

DATASET_DEFAULT = Path("kaggle_RC_2019-05_ngrams.csv")
GRAPHS_BASE = Path("analysis/graphs")
RESULTS_DIR = Path("analysis/results")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_results_table(rows: List[Dict], path: Path) -> None:
    _ensure_dir(path.parent)
    metrics = [
        "weighted_relative_l1",
        "mae",
        "rmse",
        "mean_relative_error",
        "total_counters",
        "estimated_bytes",
        "load_time_s",
        "stream_time_s",
        "eval_time_s",
        "total_time_s",
        "throughput_events_per_s",
    ]
    header = "model\tlabel\twidth\tdepth\t" + "\t".join(metrics)
    lines = [header]
    for row in rows:
        vals = [
            row["model"],
            row.get("label", ""),
            str(row.get("width", "")),
            str(row.get("depth", "")),
        ]
        m = row["metrics"]
        for key in metrics:
            val = m.get(key, float("nan"))
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        lines.append("\t".join(vals))
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_metric_bars(rows: List[Dict], metric: str, title: str, ylabel: str, out_path: Path) -> None:
    labels = [r.get("label", f"{r['model']}_w{r['width']}_d{r['depth']}") for r in rows]
    vals = [r["metrics"].get(metric, 0.0) for r in rows]
    colors = []
    for r in rows:
        if r["model"] == "hokusai":
            colors.append("#1f77b4")
        elif r["model"] == "ring_buffer":
            colors.append("#ff7f0e")
        elif r["model"] == "burst_sketch":
            colors.append("#2ca02c")
        else:
            colors.append("#9467bd")

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(rows)), 5))
    ax.bar(x, vals, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    for rect in ax.patches:
        height = rect.get_height()
        if metric in {"estimated_bytes", "total_counters", "throughput_events_per_s"}:
            label = str(int(round(height)))
        else:
            label = f"{height:.3f}"
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)


def _plot_error_vs_memory(rows: List[Dict], out_path: Path) -> None:
    """Scatter of error vs memory (log-log) with zoom for clustered points."""
    markers = {"hokusai": "o", "ring_buffer": "s", "burst_sketch": "D", "ada_sketch": "^"}
    colors = {"hokusai": "#1f77b4", "ring_buffer": "#ff7f0e", "burst_sketch": "#2ca02c", "ada_sketch": "#9467bd"}

    mems = [r["metrics"].get("estimated_bytes", 0.0) for r in rows]
    errs = [r["metrics"].get("weighted_relative_l1", 0.0) for r in rows]
    mem_min, mem_max = min(mems), max(mems)
    err_min, err_max = min(errs), max(errs)

    fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(12, 6))

    def _scatter(ax, rows_subset, annotate_top: bool = False):
        seen = set()
        handles = []
        for r in rows_subset:
            m = r["metrics"]
            mem = m.get("estimated_bytes", 0.0)
            err = m.get("weighted_relative_l1", 0.0)
            ax.scatter(mem, err, marker=markers.get(r["model"], "o"), color=colors.get(r["model"], "#888"))
        # legend
        for r in rows_subset:
            model = r["model"]
            if model in seen:
                continue
            seen.add(model)
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=markers.get(model, "o"),
                    color="w",
                    markerfacecolor=colors.get(model, "#888"),
                    markersize=7,
                    label=model,
                )
            )
        ax.legend(handles=handles, title="Model", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Estimated memory (bytes)")
        ax.set_ylabel("Weighted relative L1 (lower is better)")

    # Main plot: full range, annotate top memory points
    _scatter(ax_main, rows, annotate_top=True)
    ax_main.set_title("Error vs Memory (log-log, full range)")

    # Zoomed plot: focus on lower-memory cluster (exclude top 10% memory)
    threshold = np.percentile(mems, 90) if len(mems) > 1 else mem_max
    cluster_rows = [r for r in rows if r["metrics"].get("estimated_bytes", 0.0) <= threshold]
    if cluster_rows:
        _scatter(ax_zoom, cluster_rows, annotate_top=False)
        ax_zoom.set_title("Error vs Memory (log-log, zoomed cluster)")
        # adjust x/y limits to cluster range with padding
        mems_c = [r["metrics"].get("estimated_bytes", 0.0) for r in cluster_rows]
        errs_c = [r["metrics"].get("weighted_relative_l1", 0.0) for r in cluster_rows]
        ax_zoom.set_xlim(min(mems_c) * 0.8, max(mems_c) * 1.2)
        ax_zoom.set_ylim(min(errs_c) * 0.8 if min(errs_c) > 0 else 1e-6, max(errs_c) * 1.2)
    else:
        ax_zoom.set_visible(False)

    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)

def _plot_core_topk(core_files: List[Tuple[str, Path]], out_path: Path) -> None:
    import numpy as np

    fig, axes = plt.subplots(len(core_files), 1, figsize=(10, 5 * len(core_files)))
    if len(core_files) == 1:
        axes = [axes]
    for ax, (label, path) in zip(axes, core_files):
        if not path.exists():
            ax.set_visible(False)
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            ax.set_visible(False)
            continue
        if not data:
            ax.set_visible(False)
            continue
        labels = [d.get("target", "?") for d in data]
        true_vals = [d.get("true", 0) for d in data]
        est_vals = [d.get("est", 0) for d in data]
        idx = np.arange(len(labels))
        width = 0.35
        ax.bar(idx - width / 2, true_vals, width, label="True")
        ax.bar(idx + width / 2, est_vals, width, label="Estimated")
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(f"Core Top-K: {label}")
        ax.legend()
        ax.grid(axis="y")
    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)


def _parse_int_list(raw: str) -> List[int]:
    vals: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return vals


def _parse_float_list(raw: str) -> List[float]:
    vals: List[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def _parse_burst_configs(raw: str) -> List[Tuple[int, int, int]]:
    """
    Parse burst configs of the form \"bucket:recent:history\" separated by commas.
    Example: \"10:4:12,20:6:18\".
    """
    cfgs: List[Tuple[int, int, int]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) != 3:
            raise ValueError("Burst config must be bucket:recent:history")
        bucket, recent, history = (int(fields[0]), int(fields[1]), int(fields[2]))
        cfgs.append((bucket, recent, history))
    return cfgs


def _run_sanity_check(args: argparse.Namespace, widths: List[int], depths: List[int]) -> None:
    """
    Lightweight validation: small subset, check BurstSketch recent/history and Ada-Sketch decayed
    counts against exact computations to catch obvious zero/flat issues.
    """
    print("\n=== Sanity check: loading small subset of events ===")
    events, max_slot = load_events_burst(
        csv_path=args.dataset,
        base_slot_seconds=args.base_slot_seconds,
        max_events=args.sanity_max_events,
    )
    if not events:
        print("No events loaded for sanity check; aborting.")
        return
    print(f"Loaded {len(events)} events; max_slot={max_slot}")

    # pick width/depth (allow override)
    w = args.sanity_width if args.sanity_width and args.sanity_width > 0 else widths[0]
    d = depths[0]

    # Top targets from subset
    freq = Counter(e.ngram for e in events)
    target_ngrams = [ng for ng, _ in freq.most_common(max(3, args.top_k))]
    if not target_ngrams:
        print("No targets found; aborting sanity check.")
        return
    focus = target_ngrams[0]
    print(f"Sanity targets: {target_ngrams[:3]}")

    # Ada-Sketch sanity: streaming ingest + decayed true/est at a few checkpoints
    print("\n[Sanity] Ada-Sketch decayed counts (true vs est) [streaming checkpoints]")
    ada = AdaSketchCMS(width=w, depth=d, decay_lambda=args.ada_lambda)
    events_sorted = sorted(events, key=lambda e: e.slot)
    event_idx = 0
    eval_times = []
    if max_slot >= 0:
        step = max(1, (max_slot + 1) // 8)
        eval_times = list(range(0, max_slot + 1, step))
        if eval_times[-1] != max_slot:
            eval_times.append(max_slot)
    for T in eval_times:
        while event_idx < len(events_sorted) and events_sorted[event_idx].slot <= T:
            e = events_sorted[event_idx]
            ada.add(e.ngram, slot=e.slot, count=1)
            event_idx += 1
        true_counts = ada_true_at_T(events=events_sorted, T=T, decay_lambda=args.ada_lambda, target_ngrams=[focus])
        est_val = ada.estimate_decayed(focus, T)
        t_val = true_counts.get(focus, 0.0)
        print(f"  T={T:5d} | true={t_val:.3f} est={est_val:.3f} abs_err={abs(est_val - t_val):.3f}")

    # BurstSketch sanity: recent/history counts and burst score for last active bucket
    print("\n[Sanity] BurstSketch recent/history counts and burst score")
    bucket_slots = args.burst_bucket_slots
    recent_buckets = args.burst_recent_buckets
    history_buckets = args.burst_history_buckets
    true_buckets, max_bucket_id = burst_build_true_buckets(events, bucket_slots, target_ngrams)
    if max_bucket_id < 0:
        print("  No buckets; aborting burst sanity.")
        return
    # choose last bucket with any recent activity if possible
    eval_buckets = list(range(max_bucket_id, max(-1, max_bucket_id - 20), -1))
    chosen_B = max_bucket_id
    for b in eval_buckets:
        if any(burst_true_count_range(true_buckets, ng, b - recent_buckets + 1, b) > 0 for ng in target_ngrams):
            chosen_B = b
            break
    recent_start = chosen_B - recent_buckets + 1
    hist_end = chosen_B - recent_buckets
    hist_start = hist_end - history_buckets + 1
    rec_true = burst_true_count_range(true_buckets, focus, recent_start, chosen_B)
    hist_true = burst_true_count_range(true_buckets, focus, hist_start, hist_end)
    rate_recent_true = rec_true / float(recent_buckets) if recent_buckets > 0 else 0.0
    rate_hist_true = hist_true / float(history_buckets) if history_buckets > 0 else 0.0
    burst_true = rate_recent_true / (rate_hist_true if rate_hist_true > 0 else 1e-9)
    print(
        f"  Bucket B={chosen_B} focus={focus!r} "
        f"recent[{recent_start}:{chosen_B}]={rec_true} "
        f"history[{hist_start}:{hist_end}]={hist_true} "
        f"burst_score={burst_true:.3f}"
    )
    # Synthetic micro-test to isolate Ada decay math vs collisions.
    print("\n[Sanity] Ada-Sketch synthetic micro-test (single key)")
    synth_events = [(0, 1), (10, 2), (20, 3), (40, 4)]  # (slot, count)
    synth_times = [0, 10, 20, 40, 60, 100]
    ada_synth = AdaSketchCMS(width=w, depth=d, decay_lambda=args.ada_lambda)
    synth_idx = 0
    synth_events_sorted = sorted(synth_events, key=lambda t: t[0])
    for T in synth_times:
        while synth_idx < len(synth_events_sorted) and synth_events_sorted[synth_idx][0] <= T:
            s, c = synth_events_sorted[synth_idx]
            ada_synth.add("synth_key", slot=s, count=c)
            synth_idx += 1
        true_val = sum(
            c * (math.exp(-args.ada_lambda * (T - s)) if T >= s else 0.0)
            for s, c in synth_events_sorted
            if s <= T
        )
        est_val = ada_synth.estimate_decayed("synth_key", T)
        print(f"  T={T:3d} | true={true_val:.4f} est={est_val:.4f} abs_err={abs(est_val - true_val):.4f}")

    print("\nSanity check complete. Rerun with --sanity-max-events or --sanity-width to adjust subset size.")


def _run_exact_check(args: argparse.Namespace) -> None:
    """
    Tiny exact validation: load a very small subset, pick a few targets, compute exact window counts,
    and compare to each model's estimates on the same horizon.
    """
    max_events = min(args.sanity_max_events, 50_000) if hasattr(args, "sanity_max_events") else 50_000
    print(f"\n=== Exact check: loading up to {max_events} events ===")
    events, max_slot = load_events_burst(
        csv_path=args.dataset,
        base_slot_seconds=args.base_slot_seconds,
        max_events=max_events,
    )
    if not events:
        print("No events loaded for exact check; aborting.")
        return
    freq = Counter(e.ngram for e in events)
    targets = [ng for ng, _ in freq.most_common(max(3, args.top_k))]
    horizon_len = min(args.eval_last_slots or (max_slot + 1), max_slot + 1)
    horizon_start = max(0, max_slot - horizon_len + 1)
    horizon_end = max_slot
    window_sizes = [16, 64, 256] if horizon_len >= 256 else [max(1, horizon_len // 4), max(1, horizon_len // 2), horizon_len]

    print(f"Targets: {targets[:5]}")
    print(f"Horizon: slots {horizon_start}-{horizon_end}, window sizes={window_sizes}")

    # Exact counts per target per window
    exact_counts = {t: {} for t in targets}
    for t in targets:
        for w in window_sizes:
            start = max(0, horizon_end - w + 1)
            end = horizon_end
            exact_counts[t][w] = sum(1 for e in events if start <= e.slot <= end and e.ngram == t)

    results = []

    # Ring buffer exact check
    w = _parse_int_list(args.widths)[0]
    d = _parse_int_list(args.depths)[0]
    rb_metrics = run_ring_buffer(
        input_path=args.dataset,
        max_events=max_events,
        base_slot_seconds=args.base_slot_seconds,
        base_width=w,
        depth=d,
        top_k=args.top_k,
        output_prefix="exact_rb",
        max_seconds=args.max_seconds,
        eval_last_slots=horizon_len,
        buffer_slots=horizon_len,
        graphs_dir=GRAPHS_BASE / "ring_buffer",
        generate_plots=False,
        metrics_out=None,
        core_data_out=None,
    )
    results.append(("ring_buffer", rb_metrics))

    # Hokusai
    h_metrics = run_hokusai(
        input_path=args.dataset,
        max_events=max_events,
        base_slot_seconds=args.base_slot_seconds,
        max_levels=args.hokusai_max_levels,
        base_width=w,
        depth=d,
        top_k=args.top_k,
        output_prefix="exact_hok",
        max_seconds=args.max_seconds,
        eval_last_slots=horizon_len,
        graphs_dir=GRAPHS_BASE / "hakusai",
        generate_plots=False,
        metrics_out=None,
        core_data_out=None,
    )
    results.append(("hokusai", h_metrics))

    # BurstSketch
    b_metrics = run_burst(
        input_path=args.dataset,
        max_events=max_events,
        base_slot_seconds=args.base_slot_seconds,
        bucket_slots=args.burst_bucket_slots,
        width=w,
        depth=d,
        recent_buckets=args.burst_recent_buckets,
        history_buckets=args.burst_history_buckets,
        top_k=args.top_k,
        output_prefix="exact_burst",
        graphs_dir=GRAPHS_BASE / "burst_sketch",
        generate_plots=False,
        metrics_out=None,
        eval_last_slots=horizon_len,
        core_data_out=None,
    )
    results.append(("burst_sketch", b_metrics))

    # Ada-Sketch
    a_metrics = run_ada(
        input_path=args.dataset,
        max_events=max_events,
        base_slot_seconds=args.base_slot_seconds,
        width=w,
        depth=d,
        decay_lambda=_parse_float_list(args.ada_lambdas)[0] if args.ada_lambdas else 1e-6,
        top_k=args.top_k,
        output_prefix="exact_ada",
        graphs_dir=GRAPHS_BASE / "ada_sketch",
        generate_plots=False,
        metrics_out=None,
        eval_last_slots=horizon_len,
        core_data_out=None,
    )
    results.append(("ada_sketch", a_metrics))

    # Summarize
    print("\n=== Exact check summary (metrics only) ===")
    for name, m in results:
        print(f"{name}: weighted_relative_l1={m.get('weighted_relative_l1', 0):.6f}, mae={m.get('mae', 0):.6f}, rmse={m.get('rmse', 0):.6f}")
    print("Exact check complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep widths/depths and compare Hokusai vs Ring Buffer."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DEFAULT,
        help=f"Path to ngram CSV (default={DATASET_DEFAULT})",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=5_000_000,
        help="Maximum events to load (0 = no limit; default=5,000,000)",
    )
    parser.add_argument(
        "--base-slot-seconds",
        type=int,
        default=60,
        help="Base slot seconds for both models (default=60)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=14 * 24 * 3600,
        help="Max seconds span from first timestamp (default matches model scripts)",
    )
    parser.add_argument(
        "--eval-last-slots",
        type=int,
        default=512,
        help="Evaluation horizon length in slots (default=512)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K ngrams to evaluate (default=5)",
    )
    parser.add_argument(
        "--widths",
        type=str,
        default="2048,16384,131072",
        help="Comma-separated base widths to sweep (default=2048,16384,131072)",
    )
    parser.add_argument(
        "--depths",
        type=str,
        default="5,7,10",
        help="Comma-separated depths to sweep (default=5,7,10)",
    )
    parser.add_argument(
        "--ada-lambdas",
        type=str,
        default="1e-6",
        help="Comma-separated lambdas for Ada-Sketch sweep (default=1e-6)",
    )
    parser.add_argument(
        "--hokusai-max-levels",
        type=int,
        default=8,
        help="Hokusai max levels (default=8)",
    )
    parser.add_argument(
        "--ring-buffer-slots",
        type=int,
        default=0,
        help="Ring Buffer slots retained; default (0) uses eval_last_slots",
    )
    parser.add_argument(
        "--burst-bucket-slots",
        type=int,
        default=10,
        help="BurstSketch bucket size in slots (default=10)",
    )
    parser.add_argument(
        "--burst-recent-buckets",
        type=int,
        default=4,
        help="BurstSketch recent window in buckets (default=4)",
    )
    parser.add_argument(
        "--burst-history-buckets",
        type=int,
        default=12,
        help="BurstSketch history window in buckets (default=12)",
    )
    parser.add_argument(
        "--burst-configs",
        type=str,
        default="",
        help="Optional comma list of burst configs bucket:recent:history (overrides single values).",
    )
    parser.add_argument(
        "--ada-lambda",
        type=float,
        default=1e-6,
        help="AdaSketch decay lambda (default=1e-6; use smaller to avoid overflow on long horizons)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="sweep",
        help="Prefix for output artifacts",
    )
    parser.add_argument(
        "--skip-model-plots",
        action="store_true",
        help="If set, skip per-model plots (combined plots still generated)",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run a small sanity check on BurstSketch/Ada-Sketch and exit.",
    )
    parser.add_argument(
        "--sanity-max-events",
        type=int,
        default=200_000,
        help="Max events for sanity check subset (default=200,000).",
    )
    parser.add_argument(
        "--sanity-width",
        type=int,
        default=0,
        help="Override width for sanity check (0 = use first from --widths).",
    )
    parser.add_argument(
        "--exact-check",
        action="store_true",
        help="Run a tiny exact validation (small subset, exact counts) and exit.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    widths = _parse_int_list(args.widths)
    depths = _parse_int_list(args.depths)
    ada_lambdas = _parse_float_list(args.ada_lambdas)
    burst_cfgs = (
        _parse_burst_configs(args.burst_configs)
        if args.burst_configs
        else [(args.burst_bucket_slots, args.burst_recent_buckets, args.burst_history_buckets)]
    )
    if not widths:
        raise SystemExit("No widths provided.")
    if not depths:
        raise SystemExit("No depths provided.")
    # Simplify: use only the first depth, first burst config, first lambda to avoid over-sweeping.
    if len(depths) > 1:
        print(f"Simplifying depths to first entry: {depths[0]} (ignored {depths[1:]})")
        depths = [depths[0]]
    if len(ada_lambdas) > 1:
        print(f"Simplifying ada_lambdas to first entry: {ada_lambdas[0]} (ignored {ada_lambdas[1:]})")
        ada_lambdas = [ada_lambdas[0]]
    if len(burst_cfgs) > 1:
        print(f"Simplifying burst configs to first entry: {burst_cfgs[0]} (ignored {burst_cfgs[1:]})")
        burst_cfgs = [burst_cfgs[0]]

    if args.sanity_check:
        _run_sanity_check(args, widths, depths)
        return
    if args.exact_check:
        _run_exact_check(args)
        return

    _ensure_dir(GRAPHS_BASE / "hakusai")
    _ensure_dir(GRAPHS_BASE / "ring_buffer")
    _ensure_dir(GRAPHS_BASE / "burst_sketch")
    _ensure_dir(GRAPHS_BASE / "ada_sketch")
    _ensure_dir(GRAPHS_BASE / "combined")
    _ensure_dir(RESULTS_DIR)

    buffer_slots = args.ring_buffer_slots or args.eval_last_slots or 1

    rows: List[Dict] = []

    for w in widths:
        for d in depths:
            tag = f"w{w}_d{d}"
            print(f"\n=== Running Hokusai (width={w}, depth={d}) ===")
            core_h_path = RESULTS_DIR / f"{args.prefix}_hokusai_{tag}_core.json"
            h_metrics = run_hokusai(
                input_path=args.dataset,
                max_events=args.max_events,
                base_slot_seconds=args.base_slot_seconds,
                max_levels=args.hokusai_max_levels,
                base_width=w,
                depth=d,
                top_k=args.top_k,
                output_prefix=f"{args.prefix}_hokusai_{tag}",
                max_seconds=args.max_seconds,
                eval_last_slots=args.eval_last_slots,
                graphs_dir=GRAPHS_BASE / "hakusai",
                generate_plots=not args.skip_model_plots,
                metrics_out=RESULTS_DIR / f"{args.prefix}_hokusai_{tag}_metrics.json",
                core_data_out=core_h_path,
            )
            rows.append({"model": "hokusai", "label": f"hokusai_w{w}_d{d}", "width": w, "depth": d, "metrics": h_metrics})

            print(f"\n=== Running Ring Buffer (width={w}, depth={d}) ===")
            core_rb_path = RESULTS_DIR / f"{args.prefix}_ring_{tag}_core.json"
            rb_metrics = run_ring_buffer(
                input_path=args.dataset,
                max_events=args.max_events,
                base_slot_seconds=args.base_slot_seconds,
                base_width=w,
                depth=d,
                top_k=args.top_k,
                output_prefix=f"{args.prefix}_ring_{tag}",
                max_seconds=args.max_seconds,
                eval_last_slots=args.eval_last_slots,
                buffer_slots=buffer_slots,
                graphs_dir=GRAPHS_BASE / "ring_buffer",
                generate_plots=not args.skip_model_plots,
                metrics_out=RESULTS_DIR / f"{args.prefix}_ring_{tag}_metrics.json",
                core_data_out=core_rb_path,
            )
            rows.append({"model": "ring_buffer", "label": f"ring_w{w}_d{d}", "width": w, "depth": d, "metrics": rb_metrics})

            for (b_bucket, b_recent, b_hist) in burst_cfgs:
                tag_burst = f"{tag}_bs{b_bucket}_r{b_recent}_h{b_hist}"
                print(
                    f"\n=== Running BurstSketch (width={w}, depth={d}, bucket={b_bucket}, recent={b_recent}, hist={b_hist}) ==="
                )
                core_burst_path = RESULTS_DIR / f"{args.prefix}_burst_{tag_burst}_core.json"
                burst_metrics = run_burst(
                    input_path=args.dataset,
                    max_events=args.max_events,
                    base_slot_seconds=args.base_slot_seconds,
                    bucket_slots=b_bucket,
                    width=w,
                    depth=d,
                    recent_buckets=b_recent,
                    history_buckets=b_hist,
                    top_k=args.top_k,
                    output_prefix=f"{args.prefix}_burst_{tag_burst}",
                    graphs_dir=GRAPHS_BASE / "burst_sketch",
                    generate_plots=not args.skip_model_plots,
                    metrics_out=RESULTS_DIR / f"{args.prefix}_burst_{tag_burst}_metrics.json",
                    eval_last_slots=args.eval_last_slots,
                    core_data_out=core_burst_path,
                    max_seconds=args.max_seconds,
                )
                rows.append(
                    {
                        "model": "burst_sketch",
                        "label": f"burst_w{w}_d{d}",
                        "width": w,
                        "depth": d,
                        "metrics": burst_metrics,
                    }
                )

            for lam in ada_lambdas:
                tag_ada = f"{tag}_lam{lam:g}"
                print(f"\n=== Running Ada-Sketch (width={w}, depth={d}, lambda={lam}) ===")
                core_ada_path = RESULTS_DIR / f"{args.prefix}_ada_{tag_ada}_core.json"
                ada_metrics = run_ada(
                    input_path=args.dataset,
                    max_events=args.max_events,
                    base_slot_seconds=args.base_slot_seconds,
                    width=w,
                    depth=d,
                    decay_lambda=lam,
                    top_k=args.top_k,
                    output_prefix=f"{args.prefix}_ada_{tag_ada}",
                    graphs_dir=GRAPHS_BASE / "ada_sketch",
                    generate_plots=not args.skip_model_plots,
                    metrics_out=RESULTS_DIR / f"{args.prefix}_ada_{tag_ada}_metrics.json",
                    eval_last_slots=args.eval_last_slots,
                    core_data_out=core_ada_path,
                    max_seconds=args.max_seconds,
                )
                rows.append(
                    {
                        "model": "ada_sketch",
                        "label": f"ada_w{w}_d{d}",
                        "width": w,
                        "depth": d,
                        "metrics": ada_metrics,
                    }
                )

    table_path = RESULTS_DIR / f"{args.prefix}_comparison.tsv"
    # Order rows by model first, then width, then depth for clearer grouping.
    model_order = {"hokusai": 0, "ring_buffer": 1, "burst_sketch": 2, "ada_sketch": 3}
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            model_order.get(r.get("model"), 99),
            r.get("width", 0),
            r.get("depth", 0),
        ),
    )

    _write_results_table(rows_sorted, table_path)
    print(f"\nSaved numeric summary to {table_path}")

    combined_graphs = GRAPHS_BASE / "combined"
    plot_specs = [
        ("weighted_relative_l1", "Weighted Relative L1 (lower is better)"),
        ("mae", "Mean Absolute Error"),
        ("rmse", "RMSE"),
        ("total_counters", "Total CMS Counters (memory proxy)"),
        ("estimated_bytes", "Estimated Memory (bytes)"),
        ("total_time_s", "Total Time (s)"),
        ("stream_time_s", "Stream Time (s)"),
        ("throughput_events_per_s", "Throughput (events/s)"),
    ]
    for metric, title in plot_specs:
        out = combined_graphs / f"{args.prefix}_{metric}.png"
        _plot_metric_bars(rows_sorted, metric, title, metric, out)

    # Scatter: error vs memory
    scatter_path = combined_graphs / f"{args.prefix}_error_vs_memory.png"
    _plot_error_vs_memory(rows_sorted, scatter_path)

    print("\nComparison plots saved to:")
    for metric, _ in plot_specs:
        print("  ", combined_graphs / f"{args.prefix}_{metric}.png")
    print("  ", scatter_path)

    # Combined core top-K overlays (per-model subplots).
    # Use first configs for combined core overlay.
    first_burst = burst_cfgs[0]
    first_lam = ada_lambdas[0]
    core_files = [
        ("hokusai", RESULTS_DIR / f"{args.prefix}_hokusai_w{widths[0]}_d{depths[0]}_core.json"),
        ("ring_buffer", RESULTS_DIR / f"{args.prefix}_ring_w{widths[0]}_d{depths[0]}_core.json"),
        (
            "burst_sketch",
            RESULTS_DIR
            / f"{args.prefix}_burst_w{widths[0]}_d{depths[0]}_bs{first_burst[0]}_r{first_burst[1]}_h{first_burst[2]}_core.json",
        ),
        ("ada_sketch", RESULTS_DIR / f"{args.prefix}_ada_w{widths[0]}_d{depths[0]}_lam{first_lam:g}_core.json"),
    ]
    core_out = combined_graphs / f"{args.prefix}_core_topk_combined.png"
    _plot_core_topk(core_files, core_out)
    print("  ", core_out)

    print("\nDone.")


if __name__ == "__main__":
    main()
