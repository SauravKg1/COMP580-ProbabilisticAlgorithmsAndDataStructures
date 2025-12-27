#!/usr/bin/env python3
"""
Ada-Sketch / Forward-Decay Count-Min Sketch experiment on n-gram streams.

Programmatic-friendly additions:
  - run_experiment(...) returning metrics dict
  - configurable graphs directory (default: analysis/graphs/ada_sketch)
  - optional plot skipping (--no-plot)
  - optional metrics JSON output (--metrics-out)
"""

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

GRAPHS_DIR = Path("analysis/graphs/ada_sketch")
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _label_bars(ax, fmt: str = "{:.2f}") -> None:
    for rect in ax.patches:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


class SimpleCountMinSketch:
    """Count-Min Sketch with floating counts for weighted updates."""

    def __init__(self, width: int, depth: int, seed: int = 0):
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive")
        self.width = width
        self.depth = depth
        self.seed = seed
        self._counts = np.zeros((depth, width), dtype=float)
        # Track per-row totals to enable count-mean-min noise correction.
        self._row_totals = np.zeros(depth, dtype=float)

    def _hash(self, item: str, i: int) -> int:
        import hashlib

        msg = f"{self.seed}-{i}-{item}".encode("utf-8")
        digest = hashlib.sha256(msg).digest()
        value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return value % self.width

    def add(self, item: str, weight: float = 1.0) -> None:
        if weight <= 0.0:
            return
        # Conservative update to reduce overestimation from collisions:
        # compute the minimum current estimate, then only raise counters up to min+weight.
        positions = [self._hash(item, i) for i in range(self.depth)]
        min_val = min(self._counts[i, j] for i, j in enumerate(positions))
        target_val = min_val + weight
        for i, j in enumerate(positions):
            if self._counts[i, j] < target_val:
                delta = target_val - self._counts[i, j]
                self._counts[i, j] = target_val
                self._row_totals[i] += delta

    def query(self, item: str) -> float:
        # Count-mean-min: subtract expected noise in each row, then take the minimum.
        best = float("inf")
        for i in range(self.depth):
            j = self._hash(item, i)
            c = self._counts[i, j]
            row_total = self._row_totals[i]
            noise = (row_total - c) / (self.width - 1) if self.width > 1 else 0.0
            est_i = c - noise
            if est_i < best:
                best = est_i
        if best == float("inf"):
            return 0.0
        return max(0.0, best)

    @property
    def memory_proxy(self) -> int:
        return int(self.width * self.depth)


class AdaSketchCMS:
    """Forward-decay CMS: store exp(lambda * t) weights; query decays at T."""

    def __init__(self, width: int, depth: int, decay_lambda: float):
        self.sketch = SimpleCountMinSketch(width=width, depth=depth, seed=12345)
        self.decay_lambda = decay_lambda
        self.max_slot_seen: int = -1

    def add(self, item: str, slot: int, count: int = 1) -> None:
        if slot < 0:
            return
        exp_arg = self.decay_lambda * slot
        # Guard against overflow in exp for long horizons.
        if exp_arg > 700.0:
            exp_arg = 700.0
        weight = math.exp(exp_arg) * float(count)
        self.sketch.add(item, weight)
        if slot > self.max_slot_seen:
            self.max_slot_seen = slot

    def estimate_decayed(self, item: str, T: int) -> float:
        if T < 0:
            return 0.0
        raw = self.sketch.query(item)
        return raw * math.exp(-self.decay_lambda * T)

    @property
    def memory_proxy(self) -> int:
        return self.sketch.memory_proxy


@dataclass
class Event:
    slot: int
    ngram: str


def parse_timestamp(ts: str) -> datetime:
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1]
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_events(
    csv_path: Path, base_slot_seconds: int, max_events: Optional[int] = None, max_seconds: Optional[float] = None
) -> Tuple[List[Event], int]:
    events: List[Event] = []
    first_ts: Optional[datetime] = None

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"timestamp", "ngram"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must have columns {required_cols}, found {reader.fieldnames}")

        for i, row in enumerate(reader):
            if max_events is not None and max_events > 0 and i >= max_events:
                break
            ts = parse_timestamp(row["timestamp"])
            if first_ts is None:
                first_ts = ts
            delta_sec = (ts - first_ts).total_seconds()
            if max_seconds is not None and delta_sec > max_seconds:
                break
            slot = int(delta_sec // base_slot_seconds)
            ngram = row["ngram"].strip()
            events.append(Event(slot=slot, ngram=ngram))

    if not events:
        raise ValueError("No events loaded from CSV.")

    max_slot = max(e.slot for e in events)
    return events, max_slot


def compute_true_decayed_counts_at_T(
    events: List[Event], T: int, decay_lambda: float, target_ngrams: List[str]
) -> Dict[str, float]:
    true_counts: Dict[str, float] = defaultdict(float)
    target_set = set(target_ngrams)
    for e in events:
        if e.ngram not in target_set or e.slot > T:
            continue
        weight = math.exp(-decay_lambda * (T - e.slot))
        true_counts[e.ngram] += weight
    for g in target_ngrams:
        true_counts.setdefault(g, 0.0)
    return true_counts


def evaluate_ada_sketch(
    events: List[Event],
    max_slot: int,
    width: int,
    depth: int,
    decay_lambda: float,
    top_k: int,
    output_prefix: str,
    graphs_dir: Path,
    generate_plots: bool,
    eval_last_slots: int,
) -> Tuple[Dict[str, float], List[str], Dict[str, float], Dict[str, float]]:
    freq_counter: Counter[str] = Counter()
    for e in events:
        freq_counter[e.ngram] += 1
    target_ngrams = [ng for ng, _ in freq_counter.most_common(top_k)]

    # Streaming ingest with on-the-fly evaluation so time-series plots are meaningful.
    ada = AdaSketchCMS(width=width, depth=depth, decay_lambda=decay_lambda)

    num_eval_times = min(10, max_slot + 1)
    if num_eval_times <= 0:
        num_eval_times = 1
    step = max(1, (max_slot + 1) // num_eval_times)
    eval_times: List[int] = list(range(0, max_slot + 1, step))
    if eval_times[-1] != max_slot:
        eval_times.append(max_slot)

    true_matrix: Dict[int, Dict[str, float]] = {}
    est_matrix: Dict[int, Dict[str, float]] = {}
    rel_error_matrix: Dict[int, Dict[str, float]] = {}
    abs_error_matrix: Dict[int, Dict[str, float]] = {}

    # Maintain running decayed truth for targets only.
    true_running: Dict[str, float] = {ng: 0.0 for ng in target_ngrams}
    events_sorted = sorted(events, key=lambda e: e.slot)
    event_idx = 0
    last_slot = 0
    eval_idx = 0
    next_eval = eval_times[eval_idx] if eval_times else None
    next_event_slot = events_sorted[0].slot if events_sorted else None

    while event_idx < len(events_sorted) or (next_eval is not None and eval_idx < len(eval_times)):
        candidates: List[int] = []
        if next_event_slot is not None:
            candidates.append(next_event_slot)
        if next_eval is not None:
            candidates.append(next_eval)
        if not candidates:
            break
        current = min(candidates)

        # Decay the running true counts up to the current slot.
        if current > last_slot:
            decay_factor = math.exp(-decay_lambda * (current - last_slot))
            for ng in true_running:
                true_running[ng] *= decay_factor
            last_slot = current

        # Process all events in this slot.
        if next_event_slot is not None and current == next_event_slot:
            while event_idx < len(events_sorted) and events_sorted[event_idx].slot == current:
                e = events_sorted[event_idx]
                ada.add(e.ngram, slot=e.slot, count=1)
                if e.ngram in true_running:
                    true_running[e.ngram] += 1.0
                event_idx += 1
            next_event_slot = events_sorted[event_idx].slot if event_idx < len(events_sorted) else None

        # Record evaluation snapshot after applying events at this slot.
        if next_eval is not None and current == next_eval:
            est_T: Dict[str, float] = {}
            rel_T: Dict[str, float] = {}
            abs_T: Dict[str, float] = {}
            true_T = dict(true_running)
            for ng in target_ngrams:
                est_val = ada.estimate_decayed(ng, current)
                t_val = true_T.get(ng, 0.0)
                est_T[ng] = est_val
                abs_err = abs(est_val - t_val)
                abs_T[ng] = abs_err
                rel_T[ng] = abs_err / t_val if t_val > 0 else 0.0
            true_matrix[current] = true_T
            est_matrix[current] = est_T
            rel_error_matrix[current] = rel_T
            abs_error_matrix[current] = abs_T
            eval_idx += 1
            next_eval = eval_times[eval_idx] if eval_idx < len(eval_times) else None

    T_final = eval_times[-1]
    true_final = true_matrix.get(T_final, {ng: 0.0 for ng in target_ngrams})
    est_final = est_matrix.get(T_final, {ng: 0.0 for ng in target_ngrams})
    rel_final = rel_error_matrix.get(T_final, {ng: 0.0 for ng in target_ngrams})
    abs_final = abs_error_matrix.get(T_final, {ng: 0.0 for ng in target_ngrams})

    all_rel_errors: List[float] = []
    all_abs_errors: List[float] = []
    all_true_values: List[float] = []
    for T in eval_times:
        if T not in true_matrix:
            continue
        for ng in target_ngrams:
            all_rel_errors.append(rel_error_matrix[T][ng])
            all_abs_errors.append(abs_error_matrix[T][ng])
            all_true_values.append(true_matrix[T][ng])

    saved_paths: List[Path] = []
    if generate_plots:
        x_idx = np.arange(len(target_ngrams))
        width_bar = 0.35

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(x_idx - width_bar / 2, [true_final[ng] for ng in target_ngrams], width_bar, label="True decayed")
        ax1.bar(x_idx + width_bar / 2, [est_final[ng] for ng in target_ngrams], width_bar, label="Estimated decayed")
        ax1.set_xticks(x_idx)
        ax1.set_xticklabels(target_ngrams, rotation=45, ha="right")
        ax1.set_ylabel("Decayed count")
        ax1.set_title("Ada-Sketch: True vs Estimated Decayed Counts (final time)")
        ax1.legend()
        ax1.grid(axis="y")
        _label_bars(ax1, fmt="{:.2f}")
        fig1.tight_layout()
        p1 = graphs_dir / f"{output_prefix}_ada_counts_bar.png"
        _ensure_parent_dir(p1)
        fig1.savefig(p1, dpi=150)
        saved_paths.append(p1)
        plt.close(fig1)

        fig2, (ax2a, ax2b) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
        abs_vals = [abs_final[ng] for ng in target_ngrams]
        rel_vals = [rel_final[ng] for ng in target_ngrams]
        ax2a.bar(list(range(len(target_ngrams))), abs_vals)
        ax2a.set_ylabel("Absolute error")
        ax2a.set_title("Ada-Sketch: Absolute & Relative Error (final time)")
        ax2a.grid(axis="y")
        ax2b.bar(list(range(len(target_ngrams))), rel_vals)
        ax2b.set_xlabel("n-gram index (Top-K)")
        ax2b.set_ylabel("Relative error")
        ax2b.grid(axis="y")
        _label_bars(ax2a, fmt="{:.2f}")
        _label_bars(ax2b, fmt="{:.2f}")
        fig2.tight_layout()
        p2 = graphs_dir / f"{output_prefix}_ada_abs_rel_error.png"
        _ensure_parent_dir(p2)
        fig2.savefig(p2, dpi=150)
        saved_paths.append(p2)
        plt.close(fig2)

        ts_target = target_ngrams[0]
        ts_true = [true_matrix[T][ts_target] for T in eval_times]
        ts_est = [est_matrix[T][ts_target] for T in eval_times]
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(eval_times, ts_true, marker="o", linestyle="-", label="True")
        ax3.plot(eval_times, ts_est, marker="x", linestyle="--", label="Estimated")
        ax3.set_xlabel("Slot index (T)")
        ax3.set_ylabel("Decayed count")
        ax3.set_title(f"Ada-Sketch: Time-series decayed counts for {ts_target!r}")
        ax3.grid(True)
        ax3.legend()
        fig3.tight_layout()
        p3 = graphs_dir / f"{output_prefix}_ada_timeseries.png"
        _ensure_parent_dir(p3)
        fig3.savefig(p3, dpi=150)
        saved_paths.append(p3)
        plt.close(fig3)

        fig4, ax4 = plt.subplots(figsize=(8, 5))
        ax4.hist(all_rel_errors, bins=20, edgecolor="black")
        ax4.set_xlabel("Relative error")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Ada-Sketch: Relative Error Distribution (all targets × times)")
        ax4.grid(axis="y")
        fig4.tight_layout()
        p4 = graphs_dir / f"{output_prefix}_ada_rel_error_hist.png"
        _ensure_parent_dir(p4)
        fig4.savefig(p4, dpi=150)
        saved_paths.append(p4)
        plt.close(fig4)

        fig5, ax5 = plt.subplots(figsize=(6, 6))
        t_vals = np.array(all_true_values, dtype=float)
        e_vals = np.array(all_abs_errors, dtype=float)
        ax5.scatter(t_vals, e_vals, alpha=0.7)
        ax5.set_xlabel("True decayed count")
        ax5.set_ylabel("Absolute error")
        ax5.set_title("Ada-Sketch: True decayed vs Absolute error")
        ax5.grid(True)
        fig5.tight_layout()
        p5 = graphs_dir / f"{output_prefix}_ada_true_vs_abs_error.png"
        _ensure_parent_dir(p5)
        fig5.savefig(p5, dpi=150)
        saved_paths.append(p5)
        plt.close(fig5)

        eval_sorted = sorted(eval_times)
        heat_data: List[List[float]] = []
        for ng in target_ngrams:
            row = []
            for T in eval_sorted:
                row.append(rel_error_matrix[T][ng])
            heat_data.append(row)
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        cax = ax6.imshow(heat_data, aspect="auto", interpolation="nearest")
        ax6.set_xticks(range(len(eval_sorted)))
        ax6.set_xticklabels(eval_sorted, rotation=45, ha="right")
        ax6.set_yticks(range(len(target_ngrams)))
        ax6.set_yticklabels(target_ngrams)
        ax6.set_xlabel("Evaluation time T (slot index)")
        ax6.set_ylabel("Target n-gram")
        ax6.set_title("Ada-Sketch: Relative Error Heatmap")
        fig6.colorbar(cax, ax=ax6, label="Relative error")
        fig6.tight_layout()
        p6 = graphs_dir / f"{output_prefix}_ada_rel_error_heatmap.png"
        _ensure_parent_dir(p6)
        fig6.savefig(p6, dpi=150)
        saved_paths.append(p6)
        plt.close(fig6)

        fig7, ax7 = plt.subplots(figsize=(8, 5))
        sorted_rel = sorted(rel_final.values())
        ax7.plot(range(len(sorted_rel)), sorted_rel, marker="o", linestyle="-")
        ax7.set_xlabel("n-gram index (sorted by error)")
        ax7.set_ylabel("Relative error (final time)")
        ax7.set_title("Ada-Sketch: Sorted Relative Errors (final time)")
        ax7.grid(True)
        fig7.tight_layout()
        p7 = graphs_dir / f"{output_prefix}_ada_rel_error_sorted.png"
        _ensure_parent_dir(p7)
        fig7.savefig(p7, dpi=150)
        saved_paths.append(p7)
        plt.close(fig7)

        fig8, ax8 = plt.subplots(figsize=(6, 4))
        mem_proxy = ada.memory_proxy
        ax8.bar(["Ada-Sketch CMS"], [mem_proxy])
        ax8.set_ylabel("Number of counters (width × depth)")
        ax8.set_title("Ada-Sketch: Memory Proxy")
        ax8.grid(axis="y")
        _label_bars(ax8, fmt="{:.0f}")
        fig8.tight_layout()
        p8 = graphs_dir / f"{output_prefix}_ada_memory_proxy.png"
        _ensure_parent_dir(p8)
        fig8.savefig(p8, dpi=150)
        saved_paths.append(p8)
        plt.close(fig8)

    mem_proxy = ada.memory_proxy
    mae = float(np.mean(all_abs_errors)) if all_abs_errors else 0.0
    rmse = float(np.sqrt(np.mean([e * e for e in all_abs_errors]))) if all_abs_errors else 0.0
    mean_rel_error = float(np.mean(all_rel_errors)) if all_rel_errors else 0.0
    sum_true = float(np.sum(all_true_values)) if all_true_values else 0.0
    sum_abs = float(np.sum(all_abs_errors)) if all_abs_errors else 0.0
    weighted_relative_l1 = sum_abs / sum_true if sum_true > 0 else 0.0

    metrics: Dict[str, float] = {
        "num_queries": float(len(all_abs_errors)),
        "mae": mae,
        "rmse": rmse,
        "mean_relative_error": mean_rel_error,
        "weighted_relative_l1": weighted_relative_l1,
        "total_counters": float(mem_proxy),
    }
    metrics["estimated_bytes"] = metrics["total_counters"] * 8.0

    if generate_plots:
        # Core shared-horizon plots using decayed counts at T_final
        fig_core, ax_core = plt.subplots(figsize=(10, 5))
        idx_core = np.arange(len(target_ngrams))
        width_core = 0.35
        ax_core.bar(idx_core - width_core / 2, [true_final[ng] for ng in target_ngrams], width_core, label="True decayed")
        ax_core.bar(idx_core + width_core / 2, [est_final[ng] for ng in target_ngrams], width_core, label="Estimated decayed")
        ax_core.set_xticks(idx_core)
        ax_core.set_xticklabels(target_ngrams, rotation=45, ha="right")
        ax_core.set_ylabel("Decayed count (core horizon)")
        ax_core.set_title("Core Top-K decayed counts (Ada-Sketch)")
        ax_core.legend()
        ax_core.grid(axis="y")
        _label_bars(ax_core, fmt="{:.2f}")
        fig_core.tight_layout()
        core_path = graphs_dir / f"{output_prefix}_core_topk.png"
        _ensure_parent_dir(core_path)
        fig_core.savefig(core_path, dpi=150)
        saved_paths = []
        saved_paths.append(core_path)
        plt.close(fig_core)

        fig_core_hist, ax_core_hist = plt.subplots()
        ax_core_hist.hist(all_rel_errors, bins=20, edgecolor="black")
        ax_core_hist.set_xlabel("Relative error (decayed)")
        ax_core_hist.set_ylabel("Frequency")
        ax_core_hist.set_title("Core relative error distribution (Ada-Sketch)")
        ax_core_hist.grid(axis="y")
        fig_core_hist.tight_layout()
        core_hist_path = graphs_dir / f"{output_prefix}_core_rel_error_hist.png"
        _ensure_parent_dir(core_hist_path)
        fig_core_hist.savefig(core_hist_path, dpi=150)
        saved_paths.append(core_hist_path)
        plt.close(fig_core_hist)

        fig_core_scatter, ax_core_scatter = plt.subplots()
        ax_core_scatter.scatter(all_true_values, all_abs_errors, alpha=0.7)
        ax_core_scatter.set_xlabel("True decayed count")
        ax_core_scatter.set_ylabel("Absolute error")
        ax_core_scatter.set_title("Core true vs absolute error (Ada-Sketch)")
        ax_core_scatter.grid(True)
        fig_core_scatter.tight_layout()
        core_scatter_path = graphs_dir / f"{output_prefix}_core_true_vs_abs_error.png"
        _ensure_parent_dir(core_scatter_path)
        fig_core_scatter.savefig(core_scatter_path, dpi=150)
        saved_paths.append(core_scatter_path)
        plt.close(fig_core_scatter)

    print("=== Ada-Sketch Evaluation ===")
    print(f"Number of events: {len(events)}")
    print(f"Time horizon (max slot): {max_slot}")
    print(f"Width={width}, Depth={depth}, lambda={decay_lambda}")
    print(f"Memory proxy (counters): {mem_proxy}")
    print(f"Average absolute error (all targets × times): {mae:.6f}")
    print(f"Average relative error (all targets × times): {mean_rel_error:.6f}")
    if generate_plots and saved_paths:
        print("Saved plots:")
        for p in saved_paths:
            print("  ", p)

    return metrics, target_ngrams, true_final, est_final


def run_experiment(
    input_path: Path,
    max_events: int,
    base_slot_seconds: int,
    width: int,
    depth: int,
    decay_lambda: float,
    top_k: int,
    output_prefix: str,
    graphs_dir: Path = None,
    generate_plots: bool = True,
    metrics_out: Path = None,
    eval_last_slots: int = 0,
    core_data_out: Path = None,
    max_seconds: float = None,
) -> Dict[str, float]:
    t_total_start = time.perf_counter()
    graphs_dir = graphs_dir or GRAPHS_DIR
    graphs_dir.mkdir(parents=True, exist_ok=True)

    t_load_start = time.perf_counter()
    events, max_slot = load_events(
        csv_path=input_path, base_slot_seconds=base_slot_seconds, max_events=max_events, max_seconds=max_seconds
    )
    t_load_end = time.perf_counter()

    t_eval_start = time.perf_counter()
    metrics, core_targets, core_true, core_est = evaluate_ada_sketch(
        events=events,
        max_slot=max_slot,
        width=width,
        depth=depth,
        decay_lambda=decay_lambda,
        top_k=top_k,
        output_prefix=output_prefix,
        graphs_dir=graphs_dir,
        generate_plots=generate_plots,
        eval_last_slots=eval_last_slots,
    )
    t_eval_end = time.perf_counter()
    t_total_end = time.perf_counter()

    if metrics_out:
        _ensure_parent_dir(metrics_out)
        with Path(metrics_out).open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)

    num_events = len(events)
    metrics.update(
        {
            "load_time_s": t_load_end - t_load_start,
            "stream_time_s": t_eval_end - t_eval_start,
            "eval_time_s": t_eval_end - t_eval_start,
            "total_time_s": t_total_end - t_total_start,
            "throughput_events_per_s": (num_events / (t_eval_end - t_eval_start))
            if (t_eval_end - t_eval_start) > 0
            else 0.0,
        }
    )


    if core_data_out:
        core_payload = []
        for ng in core_targets:
            core_payload.append({"target": ng, "true": core_true.get(ng, 0), "est": core_est.get(ng, 0)})
        cpath = Path(core_data_out)
        _ensure_parent_dir(cpath)
        with cpath.open("w", encoding="utf-8") as f:
            json.dump(core_payload, f, indent=2, sort_keys=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ada-Sketch / Forward-Decay CMS experiment on n-gram stream."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to cleaned CSV with columns: timestamp,ngram")
    parser.add_argument("--width", type=int, default=2048, help="CMS width (default=2048)")
    parser.add_argument("--depth", type=int, default=5, help="CMS depth (default=5)")
    parser.add_argument("--lambda", dest="decay_lambda", type=float, default=1e-6, help="Forward-decay parameter λ (default=1e-6)")
    parser.add_argument("--base-slot-seconds", type=int, default=60, help="Size of a logical time slot in seconds (default=60).")
    parser.add_argument("--max-events", type=int, default=0, help="Optional cap on number of events to load (0 = all).")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top n-grams to evaluate (default=20).")
    parser.add_argument("--output-prefix", type=str, default="ada_output", help="Prefix for output graph filenames (default='ada_output').")
    parser.add_argument("--graphs-dir", type=str, default="analysis/graphs/ada_sketch", help="Base directory for saving plots.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation.")
    parser.add_argument("--metrics-out", type=str, default="", help="Optional path to write metrics JSON.")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    max_events = args.max_events if args.max_events and args.max_events > 0 else None
    graphs_dir = Path(args.graphs_dir)
    generate_plots = not args.no_plot
    metrics_out = Path(args.metrics_out) if args.metrics_out else None

    run_experiment(
        input_path=input_path,
        max_events=max_events or 0,
        base_slot_seconds=args.base_slot_seconds,
        width=args.width,
        depth=args.depth,
        decay_lambda=args.decay_lambda,
        top_k=args.top_k,
        output_prefix=args.output_prefix,
        graphs_dir=graphs_dir,
        generate_plots=generate_plots,
        metrics_out=metrics_out,
    )


if __name__ == "__main__":
    main()
