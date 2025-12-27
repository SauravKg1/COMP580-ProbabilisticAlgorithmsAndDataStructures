#!/usr/bin/env python3

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

GRAPHS_DIR = Path("analysis/graphs/burst_sketch")
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


class CountMinSketch:
    """Standard Count-Min Sketch with integer counts."""

    def __init__(self, width: int, depth: int, seed: int = 0):
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive")
        self.width = width
        self.depth = depth
        self.seed = seed
        self._counts = np.zeros((depth, width), dtype=np.int64)

    def _hash(self, item: str, i: int) -> int:
        import hashlib

        msg = f"{self.seed}-{i}-{item}".encode("utf-8")
        digest = hashlib.sha256(msg).digest()
        value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return value % self.width

    def add(self, item: str, count: int = 1) -> None:
        if count <= 0:
            return
        for i in range(self.depth):
            j = self._hash(item, i)
            self._counts[i, j] += count

    def query(self, item: str) -> int:
        est = math.inf
        for i in range(self.depth):
            j = self._hash(item, i)
            est = min(est, self._counts[i, j])
        if est is math.inf:
            return 0
        return int(est)

    @property
    def memory_proxy(self) -> int:
        return int(self.width * self.depth)


class BurstSketchCMS:
    def __init__(self, bucket_slots: int, width: int, depth: int, seed: int = 0):
        if bucket_slots <= 0:
            raise ValueError("bucket_slots must be positive")
        self.bucket_slots = bucket_slots
        self.width = width
        self.depth = depth
        self.seed = seed
        self._buckets: List[CountMinSketch] = []
        self.max_bucket_id: int = -1

    def _ensure_bucket(self, bucket_id: int) -> None:
        while len(self._buckets) <= bucket_id:
            cms = CountMinSketch(self.width, self.depth, seed=self.seed + len(self._buckets))
            self._buckets.append(cms)

    def add(self, item: str, slot: int, count: int = 1) -> None:
        if slot < 0:
            return
        bucket_id = slot // self.bucket_slots
        self._ensure_bucket(bucket_id)
        self._buckets[bucket_id].add(item, count)
        if bucket_id > self.max_bucket_id:
            self.max_bucket_id = bucket_id

    def count_in_bucket_range(self, item: str, start_bucket: int, end_bucket: int) -> int:
        if start_bucket > end_bucket:
            return 0
        if end_bucket < 0:
            return 0
        start_bucket = max(0, start_bucket)
        end_bucket = min(end_bucket, len(self._buckets) - 1)
        if start_bucket > end_bucket:
            return 0
        total = 0
        for b in range(start_bucket, end_bucket + 1):
            total += self._buckets[b].query(item)
        return total

    @property
    def num_buckets(self) -> int:
        return len(self._buckets)

    @property
    def memory_counters(self) -> int:
        return self.num_buckets * self.width * self.depth


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


def build_true_bucket_counts(
    events: List[Event],
    bucket_slots: int,
    target_ngrams: List[str],
) -> Tuple[Dict[int, Dict[str, int]], int]:
    true_buckets: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    target_set = set(target_ngrams)
    max_bucket_id = 0

    for e in events:
        bucket_id = e.slot // bucket_slots
        if bucket_id > max_bucket_id:
            max_bucket_id = bucket_id
        if e.ngram in target_set:
            true_buckets[bucket_id][e.ngram] += 1

    return true_buckets, max_bucket_id


def true_count_in_bucket_range(
    true_buckets: Dict[int, Dict[str, int]],
    ngram: str,
    start_bucket: int,
    end_bucket: int,
) -> int:
    if start_bucket > end_bucket:
        return 0
    total = 0
    for b in range(start_bucket, end_bucket + 1):
        if b < 0:
            continue
        bucket_dict = true_buckets.get(b)
        if bucket_dict:
            total += bucket_dict.get(ngram, 0)
    return total


def evaluate_burst_sketch(
    events: List[Event],
    max_slot: int,
    base_slot_seconds: int,
    bucket_slots: int,
    width: int,
    depth: int,
    recent_buckets: int,
    history_buckets: int,
    top_k: int,
    output_prefix: str,
    graphs_dir: Path,
    generate_plots: bool,
    eval_last_slots: int,
) -> Tuple[Dict[str, float], List[str], Dict[str, int], Dict[str, int], Dict[int, Dict[str, Dict[str, float]]]]:
    # Focus evaluation on the tail horizon if requested, similar to other models.
    if eval_last_slots > 0:
        horizon_start = max(0, max_slot - eval_last_slots + 1)
        events_eval = [e for e in events if e.slot >= horizon_start]
        if events_eval:
            eval_max_slot = max(e.slot for e in events_eval)
        else:
            events_eval = events
            eval_max_slot = max_slot
    else:
        events_eval = events
        eval_max_slot = max_slot

    print("Building frequency table for top-K selection...")
    freq_counter: Counter[str] = Counter()
    for e in events_eval:
        freq_counter[e.ngram] += 1

    target_ngrams = [ng for ng, _ in freq_counter.most_common(top_k)]

    print("Initializing BurstSketch CMS and true per-bucket counts...")
    burst = BurstSketchCMS(bucket_slots=bucket_slots, width=width, depth=depth, seed=777)
    true_buckets, max_bucket_id = build_true_bucket_counts(events_eval, bucket_slots, target_ngrams)

    for idx, e in enumerate(events_eval):
        burst.add(e.ngram, slot=e.slot, count=1)
        if idx > 0 and idx % 200000 == 0:
            print(f"  streamed {idx} events into BurstSketch...")

    print(f"Number of buckets in BurstSketch: {burst.num_buckets} (max_bucket_id={max_bucket_id})")

    min_B = recent_buckets + history_buckets
    if max_bucket_id < min_B:
        eval_buckets = [max_bucket_id]
    else:
        span = max_bucket_id - min_B + 1
        num_eval = min(10, span)
        step = max(1, span // num_eval)
        eval_buckets = list(range(min_B, max_bucket_id + 1, step))
        if eval_buckets[-1] != max_bucket_id:
            eval_buckets.append(max_bucket_id)

    print(f"Evaluation bucket times: {eval_buckets}")

    epsilon = 1e-9
    true_burst: Dict[int, Dict[str, float]] = {}
    est_burst: Dict[int, Dict[str, float]] = {}
    rel_error_burst: Dict[int, Dict[str, float]] = {}
    abs_error_burst: Dict[int, Dict[str, float]] = {}

    # Keep per-bucket recent/history to choose a representative bucket with activity.
    true_recent_map: Dict[int, Dict[str, int]] = {}
    true_history_map: Dict[int, Dict[str, int]] = {}
    est_recent_map: Dict[int, Dict[str, int]] = {}
    est_history_map: Dict[int, Dict[str, int]] = {}

    for B in eval_buckets:
        true_burst_B: Dict[str, float] = {}
        est_burst_B: Dict[str, float] = {}
        rel_B: Dict[str, float] = {}
        abs_B: Dict[str, float] = {}
        true_recent_map[B] = {}
        est_recent_map[B] = {}
        true_history_map[B] = {}
        est_history_map[B] = {}

        for ng in target_ngrams:
            recent_start = B - recent_buckets + 1
            recent_end = B
            hist_end = B - recent_buckets
            hist_start = hist_end - history_buckets + 1

            c_recent_true = true_count_in_bucket_range(true_buckets, ng, recent_start, recent_end)
            c_hist_true = true_count_in_bucket_range(true_buckets, ng, hist_start, hist_end)

            c_recent_est = burst.count_in_bucket_range(ng, recent_start, recent_end)
            c_hist_est = burst.count_in_bucket_range(ng, hist_start, hist_end)

            rate_recent_true = c_recent_true / float(recent_buckets) if recent_buckets > 0 else 0.0
            rate_hist_true = c_hist_true / float(history_buckets) if history_buckets > 0 else 0.0
            rate_recent_est = c_recent_est / float(recent_buckets) if recent_buckets > 0 else 0.0
            rate_hist_est = c_hist_est / float(history_buckets) if history_buckets > 0 else 0.0

            burst_true = rate_recent_true / (rate_hist_true + epsilon)
            burst_est = rate_recent_est / (rate_hist_est + epsilon)

            true_burst_B[ng] = burst_true
            est_burst_B[ng] = burst_est
            abs_err = abs(burst_est - burst_true)
            abs_B[ng] = abs_err
            rel_B[ng] = abs_err / burst_true if burst_true > 0 else 0.0
            true_recent_map[B][ng] = c_recent_true
            true_history_map[B][ng] = c_hist_true
            est_recent_map[B][ng] = c_recent_est
            est_history_map[B][ng] = c_hist_est

        true_burst[B] = true_burst_B
        est_burst[B] = est_burst_B
        rel_error_burst[B] = rel_B
        abs_error_burst[B] = abs_B

    all_rel_errors: List[float] = []
    all_abs_errors: List[float] = []
    all_true_burst_scores: List[float] = []
    all_est_burst_scores: List[float] = []
    for B in eval_buckets:
        for ng in target_ngrams:
            t_val = true_burst[B][ng]
            e_val = est_burst[B][ng]
            all_true_burst_scores.append(t_val)
            all_est_burst_scores.append(e_val)
            all_rel_errors.append(rel_error_burst[B][ng])
            all_abs_errors.append(abs_error_burst[B][ng])

    # Plotting (optional)
    saved_paths: List[Path] = []
    if generate_plots:
        # Pick a bucket with non-zero recent activity if possible; otherwise use last.
        B_final = eval_buckets[-1]
        for b in reversed(eval_buckets):
            if any(v > 0 for v in true_recent_map.get(b, {}).values()) or any(
                v > 0 for v in est_recent_map.get(b, {}).values()
            ):
                B_final = b
                break

        true_recent_final = true_recent_map.get(B_final, {ng: 0 for ng in target_ngrams})
        true_history_final = true_history_map.get(B_final, {ng: 0 for ng in target_ngrams})
        est_recent_final = est_recent_map.get(B_final, {ng: 0 for ng in target_ngrams})
        est_history_final = est_history_map.get(B_final, {ng: 0 for ng in target_ngrams})

        idx = np.arange(len(target_ngrams))
        bar_width = 0.2

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(idx - bar_width * 1.5, [true_recent_final[ng] for ng in target_ngrams], bar_width, label="True recent")
        ax1.bar(idx - bar_width * 0.5, [true_history_final[ng] for ng in target_ngrams], bar_width, label="True history")
        ax1.bar(idx + bar_width * 0.5, [est_recent_final[ng] for ng in target_ngrams], bar_width, label="Est recent")
        ax1.bar(idx + bar_width * 1.5, [est_history_final[ng] for ng in target_ngrams], bar_width, label="Est history")
        ax1.set_xticks(idx)
        ax1.set_xticklabels(target_ngrams, rotation=45, ha="right")
        ax1.set_ylabel("Counts (bucketized)")
        ax1.set_title(f"BurstSketch: True vs Est recent/history counts at bucket B={B_final}")
        ax1.legend()
        ax1.grid(axis="y")
        _label_bars(ax1, fmt="{:.0f}")
        fig1.tight_layout()
        p1 = graphs_dir / f"{output_prefix}_burst_counts_bar.png"
        _ensure_parent_dir(p1)
        fig1.savefig(p1, dpi=150)
        saved_paths.append(p1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.bar(idx - bar_width / 2, [true_burst[B_final][ng] for ng in target_ngrams], bar_width, label="True burst score")
        ax2.bar(idx + bar_width / 2, [est_burst[B_final][ng] for ng in target_ngrams], bar_width, label="Estimated burst score")
        ax2.set_xticks(idx)
        ax2.set_xticklabels(target_ngrams, rotation=45, ha="right")
        ax2.set_ylabel("Burst score")
        ax2.set_title(f"BurstSketch: True vs Est Burst Scores at B={B_final}")
        ax2.legend()
        ax2.grid(axis="y")
        _label_bars(ax2, fmt="{:.2f}")
        fig2.tight_layout()
        p2 = graphs_dir / f"{output_prefix}_burst_scores_bar.png"
        _ensure_parent_dir(p2)
        fig2.savefig(p2, dpi=150)
        saved_paths.append(p2)
        plt.close(fig2)

        abs_vals_final = [abs_error_burst[B_final][ng] for ng in target_ngrams]
        rel_vals_final = [rel_error_burst[B_final][ng] for ng in target_ngrams]
        fig3, (ax3a, ax3b) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
        ax3a.bar(idx, abs_vals_final)
        ax3a.set_ylabel("Absolute error")
        ax3a.set_title("BurstSketch: Burst score errors at final bucket")
        ax3a.grid(axis="y")
        ax3b.bar(idx, rel_vals_final)
        ax3b.set_xlabel("n-gram index (Top-K)")
        ax3b.set_ylabel("Relative error")
        ax3b.grid(axis="y")
        _label_bars(ax3a, fmt="{:.2f}")
        _label_bars(ax3b, fmt="{:.2f}")
        fig3.tight_layout()
        p3 = graphs_dir / f"{output_prefix}_burst_abs_rel_error.png"
        _ensure_parent_dir(p3)
        fig3.savefig(p3, dpi=150)
        saved_paths.append(p3)
        plt.close(fig3)

        ts_target = target_ngrams[0]
        ts_true = [true_burst[B][ts_target] for B in eval_buckets]
        ts_est = [est_burst[B][ts_target] for B in eval_buckets]
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.plot(eval_buckets, ts_true, marker="o", linestyle="-", label="True")
        ax4.plot(eval_buckets, ts_est, marker="x", linestyle="--", label="Estimated")
        ax4.set_xlabel("Bucket index B")
        ax4.set_ylabel("Burst score")
        ax4.set_title(f"BurstSketch: Burst score over time for {ts_target!r}")
        ax4.grid(True)
        ax4.legend()
        fig4.tight_layout()
        p4 = graphs_dir / f"{output_prefix}_burst_timeseries.png"
        _ensure_parent_dir(p4)
        fig4.savefig(p4, dpi=150)
        saved_paths.append(p4)
        plt.close(fig4)

        fig5, ax5 = plt.subplots(figsize=(8, 5))
        ax5.hist(all_rel_errors, bins=20, edgecolor="black")
        ax5.set_xlabel("Relative error (burst score)")
        ax5.set_ylabel("Frequency")
        ax5.set_title("BurstSketch: Relative error distribution (all B × Top-K)")
        ax5.grid(axis="y")
        fig5.tight_layout()
        p5 = graphs_dir / f"{output_prefix}_burst_rel_error_hist.png"
        _ensure_parent_dir(p5)
        fig5.savefig(p5, dpi=150)
        saved_paths.append(p5)
        plt.close(fig5)

        fig6, ax6 = plt.subplots(figsize=(6, 6))
        t_vals = np.array(all_true_burst_scores, dtype=float)
        e_vals = np.array(all_est_burst_scores, dtype=float)
        ax6.scatter(t_vals, e_vals, alpha=0.7)
        max_val = max(t_vals.max() if len(t_vals) else 1.0, e_vals.max() if len(e_vals) else 1.0)
        ax6.plot([0, max_val], [0, max_val], "k--", linewidth=1.0)
        ax6.set_xlabel("True burst score")
        ax6.set_ylabel("Estimated burst score")
        ax6.set_title("BurstSketch: True vs Estimated Burst Scores")
        ax6.grid(True)
        fig6.tight_layout()
        p6 = graphs_dir / f"{output_prefix}_burst_true_vs_est_scatter.png"
        _ensure_parent_dir(p6)
        fig6.savefig(p6, dpi=150)
        saved_paths.append(p6)
        plt.close(fig6)

        eval_sorted = sorted(eval_buckets)
        heat_data: List[List[float]] = []
        for ng in target_ngrams:
            row = []
            for B in eval_sorted:
                row.append(rel_error_burst[B][ng])
            heat_data.append(row)
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        cax = ax7.imshow(heat_data, aspect="auto", interpolation="nearest")
        ax7.set_xticks(range(len(eval_sorted)))
        ax7.set_xticklabels(eval_sorted, rotation=45, ha="right")
        ax7.set_yticks(range(len(target_ngrams)))
        ax7.set_yticklabels(target_ngrams)
        ax7.set_xlabel("Bucket index B")
        ax7.set_ylabel("Target n-gram")
        ax7.set_title("BurstSketch: Relative error heatmap (burst scores)")
        fig7.colorbar(cax, ax=ax7, label="Relative error")
        fig7.tight_layout()
        p7 = graphs_dir / f"{output_prefix}_burst_rel_error_heatmap.png"
        _ensure_parent_dir(p7)
        fig7.savefig(p7, dpi=150)
        saved_paths.append(p7)
        plt.close(fig7)

        fig8, ax8 = plt.subplots(figsize=(6, 4))
        mem_counters = burst.memory_counters
        ax8.bar(["BurstSketch CMS"], [mem_counters])
        ax8.set_ylabel("Number of counters (num_buckets × width × depth)")
        ax8.set_title("BurstSketch: Memory Proxy")
        ax8.grid(axis="y")
        _label_bars(ax8, fmt="{:.0f}")
        fig8.tight_layout()
        p8 = graphs_dir / f"{output_prefix}_burst_memory_proxy.png"
        _ensure_parent_dir(p8)
        fig8.savefig(p8, dpi=150)
        saved_paths.append(p8)
        plt.close(fig8)

    # Metrics summary
    mem_counters = burst.memory_counters
    mae = float(np.mean(all_abs_errors)) if all_abs_errors else 0.0
    rmse = float(np.sqrt(np.mean([e * e for e in all_abs_errors]))) if all_abs_errors else 0.0
    mean_rel_error = float(np.mean(all_rel_errors)) if all_rel_errors else 0.0
    sum_true = float(np.sum(all_true_burst_scores)) if all_true_burst_scores else 0.0
    sum_abs = float(np.sum(all_abs_errors)) if all_abs_errors else 0.0
    weighted_relative_l1 = sum_abs / sum_true if sum_true > 0 else 0.0

    metrics: Dict[str, float] = {
        "num_queries": float(len(all_abs_errors)),
        "mae": mae,
        "rmse": rmse,
        "mean_relative_error": mean_rel_error,
        "weighted_relative_l1": weighted_relative_l1,
        "total_counters": float(mem_counters),
    }
    metrics["estimated_bytes"] = metrics["total_counters"] * 8.0

    # Core shared-horizon counts (approximate counts over last eval_last_slots) available even if plots are skipped.
    horizon_len = eval_last_slots if eval_last_slots > 0 else (max_slot + 1)
    start_slot = max(0, max_slot - horizon_len + 1)
    # True counts over horizon
    true_core: Dict[str, int] = {ng: 0 for ng in target_ngrams}
    for e in events:
        if e.slot < start_slot or e.slot > max_slot:
            continue
        if e.ngram in true_core:
            true_core[e.ngram] += 1
    # Estimated counts over horizon via bucket aggregation (full-bucket approximation)
    start_bucket = start_slot // bucket_slots
    end_bucket = max_slot // bucket_slots
    est_core: Dict[str, int] = {}
    for ng in target_ngrams:
        est_core[ng] = burst.count_in_bucket_range(ng, start_bucket, end_bucket)

    if generate_plots:
        fig_core, ax_core = plt.subplots(figsize=(10, 5))
        idx_core = np.arange(len(target_ngrams))
        width_core = 0.35
        ax_core.bar(idx_core - width_core / 2, [true_core[ng] for ng in target_ngrams], width_core, label="True")
        ax_core.bar(idx_core + width_core / 2, [est_core[ng] for ng in target_ngrams], width_core, label="Estimated")
        ax_core.set_xticks(idx_core)
        ax_core.set_xticklabels(target_ngrams, rotation=45, ha="right")
        ax_core.set_ylabel(f"Count (last {horizon_len} slots)")
        ax_core.set_title("Core Top-K counts over shared horizon")
        ax_core.legend()
        ax_core.grid(axis="y")
        _label_bars(ax_core, fmt="{:.0f}")
        fig_core.tight_layout()
        core_path = graphs_dir / f"{output_prefix}_core_topk.png"
        _ensure_parent_dir(core_path)
        fig_core.savefig(core_path, dpi=150)
        saved_paths.append(core_path)

        # Core error histogram and scatter
        core_abs = []
        core_rel = []
        core_true_vals = []
        for ng in target_ngrams:
            t_val = true_core.get(ng, 0)
            e_val = est_core.get(ng, 0)
            core_true_vals.append(t_val)
            abs_err = abs(e_val - t_val)
            core_abs.append(abs_err)
            core_rel.append(abs_err / t_val if t_val > 0 else 0.0)

        fig_core_hist, ax_core_hist = plt.subplots()
        ax_core_hist.hist(core_rel, bins=20, edgecolor="black")
        ax_core_hist.set_xlabel("Relative error (core horizon)")
        ax_core_hist.set_ylabel("Frequency")
        ax_core_hist.set_title("Core relative error distribution (burst sketch)")
        ax_core_hist.grid(axis="y")
        fig_core_hist.tight_layout()
        core_hist_path = graphs_dir / f"{output_prefix}_core_rel_error_hist.png"
        _ensure_parent_dir(core_hist_path)
        fig_core_hist.savefig(core_hist_path, dpi=150)
        saved_paths.append(core_hist_path)
        plt.close(fig_core_hist)

        fig_core_scatter, ax_core_scatter = plt.subplots()
        ax_core_scatter.scatter(core_true_vals, core_abs, alpha=0.7)
        ax_core_scatter.set_xlabel("True count (core horizon)")
        ax_core_scatter.set_ylabel("Absolute error")
        ax_core_scatter.set_title("Core true vs absolute error (burst sketch)")
        ax_core_scatter.grid(True)
        fig_core_scatter.tight_layout()
        core_scatter_path = graphs_dir / f"{output_prefix}_core_true_vs_abs_error.png"
        _ensure_parent_dir(core_scatter_path)
        fig_core_scatter.savefig(core_scatter_path, dpi=150)
        saved_paths.append(core_scatter_path)
        plt.close(fig_core_scatter)

    print("=== BurstSketch Evaluation ===")
    print(f"Number of events: {len(events)}")
    print(f"Time horizon (max slot): {max_slot}")
    print(f"Bucket size (slots): {bucket_slots}")
    print(f"Width={width}, Depth={depth}")
    print(f"Recent buckets={recent_buckets}, History buckets={history_buckets}")
    print(f"Total CMS counters (memory proxy): {mem_counters}")
    print(f"Average absolute error (all B × targets): {mae:.6f}")
    print(f"Average relative error (all B × targets): {mean_rel_error:.6f}")
    if generate_plots and saved_paths:
        print("Saved plots:")
        for p in saved_paths:
            print("  ", p)

    return metrics, target_ngrams, true_core, est_core, {
        "true_burst": true_burst,
        "est_burst": est_burst,
        "true_recent": true_recent_map,
        "true_history": true_history_map,
        "est_recent": est_recent_map,
        "est_history": est_history_map,
        "eval_buckets": eval_buckets,
    }


def run_experiment(
    input_path: Path,
    max_events: int,
    base_slot_seconds: int,
    bucket_slots: int,
    width: int,
    depth: int,
    recent_buckets: int,
    history_buckets: int,
    top_k: int,
    output_prefix: str,
    graphs_dir: Path = None,
    generate_plots: bool = True,
    metrics_out: Path = None,
    eval_last_slots: int = 0,
    core_data_out: Path = None,
    burst_scores_out: Path = None,
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

    t_stream_start = time.perf_counter()
    metrics, core_targets, core_true, core_est, burst_debug = evaluate_burst_sketch(
        events=events,
        max_slot=max_slot,
        base_slot_seconds=base_slot_seconds,
        bucket_slots=bucket_slots,
        width=width,
        depth=depth,
        recent_buckets=recent_buckets,
        history_buckets=history_buckets,
        top_k=top_k,
        output_prefix=output_prefix,
        graphs_dir=graphs_dir,
        generate_plots=generate_plots,
        eval_last_slots=eval_last_slots,
    )
    t_stream_end = t_total_end = time.perf_counter()

    if metrics_out:
        _ensure_parent_dir(metrics_out)
        with Path(metrics_out).open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)

    if core_data_out:
        core_payload = []
        for ng in core_targets:
            core_payload.append({"target": ng, "true": core_true.get(ng, 0), "est": core_est.get(ng, 0)})
        cpath = Path(core_data_out)
        _ensure_parent_dir(cpath)
        with cpath.open("w", encoding="utf-8") as f:
            json.dump(core_payload, f, indent=2, sort_keys=True)

    num_events = len(events)
    metrics.update(
        {
            "load_time_s": t_load_end - t_load_start,
            "stream_time_s": t_stream_end - t_stream_start,
            "eval_time_s": 0.0,  # evaluation done inside streaming step here
            "total_time_s": t_total_end - t_total_start,
            "throughput_events_per_s": (num_events / (t_stream_end - t_stream_start))
            if (t_stream_end - t_stream_start) > 0
            else 0.0,
        }
    )

    if burst_scores_out:
        payload = []
        eval_buckets = burst_debug.get("eval_buckets", [])
        for B in eval_buckets:
            for ng in core_targets:
                payload.append(
                    {
                        "bucket": B,
                        "target": ng,
                        "true_burst": burst_debug["true_burst"].get(B, {}).get(ng, 0.0),
                        "est_burst": burst_debug["est_burst"].get(B, {}).get(ng, 0.0),
                        "true_recent": burst_debug["true_recent"].get(B, {}).get(ng, 0),
                        "true_history": burst_debug["true_history"].get(B, {}).get(ng, 0),
                        "est_recent": burst_debug["est_recent"].get(B, {}).get(ng, 0),
                        "est_history": burst_debug["est_history"].get(B, {}).get(ng, 0),
                    }
                )
        bpath = Path(burst_scores_out)
        _ensure_parent_dir(bpath)
        with bpath.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BurstSketch-style bucketed CMS burst detection experiment."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to cleaned CSV with columns: timestamp,ngram")
    parser.add_argument("--base-slot-seconds", type=int, default=60, help="Base slot size in seconds (default=60).")
    parser.add_argument("--bucket-slots", type=int, default=10, help="Number of base slots per bucket (default=10).")
    parser.add_argument("--width", type=int, default=1024, help="CMS width (default=1024).")
    parser.add_argument("--depth", type=int, default=4, help="CMS depth (default=4).")
    parser.add_argument("--recent-buckets", type=int, default=4, help="Number of buckets in the recent window (default=4).")
    parser.add_argument("--history-buckets", type=int, default=12, help="Number of buckets in the history window (default=12).")
    parser.add_argument("--max-events", type=int, default=200000, help="Cap on number of events to load (default=200000; <=0 means all).")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top n-grams (by raw frequency) to evaluate (default=20).")
    parser.add_argument("--output-prefix", type=str, default="burst_output", help="Prefix for output graph filenames (default='burst_output').")
    parser.add_argument("--graphs-dir", type=str, default="analysis/graphs/burst_sketch", help="Base directory for saving plots.")
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
        bucket_slots=args.bucket_slots,
        width=args.width,
        depth=args.depth,
        recent_buckets=args.recent_buckets,
        history_buckets=args.history_buckets,
        top_k=args.top_k,
        output_prefix=args.output_prefix,
        graphs_dir=graphs_dir,
        generate_plots=generate_plots,
        metrics_out=metrics_out,
    )


if __name__ == "__main__":
    main()
