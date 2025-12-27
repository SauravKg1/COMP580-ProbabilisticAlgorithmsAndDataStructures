#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

# Default location for model-specific plots; can be overridden via CLI.
GRAPHS_DIR = Path("analysis/graphs/hakusai")
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_parent_dir(path: Path) -> None:
    """
    Ensure that the parent directory of `path` exists.

    This is a defensive helper so that even if the graphs directory is deleted
    between import time and save time, all figure save operations still succeed.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)


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
            rotation=0,
        )

class CountMinSketch:

    def __init__(self, width: int, depth: int, seed: int = 0):
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive integers")

        self.width = width
        self.depth = depth
        self.seed = seed
        self._counts: List[List[int]] = [[0] * width for _ in range(depth)]
        self._hash_seeds: List[int] = [seed + i for i in range(depth)]

    def _hash(self, item: Any, seed: int) -> int:
        """
        Simple deterministic hash: combine python hash with a seed,
        then normalize to the [0, width) range.
        """
        h = hash((seed, item))
        if h < 0:
            h = -h
        return h % self.width

    def add(self, item: Any, count: int = 1) -> None:
        if count <= 0:
            return
        for row, s in enumerate(self._hash_seeds):
            col = self._hash(item, s)
            self._counts[row][col] += count

    def query(self, item: Any) -> int:
        """Return CMS estimate of the item's frequency."""
        est = math.inf
        for row, s in enumerate(self._hash_seeds):
            col = self._hash(item, s)
            est = min(est, self._counts[row][col])
        if est is math.inf:
            return 0
        return int(est)

    def clear(self) -> None:
        """Reset all counters to zero."""
        for r in range(self.depth):
            row = self._counts[r]
            for c in range(self.width):
                row[c] = 0

    def merge(self, other: "CountMinSketch") -> None:
        if self.width != other.width or self.depth != other.depth:
            raise ValueError("Cannot merge CMS with different shapes")
        for r in range(self.depth):
            row = self._counts[r]
            other_row = other._counts[r]
            for c in range(self.width):
                row[c] += other_row[c]

    @classmethod
    def merged(cls, sketches: Iterable["CountMinSketch"]) -> "CountMinSketch":
        sketches = list(sketches)
        if not sketches:
            raise ValueError("Cannot merge empty list")
        first = sketches[0]
        result = CountMinSketch(first.width, first.depth, first.seed)
        for s in sketches:
            result.merge(s)
        return result

    def fold_width(self, factor: int = 2) -> "CountMinSketch":
        if factor <= 1:
            return self
        if self.width % factor != 0:
            raise ValueError("Width must be divisible by factor to fold")
        new_width = self.width // factor
        folded = CountMinSketch(width=new_width, depth=self.depth, seed=self.seed)
        for r in range(self.depth):
            for j in range(new_width):
                s = 0
                base = j * factor
                for k in range(factor):
                    s += self._counts[r][base + k]
                folded._counts[r][j] = s
        return folded

    def __repr__(self) -> str:
        return f"CountMinSketch(width={self.width}, depth={self.depth}, seed={self.seed})"

@dataclass
class Event:
    ts: datetime
    ngram: str
    slot: int  # integer time index (base slot units)


class HokusaiMultiResolutionCMS:
    
    def __init__(
        self,
        base_slot_seconds: int,
        base_width: int,
        depth: int,
        max_levels: int,
        min_width: int = 32,
        seed: int = 0,
    ):
        if base_slot_seconds <= 0:
            raise ValueError("base_slot_seconds must be positive")
        if base_width <= 0 or depth <= 0:
            raise ValueError("base_width and depth must be positive")
        if max_levels <= 0:
            raise ValueError("max_levels must be positive")

        self.base_slot_seconds = base_slot_seconds
        self.base_width = base_width
        self.depth = depth
        self.max_levels = max_levels
        self.min_width = min_width
        self.seed = seed

        # For each level j: list of CMS segments.
        self.segments: List[List[CountMinSketch]] = [[] for _ in range(max_levels)]

        # Track the maximum slot we have seen so far (for "last window" queries).
        self.max_slot_seen: int = -1

        # Small epsilon to avoid division by zero in burst_score.
        self.epsilon: float = 1e-9

    def _width_for_level(self, level: int) -> int:
        """
        Item-aggregation / folding: higher levels use smaller width.
        """
        w = self.base_width >> level
        if w < self.min_width:
            w = self.min_width
        return w

    def _ensure_segment(self, level: int, seg_idx: int) -> CountMinSketch:
        """
        Ensure that self.segments[level][seg_idx] exists, creating CMSes as needed.
        """
        segs = self.segments[level]
        while len(segs) <= seg_idx:
            width = self._width_for_level(level)
            seed = self.seed + level * 1000 + len(segs)
            cms = CountMinSketch(width=width, depth=self.depth, seed=seed)
            segs.append(cms)
        return segs[seg_idx]

    def add(self, item: Any, slot: int, count: int = 1) -> None:
        """
        Add an event for `item` at discrete time index `slot`.
        This is the real-time update step.
        """
        if slot < 0:
            return
        if slot > self.max_slot_seen:
            self.max_slot_seen = slot

        # For each level j, compute segment index and update corresponding CMS.
        for level in range(self.max_levels):
            seg_len = 1 << level  # length of segment in slots
            seg_idx = slot // seg_len
            cms = self._ensure_segment(level, seg_idx)
            cms.add(item, count=count)

    
    @staticmethod
    def _decompose_interval(start_slot: int, end_slot: int, max_level_limit: int) -> List[Tuple[int, int]]:
        if start_slot > end_slot:
            return []
        res: List[Tuple[int, int]] = []
        start = start_slot
        end = end_slot
        while start <= end:
            max_len = end - start + 1
            max_level = int(math.floor(math.log2(max_len)))
            if max_level_limit > 0:
                max_level = min(max_level, max_level_limit - 1)

            level = max_level
            # Ensure alignment: start % 2^level == 0
            while level > 0 and (start % (1 << level)) != 0:
                level -= 1

            seg_len = 1 << level
            seg_idx = start // seg_len
            res.append((level, seg_idx))
            start += seg_len
        return res

    
    def query_interval(self, item: Any, start_slot: int, end_slot: int) -> int:
        """
        Approximate the count of `item` in [start_slot, end_slot] using the dyadic tree.

        This is O(log T) segments, and each segment query is O(depth).
        """
        if start_slot < 0:
            start_slot = 0
        if end_slot < start_slot:
            return 0
        segments = self._decompose_interval(start_slot, end_slot, self.max_levels)
        total = 0
        for level, seg_idx in segments:
            if level >= self.max_levels:
                continue
            if seg_idx >= len(self.segments[level]):
                continue
            cms = self.segments[level][seg_idx]
            total += cms.query(item)
        return total

    def query_last_window(self, item: Any, window_slots: int) -> int:
        """
        Convenience: query count of `item` in the last `window_slots` slots
        relative to the *latest* slot we have seen.
        """
        if self.max_slot_seen < 0 or window_slots <= 0:
            return 0
        end_slot = self.max_slot_seen
        start_slot = max(0, end_slot - window_slots + 1)
        return self.query_interval(item, start_slot, end_slot)

    def get_burst_score(
        self,
        item: Any,
        recent_slots: int,
        history_slots: int,
    ) -> float:
        if self.max_slot_seen < 0:
            return 0.0
        if recent_slots <= 0 or history_slots <= 0:
            return 0.0

        T = self.max_slot_seen
        recent_start = max(0, T - recent_slots + 1)
        recent_end = T
        hist_end = recent_start - 1
        hist_start = max(0, hist_end - history_slots + 1)

        if hist_end < hist_start:
            return 0.0

        count_recent = self.query_interval(item, recent_start, recent_end)
        count_history = self.query_interval(item, hist_start, hist_end)

        rate_recent = count_recent / float(recent_slots) if recent_slots > 0 else 0.0
        rate_history = count_history / float(history_slots) if history_slots > 0 else 0.0

        denom = rate_history + self.epsilon
        if denom == 0.0:
            return float("inf") if rate_recent > 0.0 else 0.0
        return rate_recent / denom

    def memory_counters_per_level(self) -> List[int]:
        """
        Approximate memory usage per level:
          counters = sum(width_j * depth for each segment at that level)
        """
        usage: List[int] = []
        for level in range(self.max_levels):
            segs = self.segments[level]
            if not segs:
                usage.append(0)
            else:
                width = segs[0].width
                depth = segs[0].depth
                counters = width * depth * len(segs)
                usage.append(counters)
        return usage

    def __repr__(self) -> str:
        return (
            f"HokusaiMultiResolutionCMS(base_slot_seconds={self.base_slot_seconds}, "
            f"base_width={self.base_width}, depth={self.depth}, "
            f"max_levels={self.max_levels}, min_width={self.min_width})"
        )

def parse_timestamp(ts: str) -> datetime:
    """
    Parse timestamps like '2019-05-01T00:00:00Z' into timezone-aware UTC datetimes.
    """
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1]
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_events(
    path: Path,
    max_events: int,
    base_slot_seconds: int,
    max_seconds: float,
) -> List[Event]:
    events: List[Event] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return events

        # Map column names to indices.
        name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(header)}

        # Required column for all current datasets.
        if "ngram" not in name_to_idx:
            raise ValueError("Expected 'ngram' column in CSV header")

        idx_ngram = name_to_idx["ngram"]
        idx_ts = name_to_idx.get("timestamp")
        idx_slot = name_to_idx.get("slot")

        # If a precomputed integer 'slot' column exists, we can avoid datetime parsing.
        # Slots will be rebased so that the first event seen has slot 0, matching the
        # behavior of the timestamp-based path.
        use_precomputed_slot = idx_slot is not None

        first_ts: datetime = None
        first_slot_value: int = None

        for i, row in enumerate(reader):
            if max_events > 0 and i >= max_events:
                break

            if use_precomputed_slot:
                try:
                    raw_slot = int(row[idx_slot])
                except (ValueError, TypeError, IndexError):
                    # Fallback: skip malformed rows.
                    continue
                if first_slot_value is None:
                    first_slot_value = raw_slot
                # Rebase so the first event has slot 0.
                slot = raw_slot - first_slot_value
                # Approximate elapsed seconds from the first event.
                delta_seconds = slot * base_slot_seconds
                if max_seconds > 0 and delta_seconds > max_seconds:
                    break
                ts = None  # We don't currently use the timestamp field elsewhere.
            else:
                if idx_ts is None:
                    raise ValueError(
                        "CSV must contain either a 'timestamp' column "
                        "or a precomputed integer 'slot' column."
                    )
                try:
                    ts = parse_timestamp(row[idx_ts])
                except (IndexError, ValueError):
                    # Skip malformed rows.
                    continue
                if first_ts is None:
                    first_ts = ts
                delta_seconds = (ts - first_ts).total_seconds()
                if max_seconds > 0 and delta_seconds > max_seconds:
                    break
                slot = int(delta_seconds // base_slot_seconds)

            try:
                ngram = row[idx_ngram]
            except IndexError:
                continue

            events.append(Event(ts=ts, ngram=ngram, slot=slot))
    return events


def build_ground_truth(events: List[Event], top_k: int) -> Tuple[List[str], int]:
    """
    Compute global counts and return:
        - list of top_k most frequent n-grams
        - maximum slot index seen
    """
    total_counts = Counter()
    max_slot = 0
    for e in events:
        total_counts[e.ngram] += 1
        if e.slot > max_slot:
            max_slot = e.slot
    targets = [ng for (ng, _) in total_counts.most_common(top_k)]
    return targets, max_slot


def build_horizon_targets(
    events: List[Event],
    top_k: int,
    start_slot: int,
    end_slot: int,
) -> List[str]:
    """
    Compute top_k n-grams restricted to a time horizon [start_slot, end_slot].
    This is used to make evaluation targets local in time instead of global.
    """
    if start_slot > end_slot:
        return []
    local_counts = Counter()
    for e in events:
        if e.slot < start_slot:
            continue
        if e.slot > end_slot:
            break
        local_counts[e.ngram] += 1
    return [ng for (ng, _) in local_counts.most_common(top_k)]


def true_count_interval(
    events: List[Event],
    target: str,
    start_slot: int,
    end_slot: int,
) -> int:
    """
    Exact count of `target` in [start_slot, end_slot].
    (Naive but fine for evaluation if we only use a subset of the data.)
    """
    if start_slot > end_slot:
        return 0
    c = 0
    for e in events:
        if e.slot < start_slot:
            continue
        if e.slot > end_slot:
            break
        if e.ngram == target:
            c += 1
    return c

def run_experiment(
    input_path: Path,
    max_events: int,
    base_slot_seconds: int,
    max_levels: int,
    base_width: int,
    depth: int,
    top_k: int,
    output_prefix: str,
    max_seconds: float,
    eval_last_slots: int,
    graphs_dir: Path = None,
    generate_plots: bool = True,
    metrics_out: Path = None,
    core_data_out: Path = None,
) -> Dict[str, float]:
    t_total_start = time.perf_counter()
    print("Loading events...")
    t_load_start = time.perf_counter()
    events = load_events(
        input_path,
        max_events=max_events,
        base_slot_seconds=base_slot_seconds,
        max_seconds=max_seconds,
    )
    t_load_end = time.perf_counter()
    if not events:
        print("No events loaded; nothing to do.")
        return {}
    events.sort(key=lambda e: e.slot)

    graphs_base = graphs_dir or GRAPHS_DIR
    graphs_base.mkdir(parents=True, exist_ok=True)

    # Global stats (mainly to get max_slot).
    global_targets, max_slot = build_ground_truth(events, top_k=top_k)

    # Determine evaluation horizon near the tail.
    if eval_last_slots > 0:
        # Use at most eval_last_slots, but not more than the stream length.
        horizon_len = min(eval_last_slots, max_slot + 1)
        horizon_start = max(0, max_slot - horizon_len + 1)
    else:
        # Use the entire time range.
        horizon_start = 0
        horizon_len = max_slot + 1
    horizon_end = max_slot

    targets = build_horizon_targets(
        events, top_k=top_k, start_slot=horizon_start, end_slot=horizon_end
    )
    if targets:
        print(
            f"Selected top {top_k} n-grams in evaluation horizon "
            f"[{horizon_start}, {horizon_end}] (length={horizon_len} slots)."
        )
    else:
        # Fallback to global if no activity in horizon (should be rare).
        targets = global_targets
        print(
            "No targets found in the evaluation horizon; "
            "falling back to global top-K."
        )

    print(f"Loaded {len(events)} events.")
    print(f"Max slot index: {max_slot}")
    print(f"Top {top_k} n-grams:")
    for ng in targets:
        print("  ", ng)

    # Cap levels for short horizons to reduce folding error. If the evaluation horizon is shorter than
    # the base width, stick to a single level (behaves like a flat CMS).
    eval_horizon = horizon_len
    if eval_horizon <= base_width:
        effective_levels = 1
    else:
        effective_levels = max(1, min(max_levels, math.ceil(math.log2(max_slot + 1)) if max_slot > 0 else 1))
    print(f"\nInitializing Hokusai Multi-Resolution CMS (effective_levels={effective_levels})...")
    hok = HokusaiMultiResolutionCMS(
        base_slot_seconds=base_slot_seconds,
        base_width=base_width,
        depth=depth,
        max_levels=effective_levels,
        min_width=base_width,
        seed=42,
    )
    print(hok)

    print("\nStreaming events into Hokusai structure...")
    t_stream_start = time.perf_counter()
    for e in events:
        hok.add(e.ngram, e.slot)
    t_stream_end = time.perf_counter()

    # Choose some window sizes in slots for evaluation.
    # We use a small set of exponentially growing windows to keep the number of
    # (target, window) queries modest while still covering multiple scales.
    max_window = min(eval_last_slots or (max_slot + 1), max_slot + 1)
    window_sizes_slots: List[int] = []
    w = 1
    while w <= max_window:
        window_sizes_slots.append(w)
        w *= 4  # 1, 4, 16, 64, 256, ...
    if not window_sizes_slots:
        window_sizes_slots = [1]

    # ---- Precompute ground-truth counts for targets in the evaluation horizon ----
    print(
        "\nPrecomputing exact ground-truth counts for targets "
        f"in horizon [{horizon_start}, {horizon_end}]..."
    )
    horizon_len = horizon_end - horizon_start + 1
    targets_set = set(targets)
    # Per-target per-slot counts within [horizon_start, horizon_end]
    per_target_counts: Dict[str, List[int]] = {
        t: [0] * horizon_len for t in targets
    }
    for e in events:
        if e.slot < horizon_start:
            continue
        if e.slot > horizon_end:
            break
        if e.ngram in targets_set:
            idx = e.slot - horizon_start
            per_target_counts[e.ngram][idx] += 1

    # Build prefix sums so we can answer any interval query in O(1).
    per_target_prefix: Dict[str, List[int]] = {}
    for t in targets:
        counts = per_target_counts[t]
        prefix = [0] * horizon_len
        running = 0
        for i, c in enumerate(counts):
            running += c
            prefix[i] = running
        per_target_prefix[t] = prefix

    def true_count_interval_fast(target: str, start_slot: int, end_slot: int) -> int:
        """
        Exact count of `target` in [start_slot, end_slot] using precomputed
        prefix sums over the evaluation horizon.
        """
        if target not in per_target_prefix:
            return 0
        if start_slot > end_slot:
            return 0
        # If the query interval lies entirely outside the horizon, count is 0.
        if end_slot < horizon_start or start_slot > horizon_end:
            return 0
        # Clamp to horizon.
        s = max(start_slot, horizon_start)
        e = min(end_slot, horizon_end)
        if s > e:
            return 0
        start_idx = s - horizon_start
        end_idx = e - horizon_start
        prefix = per_target_prefix[target]
        if start_idx == 0:
            return prefix[end_idx]
        return prefix[end_idx] - prefix[start_idx - 1]

    # Evaluate at final time (last slot).
    T = max_slot
    results: Dict[str, Dict[int, Tuple[int, int]]] = defaultdict(dict)

    t_eval_start = time.perf_counter()
    print("\nEvaluating accuracy at final time (slot index T = {})...".format(T))
    for target in targets:
        print(f"  Target: {target!r}")
        for w in window_sizes_slots:
            start = max(0, T - w + 1)
            end = T
            true_c = true_count_interval_fast(target, start, end)
            est_c = hok.query_interval(target, start, end)
            results[target][w] = (true_c, est_c)
            print(
                f"    window={w:3d} slots: true={true_c:6d}, "
                f"est={est_c:6d}, abs_err={abs(est_c - true_c):6d}"
            )

    # Collect error stats flattened across targets & windows
    all_rel_errors: List[float] = []
    all_true_counts: List[int] = []
    all_abs_errors: List[int] = []

    # Split statistics into "active" (true > 0) and "inactive" (true == 0).
    active_true_counts: List[int] = []
    active_abs_errors: List[int] = []
    active_rel_errors: List[float] = []

    inactive_total = 0
    inactive_false_positives = 0
    inactive_estimates: List[int] = []

    for target in targets:
        for w in window_sizes_slots:
            true_c, est_c = results[target][w]
            abs_err = abs(est_c - true_c)
            if true_c > 0:
                rel_err = abs_err / float(true_c)
            else:
                rel_err = 0.0
            all_true_counts.append(true_c)
            all_abs_errors.append(abs_err)
            all_rel_errors.append(rel_err)
            if true_c > 0:
                active_true_counts.append(true_c)
                active_abs_errors.append(abs_err)
                active_rel_errors.append(rel_err)
            else:
                inactive_total += 1
                inactive_estimates.append(est_c)
                if est_c > 0:
                    inactive_false_positives += 1

    mem_usage = hok.memory_counters_per_level()

    # Export core top-K (largest window) for combined overlays.
    core_payload: List[Dict[str, float]] = []
    if targets:
        largest_window = max(window_sizes_slots)
        true_bar = [results[target][largest_window][0] for target in targets]
        est_bar = [results[target][largest_window][1] for target in targets]
        if core_data_out:
            cpath = Path(core_data_out)
            _ensure_parent_dir(cpath)
            for t, tv, ev in zip(targets, true_bar, est_bar):
                core_payload.append({"target": t, "true": tv, "est": ev})
            with cpath.open("w", encoding="utf-8") as f:
                json.dump(core_payload, f, indent=2, sort_keys=True)
    else:
        true_bar = []
        est_bar = []

    saved_paths: List[Path] = []
    if generate_plots:
        print("\nPlotting True vs Estimated counts...")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        for target in targets:
            ws = sorted(results[target].keys())
            true_vals = [results[target][w][0] for w in ws]
            est_vals = [results[target][w][1] for w in ws]
            ax1.plot(ws, true_vals, marker="o", linestyle="-", label=f"{target} (true)")
            ax1.plot(ws, est_vals, marker="x", linestyle="--", label=f"{target} (est)")

        ax1.set_xlabel("Window size (slots)")
        ax1.set_ylabel("Count in last window")
        ax1.set_title(
            "Hokusai Multi-Resolution CMS: True vs Estimated",
            fontsize=13,
            pad=12,
        )
        ax1.grid(True)

        ax1.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )

        fig1.subplots_adjust(left=0.08, right=0.80, top=0.90, bottom=0.12)

        counts_path = graphs_base / f"{output_prefix}_counts.png"
        _ensure_parent_dir(counts_path)
        fig1.savefig(counts_path, dpi=150)
        saved_paths.append(counts_path)

        print("Plotting Absolute and Relative Error...")
        fig2, (ax2a, ax2b) = plt.subplots(2, 1, sharex=True, figsize=(12, 7))

        for target in targets:
            ws = sorted(results[target].keys())
            abs_errs = [abs(results[target][w][1] - results[target][w][0]) for w in ws]
            rel_errs = []
            for w in ws:
                true_c, est_c = results[target][w]
                if true_c == 0:
                    rel = 0.0
                else:
                    rel = abs(est_c - true_c) / float(true_c)
                rel_errs.append(rel)

            ax2a.plot(ws, abs_errs, marker="o", linestyle="-", label=target)
            ax2b.plot(ws, rel_errs, marker="o", linestyle="-", label=target)

        ax2a.set_ylabel("Absolute error")
        ax2a.set_title(
            "Hokusai Multi-Resolution CMS: Error vs Window Size",
            fontsize=13,
            pad=12,
        )
        ax2a.grid(True)
        ax2a.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )

        ax2b.set_xlabel("Window size (slots)")
        ax2b.set_ylabel("Relative error")
        ax2b.grid(True)
        ax2b.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )

        fig2.subplots_adjust(left=0.08, right=0.80, top=0.93, bottom=0.08, hspace=0.25)

        errors_path = graphs_base / f"{output_prefix}_errors.png"
        _ensure_parent_dir(errors_path)
        fig2.savefig(errors_path, dpi=150)
        saved_paths.append(errors_path)

        print("Computing burst score time series...")
        burst_target = targets[0]  # pick the most frequent
        recent_slots = min(10, max_slot + 1)
        history_slots = min(50, max_slot + 1)

        eval_points = []
        burst_scores = []
        step = max(1, (max_slot + 1) // 50)
        original_max_slot = hok.max_slot_seen
        for t in range(0, max_slot + 1, step):
            hok.max_slot_seen = t
            score = hok.get_burst_score(burst_target, recent_slots, history_slots)
            eval_points.append(t)
            burst_scores.append(score)
        hok.max_slot_seen = original_max_slot

        fig3, ax3 = plt.subplots()
        ax3.plot(eval_points, burst_scores, marker="o", linestyle="-")
        ax3.set_xlabel("Slot index")
        ax3.set_ylabel("Burst score")
        ax3.set_title(f"Burst score over time for n-gram: {burst_target!r}")
        ax3.grid(True)
        fig3.tight_layout()
        burst_path = graphs_base / f"{output_prefix}_burst.png"
        _ensure_parent_dir(burst_path)
        fig3.savefig(burst_path, dpi=150)
        saved_paths.append(burst_path)

        print("Plotting top-K bar chart...")
        largest_window = max(window_sizes_slots)
        true_bar = [results[target][largest_window][0] for target in targets]
        est_bar = [results[target][largest_window][1] for target in targets]
        x = list(range(len(targets)))
        width_bar = 0.35

        fig4, ax4 = plt.subplots()
        ax4.bar([i - width_bar / 2 for i in x], true_bar, width_bar, label="True")
        ax4.bar([i + width_bar / 2 for i in x], est_bar, width_bar, label="Estimated")
        ax4.set_xticks(x)
        ax4.set_xticklabels(targets, rotation=45, ha="right")
        ax4.set_ylabel(f"Count (window={largest_window} slots)")
        ax4.set_title("Top-K n-grams: True vs Estimated (largest window)")
        ax4.legend()
        ax4.grid(axis="y")
        _label_bars(ax4, fmt="{:.0f}")
        fig4.tight_layout()
        bar_path = graphs_base / f"{output_prefix}_topk_bar.png"
        _ensure_parent_dir(bar_path)
        fig4.savefig(bar_path, dpi=150)
        saved_paths.append(bar_path)

        print("Plotting relative error histogram...")
        fig5, ax5 = plt.subplots()
        ax5.hist(all_rel_errors, bins=20, edgecolor="black")
        ax5.set_xlabel("Relative error")
        ax5.set_ylabel("Frequency")
        ax5.set_title("Distribution of relative errors (all targets × windows)")
        ax5.grid(axis="y")
        fig5.tight_layout()
        hist_path = graphs_base / f"{output_prefix}_rel_error_hist.png"
        _ensure_parent_dir(hist_path)
        fig5.savefig(hist_path, dpi=150)
        saved_paths.append(hist_path)

        print("Plotting true count vs absolute error scatter...")
        fig6, ax6 = plt.subplots()
        ax6.scatter(all_true_counts, all_abs_errors, alpha=0.7)
        ax6.set_xlabel("True count")
        ax6.set_ylabel("Absolute error")
        ax6.set_title("True count vs Absolute error")
        ax6.grid(True)
        fig6.tight_layout()
        scatter_path = graphs_base / f"{output_prefix}_true_vs_abs_error.png"
        _ensure_parent_dir(scatter_path)
        fig6.savefig(scatter_path, dpi=150)
        saved_paths.append(scatter_path)

        print("Plotting heatmap of relative errors...")
        ws_sorted = sorted(window_sizes_slots)
        heat_data: List[List[float]] = []
        for target in targets:
            row = []
            for w in ws_sorted:
                true_c, est_c = results[target][w]
                abs_err = abs(est_c - true_c)
                if true_c == 0:
                    rel = 0.0
                else:
                    rel = abs_err / float(true_c)
                row.append(rel)
            heat_data.append(row)

        fig7, ax7 = plt.subplots()
        cax = ax7.imshow(heat_data, aspect="auto", interpolation="nearest")
        ax7.set_xticks(range(len(ws_sorted)))
        ax7.set_xticklabels(ws_sorted)
        ax7.set_yticks(range(len(targets)))
        ax7.set_yticklabels(targets)
        ax7.set_xlabel("Window size (slots)")
        ax7.set_ylabel("Target n-gram")
        ax7.set_title("Relative error heatmap")
        fig7.colorbar(cax, ax=ax7, label="Relative error")
        fig7.tight_layout()
        heatmap_path = graphs_base / f"{output_prefix}_rel_error_heatmap.png"
        _ensure_parent_dir(heatmap_path)
        fig7.savefig(heatmap_path, dpi=150)
        saved_paths.append(heatmap_path)

        print("Plotting time-series counts for one target...")
        timeseries_target = targets[0]
        window_for_ts = min(32, max_slot + 1)  # fixed sliding window
        ts_eval_points = []
        ts_true = []
        ts_est = []
        step_ts = max(1, (max_slot + 1) // 50)  # about 50 points

        for t in range(0, max_slot + 1, step_ts):
            start = max(0, t - window_for_ts + 1)
            end = t
            true_c = true_count_interval(events, timeseries_target, start, end)
            est_c = hok.query_interval(timeseries_target, start, end)
            ts_eval_points.append(t)
            ts_true.append(true_c)
            ts_est.append(est_c)

        fig8, ax8 = plt.subplots()
        ax8.plot(ts_eval_points, ts_true, marker="o", linestyle="-", label="True")
        ax8.plot(ts_eval_points, ts_est, marker="x", linestyle="--", label="Estimated")
        ax8.set_xlabel("Slot index")
        ax8.set_ylabel(f"Count in last {window_for_ts} slots")
        ax8.set_title(f"Time-series counts for {timeseries_target!r}")
        ax8.grid(True)
        ax8.legend()
        fig8.tight_layout()
        ts_path = graphs_base / f"{output_prefix}_timeseries.png"
        _ensure_parent_dir(ts_path)
        fig8.savefig(ts_path, dpi=150)
        saved_paths.append(ts_path)

        print("Plotting memory usage per Hokusai level...")
        levels = list(range(len(mem_usage)))
        fig9, ax9 = plt.subplots()
        ax9.bar(levels, mem_usage)
        ax9.set_xlabel("Level (j)")
        ax9.set_ylabel("Number of CMS counters (width × depth × segments)")
        ax9.set_title("Memory usage per Hokusai level")
        ax9.set_xticks(levels)
        ax9.grid(axis="y")
        _label_bars(ax9, fmt="{:.0f}")
        fig9.tight_layout()
        mem_path = graphs_base / f"{output_prefix}_memory_per_level.png"
        _ensure_parent_dir(mem_path)
        fig9.savefig(mem_path, dpi=150)
        saved_paths.append(mem_path)
    else:
        print("\nSkipping plot generation (--no-plot enabled).")

    # ---- Scalar accuracy / error metrics ----
    total_counters = sum(mem_usage)
    metrics_all = compute_accuracy_metrics(
        all_true_counts=all_true_counts,
        all_abs_errors=all_abs_errors,
        all_rel_errors=all_rel_errors,
        total_counters=total_counters,
    )
    metrics_active = compute_accuracy_metrics(
        all_true_counts=active_true_counts,
        all_abs_errors=active_abs_errors,
        all_rel_errors=active_rel_errors,
        total_counters=total_counters,
    )

    if inactive_total > 0:
        false_positive_rate = inactive_false_positives / float(inactive_total)
        avg_false_positive = sum(inactive_estimates) / float(inactive_total)
        max_false_positive = max(inactive_estimates)
    else:
        false_positive_rate = 0.0
        avg_false_positive = 0.0
        max_false_positive = 0

    metrics: Dict[str, float] = dict(metrics_all)
    metrics.update(
        {
            "mae_active": metrics_active["mae"],
            "rmse_active": metrics_active["rmse"],
            "mean_relative_error_active": metrics_active["mean_relative_error"],
            "weighted_relative_l1_active": metrics_active[
                "weighted_relative_l1"
            ],
            "false_positive_rate": false_positive_rate,
            "avg_false_positive_estimate": avg_false_positive,
            "max_false_positive_estimate": float(max_false_positive),
        }
    )
    bytes_per_counter = 4.0
    metrics["estimated_bytes"] = metrics["total_counters"] * bytes_per_counter
    t_eval_end = time.perf_counter()
    t_total_end = time.perf_counter()

    num_events = len(events)
    metrics.update(
        {
            "load_time_s": t_load_end - t_load_start,
            "stream_time_s": t_stream_end - t_stream_start,
            "eval_time_s": t_eval_end - t_eval_start,
            "total_time_s": t_total_end - t_total_start,
            "throughput_events_per_s": (num_events / (t_stream_end - t_stream_start))
            if (t_stream_end - t_stream_start) > 0
            else 0.0,
        }
    )

    if metrics_out:
        metrics_path = Path(metrics_out)
        _ensure_parent_dir(metrics_path)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)

    print("\nAccuracy / error summary for this configuration (all queries):")
    print(f"  num_queries          = {int(metrics_all['num_queries'])}")
    print(
        f"  mae                  = {metrics_all['mae']:.6f} "
        "(mean absolute error across targets×windows)"
    )
    print(
        f"  rmse                 = {metrics_all['rmse']:.6f} "
        "(root mean squared error)"
    )
    print(
        f"  mean_relative_error  = {metrics_all['mean_relative_error']:.6f} "
        "(unweighted mean of per-query relative error)"
    )
    print(
        f"  weighted_relative_l1 = {metrics_all['weighted_relative_l1']:.6f} "
        "(∑|err| / ∑true, LOWER is better)"
    )
    print(
        f"  median_relative_err  = {metrics_all['median_relative_error']:.6f}, "
        f"p90_relative_err = {metrics_all['p90_relative_error']:.6f}, "
        f"max_relative_err = {metrics_all['max_relative_error']:.6f}"
    )
    print(
        f"  total_counters       = {int(metrics_all['total_counters'])} "
        "(approximate memory footprint proxy)"
    )

    print("\nAccuracy / error summary (ACTIVE queries only, true > 0):")
    print(
        f"  mae_active           = {metrics['mae_active']:.6f}, "
        f"rmse_active = {metrics['rmse_active']:.6f}"
    )
    print(
        "  mean_relative_error_active  = "
        f"{metrics['mean_relative_error_active']:.6f}"
    )
    print(
        "  weighted_relative_l1_active = "
        f"{metrics['weighted_relative_l1_active']:.6f} (LOWER is better)"
    )

    print("\nInactive-query (true = 0) noise statistics:")
    print(f"  num_inactive_queries = {inactive_total}")
    print(f"  false_positive_rate  = {false_positive_rate:.6f}")
    print(
        f"  avg_false_positive_estimate = {avg_false_positive:.6f}, "
        f"max_false_positive_estimate = {max_false_positive}"
    )

    if generate_plots and saved_paths:
        print("\nSaved plots to:")
        for p in saved_paths:
            print("  ", p)
    else:
        print("\nPlots were skipped (no-plot mode).")
    print("\nDone.")

    return metrics


def _percentile(sorted_values: List[float], pct: float) -> float:
    """
    Compute a simple percentile (0–100) from a pre-sorted list of floats.
    """
    if not sorted_values:
        return 0.0
    if pct <= 0.0:
        return float(sorted_values[0])
    if pct >= 100.0:
        return float(sorted_values[-1])
    k = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    return float(sorted_values[k])


def compute_accuracy_metrics(
    all_true_counts: List[int],
    all_abs_errors: List[int],
    all_rel_errors: List[float],
    total_counters: int,
) -> Dict[str, float]:
    """
    Compute scalar accuracy/error metrics that can be compared across models.

    Recommended comparison metric: `weighted_relative_l1` (lower is better).
    """
    n = len(all_true_counts)
    if n == 0:
        return {
            "num_queries": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "mean_relative_error": 0.0,
            "weighted_relative_l1": 0.0,
            "median_relative_error": 0.0,
            "p90_relative_error": 0.0,
            "max_relative_error": 0.0,
            "total_counters": float(total_counters),
        }

    mae = sum(all_abs_errors) / float(n)
    mse = sum(e * e for e in all_abs_errors) / float(n)
    rmse = math.sqrt(mse)
    mean_rel_error = sum(all_rel_errors) / float(n)

    sum_true = sum(all_true_counts)
    sum_abs = sum(all_abs_errors)
    if sum_true > 0:
        weighted_relative_l1 = sum_abs / float(sum_true)
    else:
        weighted_relative_l1 = 0.0
    sorted_rels = sorted(all_rel_errors)
    median_rel = _percentile(sorted_rels, 50.0)
    p90_rel = _percentile(sorted_rels, 90.0)
    max_rel = sorted_rels[-1] if sorted_rels else 0.0

    return {
        "num_queries": float(n),
        "mae": mae,
        "rmse": rmse,
        "mean_relative_error": mean_rel_error,
        "weighted_relative_l1": weighted_relative_l1,
        "median_relative_error": median_rel,
        "p90_relative_error": p90_rel,
        "max_relative_error": max_rel,
        "total_counters": float(total_counters),
    }


def run_width_sweep(
    input_path: Path,
    max_events: int,
    base_slot_seconds: int,
    max_levels: int,
    base_widths: List[int],
    depth: int,
    top_k: int,
    output_prefix: str,
    max_seconds: float,
    eval_last_slots: int,
    graphs_dir: Path = None,
    generate_plots: bool = True,
) -> None:
    """
    Run a simple parameter sweep over base_width values, using the same dataset.

    For each base_width, we:
      * run the standard experiment (including plots, prefixed by base_width)
      * compute scalar accuracy metrics
    and then print a summary table along with the "best" width according to
    weighted_relative_l1.
    """
    print("\n=== Running base_width sweep ===")
    print("Candidate base_widths:", base_widths)

    sweep_results: List[Tuple[int, Dict[str, float]]] = []
    for bw in base_widths:
        print("\n========================================")
        print(f"Evaluating configuration base_width={bw}, depth={depth}")
        print("========================================")
        metrics = run_experiment(
            input_path=input_path,
            max_events=max_events,
            base_slot_seconds=base_slot_seconds,
            max_levels=max_levels,
            base_width=bw,
            depth=depth,
            top_k=top_k,
            output_prefix=f"{output_prefix}_w{bw}",
            max_seconds=max_seconds,
            eval_last_slots=eval_last_slots,
            graphs_dir=graphs_dir,
            generate_plots=generate_plots,
        )
        if metrics:
            sweep_results.append((bw, metrics))

    if not sweep_results:
        print("No results collected in width sweep (no events or errors).")
        return

    sweep_results.sort(key=lambda x: x[1].get("weighted_relative_l1", float("inf")))

    print("\n=== Width sweep summary (sorted by weighted_relative_l1, lower is better) ===")
    header = (
        "base_width\t"
        "total_counters\t"
        "weighted_rel_L1\t"
        "mean_rel_err\t"
        "mae\t"
        "rmse"
    )
    print(header)
    for bw, m in sweep_results:
        print(
            f"{bw}\t"
            f"{int(m.get('total_counters', 0.0))}\t"
            f"{m.get('weighted_relative_l1', float('nan')):.6f}\t"
            f"{m.get('mean_relative_error', float('nan')):.6f}\t"
            f"{m.get('mae', float('nan')):.6f}\t"
            f"{m.get('rmse', float('nan')):.6f}"
        )

    best_bw, best_metrics = sweep_results[0]
    print(
        "\nBest base_width by weighted_relative_l1: "
        f"{best_bw} (weighted_relative_l1={best_metrics['weighted_relative_l1']:.6f})"
    )

def main():
    parser = argparse.ArgumentParser(
        description="Hokusai Multi-Resolution Count-Min Sketch experiment."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to timestamp,ngram CSV (from create_bigrams.py)",
    )
    # Everything below has defaults – you rarely need to touch these.
    parser.add_argument(
        "--max-events",
        type=int,
        default=200000,
        help="Maximum number of events to load (0 = no limit, default=200000)",
    )
    parser.add_argument(
        "--base-slot-seconds",
        type=int,
        default=60,
        help="Base time unit in seconds (default=60 for 1-minute slots)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=14 * 24 * 3600,
        help=(
            "Maximum time span in seconds from the first timestamp "
            "(default = first 14 days; use 0 for no limit)"
        ),
    )
    parser.add_argument(
        "--max-levels",
        type=int,
        default=8,
        help="Maximum levels in dyadic hierarchy (default=8)",
    )
    parser.add_argument(
        "--base-width",
        type=int,
        default=2048,
        help="CMS width at level 0 (higher levels get folded widths, default=2048)",
    )
    parser.add_argument(
        "--eval-last-slots",
        type=int,
        default=512,
        help=(
            "Number of most recent time slots to use for target selection and "
            "evaluation (default=512; use 0 to use entire range). With "
            "base-slot-seconds=60, 512 slots ≈ 8.5 hours."
        ),
    )
    parser.add_argument(
        "--width-grid",
        type=str,
        default="",
        help=(
            "Optional comma-separated list of base_width values for a parameter "
            "sweep (e.g. '512,1024,2048'). If provided, overrides --base-width."
        ),
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="CMS depth (number of hash rows, default=5)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top n-grams to evaluate and plot (default=5)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="hokusai_output",
        help="Prefix for output PNG files (default='hokusai_output')",
    )
    parser.add_argument(
        "--graphs-dir",
        type=str,
        default="analysis/graphs/hakusai",
        help="Base directory for saving plots (default='analysis/graphs/hakusai')",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation (useful for programmatic runs)",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="",
        help="Optional path to write metrics JSON",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    graphs_dir = Path(args.graphs_dir)
    generate_plots = not args.no_plot
    metrics_out = Path(args.metrics_out) if args.metrics_out else None

    if args.width_grid:
        raw_values = [v.strip() for v in args.width_grid.split(",") if v.strip()]
        try:
            base_widths = [int(v) for v in raw_values]
        except ValueError:
            raise SystemExit(
                f"Invalid --width-grid value: {args.width_grid!r}. "
                "Expected a comma-separated list of integers."
            )
        run_width_sweep(
            input_path=input_path,
            max_events=args.max_events,
            base_slot_seconds=args.base_slot_seconds,
            max_levels=args.max_levels,
            base_widths=base_widths,
            depth=args.depth,
            top_k=args.top_k,
            output_prefix=args.output_prefix,
            max_seconds=args.max_seconds,
            eval_last_slots=args.eval_last_slots,
            graphs_dir=graphs_dir,
            generate_plots=generate_plots,
        )
    else:
        run_experiment(
            input_path=input_path,
            max_events=args.max_events,
            base_slot_seconds=args.base_slot_seconds,
            max_levels=args.max_levels,
            base_width=args.base_width,
            depth=args.depth,
            top_k=args.top_k,
            output_prefix=args.output_prefix,
            max_seconds=args.max_seconds,
            eval_last_slots=args.eval_last_slots,
            graphs_dir=graphs_dir,
            generate_plots=generate_plots,
            metrics_out=metrics_out,
        )

if __name__ == "__main__":
    main()
