# CMS Sketch Experiments

Compare several count-min-sketch variants (Hokusai, ring buffer, BurstSketch, Ada-Sketch) on the n-gram stream dataset in `kaggle_RC_2019-05_ngrams.csv`.

## Setup
- Requires Python 3.10+ and the n-gram CSV (already present as `kaggle_RC_2019-05_ngrams.csv`).
- Install dependencies:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

## Run the full comparison (plots + metrics)
The orchestration script sweeps widths/depths (and optional BurstSketch/Ada configs), runs all four models, and writes metrics and plots.
```bash
python analysis/analysis.py \
  --dataset kaggle_RC_2019-05_ngrams.csv \
  --prefix sweep \
  --max-events 0 \
  --widths 8192,16384,32768 \
  --depths 4 \
  --ada-lambdas 1e-6 \
  --burst-configs 10:4:12
```
- Outputs:
  - Metrics JSONs and the combined TSV summary in `analysis/results/` (e.g., `sweep_*_metrics.json`, `sweep_comparison.tsv`).
  - Per-model plots under `analysis/graphs/{hakusai,ring_buffer,burst_sketch,ada_sketch}` and combined comparison plots in `analysis/graphs/combined/`.
- Flags to speed up iteration:
  - `--skip-model-plots` skips per-model PNGs (still produces combined comparison plots).
  - `--sanity-check` runs a fast subset validation for BurstSketch/Ada-Sketch and exits.
  - `--exact-check` runs a tiny exact-window validation for all models and exits.

## Run individual models
Each model script can be run alone if you want targeted plots/metrics. All expect a CSV with `timestamp,ngram` columns and share `--no-plot` and `--metrics-out` options.
- Hokusai multi-resolution CMS:
  ```bash
  python cms_hakusai/hokusai.py --input kaggle_RC_2019-05_ngrams.csv --base-width 2048 --depth 5 --max-events 200000
  ```
- CMS ring buffer:
  ```bash
  python cms_ring_buffer/ring_buffer.py --input kaggle_RC_2019-05_ngrams.csv --base-width 2048 --depth 5 --buffer-slots 512
  ```
- BurstSketch burst detection:
  ```bash
  python cms_burst_sketch/burst_sketch.py --input kaggle_RC_2019-05_ngrams.csv --width 1024 --depth 4 --bucket-slots 10 --recent-buckets 4 --history-buckets 12
  ```
- Ada-Sketch (forward-decay):
  ```bash
  python cms_ada_sketch/ada_sketch.py --input kaggle_RC_2019-05_ngrams.csv --width 2048 --depth 5 --lambda 1e-6
  ```

## Notes
- `analysis/analysis.py` creates `analysis/.mplconfig` automatically so matplotlib can cache fonts without needing a home directory.
- Use `--max-events` and `--max-seconds` to shrink runtime when iterating; set them to `0` to process the full dataset.
