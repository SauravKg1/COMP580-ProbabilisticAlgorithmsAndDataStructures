import argparse, os, re, sys, random, hashlib, heapq
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- Config -----------------
SEED = 5802
random.seed(SEED)
D = 5
R_LIST = [2**10, 2**14, 2**18]
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
P = 2305843009213693951  # 2^61 - 1
# -----------------------------------------

# ----------------- Utilities --------------
def tok_id(s: str) -> int:
    """Deterministic 64-bit token id (for cross-run stability)."""
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "big", signed=False)

def toks(s: str):
    return TOKEN_RE.findall(s.lower())

def stream_tokens(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                for t in toks(parts[1]):
                    yield t

def approx_counter_bytes(c: Counter) -> int:
    size = sys.getsizeof(c)
    for k, v in c.items():
        size += sys.getsizeof(k)
        size += sys.getsizeof(v)
    return size
# ------------------------------------------

# -------- 2-universal (k=2) hash family ----
class Hash2U:
    """
    2-universal (pairwise independent) hashing.
    Index hash: h_i(x) = ((a_i * x + b_i) mod P) mod R
    Sign hash:  s_i(x) = (-1)^( ((a'_i * x + b'_i) mod P) & 1 ), i.e., parity bit → {+1,-1}
    Each row i has independent coefficients.
    """
    def __init__(self, d: int, R: int, seed: int):
        self.d, self.R = d, R
        rng = random.Random(seed + 17*R + 13*d + 3)

        # a in [1, P-1], b in [0, P-1]
        self.a  = [rng.randrange(1, P-1) for _ in range(d)]
        self.b  = [rng.randrange(0, P-1) for _ in range(d)]

        # independent coefficients for the sign hash
        self.asg = [rng.randrange(1, P-1) for _ in range(d)]
        self.bsg = [rng.randrange(0, P-1) for _ in range(d)]

    def h(self, i: int, x: int) -> int:
        return ( (self.a[i] * x + self.b[i]) % P ) % self.R

    def s(self, i: int, x: int) -> int:
        return 1 if ( (self.asg[i] * x + self.bsg[i]) % P ) & 1 == 0 else -1
# ------------------------------------------

# -------------- Sketch builders ------------
def make_countmin(hf: Hash2U):
    D, R = hf.d, hf.R
    table = [[0]*R for _ in range(D)]
    def insert(x: int):
        for i in range(D): table[i][hf.h(i,x)] += 1
    def query(x: int) -> int:
        return min(table[i][hf.h(i,x)] for i in range(D))
    return query, insert

def make_countmedian(hf: Hash2U):
    D, R = hf.d, hf.R
    table = [[0]*R for _ in range(D)]
    def insert(x: int):
        for i in range(D): table[i][hf.h(i,x)] += 1
    def query(x: int) -> int:
        vals = [table[i][hf.h(i,x)] for i in range(D)]
        vals.sort(); m = len(vals)//2
        return vals[m] if len(vals)%2 else (vals[m-1]+vals[m])//2
    return query, insert

def make_countsketch(hf: Hash2U):
    D, R = hf.d, hf.R
    table = [[0]*R for _ in range(D)]
    def insert(x: int):
        for i in range(D):
            idx = hf.h(i,x); s = hf.s(i,x)
            table[i][idx] += s
    def query(x: int) -> int:
        vals = []
        for i in range(D):
            idx = hf.h(i,x); s = hf.s(i,x)
            vals.append(s*table[i][idx])
        vals.sort(); m = len(vals)//2
        return vals[m] if len(vals)%2 else (vals[m-1]+vals[m])//2
    return query, insert
# ------------------------------------------

# -------------- Top-k helpers --------------
def topk_update(heap, entry, K, token, est):
    # min-heap (est, token) + lazy deletion
    old = entry.get(token)
    if old is None:
        if len(heap) < K:
            entry[token] = est; heapq.heappush(heap, (est, token))
        else:
            while heap and entry.get(heap[0][1], None) != heap[0][0]:
                heapq.heappop(heap)
            if not heap or est > heap[0][0]:
                entry[token] = est; heapq.heappush(heap, (est, token))
    elif est > old:
        entry[token] = est; heapq.heappush(heap, (est, token))

def topk_list(heap, entry, K):
    tmp = []
    for v,t in heap:
        if entry.get(t, None) == v:
            tmp.append((t,v))
    tmp.sort(key=lambda x: x[1], reverse=True)
    return tmp[:K]
# ------------------------------------------

# -------------- Eval & Plotting ------------
def build_truth(aol_path: str):
    cnt = Counter(); n = 0
    for t in stream_tokens(aol_path):
        cnt[t] += 1; n += 1
        if n % 1_000_000 == 0:
            print(f"  [truth] {n:,} tokens...")
    print(f"[truth] total={n:,} tokens, unique={len(cnt):,}")
    return cnt

def eval_sets(cnt: Counter):
    freq100 = [w for (w, _) in cnt.most_common(100)]
    infreq100 = [w for (w, c) in sorted(cnt.items(), key=lambda x:(x[1], x[0]))[:100]]
    keys = list(cnt.keys())
    rand100 = random.Random(SEED+12345).sample(keys, 100) if len(keys)>=100 else keys
    return freq100, rand100, infreq100

def rel_errors(words, query_fn, truth: Counter):
    # x-axis sorted by TRUE frequency (desc), per assignment
    words_sorted = sorted(words, key=lambda w: truth[w], reverse=True)
    errs = []
    for w in words_sorted:
        c = truth[w]
        est = query_fn(tok_id(w))
        errs.append(0.0 if c == 0 else abs(est - c)/c)
    return words_sorted, errs

def plot_errors(xwords, series, title, out):
    # Wider so word labels remain readable; rotate labels vertically
    fig_w = max(12, 0.18 * len(xwords))
    plt.figure(figsize=(fig_w, 4.2), dpi=150)

    positions = list(range(len(xwords)))  # numeric positions for plotting
    for label, ys in series:
        plt.plot(positions, ys, label=label, linewidth=1.4)

    # Use the exact words as tick labels
    plt.xticks(positions, xwords, rotation=90, fontsize=7)
    plt.margins(x=0.005)
    plt.xlabel("Tokens (sorted by true frequency)")
    plt.ylabel("Relative error |ĉ - c| / c")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def plot_intersection_vs_R(R_LIST, inter_curve, out_path):
    plt.figure(figsize=(6,4))
    for label, ys in inter_curve.items():
        plt.plot(R_LIST, ys, marker="o", label=label, linewidth=1.4)
    plt.xticks(R_LIST, [str(r) for r in R_LIST])
    plt.xlabel("R")
    plt.ylabel("|Top-500 (approx) ∩ Top-100 (true)|")
    plt.title("Intersection vs R")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path); plt.close()
# ------------------------------------------

# -------------- Main pipeline --------------
def run(aol_path: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    pdir = os.path.join(outdir, "plots"); os.makedirs(pdir, exist_ok=True)

    print(f"[{datetime.now()}] Building exact counts…")
    truth = build_truth(aol_path)

    # dictionary memory stats
    mem_bytes = approx_counter_bytes(truth)
    with open(os.path.join(outdir, "dictionary_stats.txt"), "w") as fh:
        fh.write(f"Approx dictionary memory usage: {mem_bytes/(1024**2):.2f} MB\n")
        fh.write(f"Unique tokens: {len(truth):,}\n")

    freq100, rand100, infreq100 = eval_sets(truth)
    true_top100 = set([w for (w, _) in truth.most_common(100)])

    inter_curve = {"Count-Min":[], "Count-Median":[], "Count-Sketch":[]}

    for R in R_LIST:
        print(f"\n[{datetime.now()}] R={R} with 2-universal hashing…")
        hf_cm   = Hash2U(D, R, SEED + 0)
        hf_cmed = Hash2U(D, R, SEED + 1)
        hf_csk  = Hash2U(D, R, SEED + 2)

        q_cm,   ins_cm   = make_countmin(hf_cm)
        q_cmed, ins_cmed = make_countmedian(hf_cmed)
        q_csk,  ins_csk  = make_countsketch(hf_csk)

        # Top-500 trackers
        h_cm, e_cm = [], {}
        h_cmed, e_cmed = [], {}
        h_csk, e_csk = [], {}
        K = 500

        n = 0
        for t in stream_tokens(aol_path):
            x = tok_id(t)
            ins_cm(x); ins_cmed(x); ins_csk(x)
            topk_update(h_cm,   e_cm,   K, t, q_cm(x))
            topk_update(h_cmed, e_cmed, K, t, q_cmed(x))
            topk_update(h_csk,  e_csk,  K, t, q_csk(x))
            n += 1
            if n % 1_000_000 == 0:
                print(f"  streamed {n:,} tokens for R={R}")

        # Error plots (Freq-100 / Rand-100 / Infreq-100)
        for name, words in [("Freq-100",freq100), ("Rand-100",rand100), ("Infreq-100",infreq100)]:
            xs, e1 = rel_errors(words, q_cm,   truth)
            _,  e2 = rel_errors(words, q_cmed, truth)
            _,  e3 = rel_errors(words, q_csk,  truth)
            outp = os.path.join(pdir, f"errors_{name.replace('-','').lower()}_R{R}_2u.png")
            plot_errors(xs, [("Count-Min",e1), ("Count-Median",e2), ("Count-Sketch",e3)],
                        f"Relative Errors – {name} (R={R}, 2-universal)", outp)
            print("  saved", outp)

        # Intersection sizes with true Top-100
        approx_cm   = [w for (w,_) in topk_list(h_cm,   e_cm,   K)]
        approx_cmed = [w for (w,_) in topk_list(h_cmed, e_cmed, K)]
        approx_csk  = [w for (w,_) in topk_list(h_csk,  e_csk,  K)]

        inter_cm   = len(true_top100.intersection(approx_cm))
        inter_cmed = len(true_top100.intersection(approx_cmed))
        inter_csk  = len(true_top100.intersection(approx_csk))
        inter_curve["Count-Min"].append(inter_cm)
        inter_curve["Count-Median"].append(inter_cmed)
        inter_curve["Count-Sketch"].append(inter_csk)
        print(f"  intersections |approx500 ∩ true100|  CM={inter_cm}  CMED={inter_cmed}  CSK={inter_csk}")

    # Intersection vs R plot
    inter_path = os.path.join(pdir, "intersection_vs_R_2u.png")
    plot_intersection_vs_R(R_LIST, inter_curve, inter_path)
    print("saved", inter_path)

    # Report
    with open(os.path.join(outdir, "report.md"), "w") as f:
        f.write("# Assignment 2 Report (2-universal hashing)\n\n")
        f.write(f"- Generated: {datetime.now()}\n")
        f.write(f"- Unique tokens: {len(truth):,}\n")
        f.write(f"- Approx dictionary memory: {mem_bytes/(1024**2):.2f} MB\n")
        f.write(f"- Parameters: d={D}, R={R_LIST}\n\n")
        for R in R_LIST:
            f.write(f"## R={R}\n")
            f.write(f"- plots/errors_freq100_R{R}_2u.png\n")
            f.write(f"- plots/errors_rand100_R{R}_2u.png\n")
            f.write(f"- plots/errors_infreq100_R{R}_2u.png\n\n")
        f.write("## Intersection vs R\n")
        f.write(f"- plots/intersection_vs_R_2u.png\n")
        f.write("- Intersections by sketch:\n")
        for name, vals in inter_curve.items():
            f.write(f"  - {name}: {vals}\n")
        f.write("\n## Notes\n")
        f.write("- Count-Min overestimates; Count-Median reduces collision impact; Count-Sketch leverages ±1 signs.\n")
        f.write("- As R increases, collisions drop → lower relative errors and better heavy-hitter recovery.\n")
        f.write("- 2-universal hashing gives pairwise independence (a*x+b mod P); it is weaker than 4-universal but faster and often sufficient in practice.\n")

def main():
    p = argparse.ArgumentParser(description="COMP 480/580 Assignment 2 – 2-universal sketches")
    p.add_argument("--aol_path", default="./user-ct-test-collection-01.txt")
    p.add_argument("--out", default="./outputs")
    a = p.parse_args()

    if not os.path.isfile(a.aol_path):
        print("ERROR: AOL file not found at", a.aol_path); sys.exit(1)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.out, "plots"), exist_ok=True)

    run(a.aol_path, a.out)

if __name__ == "__main__":
    main()
