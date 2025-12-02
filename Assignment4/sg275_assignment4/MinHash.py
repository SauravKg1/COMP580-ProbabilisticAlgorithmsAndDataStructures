import random
import time
from collections import defaultdict
from typing import List, Set, Iterable, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ssl
import urllib.request
import io

DATA_URL = "https://cim.mcgill.ca/~dudek/206/Logs/AOL-user-ct-collection/user-ct-test-collection-01.txt"

DEFAULT_K = 2
DEFAULT_L = 50
DEFAULT_B = 64
DEFAULT_R = 2 ** 20

NUM_QUERIES = 200
RANDOM_SEED = 42

_PRIME = 4294967311
_MAX_HASHES = 2000

random.seed(42)
_A_COEFFS = [random.randint(1, _PRIME - 1) for _ in range(_MAX_HASHES)]
_B_COEFFS = [random.randint(0, _PRIME - 1) for _ in range(_MAX_HASHES)]

def get_3grams(s: str, n: int = 3) -> Set[str]:
    s = s.lower()
    if len(s) < n:
        return {s}
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    """
    Compute Jaccard similarity between two sets.
    """
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def _base_hash_shingle(shingle: str) -> int:
    return hash(shingle) & 0xFFFFFFFFFFFFFFFF

def MinHash(A: Iterable[str], k: int) -> List[int]:
    if k <= 0:
        raise ValueError("k must be positive")
    if k > _MAX_HASHES:
        raise ValueError(f"k must be <= {_MAX_HASHES}")

    if not isinstance(A, set):
        A = set(A)

    if not A:
        # Empty set: use a large constant
        return [(_PRIME - 1)] * k

    base_hashes = [_base_hash_shingle(s) for s in A]

    signature = [_PRIME - 1] * k
    for i in range(k):
        a = _A_COEFFS[i]
        b = _B_COEFFS[i]
        min_val = _PRIME - 1
        for h in base_hashes:
            val = (a * h + b) % _PRIME
            if val < min_val:
                min_val = val
        signature[i] = min_val
    return signature


class HashTable:
    def __init__(self, K: int, L: int, B: int, R: int):
        if K <= 0 or L <= 0 or B <= 0 or R <= 0:
            raise ValueError("K, L, B, and R must all be positive integers")

        self.K = K
        self.L = L
        self.B = B
        self.R = R

        self.tables: List[Dict[int, set]] = [defaultdict(set) for _ in range(L)]

    def _get_band_bucket(self, band_values: List[int], table_index: int) -> int:
        combined = (table_index, *band_values)
        h = hash(combined)
        h = (h & 0x7FFFFFFF) % self.R
        bucket = h % self.B
        return bucket

    def insert(self, hashcodes: List[int], id_val: int) -> None:
        m = len(hashcodes)
        required = self.K * self.L
        if m < required:
            raise ValueError(
                f"HashTable.insert expected at least {required} hashcodes, got {m}."
            )

        for table_index in range(self.L):
            start = table_index * self.K
            band = hashcodes[start:start + self.K]
            bucket = self._get_band_bucket(band, table_index)
            self.tables[table_index][bucket].add(id_val)

    def lookup(self, hashcodes: List[int]) -> List[int]:
        m = len(hashcodes)
        required = self.K * self.L
        if m < required:
            raise ValueError(
                f"HashTable.lookup expected at least {required} hashcodes, got {m}."
            )

        candidates = set()
        for table_index in range(self.L):
            start = table_index * self.K
            band = hashcodes[start:start + self.K]
            bucket = self._get_band_bucket(band, table_index)
            bucket_set = self.tables[table_index].get(bucket)
            if bucket_set:
                candidates.update(bucket_set)
        return list(candidates)


def task0_demo():
    """
    Task 0:
      - Generate 100 MinHash values (m = 100) for the two given strings S1 and S2.
      - Verify that the MinHash estimate is close to the actual 3-gram Jaccard similarity.
    """

    S1 = (
        "The mission statement of the WCSCC and area employers recognize the importance of good "
        "attendance on the job. Any student whose absences exceed 18 days is jeopardizing their "
        "opportunity for advanced placement as well as hindering his/her likelihood for successfully "
        "completing their program."
    )

    S2 = (
        "The WCSCC's mission statement and surrounding employers recognize the importance of great "
        "attendance. Any student who is absent more than 18 days will loose the opportunity for "
        "successfully completing their trade program."
    )

    A = get_3grams(S1)
    B = get_3grams(S2)

    true_j = jaccard_similarity(A, B)

    m = 100
    mh1 = MinHash(A, m)
    mh2 = MinHash(B, m)
    est_j = sum(1 for x, y in zip(mh1, mh2) if x == y) / m

    print("=== Task 0: MinHash Demo ===")
    print(f"True Jaccard similarity (3-grams) : {true_j:.6f}")
    print(f"Estimated Jaccard via MinHash (m={m}) : {est_j:.6f}")
    print("")

def load_urls(url: str = DATA_URL) -> List[str]:
    ctx = ssl._create_unverified_context()

    with urllib.request.urlopen(url, context=ctx) as resp:
        data_bytes = resp.read()

    df = pd.read_csv(io.BytesIO(data_bytes), sep="\t")

    urllist = df["ClickURL"].dropna().unique()
    return list(urllist)


def build_lsh_index(
    urls: List[str],
    K: int,
    L: int,
    B: int,
    R: int,
) -> Tuple[HashTable, List[Set[str]], List[List[int]]]:
    ht = HashTable(K=K, L=L, B=B, R=R)
    url_shingles: List[Set[str]] = []
    url_signatures: List[List[int]] = []

    num_hashes = K * L  # Using K * L MinHash values for L bands of size K

    for idx, url in enumerate(urls):
        shingles = get_3grams(url)
        sig = MinHash(shingles, num_hashes)
        ht.insert(sig, idx)
        url_shingles.append(shingles)
        url_signatures.append(sig)

    return ht, url_shingles, url_signatures


def sample_queries(n_urls: int, num_queries: int, seed: int) -> List[int]:
    if num_queries > n_urls:
        raise ValueError("num_queries cannot exceed number of URLs")
    random.seed(seed)
    return random.sample(range(n_urls), num_queries)

def task1_lsh_retrieval(
    urls: List[str],
    url_shingles: List[Set[str]],
    url_signatures: List[List[int]],
    ht: HashTable,
    query_indices: List[int],
) -> None:
    all_sims = []
    top10_sims = []
    total_lookup_time = 0.0

    for q_idx in query_indices:
        query_sig = url_signatures[q_idx]
        query_set = url_shingles[q_idx]

        t0 = time.time()
        candidate_ids = ht.lookup(query_sig)
        t1 = time.time()
        total_lookup_time += (t1 - t0)

        candidate_ids = [cid for cid in candidate_ids if cid != q_idx]

        if not candidate_ids:
            continue

        sims_for_query = []
        for cid in candidate_ids:
            sim = jaccard_similarity(query_set, url_shingles[cid])
            sims_for_query.append((cid, sim))
            all_sims.append(sim)

        # Top-10 by actual Jaccard similarity
        sims_for_query.sort(key=lambda x: x[1], reverse=True)
        for _, sim in sims_for_query[:10]:
            top10_sims.append(sim)

    mean_all = sum(all_sims) / len(all_sims) if all_sims else 0.0
    mean_top10 = sum(top10_sims) / len(top10_sims) if top10_sims else 0.0
    avg_query_time = total_lookup_time / len(query_indices) if query_indices else 0.0

    print("=== Task 1: LSH Retrieval (K=2, L=50, B=64, R=2^20) ===")
    print(f"Number of URLs: {len(urls)}")
    print(f"Number of queries: {len(query_indices)}")
    print(f"Mean Jaccard similarity (all retrieved URLs): {mean_all:.6f}")
    print(f"Mean Jaccard similarity (top-10 URLs per query): {mean_top10:.6f}")
    print(f"Average lookup time per query (seconds): {avg_query_time:.6f}")
    print("")

def task2_bruteforce(
    urls: List[str],
    url_shingles: List[Set[str]],
    query_indices: List[int],
) -> None:
    n = len(urls)
    num_queries = len(query_indices)

    print("=== Task 2: Brute-force Jaccard Similarity ===")
    print(f"Number of URLs: {n}")
    print(f"Number of queries: {num_queries}")

    start_time = time.time()
    pair_count = 0

    for q_idx in query_indices:
        q_set = url_shingles[q_idx]
        for j in range(n):
            sim = jaccard_similarity(q_set, url_shingles[j])
            pair_count += 1

    elapsed = time.time() - start_time
    time_per_query = elapsed / num_queries if num_queries else 0.0
    time_per_pair = elapsed / pair_count if pair_count else 0.0

    total_pairs_all = n * (n - 1) // 2
    estimated_total_all = time_per_pair * total_pairs_all

    print(f"Total brute-force time for {pair_count} pairs (seconds): {elapsed:.6f}")
    print(f"Average time per query (seconds): {time_per_query:.6f}")
    print(f"Estimated time for all ~{total_pairs_all} URL pairs (seconds): {estimated_total_all:.2f}")
    print("")

def task3_tune_K_L(
    urls: List[str],
    url_shingles: List[Set[str]],
    query_indices: List[int],
    K_values = (2, 3, 4, 5, 6),
    L_values = (20, 50, 100),
    B: int = DEFAULT_B,
    R: int = DEFAULT_R,
) -> None:
    print("=== Task 3: Tuning K and L ===")
    print("K values:", K_values)
    print("L values:", L_values)
    print("")

    results = []  # (K, L, mean_jaccard, avg_time_per_query)

    for K in K_values:
        for L in L_values:
            num_hashes = K * L
            ht = HashTable(K=K, L=L, B=B, R=R)
            url_signatures: List[List[int]] = []

            # Build index for this (K, L)
            for idx in range(len(urls)):
                shingles = url_shingles[idx]
                sig = MinHash(shingles, num_hashes)
                ht.insert(sig, idx)
                url_signatures.append(sig)

            # Evaluate retrieval performance
            all_sims = []
            total_lookup_time = 0.0

            for q_idx in query_indices:
                query_sig = url_signatures[q_idx]
                query_set = url_shingles[q_idx]

                t0 = time.time()
                candidate_ids = ht.lookup(query_sig)
                t1 = time.time()
                total_lookup_time += (t1 - t0)

                candidate_ids = [cid for cid in candidate_ids if cid != q_idx]

                for cid in candidate_ids:
                    sim = jaccard_similarity(query_set, url_shingles[cid])
                    all_sims.append(sim)

            mean_sim = sum(all_sims) / len(all_sims) if all_sims else 0.0
            avg_query_time = total_lookup_time / len(query_indices) if query_indices else 0.0

            results.append((K, L, mean_sim, avg_query_time))

            print(f"K={K:2d}, L={L:3d} -> "
                  f"mean Jaccard (retrieved): {mean_sim:.6f}, "
                  f"avg query time (s): {avg_query_time:.6f}")

    print("")
    print("Summary table (K, L, mean_Jaccard, avg_query_time_seconds):")
    for K, L, mean_sim, avg_time in results:
        print(f"{K:2d}, {L:3d}, {mean_sim:.6f}, {avg_time:.6f}")
    print("")

def s_curve_probability(Jx: np.ndarray, K: int, L: int) -> np.ndarray:
    return 1.0 - np.power(1.0 - np.power(Jx, K), L)


def task4_plot_s_curves():
    Jx = np.linspace(0.0, 1.0, 1001)

    # Plot 1: L = 50, vary K
    L_fixed = 50
    K_values = [1, 2, 3, 4, 5, 6, 7]

    plt.figure()
    for K in K_values:
        Px = s_curve_probability(Jx, K, L_fixed)
        plt.plot(Jx, Px, label=f"K={K}")
    plt.xlabel("Jaccard similarity Jx")
    plt.ylabel("Retrieval probability Px")
    plt.title(f"S-curves: L = {L_fixed}, varying K")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("s_curve_L50_vary_K.png")

    K_fixed = 4
    L_values = [5, 10, 20, 50, 100, 150, 200]

    plt.figure()
    for L in L_values:
        Px = s_curve_probability(Jx, K_fixed, L)
        plt.plot(Jx, Px, label=f"L={L}")
    plt.xlabel("Jaccard similarity Jx")
    plt.ylabel("Retrieval probability Px")
    plt.title(f"S-curves: K = {K_fixed}, varying L")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("s_curve_K4_vary_L.png")

    print("=== Task 4: S-curves plotted ===")
    print("Saved: s_curve_L50_vary_K.png")
    print("Saved: s_curve_K4_vary_L.png")
    print("")

def main():
    # Task 0: demo on the two given strings
    task0_demo()

    # Load URLs from AOL dataset via URL (no local file dependency)
    print(f"Downloading AOL dataset from:\n  {DATA_URL}")
    urls = load_urls(DATA_URL)
    print(f"Loaded {len(urls)} unique URLs from remote AOL file")

    # Build LSH index for Task 1 (K=2, L=50, B=64, R=2^20)
    ht, url_shingles, url_signatures = build_lsh_index(
        urls, K=DEFAULT_K, L=DEFAULT_L, B=DEFAULT_B, R=DEFAULT_R
    )

    # Sample 200 query URLs
    query_indices = sample_queries(len(urls), NUM_QUERIES, seed=RANDOM_SEED)

    # Task 1
    task1_lsh_retrieval(urls, url_shingles, url_signatures, ht, query_indices)

    # Task 2
    task2_bruteforce(urls, url_shingles, query_indices)

    # Task 3
    task3_tune_K_L(urls, url_shingles, query_indices)

    # Task 4
    task4_plot_s_curves()


if __name__ == "__main__":
    main()
