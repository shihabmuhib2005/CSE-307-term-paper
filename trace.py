"""Synthetic page-access trace generator with a deliberate mid-trace shift.

First half  : locality-heavy. A small, slowly drifting hot set (Zipf-like)
              plus occasional sequential scans through a cold region.
Second half : random / bursty. Uniform random accesses over the whole page
              universe, mixed with short bursts on a few random pages.
"""
import numpy as np


def make_trace(n=10000, universe=300, seed=0, pre_only=False, hot_size=20):
    """Return (trace, shift_index).

    pre_only=True gives a trace that never shifts (used to train the
    'pre-shift' models); shift_index is then n.
    """
    rng = np.random.default_rng(seed)
    shift = n if pre_only else n // 2
    out = []

    # ---------- phase 1: locality-heavy + sequential scans ----------
    hot = list(range(hot_size))
    w = 1.0 / np.arange(1, hot_size + 1) ** 0.8
    w /= w.sum()
    scan_ptr, scan_left = 0, 0
    while len(out) < shift:
        if scan_left > 0:                       # inside a sequential scan
            out.append(100 + scan_ptr % 200)
            scan_ptr += 1
            scan_left -= 1
        elif rng.random() < 0.01:               # start a new scan
            scan_left = int(rng.integers(20, 60))
        else:
            out.append(int(rng.choice(hot, p=w)))
        if len(out) % 500 == 0:                 # slow drift of the hot set
            hot[int(rng.integers(0, hot_size))] = int(rng.integers(20, 100))

    # ---------- phase 2: random / bursty ----------
    while len(out) < n:
        if rng.random() < 0.9:                  # uniform random access
            out.append(int(rng.integers(0, universe)))
        else:                                   # burst on a few pages
            burst = rng.integers(0, universe, size=int(rng.integers(3, 7)))
            for _ in range(int(rng.integers(4, 10))):
                out.append(int(rng.choice(burst)))
    return out[:n], shift
