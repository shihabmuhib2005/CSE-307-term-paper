"""Sanity checks. Run:  python tests/test_policies.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from policies import FIFO, LRU, Optimal, simulate

# Classic textbook reference string, 3 frames
REF = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]


def faults(policy):
    return int((~simulate(REF, 3, policy)).sum())


def test_textbook():
    assert faults(FIFO()) == 15
    assert faults(LRU()) == 12
    assert faults(Optimal()) == 9


def test_optimal_is_lower_bound():
    from trace import make_trace
    tr, _ = make_trace(n=3000, seed=5)
    f = {p.name: (~simulate(tr, 32, p)).sum() for p in (FIFO(), LRU(), Optimal())}
    assert f["Optimal"] <= f["LRU"] and f["Optimal"] <= f["FIFO"]


if __name__ == "__main__":
    test_textbook()
    test_optimal_is_lower_bound()
    print("all tests passed")
