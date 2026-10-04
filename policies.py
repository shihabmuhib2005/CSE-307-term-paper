"""Classical page replacement policies (FIFO, LRU, Optimal) + simulator."""
import numpy as np


class Policy:
    name = "base"

    def reset(self, trace, frames):
        pass

    def observe(self, t, page):          # called on every access
        pass

    def on_load(self, t, page):          # page brought into memory
        pass

    def victim(self, t, cache):          # choose page to evict
        raise NotImplementedError


class FIFO(Policy):
    name = "FIFO"

    def reset(self, trace, frames):
        self.loaded = {}

    def on_load(self, t, page):
        self.loaded[page] = t

    def victim(self, t, cache):
        return min(cache, key=self.loaded.__getitem__)


class LRU(Policy):
    name = "LRU"

    def reset(self, trace, frames):
        self.last = {}

    def observe(self, t, page):
        self.last[page] = t

    def victim(self, t, cache):
        return min(cache, key=self.last.__getitem__)


class Optimal(Policy):
    """Belady's MIN: evict the page whose next use is farthest in the future."""
    name = "Optimal"

    def reset(self, trace, frames):
        n = len(trace)
        self.nxt = [0] * n
        seen = {}
        for t in range(n - 1, -1, -1):
            self.nxt[t] = seen.get(trace[t], float("inf"))
            seen[trace[t]] = t
        self.nu = {}

    def observe(self, t, page):
        self.nu[page] = self.nxt[t]

    def victim(self, t, cache):
        return max(cache, key=self.nu.__getitem__)


def simulate(trace, frames, policy):
    """Run the trace; return boolean numpy array hits[t]."""
    policy.reset(trace, frames)
    cache = set()
    hits = np.zeros(len(trace), dtype=bool)
    for t, p in enumerate(trace):
        policy.observe(t, p)
        if p in cache:
            hits[t] = True
            continue
        if len(cache) >= frames:
            cache.remove(policy.victim(t, cache))
        cache.add(p)
        policy.on_load(t, p)
    return hits
