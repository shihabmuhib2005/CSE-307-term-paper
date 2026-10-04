"""Run all policies on shifted traces, save tables + charts to ../results.

Usage:  python src/run_experiments.py [--seeds 10] [--frames 32] [--n 10000]
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from trace import make_trace
from policies import FIFO, LRU, Optimal, simulate
from learned import LearnedPolicy, train_static, label_rows, T_HORIZON

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
ORDER = ["FIFO", "LRU", "Optimal", "Learned-DT (static)", "Learned-LR (static)",
         "Learned-DT (online)", "Learned-DT (mixed-trained)"]


def run(seeds, frames, n):
    rows, rolling, decs = [], {k: [] for k in ORDER}, []
    for s in range(seeds):
        trace, shift = make_trace(n=n, seed=s)
        pre, _ = make_trace(n=n // 2, seed=1000 + s, pre_only=True)
        mixed, _ = make_trace(n=n, seed=2000 + s)
        policies = [
            FIFO(), LRU(), Optimal(),
            LearnedPolicy(train_static("dt", pre, frames), label="Learned-DT (static)"),
            LearnedPolicy(train_static("lr", pre, frames), label="Learned-LR (static)"),
            LearnedPolicy(None, online=True, label="Learned-DT (online)"),
            LearnedPolicy(train_static("dt", mixed, frames), label="Learned-DT (mixed-trained)"),
        ]
        for p in policies:
            hits = simulate(trace, frames, p)
            for phase, sl in (("before", slice(0, shift)), ("after", slice(shift, n))):
                h = hits[sl]
                rows.append(dict(seed=s, policy=p.name, phase=phase, accesses=len(h),
                                 faults=int((~h).sum()), hit_ratio=float(h.mean())))
            rolling[p.name].append(np.convolve(hits, np.ones(250) / 250, mode="valid"))
            if isinstance(p, LearnedPolicy) and p.name in ("Learned-DT (static)", "Learned-DT (online)"):
                for t, v, conf, _ in p.decisions:
                    if conf is None:
                        continue
                    y = label_rows(p.times, t, [v])[0]
                    decs.append(dict(seed=s, policy=p.name, t=t, phase="before" if t < shift else "after",
                                     confidence=conf, correct=int(y)))
        print(f"seed {s} done", flush=True)
    return pd.DataFrame(rows), rolling, pd.DataFrame(decs)


def summarise(df):
    g = df.groupby(["policy", "phase"]).agg(hit_mean=("hit_ratio", "mean"), hit_std=("hit_ratio", "std"),
                                            faults_mean=("faults", "mean"), faults_std=("faults", "std")).reset_index()
    out = []
    for pol in ORDER:
        b = g[(g.policy == pol) & (g.phase == "before")].iloc[0]
        a = g[(g.policy == pol) & (g.phase == "after")].iloc[0]
        out.append({"Policy": pol,
                    "Hit ratio before": f"{b.hit_mean:.3f} ± {b.hit_std:.3f}",
                    "Hit ratio after": f"{a.hit_mean:.3f} ± {a.hit_std:.3f}",
                    "Drop (pp)": round(100 * (b.hit_mean - a.hit_mean), 1),
                    "Faults before": f"{b.faults_mean:.0f} ± {b.faults_std:.0f}",
                    "Faults after": f"{a.faults_mean:.0f} ± {a.faults_std:.0f}",
                    "_b": b.hit_mean, "_a": a.hit_mean})
    return pd.DataFrame(out)


def plots(df, rolling, shift, summ):
    colors = {"FIFO": "#7f7f7f", "LRU": "#1f77b4", "Optimal": "#2ca02c", "Learned-DT (static)": "#d62728",
              "Learned-LR (static)": "#ff7f0e", "Learned-DT (online)": "#9467bd",
              "Learned-DT (mixed-trained)": "#8c564b"}
    plt.figure(figsize=(9, 4.5))
    for k in ORDER:
        plt.plot(np.mean(rolling[k], axis=0), label=k, color=colors[k], lw=1.5)
    plt.axvline(shift - 125, color="k", ls="--", lw=1)
    plt.text(shift - 100, 0.02, "workload shift", fontsize=8)
    plt.xlabel("access index (250-access rolling window)")
    plt.ylabel("hit ratio (mean over seeds)")
    plt.title("Hit ratio over time: locality-heavy -> random/bursty")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig1_rolling_hit_ratio.png"), dpi=160)
    plt.close()

    x = np.arange(len(ORDER))
    plt.figure(figsize=(9, 4.5))
    plt.bar(x - .2, summ["_b"], .4, label="before shift", color="#4c72b0")
    plt.bar(x + .2, summ["_a"], .4, label="after shift", color="#dd8452")
    plt.xticks(x, [o.replace(" (", "\n(") for o in ORDER], fontsize=7)
    plt.ylabel("hit ratio")
    plt.title("Hit ratio before vs after the shift")
    plt.legend()
    plt.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig2_before_after.png"), dpi=160)
    plt.close()


def bonus(decs):
    """Confidence vs correctness of the learned eviction decisions."""
    rows = []
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for pol, c in (("Learned-DT (static)", "#d62728"), ("Learned-DT (online)", "#9467bd")):
        d = decs[decs.policy == pol]
        bins = np.linspace(0, 1, 6)
        d = d.assign(b=np.clip(np.digitize(d.confidence, bins) - 1, 0, 4))
        cal = d.groupby("b").agg(conf=("confidence", "mean"), acc=("correct", "mean"), n=("correct", "size"))
        ax.plot(cal.conf, cal.acc, "o-", color=c, label=pol)
        ece = float((cal.n * (cal.conf - cal.acc).abs()).sum() / cal.n.sum())
        for ph in ("before", "after", "all"):
            dd = d if ph == "all" else d[d.phase == ph]
            rows.append({"policy": pol, "phase": ph, "decisions": len(dd), "accuracy": round(dd.correct.mean(), 3),
                         "mean_conf_correct": round(dd[dd.correct == 1].confidence.mean(), 3),
                         "mean_conf_wrong": round(dd[dd.correct == 0].confidence.mean(), 3) if (dd.correct == 0).any() else np.nan,
                         "ECE(all)": round(ece, 3) if ph == "all" else ""})
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("fraction of evictions that were correct")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_confidence_calibration.png"), dpi=160)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "bonus_confidence.csv"), index=False)
    return pd.DataFrame(rows)


def explanation_sample(frames, n, seed=0):
    """Rule-based natural-language explanation for each decision (bonus)."""
    trace, shift = make_trace(n=n, seed=seed)
    pre, _ = make_trace(n=n // 2, seed=1000 + seed, pre_only=True)
    p = LearnedPolicy(train_static("dt", pre, frames), label="Learned-DT (static)")
    simulate(trace, frames, p)
    lines = []
    for t, v, conf, x in [d for d in p.decisions if d[2] is not None][:5] + \
                         [d for d in p.decisions if d[2] is not None and d[0] >= shift][:5]:
        idle = int(round(np.expm1(x[0])))
        lines.append(f"t={t}: cache full, evicting page {v}. It was last used {idle} accesses ago and "
                     f"was touched {int(x[1])} times in the last 200 accesses, so I expect it not to be needed "
                     f"in the next {T_HORIZON} accesses. Self-rated confidence: {conf:.2f}.")
    open(os.path.join(OUT, "explanations_sample.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df, rolling, decs = run(a.seeds, a.frames, a.n)
    df.to_csv(os.path.join(OUT, "raw_results.csv"), index=False)
    summ = summarise(df)
    summ.drop(columns=["_b", "_a"]).to_csv(os.path.join(OUT, "summary.csv"), index=False)
    open(os.path.join(OUT, "summary.md"), "w").write(summ.drop(columns=["_b", "_a"]).to_markdown(index=False))
    plots(df, rolling, a.n // 2, summ)
    print(summ.drop(columns=["_b", "_a"]).to_string(index=False))
    print(bonus(decs).to_string(index=False))
    explanation_sample(a.frames, a.n)
