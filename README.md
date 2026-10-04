# Learned Page Replacement under Workload Shift (CSE-307, Section B, Track 1)

Term paper project for **CSE-307: Operating Systems**.
Track chosen: **Track 1 - Learned Page Replacement (Memory Management)**, with the optional bonus (explanation + confidence).

**Name:** Md Muhibuzzaman Khan  
**ID:** 202414085  
**Course:** CSE-307 (Operating Systems)

## What this project does
1. Implements **FIFO, LRU and Optimal (Belady)** page replacement (`src/policies.py`).
2. Adds a **learned eviction policy** (`src/learned.py`): a decision tree / logistic regression that looks at
   recency, access frequency (last 200 accesses), total access count and residency age of each cached page and
   predicts whether the page is *safe to evict* (not reused within the next 64 accesses). Variants:
   - `Learned-DT (static)` and `Learned-LR (static)`: trained offline on a trace with only the *pre-shift* pattern
   - `Learned-DT (online)`: starts as LRU, then is retrained every 100 evictions on recent eviction decisions whose outcome is already known (the adaptive layer)
   - `Learned-DT (mixed-trained)`: ablation, trained on a trace that already contains the shift
3. Generates a **synthetic trace with a deliberate shift** (`src/trace.py`): first half = locality-heavy (small drifting
   hot set + sequential scans), second half = mostly uniform random with short bursts.
4. Measures hit ratio and page faults **before and after the shift** (10 random seeds, 32 frames, 10,000 accesses each).
5. Bonus: rule-based explanation text for each eviction with a self-rated confidence, compared with whether the decision was correct.

## Setup
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to run
```bash
python tests/test_policies.py                 # sanity checks (textbook example: FIFO 15, LRU 12, OPT 9 faults)
python src/run_experiments.py                 # full experiment (~1-2 min), writes to results/
python src/run_experiments.py --seeds 3 --frames 16   # optional: quick / different setting
python report/make_report.py                  # rebuilds report/report.pdf from results/
```

## Results (mean of 10 seeds, 5,000 accesses per phase)
| Policy | Hit ratio before | Hit ratio after | Drop (pp) | Faults before | Faults after |
|---|---|---|---|---|---|
| FIFO | 0.619 | 0.282 | 33.7 | 1907 | 3592 |
| LRU | 0.622 | 0.282 | 34.0 | 1891 | 3592 |
| Optimal | 0.736 | 0.510 | 22.6 | 1321 | 2451 |
| Learned-DT (static) | 0.719 | 0.255 | 46.4 | 1407 | 3726 |
| Learned-LR (static) | 0.724 | 0.198 | 52.6 | 1382 | 4010 |
| Learned-DT (online) | 0.701 | 0.263 | 43.8 | 1495 | 3685 |
| Learned-DT (mixed-trained) | 0.719 | 0.220 | 49.9 | 1405 | 3902 |

Summary: before the shift the learned policies beat LRU/FIFO by about 10 points and come close to Optimal; after the
shift they degrade the most (LR worst), and the online retraining recovers only a little. Full discussion is in the report.
Charts and tables are in `results/` (`fig1_rolling_hit_ratio.png`, `fig2_before_after.png`, `fig3_confidence_calibration.png`,
`summary.csv`, `bonus_confidence.csv`, `explanations_sample.txt`, `raw_results.csv`).

## Repository layout
```
src/policies.py         FIFO, LRU, Optimal + simulator
src/learned.py          learned / adaptive eviction policy
src/trace.py            synthetic trace generator with shift
src/run_experiments.py  experiments, tables, charts, bonus
tests/test_policies.py  correctness checks
report/                 report.pdf and make_report.py
results/                charts and tables
```

## Notes / limitations
- Traces are synthetic; numbers depend on the chosen workload, 32 frames and a 64-access "reuse" horizon.
- Hit ratio before the shift includes cold-start misses.
- In the bonus part, "correct" means the evicted page was not reused within 64 accesses; this is easier to satisfy in the random phase, so high accuracy there does not mean a high hit ratio.

## AI assistance disclosure
An AI assistant (Claude) was used for implementation help, including drafting the code in this repository
and the first draft of this README and the report. As required by the brief, the experimental design,
results and analysis should be reviewed, understood and owned by the student before submission.
