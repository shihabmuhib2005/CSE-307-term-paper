"""Builds report/report.pdf (2-3 pages) from the files in results/."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, "..", "results")
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.3, leading=12, alignment=4, spaceAfter=4)
h = ParagraphStyle("h", parent=ss["Heading2"], fontSize=11, spaceBefore=6, spaceAfter=3)
title = ParagraphStyle("t", parent=ss["Title"], fontSize=15, spaceAfter=2)
sub = ParagraphStyle("s", parent=ss["Normal"], fontSize=9, alignment=1, spaceAfter=6)
cap = ParagraphStyle("c", parent=ss["Normal"], fontSize=8, alignment=1, textColor=colors.HexColor("#444444"))

S = []
P = lambda t, st=body: S.append(Paragraph(t, st))

P("Learned Page Replacement under Workload Shift", title)
P("Course: CSE-307 Operating Systems - Term Paper (Section B), Track 1<br/>Name: Md Muhibuzzaman Khan &nbsp;|&nbsp; ID: 202414085", sub)

P("1. Problem framing", h)
P("FIFO and LRU assume the access pattern stays roughly the same. Real programs change phase: a loop over a small "
  "working set can be followed by a scan or by almost random accesses. I wanted to see how much three classical policies "
  "(FIFO, LRU, Belady's Optimal) and a small learned policy suffer when the pattern changes in the middle of a run. "
  "The questions were: (a) can a lightweight model beat LRU on a stable pattern, (b) what happens to it after the shift, "
  "and (c) does retraining it online help.")

P("2. Implementation summary", h)
P("All code is in Python (<i>src/</i>). The simulator walks through the trace with a fixed number of frames (32) and counts a hit "
  "or a page fault for each access. FIFO evicts the oldest loaded page, LRU the page with the oldest last access, and Optimal "
  "the page whose next use is farthest in the future (checked against the textbook string 7,0,1,2,0,3,0,4,2,3,0,3,2,1,2,0,1,7,0,1 "
  "with 3 frames: 15 / 12 / 9 faults).")
P("<b>Learned policy.</b> Whenever a fault happens with a full memory, each resident page gets four features: time since last access, "
  "number of accesses in the last 200 references, total accesses so far, and time since it was loaded. A classifier predicts "
  "whether the page is <i>not</i> reused in the next 64 accesses (label 1 = good to evict, labels come from the future of the trace "
  "in hindsight, in the spirit of Belady). The page with the highest probability is evicted. I tried a depth-4 decision tree and "
  "logistic regression trained offline on a pre-shift trace (<i>static</i>), a decision tree trained on a trace that already "
  "contains the shift (<i>mixed-trained</i>, used as an ablation), and a decision tree that starts as LRU and is retrained every "
  "100 evictions on recent decisions whose outcome is already known (<i>online</i>).")

P("3. Experimental setup", h)
P("Each trace has 10,000 accesses over 300 pages. In the first 5,000 accesses about 70% of references go to a small hot set of 20 pages "
  "(Zipf-like, slowly drifting) and about 30% are sequential scans (20-60 pages long) through a cold region. In the last 5,000 accesses "
  "90% of events are single uniform random accesses over all 300 pages and 10% are short bursts (4-9 accesses on 3-6 pages). Models were trained on separate "
  "traces (different seeds). Everything was repeated for 10 seeds; I report the mean and the standard deviation. Hit ratio and fault "
  "counts are computed separately for the two halves (5,000 accesses each). The hit ratio before the shift includes the cold-start misses.")

P("4. Results", h)
rows = [["Policy", "Hit ratio before", "Hit ratio after", "Drop (pp)", "Faults before", "Faults after"]]
import csv
for r in csv.DictReader(open(os.path.join(R, "summary.csv"))):
    rows.append([r["Policy"], r["Hit ratio before"], r["Hit ratio after"], r["Drop (pp)"], r["Faults before"], r["Faults after"]])
t = Table(rows, colWidths=[4.3*cm, 2.9*cm, 2.9*cm, 1.7*cm, 2.6*cm, 2.6*cm])
t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.6), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde4ee")),
                       ("GRID", (0, 0), (-1, -1), .3, colors.grey), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                       ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
S.append(KeepTogether([t, Paragraph("Table 1: hit ratio and page faults per 5,000 accesses (mean ± std over 10 seeds, 32 frames).", cap)]))
S.append(Spacer(1, 4))
S.append(KeepTogether([Image(os.path.join(R, "fig1_rolling_hit_ratio.png"), width=14.5*cm, height=7.25*cm),
                       Paragraph("Figure 1: rolling hit ratio (250 accesses), averaged over seeds. FIFO and LRU lie on top of each other.", cap)]))

P("5. Analysis: what changed under the shift", h)
P("<b>FIFO and LRU became the same.</b> After the workload shift, the performance of FIFO and LRU became equal "
  "(hit ratio 0.282, about 3,592 faults for both). The accesses became random, so the recently used pages no longer "
  "tell which page will be needed next, and LRU has no advantage over FIFO.")
P("<b>The learned models got worse.</b> The models were trained on the old workload, but the access pattern changed. "
  "Therefore, their predictions became less accurate than LRU. After the shift, static DT dropped to 0.255 and static LR "
  "to 0.198, both below LRU (0.282), so the learned policies degraded the most (drops of 46.4 and 52.6 points "
  "against 34.0 for LRU).")
P("<b>Online retraining helped only a little.</b> Retraining allowed the model to learn the new workload and the hit ratio "
  "improved to 0.263 (static DT: 0.255). However, in the random pattern pages are rarely reused soon, so there is little "
  "pattern to learn and the improvement was limited. It stayed below LRU (0.282).")

P("6. Bonus: explanation confidence", h)
P("For each eviction a template generates a sentence such as <i>'evicting page 8; last used 71 accesses ago, touched once in the last 200 accesses; "
  "expect no reuse within 64 accesses; confidence 0.98'</i> (samples in <i>results/explanations_sample.txt</i>). The confidence is the model's probability "
  "for the victim, and a decision counts as correct if the page is really not reused within 64 accesses. For the static DT, confidence is high "
  "before the shift (0.99 when correct, accuracy 98%) but after the shift it is about 0.58 and is not higher on correct decisions (0.579 correct vs 0.592 wrong), "
  "so it is poorly calibrated under shift (ECE 0.23). The online tree is better calibrated (ECE 0.09, 0.93 vs 0.92 mean confidence correct/wrong), though the "
  "gap between correct and wrong is small. Note that 'correct' is an easy condition in the random half, so high accuracy there does not mean a high hit ratio.")

P("7. Limitations", h)
P("The traces are synthetic and the results depend on the chosen mix, the number of frames and the 64-access horizon. The features are simple on purpose; "
  "richer features (for example scan detection) or a change detector would probably help. Code, charts and tables: see the GitHub repository (README).")

doc = SimpleDocTemplate(os.path.join(HERE, "report.pdf"), pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm,
                        topMargin=1.5*cm, bottomMargin=1.4*cm, title="Learned Page Replacement under Workload Shift")
doc.build(S)
print("report.pdf written")
