"""Generate docs/TXN_REPR_PAPER.pdf - an arXiv-style research paper for the POC.

Regenerate with:  python docs/build_paper.py
Every number is the measured one from the H200 full runs (results_*.json,
RESULTS.md, docs/V2_DIRECTION.md). No number in this paper is aspirational.
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

OUT = Path(__file__).resolve().parent / "TXN_REPR_PAPER.pdf"

INK = colors.HexColor("#1d1d1f")
MUTED = colors.HexColor("#5f6168")
ACCENT = colors.HexColor("#1E2761")
LINE = colors.HexColor("#d9d9dd")
HEAD_BG = colors.HexColor("#eef2f8")

ss = getSampleStyleSheet()
TITLE = ParagraphStyle("TITLE", parent=ss["Title"], fontSize=17, leading=21,
                       textColor=INK, alignment=TA_CENTER, spaceAfter=6)
AUTH = ParagraphStyle("AUTH", parent=ss["Normal"], fontSize=9.5, leading=13,
                      textColor=MUTED, alignment=TA_CENTER, spaceAfter=2)
ABS = ParagraphStyle("ABS", parent=ss["Normal"], fontSize=9, leading=13,
                     textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6,
                     leftIndent=14, rightIndent=14)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, leading=15,
                    textColor=ACCENT, spaceBefore=12, spaceAfter=4)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10.5, leading=13,
                    textColor=INK, spaceBefore=7, spaceAfter=2)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=9.5, leading=13.8,
                      textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6)
REF = ParagraphStyle("REF", parent=ss["Normal"], fontSize=8.3, leading=11,
                     textColor=INK, alignment=TA_LEFT, leftIndent=12,
                     firstLineIndent=-12, spaceAfter=3)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.5, leading=11,
                      alignment=TA_LEFT, spaceAfter=0)
CELLH = ParagraphStyle("CELLH", parent=CELL, textColor=colors.white,
                       fontName="Helvetica-Bold")
CAP = ParagraphStyle("CAP", parent=ss["Normal"], fontSize=8.3, leading=11,
                     textColor=MUTED, alignment=TA_JUSTIFY, spaceAfter=8)

story = []


def h2(t): story.append(Paragraph(t, H2))
def h3(t): story.append(Paragraph(t, H3))
def body(t): story.append(Paragraph(t, BODY))
def ref(t): story.append(Paragraph(t, REF))
def cap(t): story.append(Paragraph(t, CAP))


def table(headers, rows, widths):
    data = [[Paragraph(h, CELLH) for h in headers]]
    for r in rows:
        data.append([Paragraph(c, CELL) for c in r])
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HEAD_BG]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))


# ====================================================== TITLE + ABSTRACT
story.append(Paragraph(
    "PULSE: Payment Understanding, Lifecycle Scoring &amp; Embeddings<br/>"
    "A Transaction Foundation Model on Synthetic ISO 20022 Rails -<br/>"
    "a Falsifiable Replication and Extension of arXiv:2410.07851", TITLE))
story.append(Paragraph("Subrato B. &nbsp;·&nbsp; subrato.b@outlook.com", AUTH))
story.append(Paragraph(
    "Preprint · July 2026 · code and generators: github.com/Bratz/txn-repr-poc", AUTH))
story.append(Spacer(1, 8))
story.append(HRFlowable(width="100%", thickness=0.6, color=LINE))
story.append(Spacer(1, 6))
story.append(Paragraph(
    "<b>Abstract.</b> "
    "We present PULSE, a transaction foundation model built by replicating the architecture "
    "of Raman et al. (arXiv:2410.07851) on synthetic ISO 20022 payments and measuring, "
    "against thresholds fixed before each run, "
    "what a single frozen tabular encoder can and cannot predict about a payment. Five "
    "claims were pre-registered. Two replicate the source paper: the partitioned "
    "high-cardinality embedder matches dense masked-column reconstruction at 5.8% of the "
    "embedding parameters (C1), and the frozen-encoder-plus-adapter stack trains 0.59% of "
    "a full fine-tune's parameters but loses to CatBoost by 44.5 PR-AUC points on a "
    "rule-generated risk label (C2) - trees read rule labels directly; the encoder "
    "bottleneck cannot. Three claims concern sequences. A small history encoder over frozen "
    "per-row embeddings beats an order-blind pooled baseline by 38.5 PR-AUC points (C3) and "
    "CatBoost on hand-built aggregates by 41.3 points (C4) on a regime-change task whose "
    "order-invariant statistics are matched across classes. Feeding the same history "
    "representation through a frozen Phi-1.5 with 7.6M trainable adapter parameters buys "
    "1.3 points over a linear probe (C5) - under our 2-point bar, so serving drops the LLM. "
    "We then extend the corpus to a nine-message ISO 20022 lifecycle (pain.001 through "
    "camt.056/pacs.004) and report, wins and nulls alike: settlement-time regression beats "
    "its naive baseline at initiation (MAE 172 vs 234 minutes), in-flight outcome scoring "
    "sharpens from 0.86 to 0.94 PR-AUC as messages arrive, and fusing pain.001 origination "
    "context lifts account-takeover detection from chance (0.03) to 1.00; message-order "
    "signal is a measured null (-0.5 points), and cancel/return heads sit at prevalence. "
    "All data is synthetic and seeded; 197 tests reproduce every number.", ABS))
story.append(Spacer(1, 4))
story.append(HRFlowable(width="100%", thickness=0.6, color=LINE))

# ====================================================== 1 INTRODUCTION
h2("1&nbsp;&nbsp;Introduction")
body("A payments engine asks many questions about the same row. Which rail should carry "
     "this transfer? Will it settle straight through, and if not, which exception fires? "
     "When does the beneficiary see funds? Is this account suddenly bursting with activity "
     "it never showed before? Production systems typically answer each question with its "
     "own model, its own features, and its own retraining calendar. Raman et al. [1] "
     "propose the alternative this paper tests: pretrain one tabular encoder on the "
     "payment corpus, freeze it, and answer every downstream question with a small adapter "
     "or probe on the frozen representation.")
body("Replications in this space tend to fail quietly - a retuned learning rate here, a "
     "relabeled dataset there, and the headline number survives while the claim it "
     "supported dies. We took the opposite discipline. Every architectural choice is "
     "pinned to the source paper's section that mandates it, thresholds live in a config "
     "file committed before the runs, and a miss stays a miss: no post-hoc retuning "
     "closed any gap reported here. The result is a ledger rather than a victory lap. "
     "Some claims replicate strongly. One fails against our own bar, and the failure is "
     "informative. One resolves a design fork - whether the frozen LLM earns its "
     "inference cost - with a number instead of an opinion.")
body("Our contributions: (i) an independent replication of the source paper's "
     "parameter-efficiency claims at near-paper vocabulary scale (119,819 accounts) on "
     "fully synthetic data; (ii) three pre-registered sequence claims measured on a corpus "
     "whose order-invariant statistics are matched across classes, isolating temporal "
     "signal by construction; (iii) a measured answer to the frozen-LLM question - 1.3 "
     "PR-AUC points for 7.6M adapter parameters plus a 1.3B-parameter forward pass; "
     "(iv) an extension of the single-table setting to a nine-message ISO 20022 lifecycle "
     "with event-anchored timestamps, recall and return legs, and a second data source "
     "(pain.001 origination context) whose fusion we evaluate; and (v) a capability-routing "
     "discipline that names which questions the model refuses to answer because a rule, a "
     "lookup, or a template answers them exactly.")

# ====================================================== 2 RELATED WORK
h2("2&nbsp;&nbsp;Related work")
body("Transaction representation learning has moved from sequence models over card streams "
     "- TabBERT [7], CoLES [8], FATA-Trans [9] - toward foundation-model framings in which "
     "one pretrained backbone serves many tasks: TransactionGPT [10] at Visa and PRAGMA "
     "[11] at Revolut are the production-scale statements of that position. Raman et al. "
     "[1], our source, contribute the tabular side: columns as tokens, a partitioned "
     "embedder for high-cardinality identifiers, an offline party encoder, and a frozen "
     "LLM decoder adapted with small trainable modules in the prefix-tuning family [4]. "
     "The C2 finding we report - gradient-boosted trees [6] winning on labels that are "
     "rules over raw features - repeats a decade of tabular-benchmark results and is why "
     "we treat CatBoost as the bar every claim must clear, not as a strawman. TabPFN [12] "
     "offers a different foundation-model bet (in-context learning over small tables); we "
     "assessed it as out of scope because our tables exceed its row and class budgets and "
     "our deployment needs a persistent, incrementally probed representation.")

# ====================================================== 3 DATA
h2("3&nbsp;&nbsp;Synthetic corpora")
body("No real transaction data appears anywhere in this work. Three generators, all "
     "seeded and committed, produce the corpora; identical commands reproduce "
     "byte-identical files on any machine.")
h3("3.1&nbsp;&nbsp;pacs.008 corpus (claims C1, C2)")
body("One million interbank credit transfers over 19,895 corporate parents and a combined "
     "account vocabulary of 119,819 - close to the source paper's ~125K. The risk label is "
     "deliberately a transparent rule over raw fields (cross-border flag, currency, "
     "region, industry, amount bands), with the High class at 1.8% prevalence. That "
     "choice is honest hostility: a label a tree can read directly is the hardest case "
     "for an encoder that must compress the row into one vector before any head sees it.")
h3("3.2&nbsp;&nbsp;Behavioural corpus (claims C3-C5)")
body("Per-account payment histories, 8 to 40 events each, where 45% of accounts undergo a "
     "regime change midway through their stream. The generator matches the gap multiset "
     "and the amount distribution across the two classes, so every order-invariant "
     "aggregate - means, quantiles, counts, totals - carries no class information. Only a "
     "model that reads the ordered, timed sequence can separate the classes. This "
     "construction is what lets us attribute C3 and C4 to temporal signal rather than to "
     "feature leakage. The full corpus yields 24,016 sequences; evaluation holds out 20% "
     "of accounts (4,803 sequences, positive prevalence 0.450), never rows within a "
     "training account.")
h3("3.3&nbsp;&nbsp;India-rails lifecycle corpus (Section 6)")
body("Twenty thousand payments across RTGS, NEFT, IMPS, and SWIFT, each expanded into its "
     "ISO 20022 message trail: pain.001 acceptance, pain.002 status, pacs.008 clearing "
     "(pacs.009 cover where correspondent banking requires it), pacs.002 as the gpi "
     "tracker backbone, camt.054 credit notification, camt.056/camt.029 recall legs, and "
     "pacs.004 returns. Message timestamps anchor to generated exception events rather "
     "than to interpolation, and each message carries a direction and a visibility flag - "
     "an inward payment's pain.* legs live at the remote bank and never enter our "
     "prefixes, which is the information set a real engine has. A separate origination "
     "table attaches channel, device, session, and authentication attributes to each "
     "outward pain.001 at t=0 (10,943 rows), simulating the data a bank's own channel "
     "layer possesses before any ISO message exists. A cadence-enabled variant of the "
     "same generator (salary, rent, utility, supplier schedules) supports the forecasting "
     "heads.")

# ====================================================== 4 ARCHITECTURE
h2("4&nbsp;&nbsp;Architecture")
body("The pinned encoder follows the source paper: each column embeds as one token "
     "(25M parameters total), high-cardinality identifiers route through the partitioned "
     "embedder of their Section 3.1, party identities come from an offline-pretrained "
     "persistent store, and amounts pass through a currency-conditioned adaptive "
     "quantizer. Pretraining combines masked-column reconstruction with a batch-hard "
     "triplet objective [2], three epochs, then the encoder freezes for good. Nothing "
     "downstream ever updates it.")
body("For sequences, a small transformer (the history encoder) consumes the frozen "
     "per-row embeddings of an account's ordered payments together with time deltas and "
     "calendar features, pretrained by reconstructing held-out field values, and emits "
     "one history vector per prefix. Downstream heads then take one of two forms. "
     "Option A is a linear probe on the history vector. Option B routes the same vector "
     "into a frozen Phi-1.5 [5] through trainable adapters in the prefix family [3, 4] - "
     "7,638,528 trainable parameters against the LLM's 1.3B frozen ones. C5 is the "
     "measured comparison between the two.")
body("Not every question reaches the model. Rail caps and floors, assigned reference "
     "values, identifier lookups, and duplicate checks are answered by guards and "
     "templates that are exact by construction; we delisted those tasks from the model's "
     "advertised capabilities after finding early prototypes claiming credit for "
     "predictions a two-line rule makes with certainty. The serving layer routes each "
     "field to model, rule, lookup, or template, and the documentation names which is "
     "which. A model that answers everything answers some things worse than the "
     "infrastructure it runs on.")

# ====================================================== 5 CLAIMS
h2("5&nbsp;&nbsp;Pre-registered claims")
body("Table 1 states each claim, its threshold as committed in configuration before the "
     "run, and the measured outcome. Full-scale numbers come from single seeded runs on "
     "one NVIDIA H200; smoke-scale runs on CPU reproduce every qualitative verdict.")
table(
    ["Claim", "Measured", "Threshold", "Verdict"],
    [
        ["<b>C1</b> partitioned embedder: parameter ratio vs dense, at matched "
         "reconstruction", "<b>0.058</b>; recon gap -0.012 pp (partitioned marginally "
         "better)", "ratio &lt;= 0.55; gap &lt;= 1 pp", "<b>pass</b>"],
        ["<b>C2a</b> adapter parameter efficiency", "<b>0.0059</b> of full fine-tune "
         "(7.64M / 1.3B)", "&lt;= 0.10", "<b>pass</b>"],
        ["<b>C2b</b> adapter beats CatBoost on rule-label risk", "PR-AUC 0.210 vs "
         "<b>0.655</b> (-44.5 pp)", "&gt;= +10 pp", "<b>fail</b>"],
        ["<b>C3</b> temporal lift: history encoder vs order-blind pooling",
         "0.919 vs 0.533 (<b>+38.5 pp</b>)", "&gt;= +10 pp", "<b>pass</b>"],
        ["<b>C4</b> held-out generalisation vs CatBoost on aggregates",
         "0.919 vs 0.506 (<b>+41.3 pp</b>)", "&gt;= +5 pp", "<b>pass</b>"],
        ["<b>C5</b> LLM necessity: frozen Phi-1.5 + adapters vs linear probe",
         "0.932 vs 0.919 (<b>+1.3 pp</b> for the LLM)",
         "drop the LLM if within 2 pp", "<b>drop the LLM</b>"],
    ],
    [52, 52, 32, 26])
cap("Table 1: the five claims. C1/C2 on the 1M-row pacs.008 corpus (positive class 1.8%); "
    "C3-C5 on the behavioural corpus, held-out accounts, PR-AUC on regime change. "
    "Thresholds predate the runs; no configuration was retuned after a miss.")
h3("5.1&nbsp;&nbsp;What replicated")
body("C1 is the source paper's core engineering claim and it replicates cleanly: the "
     "partitioned embedder holds reconstruction quality while spending 5.8% of the "
     "dense embedding budget, at a vocabulary within a few percent of the paper's. C3 "
     "and C4 are the strongest results in this report. On data where every "
     "order-invariant aggregate is uninformative by construction, CatBoost sits near "
     "chance (0.506) while the history encoder reaches 0.919 on unseen accounts - the "
     "first configuration in this project where the learned representation beats the "
     "tree, and it does so because the signal is genuinely temporal. A companion "
     "velocity task (does this account's next window contain an activity burst, "
     "prevalence 0.131) shows the same shape: time-aware 0.822 against order-blind "
     "0.222.")
h3("5.2&nbsp;&nbsp;What failed, and why we kept it")
body("C2b failed against our own bar. The adapter learns real signal - 0.210 PR-AUC is "
     "roughly ten times the prevalence floor - but CatBoost reads the label-generating "
     "rule straight off the raw features, and no amount of frozen-representation "
     "elegance competes with direct access to the rule. We could have made this number "
     "respectable by softening the label. We left it, because it draws the deployment "
     "boundary honestly: on complete rows with rule-like labels, ship the tree. The "
     "encoder earns its keep where trees cannot follow - order, timing, partial "
     "information, and fusion across sources - which is precisely where Sections 5.1 "
     "and 6 place it.")
h3("5.3&nbsp;&nbsp;The LLM verdict")
body("C5 asks whether the frozen LLM belongs in the serving path. On regime "
     "classification from the same frozen history vector, Option B (Phi-1.5 plus "
     "adapters) reaches 0.932 against the probe's 0.919. The gap is real and it is "
     "small: 1.3 points, under the 2-point bar we set in advance, purchased with 7.6M "
     "trained parameters and a 1.3B-parameter forward pass per prediction. An earlier "
     "mock-LLM control had suggested the LLM subtracts 13 points; the real model "
     "corrects the sign but not the decision. Serving therefore uses linear probes and "
     "isotonic calibration on the frozen representations, and nothing in the deployed "
     "path executes an LLM. We note the untested cases plainly: tasks that emit text, "
     "few-shot task switching, and instruction-conditioned multi-task heads are the "
     "settings where the LLM could still earn its cost, and none of them is measured "
     "here.")

# ====================================================== 6 LIFECYCLE RESULTS
h2("6&nbsp;&nbsp;The lifecycle extension: wins and nulls")
body("The India-rails corpus turns the single-table problem into an engine's problem: "
     "score at intake, rescore as each message lands, and answer questions no single "
     "message contains. Table 2 reports every prediction the system exposes, including "
     "the ones that lose. Splits hold out accounts; in-flight prefixes exclude the "
     "messages that state the outcome being predicted.")
table(
    ["Prediction", "Measured", "Baseline", "Reading"],
    [
        ["Settlement ETA at initiation (pain.001 only)", "MAE <b>172 min</b>",
         "234 (naive)", "win, pre-clearing"],
        ["Settlement ETA at intake (full row, n=4,000 eval)", "MAE <b>195 min</b>",
         "392 (naive)", "win"],
        ["Charges", "MAE <b>117</b>", "203 (naive)", "win"],
        ["'Will it credit?' as messages arrive", "PR-AUC 0.860 -&gt; <b>0.942</b> "
         "(prefix 1 -&gt; 3)", "0.856 (prevalence)", "win, sharpens in-flight"],
        ["Account takeover at origination", "0.029 ISO-only -&gt; <b>1.000</b> fused "
         "with pain.001 channel context", "ISO fields alone", "win, multi-source"],
        ["Velocity burst (dense fleet)", "PR-AUC <b>0.822</b>", "0.222 order-blind",
         "win, timing-only"],
        ["Next payment: days until (held-out customers)", "MAE <b>23.5 d</b>",
         "25.9 / 32.5 (median / last gap)", "win, modest"],
        ["Next payment: occurrence within 35 d", "PR-AUC 0.579", "tree 0.655, "
         "rule 0.637", "loss - trees win"],
        ["Next payment: amount", "MAE 33.0k", "21.8k (history median)", "loss"],
        ["Rail routing", "acc 0.556", "tree 0.645, majority 0.478",
         "loss - C2 territory (SWIFT F1 = 1.00)"],
        ["Terminal status at intake", "acc 0.279", "majority 0.740", "loss - C2 "
         "territory"],
        ["Reject reason (ISO code)", "acc 0.102", "'none' dominates", "loss"],
        ["Cancel / return likelihood at intake", "PR-AUC at prevalence "
         "(0.030 / 0.018)", "-", "null - event-driven, not intent-visible"],
        ["Message-order signal (tracker outcome)", "sequence macro-F1 0.270 vs pooled "
         "0.275 (<b>-0.5 pp</b>)", "bag of messages", "null - prefixes are 2-3 "
         "near-fixed-order messages"],
    ],
    [50, 46, 36, 36])
cap("Table 2: every exposed prediction on the lifecycle corpus, H200 full runs. "
    "The losses and nulls are load-bearing: they route those fields to trees, rules, "
    "or nothing.")
body("Two rows deserve prose. The account-takeover result is the multi-source argument "
     "in one line: ISO clearing fields carry no trace of a hijacked session (0.029, "
     "chance at 2.7% prevalence), while the same frozen encoder reading the pain.001 "
     "origination context alongside them separates the classes completely - on "
     "synthetic data whose generator plants exactly that correlation, so the number "
     "demonstrates the fusion path works, not that real fraud is this easy. The "
     "message-order null is the opposite lesson. A single payment's pre-outcome trail "
     "is two or three messages in near-fixed order; there is nothing for a sequence "
     "model to read that a bag of messages does not already say, and the measured "
     "-0.5 points says exactly that. Temporal signal in payments lives at the entity "
     "level (C3, velocity), not inside one payment's message chain.")

# ====================================================== 7 SERVING
h2("7&nbsp;&nbsp;Serving")
body("The deployed artifact is the frozen encoder, the frozen history encoder, and a "
     "set of linear heads with per-head isotonic calibration fitted on held-out "
     "prefixes - roughly 260MB, CPU-servable. Scoring is stateless: the engine sends a "
     "payment row, or a message prefix, or an account's recent history, and receives "
     "calibrated probabilities plus a predicted-lifecycle object (settlement outcome "
     "distribution, remaining ETA, cancel and return probabilities). Uncalibrated "
     "heads had produced scores like 0.94 for a 3%-prevalence event; calibration "
     "restored readability (the same head now emits 0.19). Explanations come from "
     "column occlusion against the frozen encoder - occlude a field, re-embed, measure "
     "the shift - which is faithful to the model by construction and costs about "
     "twenty embeddings per explained row. Two operational caveats are measured rather "
     "than suspected: brand-new accounts degrade routing confidence to near-uniform "
     "(identity features carry most of the rail signal), and a bundle trained under "
     "scikit-learn 1.9 requires a one-line attribute shim to load under 1.7.")

# ====================================================== 8 LIMITATIONS
h2("8&nbsp;&nbsp;Limitations")
body("Everything here is synthetic, and the generators were written by the same team "
     "that wrote the models. We defended against self-deception where we knew how - "
     "matched aggregates for C3/C4, prevalence-honest PR-AUC, held-out actors, "
     "visibility-filtered prefixes, thresholds committed in advance - but a generator "
     "cannot invent the correlations of real payment traffic, and the ATO fusion "
     "number in particular measures a planted signal. Each full-scale number is one "
     "seeded run; we report no variance across seeds. C5 is measured on one task "
     "family with one 1.3B LLM, and says nothing about text-emitting or few-shot "
     "settings. The rare-event heads (cancel at 3.0%, return at 1.8% of 20,000 "
     "payments) sit at prevalence and would need two orders of magnitude more data "
     "before failure means anything. The published v1 checkpoint predates the "
     "lifecycle work and scores only the original task suite. Real transaction data, "
     "multiple seeds, and an entity-level generator whose successive payments "
     "correlate are the three shortest paths to making these numbers mean more.")

# ====================================================== 9 CONCLUSION
h2("9&nbsp;&nbsp;Conclusion")
body("PULSE's frozen encoder does carry many payment decisions - provided one is honest "
     "about which ones. Where the signal is a rule over a complete row, a "
     "gradient-boosted tree wins and should be shipped instead (C2b, routing, status). "
     "Where the signal is order, timing, partial information, or a second source the "
     "tree never sees, the frozen representation wins by margins of 38 to 60 PR-AUC "
     "points (C3, C4, velocity, in-flight, fusion). The frozen LLM, the most "
     "expensive component the source architecture allows, contributes 1.3 points and "
     "was removed from serving (C5). The claims ledger, generators, and 197 tests "
     "reproduce every number in this paper from two commands.")

# ====================================================== REFERENCES
h2("References")
ref("[1] Raman, B., Ganesh, S., Veloso, M. (2024). Scalable Representation Learning "
    "for Multimodal Tabular Transactions. arXiv:2410.07851. NeurIPS 2024 Table "
    "Representation Learning workshop.")
ref("[2] Hermans, A., Beyer, L., Leibe, B. (2017). In Defense of the Triplet Loss for "
    "Person Re-Identification. arXiv:1703.07737.")
ref("[3] Li, X. L., Liang, P. (2021). Prefix-Tuning: Optimizing Continuous Prompts "
    "for Generation. ACL. arXiv:2101.00190.")
ref("[4] Lester, B., Al-Rfou, R., Constant, N. (2021). The Power of Scale for "
    "Parameter-Efficient Prompt Tuning. EMNLP. arXiv:2104.08691.")
ref("[5] Li, Y., et al. (2023). Textbooks Are All You Need II: phi-1.5 Technical "
    "Report. arXiv:2309.05463.")
ref("[6] Prokhorenkova, L., et al. (2018). CatBoost: Unbiased Boosting with "
    "Categorical Features. NeurIPS.")
ref("[7] Padhi, I., et al. (2021). Tabular Transformers for Modeling Multivariate "
    "Time Series (TabBERT). ICASSP. arXiv:2011.01843.")
ref("[8] Babaev, D., et al. (2022). CoLES: Contrastive Learning for Event Sequences "
    "with Self-Supervision. SIGMOD. arXiv:2002.08232.")
ref("[9] Zhang, D., et al. (2023). FATA-Trans: Field And Time-Aware Transformer for "
    "Sequential Tabular Data. CIKM. arXiv:2310.13818.")
ref("[10] Visa Research (2025). TransactionGPT. arXiv:2511.08939.")
ref("[11] Revolut (2026). PRAGMA: Revolut Foundation Model. arXiv:2604.08649.")
ref("[12] Hollmann, N., et al. (2025). Accurate Predictions on Small Data with a "
    "Tabular Foundation Model (TabPFN). Nature 637.")
ref("[13] Zadrozny, B., Elkan, C. (2002). Transforming Classifier Scores into "
    "Accurate Multiclass Probability Estimates. KDD.")


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 11 * mm, "PULSE (txn-repr-poc) - preprint")
    canvas.drawRightString(A4[0] - 20 * mm, 11 * mm, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        leftMargin=20 * mm, rightMargin=20 * mm,
                        topMargin=18 * mm, bottomMargin=18 * mm,
                        title="PULSE: Payment Understanding, Lifecycle Scoring & Embeddings",
                        author="txn-repr-poc")
doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
print(f"wrote {OUT}")
