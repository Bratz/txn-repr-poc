"""Generate docs/SCIENCE_PAPER.pdf - the science behind each layer.

Regenerate with:  python docs/build_science_paper.py
A layer-by-layer account of the txn-repr-poc replication of arXiv:2410.07851.
All math is ASCII (Phi/psi/phi, alpha, ->, <<) so the built-in PDF fonts render it.
"""
import math
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.platypus import (
    HRFlowable, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
)

OUT = Path(__file__).resolve().parent / "SCIENCE_PAPER.pdf"

INK = colors.HexColor("#1d1d1f")
MUTED = colors.HexColor("#5f6168")
ACCENT = colors.HexColor("#2c6ecb")
CODE_BG = colors.HexColor("#f4f5f7")
EQ_BG = colors.HexColor("#f7f8fa")
LINE = colors.HexColor("#d9d9dd")
HEAD_BG = colors.HexColor("#eef2f8")

ss = getSampleStyleSheet()
TITLE = ParagraphStyle("TITLE", parent=ss["Title"], fontSize=18, leading=22,
                       textColor=INK, alignment=TA_CENTER, spaceAfter=4)
AUTH = ParagraphStyle("AUTH", parent=ss["Normal"], fontSize=9.5, leading=13,
                      textColor=MUTED, alignment=TA_CENTER, spaceAfter=2)
ABS = ParagraphStyle("ABS", parent=ss["Normal"], fontSize=9, leading=13,
                     textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6,
                     leftIndent=10, rightIndent=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, leading=15,
                    textColor=ACCENT, spaceBefore=12, spaceAfter=4)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10.5, leading=13,
                    textColor=INK, spaceBefore=7, spaceAfter=2)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=9.5, leading=13.8,
                      textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6)
EQ = ParagraphStyle("EQ", parent=ss["Normal"], fontName="Helvetica-Oblique",
                    fontSize=9.5, leading=14, textColor=INK, alignment=TA_CENTER,
                    spaceAfter=2, spaceBefore=2)
REF = ParagraphStyle("REF", parent=ss["Normal"], fontSize=8.3, leading=11,
                     textColor=INK, alignment=TA_LEFT, leftIndent=12,
                     firstLineIndent=-12, spaceAfter=3)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.5, leading=11,
                      alignment=TA_LEFT, spaceAfter=0)
CELLH = ParagraphStyle("CELLH", parent=CELL, textColor=colors.white,
                       fontName="Helvetica-Bold")

story = []


def h2(t): story.append(Paragraph(t, H2))
def h3(t): story.append(Paragraph(t, H3))
def body(t): story.append(Paragraph(t, BODY))
def ref(t): story.append(Paragraph(t, REF))


def eq(t):
    p = Paragraph(t, EQ)
    tbl = Table([[p]], colWidths=[168 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), EQ_BG),
        ("BOX", (0, 0), (-1, -1), 0.4, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl); story.append(Spacer(1, 5))


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
    story.append(t); story.append(Spacer(1, 7))


def design_drawing():
    d = Drawing(460, 322)
    ac_f, ac_s = colors.HexColor("#dce8f7"), colors.HexColor("#2c6ecb")
    fr_f, fr_s = colors.HexColor("#e7e8ea"), colors.HexColor("#9a9aa0")
    pl_f, pl_s = colors.white, colors.HexColor("#c4c4c8")
    grey = colors.HexColor("#6b6b70")

    def txt(x, y, s, size=8, bold=False, col=INK, anchor="middle"):
        d.add(String(x, y, s, fontName="Helvetica-Bold" if bold else "Helvetica",
                     fontSize=size, fillColor=col, textAnchor=anchor))

    def cbox(x, y, w, h, kind, title, sub=None, tsize=8):
        f, s = {"a": (ac_f, ac_s), "f": (fr_f, fr_s), "p": (pl_f, pl_s)}[kind]
        d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=f, strokeColor=s, strokeWidth=1))
        cx = x + w / 2
        if sub:
            txt(cx, y + h / 2 + 2, title, tsize, True)
            txt(cx, y + h / 2 - 8, sub, 6, col=MUTED)
        else:
            txt(cx, y + h / 2 - 3, title, tsize, True)

    def arrow(x1, y1, x2, y2, dash=False):
        ln = Line(x1, y1, x2, y2, strokeColor=grey, strokeWidth=1.2)
        if dash:
            ln.strokeDashArray = [3, 2]
        d.add(ln)
        ang = math.atan2(y2 - y1, x2 - x1)
        bx, by = x2 - 5.5 * math.cos(ang), y2 - 5.5 * math.sin(ang)
        px, py = -math.sin(ang) * 2.6, math.cos(ang) * 2.6
        d.add(Polygon([x2, y2, bx + px, by + py, bx - px, by - py],
                      fillColor=grey, strokeColor=grey))

    cbox(0, 288, 64, 28, "p", "Payment", "pacs.008")
    arrow(64, 302, 78, 302)
    cbox(78, 288, 64, 28, "p", "Projection", "to row")
    arrow(142, 302, 156, 302)
    cbox(156, 288, 138, 28, "f", "Transaction encoder", "BERT 25M . frozen")
    arrow(294, 302, 308, 302)
    cbox(308, 288, 50, 28, "f", "f(x)", "embedding")

    d.add(Rect(0, 16, 460, 248, rx=6, ry=6, fillColor=colors.white,
               strokeColor=colors.HexColor("#d9d9dd"), strokeWidth=1))
    txt(8, 250, "Layer 4 - Decoder", 8, True, ACCENT, anchor="start")
    cbox(10, 208, 56, 32, "a", "phi prompt", "soft", 7.5)
    cbox(72, 208, 34, 32, "a", "[R1]", "mark", 7.5)
    cbox(112, 208, 84, 32, "a", "Phi( f(x) )", "payment -> token", 7.5)
    cbox(202, 208, 116, 32, "p", "instruction", "classify risk ...", 7.5)
    cbox(324, 208, 56, 32, "a", "psi task", "which Q", 7.5)
    txt(196, 196, "one interleaved input (Eq. 5)", 6.5, col=MUTED)
    arrow(330, 288, 154, 242, dash=True)
    arrow(196, 193, 196, 154)
    cbox(88, 118, 216, 36, "f", "Phi-1.5", "frozen . ~1.3B params . fp32", 10)
    arrow(196, 118, 196, 86)
    cbox(88, 52, 216, 34, "p", "next word -> A / B / C", "= the task's label")
    cbox(320, 118, 134, 36, "a", "Trains: Phi . psi . phi", "~7.64M (<1% full tune)", 7.3)
    cbox(320, 74, 134, 36, "f", "Frozen: encoder f, Phi-1.5", None, 7.3)
    d.add(Rect(150, 26, 10, 10, fillColor=ac_f, strokeColor=ac_s, strokeWidth=1))
    txt(166, 28, "Trained (small)", 7, col=MUTED, anchor="start")
    d.add(Rect(262, 26, 10, 10, fillColor=fr_f, strokeColor=fr_s, strokeWidth=1))
    txt(278, 28, "Frozen", 7, col=MUTED, anchor="start")
    return d


# ============================================================== TITLE + ABSTRACT
story.append(Paragraph("The Science of a Transaction Representation Model, Layer by Layer",
                       TITLE))
story.append(Paragraph(
    "A faithful replication of Raman, Ganesh and Veloso, <i>Scalable Representation "
    "Learning for Multimodal Tabular Transactions</i> (arXiv:2410.07851, NeurIPS 2024), "
    "on synthetic ISO 20022 (pacs.008) payments.", AUTH))
story.append(Spacer(1, 8))
story.append(Paragraph("<b>Abstract.</b> "
    "This paper documents the science of each layer of a faithful replication of the "
    "multimodal tabular-transaction method of arXiv:2410.07851. A power-law partitioned "
    "embedding encodes roughly 120,000 account identifiers in 5.8% of the parameters a "
    "dense table needs, with no loss of reconstruction accuracy (Layer 2, Sec. 3.1). A "
    "magnitude-adaptive, currency-conditioned quantizer turns amounts into tokens (3.3). "
    "An offline party encoder amortizes counterparty representation into a persistent "
    "store, so inference is a lookup rather than a forward pass (3.2). A bidirectional "
    "encoder is trained with a joint masked-reconstruction and batch-hard-triplet "
    "objective and then frozen (3.4). A frozen-encoder, frozen-LLM decoder learns four "
    "downstream tasks through about 7.64 million adapter parameters - 0.59% of a full "
    "fine-tune (Sec. 4). We state the two falsifiable claims the replication was built to "
    "test and report both, including the honest negative: on a synthetic, rule-derived "
    "risk label the adapter is extremely parameter-efficient but loses to a gradient-boosted "
    "tree, for reasons we trace to the data, not the architecture.", ABS))
story.append(HRFlowable(width="100%", thickness=0.6, color=LINE, spaceBefore=6, spaceAfter=4))

# ============================================================== 1. INTRODUCTION
h2("1. The problem, and the two planes")
body("A payment is a wide, mixed-type record: high-cardinality identifiers (accounts, "
     "parents), a numeric amount spanning several orders of magnitude, categorical "
     "context, and counterparty attributes. Three properties make it hard to model. The "
     "identifiers have vocabularies in the hundreds of thousands. The interesting labels "
     "are rare - in our data the High-risk class is 1.8% of rows. And the data is "
     "regulated, so the representation has to be learned in a way that is auditable and "
     "cheap to serve.")
body("The method answers this by splitting along a frozen-versus-trained boundary. An "
     "offline plane learns the expensive things once - the partitioned identifier space, "
     "the counterparty store, the encoder - and freezes them. An online plane scores a new "
     "payment with a store lookup and two frozen forward passes plus a small trained head. "
     "Everything below describes one stage of that pipeline, the science it rests on, and "
     "the design choices the paper left open. The deliverable of this replication is "
     "fidelity to the method, not improved metrics; where we depart, we say so.")

# ============================================================== 2. LAYER 1
h2("2. Layer 1 - the input contract")
body("Layer 1 is deterministic data engineering with no paper anchor: it parses each "
     "pacs.008 credit-transfer and projects it into a fixed-width row "
     "x = (x<sub>1</sub>, ..., x<sub>C</sub>) over C typed columns, then writes a bucket "
     "manifest (column_schema.json) that every later stage reads. Each column is typed into "
     "one of four buckets - high-cardinality categorical, numerical, core, and meta-party - "
     "and the bucket decides which Layer-2 encoder the column flows through. The contract "
     "matters more than it looks: pinning it to the schema, rather than to hard-coded "
     "column lists, is what lets a schema change reach every downstream module at once.")

# ============================================================== 3. LAYER 2.1
h2("3. Layer 2 - field encoders")
h3("3.1  Partitioning embedder (paper Sec. 3.1)")
body("A classical embedding stores one D-dimensional row per token, E in "
     "R<super>|V| x D</super>. For account identifiers with |V| above 10<super>5</super> "
     "this table dominates the model's parameters, and most of it is barely trained - "
     "transaction identifiers follow a steep power law, so the long tail of rare accounts "
     "each appears a handful of times. Spending a full D-dimensional vector on a token seen "
     "three times is waste.")
body("The fix is to size capacity to frequency. The vocabulary is split into B bins and "
     "the embedding dimension into B matching slices, each by a power law (Eq. 2 and its "
     "dimension analogue):")
eq("|V<super>b</super>| = |V| &middot; b<super>-a_v</super> / "
   "( sum<sub>j=1..B</sub> j<super>-a_v</super> ) &nbsp;&nbsp;&nbsp; "
   "D<super>b</super> = D &middot; b<super>-a_d</super> / "
   "( sum<sub>j=1..B</sub> j<super>-a_d</super> )")
body("With the paper's exponents a_v = -3 and a_d = 2.25, the negative a_v makes later "
     "bins hold far more tokens (|V<super>1</super>| &lt;&lt; ... &lt;&lt; |V<super>B</super>|) "
     "while the positive a_d makes earlier bins far wider (D<super>1</super> &gt; ... &gt; "
     "D<super>B</super>). So the few frequent tokens land in a small, high-dimensional bin; "
     "the many rare tokens share a large, low-dimensional bin. The shared output space is a "
     "direct sum, R<super>D</super> = R<super>D^1</super> (+) ... (+) R<super>D^B</super>: "
     "a token assigned to bin b receives its D<super>b</super>-dimensional row placed into "
     "bin b's contiguous coordinate slice, and the other coordinates stay zero. There is no "
     "up-projection back to D - the partitioned subspaces are kept. The net parameter count "
     "is sum<sub>b</sub> |V<super>b</super>| &middot; D<super>b</super>, far below |V| "
     "&middot; D.")
body("The paper does not state how tokens are assigned to bins; we assign by frequency "
     "rank, the only reading consistent with the power-law-on-frequency motivation, and "
     "flag the choice in the code. The claim under test (C1) is that this matches a dense "
     "table on reconstruction while using far fewer parameters. It holds: at a realized "
     "account vocabulary of 119,819, the partitioned tables use a 0.058 fraction of the "
     "dense parameter count and the masked-column reconstruction top-1 accuracy is "
     "0.012 points <i>better</i> than the dense control, not worse.")

h3("3.2  Adaptive numerical quantization (Sec. 3.3)")
body("Amounts are continuous and heavy-tailed. A uniform grid wastes most of its levels on "
     "the sparse high end and starves the dense low end of resolution. The method maps a "
     "numeric column to a custom vocabulary Q = {Q<sub>1</sub>, ..., Q<sub>m</sub>} that "
     "adapts to magnitude - finer spacing for small numbers, coarser for large - and "
     "assigns a value to its nearest level, argmin<sub>i</sub> |x - Q<sub>i</sub>|, which is "
     "then embedded like any categorical token. The paper leaves the spacing law and the "
     "level count open. We use geometric (log-spaced) levels, which give absolute gaps that "
     "grow with magnitude - exactly the stated behaviour, keyed to magnitude rather than to "
     "data density as a quantile grid would be - and m = 128 levels.")
body("One forced departure lives here. Settlement amounts arrive in many currencies on "
     "wildly different scales (JPY versus USD), so a single global grid mis-quantizes "
     "multi-currency data. We build one grid per currency. The level-index vocabulary stays "
     "shared at size m, so level i means the i-th magnitude band <i>for its own currency</i>, "
     "and two currencies' level i share an embedding row; currency identity rides separately "
     "as a core column rather than being folded into the numeric token. This is a "
     "correctness fix, not an enhancement.")

h3("3.3  Offline party encoder and the persistent store (Sec. 3.2)")
body("Counterparty attributes are meta-columns: contextual, slowly changing, and shared "
     "across many transactions. The method encodes them offline with a separate function "
     "Xi_g : x_g -> R<super>C_g x D</super> whose output occupies a single element of the "
     "column sequence, with C_g &lt; |x_g| so the result is a compact pooled summary rather "
     "than a flattened block. The paper deliberately leaves the encoder, the pooling, and "
     "the training objective unspecified.")
body("We encode the structured trio {Ctry, Industry, SubIndustry} with a small transformer "
     "over per-attribute embeddings plus a learned summary (CLS-style) token, and train it "
     "by masked-attribute reconstruction - mask one attribute, reconstruct it from the rest "
     "- which keeps the objective methodologically consistent with the Layer-3 loss. Party "
     "names are treated as identity, not as learnable attributes, because free-text names "
     "are high-cardinality identifiers already served by the Sec. 3.1 embedder through the "
     "account keys. The summaries are written once into a persistent store keyed by account "
     "id. At inference this store is a lookup, not a forward pass - the amortization that "
     "makes the online plane cheap, and the one place a careful reviewer will push, which is "
     "why the objective choice is documented rather than hidden.")

h3("3.4  Column assembler (Eq. 4)")
body("The assembler routes each column to the encoder its bucket names and concatenates the "
     "results into the encoder's input sequence:")
eq("embedding(x) = ( Xi(x<sub>1</sub>), ..., Xi_g(x_g), ..., Xi(x<sub>C</sub>) )")
body("High-cardinality columns get one partitioning embedder each - debtor and creditor "
     "accounts are kept separate, not merged into a shared vocabulary, which is the literal "
     "reading of Sec. 3.1. The party block is replaced by its pooled summary token, a frozen "
     "store lookup. Routing is driven entirely by the schema.")

# ============================================================== 4. LAYER 3
h2("4. Layer 3 - the tabular encoder (Sec. 3.4)")
body("A bidirectional transformer in the standard BERT configuration runs over the assembled "
     "column sequence; the output at a prepended [CLS] position is the row embedding f(x). "
     "Training uses a composite objective that adds, to masked-column reconstruction, a "
     "batch-hard triplet loss:")
eq("L = L<sub>recon</sub> + lambda &middot; L<sub>triplet</sub> &nbsp;&nbsp;(lambda = 1)")
body("Reconstruction masks a subset of the reconstructable columns and predicts them with "
     "per-column heads (the frozen party-summary tokens serve as context and are not "
     "reconstructed). The triplet term is the part the paper defends in its ablation, and "
     "it is the claim under test, so it must not be swapped for a generic contrastive loss. "
     "We form two independently masked views of each row as a positive pair, and apply the "
     "batch-hard rule of Hermans et al. (2017): for every anchor, take the hardest positive "
     "(the farthest view of the same row) and the hardest negative (the nearest other row), "
     "and push them apart by a margin:")
eq("L<sub>triplet</sub> = mean over anchors of "
   "relu( margin + d(anchor, hardest_pos) - d(anchor, hardest_neg) )")
body("The reconstruction loss teaches the encoder what each column says; the triplet loss "
     "shapes the geometry, pulling two corrupted views of one transaction together while "
     "pushing different transactions apart, so that distance in embedding space means "
     "something the per-column heads alone never enforce. The backbone is sized to the "
     "paper's pinned 25M parameters and trained for 3 epochs, the ablation sweet spot, then "
     "frozen. From here on f is a fixed feature extractor.")

# ============================================================== 5. LAYER 4
h2("5. Layer 4 - the multimodal decoder (Sec. 4 / 4.1)")
body("Layer 4 reads the frozen embedding with a frozen language model and learns the "
     "downstream tasks through three small trainable modules. The figure shows the runtime "
     "data flow.")
story.append(design_drawing())
story.append(Spacer(1, 8))
body("The frozen tabular encoder f produces one embedding per payment. The frozen LLM is a "
     "Phi-1.5-class model. Only three modules train. The adapter Phi is a small transformer "
     "that projects f(x) into the LLM's token-embedding space, emitting one soft token per "
     "record - this is what lets a frozen language model read a non-text object. The task "
     "embedding psi is a map Xi_task : 1..K -> R<super>D</super> built as "
     "concat( unique[k], shared ): a subspace unique to each task and a subspace shared "
     "across all tasks. The prompt parameters phi augment the model the way prompt-tuning "
     "does - either a per-layer learnable prefix (Li and Liang, 2021) or an input-level soft "
     "prompt, the robust default on a real HF model. The inputs to the LLM interleave a row "
     "sentinel, the payment token, the instruction text, and the task signal (Eq. 5):")
eq("z<sub>i</sub> = Xi_LLM(s(1)) (+) Phi(f(x<sub>i1</sub>)) (+) ... (+) "
   "Xi_LLM(t<sub>i</sub>) (+) Xi_task(k<sub>i</sub>)")
body("The row sentinels s(.) = [R1], [R2], ... uniquely identify records. Training is a "
     "single cross-entropy objective over the answer, supervised only at the response "
     "position (Eq. 6):")
eq("L = - sum<sub>i</sub> log P( y<sub>i</sub> | z<sub>i</sub> ; Phi, psi, phi )")
body("Single-record tasks pass one payment; the recurrence task passes several of an "
     "entity's payments together, repeating the sentinel-and-token block before the "
     "instruction, because recurrence is a pattern across records rather than a property of "
     "one. The whole trainable trio is about 7.64 million parameters against a 1.3-billion "
     "frozen LLM - 0.59%. Keeping f and the LLM frozen is both the deployability argument "
     "and the integrity of the experiment: unfreeze either and the headline numbers describe "
     "a different model.")

# ============================================================== 6. LAYER 5
h2("6. Layer 5 - baselines and falsifiable evaluation")
body("The baseline is CatBoost trained on the raw, flattened row - no representation "
     "learning, the floor the adapter must clear. The recurrence task is excluded from the "
     "CatBoost comparison, following the paper (Sec. 5.3): it passes multiple transactions "
     "and 'is not suitable for non-sequential classifiers'. Because the risk label is "
     "imbalanced, accuracy is misleading by construction, so the primary metrics are a "
     "forced departure: PR-AUC, recall at a fixed 1% false-positive rate, and F1 at the "
     "operating point; accuracy is reported alongside only for comparability to the paper's "
     "balanced-task tables.")
body("The replication is organized around two falsifiable claims with thresholds set before "
     "the run. C1: the partitioning embedder matches dense reconstruction while using "
     "substantially fewer embedding parameters. C2: the frozen-encoder, frozen-LLM adapter "
     "beats CatBoost and rivals a full fine-tune at a fraction of the trainable parameters. "
     "A run passes only if the measured number clears its pre-registered threshold.")

# ============================================================== 7. RESULTS
h2("7. Results, stated honestly")
table(["Claim / metric", "Measured", "Threshold", "Verdict"], [
    ["C1 embedding param ratio (partitioned / dense)", "0.058", "&lt;= 0.55", "pass"],
    ["C1 top-1 reconstruction gap (dense - partitioned)", "-0.012 pp", "&lt;= 1.0 pp", "pass"],
    ["C2 trainable param ratio (adapter / full tune)", "0.0059 (7.64M / 1.3B)", "&lt;= 0.10", "pass"],
    ["C2 risk PR-AUC, adapter vs CatBoost", "0.21 vs 0.66", "gain &gt;= +10 pp", "fail (-44.5 pp)"],
], [78, 42, 28, 30])
body("C1 is confirmed at near-paper scale and is the headline replication: the partitioned "
     "embedder reaches the same reconstruction quality as a dense table at 5.8% of the "
     "embedding parameters. C2 is half-confirmed. The adapter is extremely parameter-"
     "efficient - it trains under 1% of a full fine-tune - but on the risk task it loses to "
     "CatBoost, and not narrowly.")
body("The loss is a property of the data, not the architecture, and the distinction is the "
     "point. Our synthetic risk label is a transparent rule over cross-border status, "
     "currency, region, industry, and amount - the exact fields CatBoost reads off the row - "
     "so the tree recovers the rule. The adapter has to carry the same signal through a "
     "frozen encoder compressed to a single token and a head trained for one epoch. There is "
     "a general argument underneath this: for a single-record label that is a function of "
     "visible fields, CatBoost has strictly more direct access to those fields than a frozen, "
     "compressed embedding does, so the adapter can at best tie. A representation earns its "
     "advantage only where the flattened tabular view loses information - across records, "
     "and over time.")

# ============================================================== 8. WHERE IT POINTS
h2("8. Limitations and where the science points")
body("The result is honest about its own ceiling. On single-record, full-information tasks a "
     "gradient-boosted tree is near-optimal, so a representation model cannot win there. Its "
     "advantage lives in structure the tree cannot see: the timing between an entity's "
     "payments, a dormant account reactivating, a pattern that only reads as fraud across a "
     "history. The repository already carries the multi-record machinery for that case "
     "(recurrence, Eq. 5); what it does not carry is real transactions over time. That is "
     "the same direction the production transaction foundation models have taken - Visa's "
     "TransactionGPT (arXiv:2511.08939) and Revolut's PRAGMA (arXiv:2604.08649) both model "
     "behaviour as a sequence over time. The concrete next steps are temporal-sequence "
     "modelling over entity histories (TabBERT, FATA-Trans, CoLES) and a held-out-entity "
     "split that tests generalization rather than memorization.")

# ============================================================== 9. DEVIATIONS
h2("9. Deviations ledger")
body("Three departures from the paper are sanctioned and visible throughout: the pacs.008 "
     "column schema in place of the paper's generic synthetic columns; currency-conditioned "
     "quantization, a correctness fix for multi-currency amounts; and imbalance-aware metrics "
     "in place of accuracy, forced by the rare positive class. Three extensions are "
     "deliberately walked back and excluded: a data-completeness feature vector, a "
     "structuring/layering chain task, and a held-out-typology split. The recurrence task is "
     "a paper task and is in scope - it should not be confused with the walked-back chain "
     "task.")

# ============================================================== REFERENCES
h2("References")
ref("Raman, B., Ganesh, S., Veloso, M. (2024). Scalable Representation Learning for "
    "Multimodal Tabular Transactions. arXiv:2410.07851. NeurIPS 2024 Table Representation "
    "Learning workshop.")
ref("Hermans, A., Beyer, L., Leibe, B. (2017). In Defense of the Triplet Loss for Person "
    "Re-Identification. arXiv:1703.07737.")
ref("Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). BERT: Pre-training of Deep "
    "Bidirectional Transformers for Language Understanding. NAACL-HLT.")
ref("Li, X. L., Liang, P. (2021). Prefix-Tuning: Optimizing Continuous Prompts for "
    "Generation. ACL. arXiv:2101.00190.")
ref("Li, Y., et al. (2023). Textbooks Are All You Need II: phi-1.5 Technical Report. "
    "arXiv:2309.05463.")
ref("Prokhorenkova, L., et al. (2018). CatBoost: Unbiased Boosting with Categorical "
    "Features. NeurIPS.")
ref("Padhi, I., et al. (2021). Tabular Transformers for Modeling Multivariate Time Series "
    "(TabBERT). ICASSP. arXiv:2011.01843.")
ref("Babaev, D., et al. (2022). CoLES: Contrastive Learning for Event Sequences with "
    "Self-Supervision. SIGMOD. arXiv:2002.08232.")
ref("Zhang, D., et al. (2023). FATA-Trans: Field And Time-Aware Transformer for Sequential "
    "Tabular Data. CIKM. arXiv:2310.13818.")
ref("Visa Research (2025). TransactionGPT. arXiv:2511.08939.")
ref("Revolut (2026). PRAGMA: Revolut Foundation Model. arXiv:2604.08649.")


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 11 * mm, "txn-repr-poc - the science, layer by layer")
    canvas.drawRightString(A4[0] - 20 * mm, 11 * mm, str(doc.page))
    canvas.restoreState()


SimpleDocTemplate(
    str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
    topMargin=18 * mm, bottomMargin=18 * mm,
    title="The Science of a Transaction Representation Model, Layer by Layer",
    author="txn-repr-poc",
).build(story, onFirstPage=_footer, onLaterPages=_footer)
print(f"wrote {OUT}")
