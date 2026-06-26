"""Generate docs/DEVELOPER_GUIDE.pdf - maintainer's guide for txn-repr-poc.

Regenerate with:  python docs/build_developer_guide.py
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
)
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
import math

OUT = Path(__file__).resolve().parent / "DEVELOPER_GUIDE.pdf"

INK = colors.HexColor("#1d1d1f")
MUTED = colors.HexColor("#5f6168")
ACCENT = colors.HexColor("#2c6ecb")
CODE_BG = colors.HexColor("#f4f5f7")
LINE = colors.HexColor("#d9d9dd")
HEAD_BG = colors.HexColor("#eef2f8")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=20, leading=24, textColor=INK,
                    spaceAfter=2, alignment=TA_LEFT)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=10, leading=13.5,
                     textColor=MUTED, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, leading=16,
                    textColor=ACCENT, spaceBefore=13, spaceAfter=5)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=10, leading=14.5,
                      textColor=INK, spaceAfter=7)
CODE = ParagraphStyle("CODE", parent=ss["Code"], fontName="Courier", fontSize=8.6,
                      leading=11.5, textColor=INK)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=9, leading=12, spaceAfter=0)
CELLH = ParagraphStyle("CELLH", parent=CELL, textColor=colors.white,
                       fontName="Helvetica-Bold")
CELLC = ParagraphStyle("CELLC", parent=CELL, fontName="Courier", fontSize=8.3, leading=11)

story = []
mono0 = False


def h2(t): story.append(Paragraph(t, H2))
def body(t): story.append(Paragraph(t, BODY))


def code(t):
    inner = Preformatted(t, CODE)
    tbl = Table([[inner]], colWidths=[170 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl); story.append(Spacer(1, 6))


def table(headers, rows, widths):
    data = [[Paragraph(h, CELLH) for h in headers]]
    for r in rows:
        data.append([Paragraph(c, CELLC if (j == 0 and mono0) else CELL)
                     for j, c in enumerate(r)])
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HEAD_BG]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t); story.append(Spacer(1, 8))


def design_drawing():
    """Native reportlab redraw of docs/architecture-phi.svg (runtime data flow)."""
    d = Drawing(470, 332)
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

    # top pipeline: payment -> projection -> frozen encoder -> f(x)
    cbox(0, 292, 66, 30, "p", "Payment", "pacs.008")
    arrow(66, 307, 80, 307)
    cbox(80, 292, 66, 30, "p", "Projection", "to row")
    arrow(146, 307, 160, 307)
    cbox(160, 292, 140, 30, "f", "Transaction encoder", "BERT 25M . frozen")
    arrow(300, 307, 314, 307)
    cbox(314, 292, 52, 30, "f", "f(x)", "embedding")

    # Layer 4 panel
    d.add(Rect(0, 18, 470, 250, rx=6, ry=6, fillColor=colors.white,
               strokeColor=colors.HexColor("#d9d9dd"), strokeWidth=1))
    txt(8, 254, "Layer 4 - Decoder", 8, True, ACCENT, anchor="start")

    # interleaved input (Eq. 5)
    cbox(10, 212, 58, 34, "a", "phi prompt", "soft", 7.5)
    cbox(74, 212, 36, 34, "a", "[R1]", "mark", 7.5)
    cbox(116, 212, 86, 34, "a", "Phi( f(x) )", "payment -> 1 token", 7.5)
    cbox(208, 212, 120, 34, "p", "instruction", "classify the risk ...", 7.5)
    cbox(334, 212, 58, 34, "a", "psi task", "which Q", 7.5)
    txt(201, 200, "one interleaved input (Eq. 5)", 6.5, col=MUTED)

    arrow(338, 292, 160, 247, dash=True)   # f(x) feeds the adapter token
    arrow(201, 197, 201, 158)              # input -> Phi
    cbox(90, 120, 220, 38, "f", "Phi-1.5", "frozen . ~1.3B params . fp32", 10)
    arrow(200, 120, 200, 88)               # Phi -> output
    cbox(90, 52, 220, 36, "p", "next word -> A / B / C", "= the task's label")

    cbox(326, 120, 138, 38, "a", "Trains: Phi . psi . phi", "~7.64M (<1% of full tune)", 7.5)
    cbox(326, 74, 138, 38, "f", "Frozen: encoder f, Phi-1.5", None, 7.5)

    # legend
    d.add(Rect(150, 28, 10, 10, fillColor=ac_f, strokeColor=ac_s, strokeWidth=1))
    txt(166, 30, "Trained (small)", 7, col=MUTED, anchor="start")
    d.add(Rect(264, 28, 10, 10, fillColor=fr_f, strokeColor=fr_s, strokeWidth=1))
    txt(280, 30, "Frozen", 7, col=MUTED, anchor="start")
    return d


story.append(Paragraph("txn-repr-poc - a maintainer's guide", H1))
story.append(Paragraph(
    "Setting up, running, and not breaking the transaction-representation prototype. "
    "It replicates Raman, Ganesh and Veloso, arXiv:2410.07851 (NeurIPS 2024), on "
    "synthetic ISO 20022 payments.", SUB))
story.append(HRFlowable(width="100%", thickness=0.7, color=LINE, spaceAfter=8))

h2("Start here")
body("You can run the whole project on a laptop, with no GPU and no language model "
     "downloaded. For something built around a 1.3-billion-parameter model that sounds "
     "wrong, and it's the first thing worth knowing, because it's what makes the edit loop "
     "fast. Two commands:")
code("python run_gpu.py --smoke --limit 2000\npytest -q")
body("Between them they drive every layer - generator, encoder, decoder, scorer - on CPU "
     "in well under a minute. The reason it works is one seam: the decoder talks to the "
     "language model through a small interface (<font face='Courier'>LLMInterface</font>), "
     "and in smoke mode a 32-dimension MockLLM stands in for Phi-1.5. Nothing downloads. "
     "The 95 tests use the same stand-in, so a fresh checkout is green before you've "
     "installed <font face='Courier'>transformers</font> or touched a GPU. That's the "
     "property you want while changing code: a one-minute answer to whether you broke "
     "something.")

h2("What you need installed")
body("Day to day, you need Python and seven packages. The GPU and the real model are for "
     "the headline run only; you can maintain this code for weeks without either.")
mono0 = False
table(["What", "Version", "Note"], [
    ["Python", "3.10 - 3.12", "developed on 3.10.11"],
    ["git, pip, venv", "recent", "stdlib venv is fine"],
    ["requirements.txt", "numpy, pandas, pyarrow, pyyaml, torch, scikit-learn, catboost",
     "the torch CPU wheel is enough for dev"],
    ["pytest", "any", "not pinned in requirements - install it yourself"],
], [30, 86, 54])
body("Three more are optional and easy to over-install. "
     "<font face='Courier'>transformers</font> pulls in the real Phi-1.5 and matters only "
     "for the GPU run or for scoring a real checkpoint - the CPU loop never imports it. A "
     "CUDA GPU (the headline run used one H200, fp32) turns the real run from overnight "
     "into minutes. <font face='Courier'>huggingface_hub</font> plus a token only matters "
     "if you're pulling the published weights, which MODEL.md covers. Install them when you "
     "reach for them.")

h2("Setup, start to finish")
code("git clone https://github.com/Bratz/txn-repr-poc.git\n"
     "cd txn-repr-poc\n"
     "python -m venv .venv\n"
     "source .venv/bin/activate          # Windows: .venv\\Scripts\\Activate.ps1\n"
     "pip install -r requirements.txt\n"
     "pip install pytest\n"
     "# only for the real model run / scoring real checkpoints:\n"
     "# pip install transformers huggingface_hub")
body("Then prove the environment before you trust it:")
code("python data/synth_pacs008.py --parents 4000 --transactions 200000 \\\n"
     "  --out data/pacs008_synth.parquet --schema-out data/column_schema.json\n"
     "python run_gpu.py --smoke --limit 2000\n"
     "pytest -q")
body("Green tests mean you're set. Skip the data step and the tests still pass - they fall "
     "back to the committed <font face='Courier'>pacs008_sample_500.csv</font> and "
     "<font face='Courier'>column_schema.example.json</font>. That's deliberate: a clone "
     "should be testable before it has generated a single row.")

h2("How the pieces fit")
body("Every module names the paper section it implements in a comment at the top. Read "
     "them top to bottom - that's the order the data flows.")
mono0 = True
table(["Path", "Role"], [
    ["data/synth_pacs008.py", "Layer 1: the Algorithm-1 generator, the pacs.008 projection, the four task labels, and the schema 'tasks' manifest"],
    ["encoders/partitioning_embedder.py", "Sec 3.1 - frequency-split embedding for high-card account IDs"],
    ["encoders/quantizer.py", "Sec 3.3 - currency-conditioned amount quantizer"],
    ["encoders/party_encoder.py", "Sec 3.2 - offline party encoder + the persistent party store"],
    ["encoders/column_assembler.py", "Eq. 4 - routes each column to its encoder, builds the token sequence"],
    ["encoder/tabular_encoder.py", "Sec 3.4 - BERT (25M) with the reconstruction + batch-hard-triplet loss"],
    ["decoder/multimodal_decoder.py", "Sec 4 - frozen f + frozen LLM + trainable {Phi, psi, phi}; multi-record Eq. 5"],
    ["eval/metrics.py, baselines.py", "Layer 5 - per-task metrics and the CatBoost baseline"],
    ["run_gpu.py", "the orchestrator (see below)"],
    ["predict.py", "online scoring - save/load a checkpoint, score by task"],
    ["configs/default.yaml", "pinned hyperparameters + the falsifiable claims and thresholds"],
    ["data/column_schema.json", "the contract everything reads - buckets + task manifest"],
], [60, 110])
mono0 = False
body("Two files carry more than their names admit. "
     "<font face='Courier'>run_gpu.py</font> builds and freezes the encoder (the C1 "
     "experiment), then instruction-tunes the adapters across all four tasks and scores "
     "them against CatBoost (C2), and writes results.json. "
     "<font face='Courier'>column_schema.json</font> is the contract: buckets and the task "
     "list live there, and nothing downstream may hard-code a column list. Break that one "
     "rule and a schema change silently stops reaching the code that depends on it.")

h2("The design in one picture")
body("Here's how a payment becomes an answer at run time. The encoder turns a payment "
     "into one embedding; the adapter Phi makes that embedding a single token a frozen "
     "Phi-1.5 reads alongside the instruction and the task signal. Only the blue pieces "
     "train.")
story.append(design_drawing())
story.append(Spacer(1, 10))

h2("Running the real thing")
body("The full run swaps the MockLLM for a frozen Phi-1.5 and writes results.json. "
     "<font face='Courier'>--save-dir</font> persists a scorer; "
     "<font face='Courier'>--full-tune</font> adds the unfrozen-LLM comparator for C2.")
code("python run_gpu.py --save-dir ckpt\n"
     "python run_gpu.py --full-tune --save-dir ckpt")
body("Scoring a saved model is <font face='Courier'>predict.py</font>. Single-record tasks "
     "score per row; recurrence scores per debtor-creditor group, because recurrence is a "
     "pattern across several payments rather than a property of one.")
code("python predict.py --model-dir ckpt --input new_rows.parquet \\\n"
     "  --out scored.csv --task risk     # or geography | expense | recurrence")
body("One trap, and it has caught people. The predict CLI rebuilds a real language model "
     "from the name stored in the checkpoint. Point it at a smoke checkpoint, whose model "
     "is 'mock', and it tries to download a model called mock and dies. Smoke checkpoints "
     "can only be scored through the Python API with a MockLLM passed in - "
     "<font face='Courier'>load_model(dir, llm=MockLLM(...))</font>, which is what the "
     "tests do. Worth saying plainly: whether the four-task numbers hold at production "
     "scale, nobody knows yet - the published checkpoint is the risk-only run, and the "
     "multi-task GPU pass hasn't been done.")

h2("The thing most likely to break - the freeze invariant")
body("If you break one thing in this repo by accident, it'll be the freeze. At Layer 4 the "
     "encoder f and the language model are frozen; only three small modules train - the "
     "adapter Phi, the task embedding psi, and the prompt parameters phi. The headline "
     "result depends on it: unfreeze either the encoder or the LLM and you haven't tuned "
     "the model, you've run a different experiment, which means the C1 and C2 numbers in "
     "RESULTS.md stop describing the thing you're running and start describing something "
     "that no longer exists. There's an "
     "<font face='Courier'>assert_frozen()</font> guard, but it only fires if you call it. "
     "Treat the frozen state as load-bearing.")
body("Five more rules travel with the code, for the same reason: the value here is fidelity "
     "to the paper, not better metrics. Don't retune the pinned hyperparameters (B=4, "
     "alpha_v=-3, alpha_d=2.25; a 25M encoder, 3 epochs). Read buckets and tasks from "
     "column_schema.json, never from a list in the code. Keep to the three sanctioned "
     "departures - the pacs.008 schema, currency-conditioned quantization, imbalance-aware "
     "metrics - and raise a fourth out loud instead of slipping it in. Leave the walked-back "
     "extensions walked back: the completeness vector, the structuring/layering chain task, "
     "the held-out-typology split - and don't confuse that chain task with recurrence, "
     "which is a paper task and belongs here. When unsure, do what the paper did and leave "
     "a '# PAPER: section x.y' comment so the next person can check you.")

h2("Footguns")
body("A few things will cost you an afternoon if nobody warns you.")
body("<b>results.json lies.</b> It's committed and also gitignored - tracked from before "
     "someone added the ignore line - and it holds an old CPU smoke run. RESULTS.md is the "
     "real record of the GPU run; the JSON is a leftover. Don't quote it.")
body("<b>There's a real private key in the repo.</b> A file named "
     "<font face='Courier'>sshkey</font> sits in the root, left over from a GPU box. It's "
     "gitignored, so it isn't on GitHub, but it's in your working tree. Delete it or move "
     "it to ~/.ssh. A private key has no business next to source.")
body("<b>Phi loads in fp32 on purpose.</b> The adapter and prompt vectors are fp32, and an "
     "fp16 Phi throws at its first LayerNorm when they meet. If you 'save memory' by "
     "switching it to fp16, that's the crash you'll get.")
body("<b>Generated files aren't in the clone.</b> column_schema.json and the parquet are "
     "gitignored - regenerate them. The committed example files are what the tests fall "
     "back to. And on Windows, git's LF-to-CRLF warnings are just noise.")

h2("Changing things")
body("Adding a fifth task is the test of whether you've understood the shape. You write a "
     "label rule and a manifest entry in synth_pacs008.py; "
     "<font face='Courier'>build_task_specs</font> in run_gpu.py reads the manifest and "
     "wires the instruction tokens; predict.py scores it by name. A multi-record task sets "
     "<font face='Courier'>records='multi'</font> and names its group column. You shouldn't "
     "have to touch the encoder, the decoder internals, or the LLM. If you do, stop - "
     "that's usually the freeze invariant about to be broken.")
body("The whole job comes down to three habits: keep the one-minute CPU loop green, keep f "
     "and Phi frozen, and keep every claim traceable to a paper section or a sanctioned "
     "departure. For anything deeper, architecture.md is the source of truth on what each "
     "component is, and the module you need usually has the comment that answers your "
     "question at the top of the file.")

SimpleDocTemplate(
    str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
    topMargin=18 * mm, bottomMargin=16 * mm,
    title="txn-repr-poc - a maintainer's guide", author="txn-repr-poc",
).build(story)
print(f"wrote {OUT}")
