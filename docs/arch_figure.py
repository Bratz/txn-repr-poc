"""Shared architecture figure (v1 + v2) as a native reportlab Drawing.

Imported by build_developer_guide.py and build_science_paper.py so both PDFs and the
standalone docs/architecture-phi.svg stay in sync. Self-contained (defines its own
colours) and ASCII-only so the built-in PDF fonts render it.
"""

import math

from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String

_INK = colors.HexColor("#1d1d1f")
_MUTED = colors.HexColor("#5f6168")
_ACC = colors.HexColor("#2c6ecb")
_GREY = colors.HexColor("#6b6b70")
_AC_F, _AC_S = colors.HexColor("#dce8f7"), colors.HexColor("#2c6ecb")   # trained
_FR_F, _FR_S = colors.HexColor("#e7e8ea"), colors.HexColor("#9a9aa0")   # frozen
_PL_F, _PL_S = colors.white, colors.HexColor("#c4c4c8")                 # plain
_RAIL = colors.HexColor("#eef2f8")


def design_drawing():
    """v1 (paper) + v2 (sequence extension) architecture, ~474 x 430 pt."""
    d = Drawing(474, 430)

    def txt(x, y, s, size=7.5, bold=False, col=_INK, anchor="middle"):
        d.add(String(x, y, s, fontName="Helvetica-Bold" if bold else "Helvetica",
                     fontSize=size, fillColor=col, textAnchor=anchor))

    def cbox(x, y, w, h, kind, title, sub=None, tsize=8):
        f, s = {"a": (_AC_F, _AC_S), "f": (_FR_F, _FR_S), "p": (_PL_F, _PL_S)}[kind]
        d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=f, strokeColor=s, strokeWidth=1))
        cx = x + w / 2
        if sub:
            txt(cx, y + h / 2 + 1, title, tsize, True)
            txt(cx, y + h / 2 - 8, sub, 6, col=_MUTED)
        else:
            txt(cx, y + h / 2 - 3, title, tsize, True)

    def arrow(x1, y1, x2, y2, dash=False):
        ln = Line(x1, y1, x2, y2, strokeColor=_GREY, strokeWidth=1.2)
        if dash:
            ln.strokeDashArray = [3, 2]
        d.add(ln)
        ang = math.atan2(y2 - y1, x2 - x1)
        bx, by = x2 - 5 * math.cos(ang), y2 - 5 * math.sin(ang)
        px, py = -math.sin(ang) * 2.4, math.cos(ang) * 2.4
        d.add(Polygon([x2, y2, bx + px, by + py, bx - px, by - py],
                      fillColor=_GREY, strokeColor=_GREY))

    # ---- shared representation row ----
    txt(2, 426, "SHARED REPRESENTATION (v1 + v2)", 6.5, True, _MUTED, anchor="start")
    cbox(2, 394, 64, 30, "p", "Payment", "pacs.008")
    arrow(66, 409, 76, 409)
    cbox(76, 394, 66, 30, "p", "Projection", "to row")
    arrow(142, 409, 152, 409)
    cbox(152, 394, 120, 30, "p", "Layer 2 encoders", "3.1 . 3.3 . 3.2")
    arrow(272, 409, 282, 409)
    cbox(282, 394, 92, 30, "f", "Layer 3 enc", "BERT 25M frozen")
    arrow(374, 409, 384, 409)
    cbox(384, 394, 56, 30, "f", "f(x)", "per txn")

    # ---- f(x) rail ----
    arrow(412, 394, 412, 388)
    d.add(Rect(2, 372, 470, 16, rx=5, ry=5, fillColor=_RAIL, strokeColor=_PL_S, strokeWidth=1))
    txt(237, 377, "f(x) - frozen per-transaction representation, shared by both paths",
        7, True, _ACC)

    # ---- v1 lane (left, solid) ----
    d.add(Rect(2, 44, 228, 320, rx=8, ry=8, fillColor=None, strokeColor=_PL_S, strokeWidth=1.3))
    txt(14, 350, "v1 . per transaction", 8, True, _INK, anchor="start")
    txt(14, 340, "faithful to the paper", 6, False, _MUTED, anchor="start")
    arrow(116, 372, 116, 306)
    cbox(24, 258, 184, 48, "f", "Layer 4 - decoder", "frozen Phi + trainable {Phi,psi,phi}", 8)
    arrow(116, 258, 116, 236)
    cbox(24, 190, 184, 46, "p", "per-transaction task", "risk . geo . expense . recurrence", 8)

    # ---- v2 lane (right, dashed = beyond the paper) ----
    v2 = Rect(240, 30, 232, 334, rx=8, ry=8, fillColor=None, strokeColor=_ACC, strokeWidth=1.2)
    v2.strokeDashArray = [5, 3]
    d.add(v2)
    txt(252, 350, "v2 . per entity over time", 8, True, _INK, anchor="start")
    d.add(Rect(398, 345, 66, 13, rx=6, ry=6, fillColor=_AC_F, strokeColor=_ACC, strokeWidth=0.8))
    txt(431, 349, "beyond paper", 5.5, False, _ACC)
    arrow(356, 372, 356, 332)
    cbox(256, 302, 200, 30, "p", "ordered history: e1..en (+ time)", None, 7.5)
    arrow(356, 302, 356, 284)
    cbox(256, 240, 200, 44, "a", "Layer 3b - history encoder", "[USR] + events  ->  h_USR", 8)
    arrow(356, 240, 356, 220)
    cbox(316, 182, 80, 36, "f", "h_USR", "entity vec", 8)
    arrow(344, 182, 304, 158)
    arrow(372, 182, 412, 158, dash=True)
    cbox(250, 116, 104, 40, "a", "Option A", "linear / LoRA probe", 8)
    cbox(366, 116, 104, 40, "p", "Option B", "Phi - C5: drop", 8)
    arrow(302, 116, 302, 96)
    cbox(250, 60, 104, 34, "p", "entity task", "regime change", 8)

    # ---- legend ----
    d.add(Rect(110, 10, 12, 12, rx=2, ry=2, fillColor=_AC_F, strokeColor=_AC_S, strokeWidth=1))
    txt(126, 13, "Trained (small)", 6.5, False, _MUTED, anchor="start")
    d.add(Rect(210, 10, 12, 12, rx=2, ry=2, fillColor=_FR_F, strokeColor=_FR_S, strokeWidth=1))
    txt(226, 13, "Frozen", 6.5, False, _MUTED, anchor="start")
    leg = Rect(280, 10, 12, 12, rx=2, ry=2, fillColor=_PL_F, strokeColor=_ACC, strokeWidth=1)
    leg.strokeDashArray = [3, 2]
    d.add(leg)
    txt(296, 13, "v2 / optional (beyond the paper)", 6.5, False, _MUTED, anchor="start")
    return d
