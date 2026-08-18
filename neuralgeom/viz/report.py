"""
neuralgeom.viz.report — the PDF report builder used by every write-up.
=================================================================

A thin wrapper over reportlab: title, headings, paragraphs, highlighted
boxes, tables, definition lists and captioned figures. Extracted from the
per-experiment helper module so all reports share one implementation.

    rep = Report(PDF_DIR / "out.pdf", "Title", "subtitle")
    rep.h1("Section").p("text").figure(path, "caption").build()
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

from PIL import Image as PILImage                                # noqa: E402
from reportlab.lib.colors import HexColor                        # noqa: E402
from reportlab.lib.enums import TA_CENTER                        # noqa: E402
from reportlab.lib.pagesizes import letter                       # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch                             # noqa: E402
from reportlab.platypus import (Image, PageBreak, Paragraph,     # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)


class Report:
    """Minimal document builder shared by the experiment scripts."""

    def __init__(self, out_path: Path, title: str, subtitle: str,
                 accent: str = "#0f766e"):
        self.out = Path(out_path)
        self.accent = HexColor(accent)
        self.grey = HexColor("#555555")
        ss = getSampleStyleSheet()
        self.s_title = ParagraphStyle("t", parent=ss["Title"], fontSize=19,
                                      textColor=self.accent, spaceAfter=3)
        self.s_sub = ParagraphStyle("st", parent=ss["Normal"], fontSize=10.5,
                                    textColor=self.grey, alignment=TA_CENTER,
                                    spaceAfter=15)
        self.s_h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=13.5,
                                   textColor=self.accent, spaceBefore=14,
                                   spaceAfter=5)
        self.s_h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11,
                                   spaceBefore=9, spaceAfter=3)
        self.s_body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.7,
                                     leading=13.4, spaceAfter=6.5)
        self.s_cap = ParagraphStyle("c", parent=ss["Normal"], fontSize=8.5,
                                    leading=11.3, textColor=self.grey,
                                    spaceBefore=2, spaceAfter=11,
                                    leftIndent=12, rightIndent=12)
        self.s_box = ParagraphStyle("bx", parent=self.s_body,
                                    backColor=HexColor("#eaf3f2"),
                                    borderPadding=6, borderColor=self.accent,
                                    borderWidth=0.7, spaceBefore=5,
                                    spaceAfter=9)
        self.s_cell = ParagraphStyle("cl", parent=self.s_body, fontSize=8.8,
                                     leading=11.8, spaceAfter=0)
        self.s_mono = ParagraphStyle("m", parent=self.s_body,
                                     fontName="Courier", fontSize=8.2,
                                     leading=11.2,
                                     backColor=HexColor("#f2f5f5"),
                                     borderPadding=5, spaceAfter=8)
        self.story = [Paragraph(title, self.s_title),
                      Paragraph(subtitle, self.s_sub)]

    # -- content ---------------------------------------------------------- #
    def h1(self, t): self.story.append(Paragraph(t, self.s_h1)); return self
    def h2(self, t): self.story.append(Paragraph(t, self.s_h2)); return self
    def p(self, t): self.story.append(Paragraph(t, self.s_body)); return self
    def box(self, t): self.story.append(Paragraph(t, self.s_box)); return self
    def mono(self, t): self.story.append(Paragraph(t, self.s_mono)); return self
    def pagebreak(self): self.story.append(PageBreak()); return self
    def space(self, h=8): self.story.append(Spacer(1, h)); return self

    def figure(self, path: Path, caption: str = "", width: float = 6.4):
        w, h = PILImage.open(path).size
        self.story.append(Image(str(path), width=width * inch,
                                height=width * inch * h / w))
        if caption:
            self.story.append(Paragraph(caption, self.s_cap))
        return self

    def table(self, header: Sequence[str], rows: Sequence[Sequence],
              widths: Optional[Sequence[float]] = None, fontsize=8.8):
        st = ParagraphStyle("tc", parent=self.s_cell, fontSize=fontsize)
        data = [[Paragraph(f"<b>{c}</b>", st) for c in header]] + \
               [[Paragraph(str(c), st) for c in r] for r in rows]
        n = len(header)
        widths = widths or [6.4 / n] * n
        t = Table(data, colWidths=[w * inch for w in widths])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#d5e8e5")),
            ("GRID", (0, 0), (-1, -1), 0.45, self.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        self.story.append(t)
        self.space(7)
        return self

    def deflist(self, rows: Sequence[Tuple[str, str]], w0: float = 1.5):
        st_term = ParagraphStyle("dt", parent=self.s_cell,
                                 textColor=self.accent)
        data = [[Paragraph(f"<b>{a}</b>", st_term),
                 Paragraph(b, self.s_cell)] for a, b in rows]
        t = Table(data, colWidths=[w0 * inch, (6.4 - w0) * inch])
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#9fc7c2")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4.5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [HexColor("#ffffff"), HexColor("#f3f8f7")]),
        ]))
        self.story.append(t)
        self.space(7)
        return self

    def build(self):
        doc = SimpleDocTemplate(
            str(self.out), pagesize=letter, leftMargin=0.85 * inch,
            rightMargin=0.85 * inch, topMargin=0.75 * inch,
            bottomMargin=0.75 * inch, title=self.out.stem)
        doc.build(self.story)
        print(f"\nwrote {self.out}")
        return self.out
