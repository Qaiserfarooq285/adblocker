#!/usr/bin/env python3
from pathlib import Path
import re

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"
OUTPUT_PATH = REPO_ROOT / "betting_ad_blocker_documentation.pdf"


def parse_markdown_to_story(text: str):
    styles = getSampleStyleSheet()
    normal = styles["BodyText"]
    normal.fontName = "Helvetica"
    normal.fontSize = 10
    normal.leading = 13

    heading1 = ParagraphStyle(
        "Heading1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        spaceAfter=8,
        spaceBefore=12,
    )
    heading2 = ParagraphStyle(
        "Heading2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        spaceAfter=6,
        spaceBefore=8,
    )
    heading3 = ParagraphStyle(
        "Heading3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        spaceAfter=4,
        spaceBefore=6,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=normal,
        leftIndent=18,
        bulletIndent=0,
        spaceAfter=2,
    )
    code = ParagraphStyle(
        "Code",
        parent=normal,
        fontName="Courier",
        fontSize=8.5,
        leading=10.5,
        backColor="#f5f5f5",
        borderPadding=4,
        leftIndent=8,
        rightIndent=8,
        spaceAfter=6,
    )

    story = []
    lines = text.splitlines()
    i = 0

    while i < len(lines):
        raw = lines[i].rstrip()
        if not raw.strip():
            story.append(Spacer(1, 4))
            i += 1
            continue

        if raw.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            if code_lines:
                story.append(Paragraph("<font name=Courier>" + "<br/>".join(code_lines) + "</font>", code))
                story.append(Spacer(1, 4))
            continue

        if re.match(r"^#{1,6}\s+", raw):
            level = len(raw) - len(raw.lstrip("#"))
            text_part = raw.lstrip("#").strip()
            if level == 1:
                story.append(Paragraph(text_part, heading1))
            elif level == 2:
                story.append(Paragraph(text_part, heading2))
            else:
                story.append(Paragraph(text_part, heading3))
            i += 1
            continue

        if re.match(r"^[-*]\s+", raw):
            story.append(Paragraph(re.sub(r"^[-*]\s+", "• ", raw), bullet))
            i += 1
            continue

        if re.match(r"^\d+\.\s+", raw):
            story.append(Paragraph(re.sub(r"^\d+\.\s+", "• ", raw), bullet))
            i += 1
            continue

        para_lines = [raw]
        i += 1
        while i < len(lines):
            nxt = lines[i].rstrip()
            if not nxt.strip():
                break
            if nxt.startswith("```") or re.match(r"^#{1,6}\s+", nxt) or re.match(r"^[-*]\s+", nxt) or re.match(r"^\d+\.\s+", nxt):
                break
            para_lines.append(nxt)
            i += 1
        story.append(Paragraph(" ".join(para_lines), normal))
        story.append(Spacer(1, 4))

    return story


def build_pdf():
    if not README_PATH.exists():
        raise FileNotFoundError(f"README not found: {README_PATH}")

    text = README_PATH.read_text(encoding="utf-8")
    story = parse_markdown_to_story(text)
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    doc.build(story)


if __name__ == "__main__":
    build_pdf()
    print(f"Wrote {OUTPUT_PATH}")
