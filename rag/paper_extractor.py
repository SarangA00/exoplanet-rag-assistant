"""
PDF text extraction for research papers, feeding the same chunking/embedding
pipeline as the findings text files.

Settings here were tuned against 5 real test papers (see papers/):
- x_tolerance=1: pdfplumber's default word-spacing heuristic glues words
  together on some PDFs (fonts without explicit space characters). A
  tighter x_tolerance fixes this without regressing PDFs that were already
  extracting cleanly.
- filter(upright=True): drops rotated characters before extracting text.
  arXiv preprints print a vertical identifier watermark down the left
  margin of page 1; extracted naively (without rotation awareness) it comes
  out as reversed gibberish mixed into the start of the text. Filtering to
  upright-only characters removes it cleanly since it's the only rotated
  content on the page.

- Repeated running headers/footers (e.g. a page's running byline "N Author
  Name") are stripped via a data-driven check: any line that repeats
  verbatim across 3+ pages -- or repeats 3+ times once its leading page
  number is stripped off -- is treated as a running header, not content.

Known, accepted limitations (not worth solving for a small test batch):
- Equation-heavy passages can still contain broken glyph codes like
  "(cid:18)" where pdfminer couldn't map a math-font glyph to Unicode.
- A paper's two-column frontmatter (e.g. "EDITED BY" / "REVIEWED BY"
  sidebars) can interleave with the adjacent text column, since this is
  naive top-to-bottom extraction, not column-aware. Only seen affecting
  front-matter metadata, not body/results text, in the test batch.
"""

import re
from collections import Counter
from pathlib import Path

import pdfplumber

PAPERS_DIR = Path(__file__).resolve().parent / "papers"

# Glyph-code artifacts pdfminer emits when it can't map a math-font glyph to
# Unicode (common in equation-heavy passages), e.g. "(cid:18)".
CID_TOKEN_RE = re.compile(r"\(cid:\d+\)")


def _normalize_running_header(line: str) -> str:
    """Strip a leading page number so 'N Author Name' lines compare equal
    across pages regardless of which page number N is."""
    return re.sub(r"^\d+\s*", "", line).strip()


def _find_running_headers(pages_lines: list[list[str]]) -> set[str]:
    """Identify lines that repeat verbatim (or modulo a leading page number)
    across 3+ pages -- these are running headers/footers, not content."""
    all_lines = [line for lines in pages_lines for line in lines]
    line_counts = Counter(all_lines)
    normalized_counts = Counter(_normalize_running_header(line) for line in all_lines)

    running = set()
    for line in set(all_lines):
        if line_counts[line] >= 3:
            running.add(line)
            continue
        norm = _normalize_running_header(line)
        if len(norm) >= 8 and normalized_counts[norm] >= 3:
            running.add(line)
    return running


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF, cleaning up rotated watermark text,
    cid glyph codes, and repeated running headers/footers. Pages are
    joined with blank lines."""
    with pdfplumber.open(pdf_path) as pdf:
        pages_lines = []
        for page in pdf.pages:
            upright_only = page.filter(lambda obj: obj.get("upright", True))
            text = CID_TOKEN_RE.sub("", upright_only.extract_text(x_tolerance=1) or "")
            pages_lines.append([line.strip() for line in text.split("\n") if line.strip()])

    running = _find_running_headers(pages_lines)

    pages = []
    for lines in pages_lines:
        kept_lines = [line for line in lines if line not in running]
        if kept_lines:
            pages.append("\n".join(kept_lines))
    return "\n\n".join(pages)


if __name__ == "__main__":
    for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
        text = extract_pdf_text(pdf_path)
        print(f"{pdf_path.name}: {len(text.split())} words extracted")
