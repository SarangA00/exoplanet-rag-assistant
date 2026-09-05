"""
Chunking for research papers -- different from the findings-file chunker
because papers have real structure (sections/subsections) but no reliable
blank-line paragraph breaks once extracted from PDF.
1. Detect section header lines (e.g. "ABSTRACT", "3.1. Necessary
   Information", "2 Methods") and split the paper into (header, body)
   sections at those boundaries.
2. Drop the References/Bibliography section entirely -- citations are low
   value for retrieval.
3. Within each remaining section, split into sentences (there are no
   paragraph breaks to split on) and merge sentences up to ~200-400 words
   using the same merge_units() logic the findings chunker uses.

Header detection was tuned against 5 real test papers and deliberately
favors precision over recall, to avoid false-positive headers splitting a
chunk mid-paragraph:
- Exact match against known section names (Abstract, Introduction, ...).
- Or a numbered prefix ("3.", "2.1", ...) followed by at least one real
  word (>=3 letters) where every word starts with a capital letter -- this
  rejects numbered list items inside body text (e.g. "3. To provide an
  interactive tool...", which is a sentence, not Title Case/ALL CAPS) and
  rejects table/axis-label noise ("100 101 102 ⊕", "1 TP TN", which have no
  real words after the number).

Known, accepted limitation: when a paper's column-interleaving (see
paper_extractor.py) fuses a real header with adjacent body text on the
same line, the merged line usually fails the "all words capitalized"
check and the header goes undetected -- that section just doesn't get its
own boundary and ends up merged into whatever chunk follows. Seen on 1 of
5 test papers, affecting subsection granularity only, not correctness.
"""

import re

from chunker import merge_units

MIN_WORDS = 200
MAX_WORDS = 400

KNOWN_HEADERS = {
    "abstract", "introduction", "methods", "results", "discussion",
    "conclusion", "conclusions", "references", "bibliography",
    "acknowledgments", "acknowledgements", "appendix",
}
DROP_SECTION_HEADERS = {"references", "bibliography"}

NUMBERED_HEADER_RE = re.compile(r"^(\d+(\.\d+)*)\.?\s+(.+)$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def is_header(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 55:
        return False
    if line.lower().rstrip(".") in KNOWN_HEADERS:
        return True

    m = NUMBERED_HEADER_RE.match(line)
    if not m:
        return False
    rest_words = m.group(3).split()
    if not rest_words:
        return False
    if not all(w[0].isupper() for w in rest_words if w[0].isalpha()):
        return False
    # Require at least one real word, not just digits/symbols/2-letter abbrevs
    # (rejects table rows and axis-label noise like "100 101 102 ⊕").
    return any(w.isalpha() and len(w) >= 3 for w in rest_words)


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (header, body) pairs using detected header lines as
    boundaries. Text before the first header (title, authors, ...) is kept
    under an empty-string header."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and is_header(stripped):
            sections.append((stripped, []))
        else:
            sections[-1][1].append(line)

    return [
        (header, "\n".join(body_lines).strip())
        for header, body_lines in sections
        if "\n".join(body_lines).strip()
    ]


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())  # collapse newlines/whitespace first
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def chunk_paper(text: str, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> list[str]:
    chunks = []
    for header, body in split_into_sections(text):
        if header.lower().rstrip(".") in DROP_SECTION_HEADERS:
            continue
        sentences = split_sentences(body)
        if sentences:
            chunks.extend(merge_units(sentences, min_words, max_words, join_str=" "))
    return chunks


if __name__ == "__main__":
    from pathlib import Path

    from paper_extractor import PAPERS_DIR, extract_pdf_text

    for pdf_path in sorted(PAPERS_DIR.glob("*.pdf")):
        text = extract_pdf_text(pdf_path)
        chunks = chunk_paper(text)
        print(f"\n{pdf_path.name}: {len(chunks)} chunk(s)")
        for i, c in enumerate(chunks, start=1):
            print(f"  [{i}] ({len(c.split())} words) {c[:80]!r}...")
