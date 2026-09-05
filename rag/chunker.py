"""
Splits a findings text file into smaller chunks for embedding.

Strategy (deliberately simple):
1. Split the text on blank lines into paragraphs.
2. Walk through the paragraphs, accumulating them into a "buffer" chunk.
3. Once the buffer reaches MIN_WORDS, cut it loose as a finished chunk and
   start a new one. This merges small paragraphs together (like a 15-word
   summary line) while stopping a chunk from growing forever.
4. If adding the next paragraph would push the buffer past MAX_WORDS, cut
   the buffer first, then start the new paragraph in a fresh buffer — so a
   chunk never grows unboundedly even if MIN_WORDS is never reached.

Note: if a single paragraph is *by itself* already longer than MAX_WORDS,
it's kept as its own (oversized) chunk rather than split mid-paragraph.
That's a known limitation of this simple version — fine for the current
findings files, but worth revisiting if a source document ever has one
huge paragraph.
"""

MIN_WORDS = 200
MAX_WORDS = 250


def merge_units(
    units: list[str], min_words: int, max_words: int, join_str: str = "\n\n"
) -> list[str]:
    """
    Greedily merge small text units (paragraphs, sentences, ...) into chunks
    within roughly [min_words, max_words], preserving order. Shared by the
    findings-file chunker (paragraph units) and the paper chunker (sentence
    units within a section) below.
    """
    chunks = []
    buffer_units: list[str] = []
    buffer_words = 0

    for unit in units:
        unit_words = len(unit.split())

        # The current buffer already hit the target size — start fresh.
        if buffer_units and buffer_words >= min_words:
            chunks.append(join_str.join(buffer_units))
            buffer_units = []
            buffer_words = 0

        # Adding this unit would overflow the max — flush what we have
        # first, so the new unit starts its own chunk instead.
        if buffer_units and buffer_words + unit_words > max_words:
            chunks.append(join_str.join(buffer_units))
            buffer_units = []
            buffer_words = 0

        buffer_units.append(unit)
        buffer_words += unit_words

    if buffer_units:
        chunks.append(join_str.join(buffer_units))

    return chunks


def chunk_text(text: str, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return merge_units(paragraphs, min_words, max_words, join_str="\n\n")


if __name__ == "__main__":
    from pathlib import Path

    findings_dir = Path(__file__).resolve().parent / "findings"
    for path in sorted(findings_dir.glob("*.txt")):
        chunks = chunk_text(path.read_text())
        print(f"\n{path.name}: {len(chunks)} chunk(s)")
        for i, c in enumerate(chunks, start=1):
            print(f"  [{i}] ({len(c.split())} words) {c[:80]!r}...")
