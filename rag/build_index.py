"""
Step 3 of the RAG pipeline.

Reads the findings text files from rag/findings/ and the research paper
PDFs from rag/papers/, chunks each with the appropriate chunker (paragraph
merging for findings, section+sentence merging for papers), embeds every
chunk with the local Embedder, and stores the embeddings (plus chunk text
and source metadata) in a sqlite-vec database for similarity search.
"""

import sqlite3
from pathlib import Path

import sqlite_vec

from chunker import chunk_text
from embedder import Embedder
from paper_chunker import chunk_paper
from paper_extractor import extract_pdf_text

FINDINGS_DIR = Path(__file__).resolve().parent / "findings"
PAPERS_DIR = Path(__file__).resolve().parent / "papers"
DB_PATH = Path(__file__).resolve().parent / "rag.db"


def get_connection() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def create_tables(db: sqlite3.Connection, dim: int) -> None:
    # Regular table for the human-readable content and metadata — one row
    # per chunk. chunk_index records the chunk's position within its
    # source document (0, 1, 2, ...). source_type distinguishes findings
    # write-ups ("analysis") from ingested research papers ("paper");
    # source_title is a human-readable label (the filename stem) so you
    # can identify a source without parsing the raw filename.
    db.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_title TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL
        )
    """)
    # vec0 virtual table for the embeddings, keyed by the same rowid/id so
    # the two tables can be joined at query time.
    db.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            embedding float[{dim}]
        )
    """)
    db.commit()


def collect_findings_chunks() -> list[dict]:
    rows = []
    for f in sorted(FINDINGS_DIR.glob("*.txt")):
        for i, text in enumerate(chunk_text(f.read_text())):
            rows.append({
                "source": f.name,
                "source_type": "analysis",
                "source_title": f.stem,
                "chunk_index": i,
                "text": text,
            })
    return rows


def collect_paper_chunks() -> list[dict]:
    if not PAPERS_DIR.exists():
        return []
    rows = []
    for f in sorted(PAPERS_DIR.glob("*.pdf")):
        paper_text = extract_pdf_text(f)
        for i, text in enumerate(chunk_paper(paper_text)):
            rows.append({
                "source": f.name,
                "source_type": "paper",
                "source_title": f.stem,
                "chunk_index": i,
                "text": text,
            })
    return rows


def main():
    embedder = Embedder()

    findings_rows = collect_findings_chunks()
    if not findings_rows:
        raise SystemExit(f"No .txt files found in {FINDINGS_DIR} — run build_findings.py first.")
    paper_rows = collect_paper_chunks()

    rows = findings_rows + paper_rows
    texts = [r["text"] for r in rows]

    print(
        f"Split {len(findings_rows)} findings chunks + {len(paper_rows)} paper "
        f"chunks = {len(rows)} total."
    )
    print(f"Embedding {len(texts)} chunks with {embedder.model.__class__.__name__}...")
    vectors = embedder.embed(texts)

    # Rebuild fresh each run — simplest approach for a handful of docs.
    if DB_PATH.exists():
        DB_PATH.unlink()

    db = get_connection()
    create_tables(db, embedder.dim)

    for i, (row, vector) in enumerate(zip(rows, vectors), start=1):
        db.execute(
            """
            INSERT INTO chunks (id, source, source_type, source_title, chunk_index, text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (i, row["source"], row["source_type"], row["source_title"], row["chunk_index"], row["text"]),
        )
        db.execute(
            "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
            (i, sqlite_vec.serialize_float32(vector)),
        )
    db.commit()

    row_count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    vec_count = db.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    db.close()

    print(f"Wrote {DB_PATH.name}: {row_count} chunk rows, {vec_count} embeddings.")


if __name__ == "__main__":
    main()
