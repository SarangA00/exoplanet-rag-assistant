"""
Step 4 of the RAG pipeline.

Command-line tool: ask a question, embed it, retrieve the closest matching
chunk(s) from rag.db via sqlite-vec similarity search, then ask a local
Ollama model to synthesize an answer grounded in those chunks.
"""

# Importing main libraries
import argparse
import sqlite3
from pathlib import Path

import ollama
import sqlite_vec

from embedder import Embedder

DB_PATH = Path(__file__).resolve().parent / "rag.db"
DEFAULT_K = 2

OLLAMA_MODEL = "llama3.2:3b"


# Main system prompt
SYSTEM_PROMPT = (
    "You answer questions about an exoplanet dataset using only the context "
    "provided below. If the context doesn't contain the answer, say so "
    "plainly instead of guessing or using outside knowledge."
)

# Distance threshold set to 1.2, ensures that only relevant chunks are pulled
# Distances exceeding 1.2 will return a "Not found" answer
DISTANCE_THRESHOLD = 1.2

# Opens the rag database and loads the sqlite vector extention, allows for the similarity search
def get_connection() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


# This runs the actual nearest match query
def search(db: sqlite3.Connection, query_vector: list[float], k: int) -> list[dict]:
    # vec0 KNN queries need the result count as a literal `k = N` constraint
    # in the WHERE clause — it can't be a bound parameter, since the query
    # planner needs it up front to run the search. `k` comes from our own
    # CLI arg, not user text, so interpolating it directly here is safe.
    rows = db.execute(
        f"""
        SELECT c.source, c.source_type, c.source_title, c.chunk_index, c.text, v.distance
        FROM vec_chunks AS v
        JOIN chunks AS c ON c.id = v.rowid
        WHERE v.embedding MATCH ? AND k = {int(k)}
        ORDER BY v.distance
        """,
        (sqlite_vec.serialize_float32(query_vector),),
    ).fetchall()

    return [
        {
            "source": r[0],
            "source_type": r[1],
            "source_title": r[2],
            "chunk_index": r[3],
            "text": r[4],
            "distance": r[5],
        }
        for r in rows
    ]


# Takes the remaining chunks from the filter step below and sends them to the llama AI along with the system prompt, grounding step
def generate_answer(question: str, chunks: list[dict]) -> str:
    """Ask a local Ollama model to answer the question using only the given chunks."""
    context = "\n\n".join(
        f"[{c['source']} chunk {c['chunk_index']}]\n{c['text']}" for c in chunks
    )
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response["message"]["content"]


# The main filter step, it removes chunks past the distance threshold
def answer_from_results(
    question: str, results: list[dict], threshold: float = DISTANCE_THRESHOLD
) -> str:
    """
    Turn retrieved chunks into an answer. Chunks past the distance threshold
    are dropped first — otherwise nearest-neighbor search would always
    return *something*, even for questions with no real answer in the data.
    Everything under threshold is handed to the LLM as context so it can
    synthesize an answer instead of returning raw chunk text.
    """
    relevant = [r for r in results if r["distance"] <= threshold]
    if not relevant:
        return "No relevant information found for this question."
    return generate_answer(question, relevant)


# Calls all of the functions in a logical fashion
def main():
    parser = argparse.ArgumentParser(
        description="Ask a question about the exoplanet analysis findings."
    )
    parser.add_argument("question", help="Your question, in quotes.")
    parser.add_argument(
        "-k", type=int, default=DEFAULT_K,
        help=f"Number of results to retrieve (default {DEFAULT_K}).",
    )
    parser.add_argument(
        "--threshold", type=float, default=DISTANCE_THRESHOLD,
        help=f"Max distance to accept as relevant (default {DISTANCE_THRESHOLD}).",
    )
    args = parser.parse_args()

    embedder = Embedder()
    query_vector = embedder.embed_one(args.question)

    db = get_connection()
    results = search(db, query_vector, args.k)
    db.close()

    print(f"\nQuestion: {args.question}\n")
    print("Retrieved:")
    for r in results:
        flag = "" if r["distance"] <= args.threshold else "  (over threshold, ignored)"
        print(
            f"  [{r['source_type']}:{r['source_title']} chunk {r['chunk_index']}] "
            f"distance={r['distance']:.4f}{flag}"
        )

    print("\nAnswer:")
    print(answer_from_results(args.question, results, args.threshold))


if __name__ == "__main__":
    main()
