"""
Step 2 of the RAG pipeline.

Wraps a local sentence-transformers model behind a small `Embedder` class.
Every other script (build_index.py, ask.py) talks to embeddings only through
`Embedder.embed()` — so swapping this out for an API-based embedding model
later means writing a new class with the same `embed()` method, not touching
any of the retrieval or storage code.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning one vector per text."""
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for embedding a single string."""
        return self.embed([text])[0]


if __name__ == "__main__":
    embedder = Embedder()
    print(f"Loaded {MODEL_NAME}, embedding dimension = {embedder.dim}")
    vec = embedder.embed_one("How many exoplanets were analyzed?")
    print(f"Sample embedding (first 5 of {len(vec)} dims): {vec[:5]}")
