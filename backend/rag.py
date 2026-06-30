"""
rag.py — Simple RAG (Retrieval-Augmented Generation) layer for StayFlow AI,
the AegeanStay Hotels Guest Support Copilot.

What this module does:
1. Loads the AegeanStay Hotels guest support knowledge base from markdown files
   in kb/.
2. Splits each file into small text chunks.
3. Creates an embedding (a list of numbers) for each chunk using Gemini.
4. Stores everything in memory and retrieves the most relevant chunks for a
   user query using cosine similarity (with numpy).

It is intentionally simple and beginner-friendly:
- No external vector database, no ChromaDB.
- Just in-memory numpy math, which is easy to read and explain.

This module does NOT generate answers with an LLM yet — it only retrieves the
most relevant support text. Answer generation comes in a later step.
"""

import os

import numpy as np
from dotenv import load_dotenv
from google import genai

# Load variables from a .env file (e.g. GEMINI_API_KEY) if present.
load_dotenv()

# The Gemini embedding model we use to turn text into vectors.
EMBEDDING_MODEL = "gemini-embedding-001"

# The knowledge base files we want to load.
KB_FILES = [
    "hotel_faq.md",
    "booking_policy.md",
    "checkin_checkout.md",
    "cancellations_refunds.md",
    "amenities_services.md",
]

# In-memory cache of the built index so we don't re-embed on every query.
_INDEX = None

# In-memory cache of the Gemini client.
_CLIENT = None


# --- Paths ------------------------------------------------------------------


def _kb_dir() -> str:
    """Return the absolute path to the kb/ folder.

    rag.py lives in backend/, but kb/ is one level above backend/ (in the
    project root). We resolve the path relative to THIS file, so it works no
    matter whether you run from the project root or from the backend folder.
    """
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    return os.path.join(project_root, "kb")


# --- Gemini client ----------------------------------------------------------


def _get_client() -> genai.Client:
    """Create (once) and return the Gemini client.

    Raises a clear error if GEMINI_API_KEY is not set.
    """
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to backend/.env "
                "(see backend/.env.example) before using the RAG layer."
            )
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


# --- Chunking ---------------------------------------------------------------


def chunk_text(text: str, max_chars: int = 800) -> list:
    """Split text into small chunks, grouping paragraphs up to max_chars.

    Simple strategy:
    - Split the text on blank lines into paragraphs.
    - Add paragraphs to the current chunk until it would exceed max_chars.
    - Skip empty pieces so we never return blank chunks.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        # If adding this paragraph keeps us under the limit, append it.
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}" if current else paragraph
        else:
            # Otherwise, store what we have and start a new chunk.
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks


# --- Document loading -------------------------------------------------------


def load_documents() -> list:
    """Load and chunk the knowledge base files.

    Returns a list of dicts like: {"text": "...", "source": "booking_policy.md"}.
    Raises a clear error if the kb/ folder is missing or has no usable content.
    """
    kb_dir = _kb_dir()
    if not os.path.isdir(kb_dir):
        raise RuntimeError(
            f"Knowledge base folder not found at: {kb_dir}. "
            "Make sure the kb/ folder exists in the project root."
        )

    documents = []
    for filename in KB_FILES:
        path = os.path.join(kb_dir, filename)
        if not os.path.isfile(path):
            # Skip a missing file but keep going with the others.
            continue

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        for chunk in chunk_text(content):
            documents.append({"text": chunk, "source": filename})

    if not documents:
        raise RuntimeError(
            f"No knowledge base content found in: {kb_dir}. "
            "Make sure the kb/ markdown files exist and are not empty."
        )

    return documents


# --- Embeddings -------------------------------------------------------------


def get_embedding(text: str) -> list:
    """Return the Gemini embedding (a list of floats) for a piece of text."""
    client = _get_client()
    response = client.models.embed_content(model=EMBEDDING_MODEL, contents=text)
    # The SDK returns a list of embeddings; we sent one text, so take the first.
    return list(response.embeddings[0].values)


# --- Similarity -------------------------------------------------------------


def cosine_similarity(a, b) -> float:
    """Return the cosine similarity (between -1 and 1) of two vectors."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


# --- Index ------------------------------------------------------------------


def build_index() -> list:
    """Load documents and create an embedding for each chunk.

    Returns a list of dicts like:
        {"text": "...", "source": "...", "embedding": [...]}.
    """
    documents = load_documents()

    index = []
    for doc in documents:
        embedding = get_embedding(doc["text"])
        index.append(
            {
                "text": doc["text"],
                "source": doc["source"],
                "embedding": embedding,
            }
        )

    return index


def _get_index() -> list:
    """Build the index once and reuse it for later queries (in-memory cache)."""
    global _INDEX
    if _INDEX is None:
        _INDEX = build_index()
    return _INDEX


# --- Retrieval --------------------------------------------------------------


def retrieve(query: str, top_k: int = 3) -> list:
    """Return the top_k most relevant knowledge base chunks for a query.

    Each result looks like: {"text": "...", "source": "...", "score": 0.82}.
    """
    index = _get_index()
    query_embedding = get_embedding(query)

    # Score every chunk by how similar it is to the query.
    scored = []
    for item in index:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append(
            {
                "text": item["text"],
                "source": item["source"],
                "score": round(score, 4),
            }
        )

    # Sort by score (highest first) and keep only the top_k results.
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


# --- Manual test ------------------------------------------------------------
# Run this file directly to do a quick retrieval check:
#   python rag.py

if __name__ == "__main__":
    # A couple of quick retrieval checks. The cancellation query should pull in
    # cancellations_refunds.md as one of the top sources.
    for query in ["What time is check-in?", "Can I cancel my booking?"]:
        print(f"\n=== Query: {query} ===")
        results = retrieve(query)
        for r in results:
            print(r["source"], r["score"])
            print(r["text"][:300])
