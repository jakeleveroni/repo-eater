# pip install "psycopg[binary,pool]" pgvector cocoindex numpy openai
# If using Ollama: pip install openai==1.*   (the official OpenAI sdk)

import os
import textwrap
from typing import List, Dict

import cocoindex
from numpy.typing import NDArray
import numpy as np
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector
from psycopg import sql

from openai import OpenAI

# --- CONFIG ---
# Postgres where CocoIndex exported your vectors
DB_URL = os.getenv("COCOINDEX_DATABASE_URL", "postgresql://cocoindex:cocoindex@localhost:5432/cocoindex")

# Name of the PG table that CocoIndex exported, e.g. "doc_embeddings".
# If you don't know it, check your CocoIndex export target or list tables.
TARGET_TABLE = os.getenv("COCOINDEX_TARGET_TABLE", "codeembedding_code_embeddings")

# LLM server (Ollama by default)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "does-not-matter-its-local")   # ignored by Ollama
MODEL_NAME      = os.getenv("MODEL_NAME", "qwen2.5-coder:7b")  # for Ollama
TOP_K           = int(os.getenv("TOP_K", "6"))
CHAR_BUDGET     = int(os.getenv("CHAR_BUDGET", "20000"))  # cap context size

# --- INIT ---
cocoindex.init()
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
pool = ConnectionPool(DB_URL)

# Use the SAME embedder as your indexing flow (example uses MiniLM)
@cocoindex.transform_flow()
def text_to_embedding(text: cocoindex.DataSlice[str]) -> cocoindex.DataSlice[NDArray[np.float32]]:
    return text.transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    )

def embed_query(q: str) -> NDArray[np.float32]:
    return text_to_embedding.eval(q)

def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    qvec = embed_query(query)
    with pool.connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # cosine distance with pgvector: smaller is closer, so ORDER BY distance ASC
            dbQuery = sql.SQL("""
                SELECT filename, location, code, embedding <=> %s AS distance
                FROM {table}
                ORDER BY distance ASC
                LIMIT %s
            """).format(table=sql.Identifier(TARGET_TABLE))

            cur.execute(dbQuery, (qvec, top_k))
            rows = cur.fetchall()
    # Convert to (score=similarity), keep useful metadata
    results = []
    for filename, location, text, dist in rows:
        similarity = 1.0 - float(dist)
        results.append({
            "filename": filename,
            "location": location,
            "text": text,
            "score": similarity,
        })
    return results

def build_context(chunks: List[Dict], char_budget: int = CHAR_BUDGET) -> str:
    blocks, used = [], 0
    for i, ch in enumerate(chunks, 1):
        snippet = ch["text"].strip()
        # lightly trim long snippets to fit budget
        snippet = snippet[: min(4000, len(snippet))]  # per-chunk cap
        block = f"""[#{i}] FILE: {ch['filename']} @ {ch['location']}
"""
        if used + len(block) > char_budget:
            break
        used += len(block)
        blocks.append(block)
    return "\n".join(blocks)

def answer(question: str) -> None:
    # 1) retrieve
    hits = retrieve(question, TOP_K)
    context = build_context(hits, CHAR_BUDGET)

    system_prompt = """\
You are a senior code assistant answering from the provided CONTEXT along with your existing code related training data.
Rules:
- Be concise and correct.
- If the CONTEXT lacks the answer, say what's missing and suggest where to look.
- When you reference code, include file paths and keep code blocks small.
- Return a short "Sources:" section listing the most relevant file paths.
"""

    user_prompt = f"""
QUESTION: {question} 
CONTEXT: {context if context.strip() else "(no matches)"}
"""

    with client.chat.completions.stream(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    ) as stream:
        for event in stream:
            # Each event is a dict with incremental delta
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
        print()

if __name__ == "__main__":
    while True:
        try:
            q = input("\nAsk about your codebase (empty to quit): ").strip()
            if not q:
                break
            answer(q)
        except KeyboardInterrupt:
            break
