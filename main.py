from dotenv import load_dotenv
import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from numpy.typing import NDArray
from typing import TypeVar
import cocoindex
import os
import numpy as np
from sentence_transformers import SentenceTransformer

CT = TypeVar('CT', bound=psycopg.Connection)

# Load once at startup
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text: str | list[str]) -> np.ndarray:
    """Shared embed function for both indexing and querying."""
    return model.encode(text, convert_to_numpy=True).astype(np.float32)

def search(pool: ConnectionPool[CT], query: str, top_k: int = 5):
    table_name = cocoindex.utils.get_target_default_name(text_embedding_flow, "code_embeddings")
    print(table_name, query)

    # Direct embedding for query
    query_vector = embed_text(query)
    print("going to db")

    with pool.connection() as conn:
        print("connected")
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT filename, code, embedding <=> %s AS distance
                FROM {table_name} ORDER BY distance LIMIT %s
            """, (query_vector, top_k))  # type: ignore
            print("query res")
            return [
                {"filename": row[0], "text": row[1], "score": 1.0 - row[2]}
                for row in cur.fetchall()
            ]

# Define an op so CocoIndex can use our SentenceTransformer during indexing
@cocoindex.op.function()  # type: ignore
def embed_op(text: str) -> NDArray[np.float32]:
    return embed_text(text)

@cocoindex.op.function()  # type: ignore
def extract_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()

@cocoindex.flow_def(name="CodeEmbedding")
def text_embedding_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    print("Building flow...")
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="/Users/f1253/dev/projects/js-monorepo",
            included_patterns=["**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx", "**/*.md", "**/*.mdx", "**/*.json"],
            excluded_patterns=[".*", "**/dist", "**/node_modules"],
        )
    )

    code_embeddings = data_scope.add_collector()

    with data_scope["documents"].row() as file:
        print(f"Processing file: {file['filename']}")
        file["extension"] = file["filename"].transform(extract_extension)  # type: ignore
        file["chunks"] = file["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language=file["extension"],
            chunk_size=1000,
            chunk_overlap=300,
        )

        with file["chunks"].row() as chunk:
            # use our custom embedding op
            chunk["embedding"] = chunk["text"].transform(embed_op) # type: ignore
            code_embeddings.collect(
                filename=file["filename"],
                location=chunk["location"],
                code=chunk["text"],
                embedding=chunk["embedding"],
            )

    code_embeddings.export(
        "code_embeddings",
        cocoindex.storages.Postgres(),
        primary_key_fields=["filename", "location"],
        vector_index=[("embedding", cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY)],
    )

if __name__ == "__main__":
    load_dotenv()
    print("Starting CocoIndex...")

    cocoindex.init()
    pool = ConnectionPool(os.getenv("COCOINDEX_DATABASE_URL"))  # type: ignore

    while True:
        try:
            query = input("Enter search query (or Enter to quit): ")
            if query == "":
                break
            results = search(pool, query)
            print("\nSearch results:")
            for result in results:
                print(f"[{result['score']:.3f}] {result['filename']}")
                print(f"    {result['text']}")
                print("---")
            print()
        except KeyboardInterrupt:
            break
