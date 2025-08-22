from dotenv import load_dotenv
import psycopg

from typing import TypeVar
import cocoindex
import os

CT = TypeVar('CT', bound=psycopg.Connection)

# Load once at startup
model = cocoindex.functions.SentenceTransformerEmbed(model="sentence-transformers/all-MiniLM-L6-v2")

@cocoindex.op.function()  # type: ignore
def extract_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()

@cocoindex.flow_def(name="CodeEmbedding")
def text_embedding_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="/Users/f1253/dev/projects/js-monorepo",
            included_patterns=["**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx", "**/*.md", "**/*.mdx", "**/*.json"],
            excluded_patterns=[".*", "**/dist", "**/node_modules"],
        )
    )

    code_embeddings = data_scope.add_collector()

    with data_scope["documents"].row() as file:
        file["extension"] = file["filename"].transform(extract_extension)  # type: ignore
        file["chunks"] = file["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language=file["extension"],
            chunk_size=1000,
            chunk_overlap=300,
        )

        with file["chunks"].row() as chunk:
            # use our custom embedding op
            chunk["embedding"] = chunk["text"].transform(model) # type: ignore
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