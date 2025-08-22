# Repo Embedding Generator

This project allows you to vectorize your codebase and store it in a local database

## Setup

### Install dependencies

We are using pyenv and python 3.13.0, along with pip. I did also try this with `uv` and it worked fine as well

```python
brew install pyenv
pyenv install 3.13.0
pyenv global 3.13.0

# pip
pip install .

# uv
uv sync
```

Start the database

```
docker compose up -d
```

### Update repo path

In main.py change the path to the repo you want to embed

```
cocoindex.sources.LocalFile(
    path="/Users/f1253/dev/projects/js-monorepo", # <-- change this to whatever you want
    included_patterns=["**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx", "**/*.md", "**/*.mdx", "**/*.json"],
    excluded_patterns=[".*", "**/dist", "**/node_modules"],
)
```

and now we can generate the embeddings via

```python
# pip
cocoindex update --setup main.py

#uv
uv run --with cocoindex cocoindex update --setup main.py
```

Your database should be populated by the embedding once the process completes.

## Querying

TODO i aint figured it out yet