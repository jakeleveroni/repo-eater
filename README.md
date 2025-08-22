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

### Optional Install Embedding Model Locally

The embedding generation is done via a separate model. The script is setup to pull the model from hugging face by default. If you want to pull the model locally so you dont have to download it on each run or you want to be able to run the script offline you can follow these steps:

```sh
# Make a folder to store models
mkdir -p ~/models/sentence-transformers
cd ~/models/sentence-transformers

# Use huggingface-cli to download
pip install huggingface_hub
huggingface-cli repo download sentence-transformers/all-MiniLM-L6-v2
```

And update the model path in `main.py`:

```python
# Change this
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# To this (might have to use absolute path i havent tried with relative path)
model = SentenceTransformer("~/models/sentence-transformers/all-MiniLM-L6-v2")
```

NOTE: i havent tried other models but you if you want to try them this is where you would change that out.

## Querying

TODO i aint figured it out yet