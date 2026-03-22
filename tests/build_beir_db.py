"""Build BEIR dataset SQLite DBs for flexvec testing.

Usage:
    python tests/build_beir_db.py                              # all 4, local ONNX
    python tests/build_beir_db.py scifact                      # one dataset
    python tests/build_beir_db.py --nomic-key nk-xxx           # all 4, Nomic API (fast)
    NOMIC_API_KEY=nk-xxx python tests/build_beir_db.py         # env var works too
    python tests/build_beir_db.py --nomic-key nk-xxx fiqa      # one dataset, API
"""
import json
import os
import sqlite3
import sys
import time
import urllib.request
import zipfile

import numpy as np

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
DEST_ROOT = "/tmp/beir_flexvec"

DATASETS = {
    "scifact":  {"docs": "~5K",  "domain": "science/claims"},
    "nfcorpus": {"docs": "~3.6K", "domain": "biomedical"},
    "fiqa":     {"docs": "~57K", "domain": "financial QA"},
    "scidocs":  {"docs": "~25K", "domain": "citation prediction"},
}


def download_dataset(name: str) -> str:
    """Download and extract a BEIR dataset. Returns path to extracted dir."""
    dest = f"{DEST_ROOT}/{name}"
    base = f"{dest}/{name}"
    if os.path.exists(base):
        print(f"  {name} already at {base}")
        return base

    url = f"{BEIR_BASE_URL}/{name}.zip"
    zip_path = f"{dest}.zip"
    print(f"  Downloading {name}...")
    os.makedirs(dest, exist_ok=True)
    urllib.request.urlretrieve(url, zip_path)
    print(f"  Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    os.remove(zip_path)
    return base


def load_corpus(base: str) -> list[tuple[str, str]]:
    """Load corpus from BEIR JSONL format."""
    corpus = []
    with open(f"{base}/corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            corpus.append((doc["_id"], text))
    return corpus


def load_queries(base: str) -> dict[str, str]:
    """Load queries from BEIR JSONL format."""
    queries = {}
    with open(f"{base}/queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    return queries


def load_qrels(base: str) -> list[tuple[str, str, int]]:
    """Load relevance judgments. Tries test.tsv first, falls back to dev."""
    for split in ["test.tsv", "dev.tsv"]:
        path = f"{base}/qrels/{split}"
        if os.path.exists(path):
            qrels = []
            with open(path) as f:
                f.readline()  # header
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        qrels.append((parts[0], parts[1], int(parts[2])))
            return qrels
    raise FileNotFoundError(f"No qrels found in {base}/qrels/")


def get_embedder(nomic_key: str = None):
    """Get an embedder — Nomic API if key provided, local ONNX otherwise."""
    if nomic_key:
        # Import from flex's nomic_embed (same code lives in flexvec/onnx/)
        sys.path.insert(0, os.path.dirname(__file__) + '/..')
        from flexvec.onnx.nomic_embed import NomicEmbedder
        embedder = NomicEmbedder(nomic_key)
        err = embedder.validate()
        if err:
            print(f"  Nomic API key invalid: {err}")
            sys.exit(1)
        print(f"  Using Nomic API (fast)")
        return embedder
    else:
        from flexvec.onnx.embed import get_model
        print(f"  Using local ONNX (slow on CPU)")
        return get_model()


def build_db(name: str, embedder=None):
    """Download, embed, and build SQLite DB for one BEIR dataset."""
    print(f"\n{'='*60}")
    print(f"Building {name} ({DATASETS[name]['domain']}, {DATASETS[name]['docs']} docs)")
    print(f"{'='*60}")

    base = download_dataset(name)
    db_path = f"{DEST_ROOT}/{name}.db"

    corpus = load_corpus(base)
    print(f"  Corpus: {len(corpus)} docs")

    texts = [t for _, t in corpus]

    print(f"  Embedding {len(texts)} docs...")
    t0 = time.time()
    embeddings = embedder.encode(texts, prefix='search_document: ', matryoshka_dim=128)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — shape {embeddings.shape}")

    # Build SQLite
    if os.path.exists(db_path):
        os.remove(db_path)

    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE _raw_chunks (id TEXT PRIMARY KEY, content TEXT, embedding BLOB, timestamp REAL)")
    db.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(content, content='_raw_chunks', content_rowid='rowid')")

    now = time.time()
    day = 86400
    for i, (doc_id, text) in enumerate(corpus):
        days_ago = 90 * (1 - i / len(corpus))
        ts = now - (days_ago * day)
        emb_blob = embeddings[i].astype(np.float32).tobytes()
        db.execute("INSERT INTO _raw_chunks VALUES (?, ?, ?, ?)", (doc_id, text, emb_blob, ts))

    db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")

    # Queries
    queries = load_queries(base)
    db.execute("CREATE TABLE queries (id TEXT PRIMARY KEY, text TEXT)")
    for qid, text in queries.items():
        db.execute("INSERT INTO queries VALUES (?, ?)", (qid, text))

    # Qrels
    qrels = load_qrels(base)
    db.execute("CREATE TABLE qrels (query_id TEXT, doc_id TEXT, relevance INTEGER)")
    for qid, did, rel in qrels:
        db.execute("INSERT INTO qrels VALUES (?, ?, ?)", (qid, did, rel))

    db.commit()
    db.close()

    size_mb = os.path.getsize(db_path) / 1024 / 1024
    print(f"  DB: {db_path} ({size_mb:.1f}MB)")


def main():
    # Parse --nomic-key flag
    nomic_key = os.environ.get("NOMIC_API_KEY", "").strip() or None
    args = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--nomic-key" and i + 1 < len(sys.argv):
            nomic_key = sys.argv[i + 1]
            i += 2
        else:
            args.append(sys.argv[i])
            i += 1

    names = args if args else list(DATASETS.keys())
    for name in names:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}. Available: {', '.join(DATASETS.keys())}")
            sys.exit(1)

    embedder = get_embedder(nomic_key)

    for name in names:
        build_db(name, embedder=embedder)

    print(f"\nAll done. Run tests:")
    print(f"  pytest tests/test_tokens_beir.py -v -s")


if __name__ == "__main__":
    main()
