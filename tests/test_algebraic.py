"""
Algebraic correctness tests for flexvec modulation tokens.

Unlike behavioral tests (which check that rankings change), these tests verify
that output scores equal the mathematical formula to floating-point precision.
Each modulation is a deterministic arithmetic operation on the score array;
these tests prove the implementation matches the specification.

Requires: python tests/build_beir_db.py  (one-time, builds all 4 datasets)
Run with: pytest tests/test_algebraic.py -v -s
"""
import math
import os
import sqlite3
import time

import numpy as np
import pytest

from flexvec import VectorCache, register_vec_ops, execute
from flexvec.score import parse_modifiers


DB_ROOT = "/tmp/beir_flexvec"
DATASETS = ["scifact", "nfcorpus", "fiqa", "scidocs"]
TOLERANCE = 1e-3  # float32 precision


# ── Helpers ───────────────────────────────────────────────────────

def run_query(db_conn, query_text, modifier='', limit=10):
    """Run vec_ops query, return list of (id, score) tuples."""
    mod_clause = f", '{modifier}'" if modifier else ""
    sql = f"""
        SELECT v.id, v.score
        FROM vec_ops('_raw_chunks', '{query_text.replace("'", "''")}'
            {mod_clause}) v
        ORDER BY v.score DESC LIMIT {limit}
    """
    rows = execute(db_conn, sql)
    if isinstance(rows, dict):
        return []
    return [(r['id'], r['score']) for r in rows]


# ── Per-dataset fixture ───────────────────────────────────────────

_dataset_cache = {}


def _load_dataset(name):
    db_path = f"{DB_ROOT}/{name}.db"
    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    cache = VectorCache()
    cache.load_from_db(conn, '_raw_chunks', 'embedding', 'id')
    cache.load_columns(conn, '_raw_chunks', 'id')

    try:
        from flexvec.onnx.embed import get_model
        model = get_model()
        embed_fn = lambda text: model.encode([text], prefix='search_query: ', matryoshka_dim=128)
        embed_doc_fn = lambda text: model.encode([text], prefix='search_document: ', matryoshka_dim=128)
    except Exception:
        pytest.skip("No embedder available — install flexvec[embed]")

    register_vec_ops(conn, {'_raw_chunks': cache}, embed_fn, embed_doc_fn=embed_doc_fn)

    rows = conn.execute("SELECT query_id, doc_id, relevance FROM qrels").fetchall()
    qrels = {}
    for r in rows:
        qid = r[0]
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][r[1]] = r[2]

    query_rows = conn.execute("""
        SELECT q.id, q.text FROM queries q
        WHERE q.id IN (SELECT DISTINCT query_id FROM qrels)
        LIMIT 30
    """).fetchall()
    queries = [(r[0], r[1]) for r in query_rows]

    return conn, cache, qrels, queries


def get_dataset(name):
    if name not in _dataset_cache:
        _dataset_cache[name] = _load_dataset(name)
    return _dataset_cache[name]


def _skip_if_missing(name):
    if not os.path.exists(f"{DB_ROOT}/{name}.db"):
        pytest.skip(f"{name} DB not found. Run: python tests/build_beir_db.py {name}")


def _get_embedder():
    from flexvec.onnx.embed import get_model
    model = get_model()
    def embed(text):
        vec = model.encode([text], prefix='search_query: ', matryoshka_dim=128)[0].astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
    return embed


def _get_doc_embedder():
    from flexvec.onnx.embed import get_model
    model = get_model()
    def embed(text):
        vec = model.encode([text], prefix='search_document: ', matryoshka_dim=128)[0].astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
    return embed


# ── Algebraic correctness tests ──────────────────────────────────

@pytest.mark.parametrize("dataset", DATASETS)
class TestSuppressFormula:
    """suppress score = baseline_score - 0.5 * similarity_to_suppression_text

    Verifies algebraic identity: scores -= w * (M @ embed(X))
    """

    def test_suppress_scores_match_formula(self, dataset):
        _skip_if_missing(dataset)
        conn, cache, qrels, queries = get_dataset(dataset)
        embed = _get_embedder()
        suppress_text = "clinical trial drug treatment"

        total_checked = 0
        total_mismatches = 0

        for qid, qtext in queries[:5]:
            q_vec = embed(qtext)
            n_vec = embed(suppress_text)

            # Compute expected scores via formula
            baseline_scores = cache.matrix @ q_vec
            neg_scores = cache.matrix @ n_vec
            expected = baseline_scores - 0.5 * neg_scores

            # Get actual suppress results
            suppressed = run_query(conn, qtext,
                modifier=f'suppress:{suppress_text}', limit=20)

            for rid, rscore in suppressed:
                idx = cache._id_to_idx.get(rid)
                if idx is not None:
                    total_checked += 1
                    diff = abs(expected[idx] - rscore)
                    if diff > TOLERANCE:
                        total_mismatches += 1
                        print(f"  MISMATCH {rid}: expected={expected[idx]:.6f} actual={rscore:.6f} diff={diff:.6f}")

        print(f"\n[{dataset}] suppress: checked {total_checked} scores, {total_mismatches} mismatches")
        assert total_checked > 0, f"[{dataset}] No scores checked"
        assert total_mismatches == 0, (
            f"[{dataset}] {total_mismatches}/{total_checked} scores did not match suppress formula"
        )

    def test_multi_suppress_scores_match_formula(self, dataset):
        """Multiple suppress tokens compose additively: each subtracts independently."""
        _skip_if_missing(dataset)
        conn, cache, qrels, queries = get_dataset(dataset)
        embed = _get_embedder()
        suppress_a = "clinical trial drug"
        suppress_b = "patient diagnosis treatment"

        total_checked = 0
        total_mismatches = 0

        for qid, qtext in queries[:3]:
            q_vec = embed(qtext)
            n_vec_a = embed(suppress_a)
            n_vec_b = embed(suppress_b)

            baseline_scores = cache.matrix @ q_vec
            neg_scores_a = cache.matrix @ n_vec_a
            neg_scores_b = cache.matrix @ n_vec_b
            expected = baseline_scores - 0.5 * neg_scores_a - 0.5 * neg_scores_b

            suppressed = run_query(conn, qtext,
                modifier=f'suppress:{suppress_a} suppress:{suppress_b}', limit=20)

            for rid, rscore in suppressed:
                idx = cache._id_to_idx.get(rid)
                if idx is not None:
                    total_checked += 1
                    diff = abs(expected[idx] - rscore)
                    if diff > TOLERANCE:
                        total_mismatches += 1

        print(f"\n[{dataset}] multi-suppress: checked {total_checked}, {total_mismatches} mismatches")
        assert total_checked > 0
        assert total_mismatches == 0, (
            f"[{dataset}] {total_mismatches}/{total_checked} scores did not match multi-suppress formula"
        )


@pytest.mark.parametrize("dataset", DATASETS)
class TestTrajectoryFormula:
    """from/to score = 0.5 * baseline + 0.5 * (M @ (embed(B) - embed(A)))

    Verifies algebraic identity: scores = 0.5 * sim + 0.5 * traj_scores
    """

    def test_trajectory_scores_match_formula(self, dataset):
        _skip_if_missing(dataset)
        conn, cache, qrels, queries = get_dataset(dataset)
        embed = _get_embedder()
        embed_doc = _get_doc_embedder()
        from_text = "basic observation"
        to_text = "clinical treatment application"

        total_checked = 0
        total_mismatches = 0

        for qid, qtext in queries[:5]:
            q_vec = embed(qtext)
            # Trajectory uses document embedder, not query embedder
            from_vec = embed_doc(from_text)
            to_vec = embed_doc(to_text)

            direction = to_vec - from_vec
            # Direction is L2-normalized in the implementation
            d_norm = np.linalg.norm(direction)
            if d_norm > 0:
                direction = direction / d_norm
            baseline_scores = cache.matrix @ q_vec
            traj_scores = cache.matrix @ direction
            expected = 0.5 * baseline_scores + 0.5 * traj_scores

            trajectory = run_query(conn, qtext,
                modifier=f'from:{from_text} to:{to_text}', limit=20)

            for rid, rscore in trajectory:
                idx = cache._id_to_idx.get(rid)
                if idx is not None:
                    total_checked += 1
                    diff = abs(expected[idx] - rscore)
                    if diff > TOLERANCE:
                        total_mismatches += 1
                        print(f"  MISMATCH {rid}: expected={expected[idx]:.6f} actual={rscore:.6f}")

        print(f"\n[{dataset}] trajectory: checked {total_checked}, {total_mismatches} mismatches")
        assert total_checked > 0
        assert total_mismatches == 0, (
            f"[{dataset}] {total_mismatches}/{total_checked} scores did not match trajectory formula"
        )


@pytest.mark.parametrize("dataset", DATASETS)
class TestDecayFormula:
    """decay score = baseline * (1 / (1 + days_ago / half_life))

    Verifies algebraic identity: scores *= 1.0 / (1.0 + days_ago / half_life)

    Note: BEIR timestamps are synthetic (90-day uniform spread), so this tests
    the mechanism, not real temporal ranking.
    """

    def test_decay_scores_match_formula(self, dataset):
        _skip_if_missing(dataset)
        conn, cache, qrels, queries = get_dataset(dataset)
        embed = _get_embedder()
        half_life = 7.0

        # Load all timestamps
        ts_rows = conn.execute("SELECT id, timestamp FROM _raw_chunks").fetchall()
        ts_map = {r[0]: r[1] for r in ts_rows}
        now = time.time()

        total_checked = 0
        total_mismatches = 0

        for qid, qtext in queries[:5]:
            q_vec = embed(qtext)
            baseline_scores = cache.matrix @ q_vec

            decayed = run_query(conn, qtext, modifier='decay:7', limit=20)

            for rid, rscore in decayed:
                idx = cache._id_to_idx.get(rid)
                ts = ts_map.get(rid)
                if idx is not None and ts is not None:
                    days_ago = max((now - ts) / 86400.0, 0.0)
                    decay_factor = 1.0 / (1.0 + days_ago / half_life)
                    expected = baseline_scores[idx] * decay_factor
                    total_checked += 1
                    diff = abs(expected - rscore)
                    if diff > TOLERANCE:
                        total_mismatches += 1
                        print(f"  MISMATCH {rid}: expected={expected:.6f} actual={rscore:.6f} days_ago={days_ago:.1f}")

        print(f"\n[{dataset}] decay: checked {total_checked}, {total_mismatches} mismatches")
        assert total_checked > 0
        assert total_mismatches == 0, (
            f"[{dataset}] {total_mismatches}/{total_checked} scores did not match decay formula"
        )


@pytest.mark.parametrize("dataset", DATASETS)
class TestCentroidFormula:
    """centroid shifts the query vector: q = α*q + (1-α)*centroid; q /= ||q||

    Verifies that centroid search produces scores equal to
    (shifted_query @ candidate_matrix) where shifted_query is the
    blended and renormalized query vector.
    """

    def test_centroid_scores_match_formula(self, dataset):
        _skip_if_missing(dataset)
        conn, cache, qrels, queries = get_dataset(dataset)
        embed = _get_embedder()
        alpha = 0.5

        total_checked = 0
        total_mismatches = 0

        for qid, qtext in queries[:5]:
            # Get baseline results to pick example IDs
            baseline = run_query(conn, qtext, limit=10)
            if len(baseline) < 6:
                continue

            example_ids = [baseline[i][0] for i in range(3, min(6, len(baseline)))]
            example_vecs = cache.get_vectors(example_ids)
            if len(example_vecs) == 0:
                continue

            # Compute shifted query
            q_vec = embed(qtext)
            centroid = example_vecs.mean(axis=0).astype(np.float32)
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid /= c_norm

            shifted = alpha * q_vec + (1 - alpha) * centroid
            s_norm = np.linalg.norm(shifted)
            if s_norm > 0:
                shifted /= s_norm

            expected_scores = cache.matrix @ shifted

            # Get actual centroid results
            centroid_results = run_query(conn, qtext,
                modifier=f"like:{','.join(example_ids)}", limit=20)

            for rid, rscore in centroid_results:
                idx = cache._id_to_idx.get(rid)
                if idx is not None:
                    total_checked += 1
                    diff = abs(expected_scores[idx] - rscore)
                    if diff > TOLERANCE:
                        total_mismatches += 1
                        print(f"  MISMATCH {rid}: expected={expected_scores[idx]:.6f} actual={rscore:.6f}")

        print(f"\n[{dataset}] centroid: checked {total_checked}, {total_mismatches} mismatches")
        assert total_checked > 0
        assert total_mismatches == 0, (
            f"[{dataset}] {total_mismatches}/{total_checked} scores did not match centroid formula"
        )
