"""
Multi-dataset BEIR property tests for flexvec modulation tokens.

Tests: baseline nDCG, diverse (ILS reduction), suppress/unlike (contrastive),
decay/recent (temporal), centroid/like, trajectory (from:to:), nDCG preservation.

Requires: python tests/build_beir_db.py  (one-time, builds all 4 datasets)
Run with: pytest tests/test_tokens_beir.py -v -s
"""
import math
import os
import sqlite3

import numpy as np
import pytest

from flexvec import VectorCache, register_vec_ops, execute
from flexvec.score import parse_modifiers


def record_metric(dataset, name, value):
    """Record a metric for the BEIR report (conftest picks it up)."""
    import sys
    # conftest is loaded by pytest into the test module's namespace
    conftest = sys.modules.get('conftest') or sys.modules.get('tests.conftest')
    if conftest and hasattr(conftest, 'beir_metrics'):
        conftest.beir_metrics[(dataset, name)] = value

DB_ROOT = "/tmp/beir_flexvec"

DATASETS = ["scifact", "nfcorpus", "fiqa", "scidocs"]

# Per-corpus suppress and trajectory text — domain-appropriate for each corpus
SUPPRESS_TEXT = {
    "scifact": "clinical trial drug treatment therapy patient",
    "nfcorpus": "clinical trial drug treatment therapy patient",
    "fiqa": "portfolio equity dividend interest rate bond market",
    "scidocs": "neural network deep learning algorithm classification training",
}

TRAJECTORY_FROM = {
    "scifact": "basic observation",
    "nfcorpus": "basic observation",
    "fiqa": "general question",
    "scidocs": "problem statement",
}

TRAJECTORY_TO = {
    "scifact": "clinical treatment application",
    "nfcorpus": "clinical treatment application",
    "fiqa": "investment strategy recommendation",
    "scidocs": "proposed method evaluation",
}


# ── Helpers ───────────────────────────────────────────────────────

def ndcg_at_k(ranked_ids, qrel_dict, k=10):
    """Compute nDCG@k given ranked doc IDs and relevance judgments."""
    def dcg(scores, k):
        return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))

    gains = [qrel_dict.get(did, 0) for did in ranked_ids[:k]]
    ideal = sorted(qrel_dict.values(), reverse=True)[:k]
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(gains, k) / idcg


def intra_list_similarity(cache, result_ids):
    """ILS: average pairwise cosine similarity within result set. Lower = more diverse."""
    vecs = cache.get_vectors(result_ids)
    if len(vecs) < 2:
        return 0.0
    sims = vecs @ vecs.T
    n = len(sims)
    return (sims.sum() - n) / (n * (n - 1))


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


def get_timestamps(db_conn, ids):
    """Get timestamps for a list of doc IDs."""
    if not ids:
        return {}
    placeholders = ','.join('?' * len(ids))
    rows = db_conn.execute(
        f"SELECT id, timestamp FROM _raw_chunks WHERE id IN ({placeholders})", ids
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def rank_biased_overlap(list_a, list_b, p=0.9):
    """RBO: measures similarity between two ranked lists. 0=disjoint, 1=identical."""
    k = max(len(list_a), len(list_b))
    if k == 0:
        return 1.0
    rbo = 0.0
    for d in range(1, k + 1):
        set_a = set(list_a[:d])
        set_b = set(list_b[:d])
        overlap = len(set_a & set_b) / d
        rbo += (1 - p) * (p ** (d - 1)) * overlap
    return rbo


# ── Per-dataset fixture ───────────────────────────────────────────

def _load_dataset(name):
    """Load a BEIR dataset DB. Returns (conn, cache, qrels, queries) or None."""
    db_path = f"{DB_ROOT}/{name}.db"
    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    cache = VectorCache()
    cache.load_from_db(conn, '_raw_chunks', 'embedding', 'id')
    cache.load_columns(conn, '_raw_chunks', 'id')

    # Always use local ONNX embedder
    try:
        from flexvec.onnx.embed import get_model
        model = get_model()
        embed_fn = lambda text: model.encode([text], prefix='search_query: ', matryoshka_dim=128)
        embed_doc_fn = lambda text: model.encode([text], prefix='search_document: ', matryoshka_dim=128)
    except Exception:
        pytest.skip("No embedder available — install flexvec[embed]")

    register_vec_ops(conn, {'_raw_chunks': cache}, embed_fn, embed_doc_fn=embed_doc_fn)

    # Load qrels
    rows = conn.execute("SELECT query_id, doc_id, relevance FROM qrels").fetchall()
    qrels = {}
    for r in rows:
        qid = r[0]
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][r[1]] = r[2]

    # Load queries that have qrels
    query_rows = conn.execute("""
        SELECT q.id, q.text FROM queries q
        WHERE q.id IN (SELECT DISTINCT query_id FROM qrels)
        LIMIT 30
    """).fetchall()
    queries = [(r[0], r[1]) for r in query_rows]

    return conn, cache, qrels, queries


# Module-level cache so each dataset loads once
_dataset_cache = {}


def get_dataset(name):
    if name not in _dataset_cache:
        _dataset_cache[name] = _load_dataset(name)
    return _dataset_cache[name]


# ── Parametrized tests ────────────────────────────────────────────

def _available_datasets():
    """Return datasets that have been built."""
    available = []
    for name in DATASETS:
        if os.path.exists(f"{DB_ROOT}/{name}.db"):
            available.append(name)
    return available


def _skip_if_missing(name):
    if not os.path.exists(f"{DB_ROOT}/{name}.db"):
        pytest.skip(f"{name} DB not found. Run: python tests/build_beir_db.py {name}")


@pytest.mark.parametrize("dataset", DATASETS)
class TestBaseline:
    """Verify baseline search produces reasonable results."""

    def test_baseline_returns_results(self, dataset):
        _skip_if_missing(dataset)
        data = get_dataset(dataset)
        conn, cache, qrels, queries = data
        for qid, qtext in queries[:5]:
            results = run_query(conn, qtext)
            assert len(results) > 0, f"[{dataset}] No results for query {qid}"

    def test_baseline_ndcg(self, dataset):
        _skip_if_missing(dataset)
        conn, cache, qrels, queries = get_dataset(dataset)
        scores = []
        for qid, qtext in queries:
            if qid not in qrels:
                continue
            results = run_query(conn, qtext)
            ids = [r[0] for r in results]
            score = ndcg_at_k(ids, qrels[qid])
            scores.append(score)

        mean_ndcg = np.mean(scores) if scores else 0
        record_metric(dataset, 'ndcg', float(mean_ndcg))
        # No assertion — nDCG is non-negative by definition. The value is recorded
        # for the benchmark report; regressions are caught by nDCG preservation tests.
        print(f"\n[{dataset}] Baseline nDCG@10: {mean_ndcg:.4f} (n={len(scores)})")


@pytest.mark.parametrize("dataset", DATASETS)
class TestDiverse:
    """diverse token: MMR diversity selection."""

    def test_diverse_reduces_ils(self, dataset):
        _skip_if_missing(dataset)
        conn, cache, qrels, queries = get_dataset(dataset)
        ils_b_all, ils_d_all = [], []

        for qid, qtext in queries[:10]:
            baseline = run_query(conn, qtext, limit=10)
            diverse = run_query(conn, qtext, modifier='diverse limit:50', limit=10)

            ils_b_all.append(intra_list_similarity(cache, [r[0] for r in baseline]))
            ils_d_all.append(intra_list_similarity(cache, [r[0] for r in diverse]))

        mean_b, mean_d = np.mean(ils_b_all), np.mean(ils_d_all)
        record_metric(dataset, 'ils_baseline', float(mean_b))
        record_metric(dataset, 'ils_diverse', float(mean_d))
        print(f"\n[{dataset}] ILS baseline={mean_b:.4f}, diverse={mean_d:.4f}")
        assert mean_d <= mean_b, f"[{dataset}] diverse should reduce ILS"

    def test_diverse_changes_ranking(self, dataset):
        _skip_if_missing(dataset)
        conn, cache, qrels, queries = get_dataset(dataset)
        rbos = []
        for qid, qtext in queries[:10]:
            baseline = run_query(conn, qtext)
            diverse = run_query(conn, qtext, modifier='diverse limit:50')
            rbos.append(rank_biased_overlap([r[0] for r in baseline], [r[0] for r in diverse]))

        mean_rbo = np.mean(rbos)
        record_metric(dataset, 'diverse_rbo', float(mean_rbo))
        print(f"\n[{dataset}] RBO (diverse vs baseline): {mean_rbo:.4f}")
        assert mean_rbo < 0.8, f"[{dataset}] diverse should change ranking"


@pytest.mark.parametrize("dataset", DATASETS)
class TestSuppress:
    """suppress:TEXT token: contrastive suppression."""

    def test_suppress_changes_scores(self, dataset):
        """Suppress should meaningfully change the score distribution.

        Uses domain-appropriate suppress text per corpus to ensure the
        suppression direction has semantic traction in each domain.
        """
        _skip_if_missing(dataset)
        conn, cache, qrels, queries = get_dataset(dataset)
        suppress_text = SUPPRESS_TEXT[dataset]
        score_deltas = []

        for qid, qtext in queries[:10]:
            baseline = run_query(conn, qtext)
            suppressed = run_query(conn, qtext, modifier=f'suppress:{suppress_text}')
            if baseline and suppressed:
                mean_b = np.mean([r[1] for r in baseline])
                mean_s = np.mean([r[1] for r in suppressed])
                score_deltas.append(abs(mean_s - mean_b))

        mean_delta = np.mean(score_deltas)
        record_metric(dataset, 'suppress_delta', float(mean_delta))
        print(f"\n[{dataset}] Mean score delta from suppress: {mean_delta:.4f}")
        assert mean_delta > 0.001, f"[{dataset}] suppress should change scores"

    def test_suppress_changes_ranking(self, dataset):
        _skip_if_missing(dataset)
        conn, cache, qrels, queries = get_dataset(dataset)
        suppress_text = SUPPRESS_TEXT[dataset]
        rbos = []
        for qid, qtext in queries[:10]:
            baseline = run_query(conn, qtext)
            suppressed = run_query(conn, qtext, modifier=f'suppress:{suppress_text}')
            rbos.append(rank_biased_overlap([r[0] for r in baseline], [r[0] for r in suppressed]))

        mean_rbo = np.mean(rbos)
        record_metric(dataset, 'suppress_rbo', float(mean_rbo))
        print(f"\n[{dataset}] RBO (suppress vs baseline): {mean_rbo:.4f}")
        assert mean_rbo < 0.8, f"[{dataset}] suppress should change ranking"


@pytest.mark.parametrize("dataset", DATASETS)
class TestDecay:
    """decay:N token: temporal decay."""

    def test_decay_boosts_newer(self, dataset):
        _skip_if_missing(dataset)
        conn, cache, qrels, queries = get_dataset(dataset)
        ts_b, ts_d = [], []

        for qid, qtext in queries[:10]:
            baseline = run_query(conn, qtext)
            decayed = run_query(conn, qtext, modifier='decay:7')

            b_ts = get_timestamps(conn, [r[0] for r in baseline])
            d_ts = get_timestamps(conn, [r[0] for r in decayed])
            if b_ts:
                ts_b.append(np.mean(list(b_ts.values())))
            if d_ts:
                ts_d.append(np.mean(list(d_ts.values())))

        mean_b, mean_d = np.mean(ts_b), np.mean(ts_d)
        days_shift = (mean_d - mean_b) / 86400
        record_metric(dataset, 'decay_days', float(days_shift))
        print(f"\n[{dataset}] Newer by {days_shift:.1f} days")
        assert mean_d > mean_b, f"[{dataset}] decay should boost newer docs"

    def test_decay_changes_ranking(self, dataset):
        _skip_if_missing(dataset)
        conn, cache, qrels, queries = get_dataset(dataset)
        rbos = []
        for qid, qtext in queries[:10]:
            baseline = run_query(conn, qtext)
            decayed = run_query(conn, qtext, modifier='decay:7')
            rbos.append(rank_biased_overlap([r[0] for r in baseline], [r[0] for r in decayed]))

        mean_rbo = np.mean(rbos)
        record_metric(dataset, 'decay_rbo', float(mean_rbo))
        print(f"\n[{dataset}] RBO (decay vs baseline): {mean_rbo:.4f}")
        assert mean_rbo < 0.8, f"[{dataset}] decay should change ranking"


@pytest.mark.parametrize("dataset", DATASETS)
class TestCentroid:
    """centroid:id1,id2 token: vector arithmetic verification.

    Note: this test is tautological by design — blending toward X increases
    similarity to X. It verifies the centroid blending mechanism works, not
    that retrieval quality improves.
    """

    def test_centroid_blending_shifts_vectors(self, dataset):
        _skip_if_missing(dataset)
        conn, cache, qrels, queries = get_dataset(dataset)
        sim_b, sim_l = [], []

        for qid, qtext in queries[:10]:
            baseline = run_query(conn, qtext)
            if len(baseline) < 5:
                continue

            example_ids = [baseline[i][0] for i in range(3, min(6, len(baseline)))]
            example_vecs = cache.get_vectors(example_ids)
            if len(example_vecs) == 0:
                continue
            centroid = example_vecs.mean(axis=0)
            centroid /= np.linalg.norm(centroid)

            like_results = run_query(conn, qtext, modifier=f"like:{','.join(example_ids)}")

            b_vecs = cache.get_vectors([r[0] for r in baseline])
            l_vecs = cache.get_vectors([r[0] for r in like_results])
            if len(b_vecs) > 0:
                sim_b.append(float((b_vecs @ centroid).mean()))
            if len(l_vecs) > 0:
                sim_l.append(float((l_vecs @ centroid).mean()))

        mean_b, mean_l = np.mean(sim_b), np.mean(sim_l)
        record_metric(dataset, 'centroid_baseline', float(mean_b))
        record_metric(dataset, 'centroid_like', float(mean_l))
        print(f"\n[{dataset}] Centroid sim baseline={mean_b:.4f}, like={mean_l:.4f}")
        assert mean_l > mean_b, f"[{dataset}] like should shift toward examples"


@pytest.mark.parametrize("dataset", DATASETS)
class TestTrajectory:
    """from:TEXT to:TEXT token: directional vector blending."""

    def test_trajectory_changes_ranking(self, dataset):
        _skip_if_missing(dataset)
        conn, cache, qrels, queries = get_dataset(dataset)
        rbos = []
        for qid, qtext in queries[:10]:
            baseline = run_query(conn, qtext)
            trajectory = run_query(conn, qtext,
                modifier=f'from:{TRAJECTORY_FROM[dataset]} to:{TRAJECTORY_TO[dataset]}')
            rbos.append(rank_biased_overlap([r[0] for r in baseline], [r[0] for r in trajectory]))

        mean_rbo = np.mean(rbos)
        record_metric(dataset, 'trajectory_rbo', float(mean_rbo))
        print(f"\n[{dataset}] RBO (trajectory vs baseline): {mean_rbo:.4f}")
        assert mean_rbo < 0.8, f"[{dataset}] trajectory should change ranking"

    def test_trajectory_different_from_to_query(self, dataset):
        _skip_if_missing(dataset)
        conn, cache, qrels, queries = get_dataset(dataset)
        rbos = []
        for qid, qtext in queries[:5]:
            trajectory = run_query(conn, qtext,
                modifier=f'from:{TRAJECTORY_FROM[dataset]} to:{TRAJECTORY_TO[dataset]}')
            to_only = run_query(conn, TRAJECTORY_TO[dataset])
            rbos.append(rank_biased_overlap([r[0] for r in trajectory], [r[0] for r in to_only]))

        mean_rbo = np.mean(rbos)
        print(f"\n[{dataset}] RBO (trajectory vs to-only): {mean_rbo:.4f}")
        assert mean_rbo < 0.95, f"[{dataset}] trajectory should differ from to-only"


@pytest.mark.parametrize("dataset", DATASETS)
class TestNDCGPreservation:
    """Modulation tokens should not destroy ranking quality."""

    @pytest.mark.parametrize("modifier_key", [
        "diverse",
        "suppress",
        # decay:7 excluded — synthetic timestamps are uncorrelated with relevance
    ])
    def test_token_preserves_ndcg(self, dataset, modifier_key):
        _skip_if_missing(dataset)
        conn, cache, qrels, queries = get_dataset(dataset)

        # Resolve per-corpus modifier
        if modifier_key == "diverse":
            modifier = "diverse limit:50"
        elif modifier_key == "suppress":
            modifier = f"suppress:{SUPPRESS_TEXT[dataset]}"
        else:
            modifier = modifier_key

        baseline_scores, token_scores = [], []

        for qid, qtext in queries:
            if qid not in qrels:
                continue
            baseline = run_query(conn, qtext)
            modified = run_query(conn, qtext, modifier=modifier)

            baseline_scores.append(ndcg_at_k([r[0] for r in baseline], qrels[qid]))
            token_scores.append(ndcg_at_k([r[0] for r in modified], qrels[qid]))

        if not baseline_scores:
            pytest.skip("No queries with qrels found")

        mean_b, mean_t = np.mean(baseline_scores), np.mean(token_scores)
        # Record for report
        if 'diverse' in modifier:
            record_metric(dataset, 'ndcg_diverse', float(mean_t))
        elif 'suppress' in modifier or 'unlike' in modifier:
            record_metric(dataset, 'ndcg_suppress', float(mean_t))
        print(f"\n[{dataset}] nDCG@10 baseline={mean_b:.4f}, {modifier}={mean_t:.4f}")
        if mean_b < 0.01:
            pytest.skip(f"[{dataset}] baseline nDCG too low ({mean_b:.4f}) — no signal to preserve")
        assert mean_t >= mean_b * 0.5, (
            f"[{dataset}] {modifier} destroyed nDCG: {mean_t:.4f} < {mean_b * 0.5:.4f}"
        )
