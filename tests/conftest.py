"""flexvec test fixtures and BEIR report generation."""
import struct
import os
import time
from datetime import datetime
from pathlib import Path

import pytest

EMBED_DIM = 128
REPORT_DIR = Path(__file__).parent / "benchmarks"

# Shared metric store — tests write here, report reads from here
beir_metrics = {}  # {(dataset, metric_name): value}


def _make_embedding(dim=EMBED_DIM):
    """Create a fake float32 embedding BLOB."""
    return struct.pack(f'{dim}f', *([0.1] * dim))


def record_metric(dataset, name, value):
    """Record a BEIR metric for the report."""
    beir_metrics[(dataset, name)] = value


# ── Report generation ─────────────────────────────────────────────

_start_time = None
_beir_pass = 0
_beir_fail = 0
_beir_skip = 0


def pytest_configure(config):
    global _start_time
    _start_time = time.time()


def pytest_runtest_logreport(report):
    global _beir_pass, _beir_fail, _beir_skip
    if report.when == "call" and "test_tokens_beir" in report.nodeid:
        if report.outcome == "passed":
            _beir_pass += 1
        elif report.outcome == "failed":
            _beir_fail += 1
        elif report.outcome == "skipped":
            _beir_skip += 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not beir_metrics:
        return

    elapsed = time.time() - _start_time
    now = datetime.now()
    timestamp = now.strftime("%y%m%d-%H%M")

    datasets = ['scifact', 'nfcorpus', 'fiqa', 'scidocs']
    dataset_info = {
        'scifact': ('SciFact', '5,183', 'Science/claims'),
        'nfcorpus': ('NFCorpus', '3,633', 'Biomedical'),
        'fiqa': ('FiQA', '57,638', 'Financial QA'),
        'scidocs': ('SCIDOCS', '25,657', 'Citation prediction'),
    }

    def m(ds, key, default='—'):
        v = beir_metrics.get((ds, key))
        if v is None:
            return default
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = []
    lines.append(f"# BEIR Benchmark Results — {now.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(f"**{_beir_pass} passed, {_beir_fail} failed, {_beir_skip} skipped in {elapsed:.1f}s**")
    lines.append("")
    lines.append("Embedder: Nomic Embed v1.5, 128d Matryoshka, ONNX int8")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Dataset | Domain | Docs | nDCG@10 | diverse ILS | suppress RBO | decay shift | centroid sim |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds in datasets:
        if (ds, 'ndcg') not in beir_metrics:
            continue
        name, docs, domain = dataset_info[ds]
        ils = f"{m(ds,'ils_baseline')}→{m(ds,'ils_diverse')}"
        cent = f"{m(ds,'centroid_baseline')}→{m(ds,'centroid_like')}"
        decay = f"+{m(ds,'decay_days')}d"
        lines.append(f"| **{name}** | {domain} | {docs} | {m(ds,'ndcg')} | {ils} | {m(ds,'suppress_rbo')} | {decay} | {cent} |")

    # nDCG preservation
    lines.append("")
    lines.append("## nDCG Preservation")
    lines.append("")
    lines.append("| Dataset | Baseline | diverse | diverse % | suppress | suppress % |")
    lines.append("|---|---|---|---|---|---|")
    for ds in datasets:
        ndcg = beir_metrics.get((ds, 'ndcg'))
        div = beir_metrics.get((ds, 'ndcg_diverse'))
        sup = beir_metrics.get((ds, 'ndcg_suppress'))
        if ndcg is None:
            continue
        name = dataset_info[ds][0]
        div_pct = f"{div/ndcg*100:.0f}%" if ndcg > 0 and div is not None else "—"
        sup_pct = f"{sup/ndcg*100:.0f}%" if ndcg > 0 and sup is not None else "—"
        lines.append(f"| **{name}** | {m(ds,'ndcg')} | {m(ds,'ndcg_diverse')} | {div_pct} | {m(ds,'ndcg_suppress')} | {sup_pct} |")

    # All metrics
    lines.append("")
    lines.append("## All Metrics")
    lines.append("")
    lines.append("| Dataset | Metric | Value |")
    lines.append("|---|---|---|")
    for (ds, key), val in sorted(beir_metrics.items()):
        name = dataset_info.get(ds, (ds,))[0]
        if isinstance(val, float):
            lines.append(f"| {name} | {key} | {val:.4f} |")
        else:
            lines.append(f"| {name} | {key} | {val} |")

    report = "\n".join(lines)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{timestamp}_beir-results.md"
    report_path.write_text(report)
    terminalreporter.write_sep("=", f"BEIR report: {report_path}")
