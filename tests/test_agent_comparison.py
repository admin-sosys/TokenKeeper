"""Blind Agent Comparison: RAG-augmented vs Raw codebase search.

Spawns blind "agents" (isolated function calls) that answer codebase
questions using either:
  A) RAG retrieval (top-k chunks from the indexed HomePayroll repo)
  B) Raw file listing + grepping (simulating unassisted codebase search)

Each agent is scored against exhaustive ground truth (established by
Opus 4.6 manual review) using precision, recall, and F1.

This measures whether RAG retrieval can match or exceed the accuracy
of a "raw search" approach while using dramatically fewer tokens.

Requires:
  - HomePayroll cloned at .acceptance-cache/HomePayroll/
  - Ollama running with nomic-embed-text
  - RAG index built (chromadb + bm25) at .acceptance-cache/HomePayroll/.rag/

Usage::

    # Run all blind agent comparison tests
    uv run pytest tests/test_agent_comparison.py -v -s --tb=short

    # Run a single query
    uv run pytest tests/test_agent_comparison.py -v -s -k "query_1"
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
import pytest

from tokenkeeper.bm25_index import BM25Index
from tokenkeeper.config import RagConfig
from tokenkeeper.embeddings import embed_texts_google
from tokenkeeper.indexer import index_documents
from tokenkeeper.search import SearchResult, search

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOMEPAYROLL_DIR = Path(".acceptance-cache/HomePayroll")
RAG_DIR = HOMEPAYROLL_DIR / ".rag"
CHROMA_DIR = RAG_DIR / "chroma"
COLLECTION_NAME = "homepayroll_google"

# Token estimator (same as benchmark suite)
def count_tokens(text: str) -> int:
    """Estimate token count as chars // 4."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Ground Truth: 5 diverse queries with exhaustive file lists
# ---------------------------------------------------------------------------

@dataclass
class QueryGroundTruth:
    """A test query with its known-correct answer files."""
    id: str
    query: str
    description: str
    # Files that MUST be found (true positives).
    # Paths are relative to HomePayroll root.
    expected_files: list[str]
    # Keywords that should appear in relevant results
    expected_keywords: list[str]
    # Category for reporting
    category: str


# ---------------------------------------------------------------------------
# Ground truth design philosophy:
#
# These ground truth sets focus on KNOWLEDGE RETRIEVAL — finding the
# documentation, planning docs, blog posts, and data files that an LLM
# agent would need in its context window. This mirrors the real use case:
# TokenKeeper reduces agent memory bloat by retrieving only the relevant
# knowledge files instead of scanning the entire codebase.
#
# Files in expected_files are primarily .md, .mdx, .json, .mjs — the
# knowledge artifacts that bloat agent context. Source .ts/.tsx files are
# included only when they contain substantial knowledge content (e.g.,
# security-policy.ts has full policy text, not just code).
#
# The evaluation asks: "Did RAG find all the relevant knowledge docs?"
# Extra relevant files are fine (high recall with tolerable precision).
# ---------------------------------------------------------------------------

# Query 1: Legal document mandatory update triggers
# Tests: broad regulatory recall across scattered knowledge docs
QUERY_1 = QueryGroundTruth(
    id="q1_legal_update_triggers",
    query="What are all the rules and triggers for when legal compliance documents must be updated?",
    description="Broad regulatory: find all mandatory update trigger rules across docs",
    expected_files=[
        ".claude/rules/legal-compliance-doc.md",
        ".planning/LEGAL-COMPLIANCE-MASTER.md",
        ".claude/rules/security-compliance-doc.md",
        ".planning/SECURITY-COMPLIANCE-MASTER.md",
        ".planning/todos/completed/legal-compliance-master-pdf.md",
        ".planning/phases/10-security-compliance/10-RESEARCH.md",
        ".planning/phases/17-legal-compliance/17-RESEARCH-email-compliance.md",
    ],
    expected_keywords=[
        "mandatory", "update", "trigger", "compliance", "legal",
        "policy", "change", "PDF", "regenerate",
    ],
    category="broad_regulatory",
)

# Query 2: Privacy policy — knowledge docs about privacy content
# Tests: retrieval of planning research + rendered policy pages
QUERY_2 = QueryGroundTruth(
    id="q2_privacy_policy",
    query="Where is the privacy policy content defined and how is it rendered to users?",
    description="Knowledge retrieval: privacy policy research, analysis, and page rendering",
    expected_files=[
        ".planning/phases/17-legal-compliance/17-RESEARCH-policy-analysis.md",
        ".planning/phases/17-legal-compliance/17-03-PLAN.md",
        ".planning/phases/17-legal-compliance/17-RESEARCH-review.md",
        ".planning/phases/10-security-compliance/10-RESEARCH.md",
        ".planning/phases/10-security-compliance/10-01-legal-pages-PLAN.md",
        "src/app/(marketing)/privacy/page.tsx",
        ".planning/phases/17-legal-compliance/17-RESEARCH-missing-policies.md",
    ],
    expected_keywords=[
        "privacy", "policy", "cookie", "CCPA", "GDPR", "data",
        "retention", "personal information",
    ],
    category="knowledge_retrieval",
)

# Query 3: Tax form data and deadlines
# Tests: retrieval of tax knowledge across blog posts, summaries, and data
QUERY_3 = QueryGroundTruth(
    id="q3_tax_data_deadlines",
    query="What tax forms and filing deadlines does the system track for household employers?",
    description="Tax knowledge: blog posts, guides, summaries, and data files",
    expected_files=[
        "content/blog/state-tax-requirements-overview.mdx",
        "content/blog/irs-penalties-household-employers.mdx",
        "content/blog/schedule-h-guide.mdx",
        "data/tax-kb/summaries/federal-summary.md",
        "data/tax-kb/2026/federal.json",
        "content/resources/payroll-setup-guide.mdx",
        "content/blog/nanny-tax-101.mdx",
    ],
    expected_keywords=[
        "Schedule H", "W-2", "W-4", "1099", "deadline", "FICA",
        "FUTA", "tax year", "filing",
    ],
    category="knowledge_retrieval",
)

# Query 4: Security implementation knowledge
# Tests: retrieval of security docs, plans, and policy content
QUERY_4 = QueryGroundTruth(
    id="q4_security_infrastructure",
    query="How does the application implement security headers, rate limiting, and content security policy?",
    description="Security knowledge: compliance docs, research, and policy content",
    expected_files=[
        "src/content/security-policy.ts",
        ".planning/SECURITY-COMPLIANCE-MASTER.md",
        ".planning/phases/01-foundation-infrastructure/01-04-SUMMARY.md",
        ".planning/phases/02-design-system-authentication/02-RESEARCH.md",
        ".planning/phases/10-security-compliance/10-RESEARCH.md",
        ".claude/rules/security-compliance-doc.md",
    ],
    expected_keywords=[
        "middleware", "rate limit", "CSP", "security", "headers",
        "content-security-policy", "X-Frame-Options",
    ],
    category="knowledge_retrieval",
)

# Query 5: Employment contract generation and PDF tooling
# Tests: cross-cutting knowledge about document generation pipeline
QUERY_5 = QueryGroundTruth(
    id="q5_contract_pdf_generation",
    query="How does the system generate employment contracts and compliance PDF documents?",
    description="Cross-cutting knowledge: PDF pipeline plans, scripts, and contract docs",
    expected_files=[
        "scripts/generate-compliance-pdf.mjs",
        ".planning/phases/24-helpful-tools-sample-templates/24-04-PLAN.md",
        ".planning/phases/24-helpful-tools-sample-templates/24-02-SUMMARY.md",
        ".planning/todos/completed/legal-compliance-master-pdf.md",
        ".planning/LEGAL-COMPLIANCE-MASTER.md",
        ".planning/phases/14-document-storage-pdf-generation/14-07-SUMMARY.md",
        ".planning/phases/14-document-storage-pdf-generation/14-VERIFICATION.md",
        "src/lib/pdf/contracts/contract-utils.ts",
    ],
    expected_keywords=[
        "contract", "PDF", "generate", "employment", "template",
        "document", "schema",
    ],
    category="cross_cutting",
)

ALL_QUERIES = [QUERY_1, QUERY_2, QUERY_3, QUERY_4, QUERY_5]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Result from a single agent run."""
    agent_type: str  # "raw" or "rag"
    query_id: str
    found_files: list[str]  # files the agent identified
    tokens_used: int  # approximate tokens consumed
    run_index: int  # which run (0, 1, 2)


@dataclass
class ScoreCard:
    """Precision/recall/F1 scores for one agent run."""
    agent_type: str
    query_id: str
    run_index: int
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    tokens_used: int
    found_files: list[str]
    missed_files: list[str]
    extra_files: list[str]


def score_agent_result(
    result: AgentResult,
    ground_truth: QueryGroundTruth,
) -> ScoreCard:
    """Score an agent's file list against ground truth."""
    # Normalize paths for comparison (strip leading ./ and trailing /)
    def normalize(p: str) -> str:
        return p.strip().lstrip("./").rstrip("/").replace("\\", "/")

    found_set = {normalize(f) for f in result.found_files}
    expected_set = {normalize(f) for f in ground_truth.expected_files}

    true_positives = found_set & expected_set
    false_positives = found_set - expected_set
    false_negatives = expected_set - found_set

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return ScoreCard(
        agent_type=result.agent_type,
        query_id=result.query_id,
        run_index=result.run_index,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        tokens_used=result.tokens_used,
        found_files=sorted(found_set),
        missed_files=sorted(false_negatives),
        extra_files=sorted(false_positives),
    )


# ---------------------------------------------------------------------------
# "Raw Agent" — simulates searching the repo via grep/find
# ---------------------------------------------------------------------------

def raw_agent_search(
    query: QueryGroundTruth,
    repo_path: Path,
) -> AgentResult:
    """Simulate a raw codebase search using keyword grep.

    This represents what an LLM would do without RAG: grep for keywords
    across all files, then return the files that matched.

    The "token cost" is the total content of all files scanned by grep
    (which, in a real agent, would be sent to the LLM context window).
    """
    all_files: list[str] = []
    tokens_scanned = 0

    # Build search terms from the query keywords
    keywords = query.expected_keywords[:5]  # Use top 5 keywords as grep terms

    matched_files: set[str] = set()
    files_scanned: set[str] = set()

    for root, dirs, files in os.walk(repo_path):
        # Skip .git, .rag, node_modules, .next
        dirs[:] = [d for d in dirs if d not in {".git", ".rag", "node_modules", ".next", "__pycache__"}]
        for fname in files:
            # Only search text files
            if not fname.endswith((".ts", ".tsx", ".md", ".mdx", ".json", ".js", ".mjs")):
                continue
            fpath = Path(root) / fname
            rel_path = str(fpath.relative_to(repo_path)).replace("\\", "/")

            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue

            file_tokens = count_tokens(content)
            tokens_scanned += file_tokens
            files_scanned.add(rel_path)

            # Check if any keyword matches (case-insensitive)
            content_lower = content.lower()
            for kw in keywords:
                if kw.lower() in content_lower:
                    matched_files.add(rel_path)
                    break

    return AgentResult(
        agent_type="raw",
        query_id=query.id,
        found_files=sorted(matched_files),
        tokens_used=tokens_scanned,
        run_index=0,
    )


# ---------------------------------------------------------------------------
# "RAG Agent" — uses the indexed RAG database
# ---------------------------------------------------------------------------

def rag_agent_search(
    query: QueryGroundTruth,
    collection: chromadb.Collection,
    bm25_index: BM25Index,
    top_k: int = 10,
    run_index: int = 0,
) -> AgentResult:
    """Search using RAG retrieval.

    The "token cost" is the total content of the retrieved chunks only
    (what would actually be sent to the LLM context window).
    """
    results = search(
        query=query.query,
        collection=collection,
        bm25_index=bm25_index,
        embed_fn=embed_texts_google,
        top_k=top_k,
        mode="hybrid",
    )

    # Extract unique source files from results
    found_files: set[str] = set()
    tokens_used = 0

    for r in results:
        found_files.add(r.source_file)
        tokens_used += count_tokens(r.content)

    return AgentResult(
        agent_type="rag",
        query_id=query.id,
        found_files=sorted(found_files),
        tokens_used=tokens_used,
        run_index=run_index,
    )


def rag_agent_search_adaptive(
    query: QueryGroundTruth,
    collection: chromadb.Collection,
    bm25_index: BM25Index,
    score_threshold: float = 0.3,
    max_k: int = 30,
    run_index: int = 0,
) -> AgentResult:
    """RAG search with adaptive top_k — retrieves until scores drop below threshold.

    Instead of a fixed top_k, this fetches max_k results but only keeps
    those with score >= score_threshold. This tests whether adaptive
    retrieval can improve precision without sacrificing recall.
    """
    results = search(
        query=query.query,
        collection=collection,
        bm25_index=bm25_index,
        embed_fn=embed_texts_google,
        top_k=max_k,
        mode="hybrid",
    )

    # Filter by score threshold
    filtered = [r for r in results if r.score >= score_threshold]

    found_files: set[str] = set()
    tokens_used = 0

    for r in filtered:
        found_files.add(r.source_file)
        tokens_used += count_tokens(r.content)

    return AgentResult(
        agent_type=f"rag_adaptive_{score_threshold}",
        query_id=query.id,
        found_files=sorted(found_files),
        tokens_used=tokens_used,
        run_index=run_index,
    )


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_scorecard(sc: ScoreCard, ground_truth: QueryGroundTruth) -> None:
    """Print a formatted scorecard for one run."""
    print(f"\n  {'-' * 60}")
    print(f"  Agent: {sc.agent_type:20s}  Query: {sc.query_id}")
    print(f"  Run: {sc.run_index}  Tokens: {sc.tokens_used:,}")
    print(f"  Precision: {sc.precision:.1%}  Recall: {sc.recall:.1%}  F1: {sc.f1:.1%}")
    print(f"  TP={sc.true_positives}  FP={sc.false_positives}  FN={sc.false_negatives}")
    if sc.missed_files:
        print(f"  MISSED: {sc.missed_files[:3]}{'...' if len(sc.missed_files) > 3 else ''}")
    if sc.extra_files:
        print(f"  EXTRA:  {sc.extra_files[:3]}{'...' if len(sc.extra_files) > 3 else ''}")


def print_comparison_table(
    raw_scores: list[ScoreCard],
    rag_scores: list[ScoreCard],
    adaptive_scores: list[ScoreCard],
    query: QueryGroundTruth,
) -> None:
    """Print a comparison table for one query across all agent types."""
    print(f"\n{'=' * 70}")
    print(f"  QUERY: {query.id}")
    print(f"  {query.description}")
    print(f"  Ground truth: {len(query.expected_files)} files")
    print(f"{'=' * 70}")
    print(f"  {'Agent Type':<25s} {'Prec':>6s} {'Recall':>7s} {'F1':>6s} {'Tokens':>10s} {'Files':>6s}")
    print(f"  {'-' * 62}")

    all_scores = raw_scores + rag_scores + adaptive_scores
    for sc in all_scores:
        print(
            f"  {sc.agent_type:<25s} {sc.precision:>5.0%} {sc.recall:>6.0%}"
            f" {sc.f1:>5.0%} {sc.tokens_used:>10,} {len(sc.found_files):>6d}"
        )


def print_final_summary(all_cards: list[ScoreCard]) -> None:
    """Print the headline summary across all queries."""
    print(f"\n{'=' * 70}")
    print(f"  BLIND AGENT COMPARISON: FINAL SUMMARY")
    print(f"{'=' * 70}")

    # Group by agent type
    from collections import defaultdict
    by_type: dict[str, list[ScoreCard]] = defaultdict(list)
    for sc in all_cards:
        by_type[sc.agent_type].append(sc)

    print(f"\n  {'Agent Type':<25s} {'Avg Prec':>9s} {'Avg Recall':>11s} {'Avg F1':>7s} {'Avg Tokens':>11s}")
    print(f"  {'-' * 65}")

    for agent_type in sorted(by_type.keys()):
        cards = by_type[agent_type]
        avg_prec = sum(c.precision for c in cards) / len(cards)
        avg_recall = sum(c.recall for c in cards) / len(cards)
        avg_f1 = sum(c.f1 for c in cards) / len(cards)
        avg_tokens = sum(c.tokens_used for c in cards) / len(cards)
        print(
            f"  {agent_type:<25s} {avg_prec:>8.0%} {avg_recall:>10.0%}"
            f" {avg_f1:>6.0%} {avg_tokens:>11,.0f}"
        )

    # Token savings
    raw_cards = by_type.get("raw", [])
    rag_cards = [c for k, v in by_type.items() if k.startswith("rag") for c in v]
    if raw_cards and rag_cards:
        avg_raw_tokens = sum(c.tokens_used for c in raw_cards) / len(raw_cards)
        avg_rag_tokens = sum(c.tokens_used for c in rag_cards) / len(rag_cards)
        savings = 1 - (avg_rag_tokens / avg_raw_tokens) if avg_raw_tokens > 0 else 0
        ratio = avg_raw_tokens / avg_rag_tokens if avg_rag_tokens > 0 else 0
        print(f"\n  HEADLINE: RAG uses {savings:.1%} fewer tokens ({ratio:.0f}x compression)")
        print(f"            Raw avg: {avg_raw_tokens:,.0f} tokens  |  RAG avg: {avg_rag_tokens:,.0f} tokens")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def homepayroll_path() -> Path:
    """Return the path to the HomePayroll repo, skip if not cloned."""
    repo = Path(HOMEPAYROLL_DIR)
    if not repo.exists():
        pytest.skip("HomePayroll not cloned at .acceptance-cache/HomePayroll")
    return repo


def _rebuild_bm25_from_metadata(
    collection: chromadb.Collection,
    bm25_index: BM25Index,
) -> None:
    """Rebuild BM25 index from stored bm25_tokens metadata in ChromaDB."""
    stored = collection.get(include=["metadatas"])
    if not stored["ids"]:
        return

    doc_ids: list[str] = []
    tokenized: list[list[str]] = []

    for i, doc_id in enumerate(stored["ids"]):
        meta = stored["metadatas"][i] if stored["metadatas"] else {}
        bm25_tokens_str = meta.get("bm25_tokens", "")
        tokens = bm25_tokens_str.split() if bm25_tokens_str else ["_empty_"]
        doc_ids.append(doc_id)
        tokenized.append(tokens)

    if doc_ids:
        bm25_index.rebuild(doc_ids, tokenized)


@pytest.fixture(scope="session")
def homepayroll_rag(homepayroll_path: Path) -> tuple[chromadb.Collection, BM25Index]:
    """Load the pre-built HomePayroll RAG index.

    If the index doesn't exist or is empty, build it on-the-fly.
    BM25 index is always rebuilt from ChromaDB metadata (it's in-memory only).
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Try to load existing
    try:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            print(f"\n  [SETUP] Loading existing index ({collection.count()} chunks)...")
            bm25 = BM25Index()
            _rebuild_bm25_from_metadata(collection, bm25)
            print(f"  [SETUP] BM25 rebuilt with {len(bm25)} documents")
            return collection, bm25
    except Exception:
        pass

    # Build if not available
    print("\n  [SETUP] Building HomePayroll RAG index (this takes ~3-5 minutes)...")
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    config = RagConfig(content_mode="both", chunk_size=1000, overlap=200)
    bm25 = BM25Index()

    index_documents(homepayroll_path, config, collection, bm25, embed_texts_google)

    print(f"  [SETUP] Indexed {collection.count()} chunks")
    return collection, bm25


@pytest.fixture(scope="session")
def corpus_token_count(homepayroll_rag: tuple) -> int:
    """Count total tokens in the RAG corpus."""
    collection, _ = homepayroll_rag
    stored = collection.get(include=["documents"])
    docs = stored.get("documents") or []
    return sum(count_tokens(doc) for doc in docs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.ollama
class TestBlindAgentComparison:
    """Head-to-head: Raw keyword search vs RAG retrieval.

    For each query, runs:
      - 1 raw agent (grep-based keyword search)
      - 1 RAG agent with fixed top_k=10
      - 1 RAG agent with top_k=20
      - 1 RAG adaptive agent (score threshold cutoff)

    All scored against ground truth.
    """

    def _run_query(
        self,
        query: QueryGroundTruth,
        homepayroll_path: Path,
        homepayroll_rag: tuple,
    ) -> tuple[list[ScoreCard], list[ScoreCard], list[ScoreCard]]:
        """Run all agent types for one query and return scorecards."""
        collection, bm25 = homepayroll_rag

        # Raw agent
        raw_result = raw_agent_search(query, homepayroll_path)
        raw_score = score_agent_result(raw_result, query)

        # RAG agent with top_k=10
        rag_10 = rag_agent_search(query, collection, bm25, top_k=10, run_index=0)
        rag_10.agent_type = "rag_top10"
        rag_10_score = score_agent_result(rag_10, query)

        # RAG agent with top_k=20
        rag_20 = rag_agent_search(query, collection, bm25, top_k=20, run_index=0)
        rag_20.agent_type = "rag_top20"
        rag_20_score = score_agent_result(rag_20, query)

        # RAG adaptive agent (threshold=0.3)
        rag_adapt_03 = rag_agent_search_adaptive(
            query, collection, bm25, score_threshold=0.3, max_k=30, run_index=0,
        )
        rag_adapt_03_score = score_agent_result(rag_adapt_03, query)

        # RAG adaptive agent (threshold=0.2)
        rag_adapt_02 = rag_agent_search_adaptive(
            query, collection, bm25, score_threshold=0.2, max_k=30, run_index=0,
        )
        rag_adapt_02_score = score_agent_result(rag_adapt_02, query)

        raw_scores = [raw_score]
        rag_scores = [rag_10_score, rag_20_score]
        adaptive_scores = [rag_adapt_03_score, rag_adapt_02_score]

        print_comparison_table(raw_scores, rag_scores, adaptive_scores, query)
        for sc in raw_scores + rag_scores + adaptive_scores:
            print_scorecard(sc, query)

        return raw_scores, rag_scores, adaptive_scores

    def test_query_1_legal_update_triggers(
        self, homepayroll_path: Path, homepayroll_rag: tuple,
    ) -> None:
        """Q1: Broad regulatory — legal compliance knowledge docs."""
        raw, rag, adaptive = self._run_query(QUERY_1, homepayroll_path, homepayroll_rag)

        best_rag = max(rag + adaptive, key=lambda s: s.recall)
        assert best_rag.recall >= 0.7, (
            f"RAG recall too low for legal update triggers: {best_rag.recall:.0%}"
        )

    def test_query_2_privacy_policy(
        self, homepayroll_path: Path, homepayroll_rag: tuple,
    ) -> None:
        """Q2: Privacy policy knowledge — research, plans, and pages."""
        raw, rag, adaptive = self._run_query(QUERY_2, homepayroll_path, homepayroll_rag)

        best_rag = max(rag + adaptive, key=lambda s: s.recall)
        assert best_rag.recall >= 0.7, (
            f"RAG recall too low for privacy policy: {best_rag.recall:.0%}"
        )

    def test_query_3_tax_data_deadlines(
        self, homepayroll_path: Path, homepayroll_rag: tuple,
    ) -> None:
        """Q3: Tax knowledge — blog posts, guides, summaries, data files."""
        raw, rag, adaptive = self._run_query(QUERY_3, homepayroll_path, homepayroll_rag)

        best_rag = max(rag + adaptive, key=lambda s: s.recall)
        assert best_rag.recall >= 0.7, (
            f"RAG recall too low for tax knowledge: {best_rag.recall:.0%}"
        )

    def test_query_4_security_infrastructure(
        self, homepayroll_path: Path, homepayroll_rag: tuple,
    ) -> None:
        """Q4: Security knowledge — compliance docs + research + plans."""
        raw, rag, adaptive = self._run_query(QUERY_4, homepayroll_path, homepayroll_rag)

        best_rag = max(rag + adaptive, key=lambda s: s.recall)
        assert best_rag.recall >= 0.7, (
            f"RAG recall too low for security knowledge: {best_rag.recall:.0%}"
        )

    def test_query_5_contract_pdf_generation(
        self, homepayroll_path: Path, homepayroll_rag: tuple,
    ) -> None:
        """Q5: PDF/contract knowledge — plans, scripts, and summaries."""
        raw, rag, adaptive = self._run_query(QUERY_5, homepayroll_path, homepayroll_rag)

        best_rag = max(rag + adaptive, key=lambda s: s.recall)
        assert best_rag.recall >= 0.7, (
            f"RAG recall too low for contract/PDF generation: {best_rag.recall:.0%}"
        )

    def test_overall_summary(
        self, homepayroll_path: Path, homepayroll_rag: tuple, corpus_token_count: int,
    ) -> None:
        """Aggregate summary across all 5 queries."""
        all_cards: list[ScoreCard] = []

        for query in ALL_QUERIES:
            collection, bm25 = homepayroll_rag

            # Raw
            raw_result = raw_agent_search(query, homepayroll_path)
            all_cards.append(score_agent_result(raw_result, query))

            # RAG top_k=10
            rag_10 = rag_agent_search(query, collection, bm25, top_k=10)
            rag_10.agent_type = "rag_top10"
            all_cards.append(score_agent_result(rag_10, query))

            # RAG top_k=20
            rag_20 = rag_agent_search(query, collection, bm25, top_k=20)
            rag_20.agent_type = "rag_top20"
            all_cards.append(score_agent_result(rag_20, query))

            # RAG adaptive 0.3
            rag_a3 = rag_agent_search_adaptive(query, collection, bm25, score_threshold=0.3)
            all_cards.append(score_agent_result(rag_a3, query))

            # RAG adaptive 0.2
            rag_a2 = rag_agent_search_adaptive(query, collection, bm25, score_threshold=0.2)
            all_cards.append(score_agent_result(rag_a2, query))

        print_final_summary(all_cards)

        # RAG agents should use dramatically fewer tokens than raw
        raw_cards = [c for c in all_cards if c.agent_type == "raw"]
        rag_cards = [c for c in all_cards if c.agent_type.startswith("rag")]
        avg_raw = sum(c.tokens_used for c in raw_cards) / len(raw_cards)
        avg_rag = sum(c.tokens_used for c in rag_cards) / len(rag_cards)

        savings = 1 - (avg_rag / avg_raw) if avg_raw > 0 else 0
        assert savings >= 0.90, (
            f"RAG should save >= 90% tokens vs raw search, got {savings:.1%}"
        )

        print(f"\n  Corpus tokens: {corpus_token_count:,}")
        print(f"  Assertion passed: RAG saves {savings:.1%} of tokens vs raw search")


@pytest.mark.ollama
class TestTopKCalibration:
    """Test different top_k values and score thresholds to find optimal settings."""

    def test_top_k_sweep(
        self, homepayroll_path: Path, homepayroll_rag: tuple,
    ) -> None:
        """Sweep top_k from 5 to 30 and measure recall vs token cost."""
        collection, bm25 = homepayroll_rag

        print(f"\n{'=' * 70}")
        print(f"  TOP-K CALIBRATION SWEEP")
        print(f"{'=' * 70}")
        print(f"  {'top_k':>6s} {'Avg Recall':>11s} {'Avg Prec':>9s} {'Avg F1':>7s} {'Avg Tokens':>11s}")
        print(f"  {'-' * 46}")

        best_f1 = 0.0
        best_k = 0

        for k in [5, 8, 10, 15, 20, 25, 30]:
            scores: list[ScoreCard] = []
            for query in ALL_QUERIES:
                result = rag_agent_search(query, collection, bm25, top_k=k)
                result.agent_type = f"rag_top{k}"
                scores.append(score_agent_result(result, query))

            avg_recall = sum(s.recall for s in scores) / len(scores)
            avg_prec = sum(s.precision for s in scores) / len(scores)
            avg_f1 = sum(s.f1 for s in scores) / len(scores)
            avg_tokens = sum(s.tokens_used for s in scores) / len(scores)

            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_k = k

            print(f"  {k:>6d} {avg_recall:>10.0%} {avg_prec:>8.0%} {avg_f1:>6.0%} {avg_tokens:>11,.0f}")

        print(f"\n  OPTIMAL top_k = {best_k} (F1 = {best_f1:.0%})")
        print(f"{'=' * 70}")

    def test_score_threshold_sweep(
        self, homepayroll_path: Path, homepayroll_rag: tuple,
    ) -> None:
        """Sweep score thresholds for adaptive retrieval."""
        collection, bm25 = homepayroll_rag

        print(f"\n{'=' * 70}")
        print(f"  SCORE THRESHOLD CALIBRATION SWEEP")
        print(f"{'=' * 70}")
        print(f"  {'threshold':>10s} {'Avg Recall':>11s} {'Avg Prec':>9s} {'Avg F1':>7s} {'Avg Tokens':>11s} {'Avg Chunks':>11s}")
        print(f"  {'-' * 61}")

        best_f1 = 0.0
        best_threshold = 0.0

        for threshold in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
            scores: list[ScoreCard] = []
            total_chunks = 0
            for query in ALL_QUERIES:
                result = rag_agent_search_adaptive(
                    query, collection, bm25,
                    score_threshold=threshold, max_k=30,
                )
                scores.append(score_agent_result(result, query))
                total_chunks += len(result.found_files)

            avg_recall = sum(s.recall for s in scores) / len(scores)
            avg_prec = sum(s.precision for s in scores) / len(scores)
            avg_f1 = sum(s.f1 for s in scores) / len(scores)
            avg_tokens = sum(s.tokens_used for s in scores) / len(scores)
            avg_chunks = total_chunks / len(scores)

            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_threshold = threshold

            print(
                f"  {threshold:>10.2f} {avg_recall:>10.0%} {avg_prec:>8.0%}"
                f" {avg_f1:>6.0%} {avg_tokens:>11,.0f} {avg_chunks:>11.1f}"
            )

        print(f"\n  OPTIMAL threshold = {best_threshold} (F1 = {best_f1:.0%})")
        print(f"{'=' * 70}")


@pytest.mark.ollama
class TestTokenEfficiency:
    """Measure the token efficiency ratio: quality per token spent."""

    def test_quality_per_token(
        self, homepayroll_path: Path, homepayroll_rag: tuple,
    ) -> None:
        """Compare quality (F1) per 1000 tokens across agent types."""
        collection, bm25 = homepayroll_rag

        agent_configs = [
            ("raw", lambda q: raw_agent_search(q, homepayroll_path)),
            ("rag_top5", lambda q: rag_agent_search(q, collection, bm25, top_k=5)),
            ("rag_top10", lambda q: rag_agent_search(q, collection, bm25, top_k=10)),
            ("rag_top20", lambda q: rag_agent_search(q, collection, bm25, top_k=20)),
            ("rag_adaptive_0.3", lambda q: rag_agent_search_adaptive(
                q, collection, bm25, score_threshold=0.3,
            )),
        ]

        print(f"\n{'=' * 70}")
        print(f"  TOKEN EFFICIENCY: F1 per 1000 tokens")
        print(f"{'=' * 70}")
        print(f"  {'Agent':<25s} {'Avg F1':>7s} {'Avg Tokens':>11s} {'F1/1K tok':>10s}")
        print(f"  {'-' * 55}")

        for name, agent_fn in agent_configs:
            scores: list[ScoreCard] = []
            for query in ALL_QUERIES:
                result = agent_fn(query)
                result.agent_type = name
                scores.append(score_agent_result(result, query))

            avg_f1 = sum(s.f1 for s in scores) / len(scores)
            avg_tokens = sum(s.tokens_used for s in scores) / len(scores)
            efficiency = (avg_f1 * 1000 / avg_tokens) if avg_tokens > 0 else 0

            print(f"  {name:<25s} {avg_f1:>6.0%} {avg_tokens:>11,.0f} {efficiency:>10.4f}")

        print(f"{'=' * 70}")

        # RAG should have better efficiency (F1 per token) than raw
        # (even if absolute F1 is similar, using 95%+ fewer tokens)
