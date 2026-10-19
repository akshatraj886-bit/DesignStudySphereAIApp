"""
StudySphere AI - Intelligent Multi-Strategy Document Retrieval Service
Provides scope-aware, hybrid (dense + lexical), and page-targeted retrieval:
1. Page-Specific Retrieval: Accurately targets explicit page/slide requests ('Page 3 me kya hai', 'explain slide 2').
2. Document-Wide Stratified Sampling: Distributes context evenly across all pages for full-document summaries.
3. Multi-Aspect / Comparison Retrieval: Retrieves evidence for each entity in 'difference between X and Y' questions.
4. Hybrid Dense + Lexical Acronym Boost: Combines vector similarity with exact keyword/acronym matching.
5. Multi-Tenant Document Isolation: Strictly confines retrieval to the requested document.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np

from starlette.concurrency import run_in_threadpool
from app.rag.store import store_instance
from app.rag.embeddings import embedding_service
from app.rag.generator import STOP_AND_FILLER_WORDS, term_matches_text, extract_substantive_query_terms

logger = logging.getLogger("studysphere.retrieval")

# Regular expression patterns for explicit page/slide queries
PAGE_INTENT_PATTERNS = [
    r"\b(?:page|slide|p\.|pg|section)\s*#?\s*(\d+)\b",
    r"\b(\d+)\s*(?:st|nd|rd|th)?\s*(?:page|slide)\b",
    r"\b(?:p)\s*(\d+)\b",
]

# Patterns detecting whole-document overview or summary intent
DOCUMENT_SUMMARY_PATTERNS = [
    r"\b(summarize|summarise|summary)\b",
    r"\b(entire|whole|all|full)\s+(document|notes|pdf|presentation|slides?|file|book|material|chapter)\b",
    r"\b(overview|outline|table of contents|walk me through|brief me|high level|review)\b",
    r"\b(is\s+(?:pdf|doc|document|file)\s+me\s+kya\s+hai)\b",
    r"\b(puri|saari|sab)\s+(?:notes|pdf|slide|slides|file|document)\b",
    r"\b(kya\s+kya\s+topics?\s+(?:cover|hai|hain))\b",
    r"\b(all\s+topics|all\s+chapters|main\s+topics|main\s+points)\b",
]

# Patterns detecting comparisons / differences
COMPARISON_PATTERNS = [
    r"\b(?:difference|differences|distinction|compare|comparison|versus|vs\.?|contrast)\b",
    r"\b(?:kya\s+antar\s+hai|antar|farq|kisme\s+kya\s+antar)\b",
]


def detect_page_intent(query: str, max_page: Optional[int] = None) -> Optional[int]:
    """
    Detects if the user query explicitly targets a specific page or slide number.
    Returns 1-indexed integer page number if found and valid, else None.
    """
    q_lower = query.lower()
    for pattern in PAGE_INTENT_PATTERNS:
        match = re.search(pattern, q_lower)
        if match:
            try:
                page_num = int(match.group(1))
                if page_num > 0:
                    if max_page is not None and page_num > max_page:
                        # User asked for a page beyond the document
                        return page_num
                    return page_num
            except (ValueError, IndexError):
                pass
    return None


def is_document_summary_intent(query: str) -> bool:
    """Detects if query asks for an overarching summary or broad overview of the whole document."""
    q_lower = query.lower()
    return any(re.search(pat, q_lower) for pat in DOCUMENT_SUMMARY_PATTERNS)


def extract_comparison_aspects(query: str) -> Tuple[bool, List[str]]:
    """
    Detects comparison questions and extracts the two or more contrasting concepts
    (e.g., 'difference between paging and segmentation' -> ['paging', 'segmentation']).
    """
    q_lower = query.lower()
    is_comp = any(re.search(pat, q_lower) for pat in COMPARISON_PATTERNS)
    if not is_comp:
        return False, []

    # Look for "X and Y", "X vs Y", "X or Y"
    aspects: List[str] = []
    diff_match = re.search(r"(?:between|compare|vs\.?|versus)\s+(.+?)(?:\?|$|\.|\n)", q_lower)
    if diff_match:
        clause = diff_match.group(1)
        sub_parts = re.split(r"\b(?:and|vs\.?|versus|to|with)\b", clause)
        for part in sub_parts:
            cleaned_tokens = extract_substantive_query_terms(part)
            if cleaned_tokens:
                aspects.append(" ".join(cleaned_tokens))

    return True, aspects


def compute_lexical_boost(chunk_text: str, query_terms: List[str]) -> float:
    """
    Computes a keyword overlap boost for technical terms, acronyms, and names.
    Gives extra weight to exact acronyms (e.g. 'BCNF', 'FIFO', 'LRU', 'SJF').
    """
    if not query_terms:
        return 0.0

    hits = 0
    acronym_hits = 0
    chunk_lower = chunk_text.lower()
    chunk_words = set(re.findall(r"\b[A-Za-z0-9_]+\b", chunk_text))

    for term in query_terms:
        t_clean = term.strip()
        if not t_clean:
            continue
        # Exact word match
        if t_clean in chunk_words or term_matches_text(t_clean, chunk_lower):
            hits += 1
            if t_clean.isupper() and len(t_clean) >= 2:
                acronym_hits += 1

    ratio = hits / len(query_terms)
    boost = (0.20 * ratio) + (0.10 if acronym_hits > 0 else 0.0)
    return min(0.30, boost)


class RetrievalResult:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        scope: str,
        target_page: Optional[int] = None,
        is_summary: bool = False,
    ):
        self.chunks = chunks
        self.scope = scope
        self.target_page = target_page
        self.is_summary = is_summary


async def retrieve_document_context(
    query: str,
    document_id: Optional[str] = None,
    session_token: str = "default_session",
    top_k: int = 5,
    similarity_threshold: float = 0.32,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> RetrievalResult:
    """
    Intelligent scope-aware retrieval coordinator:
    1. Detects page-specific intent -> retrieves authoritative chunks from target page.
    2. Detects whole-document summary intent -> performs stratified sampling across all pages.
    3. Detects comparison queries -> retrieves multi-aspect context for both entities.
    4. General query -> performs dense vector search + lexical keyword overlap boost.
    Strictly isolates chunks to document_id and session_token.
    """
    clean_query = query.strip()
    target_doc_id = document_id.strip() if document_id and document_id.strip() else None

    # Resolve document metadata to know total page count
    doc_meta = None
    if target_doc_id:
        doc_meta = store_instance.get_document(target_doc_id, session_token=session_token)

    max_page = doc_meta.get("page_count") if doc_meta else None

    # Check Page-Specific Intent
    page_target = detect_page_intent(clean_query, max_page=max_page)
    if target_doc_id and page_target is not None:
        # Retrieve all chunks belonging to this document and filter to page_target
        all_doc_chunks = store_instance.get_document_chunks(target_doc_id, session_token=session_token)
        page_chunks = [c for c in all_doc_chunks if c.get("page_number") == page_target]

        if page_chunks:
            # Query embedding for ranking within the page if multiple chunks exist
            try:
                q_emb = await run_in_threadpool(embedding_service.embed_query, clean_query)
                q_vec = np.asarray(q_emb, dtype=np.float32)
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 0:
                    q_vec /= q_norm
            except Exception:
                q_vec = None

            scored_page_chunks = []
            query_terms = extract_substantive_query_terms(clean_query)

            for idx, c in enumerate(page_chunks):
                c_copy = dict(c)
                base_score = 0.88 - (idx * 0.02)
                if q_vec is not None and "embedding" in c:
                    try:
                        c_vec = np.frombuffer(c["embedding"], dtype=np.float32)
                        c_norm = np.linalg.norm(c_vec)
                        if c_norm > 0:
                            sim = float(np.dot(q_vec, c_vec / c_norm))
                            base_score = max(0.50, (sim + 1.0) / 2.0 if sim < 0 else sim)
                    except Exception:
                        pass
                
                lex_boost = compute_lexical_boost(c.get("text", ""), query_terms)
                c_copy["similarity_score"] = round(min(1.0, base_score + lex_boost), 4)
                c_copy["excerpt"] = c.get("text", "")
                scored_page_chunks.append(c_copy)

            scored_page_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
            return RetrievalResult(
                chunks=scored_page_chunks[:max(top_k, len(page_chunks))],
                scope="page_specific",
                target_page=page_target,
            )
        else:
            # User requested a page that has no extractable chunks (or is out of range)
            # Fetch surrounding/intro chunks with clear notification
            logger.info("Requested page %d not found in document %s (max_page=%s)", page_target, target_doc_id, max_page)

    # Check Document-Wide Summary Intent
    is_summary = is_document_summary_intent(clean_query)
    if target_doc_id and is_summary:
        all_doc_chunks = store_instance.get_document_chunks(target_doc_id, session_token=session_token)
        if all_doc_chunks:
            # Stratified sampling across all unique pages
            chunks_by_page: Dict[int, List[Dict[str, Any]]] = {}
            for c in all_doc_chunks:
                p = c.get("page_number", 1)
                chunks_by_page.setdefault(p, []).append(c)

            sampled_chunks: List[Dict[str, Any]] = []
            sorted_pages = sorted(chunks_by_page.keys())
            total_pages = len(sorted_pages)

            if total_pages <= 8:
                # Take the first chunk (introductory/key) from every page
                for p in sorted_pages:
                    c = dict(chunks_by_page[p][0])
                    c["similarity_score"] = 0.85
                    c["excerpt"] = c.get("text", "")
                    sampled_chunks.append(c)
            else:
                # Evenly distribute sample across document
                step = total_pages / 8.0
                indices = [int(i * step) for i in range(8)]
                for idx in indices:
                    p = sorted_pages[min(idx, total_pages - 1)]
                    c = dict(chunks_by_page[p][0])
                    c["similarity_score"] = 0.85
                    c["excerpt"] = c.get("text", "")
                    if c not in sampled_chunks:
                        sampled_chunks.append(c)

            return RetrievalResult(
                chunks=sampled_chunks,
                scope="document_summary",
                is_summary=True,
            )

    # Check Comparison Intent (Multi-Aspect Search)
    is_comp, comp_aspects = extract_comparison_aspects(clean_query)
    query_terms = extract_substantive_query_terms(clean_query)

    # Compute dense query vector
    query_embedding = await run_in_threadpool(embedding_service.embed_query, clean_query)

    # Retrieve candidate chunks from store (dense search)
    raw_candidates = store_instance.search_chunks(
        query_embedding=query_embedding,
        top_k=max(top_k * 2, 8),
        document_id=target_doc_id,
        session_token=session_token,
    )

    # If comparison aspects were identified, also retrieve targeted candidates for each aspect
    if is_comp and comp_aspects and target_doc_id:
        seen_chunk_ids = {c["chunk_id"] for c in raw_candidates}
        for aspect in comp_aspects[:2]:
            try:
                aspect_emb = await run_in_threadpool(embedding_service.embed_query, aspect)
                aspect_chunks = store_instance.search_chunks(
                    query_embedding=aspect_emb,
                    top_k=2,
                    document_id=target_doc_id,
                    session_token=session_token,
                )
                for ac in aspect_chunks:
                    if ac["chunk_id"] not in seen_chunk_ids:
                        seen_chunk_ids.add(ac["chunk_id"])
                        raw_candidates.append(ac)
            except Exception as e:
                logger.debug("Aspect search error: %s", e)

    # Apply lexical boost for exact keyword matches, acronyms, and proper nouns
    scored_candidates = []
    for c in raw_candidates:
        c_copy = dict(c)
        c_text = c.get("text", "") or c.get("excerpt", "")
        lex_boost = compute_lexical_boost(c_text, query_terms)
        raw_sim = c.get("similarity_score", 0.0)
        # Weighted hybrid score: 80% dense vector + 20% lexical keyword boost
        final_score = round(min(1.0, (raw_sim * 0.80) + lex_boost), 4)
        c_copy["similarity_score"] = final_score
        c_copy["excerpt"] = c_text
        scored_candidates.append(c_copy)

    scored_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)

    # If single document is active but top similarity is modest, ensure first chunk is present
    if target_doc_id and not scored_candidates:
        doc_chunks = store_instance.get_document_chunks(target_doc_id, session_token=session_token)
        if doc_chunks:
            for idx, c in enumerate(doc_chunks[:4]):
                c_copy = dict(c)
                c_copy["similarity_score"] = 0.50
                c_copy["excerpt"] = c.get("text", "")
                scored_candidates.append(c_copy)

    return RetrievalResult(
        chunks=scored_candidates[:top_k],
        scope="comparison" if is_comp else "targeted_concept",
    )
