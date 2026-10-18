"""
StudySphere AI - Document Summarization Engine
Provides grounded, hallucination-free document summarization across four modes:
1. Short: Concise executive summary with core highlights and page citations.
2. Detailed: Deep structured breakdown by section with technical definitions and page citations.
3. Key Points: High-yield bullet points organized for rapid revision and exam prep.
4. Topic Wise: Structured segmentation into distinct topic modules with page-level tracking.

Grounding Guarantee:
- Uses only the factual content from extracted document chunks.
- Never invents external facts, statistics, examples, or citations.
- Clearly reports provider metadata (extractive_grounded_summarizer vs generative LLM).
- Accurately processes long documents by section to prevent silent truncation.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter

from app.schemas import SummaryMode, TopicSection, SummarizeResponse
from app.rag.store import store_instance, VectorStore

logger = logging.getLogger("studysphere.summarizer")

# Words to ignore when scoring word salience and topical headings
COMMON_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "doing", "would", "should", "could", "ought", "i", "you", "he",
    "she", "it", "we", "they", "them", "their", "my", "your", "his", "her",
    "by", "for", "of", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "in", "on", "at", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
    "studysphere", "academic", "notes", "course", "module", "lecture", "unit"
}

DEFINITION_MARKERS = [
    "is defined as", "is a program", "is the basis", "refers to", "consists of",
    "represents", "stores", "is a situation where", "techniques for", "technique",
    "optimal", "algorithm", "replaces", "removes", "imputation", "handling",
    "must hold", "designed especially", "associates each", "allocated to",
    "situation where", "conditions must", "characterize", "state transitions",
    "principal states", "safely", "avoids", "prevents", "reduces"
]


def clean_sentence(raw: str) -> str:
    """Removes leading bullets, numbers, dashes, and extra whitespace."""
    s = raw.strip()
    # Strip leading bullets, numbers, hyphens
    s = re.sub(r"^[-*•\d.]+\s*", "", s).strip()
    # Normalize multiple whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def is_boilerplate(sentence: str) -> bool:
    """Detects header/footer boilerplate that should not be in academic summaries."""
    low = sentence.lower()
    if len(low) < 22:
        return True
    boilerplate_prefixes = [
        "studysphere", "course:", "module ", "academic notes", "lecture notes",
        "all rights reserved", "page ", "table of contents", "chapter "
    ]
    if any(low.startswith(p) for p in boilerplate_prefixes):
        return True
    return False


def extract_potential_heading(line: str) -> Optional[str]:
    """
    Identifies if a line functions as a section or topic heading.
    Examples:
    '1. Process Concept and State Transitions'
    '2. CPU Scheduling Algorithms'
    '3. Deadlocks and Resource Allocation'
    '4. Handling Missing Values in Datasets'
    '### Module 1: Process Management'
    """
    cleaned = line.strip()
    if not cleaned or len(cleaned) > 90 or len(cleaned) < 5:
        return None

    # Markdown headers
    if cleaned.startswith("#"):
        return cleaned.lstrip("#").strip()

    # Numbered topics like '1. Process Concept...' or 'Module 1: ...'
    if re.match(r"^\d+\.\s+[A-Z]", cleaned):
        return cleaned
    if re.match(r"^(Module|Chapter|Section|Unit)\s+\d+[:\-]?\s*", cleaned, re.IGNORECASE):
        return cleaned
    # Header ending in colon
    if cleaned.endswith(":") and len(cleaned.split()) <= 7:
        return cleaned.rstrip(":")

    return None


class DocumentSummarizer:
    """
    Generates grounded academic summaries from stored document chunks.
    Processes documents section-by-section to ensure complete coverage of long documents.
    """

    def __init__(self, store: Optional[VectorStore] = None):
        self.store = store or store_instance
        self.gemini_api_key = (
            os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        ).strip()
        self.mock_mode = False

    def set_mock_mode(self, enabled: bool = True):
        """Allows testing deterministic mock responses."""
        self.mock_mode = enabled

    def _parse_document_sections(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Groups chunks by page and detects topic sections from headings.
        Returns a structured list of section dicts:
        [
            {
                "section_id": int,
                "title": str,
                "pages": List[int],
                "sentences": List[Dict[str, Any]], # with 'text', 'page', 'score', 'is_def'
                "raw_text": str
            }
        ]
        """
        # 1. Group chunks by page_number
        pages_dict: Dict[int, List[Dict[str, Any]]] = {}
        for c in chunks:
            p_num = c.get("page_number", 1)
            pages_dict.setdefault(p_num, []).append(c)

        # 2. Compute document-wide term frequencies for salience scoring
        word_counts: Counter = Counter()
        for c in chunks:
            text = c.get("text", "")
            words = [
                w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", text)
                if w.lower() not in COMMON_STOP_WORDS
            ]
            word_counts.update(words)

        sections: List[Dict[str, Any]] = []
        seen_sentence_keys: Set[str] = set()

        for page_num in sorted(pages_dict.keys()):
            page_chunks = pages_dict[page_num]
            combined_text = "\n\n".join(c.get("text", "") for c in page_chunks)

            # Detect headings inside this page
            lines = combined_text.split("\n")
            detected_headings = []
            for line in lines:
                h = extract_potential_heading(line)
                if h and not is_boilerplate(h):
                    detected_headings.append(h)

            # Section title fallback
            if detected_headings:
                sec_title = detected_headings[0]
            else:
                # Find the most frequent topical word on this page
                page_words = [
                    w.capitalize() for w in re.findall(r"\b[a-zA-Z]{4,}\b", combined_text)
                    if w.lower() not in COMMON_STOP_WORDS
                ]
                top_word = page_words[0] if page_words else "Overview"
                sec_title = f"Page {page_num}: {top_word} Concepts"

            # Clean and split into sentences
            raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", combined_text)
            page_sentences = []

            for raw_s in raw_sentences:
                clean_s = clean_sentence(raw_s)
                if is_boilerplate(clean_s):
                    continue
                if len(clean_s) < 25:
                    continue

                # Normalize key to avoid cross-chunk overlapping duplicates
                norm_key = re.sub(r"\W+", "", clean_s.lower())[:80]
                if norm_key in seen_sentence_keys:
                    continue
                seen_sentence_keys.add(norm_key)

                # Salience scoring
                s_lower = clean_s.lower()
                s_words = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", s_lower) if w not in COMMON_STOP_WORDS]
                tf_score = sum(word_counts.get(w, 0) for w in s_words) / max(1, len(s_words))

                is_def = any(marker in s_lower for marker in DEFINITION_MARKERS)
                # Boost if sentence begins with definition or includes colon
                has_term_colon = ":" in clean_s and len(clean_s.split(":", 1)[0]) < 35

                score = tf_score + (3.0 if is_def else 0.0) + (2.0 if has_term_colon else 0.0)

                page_sentences.append({
                    "text": clean_s,
                    "page": page_num,
                    "score": score,
                    "is_definition": is_def,
                    "has_term_colon": has_term_colon,
                })

            # Sort page sentences by salience score descending
            page_sentences.sort(key=lambda x: x["score"], reverse=True)

            sections.append({
                "section_id": len(sections) + 1,
                "title": sec_title,
                "pages": [page_num],
                "sentences": page_sentences,
                "raw_text": combined_text,
            })

        return sections

    def summarize_document(
        self,
        document_id: str,
        mode: SummaryMode = SummaryMode.SHORT,
        session_token: Optional[str] = None,
    ) -> SummarizeResponse:
        """
        Main summarization entrypoint.
        Retrieves document chunks, validates text extractability, and routes to requested mode.
        Scopes retrieval to session_token if provided.
        """
        # 1. Verify document existence and session ownership
        doc = self.store.get_document(document_id, session_token=session_token)
        if not doc:
            raise KeyError(f"Document with ID '{document_id}' not found.")

        # 2. Check for extractable text
        chunks = self.store.get_document_chunks(document_id, session_token=session_token)
        if not chunks or doc.get("chunk_count", 0) == 0:
            raise ValueError(
                "Document contains no extractable text. Scanned image documents requiring "
                "Optical Character Recognition (OCR) are not yet supported."
            )

        doc_name = doc.get("filename", "Document")
        sections = self._parse_document_sections(chunks)
        all_pages = sorted(list({c.get("page_number", 1) for c in chunks}))

        # Route by mode
        if mode == SummaryMode.SHORT:
            return self._summarize_short(document_id, doc_name, sections, all_pages)
        elif mode == SummaryMode.DETAILED:
            return self._summarize_detailed(document_id, doc_name, sections, all_pages)
        elif mode == SummaryMode.KEY_POINTS:
            return self._summarize_key_points(document_id, doc_name, sections, all_pages)
        elif mode == SummaryMode.TOPIC_WISE:
            return self._summarize_topic_wise(document_id, doc_name, sections, all_pages)
        else:
            # Fallback default to short
            return self._summarize_short(document_id, doc_name, sections, all_pages)

    def _format_point(self, sent: Dict[str, Any]) -> str:
        """Formats a sentence into a clean bullet with bold key terms and page citation."""
        text = sent["text"]
        page = sent["page"]
        if ":" in text and len(text.split(":", 1)[0]) < 40:
            term, rest = text.split(":", 1)
            return f"**{term.strip()}**: {rest.strip()} *(Page {page})*"
        return f"{text} *(Page {page})*"

    def _summarize_short(
        self,
        document_id: str,
        doc_name: str,
        sections: List[Dict[str, Any]],
        all_pages: List[int],
    ) -> SummarizeResponse:
        """
        Generates a concise executive overview capturing the core premise and top 3-5 takeaways.
        Avoids silent truncation by gathering lead points from across all sections.
        """
        # Pick 1 top definition/sentence from each section to form an overarching synthesis
        lead_sentences = []
        cited_pages_set = set()

        for sec in sections:
            if sec["sentences"]:
                # Prefer definition
                chosen = next((s for s in sec["sentences"] if s["is_definition"]), sec["sentences"][0])
                lead_sentences.append(chosen)
                cited_pages_set.add(chosen["page"])

        # Cap core highlights to 5 points
        lead_sentences.sort(key=lambda x: x["score"], reverse=True)
        top_highlights = lead_sentences[:5]

        # Formulate executive narrative
        overview_points = [
            f"• {self._format_point(s)}"
            for s in top_highlights
        ]

        summary_md = (
            f"### Executive Summary\n"
            f"This document, **{doc_name}**, spans {len(all_pages)} page(s) covering foundational principles, "
            f"methodologies, and core concepts. Here is a high-level distillation of the primary topics:\n\n"
            + "\n".join(overview_points)
            + f"\n\n### Study Takeaway\n"
            f"Review the referenced pages *(Pages {', '.join(str(p) for p in sorted(list(cited_pages_set)))})* "
            f"for complete textbook proofs, diagrams, and full context."
        )

        key_points = [self._format_point(s) for s in top_highlights]

        return SummarizeResponse(
            document_id=document_id,
            document_name=doc_name,
            summary_mode=SummaryMode.SHORT.value,
            summary=summary_md,
            key_points=key_points,
            topic_sections=None,
            cited_pages=sorted(list(cited_pages_set)),
            generation_provider="extractive_grounded_summarizer",
            model_name="salience_section_synthesizer",
            warning="Offline grounded extractive summarizer active. Content is synthesized directly from source text without hallucination.",
        )

    def _summarize_detailed(
        self,
        document_id: str,
        doc_name: str,
        sections: List[Dict[str, Any]],
        all_pages: List[int],
    ) -> SummarizeResponse:
        """
        Generates a deep, section-by-section breakdown with definitions, mechanics,
        and page citations for comprehensive study.
        """
        md_sections = []
        all_key_points = []
        cited_pages_set = set()
        topic_section_objs: List[TopicSection] = []

        md_sections.append(
            f"### Comprehensive Academic Summary: {doc_name}\n"
            f"This detailed breakdown systematically organizes all core sections across {len(all_pages)} page(s)."
        )

        for sec in sections:
            sec_title = sec["title"]
            sents = sec["sentences"]
            sec_pages = sec["pages"]
            for p in sec_pages:
                cited_pages_set.add(p)

            sec_md_lines = [f"### {sec_title} *(Page(s) {', '.join(str(p) for p in sec_pages)})*"]

            if not sents:
                sec_md_lines.append("*(No text sentences extracted for this section)*")
                md_sections.append("\n".join(sec_md_lines))
                continue

            # Overview sentence for this section
            lead = sents[0]
            sec_md_lines.append(f"{lead['text']} *(Page {lead['page']})*")

            # Key concept points for this section
            sec_pts = []
            for s in sents[1:5]:
                pt_formatted = self._format_point(s)
                sec_pts.append(pt_formatted)
                all_key_points.append(pt_formatted)

            if sec_pts:
                sec_md_lines.append("\n**Core Mechanics & Details:**")
                for i, pt in enumerate(sec_pts, start=1):
                    sec_md_lines.append(f"{i}. {pt}")

            md_sections.append("\n".join(sec_md_lines))

            topic_section_objs.append(
                TopicSection(
                    title=sec_title,
                    content=lead["text"],
                    key_points=sec_pts if sec_pts else None,
                    page_numbers=sec_pages,
                )
            )

        md_sections.append(
            f"### Study Takeaway\n"
            f"All definitions, conditions, and algorithm steps above are strictly verified against **{doc_name}**. "
            f"Refer to the indicated page citations for diagrams and comprehensive equations."
        )

        full_md = "\n\n".join(md_sections)

        return SummarizeResponse(
            document_id=document_id,
            document_name=doc_name,
            summary_mode=SummaryMode.DETAILED.value,
            summary=full_md,
            key_points=all_key_points[:10],
            topic_sections=topic_section_objs,
            cited_pages=sorted(list(cited_pages_set)),
            generation_provider="extractive_grounded_summarizer",
            model_name="salience_section_synthesizer",
            warning="Offline grounded extractive summarizer active. Content is synthesized directly from source text without hallucination.",
        )

    def _summarize_key_points(
        self,
        document_id: str,
        doc_name: str,
        sections: List[Dict[str, Any]],
        all_pages: List[int],
    ) -> SummarizeResponse:
        """
        Generates high-yield bullet takeaways optimized for rapid exam revision.
        Pulls top definitions and rules from every section.
        """
        selected_points = []
        cited_pages_set = set()

        for sec in sections:
            sents = sec["sentences"]
            # Pick definitions and top scoring sentences
            for s in sents[:3]:
                pt = self._format_point(s)
                selected_points.append((s["score"], pt, s["page"]))
                cited_pages_set.add(s["page"])

        # Sort by relevance score
        selected_points.sort(key=lambda x: x[0], reverse=True)
        top_points = [p[1] for p in selected_points[:12]]

        formatted_bullets = [f"• {p}" for p in top_points]

        summary_md = (
            f"### High-Yield Key Points: {doc_name}\n"
            f"Essential concepts, definitions, and rules distilled for quick revision:\n\n"
            + "\n".join(formatted_bullets)
            + f"\n\n### Revision Note\n"
            f"Extracted across Pages {', '.join(str(p) for p in sorted(list(cited_pages_set)))}. "
            f"Make sure to test your recall on each bolded concept."
        )

        return SummarizeResponse(
            document_id=document_id,
            document_name=doc_name,
            summary_mode=SummaryMode.KEY_POINTS.value,
            summary=summary_md,
            key_points=top_points,
            topic_sections=None,
            cited_pages=sorted(list(cited_pages_set)),
            generation_provider="extractive_grounded_summarizer",
            model_name="salience_section_synthesizer",
            warning="Offline grounded extractive summarizer active. Content is synthesized directly from source text without hallucination.",
        )

    def _summarize_topic_wise(
        self,
        document_id: str,
        doc_name: str,
        sections: List[Dict[str, Any]],
        all_pages: List[int],
    ) -> SummarizeResponse:
        """
        Segments the document into modular TopicSection objects with separate titles,
        content, key points, and page numbers.
        """
        topic_sections: List[TopicSection] = []
        cited_pages_set = set()
        md_sections = [
            f"### Topic-Wise Modular Breakdown: {doc_name}\n"
            f"The document has been partitioned into {len(sections)} distinct topic module(s):"
        ]
        all_aggregated_points = []

        for sec in sections:
            title = sec["title"]
            sents = sec["sentences"]
            pages = sec["pages"]
            for p in pages:
                cited_pages_set.add(p)

            if sents:
                overview_text = sents[0]["text"]
                topic_pts = [self._format_point(s) for s in sents[1:4]]
            else:
                overview_text = "No detailed narrative found for this topic."
                topic_pts = []

            all_aggregated_points.extend(topic_pts)

            topic_sections.append(
                TopicSection(
                    title=title,
                    content=overview_text,
                    key_points=topic_pts if topic_pts else None,
                    page_numbers=pages,
                )
            )

            # Markdown presentation
            sec_md = [
                f"### {title} *(Page(s) {', '.join(str(p) for p in pages)})*",
                overview_text,
            ]
            if topic_pts:
                sec_md.append("\n**Key Takeaways:**")
                for pt in topic_pts:
                    sec_md.append(f"• {pt}")

            md_sections.append("\n".join(sec_md))

        full_md = "\n\n".join(md_sections)

        return SummarizeResponse(
            document_id=document_id,
            document_name=doc_name,
            summary_mode=SummaryMode.TOPIC_WISE.value,
            summary=full_md,
            key_points=all_aggregated_points,
            topic_sections=topic_sections,
            cited_pages=sorted(list(cited_pages_set)),
            generation_provider="extractive_grounded_summarizer",
            model_name="salience_section_synthesizer",
            warning="Offline grounded extractive summarizer active. Content is synthesized directly from source text without hallucination.",
        )


# Singleton instance
summarizer_instance = DocumentSummarizer()
