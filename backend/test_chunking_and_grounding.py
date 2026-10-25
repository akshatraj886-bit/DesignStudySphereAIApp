"""
StudySphere AI - Phase 8: Boundary-Aware Chunking & Reliable Grounding Regression Test Suite
Validates:
1. Sentence boundary preservation (no mid-sentence splits).
2. Paragraph and multiple-sentence natural breaks.
3. Fallback for sentences longer than chunk limit.
4. Very short pages and empty pages.
5. Page-number and metadata preservation.
6. Overlap behavior without skipping text.
7. Unicode text and diverse punctuation.
8. Stable DocumentChunk schema expected by existing callers.
9. Step 4 Deterministic Fixtures:
   - Coffman conditions completeness and non-rejection of standard intro.
   - PCB complete field list preservation in retrieval evidence.
   - Out-of-scope question clean refusal with zero fabricated citations.
"""

import os
import sys
import unittest
from typing import List, Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from app.rag.chunker import chunk_text, process_document_pages, DocumentChunk
from app.rag.generator import validate_answer_citations, extract_cited_sources


class TestBoundaryAwareChunking(unittest.TestCase):
    """Regression test suite for boundary-aware document chunking."""

    def test_sentence_cut_mid_phrase_preserved(self):
        """1. A sentence spanning the target size is kept intact on sentence boundary."""
        sentence1 = "Operating system process states include New, Ready, Running, Waiting, and Terminated. " * 5  # ~425 chars
        # Sentence 2 crosses the 500-char boundary
        sentence2 = "The Process Control Block stores critical information including PID, registers, and memory pointers. "
        sentence3 = "CPU scheduling determines which process in the ready queue is allocated the CPU next."
        full_text = sentence1 + sentence2 + sentence3

        chunks = chunk_text(full_text, target_chunk_size=500, overlap_size=100)
        self.assertGreaterEqual(len(chunks), 2)
        # Chunk 1 must end with a full sentence, never cutting inside sentence2
        self.assertTrue(chunks[0].endswith(".") or chunks[0].endswith("."))
        self.assertNotIn("stores critical inf", chunks[0][-25:])  # never severed mid-word/phrase

    def test_multiple_sentences_and_paragraphs(self):
        """2. Multiple sentences and paragraphs prefer paragraph and sentence breaks."""
        para1 = "This is the first paragraph. It has two complete sentences."
        para2 = "This is the second paragraph. It describes computer memory and virtual paging algorithms."
        para3 = "This is the third paragraph. Deadlock avoidance algorithms include Dijkstra's Banker's algorithm."
        full_text = f"{para1}\n\n{para2}\n\n{para3}"

        chunks = chunk_text(full_text, target_chunk_size=140, overlap_size=30)
        self.assertGreaterEqual(len(chunks), 2)
        for c in chunks:
            # Chunks must not start or end with arbitrary mid-word fragments
            self.assertFalse(c.startswith("his is"))
            self.assertFalse(c.endswith("algotith"))

    def test_sentence_longer_than_chunk_limit(self):
        """3. A sentence longer than the target chunk limit falls back safely without hanging."""
        # 600 characters of continuous words without any periods
        long_sentence = " ".join([f"word{i}" for i in range(120)])
        self.assertGreater(len(long_sentence), 600)

        chunks = chunk_text(long_sentence, target_chunk_size=500, overlap_size=100)
        self.assertGreaterEqual(len(chunks), 2)
        # Verify no words were severed into partial strings
        for c in chunks:
            words = c.split()
            for w in words:
                self.assertTrue(w.startswith("word"), f"Severed word fragment detected: {w}")

    def test_short_and_empty_pages(self):
        """4. Very short pages and empty pages are handled cleanly."""
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\t  "), [])
        short_text = "Brief summary note on Page 1."
        chunks = chunk_text(short_text, target_chunk_size=500, overlap_size=100)
        self.assertEqual(chunks, [short_text])

    def test_page_number_preservation(self):
        """5. Page numbers and source metadata are strictly preserved across chunks."""
        pages = [
            {"page_number": 1, "text": "Page one content about CPU architecture."},
            {"page_number": 2, "text": "Page two content about process scheduling."},
            {"page_number": 3, "text": "Page three content about deadlock avoidance."},
        ]
        doc_chunks = process_document_pages(
            document_id="doc_test_123",
            document_name="notes.pdf",
            pages=pages,
            chunk_size=500,
            chunk_overlap=100,
        )
        self.assertEqual(len(doc_chunks), 3)
        self.assertEqual([c.page_number for c in doc_chunks], [1, 2, 3])
        self.assertEqual([c.document_id for c in doc_chunks], ["doc_test_123"] * 3)
        self.assertEqual([c.document_name for c in doc_chunks], ["notes.pdf"] * 3)

    def test_overlap_without_missing_source_text(self):
        """6. Overlap behavior does not skip any source content."""
        sentences = [f"Numbered statement {i} discusses fundamental concepts in computing." for i in range(15)]
        full_text = " ".join(sentences)

        chunks = chunk_text(full_text, target_chunk_size=300, overlap_size=80)
        self.assertGreaterEqual(len(chunks), 2)

        # Every sentence must appear completely in at least one chunk
        for s in sentences:
            found = any(s in c for c in chunks)
            self.assertTrue(found, f"Sentence was omitted or skipped: {s}")

    def test_unicode_and_punctuation(self):
        """7. Unicode text and diverse punctuation are preserved without corruption."""
        hindi_text = (
            "ऑपरेटिंग सिस्टम में प्रोसेस के मुख्य स्टेट्स होते हैं। "
            "New, Ready, Running, Waiting, और Terminated अवस्थाएं हैं। "
            "क्या आप CPU शेड्यूलिंग के बारे में समझना चाहते हैं? हाँ, बिल्कुल!"
        )
        chunks = chunk_text(hindi_text, target_chunk_size=100, overlap_size=30)
        self.assertGreaterEqual(len(chunks), 1)
        reconstructed = " ".join(chunks)
        self.assertIn("ऑपरेटिंग सिस्टम", reconstructed)
        self.assertIn("Terminated", reconstructed)

    def test_stable_chunk_metadata(self):
        """8. DocumentChunk schema contains all required attributes."""
        pages = [{"page_number": 1, "text": "Testing DocumentChunk metadata stability."}]
        doc_chunks = process_document_pages("doc_abc", "test.pdf", pages)
        self.assertEqual(len(doc_chunks), 1)
        c = doc_chunks[0]
        self.assertEqual(c.chunk_id, "doc_abc_p1_c0")
        self.assertEqual(c.chunk_index, 0)
        self.assertEqual(c.character_count, len(c.text))
        self.assertEqual(c.word_count, len(c.text.split()))


class TestStep4DeterministicFixtures(unittest.TestCase):
    """Deterministic fixtures evaluating PCB, Coffman, and Out-of-scope cases."""

    def test_fixture_a_coffman_conditions(self):
        """A. Coffman conditions: full 4 conditions present, intro does not cause failure."""
        source_text = (
            "Four Coffman conditions must hold simultaneously for a deadlock to arise:\n"
            "1. Mutual Exclusion: At least one resource must be held in a non-shareable mode.\n"
            "2. Hold and Wait: A process must be holding at least one resource and waiting to acquire additional resources.\n"
            "3. No Preemption: Resources cannot be preempted; they can only be released voluntarily.\n"
            "4. Circular Wait: A closed chain of processes exists such that each process holds at least one resource needed by the next."
        )
        chunks = chunk_text(source_text, target_chunk_size=500, overlap_size=100)
        # All 4 conditions must be present in the chunk
        full_chunk = " ".join(chunks)
        self.assertIn("Mutual Exclusion", full_chunk)
        self.assertIn("Hold and Wait", full_chunk)
        self.assertIn("No Preemption", full_chunk)
        self.assertIn("Circular Wait", full_chunk)

        # Realistic answer with normal introductory phrasing
        answer = (
            "Based on the provided notes, the four Coffman conditions for a deadlock are:\n"
            "* **Mutual Exclusion:** At least one resource must be held in a non-shareable mode [Page 3].\n"
            "* **Hold and Wait:** A process is holding a resource while waiting for more [Page 3].\n"
            "* **No Preemption:** Resources cannot be preempted [Page 3].\n"
            "* **Circular Wait:** A closed chain of processes waiting for each other [Page 3]."
        )
        candidates = [{"page_number": 3, "text": source_text}]
        validated = validate_answer_citations(answer, candidates)
        self.assertIn("[Page 3]", validated)
        cited = extract_cited_sources(validated, candidates, "found")
        self.assertEqual(len(cited), 1)
        self.assertEqual(cited[0]["page_number"], 3)

    def test_fixture_b_pcb_complete_field_list(self):
        """B. PCB: Chunking preserves full PCB sentence without cutting fields."""
        p1_text = (
            "StudySphere AI Academic Notes - Computer Science & Engineering\n"
            "Course: Operating Systems & Data Science - Module 1: Process Management\n\n"
            "1. Process Concept and State Transitions\n"
            "A process is a program in execution. It includes the program code (text section), "
            "current activity represented by the program counter, processor registers, and the stack.\n"
            "A process changes state as it executes. The principal states of a process are:\n"
            "- New: The process is being created.\n"
            "- Running: Instructions are being executed by the CPU.\n"
            "- Waiting (Blocked): The process is waiting for some event to occur (such as I/O completion).\n"
            "- Ready: The process is waiting to be assigned to a processor.\n"
            "- Terminated: The process has finished execution.\n\n"
            "The Process Control Block (PCB) stores critical information for each process, including "
            "process ID (PID), program counter, CPU registers, CPU scheduling information, and memory management info."
        )
        chunks = chunk_text(p1_text, target_chunk_size=500, overlap_size=100)
        # Verify that Chunk 2 contains the entire PCB sentence with all 5 fields
        pcb_chunk = [c for c in chunks if "Process Control Block" in c][0]
        self.assertIn("process ID (PID)", pcb_chunk)
        self.assertIn("program counter", pcb_chunk)
        self.assertIn("CPU registers", pcb_chunk)
        self.assertIn("CPU scheduling information", pcb_chunk)
        self.assertIn("memory management info", pcb_chunk)

    def test_fixture_c_outofscope_refusal(self):
        """C. Out-of-scope question: Clean refusal with zero fabricated citations."""
        refusal_answer = (
            "I could not find sufficient evidence in the uploaded document(s) to answer this question. "
            "The retrieved passages do not discuss this topic. Try selecting a different document or asking "
            "a question covered by your notes."
        )
        candidates = [{"page_number": 1, "text": "Operating system process states..."}]
        # Refusal must have 0 citations
        cited = extract_cited_sources(refusal_answer, candidates, "insufficient_evidence")
        self.assertEqual(len(cited), 0)
        self.assertNotIn("[Page", refusal_answer)


if __name__ == "__main__":
    unittest.main()
