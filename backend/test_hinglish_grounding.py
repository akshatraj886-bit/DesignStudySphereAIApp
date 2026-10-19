"""
StudySphere AI - Regression Test Suite for Hinglish Grounded QA & Insufficient Evidence Fix
Validates:
1. Exact Hinglish query:
   "Normalization kya hota hai? Iska main objective aur insert, update, delete anomalies simple language me samjhao."
   returns a useful grounded answer citing Page 1, retrieval_status == "found",
   and cited_sources properly separated from retrieved_sources.
2. Unrelated questions (Roman Empire & Hinglish Taj Mahal) correctly return
   retrieval_status == "insufficient_evidence" with 0 cited sources.
3. extract_cited_sources unit tests:
   - Strictly extracts only chunks actually referenced or matching claims.
   - Ensures cited_sources is empty when retrieval_status is 'insufficient_evidence'.
4. is_refusal_response unit tests.
"""

import os
import sys
import pymupdf
from fastapi.testclient import TestClient

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from app.main import app
from app.rag.generator import (
    generator_instance,
    extract_substantive_query_terms,
    verify_topical_evidence,
    extract_cited_sources,
    is_refusal_response,
    is_hinglish_or_hindi_query,
)


def create_normalization_notes_pdf() -> bytes:
    """Generates a realistic 2-page academic PDF on Normalization."""
    doc = pymupdf.open()

    # Page 1: Normalization Fundamentals & Objectives
    page1 = doc.new_page(width=595, height=842)
    p1_text = (
        "StudySphere AI Academic Notes - Database Management Systems\n"
        "Course: Database Management Systems - Module 4: Relational Normalization\n\n"
        "1. Concept of Normalization and Objectives\n"
        "Normalization is the process of organizing data in a database. It includes creating tables "
        "and establishing relationships between those tables according to rules designed both to protect "
        "the data and to make the database more flexible by eliminating redundancy and inconsistent dependency.\n\n"
        "Main Objectives of Normalization:\n"
        "- Eliminate redundant data (for example, storing duplicate customer information across multiple tables).\n"
        "- Ensure data dependencies make sense (only storing related data within a given table).\n"
        "- Prevent and avoid insert, update, and delete anomalies in database transactions.\n\n"
        "When tables are properly normalized, data consistency is maintained and disk storage is optimized."
    )
    page1.insert_text((50, 60), p1_text, fontsize=11)

    # Page 2: Normal Forms Overview (1NF, 2NF, 3NF)
    page2 = doc.new_page(width=595, height=842)
    p2_text = (
        "Database Management Systems - Module 4: Relational Normalization\n\n"
        "2. Overview of Standard Normal Forms\n"
        "- First Normal Form (1NF): Requires that the domain of each attribute must contain only atomic (indivisible) values, "
        "and the value of each attribute contains only a single value from that domain.\n"
        "- Second Normal Form (2NF): Must be in 1NF and all non-key attributes must be fully functionally dependent on the primary key.\n"
        "- Third Normal Form (3NF): Must be in 2NF and have no transitive dependencies on the primary key."
    )
    page2.insert_text((50, 60), p2_text, fontsize=11)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_refusal_detection_helpers():
    """Validates refusal detection and Hinglish detection helper functions."""
    assert is_refusal_response("I could not find sufficient evidence in the uploaded document(s) to answer this question.") is True
    assert is_refusal_response("The provided passages do not contain enough information to answer this question.") is True
    assert is_refusal_response("There is insufficient evidence to answer your query.") is True
    assert is_refusal_response("### Overview\nNormalization is the process of organizing data [Page 1].") is False

    assert is_hinglish_or_hindi_query("Normalization kya hota hai? Samjhao.") is True
    assert is_hinglish_or_hindi_query("What is the primary objective of Normalization?") is False


def test_cited_sources_separation_unit():
    """Validates that extract_cited_sources keeps cited and retrieved candidates strictly separate."""
    candidates = [
        {
            "document_id": "doc_1",
            "document_name": "Normalization.pptx",
            "page_number": 1,
            "chunk_index": 0,
            "similarity_score": 0.88,
            "excerpt": "Normalization is the process of organizing data in a database. Avoid insert, update, delete anomalies.",
        },
        {
            "document_id": "doc_1",
            "document_name": "Normalization.pptx",
            "page_number": 2,
            "chunk_index": 1,
            "similarity_score": 0.74,
            "excerpt": "First Normal Form (1NF) requires atomic values.",
        },
        {
            "document_id": "doc_1",
            "document_name": "Normalization.pptx",
            "page_number": 2,
            "chunk_index": 2,
            "similarity_score": 0.70,
            "excerpt": "Second Normal Form (2NF) eliminates partial dependency.",
        },
    ]

    # Test 1: Answer explicitly citing Page 1
    answer = (
        "### Overview\n"
        "**Normalization** database me data organize karne ka process hai [Page 1].\n"
        "Iska objective redundancy aur insert, update, delete anomalies ko eliminate karna hai [Page 1]."
    )
    cited = extract_cited_sources(answer, candidates, "found")
    assert len(cited) == 1
    assert cited[0]["page_number"] == 1
    assert cited[0]["chunk_index"] == 0

    # Test 2: Insufficient evidence status must return empty cited list
    refusal_answer = "I could not find sufficient evidence in the uploaded document(s) to answer this question."
    cited_empty = extract_cited_sources(refusal_answer, candidates, "insufficient_evidence")
    assert len(cited_empty) == 0


def test_end_to_end_hinglish_normalization_qa():
    """
    End-to-end regression test:
    Uploads Normalization document, asks exact Hinglish query, and verifies grounded answer.
    """
    client = TestClient(app)

    # 1. Upload Normalization document
    pdf_bytes = create_normalization_notes_pdf()
    files = {"file": ("Normalization_Academic_Notes.pdf", pdf_bytes, "application/pdf")}
    res = client.post("/api/documents/upload", files=files)
    assert res.status_code == 200
    doc_id = res.json()["document_id"]

    # 2. Ask exact Hinglish query
    hinglish_query = {
        "question": "Normalization kya hota hai? Iska main objective aur insert, update, delete anomalies simple language me samjhao.",
        "document_id": doc_id,
        "top_k": 3,
    }
    res_ask = client.post("/api/ask", json=hinglish_query)
    assert res_ask.status_code == 200
    data = res_ask.json()

    # Verify retrieval status is 'found' (NOT 'insufficient_evidence')
    assert data["retrieval_status"] == "found", f"Expected 'found', got {data['retrieval_status']}"

    # Verify cited sources exist and cite Page 1
    assert len(data["cited_sources"]) > 0, "Expected non-empty cited_sources"
    assert all(cs["page_number"] == 1 for cs in data["cited_sources"]), "Citations should be strictly Page 1"

    # Verify answer contains Page 1 citation and addresses the concepts
    ans_text = data["answer"]
    assert "Page 1" in ans_text or "[Page 1]" in ans_text, "Answer must cite Page 1"
    assert "Normalization" in ans_text or "normalization" in ans_text
    assert "redundancy" in ans_text.lower() or "redundant" in ans_text.lower()
    assert "anomal" in ans_text.lower()

    # Verify candidate sources are present and separated
    assert len(data["retrieved_sources"]) >= len(data["cited_sources"])

    # 3. Ask unrelated query (Roman Empire)
    unrelated_query = {
        "question": "Explain the history of the Roman Empire using my uploaded notes",
        "document_id": doc_id,
        "top_k": 3,
    }
    res_unrelated = client.post("/api/ask", json=unrelated_query)
    assert res_unrelated.status_code == 200
    data_unrelated = res_unrelated.json()
    assert data_unrelated["retrieval_status"] == "insufficient_evidence"
    assert len(data_unrelated["cited_sources"]) == 0

    # 4. Ask Hinglish unrelated query (Taj Mahal)
    hinglish_unrelated = {
        "question": "Taj Mahal kisne banwaya tha aur kab bana tha detail me samjhao",
        "document_id": doc_id,
        "top_k": 3,
    }
    res_taj = client.post("/api/ask", json=hinglish_unrelated)
    assert res_taj.status_code == 200
    data_taj = res_taj.json()
    assert data_taj["retrieval_status"] == "insufficient_evidence"
    assert len(data_taj["cited_sources"]) == 0


if __name__ == "__main__":
    print("Running Hinglish Grounding & Citation Separation Tests...")
    test_refusal_detection_helpers()
    print("  [PASS] test_refusal_detection_helpers")
    test_cited_sources_separation_unit()
    print("  [PASS] test_cited_sources_separation_unit")
    test_end_to_end_hinglish_normalization_qa()
    print("  [PASS] test_end_to_end_hinglish_normalization_qa")
    print("All tests passed successfully!")
