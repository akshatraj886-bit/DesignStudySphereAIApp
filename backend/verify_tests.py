"""
Automated Verification Suite for StudySphere AI Backend (Phase 1)
Tests all requirements against the live running FastAPI server on http://127.0.0.1:8000.
Uses Python's standard library urllib.request and PyMuPDF to construct genuine test files.
"""
import io
import json
import sys
import urllib.error
import urllib.request
import pymupdf

# Fix Windows console encoding for Unicode output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


BASE_URL = "http://127.0.0.1:8000"


def make_multipart_body(field_name: str, filename: str, content: bytes, content_type: str = "application/pdf"):
    boundary = "----WebKitFormBoundaryStudySphereTest7MA4YWxkTrZu0gW"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8"))
    body.write(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.write(content)
    body.write(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body.getvalue())),
    }
    return body.getvalue(), headers


def test_1_health():
    print("▶ Running Test 1: GET /api/health...")
    req = urllib.request.Request(f"{BASE_URL}/api/health", method="GET")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        data = json.loads(resp.read().decode("utf-8"))
        assert data.get("status") == "ok", f"Expected status 'ok', got {data}"
        assert "StudySphere AI Backend" in data.get("service", ""), f"Unexpected service name: {data}"
        print(f"  ✓ Passed: Status={resp.status}, Service='{data.get('service')}'")


def test_2_valid_multipage_pdf():
    print("\n▶ Running Test 2: POST /api/documents/upload with valid multi-page PDF...")
    # Create an in-memory 2-page PDF with pymupdf
    doc = pymupdf.open()
    
    # Page 1
    page1 = doc.new_page()
    page1.insert_text((50, 72), "StudySphere AI - DBMS Lecture Notes")
    page1.insert_text((50, 100), "Chapter 3: Normalization and Functional Dependencies.")
    
    # Page 2
    page2 = doc.new_page()
    page2.insert_text((50, 72), "First Normal Form (1NF) requires atomic attribute values.")
    page2.insert_text((50, 100), "Second Normal Form (2NF) eliminates partial dependencies.")
    
    pdf_bytes = doc.tobytes()
    doc.close()
    
    body, headers = make_multipart_body("file", "dbms_lecture_unit3.pdf", pdf_bytes)
    req = urllib.request.Request(f"{BASE_URL}/api/documents/upload", data=body, headers=headers, method="POST")
    
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        res = json.loads(resp.read().decode("utf-8"))
        assert res["filename"] == "dbms_lecture_unit3.pdf"
        assert res["page_count"] == 2
        assert len(res["pages"]) == 2
        assert res["pages"][0]["page_number"] == 1
        assert "Normalization" in res["pages"][0]["text"]
        assert res["pages"][1]["page_number"] == 2
        assert "First Normal Form" in res["pages"][1]["text"]
        assert res["has_extractable_text"] is True
        assert res["ocr_suggested"] is False
        assert res["total_characters"] > 50
        assert res["total_words"] > 10
        print(f"  ✓ Passed: Extracted {res['page_count']} pages, {res['total_words']} words, {res['total_characters']} characters.")
        print(f"    Page 1 excerpt: '{res['pages'][0]['text'][:40]}...'")
        print(f"    Page 2 excerpt: '{res['pages'][1]['text'][:40]}...'")


def test_3_invalid_file_extension():
    print("\n▶ Running Test 3: Reject unsupported file extension (.zip)...")
    content = b"This is just a zip archive, not a document or note."
    body, headers = make_multipart_body("file", "notes.zip", content, "application/zip")
    req = urllib.request.Request(f"{BASE_URL}/api/documents/upload", data=body, headers=headers, method="POST")
    
    try:
        urllib.request.urlopen(req)
        assert False, "Expected 400 Bad Request, but request succeeded."
    except urllib.error.HTTPError as e:
        assert e.code == 400, f"Expected 400, got {e.code}"
        err = json.loads(e.read().decode("utf-8"))
        print(f"  ✓ Passed: Correctly returned HTTP 400 - {err.get('detail')}")


def test_4_fake_pdf_header():
    print("\n▶ Running Test 4: Reject fake PDF (named .pdf but missing %PDF- magic bytes)...")
    content = b"This file is pretending to be a PDF but has no PDF header."
    body, headers = make_multipart_body("file", "imposter.pdf", content)
    req = urllib.request.Request(f"{BASE_URL}/api/documents/upload", data=body, headers=headers, method="POST")
    
    try:
        urllib.request.urlopen(req)
        assert False, "Expected 400 Bad Request, but request succeeded."
    except urllib.error.HTTPError as e:
        assert e.code == 400, f"Expected 400, got {e.code}"
        err = json.loads(e.read().decode("utf-8"))
        print(f"  ✓ Passed: Correctly returned HTTP 400 - {err.get('detail')}")


def test_5_empty_file():
    print("\n▶ Running Test 5: Reject empty 0-byte file...")
    body, headers = make_multipart_body("file", "empty.pdf", b"")
    req = urllib.request.Request(f"{BASE_URL}/api/documents/upload", data=body, headers=headers, method="POST")
    
    try:
        urllib.request.urlopen(req)
        assert False, "Expected 400 Bad Request, but request succeeded."
    except urllib.error.HTTPError as e:
        assert e.code == 400, f"Expected 400, got {e.code}"
        err = json.loads(e.read().decode("utf-8"))
        print(f"  ✓ Passed: Correctly returned HTTP 400 - {err.get('detail')}")


def test_6_scanned_image_pdf_ocr_flag():
    print("\n▶ Running Test 6: Detect image-only / blank PDF and flag OCR recommendation...")
    # Create a 1-page PDF without any text
    doc = pymupdf.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    
    body, headers = make_multipart_body("file", "scanned_handwritten_diagram.pdf", pdf_bytes)
    req = urllib.request.Request(f"{BASE_URL}/api/documents/upload", data=body, headers=headers, method="POST")
    
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        res = json.loads(resp.read().decode("utf-8"))
        assert res["page_count"] == 1
        assert res["total_characters"] == 0
        assert res["has_extractable_text"] is False
        assert res["ocr_suggested"] is True
        assert "OCR" in res["message"]
        print(f"  ✓ Passed: Returned HTTP 200 with OCR flag: has_extractable_text={res['has_extractable_text']}, ocr_suggested={res['ocr_suggested']}")
        print(f"    Message: '{res['message']}'")


def test_7_corrupted_pdf():
    print("\n▶ Running Test 7: Reject corrupted PDF structure...")
    # Starts with %PDF- but followed by broken truncated garbage
    content = b"%PDF-1.4 \n corrupted binary junk that cannot be parsed by any pdf engine \x00\xff\xfe"
    body, headers = make_multipart_body("file", "corrupt.pdf", content)
    req = urllib.request.Request(f"{BASE_URL}/api/documents/upload", data=body, headers=headers, method="POST")
    
    try:
        urllib.request.urlopen(req)
        assert False, "Expected 400 Bad Request, but request succeeded."
    except urllib.error.HTTPError as e:
        assert e.code == 400, f"Expected 400, got {e.code}"
        err = json.loads(e.read().decode("utf-8"))
        print(f"  ✓ Passed: Correctly caught corruption: HTTP 400 - {err.get('detail')}")


if __name__ == "__main__":
    print("=" * 60)
    print("StudySphere AI Phase 1 Backend Verification Suite")
    print("=" * 60)
    test_1_health()
    test_2_valid_multipage_pdf()
    test_3_invalid_file_extension()
    test_4_fake_pdf_header()
    test_5_empty_file()
    test_6_scanned_image_pdf_ocr_flag()
    test_7_corrupted_pdf()
    print("=" * 60)
    print("ALL 7 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
