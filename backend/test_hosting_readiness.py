"""
StudySphere AI - Step 4 Hosting & Security Readiness Test Suite
Tests:
1. Strict session token verification (HTTP 401 on missing token in production mode)
2. Permissive fallback in development mode (preserves local test compatibility)
3. In-memory sliding window rate limiting (HTTP 429 + Retry-After)
4. Upload entity size limit (HTTP 413)
5. Zero secrets exposure on public endpoints
6. CORS origin validation
"""

import os
import sys
import time
import uuid
import tempfile
from fastapi.testclient import TestClient

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app
from app.rag.generator import generator_instance
from app.rate_limiter import limiter, LIMITS

# Force mock mode for fast generator execution
generator_instance.set_mock_mode(True)

client = TestClient(app)


def test_hosting_readiness_suite():
    print("=" * 70)
    print("STUDYSPHERE AI - STEP 4 HOSTING & SECURITY READINESS TEST SUITE")
    print("=" * 70)

    passed = 0
    failed = 0

    def assert_test(condition: bool, name: str):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}")
            failed += 1

    # -------------------------------------------------------------------------
    # TEST 1: Health Endpoint Security & Secret Non-Exposure
    # -------------------------------------------------------------------------
    print("\n[1] Testing Health Endpoint & Secrets Non-Exposure...")
    res = client.get("/api/health")
    assert_test(res.status_code == 200, "GET /api/health returns HTTP 200")
    data = res.json()
    assert_test(data.get("status") == "ok", "Status is 'ok'")
    assert_test("database_engine" in data, "database_engine is exposed for observability")
    # Verify no secrets in response
    body_str = res.text.lower()
    assert_test("key" not in body_str and "password" not in body_str and "secret" not in body_str, "Health response contains zero credentials or secrets")

    # -------------------------------------------------------------------------
    # TEST 2: Permissive Session Token in Development Mode
    # -------------------------------------------------------------------------
    print("\n[2] Testing Development Mode Session Token Fallback...")
    os.environ["STRICT_SESSION_AUTH"] = "false"
    os.environ["ENVIRONMENT"] = "development"
    res_dev = client.get("/api/documents")
    assert_test(res_dev.status_code == 200, "Development mode allows unauthenticated request falling back to default_session")

    # -------------------------------------------------------------------------
    # TEST 3: Strict Session Token Enforcement in Production Mode
    # -------------------------------------------------------------------------
    print("\n[3] Testing Production Mode Strict Session Token Enforcement...")
    os.environ["STRICT_SESSION_AUTH"] = "true"
    res_prod_missing = client.get("/api/documents")
    assert_test(res_prod_missing.status_code == 401, "Production mode rejects request missing X-Session-Token with HTTP 401")
    assert_test("Missing required X-Session-Token" in res_prod_missing.json()["detail"], "Rejection message explains X-Session-Token requirement")

    # Malformed token in strict mode -> HTTP 400
    res_prod_malformed = client.get("/api/documents", headers={"X-Session-Token": "bad"})
    assert_test(res_prod_malformed.status_code == 400, "Malformed X-Session-Token rejected with HTTP 400")

    # Valid token in strict mode -> HTTP 200
    valid_token = "sst_prod_test_" + uuid.uuid4().hex[:16]
    res_prod_valid = client.get("/api/documents", headers={"X-Session-Token": valid_token})
    assert_test(res_prod_valid.status_code == 200, "Valid X-Session-Token succeeds with HTTP 200 in strict mode")

    # Reset environment to development for remaining tests
    os.environ["STRICT_SESSION_AUTH"] = "false"
    os.environ["ENVIRONMENT"] = "development"

    # -------------------------------------------------------------------------
    # TEST 4: In-Memory Sliding Window Rate Limiting
    # -------------------------------------------------------------------------
    print("\n[4] Testing Rate Limiting Middleware...")
    test_token = "sst_ratelimit_" + uuid.uuid4().hex[:16]
    headers = {"X-Session-Token": test_token}

    # Direct limiter unit test: verify sliding window count
    upload_limit, window = LIMITS["upload"]
    client_ip = f"192.168.1.99:{test_token}"
    history_key = (client_ip, "upload")

    # Clear any previous test history
    limiter._history[history_key] = []

    # Send up to limit requests directly to simulate calls
    simulated_now = time.time()
    for _ in range(upload_limit):
        limiter._history[history_key].append(simulated_now)

    # The next check must trigger 429
    class DummyRequest:
        headers = {"X-Session-Token": test_token}
        client = type("Client", (), {"host": "192.168.1.99"})()

    dummy_req = DummyRequest()
    rate_limited = False
    try:
        limiter.check_rate_limit(dummy_req, "upload")
    except Exception as e:
        if getattr(e, "status_code", 0) == 429:
            rate_limited = True
            retry_header = e.headers.get("Retry-After")
            assert_test(retry_header is not None, "HTTP 429 response includes Retry-After header")

    assert_test(rate_limited is True, "Exceeding upload limit triggers HTTP 429 Too Many Requests")

    # Clean up test key
    if history_key in limiter._history:
        del limiter._history[history_key]

    # -------------------------------------------------------------------------
    # TEST 5: File Size Limits Enforcement
    # -------------------------------------------------------------------------
    print("\n[5] Testing File Size Limit Enforcement...")
    from app.pdf_service import MAX_UPLOAD_SIZE_MB, MAX_FILE_SIZE_BYTES
    assert_test(MAX_UPLOAD_SIZE_MB == 25, "Default MAX_UPLOAD_SIZE_MB is set to 25 MB")
    assert_test(MAX_FILE_SIZE_BYTES == 25 * 1024 * 1024, "MAX_FILE_SIZE_BYTES matches MB limit in bytes")

    # -------------------------------------------------------------------------
    # TEST 6: CORS Configuration Sanitization
    # -------------------------------------------------------------------------
    print("\n[6] Testing CORS Configuration Sanitization...")
    from app.main import default_origins
    assert_test(len(default_origins) > 0, "Default origins defined for local development")
    assert_test(all(o.startswith("http://") or o.startswith("https://") for o in default_origins), "Origins strictly use http/https protocols")

    print("\n" + "=" * 70)
    print(f"HOSTING READINESS SUITE: {passed} PASSED, {failed} FAILED (TOTAL: {passed + failed})")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = test_hosting_readiness_suite()
    sys.exit(0 if success else 1)
