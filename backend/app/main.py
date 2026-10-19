import os
import re
import json
import asyncio
import logging
import time
import uuid
from typing import Optional, Union, List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Header, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

# Reliable local .env configuration loading
load_dotenv()

from app.schemas import (
    HealthResponse,
    PDFUploadResponse,
    ErrorResponse,
    AskRequest,
    AskResponse,
    SourceChunk,
    DocumentItem,
    DocumentListResponse,
    SummarizeRequest,
    SummarizeResponse,
    SummaryMode,
    TopicSection,
    QuestionType,
    QuestionDifficulty,
    QuestionItem,
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    QuestionAnswerSubmission,
    EvaluateQuizRequest,
    QuestionEvaluationResult,
    EvaluateQuizResponse,
    CreateChatSessionRequest,
    ChatSessionItem,
    ChatSessionListResponse,
    ChatMessageItem,
    ChatMessageListResponse,
    SaveChatMessageRequest,
    DeleteChatSessionResponse,
    ChatAskRequest,
    ChatAskResponse,
)
from app.pdf_service import process_pdf_upload
from app.rag.store import store_instance
from app.rag.embeddings import embedding_service
from app.rag.generator import generator_instance, resolve_retrieval_query
from app.rag.summarizer import summarizer_instance
from app.rag.question_generator import question_generator_instance
from app.rag.db_adapter import get_engine_type
from app.rate_limiter import limiter

logger = logging.getLogger("studysphere.api")

# Initialize the FastAPI application
app = FastAPI(
    title="StudySphere AI Backend",
    version="1.2.0",
    description="Backend service providing real PDF text extraction, persistent vector indexing, grounded RAG question answering, and multi-tenant session privacy for StudySphere AI.",
    docs_url="/docs",
    redoc_url="/redoc",
)

@app.on_event("startup")
async def startup_warmup():
    """Warm up sentence embedding model asynchronously to eliminate upload cold start."""
    try:
        asyncio.create_task(run_in_threadpool(embedding_service._load_model))
        logger.info("Triggered asynchronous background warm-up of embedding model.")
    except Exception as e:
        logger.warning("Could not trigger embedding model warm-up: %s", e)

# Determine allowed origins for CORS.
default_origins = [
    "http://localhost:8443",
    "http://127.0.0.1:8443",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://design-study-sphere-ai-app.vercel.app",
]

env_origins = os.getenv("ALLOWED_ORIGINS")
allow_creds = True
if env_origins:
    raw_origins = [origin.strip() for origin in env_origins.split(",") if origin.strip()]
    if "*" in raw_origins:
        # Wildcards cannot be used with credentials
        allow_creds = False
        allowed_origins = ["*"]
    else:
        allowed_origins = [o.rstrip("/") for o in raw_origins if o.startswith("http://") or o.startswith("https://")]
else:
    allowed_origins = default_origins

from fastapi.responses import JSONResponse

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_creds,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """
    Abuse protection middleware:
    Enforces per-tier sliding window rate limits on expensive write/generation routes.
    """
    # Allow CORS preflight requests through immediately
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    if request.method == "POST":
        tier = None
        if path == "/api/documents/upload":
            tier = "upload"
        elif path == "/api/ask" or (path.startswith("/api/chat/sessions/") and path.endswith("/ask")):
            tier = "ask"
        elif "/summarize" in path or "/questions" in path:
            tier = "generate"

        if tier:
            try:
                limiter.check_rate_limit(request, tier)
            except HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"detail": exc.detail},
                    headers=dict(exc.headers or {}),
                )

    return await call_next(request)


def is_strict_session_auth() -> bool:
    """Checks whether strict session authentication is enforced."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    strict_flag = os.getenv("STRICT_SESSION_AUTH", "").lower()
    if strict_flag in ("true", "1", "yes"):
        return True
    if strict_flag in ("false", "0", "no"):
        return False
    return env == "production"


def get_session_token(
    x_session_token: Optional[str] = Header(None, alias="X-Session-Token"),
) -> str:
    """
    Extracts and validates the anonymous client session token.
    Enforces format: 8-128 alphanumeric, hyphen, or underscore characters.
    - In production or when STRICT_SESSION_AUTH=true: rejects missing tokens with HTTP 401
      and malformed tokens with HTTP 400. Never assigns shared default_session.
    - In local development/testing: falls back to 'default_session' for test/script compatibility.
    """
    if x_session_token and x_session_token.strip():
        cleaned = x_session_token.strip()
        if re.match(r"^[a-zA-Z0-9_\-]{8,128}$", cleaned):
            return cleaned
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid X-Session-Token format. Must be 8-128 alphanumeric, underscore, or hyphen characters.",
        )
    if is_strict_session_auth():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required X-Session-Token header. Every client session must provide a valid session token in production.",
        )
    return "default_session"


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """
    Returns the current operational status of the backend and AI embedding engine.
    """
    return HealthResponse(
        status="ok",
        service="StudySphere AI Backend",
        embedding_model=embedding_service.model_name,
        device=embedding_service.device,
        database_engine=get_engine_type(),
    )


@app.get(
    "/api/documents",
    response_model=DocumentListResponse,
    summary="List all indexed documents in persistent storage for caller's session",
    tags=["Documents"],
)
async def list_documents(
    session_token: str = Depends(get_session_token),
) -> DocumentListResponse:
    """
    Retrieves indexed PDF documents stored in the persistent backend SQLite store scoped to caller session.
    """
    raw_docs = store_instance.list_documents(session_token=session_token)
    items = [DocumentItem(**doc) for doc in raw_docs]
    return DocumentListResponse(documents=items, total_count=len(items))


@app.delete(
    "/api/documents/{document_id}",
    summary="Delete an indexed document from storage",
    tags=["Documents"],
)
async def delete_document(
    document_id: str,
    session_token: str = Depends(get_session_token),
):
    """
    Removes a document and its associated semantic chunk embeddings from persistent storage,
    enforcing caller session ownership.
    """
    deleted = store_instance.delete_document(document_id, session_token=session_token)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return {"status": "deleted", "document_id": document_id}


@app.post(
    "/api/documents/upload",
    response_model=PDFUploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file, empty file, or corrupted PDF"},
        413: {"model": ErrorResponse, "description": "File exceeds 50 MB size limit"},
        500: {"model": ErrorResponse, "description": "Internal server error during PDF parsing"},
    },
    status_code=status.HTTP_200_OK,
    summary="Upload, extract, chunk, embed, and index a PDF document",
    tags=["Documents"],
)
async def upload_pdf(
    file: UploadFile = File(...),
    session_token: str = Depends(get_session_token),
) -> PDFUploadResponse:
    """
    Upload a genuine PDF file:
    - Extracts text page-by-page (PyMuPDF).
    - Splits content into overlapping semantic chunks with page citations.
    - Computes dense vector embeddings with all-MiniLM-L6-v2.
    - Saves document metadata and vectors persistently in SQLite scoped to caller session.
    """
    return await process_pdf_upload(file, session_token=session_token)


@app.post(
    "/api/ask",
    response_model=None,
    responses={
        200: {"model": AskResponse},
        400: {"model": ErrorResponse, "description": "Invalid question input"},
        404: {"model": ErrorResponse, "description": "Document not found or access denied"},
        500: {"model": ErrorResponse, "description": "Internal error during RAG retrieval or generation"},
    },
    status_code=status.HTTP_200_OK,
    summary="Grounded question answering using semantic retrieval over uploaded notes",
    tags=["RAG Q&A"],
)
async def ask_question(
    request: AskRequest,
    raw_request: Request,
    stream: Optional[bool] = False,
    session_token: str = Depends(get_session_token),
):
    """
    Genuine RAG (Retrieval-Augmented Generation) Question Answering:
    1. Computes query embedding via all-MiniLM-L6-v2.
    2. Performs cosine similarity search across persistent document chunks scoped to caller session.
    3. Restricts retrieval to request.document_id if specified (and verifies ownership).
    4. Enforces strict factual grounding without hallucinations.
    5. Returns grounded answer with exact source citations and similarity scores.
    """
    cleaned_query = request.question.strip()
    if not cleaned_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or blank whitespace.",
        )

    if request.document_id:
        doc = store_instance.get_document(request.document_id, session_token=session_token)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID '{request.document_id}' not found.",
            )

    try:
        # 1. Embed query
        embedding_started = time.perf_counter()
        query_embedding = await run_in_threadpool(embedding_service.embed_query, cleaned_query)
        embedding_seconds = time.perf_counter() - embedding_started

        # 2. Retrieve top-k chunks from vector store scoped to session
        retrieval_started = time.perf_counter()
        retrieved_raw = await run_in_threadpool(
            store_instance.search_chunks,
            query_embedding=query_embedding,
            top_k=request.top_k,
            document_id=request.document_id,
            session_token=session_token,
        )
        retrieval_seconds = time.perf_counter() - retrieval_started

        # Handle overview and shortcut queries (e.g. 'Summarize this chapter', 'Give me an example', 'is pdf me kya hai btao')
        from app.rag.generator import is_overview_or_meta_query
        effective_doc_id = request.document_id
        if not effective_doc_id:
            all_docs = store_instance.list_documents(session_token=session_token)
            if len(all_docs) == 1:
                effective_doc_id = all_docs[0]["id"]

        threshold = (
            request.similarity_threshold
            if request.similarity_threshold is not None
            else generator_instance.default_threshold
        )

        if effective_doc_id and (
            is_overview_or_meta_query(cleaned_query)
            or not retrieved_raw
        ):
            doc_chunks = store_instance.get_document_chunks(effective_doc_id, session_token=session_token)
            if doc_chunks:
                seen_ids = {r.get("id") for r in retrieved_raw}
                for idx, chunk in enumerate(doc_chunks[:6]):
                    if chunk["id"] not in seen_ids:
                        chunk_copy = dict(chunk)
                        chunk_copy["similarity_score"] = max(0.72, 0.88 - (idx * 0.03))
                        retrieved_raw.append(chunk_copy)

        # 3. Format sources
        sources = [
            SourceChunk(
                document_id=r["document_id"],
                document_name=r["document_name"],
                page_number=r["page_number"],
                chunk_index=r["chunk_index"],
                similarity_score=r["similarity_score"],
                excerpt=r["text"],
            )
            for r in retrieved_raw
        ]

        # 4. Generate grounded answer
        wants_stream = stream or "text/event-stream" in raw_request.headers.get("accept", "")
        if wants_stream:
            async def event_generator():
                full_answer = ""
                final_meta = {}
                candidate_dicts = [s.model_dump() for s in sources]
                async for chunk in generator_instance.stream_answer(
                    question=cleaned_query,
                    sources=candidate_dicts,
                    similarity_threshold=threshold,
                ):
                    if "token" in chunk:
                        full_answer += chunk["token"]
                        yield f"data: {json.dumps({'token': chunk['token']})}\n\n"
                    elif chunk.get("done"):
                        final_meta = chunk

                cited_raw = final_meta.get("cited_sources", [])
                final_event = {
                    "done": True,
                    "question": cleaned_query,
                    "answer": final_meta.get("answer", full_answer),
                    "cited_sources": cited_raw,
                    "retrieved_sources": final_meta.get("retrieved_sources", candidate_dicts),
                    "sources": cited_raw,
                    "retrieval_status": final_meta.get("retrieval_status", "found"),
                    "generation_provider": final_meta.get("generation_provider", "unknown"),
                    "model_name": final_meta.get("model_name"),
                }
                yield f"data: {json.dumps(final_event)}\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        generation_started = time.perf_counter()
        gen_result = await generator_instance.generate_answer(
            question=cleaned_query,
            sources=[s.model_dump() for s in sources],
            similarity_threshold=threshold,
        )
        logger.info(
            "Question answered: provider=%s embedding_seconds=%.2f retrieval_seconds=%.2f generation_seconds=%.2f candidates=%d",
            gen_result.get("generation_provider", "unknown"),
            embedding_seconds,
            retrieval_seconds,
            time.perf_counter() - generation_started,
            len(retrieved_raw),
        )

        cited_raw = gen_result.get("cited_sources", [])
        retrieved_raw_list = gen_result.get("retrieved_sources", [s.model_dump() for s in sources])

        cited_sources = [
            SourceChunk(
                document_id=r["document_id"],
                document_name=r["document_name"],
                page_number=r["page_number"],
                chunk_index=r["chunk_index"],
                similarity_score=r["similarity_score"],
                excerpt=r.get("excerpt") or r.get("text", ""),
            )
            for r in cited_raw
        ]

        retrieved_sources = [
            SourceChunk(
                document_id=r["document_id"],
                document_name=r["document_name"],
                page_number=r["page_number"],
                chunk_index=r["chunk_index"],
                similarity_score=r["similarity_score"],
                excerpt=r.get("excerpt") or r.get("text", ""),
            )
            for r in retrieved_raw_list
        ]

        return AskResponse(
            question=cleaned_query,
            answer=gen_result["answer"],
            cited_sources=cited_sources,
            retrieved_sources=retrieved_sources,
            sources=cited_sources,
            retrieval_status=gen_result["retrieval_status"],
            generation_provider=gen_result["generation_provider"],
            model_name=gen_result.get("model_name"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during retrieval or answer generation: {str(exc)}",
        )


@app.post(
    "/api/documents/{document_id}/summarize",
    response_model=SummarizeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid mode or document contains no extractable text"},
        404: {"model": ErrorResponse, "description": "Document not found or access denied"},
        500: {"model": ErrorResponse, "description": "Internal error during summarization"},
    },
    status_code=status.HTTP_200_OK,
    summary="Generate grounded, hallucination-free summary for an indexed document",
    tags=["Summarizer"],
)
async def summarize_document(
    document_id: str,
    request: Optional[SummarizeRequest] = None,
    session_token: str = Depends(get_session_token),
) -> SummarizeResponse:
    """
    Grounded Document Summarizer:
    1. Verifies that the requested document exists and belongs to caller session.
    2. Retrieves all extracted text chunks and page information.
    3. Rejects scanned documents lacking readable text with an OCR notice.
    4. Synthesizes a factual, grounded summary across one of 4 modes.
    """
    target_mode = request.mode if request else SummaryMode.SHORT
    try:
        return summarizer_instance.summarize_document(
            document_id=document_id,
            mode=target_mode,
            session_token=session_token,
        )
    except KeyError as k_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(k_err).strip("'\""),
        )
    except ValueError as v_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(v_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during document summarization: {str(exc)}",
        )


@app.post(
    "/api/summarize",
    response_model=SummarizeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Missing document_id or invalid mode"},
        404: {"model": ErrorResponse, "description": "Document not found or access denied"},
        500: {"model": ErrorResponse, "description": "Internal error during summarization"},
    },
    status_code=status.HTTP_200_OK,
    summary="Convenience endpoint for document summarization with document_id in body",
    tags=["Summarizer"],
)
async def summarize_document_convenience(
    request: SummarizeRequest,
    session_token: str = Depends(get_session_token),
) -> SummarizeResponse:
    if not request.document_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'document_id' is required when calling /api/summarize.",
        )
    return await summarize_document(document_id=request.document_id, request=request, session_token=session_token)


@app.post(
    "/api/documents/{document_id}/questions",
    response_model=GenerateQuestionsResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid count/type/difficulty or document contains no extractable text"},
        404: {"model": ErrorResponse, "description": "Document not found or access denied"},
        500: {"model": ErrorResponse, "description": "Internal error during question generation"},
    },
    status_code=status.HTTP_200_OK,
    summary="Generate grounded academic revision questions from an indexed document",
    tags=["Revision Questions"],
)
async def generate_document_questions(
    document_id: str,
    request: Optional[GenerateQuestionsRequest] = None,
    session_token: str = Depends(get_session_token),
) -> GenerateQuestionsResponse:
    req = request or GenerateQuestionsRequest(document_id=document_id)
    try:
        return question_generator_instance.generate_questions(
            document_id=document_id,
            request=req,
            session_token=session_token,
        )
    except KeyError as k_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(k_err).strip("'\""),
        )
    except ValueError as v_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(v_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during question generation: {str(exc)}",
        )


@app.post(
    "/api/questions/generate",
    response_model=GenerateQuestionsResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Missing document_id or invalid options"},
        404: {"model": ErrorResponse, "description": "Document not found or access denied"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    status_code=status.HTTP_200_OK,
    summary="Convenience endpoint for question generation with document_id in body",
    tags=["Revision Questions"],
)
async def generate_questions_convenience(
    request: GenerateQuestionsRequest,
    session_token: str = Depends(get_session_token),
) -> GenerateQuestionsResponse:
    if not request.document_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'document_id' is required when calling /api/questions/generate.",
        )
    return await generate_document_questions(document_id=request.document_id, request=request, session_token=session_token)


@app.post(
    "/api/questions/evaluate",
    response_model=EvaluateQuizResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid submission payload, answer format, or duplicate questions"},
        404: {"model": ErrorResponse, "description": "Quiz session not found or access denied"},
        500: {"model": ErrorResponse, "description": "Internal error during quiz evaluation"},
    },
    status_code=status.HTTP_200_OK,
    summary="Evaluate student answers against the authoritative quiz session",
    tags=["Revision Questions"],
)
async def evaluate_quiz_submission(
    request: EvaluateQuizRequest,
    session_token: str = Depends(get_session_token),
) -> EvaluateQuizResponse:
    try:
        return question_generator_instance.evaluate_quiz_session(
            session_id=request.session_id,
            submissions=request.answers,
            session_token=session_token,
            document_id=request.document_id,
        )
    except KeyError as k_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(k_err).strip("'\""),
        )
    except ValueError as v_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(v_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during quiz evaluation: {str(exc)}",
        )


# =============================================================================
# CONVERSATIONAL CHAT SESSION & MESSAGE ROUTES
# =============================================================================

@app.post(
    "/api/chat/sessions",
    response_model=ChatSessionItem,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation session attached to a document",
    tags=["Conversational Chat"],
)
async def create_chat_session_endpoint(
    request: CreateChatSessionRequest,
    session_token: str = Depends(get_session_token),
) -> ChatSessionItem:
    clean_doc_id = request.document_id.strip()
    session_id = f"csess_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    try:
        session_data = store_instance.create_chat_session(
            session_id=session_id,
            document_id=clean_doc_id,
            title=request.title,
            session_token=session_token,
        )
        return ChatSessionItem(**session_data)
    except KeyError as k_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(k_err).strip("'\""),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while creating chat session: {str(exc)}",
        )


@app.get(
    "/api/chat/sessions",
    response_model=ChatSessionListResponse,
    status_code=status.HTTP_200_OK,
    summary="List chat sessions, optionally filtered by document_id",
    tags=["Conversational Chat"],
)
async def list_chat_sessions_endpoint(
    document_id: Optional[str] = None,
    session_token: str = Depends(get_session_token),
) -> ChatSessionListResponse:
    try:
        if document_id and document_id.strip():
            clean_doc_id = document_id.strip()
            doc = store_instance.get_document(clean_doc_id, session_token=session_token)
            if not doc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document with ID '{clean_doc_id}' not found.",
                )
        sessions = store_instance.list_chat_sessions(
            document_id=document_id, session_token=session_token
        )
        items = [ChatSessionItem(**s) for s in sessions]
        return ChatSessionListResponse(sessions=items, total_count=len(items))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while listing chat sessions: {str(exc)}",
        )


@app.get(
    "/api/chat/sessions/{session_id}/messages",
    response_model=ChatMessageListResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve all messages for a chat session in chronological order",
    tags=["Conversational Chat"],
)
async def get_session_messages_endpoint(
    session_id: str,
    session_token: str = Depends(get_session_token),
) -> ChatMessageListResponse:
    clean_id = session_id.strip()
    try:
        raw_messages = store_instance.get_session_messages(clean_id, session_token=session_token)
        items = [
            ChatMessageItem(
                id=m["id"],
                session_id=m["session_id"],
                role=m["role"],
                content=m["content"],
                cited_sources=[
                    SourceChunk(
                        document_id=cs.get("document_id", ""),
                        document_name=cs.get("document_name", "Document"),
                        page_number=cs.get("page_number", 1),
                        chunk_index=cs.get("chunk_index", 0),
                        similarity_score=cs.get("similarity_score", 0.0),
                        excerpt=cs.get("excerpt") or cs.get("text") or "",
                    )
                    for cs in m.get("cited_sources", [])
                    if isinstance(cs, dict)
                ],
                retrieval_status=m.get("retrieval_status"),
                generation_provider=m.get("generation_provider"),
                created_at=m["created_at"],
            )
            for m in raw_messages
        ]
        return ChatMessageListResponse(
            session_id=clean_id,
            messages=items,
            total_count=len(items),
        )
    except KeyError as k_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(k_err).strip("'\""),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving messages: {str(exc)}",
        )


@app.delete(
    "/api/chat/sessions/{session_id}",
    response_model=DeleteChatSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a chat session and all its messages",
    tags=["Conversational Chat"],
)
async def delete_chat_session_endpoint(
    session_id: str,
    session_token: str = Depends(get_session_token),
) -> DeleteChatSessionResponse:
    clean_id = session_id.strip()
    try:
        deleted = store_instance.delete_chat_session(clean_id, session_token=session_token)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session with ID '{clean_id}' not found.",
            )
        return DeleteChatSessionResponse(
            session_id=clean_id,
            message="Chat session and conversation history successfully deleted.",
            deleted=True,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while deleting chat session: {str(exc)}",
        )


@app.post(
    "/api/chat/sessions/{session_id}/messages",
    response_model=ChatMessageItem,
    status_code=status.HTTP_201_CREATED,
    summary="Manually append a message to a chat session",
    tags=["Conversational Chat"],
)
async def save_chat_message_endpoint(
    session_id: str,
    request: SaveChatMessageRequest,
    session_token: str = Depends(get_session_token),
) -> ChatMessageItem:
    clean_session_id = session_id.strip()
    msg_id = f"cmsg_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    try:
        cited_list = [cs.model_dump() for cs in request.cited_sources] if request.cited_sources else None
        saved = store_instance.save_chat_message(
            message_id=msg_id,
            session_id=clean_session_id,
            role=request.role,
            content=request.content,
            cited_sources=cited_list,
            retrieval_status=request.retrieval_status,
            generation_provider=request.generation_provider,
            session_token=session_token,
        )
        return ChatMessageItem(
            id=saved["id"],
            session_id=saved["session_id"],
            role=saved["role"],
            content=saved["content"],
            cited_sources=[
                SourceChunk(
                    document_id=cs.get("document_id", ""),
                    document_name=cs.get("document_name", "Document"),
                    page_number=cs.get("page_number", 1),
                    chunk_index=cs.get("chunk_index", 0),
                    similarity_score=cs.get("similarity_score", 0.0),
                    excerpt=cs.get("excerpt") or cs.get("text") or "",
                )
                for cs in saved.get("cited_sources", [])
                if isinstance(cs, dict)
            ],
            retrieval_status=saved.get("retrieval_status"),
            generation_provider=saved.get("generation_provider"),
            created_at=saved["created_at"],
        )
    except KeyError as k_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(k_err).strip("'\""),
        )
    except ValueError as v_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(v_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while saving chat message: {str(exc)}",
        )


@app.post(
    "/api/chat/sessions/{session_id}/ask",
    response_model=None,
    responses={
        200: {"model": ChatAskResponse},
        400: {"model": ErrorResponse, "description": "Invalid question input"},
        404: {"model": ErrorResponse, "description": "Session or Document not found"},
        500: {"model": ErrorResponse, "description": "Internal error during generation"},
    },
    status_code=status.HTTP_200_OK,
    summary="Ask a grounded conversational question within a persistent chat session",
    tags=["Conversational Chat"],
)
async def chat_session_ask_endpoint(
    session_id: str,
    request: ChatAskRequest,
    raw_request: Request,
    stream: Optional[bool] = False,
    session_token: str = Depends(get_session_token),
):
    clean_session_id = session_id.strip()
    session = store_instance.get_chat_session(clean_session_id, session_token=session_token)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session with ID '{clean_session_id}' not found.",
        )

    # Determine document ID: from request or from session
    target_doc_id = request.document_id or session.get("document_id")
    if not target_doc_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document is associated with this session. Please specify a document_id.",
        )

    # Validate that document exists and belongs to this session
    doc = store_instance.get_document(target_doc_id, session_token=session_token)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{target_doc_id}' not found.",
        )

    # Verify if request.document_id matches session document_id
    if session.get("document_id") and request.document_id and session.get("document_id") != request.document_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document ID '{request.document_id}' does not match session document ID '{session['document_id']}'.",
        )

    # 1. Retrieve prior conversation messages for contextual reference resolution
    try:
        raw_history = store_instance.get_session_messages(clean_session_id, session_token=session_token)
    except Exception:
        raw_history = []

    # 2. Resolve retrieval query for vector search (handling follow-ups like 'explain the second point')
    retrieval_query = resolve_retrieval_query(request.question, raw_history)

    # 3. Retrieve relevant chunks using vector similarity
    top_k = request.top_k or 4
    embedding_started = time.perf_counter()
    query_embedding = await run_in_threadpool(embedding_service.embed_query, retrieval_query)
    embedding_seconds = time.perf_counter() - embedding_started
    retrieval_started = time.perf_counter()
    candidate_chunks = await run_in_threadpool(
        store_instance.search_chunks,
        query_embedding=query_embedding,
        top_k=top_k,
        document_id=target_doc_id,
        session_token=session_token,
    )
    retrieval_seconds = time.perf_counter() - retrieval_started

    from app.rag.generator import is_overview_or_meta_query
    if target_doc_id and (is_overview_or_meta_query(retrieval_query) or not candidate_chunks):
        doc_chunks = store_instance.get_document_chunks(target_doc_id, session_token=session_token)
        if doc_chunks:
            seen_ids = {r.get("id") for r in candidate_chunks}
            for idx, chunk in enumerate(doc_chunks[:6]):
                if chunk["id"] not in seen_ids:
                    chunk_copy = dict(chunk)
                    chunk_copy["similarity_score"] = max(0.72, 0.88 - (idx * 0.03))
                    candidate_chunks.append(chunk_copy)

    # 4. Format candidate sources with explicit excerpt attribute
    formatted_candidates = [
        SourceChunk(
            document_id=r["document_id"],
            document_name=r["document_name"],
            page_number=r["page_number"],
            chunk_index=r["chunk_index"],
            similarity_score=r.get("similarity_score", 0.0),
            excerpt=r.get("excerpt") or r.get("text") or "",
        )
        for r in candidate_chunks
    ]

    # 5. Resolve similarity threshold
    threshold = (
        request.similarity_threshold
        if request.similarity_threshold is not None
        else generator_instance.default_threshold
    )

    # 6. Generate answer using cloud Gemini LLM (with multi-turn history and extractive fallbacks)
    wants_stream = stream or "text/event-stream" in raw_request.headers.get("accept", "")
    if wants_stream:
        async def event_generator():
            full_answer = ""
            final_meta = {}
            candidate_dicts = [s.model_dump() for s in formatted_candidates]
            async for chunk in generator_instance.stream_answer(
                question=request.question,
                sources=candidate_dicts,
                similarity_threshold=threshold,
                conversation_history=raw_history,
            ):
                if "token" in chunk:
                    full_answer += chunk["token"]
                    yield f"data: {json.dumps({'token': chunk['token']})}\n\n"
                elif chunk.get("done"):
                    final_meta = chunk

            user_msg_id = f"cmsg_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            asst_msg_id = f"cmsg_{int(time.time()) + 1}_{uuid.uuid4().hex[:8]}"

            store_instance.save_chat_message(
                message_id=user_msg_id,
                session_id=clean_session_id,
                role="user",
                content=request.question,
                session_token=session_token,
            )

            cited_raw = final_meta.get("cited_sources", [])
            store_instance.save_chat_message(
                message_id=asst_msg_id,
                session_id=clean_session_id,
                role="assistant",
                content=final_meta.get("answer", full_answer),
                cited_sources=cited_raw,
                retrieval_status=final_meta.get("retrieval_status"),
                generation_provider=final_meta.get("generation_provider"),
                session_token=session_token,
            )

            final_event = {
                "done": True,
                "session_id": clean_session_id,
                "question": request.question,
                "answer": final_meta.get("answer", full_answer),
                "user_message_id": user_msg_id,
                "assistant_message_id": asst_msg_id,
                "cited_sources": cited_raw,
                "retrieved_sources": final_meta.get("retrieved_sources", candidate_dicts),
                "sources": cited_raw,
                "retrieval_status": final_meta.get("retrieval_status", "found"),
                "generation_provider": final_meta.get("generation_provider", "unknown"),
                "model_name": final_meta.get("model_name"),
            }
            yield f"data: {json.dumps(final_event)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    generation_started = time.perf_counter()
    res = await generator_instance.generate_answer(
        question=request.question,
        sources=[s.model_dump() for s in formatted_candidates],
        similarity_threshold=threshold,
        conversation_history=raw_history,
    )
    logger.info(
        "Chat answer generated: provider=%s embedding_seconds=%.2f retrieval_seconds=%.2f generation_seconds=%.2f candidates=%d",
        res.get("generation_provider", "unknown"),
        embedding_seconds,
        retrieval_seconds,
        time.perf_counter() - generation_started,
        len(candidate_chunks),
    )

    # 7. Persist both the user question and assistant answer in SQLite
    user_msg_id = f"cmsg_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    asst_msg_id = f"cmsg_{int(time.time()) + 1}_{uuid.uuid4().hex[:8]}"

    # Save user message
    store_instance.save_chat_message(
        message_id=user_msg_id,
        session_id=clean_session_id,
        role="user",
        content=request.question,
        session_token=session_token,
    )

    # Save assistant message with citation metadata
    cited_raw = res.get("cited_sources", [])
    store_instance.save_chat_message(
        message_id=asst_msg_id,
        session_id=clean_session_id,
        role="assistant",
        content=res["answer"],
        cited_sources=cited_raw,
        retrieval_status=res.get("retrieval_status"),
        generation_provider=res.get("generation_provider"),
        session_token=session_token,
    )

    cited_chunks = [
        SourceChunk(
            document_id=cs.get("document_id", ""),
            document_name=cs.get("document_name", "Document"),
            page_number=cs.get("page_number", 1),
            chunk_index=cs.get("chunk_index", 0),
            similarity_score=cs.get("similarity_score", 0.0),
            excerpt=cs.get("excerpt") or cs.get("text") or "",
        )
        for cs in cited_raw
    ]
    retrieved_chunks = [
        SourceChunk(
            document_id=cs.get("document_id", ""),
            document_name=cs.get("document_name", "Document"),
            page_number=cs.get("page_number", 1),
            chunk_index=cs.get("chunk_index", 0),
            similarity_score=cs.get("similarity_score", 0.0),
            excerpt=cs.get("excerpt") or cs.get("text") or "",
        )
        for cs in res.get("retrieved_sources", [s.model_dump() for s in formatted_candidates])
    ]

    return ChatAskResponse(
        session_id=clean_session_id,
        question=request.question,
        answer=res["answer"],
        user_message_id=user_msg_id,
        assistant_message_id=asst_msg_id,
        cited_sources=cited_chunks,
        retrieved_sources=retrieved_chunks,
        sources=cited_chunks,
        retrieval_status=res["retrieval_status"],
        generation_provider=res["generation_provider"],
        model_name=res.get("model_name"),
    )
