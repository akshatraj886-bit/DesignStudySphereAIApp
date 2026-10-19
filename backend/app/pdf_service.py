import io
import logging
import time
import uuid
import pymupdf
from fastapi import HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool
from typing import List
from app.schemas import PageContent, PDFUploadResponse
from app.rag.chunker import process_document_pages
from app.rag.embeddings import embedding_service
import os
from app.rag.store import store_instance

logger = logging.getLogger("studysphere.pdf_service")

# Configurable max upload size (default: 25 MB to safely prevent OOM on free-tier containers)
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
MAX_FILE_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
PDF_MAGIC_HEADER = b"%PDF-"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".pptx",
    ".ppt",
    ".docx",
    ".doc",
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
}

# Lazy-loaded OCR engine to save memory on startup
_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
        except Exception:
            _ocr_engine = False
    return _ocr_engine if _ocr_engine is not False else None


def extract_document_pages(content: bytes, ext: str) -> List[PageContent]:
    pages_data: List[PageContent] = []

    # 4. Extract Text Based on Document Type
    if ext == ".pdf":
        # Validate PDF magic bytes (%PDF-)
        if not content[:1024].startswith(PDF_MAGIC_HEADER) and PDF_MAGIC_HEADER not in content[:1024]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "File content is not a recognized PDF format. "
                    "The binary signature (%PDF-) was not found in the file header."
                ),
            )

        doc = None
        try:
            doc = pymupdf.open(stream=content, filetype="pdf")
            if doc.is_encrypted:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The uploaded PDF is password-protected or encrypted. Please remove the password before uploading.",
                )

            page_count = doc.page_count
            if page_count == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The uploaded PDF contains zero pages.",
                )

            for page_idx in range(page_count):
                page = doc.load_page(page_idx)
                text = page.get_text("text").strip()

                # If page text is very sparse or empty and contains raster images, attempt OCR on page pixmap
                if len(text) < 15 and bool(page.get_images()):
                    ocr = get_ocr_engine()
                    if ocr:
                        try:
                            pix = page.get_pixmap(dpi=150)
                            ocr_res, _ = ocr(pix.tobytes("png"))
                            if ocr_res:
                                ocr_lines = [line[1] for line in ocr_res if line and len(line) > 1]
                                if ocr_lines:
                                    text = "\n".join(ocr_lines).strip()
                        except Exception:
                            pass

                char_count = len(text)
                word_count = len(text.split()) if text else 0
                pages_data.append(
                    PageContent(
                        page_number=page_idx + 1,
                        text=text,
                        character_count=char_count,
                        word_count=word_count,
                    )
                )
        except pymupdf.FileDataError as pdf_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"The PDF file is corrupted or could not be parsed: {str(pdf_err)}",
            )
        finally:
            if doc is not None:
                doc.close()

    elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
        # Image / Photo of Notes -> Run RapidOCR
        ocr = get_ocr_engine()
        extracted_text = ""
        if ocr:
            try:
                ocr_res, _ = ocr(content)
                if ocr_res:
                    lines = [line[1] for line in ocr_res if line and len(line) > 1]
                    extracted_text = "\n".join(lines).strip()
            except Exception as ocr_err:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not extract text from image: {str(ocr_err)}",
                )
        
        char_count = len(extracted_text)
        word_count = len(extracted_text.split()) if extracted_text else 0
        pages_data.append(
            PageContent(
                page_number=1,
                text=extracted_text,
                character_count=char_count,
                word_count=word_count,
            )
        )

    elif ext in [".pptx", ".ppt"]:
        # PowerPoint presentation -> Extract slide-by-slide including tables and grouped shapes
        try:
            import pptx
            prs = pptx.Presentation(io.BytesIO(content))

            def _extract_shape_text(shape) -> List[str]:
                texts = []
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        t = paragraph.text.strip()
                        if t:
                            texts.append(t)
                elif shape.has_table:
                    for row in shape.table.rows:
                        row_txt = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                        if row_txt:
                            texts.append(row_txt)
                elif hasattr(shape, "shapes"):
                    for child in shape.shapes:
                        texts.extend(_extract_shape_text(child))
                return texts

            for slide_idx, slide in enumerate(prs.slides):
                slide_lines = []
                for shape in slide.shapes:
                    slide_lines.extend(_extract_shape_text(shape))
                slide_text = "\n".join(slide_lines).strip()
                char_count = len(slide_text)
                word_count = len(slide_text.split()) if slide_text else 0
                pages_data.append(
                    PageContent(
                        page_number=slide_idx + 1,
                        text=slide_text,
                        character_count=char_count,
                        word_count=word_count,
                    )
                )
            if not pages_data:
                pages_data.append(PageContent(page_number=1, text="", character_count=0, word_count=0))
        except Exception as ppt_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not parse PowerPoint presentation: {str(ppt_err)}",
            )

    elif ext in [".docx", ".doc"]:
        # Word document -> Extract paragraphs and tables
        try:
            import docx
            doc_file = docx.Document(io.BytesIO(content))
            paragraphs = []
            for p in doc_file.paragraphs:
                txt = p.text.strip()
                if txt:
                    paragraphs.append(txt)
            for table in doc_file.tables:
                for row in table.rows:
                    row_txt = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                    if row_txt:
                        paragraphs.append(row_txt)

            full_text = "\n\n".join(paragraphs).strip()
            # Split into ~1500 character logical pages
            chunk_page_size = 1500
            if full_text:
                slices = [full_text[i:i + chunk_page_size] for i in range(0, len(full_text), chunk_page_size)]
                for idx, slice_text in enumerate(slices):
                    pages_data.append(
                        PageContent(
                            page_number=idx + 1,
                            text=slice_text,
                            character_count=len(slice_text),
                            word_count=len(slice_text.split()),
                        )
                    )
            else:
                pages_data.append(PageContent(page_number=1, text="", character_count=0, word_count=0))
        except Exception as docx_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not parse Word document: {str(docx_err)}",
            )

    elif ext in [".txt", ".md", ".markdown", ".csv", ".json"]:
        # Plain Text or Markdown file
        try:
            try:
                full_text = content.decode("utf-8")
            except UnicodeDecodeError:
                full_text = content.decode("latin1", errors="replace")

            full_text = full_text.strip()
            chunk_page_size = 1500
            if full_text:
                slices = [full_text[i:i + chunk_page_size] for i in range(0, len(full_text), chunk_page_size)]
                for idx, slice_text in enumerate(slices):
                    pages_data.append(
                        PageContent(
                            page_number=idx + 1,
                            text=slice_text,
                            character_count=len(slice_text),
                            word_count=len(slice_text.split()),
                        )
                    )
            else:
                pages_data.append(PageContent(page_number=1, text="", character_count=0, word_count=0))
        except Exception as txt_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not read text file: {str(txt_err)}",
            )

    return pages_data

async def process_pdf_upload(
    file: UploadFile,
    session_token: str = "default_session",
) -> PDFUploadResponse:
    """
    Universal Document Ingestion Service:
    Accepts PDF, PowerPoint (PPTX), Word (DOCX), Text/Markdown (TXT/MD),
    and Photos/Images (PNG, JPG, WEBP) of notes.
    Extracts text, calculates word/page counts, chunks, generates embeddings,
    and indexes everything persistently in SQLite.
    """
    upload_started = time.perf_counter()
    filename = file.filename or "uploaded_document.pdf"
    lower_name = filename.lower()
    ext = "." + lower_name.split(".")[-1] if "." in lower_name else ""

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file format '{ext}'. StudySphere AI supports: "
                "PDF (.pdf), PowerPoint (.pptx), Word (.docx), Text (.txt, .md), "
                "and Photos/Images (.png, .jpg, .jpeg, .webp)."
            ),
        )

    # 1. Read file contents into memory safely
    try:
        content = await file.read()
    except Exception as read_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to read uploaded file: {str(read_err)}",
        )

    file_size = len(content)

    # 2. Check for empty files
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty (0 bytes). Please upload a valid document or note photo.",
        )

    # 3. Enforce maximum size limit
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size exceeds the {MAX_UPLOAD_SIZE_MB} MB limit. "
                f"Uploaded size: {file_size / (1024 * 1024):.2f} MB."
            ),
        )

    extraction_started = time.perf_counter()
    pages_data = await run_in_threadpool(extract_document_pages, content, ext)
    extraction_seconds = time.perf_counter() - extraction_started

    # 5. Aggregate Statistics & Semantic Indexing
    total_characters = sum(p.character_count for p in pages_data)
    total_words = sum(p.word_count for p in pages_data)
    page_count = len(pages_data)

    has_extractable_text = total_characters > 0
    ocr_suggested = not has_extractable_text

    document_id = f"doc_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    chunk_count = 0
    embedding_seconds = 0.0

    if has_extractable_text:
        # Chunk document into overlapping semantic windows
        chunks = await run_in_threadpool(
            process_document_pages,
            document_id=document_id,
            document_name=filename,
            pages=pages_data,
            chunk_size=500,
            chunk_overlap=100,
        )
        chunk_count = len(chunks)

        # Generate dense semantic embeddings
        chunk_texts = [c.text for c in chunks]
        embedding_started = time.perf_counter()
        embeddings = await run_in_threadpool(embedding_service.embed_texts, chunk_texts)
        embedding_seconds = time.perf_counter() - embedding_started

        # Save document and vector embeddings to persistent SQLite store
        await run_in_threadpool(
            store_instance.save_document,
            document_id=document_id,
            filename=filename,
            page_count=page_count,
            total_characters=total_characters,
            total_words=total_words,
            chunks=chunks,
            embeddings=embeddings,
            session_token=session_token,
        )

        format_label = ext.upper().replace(".", "")
        message = (
            f"Successfully processed {format_label} file ({total_words:,} words across {page_count} section/page(s)) "
            f"and indexed {chunk_count} semantic chunks."
        )
    else:
        # Store metadata record even if no extractable text
        await run_in_threadpool(
            store_instance.save_document,
            document_id=document_id,
            filename=filename,
            page_count=page_count,
            total_characters=total_characters,
            total_words=total_words,
            chunks=[],
            embeddings=[],
            session_token=session_token,
        )
        message = (
            "Document loaded successfully, but no readable text was detected. "
            "Scanned image documents requiring Optical Character Recognition (OCR) are not yet supported."
        )

    logger.info(
        "Document indexing complete: extension=%s pages=%d chunks=%d extraction_seconds=%.2f embedding_seconds=%.2f total_seconds=%.2f",
        ext,
        page_count,
        chunk_count,
        extraction_seconds,
        embedding_seconds,
        time.perf_counter() - upload_started,
    )

    return PDFUploadResponse(
        document_id=document_id,
        filename=filename,
        page_count=page_count,
        total_characters=total_characters,
        total_words=total_words,
        chunk_count=chunk_count,
        has_extractable_text=has_extractable_text,
        ocr_suggested=ocr_suggested,
        message=message,
        pages=pages_data,
    )
