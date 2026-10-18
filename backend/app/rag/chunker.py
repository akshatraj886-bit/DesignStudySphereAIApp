"""
StudySphere AI - Document Chunking Service
Splits page-by-page extracted PDF text into overlapping chunks while preserving
strict metadata (document_id, document_name, page_number, chunk_index).
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a discrete semantic chunk of a document with citation metadata."""
    chunk_id: str = Field(..., description="Unique chunk identifier: {doc_id}_p{page}_c{index}")
    document_id: str = Field(..., description="Parent document identifier")
    document_name: str = Field(..., description="Original filename of the document")
    page_number: int = Field(..., description="1-indexed source page number")
    chunk_index: int = Field(..., description="0-indexed chunk position within the document")
    text: str = Field(..., description="Extracted textual content of the chunk")
    character_count: int = Field(..., description="Length of chunk text in characters")
    word_count: int = Field(..., description="Number of words in the chunk")


def chunk_text(
    text: str,
    target_chunk_size: int = 500,
    overlap_size: int = 100,
) -> List[str]:
    """
    Splits a single block of text into overlapping windows.
    Attempts natural boundary splits (paragraphs, sentences, or word boundaries)
    rather than splitting words midway.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    if len(cleaned) <= target_chunk_size:
        return [cleaned]

    chunks: List[str] = []
    start = 0
    total_len = len(cleaned)

    while start < total_len:
        end = min(start + target_chunk_size, total_len)

        # If not at the end of text, attempt to break at a sentence or word boundary
        if end < total_len:
            # Look for a sentence-ending punctuation or newline in the last 60 chars of the window
            lookback_zone = cleaned[max(start, end - 60):end]
            best_break = -1
            for punct in [".\n", ". ", "?\n", "? ", "!\n", "! ", "\n\n", "\n"]:
                pos = lookback_zone.rfind(punct)
                if pos != -1:
                    candidate = max(start, end - 60) + pos + len(punct)
                    if candidate > start + 50:  # Avoid overly tiny chunks
                        best_break = candidate
                        break

            if best_break != -1:
                end = best_break
            else:
                # Fall back to word boundary (space)
                space_pos = cleaned.rfind(" ", start + 50, end)
                if space_pos != -1:
                    end = space_pos + 1

        chunk_slice = cleaned[start:end].strip()
        if chunk_slice:
            chunks.append(chunk_slice)

        if end >= total_len:
            break

        # Move forward, respecting the specified overlap
        next_start = end - overlap_size
        if next_start <= start:
            next_start = start + max(1, target_chunk_size // 2)
        start = next_start

    return chunks


def process_document_pages(
    document_id: str,
    document_name: str,
    pages: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[DocumentChunk]:
    """
    Processes page-by-page text extractions from PyMuPDF and produces structured
    DocumentChunk objects tagged with page numbers and document identification.
    
    Args:
        document_id: Unique string ID for the document.
        document_name: Human-readable filename.
        pages: List of dictionaries or PageContent models with 'page_number' and 'text'.
        chunk_size: Target character count per chunk.
        chunk_overlap: Character overlap between consecutive chunks.
    
    Returns:
        List of DocumentChunk instances.
    """
    all_chunks: List[DocumentChunk] = []
    global_chunk_idx = 0

    for page in pages:
        # Support both Pydantic models and dictionaries
        if hasattr(page, "page_number"):
            page_num = page.page_number
            raw_text = page.text
        else:
            page_num = page.get("page_number", 1)
            raw_text = page.get("text", "")

        if not raw_text or not raw_text.strip():
            continue

        page_text_chunks = chunk_text(
            text=raw_text,
            target_chunk_size=chunk_size,
            overlap_size=chunk_overlap,
        )

        for local_idx, chunk_content in enumerate(page_text_chunks):
            chunk_id = f"{document_id}_p{page_num}_c{local_idx}"
            words = len(chunk_content.split())
            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page_num,
                    chunk_index=global_chunk_idx,
                    text=chunk_content,
                    character_count=len(chunk_content),
                    word_count=words,
                )
            )
            global_chunk_idx += 1

    return all_chunks
