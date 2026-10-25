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
    Splits a single block of text into boundary-aware overlapping windows.
    Prioritizes natural boundaries in order of preference:
      1. Paragraph breaks (\\n\\n)
      2. Sentence terminators (. , .\\n, ? , ?\\n, ! , !\\n)
      3. Line breaks / list items (\\n)
      4. Clause delimiters (; , : )
      5. Word boundaries (space)
    Ensures sentences are not cut mid-phrase and overlap starts at clean word/sentence boundaries.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    if len(cleaned) <= target_chunk_size:
        return [cleaned]

    chunks: List[str] = []
    start = 0
    total_len = len(cleaned)

    primary_delimiters = [
        "\n\n",
        ".\n", ". ", ".\r\n",
        "?\n", "? ",
        "!\n", "! ",
    ]
    secondary_delimiters = [
        "\n",
        "; ",
        ": ",
    ]

    while start < total_len:
        # If remaining text fits within target chunk size, take it all
        if total_len - start <= target_chunk_size:
            chunk_slice = cleaned[start:total_len].strip()
            if chunk_slice:
                chunks.append(chunk_slice)
            break

        ideal_end = start + target_chunk_size
        max_end = min(total_len, ideal_end + min(80, target_chunk_size // 6))
        min_end = start + max(100, target_chunk_size // 2)

        best_break = -1

        # Search for primary sentence/paragraph boundaries in [min_end, max_end]
        search_zone = cleaned[min_end:max_end]
        best_dist = float("inf")

        for delim in primary_delimiters:
            pos = 0
            while True:
                idx = search_zone.find(delim, pos)
                if idx == -1:
                    break
                candidate = min_end + idx + len(delim)
                dist = abs(candidate - ideal_end)
                if dist < best_dist:
                    best_dist = dist
                    best_break = candidate
                pos = idx + 1

        # If no primary delimiter found, search for secondary delimiters (lines, clauses)
        if best_break == -1:
            best_dist = float("inf")
            for delim in secondary_delimiters:
                pos = 0
                while True:
                    idx = search_zone.find(delim, pos)
                    if idx == -1:
                        break
                    candidate = min_end + idx + len(delim)
                    dist = abs(candidate - ideal_end)
                    if dist < best_dist:
                        best_dist = dist
                        best_break = candidate
                    pos = idx + 1

        # If no sentence/clause boundary found, fall back to word boundary (space)
        if best_break == -1:
            space_pos = cleaned.rfind(" ", min_end, ideal_end)
            if space_pos != -1:
                best_break = space_pos + 1
            else:
                best_break = ideal_end

        end = best_break
        chunk_slice = cleaned[start:end].strip()
        if chunk_slice:
            chunks.append(chunk_slice)

        if end >= total_len:
            break

        # Calculate boundary-aware overlap
        desired_overlap_start = max(start + 1, end - overlap_size)
        overlap_search_zone = cleaned[desired_overlap_start:end]

        overlap_start = -1
        # Look for a sentence or line start in the overlap window
        for delim in ["\n\n", ".\n", ". ", "\n"]:
            idx = overlap_search_zone.find(delim)
            if idx != -1:
                cand = desired_overlap_start + idx + len(delim)
                if cand < end - 20:  # Ensure remaining overlap has substantive context
                    overlap_start = cand
                    break

        if overlap_start == -1:
            # Fall back to word boundary so we never cut words mid-letter
            space_idx = cleaned.find(" ", desired_overlap_start, end)
            if space_idx != -1:
                overlap_start = space_idx + 1
            else:
                overlap_start = desired_overlap_start

        # Safe forward progress guarantee
        if overlap_start <= start or overlap_start >= end:
            overlap_start = start + max(1, (end - start) // 2)

        start = overlap_start

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
