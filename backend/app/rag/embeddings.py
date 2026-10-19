"""
StudySphere AI - Semantic Embedding Service
Uses sentence-transformers/all-MiniLM-L6-v2 for lightweight, fast 384-dimensional
dense vector generation. Supports CUDA GPU acceleration (RTX 3050) with safe CPU fallback.
"""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger("studysphere.embeddings")

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class EmbeddingService:
    """
    Manages loading and inference for the sentence embedding model.
    Loads lazily to avoid startup latency until first retrieval/upload.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._device: Optional[str] = None

    def _load_model(self):
        if self._model is not None:
            return

        import torch
        from sentence_transformers import SentenceTransformer

        # Detect GPU / CPU availability
        if torch.cuda.is_available():
            self._device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA device for embeddings: {gpu_name}")
        else:
            self._device = "cpu"
            logger.info("Using CPU for embeddings.")

        logger.info(f"Loading embedding model '{self.model_name}' on {self._device}...")
        self._model = SentenceTransformer(self.model_name, device=self._device)
        logger.info(f"Model '{self.model_name}' loaded successfully.")

    @property
    def device(self) -> str:
        if self._device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generates normalized float32 embedding vectors for a list of text strings.
        """
        if not texts:
            return []

        self._load_model()
        # normalize_embeddings=True ensures cosine similarity equals dot product
        embeddings = self._model.encode(
            texts,
            batch_size=min(len(texts), 64),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [np.asarray(emb, dtype=np.float32) for emb in embeddings]

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generates a normalized float32 embedding vector for a single search query.
        """
        self._load_model()
        emb = self._model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(emb, dtype=np.float32)


# Global singleton
embedding_service = EmbeddingService()
