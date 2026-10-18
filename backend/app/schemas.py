from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional


class HealthResponse(BaseModel):
    """Schema for the health check endpoint response."""
    status: str = Field(..., example="ok")
    service: str = Field(..., example="StudySphere AI Backend")
    embedding_model: Optional[str] = Field(None, example="all-MiniLM-L6-v2")
    device: Optional[str] = Field(None, example="cuda")
    database_engine: Optional[str] = Field(None, example="sqlite")


class PageContent(BaseModel):
    """Schema representing extracted text and metadata for a single PDF page."""
    page_number: int = Field(..., description="1-indexed page number", example=1)
    text: str = Field(..., description="Extracted textual content from the page")
    character_count: int = Field(..., description="Number of characters in the page text", example=450)
    word_count: int = Field(..., description="Number of words in the page text", example=75)


class PDFUploadResponse(BaseModel):
    """Schema for the PDF upload, chunking, embedding, and persistent storage response."""
    document_id: str = Field(..., description="Unique persistent identifier for the document", example="doc_1729158000_a1b2c3d4")
    filename: str = Field(..., description="Original name of the uploaded PDF file", example="lecture_notes.pdf")
    page_count: int = Field(..., description="Total number of pages in the PDF", example=5)
    total_characters: int = Field(..., description="Total characters extracted across all pages", example=2450)
    total_words: int = Field(..., description="Total words extracted across all pages", example=390)
    chunk_count: int = Field(default=0, description="Number of semantic chunks indexed into the vector store", example=8)
    has_extractable_text: bool = Field(..., description="Whether any readable text was extracted", example=True)
    ocr_suggested: bool = Field(..., description="True if document appears to be a scanned image needing OCR", example=False)
    message: str = Field(..., description="Status summary message", example="Successfully extracted and indexed text from 5 page(s).")
    pages: List[PageContent] = Field(..., description="List of page-by-page extracted content")


class SourceChunk(BaseModel):
    """Schema representing a retrieved source passage with strict grounding metadata."""
    document_id: str = Field(..., description="Parent document identifier")
    document_name: str = Field(..., description="Original name of the source PDF")
    page_number: int = Field(..., description="1-indexed source page number")
    chunk_index: int = Field(..., description="Position of chunk within the document")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    excerpt: str = Field(..., description="Extracted passage text used as evidence")


class AskRequest(BaseModel):
    """Schema for document question-answering query."""
    question: str = Field(..., min_length=2, max_length=1000, description="The student's question", example="What is 3NF normalization?")
    document_id: Optional[str] = Field(None, description="Optional document ID to restrict retrieval to a specific document")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of relevant source chunks to retrieve")
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional minimum semantic similarity score threshold")


class AskResponse(BaseModel):
    """Schema for question-answering response with grounded citations."""
    question: str = Field(..., description="The original student question")
    answer: str = Field(..., description="Grounded AI or extractive answer")
    cited_sources: List[SourceChunk] = Field(default_factory=list, description="Passages actually used as evidence to compose the answer")
    retrieved_sources: List[SourceChunk] = Field(default_factory=list, description="All candidate passages retrieved by the vector similarity search")
    sources: List[SourceChunk] = Field(default_factory=list, description="Backwards-compatible alias for cited_sources")
    retrieval_status: str = Field(..., description="'found', 'insufficient_evidence', 'empty_query', or 'no_documents'")
    generation_provider: str = Field(..., description="The generation engine used ('google_gemini', 'local_ollama', 'extractive_grounded_synthesizer', 'mock_test_provider')")
    model_name: Optional[str] = Field(None, description="Specific model identifier used for answer synthesis")


class DocumentItem(BaseModel):
    """Schema representing an indexed document in the backend store."""
    id: str = Field(..., description="Persistent document identifier")
    filename: str = Field(..., description="Original filename")
    page_count: int = Field(..., description="Number of pages")
    total_characters: int = Field(..., description="Total character count")
    total_words: int = Field(..., description="Total word count")
    chunk_count: int = Field(..., description="Total indexed vector chunks")
    created_at: str = Field(..., description="ISO 8601 upload timestamp")


class DocumentListResponse(BaseModel):
    """Schema for listing indexed documents."""
    documents: List[DocumentItem]
    total_count: int


class ErrorResponse(BaseModel):
    """Standardized schema for API error responses."""
    detail: str = Field(..., description="Human-readable explanation of the error")
    error_code: Optional[str] = Field(None, description="Categorical error code")


class SummaryMode(str, Enum):
    SHORT = "short"
    DETAILED = "detailed"
    KEY_POINTS = "key_points"
    TOPIC_WISE = "topic_wise"


class SummarizeRequest(BaseModel):
    """Schema for document summarization request."""
    mode: SummaryMode = Field(
        default=SummaryMode.SHORT,
        description="Summary format mode: 'short', 'detailed', 'key_points', or 'topic_wise'",
        example="detailed",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional document ID if not provided in URL path",
    )


class TopicSection(BaseModel):
    """Schema representing a distinct topic section in topic_wise summary mode."""
    title: str = Field(..., description="Topic or section heading", example="Module 1: Process Concepts")
    content: str = Field(..., description="Summary narrative for this topic")
    key_points: Optional[List[str]] = Field(default=None, description="Key bullet takeaways for this topic")
    page_numbers: List[int] = Field(default_factory=list, description="Source pages associated with this topic", example=[1, 2])


class SummarizeResponse(BaseModel):
    """Schema for structured document summarization response."""
    document_id: str = Field(..., description="Persistent document identifier")
    document_name: str = Field(..., description="Original filename of the document")
    summary_mode: str = Field(..., description="Active summary mode: 'short', 'detailed', 'key_points', 'topic_wise'")
    summary: str = Field(..., description="Formatted markdown summary text")
    key_points: Optional[List[str]] = Field(default=None, description="Extracted bullet takeaways when applicable")
    topic_sections: Optional[List[TopicSection]] = Field(default=None, description="Topic-wise or page-wise sections when applicable")
    cited_pages: List[int] = Field(default_factory=list, description="Source pages used to synthesize the summary")
    cited_page_numbers: Optional[List[int]] = Field(default=None, description="Alias for cited_pages for client compatibility")
    generation_provider: str = Field(..., description="The summarization provider engine used")
    model_name: Optional[str] = Field(None, description="Specific model name if generative model was utilized")
    warning: Optional[str] = Field(None, description="Any limitation, note, or OCR notice")

    def __init__(self, **data):
        super().__init__(**data)
        if self.cited_page_numbers is None and self.cited_pages:
            self.cited_page_numbers = list(self.cited_pages)


class QuestionType(str, Enum):
    MCQ = "mcq"
    SHORT_ANSWER = "short_answer"
    LONG_ANSWER = "long_answer"
    TRUE_FALSE = "true_false"


class QuestionDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PublicQuestionItem(BaseModel):
    """Schema representing an individual grounded revision question without answer keys or spoiler explanations."""
    question_id: str = Field(..., description="Unique question identifier", example="q_1")
    question_text: str = Field(..., description="The student-facing question prompt")
    question_type: QuestionType = Field(..., description="Format: 'mcq', 'short_answer', 'long_answer', or 'true_false'")
    difficulty: QuestionDifficulty = Field(..., description="Difficulty level: 'easy', 'medium', or 'hard'")
    options: Optional[List[str]] = Field(default=None, description="List of options for MCQ (4 choices) or True/False (['True', 'False'])")
    source_page_numbers: List[int] = Field(default_factory=list, description="Source pages verifying this question", example=[1])


class QuestionItem(PublicQuestionItem):
    """Authoritative server-side question schema with private answer key and explanation."""
    correct_answer: str = Field(..., description="The correct option or ideal marking answer")
    explanation: str = Field(..., description="Source-grounded explanation and evidence from the notes")


class GenerateQuestionsRequest(BaseModel):
    """Schema for requesting revision question generation."""
    document_id: Optional[str] = Field(default=None, description="Optional document ID if not in URL path")
    question_count: int = Field(default=5, ge=1, le=15, description="Number of questions to generate (5, 10, or 15)")
    question_types: List[QuestionType] = Field(
        default_factory=lambda: [QuestionType.MCQ],
        description="List of question formats to generate",
        example=["mcq", "true_false"]
    )
    difficulty: QuestionDifficulty = Field(
        default=QuestionDifficulty.MEDIUM,
        description="Target difficulty level ('easy', 'medium', 'hard')",
        example="medium"
    )


class GenerateQuestionsResponse(BaseModel):
    """Schema for the public question generation response."""
    session_id: str = Field(..., description="Unique authoritative quiz session identifier")
    document_id: str = Field(..., description="Persistent document identifier")
    document_name: str = Field(..., description="Original filename of the document")
    total_questions: int = Field(..., description="Number of questions returned")
    difficulty: str = Field(..., description="Applied difficulty level")
    question_types: List[str] = Field(..., description="Question formats included in this set")
    questions: List[PublicQuestionItem] = Field(..., description="List of public question objects (answer keys redacted)")
    generation_provider: str = Field(..., description="Provider engine used (e.g. 'offline_grounded_question_generator')")
    model_name: Optional[str] = Field(None, description="Specific model name if generative model was utilized")
    warning: Optional[str] = Field(None, description="Any limitation or offline disclaimer")


class QuestionAnswerSubmission(BaseModel):
    """User answer submission for a single question."""
    question_id: str = Field(..., description="Target question identifier")
    user_answer: str = Field(..., description="The student's submitted response")


class EvaluateQuizRequest(BaseModel):
    """Schema for evaluating a completed revision quiz against an authoritative session."""
    session_id: str = Field(..., description="Authoritative quiz session identifier generated by backend")
    document_id: Optional[str] = Field(default=None, description="Optional document identifier")
    answers: List[QuestionAnswerSubmission] = Field(..., description="Student submitted answers")
    questions: Optional[List[QuestionItem]] = Field(
        default=None,
        description="Deprecated client question payload; ignored by backend in favor of authoritative server session"
    )


class QuestionEvaluationResult(BaseModel):
    """Feedback and score for a single submitted question."""
    question_id: str
    question_type: str
    user_answer: str
    correct_answer: str
    is_correct: bool
    score: float = Field(..., description="Points earned (0.0 to 1.0)")
    feedback: str = Field(..., description="Feedback explanation detailing correct answer or missed points")
    explanation: str = Field(..., description="Source-grounded explanation")
    source_page_numbers: List[int] = Field(default_factory=list)


class EvaluateQuizResponse(BaseModel):
    """Overall evaluation response for the submitted revision quiz."""
    total_questions: int
    correct_count: int
    total_score: float
    max_score: float
    percentage: float
    evaluations: List[QuestionEvaluationResult]
    grading_method: str = Field(
        default="deterministic_and_keyword_overlap",
        description="Grading methodology used"
    )


# =============================================================================
# CONVERSATIONAL CHAT SESSION SCHEMAS
# =============================================================================

class CreateChatSessionRequest(BaseModel):
    """Schema for creating a new conversation session attached to a document."""
    document_id: str = Field(..., description="Target document identifier for conversation grounding", example="doc_1729158000_a1b2c3d4")
    title: Optional[str] = Field(None, max_length=150, description="Optional conversational session title", example="Chapter 3 Normalization Discussion")


class ChatSessionItem(BaseModel):
    """Schema representing a persistent conversation session."""
    id: str = Field(..., description="Unique chat session identifier", example="csess_1729158000_a1b2c3d4")
    document_id: str = Field(..., description="Associated document identifier")
    title: str = Field(..., description="Human-readable title of the conversation")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    updated_at: str = Field(..., description="ISO 8601 last activity timestamp")
    message_count: int = Field(default=0, description="Total number of turns in this session")


class ChatSessionListResponse(BaseModel):
    """Schema for listing chat sessions."""
    sessions: List[ChatSessionItem]
    total_count: int


class ChatMessageItem(BaseModel):
    """Schema representing an individual conversational turn in a session."""
    id: str = Field(..., description="Unique message turn identifier", example="cmsg_1729158000_a1b2c3d4")
    session_id: str = Field(..., description="Parent chat session identifier")
    role: str = Field(..., description="Message author role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Text content of the message")
    cited_sources: List[SourceChunk] = Field(default_factory=list, description="Grounding passages cited for this turn")
    retrieval_status: Optional[str] = Field(None, description="'found', 'insufficient_evidence', etc.")
    generation_provider: Optional[str] = Field(None, description="Engine provider used")
    created_at: str = Field(..., description="ISO 8601 message timestamp")


class ChatMessageListResponse(BaseModel):
    """Schema for returning conversation history for a session."""
    session_id: str
    messages: List[ChatMessageItem]
    total_count: int


class SaveChatMessageRequest(BaseModel):
    """Schema for manually appending a message to a session."""
    role: str = Field(..., description="Message author role ('user' or 'assistant')")
    content: str = Field(..., min_length=1, description="Text body of the message")
    cited_sources: Optional[List[SourceChunk]] = Field(default=None, description="Optional citations")
    retrieval_status: Optional[str] = Field(default=None, description="Retrieval status")
    generation_provider: Optional[str] = Field(default=None, description="Provider identifier")


class DeleteChatSessionResponse(BaseModel):
    """Schema for chat session deletion response."""
    session_id: str
    message: str
    deleted: bool


class ChatAskRequest(BaseModel):
    """Schema for asking a grounded conversational question within a persistent chat session."""
    question: str = Field(..., min_length=1, max_length=1000, description="The student's question", example="Explain the second point in more detail.")
    document_id: Optional[str] = Field(default=None, description="Optional document ID if explicitly specified; otherwise derived from the session")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of relevant source chunks to retrieve")
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional minimum semantic similarity score threshold")


class ChatAskResponse(BaseModel):
    """Schema for response from conversational RAG endpoint with grounding citations and persisted message IDs."""
    session_id: str = Field(..., description="Active chat session identifier")
    question: str = Field(..., description="The student question")
    answer: str = Field(..., description="The grounded educational response")
    user_message_id: str = Field(..., description="Identifier of the persisted user message")
    assistant_message_id: str = Field(..., description="Identifier of the persisted assistant response")
    cited_sources: List[SourceChunk] = Field(default_factory=list, description="Passages cited as evidence")
    retrieved_sources: List[SourceChunk] = Field(default_factory=list, description="Candidate passages retrieved")
    sources: List[SourceChunk] = Field(default_factory=list, description="Backwards-compatible alias for cited_sources")
    retrieval_status: str = Field(..., description="'found', 'insufficient_evidence', 'empty_query', or 'no_documents'")
    generation_provider: str = Field(..., description="Engine provider used ('google_gemini', 'extractive_grounded_synthesizer', etc.)")
    model_name: Optional[str] = Field(None, description="Specific model identifier")




