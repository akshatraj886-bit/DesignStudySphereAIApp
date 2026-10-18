/**
 * StudySphere AI - Backend API Client
 * Provides typed methods for communication with the FastAPI RAG backend.
 * Automatically injects unguessable client session tokens for tenant privacy.
 */

export interface HealthResponse {
  status: string
  service: string
  embedding_model?: string
  device?: string
}

export interface PageContent {
  page_number: number
  text: string
  character_count: number
  word_count: number
}

export interface PDFUploadResponse {
  document_id: string
  filename: string
  page_count: number
  total_characters: number
  total_words: number
  chunk_count: number
  has_extractable_text: boolean
  ocr_suggested: boolean
  message: string
  pages: PageContent[]
}

export interface SourceChunk {
  document_id: string
  document_name: string
  page_number: number
  chunk_index: number
  similarity_score: number
  excerpt: string
}

export interface AskRequest {
  question: string
  document_id?: string
  top_k?: number
  similarity_threshold?: number
}

export interface AskResponse {
  question: string
  answer: string
  cited_sources: SourceChunk[]
  retrieved_sources: SourceChunk[]
  sources: SourceChunk[]
  retrieval_status: "found" | "insufficient_evidence" | "empty_query" | "no_documents"
  generation_provider: string
  model_name?: string
}

export interface DocumentItem {
  id: string
  filename: string
  page_count: number
  total_characters: number
  total_words: number
  chunk_count: number
  created_at: string
}

export interface DocumentListResponse {
  documents: DocumentItem[]
  total_count: number
}

export interface ApiError {
  message: string
  status?: number
  isNetworkError?: boolean
}

export type SummaryMode = "short" | "detailed" | "key_points" | "topic_wise"

export interface TopicSection {
  title: string
  content: string
  key_points?: string[]
  page_numbers: number[]
}

export interface SummarizeResponse {
  document_id: string
  document_name: string
  summary_mode: string
  summary: string
  key_points?: string[]
  topic_sections?: TopicSection[]
  cited_pages: number[]
  generation_provider: string
  model_name?: string
  warning?: string
}

export type QuestionType = "mcq" | "short_answer" | "long_answer" | "true_false"
export type QuestionDifficulty = "easy" | "medium" | "hard"

export interface QuestionItem {
  question_id: string
  question_text: string
  question_type: QuestionType
  difficulty: QuestionDifficulty
  options?: string[]
  source_page_numbers: number[]
}

export interface GenerateQuestionsRequest {
  document_id: string
  question_count?: number
  question_types?: QuestionType[]
  difficulty?: QuestionDifficulty
}

export interface GenerateQuestionsResponse {
  session_id: string
  document_id: string
  document_name: string
  total_questions: number
  difficulty: string
  question_types: string[]
  questions: QuestionItem[]
  generation_provider: string
  model_name?: string
  warning?: string
}

export interface QuestionAnswerSubmission {
  question_id: string
  user_answer: string
}

export interface EvaluateQuizRequest {
  session_id: string
  document_id?: string
  answers: QuestionAnswerSubmission[]
  questions?: QuestionItem[]
}

export interface QuestionEvaluationResult {
  question_id: string
  question_type: string
  user_answer: string
  correct_answer: string
  is_correct: boolean
  score: number
  feedback: string
  explanation: string
  source_page_numbers: number[]
}

export interface EvaluateQuizResponse {
  total_questions: number
  correct_count: number
  total_score: number
  max_score: number
  percentage: number
  evaluations: QuestionEvaluationResult[]
  results?: QuestionEvaluationResult[]
  grading_method: string
  generation_provider?: string
  model_name?: string
}

// Configurable API base URL from environment variable VITE_API_BASE_URL.
// Defaults to the deployed Render backend in production and local server in development.
// Trailing slashes are automatically removed to ensure valid URL concatenation.
const DEFAULT_PROD_API_URL = "https://studysphere-ai-backend-dqi2.onrender.com"
const envBaseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim()
const rawApiBaseUrl =
  envBaseUrl || (import.meta.env.DEV ? "http://127.0.0.1:8000" : DEFAULT_PROD_API_URL)
export const API_BASE_URL = rawApiBaseUrl.replace(/\/+$/, "")



const SESSION_TOKEN_STORAGE_KEY = "studysphere_session_token"

/**
 * Returns the client-side anonymous session token, generating a cryptographically
 * secure 32-character random hex token if one does not already exist.
 */
export function getOrCreateSessionToken(): string {
  if (typeof window === "undefined" || !window.localStorage) {
    return "default_session"
  }
  let token = window.localStorage.getItem(SESSION_TOKEN_STORAGE_KEY)
  if (!token || !/^sst_[a-zA-Z0-9_-]{16,64}$/.test(token)) {
    const randomBytes = new Uint8Array(16)
    window.crypto.getRandomValues(randomBytes)
    const hex = Array.from(randomBytes, (b) => b.toString(16).padStart(2, "0")).join("")
    token = `sst_${hex}`
    window.localStorage.setItem(SESSION_TOKEN_STORAGE_KEY, token)
  }
  return token
}

/**
 * Helper to construct request headers including the client session token.
 */
function getAuthHeaders(extraHeaders: Record<string, string> = {}): Record<string, string> {
  return {
    "X-Session-Token": getOrCreateSessionToken(),
    ...extraHeaders,
  }
}

/**
 * Checks the operational health of the FastAPI backend.
 */
export async function checkBackendHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, {
      method: "GET",
      headers: getAuthHeaders({
        Accept: "application/json",
      }),
    })

    if (!res.ok) {
      let detail = `Health check failed with HTTP ${res.status}`
      try {
        const errorData = await res.json()
        if (errorData?.detail) detail = errorData.detail
      } catch {
        // Fall back to default message
      }
      throw { message: detail, status: res.status } as ApiError
    }

    return (await res.json()) as HealthResponse
  } catch (err: unknown) {
    if (err && typeof err === "object" && "status" in err) {
      throw err as ApiError
    }
    throw {
      message: `Cannot connect to backend server at ${API_BASE_URL}. Please ensure the FastAPI backend is running.`,
      isNetworkError: true,
    } as ApiError
  }
}

/**
 * Uploads a genuine PDF file to the backend for real page-by-page text extraction,
 * semantic chunking, dense vector embedding, and persistent storage.
 *
 * @param file Actual browser File object (max 50 MB, genuine PDF).
 */
export async function uploadPDFDocument(file: File): Promise<PDFUploadResponse> {
  const formData = new FormData()
  formData.append("file", file, file.name)

  try {
    const res = await fetch(`${API_BASE_URL}/api/documents/upload`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: formData,
    })

    if (!res.ok) {
      let detail = `Upload failed with HTTP ${res.status}`
      try {
        const errorJson = await res.json()
        if (errorJson?.detail) {
          detail = errorJson.detail
        }
      } catch {
        // Fall back to default detail string
      }
      throw { message: detail, status: res.status } as ApiError
    }

    return (await res.json()) as PDFUploadResponse
  } catch (err: unknown) {
    if (err && typeof err === "object" && "status" in err) {
      throw err as ApiError
    }
    throw {
      message: `Failed to communicate with backend at ${API_BASE_URL}. Please verify the backend is running.`,
      isNetworkError: true,
    } as ApiError
  }
}

/**
 * Queries the genuine RAG system with a question and optional document restriction.
 */
export async function askQuestion(
  question: string,
  documentId?: string,
  topK: number = 4,
  similarityThreshold?: number
): Promise<AskResponse> {
  try {
    const payload: AskRequest = {
      question: question.trim(),
      document_id: documentId || undefined,
      top_k: topK,
      similarity_threshold: similarityThreshold,
    }

    const res = await fetch(`${API_BASE_URL}/api/ask`, {
      method: "POST",
      headers: getAuthHeaders({
        "Content-Type": "application/json",
        Accept: "application/json",
      }),
      body: JSON.stringify(payload),
    })

    if (!res.ok) {
      let detail = `Q&A query failed with HTTP ${res.status}`
      try {
        const errJson = await res.json()
        if (errJson?.detail) detail = errJson.detail
      } catch {
        // Fall back to default
      }
      throw { message: detail, status: res.status } as ApiError
    }

    return (await res.json()) as AskResponse
  } catch (err: unknown) {
    if (err && typeof err === "object" && "status" in err) {
      throw err as ApiError
    }
    throw {
      message: `Failed to reach StudySphere AI Q&A service at ${API_BASE_URL}.`,
      isNetworkError: true,
    } as ApiError
  }
}

/**
 * Lists all indexed documents currently stored persistently in the backend for this session.
 */
export async function fetchStoredDocuments(): Promise<DocumentListResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/documents`, {
      method: "GET",
      headers: getAuthHeaders({ Accept: "application/json" }),
    })

    if (!res.ok) {
      throw { message: `Failed to fetch documents: HTTP ${res.status}`, status: res.status } as ApiError
    }

    return (await res.json()) as DocumentListResponse
  } catch (err: unknown) {
    if (err && typeof err === "object" && "status" in err) {
      throw err as ApiError
    }
    throw {
      message: `Failed to retrieve documents from ${API_BASE_URL}.`,
      isNetworkError: true,
    } as ApiError
  }
}

/**
 * Permanently deletes an indexed document and its chunks from backend SQLite storage.
 */
export async function deleteDocument(documentId: string): Promise<{ status: string; document_id: string }> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/documents/${encodeURIComponent(documentId)}`, {
      method: "DELETE",
      headers: getAuthHeaders({ Accept: "application/json" }),
    })

    if (!res.ok) {
      let detail = `Failed to delete document: HTTP ${res.status}`
      try {
        const errorJson = await res.json()
        if (errorJson?.detail) detail = errorJson.detail
      } catch {}
      throw { message: detail, status: res.status } as ApiError
    }

    return (await res.json()) as { status: string; document_id: string }
  } catch (err: unknown) {
    if (err && typeof err === "object" && "status" in err) {
      throw err as ApiError
    }
    throw {
      message: `Failed to delete document from ${API_BASE_URL}.`,
      isNetworkError: true,
    } as ApiError
  }
}

/**
 * Generates a grounded, hallucination-free summary for an indexed document using the selected mode.
 */
export async function summarizeDocument(
  documentId: string,
  mode: SummaryMode = "short"
): Promise<SummarizeResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/documents/${encodeURIComponent(documentId)}/summarize`, {
      method: "POST",
      headers: getAuthHeaders({
        "Content-Type": "application/json",
        Accept: "application/json",
      }),
      body: JSON.stringify({ mode, document_id: documentId }),
    })

    if (!res.ok) {
      let detail = `Summarization failed with HTTP ${res.status}`
      try {
        const errJson = await res.json()
        if (errJson?.detail) detail = errJson.detail
      } catch {
        // Fall back to default
      }
      throw { message: detail, status: res.status } as ApiError
    }

    return (await res.json()) as SummarizeResponse
  } catch (err: unknown) {
    if (err && typeof err === "object" && "status" in err) {
      throw err as ApiError
    }
    throw {
      message: `Failed to reach StudySphere AI Summarizer service at ${API_BASE_URL}.`,
      isNetworkError: true,
    } as ApiError
  }
}

/**
 * Generates grounded revision questions from an indexed document.
 */
export async function generateQuestions(
  documentId: string,
  options?: {
    count?: number
    types?: QuestionType[]
    difficulty?: QuestionDifficulty
  }
): Promise<GenerateQuestionsResponse> {
  try {
    const payload: GenerateQuestionsRequest = {
      document_id: documentId,
      question_count: options?.count || 5,
      question_types: options?.types || ["mcq"],
      difficulty: options?.difficulty || "medium",
    }

    const res = await fetch(`${API_BASE_URL}/api/documents/${encodeURIComponent(documentId)}/questions`, {
      method: "POST",
      headers: getAuthHeaders({
        "Content-Type": "application/json",
        Accept: "application/json",
      }),
      body: JSON.stringify(payload),
    })

    if (!res.ok) {
      let detail = `Question generation failed with HTTP ${res.status}`
      try {
        const errJson = await res.json()
        if (errJson?.detail) detail = errJson.detail
      } catch {
        // Fall back to default
      }
      throw { message: detail, status: res.status } as ApiError
    }

    return (await res.json()) as GenerateQuestionsResponse
  } catch (err: unknown) {
    if (err && typeof err === "object" && "status" in err) {
      throw err as ApiError
    }
    throw {
      message: `Failed to reach StudySphere AI Revision service at ${API_BASE_URL}.`,
      isNetworkError: true,
    } as ApiError
  }
}

/**
 * Submits student answers for evaluation against the authoritative server quiz session.
 */
export async function evaluateQuiz(
  sessionId: string,
  answers: QuestionAnswerSubmission[],
  documentId?: string
): Promise<EvaluateQuizResponse> {
  try {
    const payload: EvaluateQuizRequest = {
      session_id: sessionId,
      answers,
      document_id: documentId,
    }

    const res = await fetch(`${API_BASE_URL}/api/questions/evaluate`, {
      method: "POST",
      headers: getAuthHeaders({
        "Content-Type": "application/json",
        Accept: "application/json",
      }),
      body: JSON.stringify(payload),
    })

    if (!res.ok) {
      let detail = `Quiz evaluation failed with HTTP ${res.status}`
      try {
        const errJson = await res.json()
        if (errJson?.detail) detail = errJson.detail
      } catch {
        // Fall back to default
      }
      throw { message: detail, status: res.status } as ApiError
    }

    return (await res.json()) as EvaluateQuizResponse
  } catch (err: unknown) {
    if (err && typeof err === "object" && "status" in err) {
      throw err as ApiError
    }
    throw {
      message: `Failed to evaluate quiz with StudySphere AI backend at ${API_BASE_URL}.`,
      isNetworkError: true,
    } as ApiError
  }
}

export interface ChatSessionItem {
  id: string
  document_id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface ChatSessionListResponse {
  sessions: ChatSessionItem[]
  total_count: number
}

export interface ChatMessageItem {
  id: string
  session_id: string
  role: "user" | "assistant" | "system"
  content: string
  cited_sources: SourceChunk[]
  retrieval_status?: string
  generation_provider?: string
  created_at: string
}

export interface ChatMessageListResponse {
  session_id: string
  messages: ChatMessageItem[]
  total_count: number
}

export interface ChatAskRequest {
  question: string
  document_id?: string
  top_k?: number
  similarity_threshold?: number
}

export interface ChatAskResponse {
  session_id: string
  question: string
  answer: string
  user_message_id: string
  assistant_message_id: string
  cited_sources: SourceChunk[]
  retrieved_sources: SourceChunk[]
  sources: SourceChunk[]
  retrieval_status: "found" | "insufficient_evidence" | "empty_query" | "no_documents"
  generation_provider: string
  model_name?: string
}

/**
 * Creates a persistent conversational session attached to an indexed document.
 */
export async function createChatSession(
  title: string,
  documentId: string
): Promise<ChatSessionItem> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/chat/sessions`, {
      method: "POST",
      headers: getAuthHeaders({
        "Content-Type": "application/json",
        Accept: "application/json",
      }),
      body: JSON.stringify({
        title: title.trim(),
        document_id: documentId,
      }),
    })

    if (!res.ok) {
      let detail = `Failed to create chat session: HTTP ${res.status}`
      try {
        const errJson = await res.json()
        if (errJson?.detail) detail = errJson.detail
      } catch {}
      throw { message: detail, status: res.status } as ApiError
    }

    return (await res.json()) as ChatSessionItem
  } catch (err: unknown) {
    if (err && typeof err === "object" && "status" in err) {
      throw err as ApiError
    }
    throw {
      message: `Failed to connect to backend at ${API_BASE_URL}.`,
      isNetworkError: true,
    } as ApiError
  }
}

/**
 * Lists conversation sessions, optionally filtered by documentId.
 */
export async function listChatSessions(
  documentId?: string
): Promise<ChatSessionListResponse> {
  try {
    const url = documentId
      ? `${API_BASE_URL}/api/chat/sessions?document_id=${encodeURIComponent(documentId)}`
      : `${API_BASE_URL}/api/chat/sessions`

    const res = await fetch(url, {
      method: "GET",
      headers: getAuthHeaders({ Accept: "application/json" }),
    })

    if (!res.ok) {
      let detail = `Failed to list chat sessions: HTTP ${res.status}`
      try {
        const errJson = await res.json()
        if (errJson?.detail) detail = errJson.detail
      } catch {}
      throw { message: detail, status: res.status } as ApiError
    }

    return (await res.json()) as ChatSessionListResponse
  } catch (err: unknown) {
    if (err && typeof err === "object" && "status" in err) {
      throw err as ApiError
    }
    throw {
      message: `Failed to retrieve chat sessions from ${API_BASE_URL}.`,
      isNetworkError: true,
    } as ApiError
  }
}

/**
 * Retrieves chronological conversation messages for a chat session.
 */
export async function getChatMessages(
  sessionId: string
): Promise<ChatMessageListResponse> {
  try {
    const res = await fetch(
      `${API_BASE_URL}/api/chat/sessions/${encodeURIComponent(sessionId)}/messages`,
      {
        method: "GET",
        headers: getAuthHeaders({ Accept: "application/json" }),
      }
    )

    if (!res.ok) {
      let detail = `Failed to get chat messages: HTTP ${res.status}`
      try {
        const errJson = await res.json()
        if (errJson?.detail) detail = errJson.detail
      } catch {}
      throw { message: detail, status: res.status } as ApiError
    }

    return (await res.json()) as ChatMessageListResponse
  } catch (err: unknown) {
    if (err && typeof err === "object" && "status" in err) {
      throw err as ApiError
    }
    throw {
      message: `Failed to retrieve chat history from ${API_BASE_URL}.`,
      isNetworkError: true,
    } as ApiError
  }
}

/**
 * Submits a question within a persistent chat session to receive a grounded answer with citations.
 */
export async function askInChatSession(
  sessionId: string,
  question: string,
  documentId?: string,
  topK: number = 4,
  similarityThreshold?: number
): Promise<ChatAskResponse> {
  try {
    const payload: ChatAskRequest = {
      question: question.trim(),
      document_id: documentId || undefined,
      top_k: topK,
      similarity_threshold: similarityThreshold,
    }

    const res = await fetch(
      `${API_BASE_URL}/api/chat/sessions/${encodeURIComponent(sessionId)}/ask`,
      {
        method: "POST",
        headers: getAuthHeaders({
          "Content-Type": "application/json",
          Accept: "application/json",
        }),
        body: JSON.stringify(payload),
      }
    )

    if (!res.ok) {
      let detail = `Conversational Q&A failed with HTTP ${res.status}`
      try {
        const errJson = await res.json()
        if (errJson?.detail) detail = errJson.detail
      } catch {}
      throw { message: detail, status: res.status } as ApiError
    }

    return (await res.json()) as ChatAskResponse
  } catch (err: unknown) {
    if (err && typeof err === "object" && "status" in err) {
      throw err as ApiError
    }
    throw {
      message: `Failed to reach StudySphere AI conversational service at ${API_BASE_URL}.`,
      isNetworkError: true,
    } as ApiError
  }
}

/**
 * Permanently deletes a chat session and its conversation history.
 */
export async function deleteChatSession(
  sessionId: string
): Promise<{ session_id: string; message: string; deleted: boolean }> {
  try {
    const res = await fetch(
      `${API_BASE_URL}/api/chat/sessions/${encodeURIComponent(sessionId)}`,
      {
        method: "DELETE",
        headers: getAuthHeaders({ Accept: "application/json" }),
      }
    )

    if (!res.ok) {
      let detail = `Failed to delete chat session: HTTP ${res.status}`
      try {
        const errorJson = await res.json()
        if (errorJson?.detail) detail = errorJson.detail
      } catch {}
      throw { message: detail, status: res.status } as ApiError
    }

    return (await res.json()) as { session_id: string; message: string; deleted: boolean }
  } catch (err: unknown) {
    if (err && typeof err === "object" && "status" in err) {
      throw err as ApiError
    }
    throw {
      message: `Failed to delete chat session at ${API_BASE_URL}.`,
      isNetworkError: true,
    } as ApiError
  }
}
