import { useEffect, useRef, useState, type ReactNode } from "react"
import GradientBackground from "./components/GradientBackground"
import GlassNavbar from "./components/GlassNavbar"
import Hero from "./components/Hero"
import Features from "./components/Features"
import Footer from "./components/Footer"
import GlassInput from "./components/GlassInput"
import { glassTheme } from "./components/glassTheme"
import {
  uploadPDFDocument,
  askQuestion,
  fetchStoredDocuments,
  deleteDocument,
  summarizeDocument,
  generateQuestions,
  evaluateQuiz,
  createChatSession,
  listChatSessions,
  getChatMessages,
  askInChatSession,
  type PageContent,
  type SourceChunk,
  type SummaryMode,
  type SummarizeResponse,
  type TopicSection,
  type QuestionItem,
  type QuestionType,
  type QuestionDifficulty,
  type EvaluateQuizResponse,
  type QuestionAnswerSubmission,
} from "./lib/api"


type IconName = "grid" | "file" | "chat" | "spark" | "cards" | "settings" | "help" | "arrow" | "upload" | "search" | "chevron" | "close" | "check" | "clock" | "plus" | "bell" | "menu" | "send" | "book" | "more" | "logout"
function Icon({
  name,
  size = 20,
  className = "",
}: {
  name: IconName
  size?: number
  className?: string
}) {
  const paths: Record<IconName, ReactNode> = {
    grid: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" />
      </>
    ),
    file: (
      <>
        <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9Z" />
        <path d="M14 3v6h6M8 13h8M8 17h5" />
      </>
    ),
    chat: (
      <>
        <path d="M21 11.5a8.5 8.5 0 0 1-9 8.5 10 10 0 0 1-4-.8L3 21l1.6-5A8 8 0 0 1 3 11.5a9 9 0 0 1 18 0Z" />
        <path d="M8 10h8M8 14h5" />
      </>
    ),
    spark: (
      <>
        <path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5Z" />
        <path d="m20 2 .5 1.5L22 4l-1.5.5L20 6l-.5-1.5L18 4l1.5-.5Z" />
      </>
    ),
    cards: (
      <>
        <rect x="6" y="7" width="14" height="14" rx="2" />
        <path d="M16 3H4a2 2 0 0 0-2 2v12M10 12h6M10 16h4" />
      </>
    ),
    settings: (
      <>
        <path d="m9 3-1 3-3 1-2 5 2 5 3 1 1 3h6l1-3 3-1 2-5-2-5-3-1-1-3Z" />
        <circle cx="12" cy="12" r="3" />
      </>
    ),
    help: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M9.5 9a2.5 2.5 0 1 1 4.2 1.8c-1.2.7-1.7 1.2-1.7 2.7M12 17h.01" />
      </>
    ),
    arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
    upload: (
      <>
        <path d="M12 16V3m-5 5 5-5 5 5M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4" />
      </>
    ),
    search: (
      <>
        <circle cx="10.5" cy="10.5" r="7" />
        <path d="m16 16 5 5" />
      </>
    ),
    chevron: <path d="m9 5 7 7-7 7" />,
    close: <path d="m6 6 12 12M6 18 18 6" />,
    check: <path d="m5 12 4 4L19 6" />,
    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    plus: <path d="M12 5v14M5 12h14" />,
    bell: (
      <>
        <path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4" />
      </>
    ),
    menu: <path d="M4 6h16M4 12h16M4 18h16" />,
    send: (
      <>
        <path d="m3 3 18 9-18 9 3-9Z" />
        <path d="M6 12h15" />
      </>
    ),
    book: (
      <>
        <path d="M12 5C8 2 3 4 3 4v15s5-2 9 1c4-3 9-1 9-1V4s-5-2-9 1Zm0 0v15" />
      </>
    ),
    more: (
      <>
        <circle cx="5" cy="12" r="1" />
        <circle cx="12" cy="12" r="1" />
        <circle cx="19" cy="12" r="1" />
      </>
    ),
    logout: <path d="M9 4H4v16h5m6-12 4 4-4 4M9 12h10" />,
  }
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.65"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  )
}
const navigation: { name: string; icon: IconName }[] = [
  { name: "Dashboard", icon: "grid" },
  { name: "My Documents", icon: "file" },
  { name: "Ask Your Notes", icon: "chat" },
  { name: "AI Summarizer", icon: "spark" },
  { name: "Revision Questions", icon: "cards" },
]
type Doc = {
  id: number
  backendDocId?: string
  name: string
  subject: string
  size: string
  pages: number
  date: string
  color: string
  // Real extraction & RAG fields from FastAPI backend
  totalCharacters?: number
  totalWords?: number
  chunkCount?: number
  hasExtractableText?: boolean
  ocrSuggested?: boolean
  extractedPages?: PageContent[]
  statusMessage?: string
}
export type ActivityItem = {
  id: string
  type: "upload" | "ask" | "summary" | "quiz"
  icon: IconName
  color: string
  title: string
  description: string
  time: string
  timestamp: number
}

function getDocFormat(filename: string): string {
  const ext = filename.split(".").pop()?.toUpperCase() || "DOC"
  if (["PNG", "JPG", "JPEG", "WEBP", "BMP", "TIFF"].includes(ext)) return "IMG"
  if (["PPT", "PPTX"].includes(ext)) return "PPT"
  if (["DOC", "DOCX"].includes(ext)) return "DOC"
  if (["TXT", "MD", "MARKDOWN"].includes(ext)) return "TXT"
  if (ext === "PDF") return "PDF"
  return ext.slice(0, 4)
}

const initialDocs: Doc[] = []
function Button({
  children,
  onClick,
  variant = "primary",
  disabled = false,
  type = "button",
}: {
  children: ReactNode
  onClick?: () => void
  variant?: string
  disabled?: boolean
  type?: "button" | "submit"
}) {
  return (
    <button
      type={type}
      className={`button ${variant}`}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  )
}
function BookArt() {
  return (
    <div className="book-art" aria-hidden="true">
      <div className="orbit orbit-one" />
      <div className="orbit orbit-two" />
      <span className="art-star star-one">✦</span>
      <span className="art-star star-two">✦</span>
      <div className="floating-note">
        <Icon name="spark" size={19} />
        <span>
          A little clarity.
          <br />
          <strong>A lot of possibility.</strong>
        </span>
      </div>
      <svg viewBox="0 0 300 210" className="book-svg">
        <defs>
          <linearGradient id="page" x1="0" x2="1">
            <stop stopColor="#fff" />
            <stop offset="1" stopColor="#e9e6fb" />
          </linearGradient>
          <linearGradient id="cover" x1="0" y1="0" x2="1" y2="1">
            <stop stopColor="#9688d8" />
            <stop offset="1" stopColor="#6553ad" />
          </linearGradient>
        </defs>
        <ellipse
          cx="152"
          cy="184"
          rx="112"
          ry="12"
          fill="#b6aad8"
          opacity=".23"
        />
        <path
          d="m28 65 118 26 122-34-8 108-113 35-112-35Z"
          fill="url(#cover)"
        />
        <path d="m38 62 108 24 111-30-6 99-104 34-104-33Z" fill="#d0c8eb" />
        <path d="M146 83Q90 35 40 47l6 101q50-4 101 34Z" fill="url(#page)" />
        <path d="M146 83q50-51 106-41l-7 104q-51 1-98 36Z" fill="#faf9ff" />
        <path d="m146 84 1 97" stroke="#c9c1e5" strokeWidth="2" />
        <g stroke="#d4cee6" strokeWidth="3" strokeLinecap="round">
          <path d="m60 70 62 21M62 82l60 20M63 95l59 20M65 108l56 19M66 122l55 18M170 87l59-22M169 100l59-22M169 113l58-22M168 126l57-22M168 139l44-17" />
        </g>
        <path d="m182 144 15-5-1 35-9-4-8 10Z" fill="#9b8acb" />
      </svg>
      <div className="art-badge">
        <Icon name="check" size={14} /> Made for your next “aha!”
      </div>
    </div>
  )
}
function FormattedAnswer({ text }: { text: string }) {
  const lines = text.split("\n")
  return (
    <div className="answer-formatted-content" style={{ fontSize: "14px", lineHeight: "1.8", color: "#f1f5f9" }}>
      {lines.map((line, idx) => {
        const trimmed = line.trim()
        if (!trimmed) return <div key={idx} style={{ height: "6px" }} />
        if (trimmed.startsWith("### ")) {
          return (
            <h4
              key={idx}
              style={{
                margin: idx === 0 ? "2px 0 8px" : "18px 0 8px",
                fontSize: "14px",
                fontWeight: 700,
                color: "#c4b5fd",
                letterSpacing: "0.3px",
                textTransform: "uppercase",
                display: "flex",
                alignItems: "center",
                gap: "6px",
              }}
            >
              {trimmed.replace(/^###\s*/, "")}
            </h4>
          )
        }
        const isListItem = /^\d+\.\s+/.test(trimmed) || trimmed.startsWith("• ") || trimmed.startsWith("- ")
        const cleanTrimmed = trimmed.replace(/^[-•]\s*/, "")
        const parts = cleanTrimmed.split(/(\*\*.*?\*\*|\*\(Page \d+\)\*)/g)
        return (
          <p
            key={idx}
            style={{
              margin: isListItem ? "6px 0" : "8px 0",
              paddingLeft: isListItem ? "12px" : "0",
              lineHeight: 1.75,
              color: "#f1f5f9",
            }}
          >
            {isListItem && <span style={{ color: "#a78bfa", marginRight: "6px", fontWeight: 700 }}>•</span>}
            {parts.map((p, pIdx) => {
              if (p.startsWith("**") && p.endsWith("**")) {
                return (
                  <strong key={pIdx} style={{ fontWeight: 700, color: "#ffffff", background: "rgba(168, 85, 247, 0.22)", border: "1px solid rgba(192, 132, 252, 0.35)", padding: "1px 6px", borderRadius: "4px" }}>
                    {p.slice(2, -2)}
                  </strong>
                )
              }
              if (p.startsWith("*(Page ") && p.endsWith(")*")) {
                return (
                  <span key={pIdx} style={{ color: "#a78bfa", fontWeight: 600, fontSize: "12px", marginLeft: "4px", background: "rgba(99, 102, 241, 0.15)", padding: "1px 5px", borderRadius: "3px" }}>
                    {p.slice(1, -1)}
                  </span>
                )
              }
              return p
            })}
          </p>
        )
      })}
    </div>
  )
}

export default function App() {
  const [page, setPage] = useState("Dashboard")
  const [mobileOpen, setMobileOpen] = useState(false)
  const [docs, setDocs] = useState<Doc[]>(initialDocs)
  const [modal, setModal] =
    useState<"upload" | "rename" | "delete" | "help" | "settings" | null>(null)
  const [activeDoc, setActiveDoc] = useState<number>(0)
  const [activities, setActivities] = useState<ActivityItem[]>(() => {
    try {
      const stored = localStorage.getItem("studysphere_activities_v2")
      return stored ? JSON.parse(stored) : []
    } catch {
      return []
    }
  })

  const addActivity = (
    type: ActivityItem["type"],
    title: string,
    description: string,
    icon: IconName,
    color: string
  ) => {
    const newItem: ActivityItem = {
      id: "act_" + Date.now() + "_" + Math.random().toString(36).substring(2, 7),
      type,
      title,
      description,
      time: "Just now",
      timestamp: Date.now(),
      icon,
      color,
    }
    setActivities((prev) => {
      const updated = [newItem, ...prev].slice(0, 30)
      try {
        localStorage.setItem("studysphere_activities_v2", JSON.stringify(updated))
      } catch {
        // ignore
      }
      return updated
    })
  }

  const [rename, setRename] = useState("")
  const [search, setSearch] = useState("")
  const [subject, setSubject] = useState("All subjects")
  const [toast, setToast] = useState("")
  const [uploadName, setUploadName] = useState("")
  const [uploadError, setUploadError] = useState("")
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [question, setQuestion] = useState("")
  const [messages, setMessages] = useState<{
    question: string
    answer: string
    cited_sources?: SourceChunk[]
    retrieved_sources?: SourceChunk[]
    sources?: SourceChunk[]
    retrieval_status?: "found" | "insufficient_evidence" | "empty_query" | "no_documents"
    generation_provider?: string
    model_name?: string
  }[]>([])
  const [expandedRetrievals, setExpandedRetrievals] = useState<number[]>([])
  const [busy, setBusy] = useState(false)
  const [summaryMode, setSummaryMode] = useState<SummaryMode>("short")
  const [summaryResult, setSummaryResult] = useState<SummarizeResponse | null>(null)
  const [summaryBusy, setSummaryBusy] = useState(false)
  const [summaryError, setSummaryError] = useState("")
  const [tab, setTab] = useState("Multiple Choice")
  const [count, setCount] = useState("5")
  const [difficulty, setDifficulty] = useState("Medium")
  const [generated, setGenerated] = useState(false)
  const [expanded, setExpanded] = useState<number[]>([])
  const [saved, setSaved] = useState<number[]>([])
  const [savedOnly, setSavedOnly] = useState(false)
  // Phase 5: Real revision questions state
  const [selectedQTypes, setSelectedQTypes] = useState<QuestionType[]>(["mcq"])
  const [quizQuestions, setQuizQuestions] = useState<QuestionItem[]>([])
  const [quizSessionId, setQuizSessionId] = useState<string>("")
  const [userAnswers, setUserAnswers] = useState<Record<string, string>>({})
  const [quizSubmitted, setQuizSubmitted] = useState(false)
  const [quizEvaluation, setQuizEvaluation] = useState<EvaluateQuizResponse | null>(null)
  const [questionsBusy, setQuestionsBusy] = useState(false)
  const [questionsError, setQuestionsError] = useState("")
  const [activeSessionId, setActiveSessionId] = useState<string>("")
  const fileInput = useRef<HTMLInputElement>(null)
  const doc = docs.find((d) => d.id === activeDoc) || docs[0] || null

  useEffect(() => {
    // Sync previously uploaded and indexed documents from backend SQLite store
    fetchStoredDocuments()
      .then((data) => {
        if (data.documents && data.documents.length > 0) {
          const loadedDocs: Doc[] = data.documents.map((d, idx) => ({
            id: 2000 + idx,
            backendDocId: d.id,
            name: d.filename,
            subject: "Uploaded Notes",
            size: `${(d.total_characters / 1024).toFixed(1)} KB`,
            pages: d.page_count,
            date: new Date(d.created_at).toLocaleDateString(),
            color: (["purple", "blue", "peach", "green"] as const)[idx % 4],
            totalCharacters: d.total_characters,
            totalWords: d.total_words,
            chunkCount: d.chunk_count,
            hasExtractableText: d.chunk_count > 0,
            statusMessage: `Indexed ${d.chunk_count} semantic chunks.`,
          }))

          setDocs(loadedDocs)
          setActiveDoc((prev) => {
            if (loadedDocs.some((ld) => ld.id === prev)) return prev
            return loadedDocs[0].id
          })
        } else {
          setDocs([])
        }
      })
      .catch(() => {
        // Backend not yet running; keep existing state
      })
  }, [])

  // Synchronize conversational chat sessions and messages for the active document
  useEffect(() => {
    const currentBackendDocId = doc?.backendDocId
    if (!currentBackendDocId || !doc) {
      setActiveSessionId("")
      setMessages([])
      return
    }

    let isCurrent = true
    listChatSessions(currentBackendDocId)
      .then(async (data) => {
        if (!isCurrent) return
        if (data.sessions && data.sessions.length > 0) {
          const session = data.sessions[0]
          setActiveSessionId(session.id)
          try {
            const history = await getChatMessages(session.id)
            if (!isCurrent) return
            if (history.messages && history.messages.length > 0) {
              const pairs: {
                question: string
                answer: string
                cited_sources?: SourceChunk[]
                retrieved_sources?: SourceChunk[]
                sources?: SourceChunk[]
                retrieval_status?: "found" | "insufficient_evidence" | "empty_query" | "no_documents"
                generation_provider?: string
                model_name?: string
              }[] = []

              let currentQ: string | null = null
              for (const m of history.messages) {
                if (m.role === "user") {
                  currentQ = m.content
                } else if (m.role === "assistant") {
                  pairs.push({
                    question: currentQ || "Question",
                    answer: m.content,
                    cited_sources: m.cited_sources,
                    sources: m.cited_sources,
                    retrieval_status: (m.retrieval_status as any) || "found",
                    generation_provider: m.generation_provider,
                  })
                  currentQ = null
                }
              }
              setMessages(pairs)
            } else {
              setMessages([])
            }
          } catch {
            // Keep existing messages on error
          }
        } else {
          // If no session exists yet, create one
          try {
            const created = await createChatSession(`Chat - ${doc.name}`, currentBackendDocId)
            if (isCurrent) {
              setActiveSessionId(created.id)
              setMessages([])
            }
          } catch {
            // Backend offline
          }
        }
      })
      .catch(() => {
        // Backend offline
      })

    return () => {
      isCurrent = false
    }
  }, [doc?.backendDocId, doc?.name])


  const notify = (message: string) => {
    setToast(message)
    window.setTimeout(() => setToast(""), 4500)
  }
  const simulate = (action: () => void) => {
    setBusy(true)
    setTimeout(() => {
      setBusy(false)
      action()
    }, 450)
  }
  const go = (destination: string) => {
    setPage(destination)
    setMobileOpen(false)
  }
  const ask = async (text = question) => {
    if (!text.trim() || busy || !doc) return
    const q = text.trim()
    setQuestion("")
    setBusy(true)

    try {
      let currentSessionId = activeSessionId
      if (!currentSessionId && doc.backendDocId) {
        try {
          const created = await createChatSession(`Chat - ${doc.name}`, doc.backendDocId)
          currentSessionId = created.id
          setActiveSessionId(created.id)
        } catch {
          // Fall back to stateless ask
        }
      }

      let res: {
        question: string
        answer: string
        cited_sources?: SourceChunk[]
        retrieved_sources?: SourceChunk[]
        sources?: SourceChunk[]
        retrieval_status: "found" | "insufficient_evidence" | "empty_query" | "no_documents"
        generation_provider: string
        model_name?: string
      }

      if (currentSessionId) {
        res = await askInChatSession(currentSessionId, q, doc.backendDocId)
      } else {
        res = await askQuestion(q, doc.backendDocId)
      }

      setMessages((prev) => [
        ...prev,
        {
          question: q,
          answer: res.answer,
          cited_sources: res.cited_sources || res.sources,
          retrieved_sources: res.retrieved_sources || res.sources,
          sources: res.cited_sources || res.sources,
          retrieval_status: res.retrieval_status,
          generation_provider: res.generation_provider,
          model_name: res.model_name,
        },
      ])
      addActivity(
        "ask",
        "Question asked",
        `Asked about "${doc.name}": "${q.length > 50 ? q.slice(0, 50) + "..." : q}"`,
        "chat",
        "blue"
      )
    } catch (err: unknown) {
      const errMsg =
        err && typeof err === "object" && "message" in err
          ? (err as { message: string }).message
          : "Could not retrieve answer from backend."
      setMessages((prev) => [
        ...prev,
        {
          question: q,
          answer: `Connection Error: ${errMsg}`,
          retrieval_status: "insufficient_evidence",
          generation_provider: "error_handler",
        },
      ])
    } finally {
      setBusy(false)
    }
  }
  const handleGenerateQuestions = async () => {
    if (!doc) {
      setQuestionsError("Please upload notes before generating revision questions.")
      return
    }
    if (!doc.backendDocId) {
      setQuestionsError(`"${doc.name}" has not been indexed in SQLite store. Please upload notes first.`)
      return
    }
    setQuestionsBusy(true)
    setQuestionsError("")
    setQuizQuestions([])
    setUserAnswers({})
    setQuizSubmitted(false)
    setQuizEvaluation(null)
    setQuizSessionId("")
    try {
      const diffMap: Record<string, QuestionDifficulty> = {
        Easy: "easy",
        Medium: "medium",
        Challenging: "hard",
      }
      const res = await generateQuestions(doc.backendDocId, {
        count: Number(count),
        types: selectedQTypes,
        difficulty: diffMap[difficulty] ?? "medium",
      })
      setQuizSessionId(res.session_id)
      setQuizQuestions(res.questions)
      setGenerated(true)
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "message" in err
          ? (err as { message: string }).message
          : "Failed to generate questions from backend."
      setQuestionsError(msg)
    } finally {
      setQuestionsBusy(false)
    }
  }

  const handleSubmitQuiz = async () => {
    if (!doc?.backendDocId || !quizSessionId || quizSubmitted) return
    const answers: QuestionAnswerSubmission[] = quizQuestions.map((q) => ({
      question_id: q.question_id,
      user_answer: userAnswers[q.question_id] ?? "",
    }))
    setQuestionsBusy(true)
    try {
      const result = await evaluateQuiz(quizSessionId, answers, doc.backendDocId)
      setQuizEvaluation(result)
      setQuizSubmitted(true)
      addActivity(
        "quiz",
        "Revision quiz completed",
        `Scored ${result.percentage}% (${result.correct_count}/${result.total_questions}) on "${doc.name}"`,
        "cards",
        "green"
      )
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "message" in err
          ? (err as { message: string }).message
          : "Failed to evaluate quiz."
      setQuestionsError(msg)
    } finally {
      setQuestionsBusy(false)
    }
  }

  const handleGenerateSummary = async () => {
    if (!doc) {
      setSummaryError("Please upload notes before generating a summary.")
      return
    }
    if (!doc.backendDocId) {
      setSummaryError(`"${doc.name}" has not yet been indexed in SQLite. Please upload notes first.`)
      setSummaryResult(null)
      return
    }

    setSummaryBusy(true)
    setSummaryError("")
    try {
      const res = await summarizeDocument(doc.backendDocId, summaryMode)
      setSummaryResult(res)
      addActivity(
        "summary",
        "Summary generated",
        `Created ${res.summary_mode.replace("_", " ")} summary for "${doc.name}"`,
        "spark",
        "peach"
      )
    } catch (err: unknown) {
      const message =
        err && typeof err === "object" && "message" in err
          ? (err as { message: string }).message
          : "Failed to generate summary from the document."
      setSummaryError(message)
      setSummaryResult(null)
    } finally {
      setSummaryBusy(false)
    }
  }

  const ALLOWED_EXTENSIONS = [
    ".pdf",
    ".pptx",
    ".ppt",
    ".docx",
    ".doc",
    ".txt",
    ".md",
    ".markdown",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
  ]
  const selectFile = (file?: File) => {
    setUploadError("")
    if (!file) {
      setSelectedFile(null)
      setUploadName("")
      return
    }
    const lower = file.name.toLowerCase()
    const isAllowed = ALLOWED_EXTENSIONS.some((ext) => lower.endsWith(ext))
    if (!isAllowed) {
      setSelectedFile(null)
      setUploadName("")
      setUploadError(
        "Supported file formats: PDF (.pdf), PowerPoint (.pptx), Word (.docx), Text (.txt, .md), and Photos/Images (.png, .jpg, .webp)."
      )
      return
    }
    if (file.size > 50 * 1024 * 1024) {
      setSelectedFile(null)
      setUploadName("")
      setUploadError("Please choose a file smaller than 50 MB.")
      return
    }
    setSelectedFile(file)
    setUploadName(file.name)
  }

  const handleUpload = async () => {
    if (!selectedFile || uploading) return
    setUploading(true)
    setUploadError("")

    try {
      const result = await uploadPDFDocument(selectedFile)

      const sizeStr =
        selectedFile.size < 1024 * 1024
          ? `${(selectedFile.size / 1024).toFixed(1)} KB`
          : `${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB`

      // Generate a client-side session identifier linked to backend document ID
      const newDocId = Date.now()

      const newDoc: Doc = {
        id: newDocId,
        backendDocId: result.document_id,
        name: result.filename,
        subject: "Study Notes",
        size: sizeStr,
        pages: result.page_count,
        date: "Just now",
        color: result.has_extractable_text ? "purple" : "peach",
        totalCharacters: result.total_characters,
        totalWords: result.total_words,
        chunkCount: result.chunk_count,
        hasExtractableText: result.has_extractable_text,
        ocrSuggested: result.ocr_suggested,
        extractedPages: result.pages,
        statusMessage: result.message,
      }

      setDocs((prev) => [newDoc, ...prev.filter((d) => d.backendDocId !== newDoc.backendDocId)])
      setActiveDoc(newDocId)
      setSelectedFile(null)
      setUploadName("")
      setModal(null)

      addActivity(
        "upload",
        "Notes added to library",
        `Uploaded "${newDoc.name}" (${result.page_count} pg, ${result.chunk_count} chunks)`,
        "file",
        "purple"
      )

      if (result.has_extractable_text) {
        notify(
          `Extracted ${result.total_words.toLocaleString()} words and indexed ${result.chunk_count} semantic chunks from "${newDoc.name}".`,
        )
      } else {
        notify(
          `"${newDoc.name}" uploaded, but no extractable text found.`,
        )
      }
    } catch (err: unknown) {
      const message =
        err && typeof err === "object" && "message" in err
          ? (err as { message: string }).message
          : "Failed to upload and extract text from the document."
      setUploadError(message)
    } finally {
      setUploading(false)
    }
  }
  const documentSelect = (
    <label className="field">
      Study material
      <select
        value={doc?.id || ""}
        onChange={(e) => {
          setActiveDoc(Number(e.target.value))
          setSummaryResult(null)
          setSummaryError("")
          setMessages([])
          setGenerated(false)
          setSaved([])
        }}
      >
        {docs.length ? (
          docs.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name} {d.backendDocId ? `(Indexed · ${d.pages} pg)` : ""}
            </option>
          ))
        ) : (
          <option value="">No notes uploaded yet</option>
        )}
      </select>
    </label>
  )
  const openDoc = (id: number) => {
    setActiveDoc(id)
    setMessages([])
    go("Ask Your Notes")
  }
  return (
    <div className="app-shell" style={glassTheme}>
      <GradientBackground />
      {mobileOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setMobileOpen(false)}
        />
      )}
      <aside className={`sidebar ${mobileOpen ? "is-open" : ""}`}>
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault()
            go("Dashboard")
          }}
        >
          <span className="brand-icon">
            <Icon name="book" size={24} />
            <i />
          </span>
          <span>
            StudySphere<span className="brand-ai">AI</span>
          </span>
        </a>
        <div className="workspace-label">YOUR WORKSPACE</div>
        <nav aria-label="Main navigation">
          {navigation.map((item) => (
            <button
              key={item.name}
              onClick={() => go(item.name)}
              className={`nav-item ${page === item.name ? "active" : ""}`}
              aria-current={page === item.name ? "page" : undefined}
            >
              <Icon name={item.icon} />
              <span>{item.name}</span>
              {item.name === "My Documents" && (
                <span className="nav-count">{docs.length}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="study-tip">
            <span className="tip-icon">
              <Icon name="spark" size={19} />
            </span>
            <strong>A little progress, every day.</strong>
            <p>
              Your next breakthrough starts
              <br />
              with one good question.
            </p>
            <button onClick={() => go("Ask Your Notes")}>
              Let’s learn something <Icon name="arrow" size={16} />
            </button>
          </div>
          <button className="nav-item" onClick={() => setModal("settings")}>
            <Icon name="settings" />
            <span>Settings</span>
          </button>
          <button className="nav-item" onClick={() => setModal("help")}>
            <Icon name="help" />
            <span>Help & support</span>
            <Icon name="arrow" size={16} />
          </button>
          <div className="profile">
            <div className="avatar">AJ</div>
            <div>
              <strong>Alex Johnson</strong>
              <span>Student workspace</span>
            </div>
            <span className="profile-dot" />
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <GlassNavbar>
          <div className="breadcrumb">
            <button
              className="icon-button mobile-toggle"
              aria-label="Open navigation"
              onClick={() => setMobileOpen(true)}
            >
              <Icon name="menu" />
            </button>
            <span className="breadcrumb-home">
              <Icon name="grid" size={16} />
              <span>Workspace</span>
              <span className="slash">/</span>
            </span>
            <strong>{page}</strong>
          </div>
          <div className="topbar-right">
            <button
              className="nav-cta"
              onClick={() => {
                setSelectedFile(null)
                setUploadName("")
                setUploadError("")
                setModal("upload")
              }}
            >
              <Icon name="plus" size={15} />
              Add notes
            </button>
            <span className="demo-badge">
              <span /> Demo workspace
            </span>
            <div className="header-divider" />
            <button
              className="icon-button notification-button"
              aria-label="Notifications"
              onClick={() =>
                notify("You’re all caught up. Your study space is ready.")
              }
            >
              <Icon name="bell" size={19} />
              <i />
            </button>
            <div className="avatar small">AJ</div>
          </div>
        </GlassNavbar>
        <main>
          {page === "Dashboard" ? (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">
                    <Icon name="spark" size={14} /> A GOOD DAY TO MAKE PROGRESS
                  </div>
                  <h1>Ready to learn something new?</h1>
                  <p>
                    Pick up where you left off or upload your next set of notes.
                  </p>
                </div>
                <Button
                  onClick={() => {
                    setSelectedFile(null)
                    setUploadName("")
                    setUploadError("")
                    setModal("upload")
                  }}
                >
                  <Icon name="plus" size={18} />
                  Upload Notes
                </Button>
              </div>
              <Hero>
                <div className="hero-content">
                  <span className="hero-kicker">
                    <Icon name="spark" size={16} /> YOUR PERSONAL STUDY
                    COMPANION
                  </span>
                  <h2>
                    Less searching.
                    <br />
                    More understanding.
                  </h2>
                  <p>
                    Turn your study notes into clear answers, simple
                    <br className="desktop-break" /> summaries, and your next
                    lightbulb moment.
                  </p>
                  <Button onClick={() => go("Ask Your Notes")}>
                    Ask your first question <Icon name="arrow" size={17} />
                  </Button>
                  <span className="hero-footnote">
                    Your notes. Your questions. Instant understanding.
                  </span>
                </div>
                <BookArt />
              </Hero>
              <div className="stats-grid">
                {[
                  {
                    label: "Documents in library",
                    value: String(docs.length).padStart(2, "0"),
                    icon: "file" as IconName,
                    color: "purple",
                    detail: docs.length === 1 ? "1 document indexed in SQLite" : `${docs.length} documents indexed in SQLite`,
                    extra: "Persistent SQLite store",
                  },
                  {
                    label: "Questions asked",
                    value: String(activities.filter((a) => a.type === "ask").length).padStart(2, "0"),
                    icon: "chat" as IconName,
                    color: "blue",
                    detail: "Grounded semantic queries",
                    extra: "Real-time log",
                  },
                  {
                    label: "Study actions",
                    value: String(activities.length).padStart(2, "0"),
                    icon: "clock" as IconName,
                    color: "peach",
                    detail: "Uploads, Q&A, and Quizzes",
                    extra: activities.length > 0 ? "Saved in session" : "Ready to begin",
                  },
                ].map((s) => (
                  <section className="stat-card" key={s.label}>
                    <div className="stat-top">
                      <span className={`icon-box ${s.color}`}>
                        <Icon name={s.icon} />
                      </span>
                      <span>{s.label}</span>
                      <Icon name="more" size={18} />
                    </div>
                    <div className="stat-value">
                      {s.value}
                      <span
                        className={s.color === "peach" ? "neutral-trend" : ""}
                      >
                        {s.color !== "peach" && <span>↗</span>} {s.extra}
                      </span>
                    </div>
                    <p>{s.detail}</p>
                  </section>
                ))}
              </div>
              <div className="dashboard-columns">
                <div className="dashboard-left">
                  <section className="panel documents-panel">
                    <div className="section-heading">
                      <div>
                        <h2>
                          Recent documents{" "}
                          <span className="count-badge">{docs.length}</span>
                        </h2>
                        <p>A little knowledge, ready when you are.</p>
                      </div>
                      <button
                        className="text-button"
                        onClick={() => go("My Documents")}
                      >
                        View all <Icon name="arrow" size={15} />
                      </button>
                    </div>
                    <div className="document-list">
                      {docs.slice(0, 3).map((d) => (
                        <button
                          className="document-row"
                          key={d.id}
                          onClick={() => openDoc(d.id)}
                        >
                          <span className={`file-icon ${d.color}`}>
                            <Icon name="file" size={23} />
                            <small>{getDocFormat(d.name)}</small>
                          </span>
                          <span className="document-info">
                            <strong>{d.name}</strong>
                            <span>
                              {d.subject}
                              <i /> {d.pages} pages <i /> {d.size}
                              {d.totalWords ? <> <i /> {d.totalWords.toLocaleString()} words</> : null}
                            </span>
                          </span>
                          <span className="document-end">
                            <span className="ready-badge">
                              <span />
                              {d.extractedPages
                                ? d.hasExtractableText
                                  ? "Text ready"
                                  : "Needs OCR"
                                : "Indexed"}
                            </span>
                            <small>{d.date}</small>
                          </span>
                          <Icon name="chevron" size={16} />
                        </button>
                      ))}
                      {!docs.length && (
                        <div className="empty-state" style={{ padding: "36px 20px", textAlign: "center" }}>
                          <span className="icon-box purple" style={{ width: "48px", height: "48px", margin: "0 auto 14px", borderRadius: "12px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                            <Icon name="file" size={26} />
                          </span>
                          <h3 style={{ fontSize: "16px", fontWeight: 700, margin: "0 0 6px", color: "#f8fafc" }}>
                            No notes uploaded yet
                          </h3>
                          <p style={{ fontSize: "12.5px", color: "#94a3b8", margin: "0 auto 16px", maxWidth: "340px", lineHeight: "1.6" }}>
                            Start studying by uploading your notes in any format — PDF, PPT slides, Word documents, text notes, or lecture photos (with OCR).
                          </p>
                          <Button
                            onClick={() => {
                              setSelectedFile(null)
                              setUploadName("")
                              setUploadError("")
                              setModal("upload")
                            }}
                          >
                            <Icon name="upload" size={16} /> Upload Your First Notes
                          </Button>
                        </div>
                      )}
                    </div>
                  </section>
                  <Features
                    features={[
                      {
                        title: "Ask a question",
                        description: "Get clarity from your own notes.",
                        icon: <Icon name="chat" />,
                        color: "purple",
                        onClick: () => go("Ask Your Notes"),
                      },
                      {
                        title: "Summarize notes",
                        description: "The big ideas. Without the overwhelm.",
                        icon: <Icon name="spark" />,
                        color: "blue",
                        onClick: () => go("AI Summarizer"),
                      },
                      {
                        title: "Create revision questions",
                        description: "A little practice goes a long way.",
                        icon: <Icon name="cards" />,
                        color: "peach",
                        onClick: () => go("Revision Questions"),
                      },
                    ]}
                  />
                </div>
                <aside className="panel activity-panel">
                  <div className="section-heading">
                    <h2>Recent activity</h2>
                    <span className="live-dot" />
                  </div>
                  <div className="activity-list">
                    {activities.length > 0 ? (
                      activities.slice(0, 5).map((a, i) => (
                        <div className="activity-item" key={a.id || i}>
                          <div className="timeline-icon">
                            <span className={`icon-box ${a.color}`}>
                              <Icon name={a.icon} size={16} />
                            </span>
                            {i < Math.min(activities.length, 5) - 1 && <i />}
                          </div>
                          <div>
                            <strong>{a.title}</strong>
                            <p>{a.description}</p>
                            <span>{a.time}</span>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="empty-state" style={{ padding: "28px 14px", textAlign: "center" }}>
                        <span className="icon-box blue" style={{ width: "40px", height: "40px", margin: "0 auto 10px", borderRadius: "10px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                          <Icon name="clock" size={20} />
                        </span>
                        <strong style={{ display: "block", fontSize: "13px", color: "#f8fafc", marginBottom: "4px" }}>
                          No recent activity yet
                        </strong>
                        <p style={{ fontSize: "11.5px", color: "#94a3b8", margin: 0, lineHeight: "1.5" }}>
                          When you upload notes, ask questions, or generate revision quizzes, your progress will be tracked here in real-time.
                        </p>
                      </div>
                    )}
                  </div>
                  <div className="activity-footer">
                    <Icon name="book" size={19} />
                    <p>
                      Every small step counts.
                      <br />
                      <strong>You’re building something good.</strong>
                    </p>
                  </div>
                </aside>
              </div>
              <Footer onNavigate={go} onHelp={() => setModal("help")} />
            </>
          ) : (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">YOUR SPACE TO UNDERSTAND</div>
                  <h1>{page}</h1>
                  <p>
                    {page === "My Documents"
                      ? "Your knowledge library. A home for every lightbulb moment."
                      : page === "Ask Your Notes"
                        ? "Big questions, clearer understanding. Start with your study material."
                        : page === "AI Summarizer"
                          ? "Less overwhelm. More of what matters."
                          : "Turn what you’ve learned into lasting understanding."}
                  </p>
                </div>
                {page === "My Documents" && (
                  <Button
                    onClick={() => {
                      setSelectedFile(null)
                      setUploadName("")
                      setUploadError("")
                      setModal("upload")
                    }}
                  >
                    <Icon name="upload" size={18} />
                    Upload Notes
                  </Button>
                )}
              </div>
              <div className="demo-notice">
                <Icon name="spark" size={17} />
                <span>
                  Universal Study Workspace · Real multi-format text extraction (PDF, PPT, Word, TXT, OCR) & persistent SQLite store.
                </span>
              </div>
              {page === "My Documents" && (
                <>
                  <button
                    className="dropzone"
                    onClick={() => {
                      setSelectedFile(null)
                      setUploadName("")
                      setUploadError("")
                      setModal("upload")
                    }}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => {
                      e.preventDefault()
                      selectFile(e.dataTransfer.files[0])
                      setModal("upload")
                    }}
                  >
                    <span className="icon-box purple">
                      <Icon name="upload" />
                    </span>
                    <strong>Give your notes a home</strong>
                    <p>
                      Drag and drop any study notes, or <span>browse files</span>
                    </p>
                    <small>PDF, PPTX, Word DOCX, TXT, or Photos (OCR) · Up to 50 MB</small>
                  </button>
                  <div className="document-toolbar">
                    <div className="search-input">
                      <Icon name="search" size={18} />
                      <GlassInput
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        placeholder="Search your documents…"
                        aria-label="Search documents"
                      />
                    </div>
                    <select
                      aria-label="Filter by subject"
                      value={subject}
                      onChange={(e) => setSubject(e.target.value)}
                    >
                      <option>All subjects</option>
                      {Array.from(new Set(docs.map((d) => d.subject))).map(
                        (s) => (
                          <option key={s}>{s}</option>
                        ),
                      )}
                    </select>
                    <select aria-label="Filter by file type">
                      <option>All file formats</option>
                    </select>
                    <span>{docs.length} documents</span>
                  </div>
                  <div className="library-grid">
                    {docs
                      .filter(
                        (d) =>
                          d.name.toLowerCase().includes(search.toLowerCase()) &&
                          (subject === "All subjects" || d.subject === subject),
                      )
                      .map((d) => (
                        <section key={d.id} className="panel library-card">
                          <div className="library-card-top">
                            <span className={`file-icon ${d.color}`}>
                              <Icon name="file" size={25} />
                              <small>{getDocFormat(d.name)}</small>
                            </span>
                            <span className="ready-badge">
                              <span />
                              {d.extractedPages
                                ? d.hasExtractableText
                                  ? "Text extracted"
                                  : "Needs OCR"
                                : "Indexed"}
                            </span>
                          </div>
                          <h2>{d.name}</h2>
                          <span className={`subject-badge ${d.color}`}>
                            {d.subject}
                          </span>
                          <p>
                            {d.pages ? `${d.pages} pages · ` : ""}
                            {d.size}
                            {d.totalWords ? ` · ${d.totalWords.toLocaleString()} words` : ""}
                            <span>{d.date}</span>
                          </p>
                          <div className="library-actions">
                            <button onClick={() => openDoc(d.id)}>
                              Open <Icon name="arrow" size={15} />
                            </button>
                            <button
                              onClick={() => {
                                setActiveDoc(d.id)
                                setRename(d.name)
                                setModal("rename")
                              }}
                            >
                              Rename
                            </button>
                            <button
                              className="danger-text"
                              onClick={() => {
                                setActiveDoc(d.id)
                                setModal("delete")
                              }}
                            >
                              Delete
                            </button>
                          </div>
                        </section>
                      ))}
                  </div>
                  {docs.length === 0 ? (
                    <div className="empty-state panel" style={{ padding: "48px 24px", textAlign: "center" }}>
                      <span className="icon-box purple" style={{ width: "56px", height: "56px", margin: "0 auto 16px", borderRadius: "14px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <Icon name="file" size={30} />
                      </span>
                      <h2 style={{ fontSize: "18px", fontWeight: 700, margin: "0 0 8px", color: "#f8fafc" }}>
                        Your notes library is empty
                      </h2>
                      <p style={{ fontSize: "13px", color: "#94a3b8", margin: "0 auto 20px", maxWidth: "420px", lineHeight: "1.6" }}>
                        No mock data. Upload your real academic notes (PDF, PowerPoint slides, Word documents, text notes, or lecture photos). All extracted content and embeddings are securely saved to the persistent database.
                      </p>
                      <Button
                        onClick={() => {
                          setSelectedFile(null)
                          setUploadName("")
                          setUploadError("")
                          setModal("upload")
                        }}
                      >
                        <Icon name="upload" size={17} /> Upload Notes Now
                      </Button>
                    </div>
                  ) : !docs.filter(
                    (d) =>
                      d.name.toLowerCase().includes(search.toLowerCase()) &&
                      (subject === "All subjects" || d.subject === subject),
                  ).length ? (
                    <div className="empty-state panel">
                      <Icon name="search" size={28} />
                      <h2>No matching documents found</h2>
                      <p>Try a different search keyword or subject filter.</p>
                      <Button
                        variant="secondary"
                        onClick={() => {
                          setSearch("")
                          setSubject("All subjects")
                        }}
                      >
                        Clear filters
                      </Button>
                    </div>
                  ) : null}
                </>
              )}
              {page === "Ask Your Notes" && (
                docs.length === 0 ? (
                  <section className="panel empty-state" style={{ padding: "56px 24px", textAlign: "center", minHeight: "380px", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
                    <span className="icon-box purple" style={{ width: "56px", height: "56px", margin: "0 auto 16px", borderRadius: "14px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <Icon name="chat" size={30} />
                    </span>
                    <h2 style={{ fontSize: "19px", fontWeight: 700, margin: "0 0 8px", color: "#f8fafc" }}>
                      Upload notes to start asking questions
                    </h2>
                    <p style={{ fontSize: "13.5px", color: "#94a3b8", margin: "0 auto 22px", maxWidth: "440px", lineHeight: "1.6" }}>
                      StudySphere AI answers are strictly grounded in your notes using semantic retrieval. Upload a PDF, PPT, Word doc, or photo to start asking questions!
                    </p>
                    <Button
                      onClick={() => {
                        setSelectedFile(null)
                        setUploadName("")
                        setUploadError("")
                        setModal("upload")
                      }}
                    >
                      <Icon name="upload" size={17} /> Upload Notes
                    </Button>
                  </section>
                ) : (
                <section className="panel conversation-panel">
                  <div className="conversation-header">
                    {documentSelect}
                    <span className="grounded-label">
                      <Icon name="book" size={16} /> Designed to use your
                      selected notes
                    </span>
                  </div>
                  <div className="conversation-body" aria-live="polite">
                    {!messages.length && !busy ? (
                      <div className="conversation-empty">
                        <div className="large-spark">
                          <Icon name="spark" size={32} />
                        </div>
                        <span className="eyebrow">
                          A LITTLE CURIOSITY GOES A LONG WAY
                        </span>
                        <h2>What would you like to understand?</h2>
                        <p>
                          Ask a question, untangle a tricky topic, or explore an
                          idea.
                          <br />
                          Your selected notes are the starting point.
                        </p>
                        <div className="suggestions">
                          {[
                            "Explain this topic simply",
                            "Give me an example",
                            "Summarize this chapter",
                          ].map((q) => (
                            <button
                              key={q}
                              disabled={!doc}
                              onClick={() => ask(q)}
                            >
                              {q}
                              <Icon name="arrow" size={15} />
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : (
                      messages.map((m, i) => (
                        <div className="message-pair" key={i}>
                          <div className="user-message">
                            <span className="avatar small">AJ</span>
                            <div>
                              <strong>You</strong>
                              <p>{m.question}</p>
                            </div>
                          </div>
                          <div className="answer-message">
                            <span className="answer-icon">
                              <Icon name="spark" size={18} />
                            </span>
                            <div>
                              <strong>
                                StudySphere AI{" "}
                                <span className="sample-tag" style={{ marginLeft: "8px" }}>
                                  {m.retrieval_status === "insufficient_evidence"
                                    ? "Insufficient Evidence"
                                    : m.generation_provider === "google_gemini"
                                    ? "Gemini Flash Grounded"
                                    : m.generation_provider === "local_ollama"
                                    ? "Local Ollama Grounded"
                                    : m.generation_provider === "extractive_grounded_synthesizer"
                                    ? "Structured Grounded Synthesis (Notes)"
                                    : "RAG Grounded Response"}
                                </span>
                              </strong>
                              <div style={{ marginTop: "10px" }}>
                                <FormattedAnswer text={m.answer} />
                              </div>
                              {((m.cited_sources && m.cited_sources.length > 0) || (m.sources && m.sources.length > 0)) && (
                                <div className="sources" style={{ marginTop: "18px" }}>
                                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
                                    <span style={{ fontWeight: 700, letterSpacing: "0.8px", textTransform: "uppercase", fontSize: "9.5px", color: "#c4b5fd" }}>
                                      EVIDENCE CITED IN ANSWER ({(m.cited_sources || m.sources || []).length} PASSAGE{(m.cited_sources || m.sources || []).length === 1 ? "" : "S"})
                                    </span>
                                    <small style={{ fontSize: "8.5px", color: "#94a3b8" }}>
                                      Scores indicate semantic similarity, not accuracy confidence
                                    </small>
                                  </div>
                                  <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                                    {(m.cited_sources || m.sources || []).map((s, sIdx) => (
                                      <div
                                        key={sIdx}
                                        className="source-item-card"
                                        style={{
                                          padding: "12px 14px",
                                          background: "rgba(21, 30, 58, 0.75)",
                                          border: "1px solid rgba(139, 92, 246, 0.25)",
                                          borderRadius: "10px",
                                        }}
                                      >
                                        <p style={{ margin: 0, fontWeight: 650, fontSize: "12px", display: "flex", alignItems: "center", gap: "7px", color: "#c4b5fd" }}>
                                          <Icon name="file" size={15} />
                                          {s.document_name} · Page {s.page_number} · Semantic similarity: {s.similarity_score.toFixed(2)}
                                        </p>
                                        <small style={{ display: "block", marginTop: "6px", fontStyle: "italic", color: "#cbd5e1", lineHeight: 1.65, fontSize: "12px" }}>
                                          "{s.excerpt.length > 220 ? s.excerpt.slice(0, 220) + "..." : s.excerpt}"
                                        </small>
                                      </div>
                                    ))}
                                  </div>

                                  {/* Optional transparency view for remaining candidate passages */}
                                  {m.retrieved_sources && m.retrieved_sources.length > (m.cited_sources?.length || m.sources?.length || 0) && (
                                    <div style={{ marginTop: "12px", borderTop: "1px dashed rgba(139, 92, 246, 0.25)", paddingTop: "10px" }}>
                                      <button
                                        type="button"
                                        onClick={() => {
                                          setExpandedRetrievals((prev) =>
                                            prev.includes(i) ? prev.filter((id) => id !== i) : [...prev, i]
                                          )
                                        }}
                                        style={{
                                          background: "none",
                                          border: "0",
                                          color: "#a78bfa",
                                          fontSize: "11px",
                                          fontWeight: 600,
                                          display: "flex",
                                          alignItems: "center",
                                          gap: "6px",
                                          padding: "4px 0",
                                          cursor: "pointer",
                                        }}
                                      >
                                        <Icon name={expandedRetrievals.includes(i) ? "close" : "search"} size={13} />
                                        {expandedRetrievals.includes(i)
                                          ? "Hide candidate search passages"
                                          : `Inspect ${m.retrieved_sources.length - (m.cited_sources?.length || m.sources?.length || 0)} other retrieved candidate passage(s)`}
                                      </button>
                                      {expandedRetrievals.includes(i) && (
                                        <div style={{ display: "flex", flexDirection: "column", gap: "8px", marginTop: "10px" }}>
                                          {m.retrieved_sources
                                            .filter(
                                              (rs) =>
                                                !(m.cited_sources || m.sources || []).some(
                                                  (cs) => cs.chunk_index === rs.chunk_index && cs.page_number === rs.page_number
                                                )
                                            )
                                            .map((rs, rsIdx) => (
                                              <div
                                                key={rsIdx}
                                                style={{
                                                  padding: "10px 12px",
                                                  background: "rgba(15, 23, 42, 0.65)",
                                                  border: "1px solid rgba(148, 163, 184, 0.15)",
                                                  borderRadius: "8px",
                                                }}
                                              >
                                                <p style={{ margin: 0, fontSize: "11px", color: "#a5b4fc", fontWeight: 600 }}>
                                                  Candidate: {rs.document_name} · Page {rs.page_number} · Similarity: {rs.similarity_score.toFixed(2)} (Not cited in answer)
                                                </p>
                                                <small style={{ display: "block", marginTop: "4px", color: "#94a3b8", fontSize: "11px", lineHeight: 1.5 }}>
                                                  "{rs.excerpt.slice(0, 160)}..."
                                                </small>
                                              </div>
                                            ))}
                                        </div>
                                      )}
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                    {busy && (
                      <div className="loading-state">
                        <span className="spinner" />
                        Retrieving semantic document chunks and synthesizing answer…
                      </div>
                    )}
                  </div>
                  <form
                    className="question-form"
                    onSubmit={(e) => {
                      e.preventDefault()
                      ask()
                    }}
                  >
                    <div className="question-input">
                      <textarea
                        aria-label="Your question"
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        placeholder={
                          doc
                            ? `Ask a question about ${doc.name}…`
                            : "Select a document to ask questions"
                        }
                        disabled={!doc}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault()
                            ask()
                          }
                        }}
                      />
                      <button
                        aria-label="Send question"
                        disabled={!question.trim() || busy || !doc}
                      >
                        <Icon name="arrow" size={22} />
                      </button>
                    </div>
                    <p>
                      <Icon name="spark" size={13} /> Answers are strictly grounded in your selected notes using dense semantic retrieval.
                    </p>
                  </form>
                </section>
              ))}
              {page === "AI Summarizer" && (
                docs.length === 0 ? (
                  <section className="panel empty-state" style={{ padding: "56px 24px", textAlign: "center", minHeight: "380px", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
                    <span className="icon-box blue" style={{ width: "56px", height: "56px", margin: "0 auto 16px", borderRadius: "14px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <Icon name="spark" size={30} />
                    </span>
                    <h2 style={{ fontSize: "19px", fontWeight: 700, margin: "0 0 8px", color: "#f8fafc" }}>
                      Upload notes to generate summaries
                    </h2>
                    <p style={{ fontSize: "13.5px", color: "#94a3b8", margin: "0 auto 22px", maxWidth: "440px", lineHeight: "1.6" }}>
                      Choose your study material to generate concise overviews, detailed breakdowns, bullet key points, or topic-wise summaries without hallucinations.
                    </p>
                    <Button
                      onClick={() => {
                        setSelectedFile(null)
                        setUploadName("")
                        setUploadError("")
                        setModal("upload")
                      }}
                    >
                      <Icon name="upload" size={17} /> Upload Notes
                    </Button>
                  </section>
                ) : (
                <div className="tool-layout">
                  <section className="panel configuration-panel">
                    <span className="icon-box blue">
                      <Icon name="spark" />
                    </span>
                    <h2>Your notes, distilled.</h2>
                    <p>Choose your material and how much detail you need.</p>
                    {documentSelect}
                    <label className="field">Summary mode</label>
                    <div
                      className="segmented"
                      style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(4, 1fr)",
                        gap: "2px",
                      }}
                    >
                      {(
                        [
                          { id: "short" as SummaryMode, label: "Short" },
                          { id: "detailed" as SummaryMode, label: "Detailed" },
                          { id: "key_points" as SummaryMode, label: "Key Points" },
                          { id: "topic_wise" as SummaryMode, label: "Topic Wise" },
                        ] as const
                      ).map((m) => (
                        <button
                          key={m.id}
                          className={summaryMode === m.id ? "selected" : ""}
                          onClick={() => {
                            setSummaryMode(m.id)
                            setSummaryResult(null)
                            setSummaryError("")
                          }}
                        >
                          {m.label}
                        </button>
                      ))}
                    </div>
                    <p className="field-hint">
                      {summaryMode === "short"
                        ? "A concise executive overview of core essentials."
                        : summaryMode === "detailed"
                          ? "Deep structured breakdown by section with definitions & mechanics."
                          : summaryMode === "key_points"
                            ? "High-yield bullet takeaways optimized for fast exam revision."
                            : "Modular structured breakdown organized by distinct topics."}
                    </p>
                    <Button
                      disabled={summaryBusy || !doc}
                      onClick={handleGenerateSummary}
                    >
                      {summaryBusy ? (
                        <span className="spinner" />
                      ) : (
                        <Icon name="spark" size={17} />
                      )}{" "}
                      {summaryBusy ? "Generating summary…" : "Generate AI Summary"}
                    </Button>
                    {!doc?.backendDocId && (
                      <p
                        style={{
                          fontSize: "9px",
                          color: "#9f8eb2",
                          marginTop: "10px",
                          lineHeight: "1.5",
                        }}
                      >
                        Note: Select an uploaded PDF with indexed chunks from the dropdown above for real AI summarization.
                      </p>
                    )}
                  </section>
                  <section className="panel output-panel">
                    {summaryBusy ? (
                      <div
                        className="loading-state"
                        style={{
                          minHeight: "360px",
                          display: "flex",
                          flexDirection: "column",
                          justifyContent: "center",
                          alignItems: "center",
                          gap: "12px",
                          textAlign: "center",
                        }}
                      >
                        <span
                          className="spinner"
                          style={{ width: "24px", height: "24px" }}
                        />
                        <strong
                          style={{
                            fontSize: "13px",
                            color: "var(--indigo)",
                          }}
                        >
                          Analyzing document sections & synthesizing grounded summary…
                        </strong>
                        <p
                          style={{
                            fontSize: "11px",
                            color: "#8d7f9d",
                            margin: 0,
                          }}
                        >
                          Strictly extracting factual propositions from{" "}
                          <strong>{doc?.name}</strong>.
                        </p>
                      </div>
                    ) : summaryError ? (
                      <div
                        style={{
                          padding: "24px",
                          background: "rgba(255, 235, 238, 0.7)",
                          border: "1px solid #ffcdd2",
                          borderRadius: "10px",
                          color: "#b71c1c",
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "8px",
                            fontWeight: 650,
                            fontSize: "13px",
                            marginBottom: "8px",
                          }}
                        >
                          <Icon name="close" size={18} />
                          <span>Summarization Notice</span>
                        </div>
                        <p
                          style={{
                            fontSize: "12px",
                            lineHeight: "1.7",
                            margin: "0 0 16px 0",
                            color: "#5d1515",
                          }}
                        >
                          {summaryError}
                        </p>
                        {summaryError.includes("OCR") && (
                          <div
                            style={{
                              fontSize: "11px",
                              color: "#491212",
                              background: "white",
                              padding: "10px 14px",
                              borderRadius: "6px",
                              marginBottom: "16px",
                              border: "1px solid #ffcdd2",
                            }}
                          >
                            <strong>Optical Character Recognition (OCR) Notice:</strong> This document contains scanned raster images without a readable text layer. OCR text extraction is planned for a future phase.
                          </div>
                        )}
                        <Button
                          variant="secondary"
                          onClick={handleGenerateSummary}
                        >
                          Try Again
                        </Button>
                      </div>
                    ) : summaryResult ? (
                      <>
                        <div
                          className="section-heading"
                          style={{ flexWrap: "wrap", gap: "10px" }}
                        >
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: "8px",
                              flexWrap: "wrap",
                            }}
                          >
                            <span
                              className="sample-tag"
                              style={{
                                background: "rgba(103, 58, 183, 0.12)",
                                color: "var(--indigo)",
                                fontWeight: 600,
                              }}
                            >
                              {summaryResult.summary_mode.replace("_", " ").toUpperCase()} MODE
                            </span>
                            <span
                              style={{
                                fontSize: "10px",
                                color: "#77658c",
                                background: "#f5f1fa",
                                padding: "3px 8px",
                                borderRadius: "4px",
                                border: "1px solid #eae2f3",
                              }}
                            >
                              {summaryResult.generation_provider === "extractive_grounded_summarizer"
                                ? "Offline Grounded Synthesizer"
                                : summaryResult.generation_provider}
                            </span>
                            {summaryResult.cited_pages && summaryResult.cited_pages.length > 0 && (
                              <span
                                style={{
                                  fontSize: "10px",
                                  color: "#695383",
                                  background: "#f0ebf8",
                                  padding: "3px 8px",
                                  borderRadius: "4px",
                                  border: "1px solid #e5daf0",
                                }}
                              >
                                Pages Cited: {summaryResult.cited_pages.join(", ")}
                              </span>
                            )}
                          </div>
                          <button
                            className="text-button"
                            onClick={() => {
                              navigator.clipboard
                                ?.writeText(summaryResult.summary)
                                .then(() => notify("Summary copied to clipboard."))
                                .catch(() =>
                                  notify("Copy is unavailable in this browser.")
                                )
                            }}
                          >
                            <Icon name="cards" size={14} /> Copy summary
                          </button>
                        </div>

                        <h2>{summaryResult.document_name}</h2>

                        {summaryResult.warning && (
                          <div
                            style={{
                              padding: "9px 13px",
                              background: "rgba(244, 240, 252, 0.8)",
                              border: "1px solid #e4d8f3",
                              borderRadius: "7px",
                              fontSize: "10.5px",
                              color: "#6e598f",
                              margin: "10px 0 18px",
                              display: "flex",
                              alignItems: "center",
                              gap: "8px",
                            }}
                          >
                            <Icon name="spark" size={15} />
                            <span>{summaryResult.warning}</span>
                          </div>
                        )}

                        {/* Structured Topic Sections if topic_wise */}
                        {summaryResult.topic_sections && summaryResult.topic_sections.length > 0 ? (
                          <div
                            style={{
                              display: "flex",
                              flexDirection: "column",
                              gap: "16px",
                              marginTop: "14px",
                            }}
                          >
                            {summaryResult.topic_sections.map((topic, tIdx) => (
                              <div
                                key={tIdx}
                                style={{
                                  padding: "16px 18px",
                                  background: "rgba(253, 252, 255, 0.75)",
                                  border: "1px solid #ebe4f3",
                                  borderRadius: "8px",
                                  boxShadow: "0 2px 6px rgba(110, 80, 160, 0.03)",
                                }}
                              >
                                <div
                                  style={{
                                    display: "flex",
                                    justifyContent: "space-between",
                                    alignItems: "center",
                                    marginBottom: "8px",
                                  }}
                                >
                                  <h3
                                    style={{
                                      margin: 0,
                                      fontSize: "13.5px",
                                      fontWeight: 700,
                                      color: "var(--indigo)",
                                    }}
                                  >
                                    {topic.title}
                                  </h3>
                                  <span
                                    style={{
                                      fontSize: "9.5px",
                                      color: "#8a75a7",
                                      background: "#f3eef9",
                                      padding: "2px 7px",
                                      borderRadius: "4px",
                                    }}
                                  >
                                    Page(s) {topic.page_numbers.join(", ")}
                                  </span>
                                </div>
                                <p
                                  style={{
                                    fontSize: "12px",
                                    lineHeight: "1.75",
                                    color: "#4f4559",
                                    margin: "0 0 10px 0",
                                  }}
                                >
                                  {topic.content}
                                </p>
                                {topic.key_points && topic.key_points.length > 0 && (
                                  <ul
                                    style={{
                                      margin: "6px 0 0 0",
                                      paddingLeft: "18px",
                                      fontSize: "11.5px",
                                      color: "#5d4e70",
                                      lineHeight: "1.7",
                                    }}
                                  >
                                    {topic.key_points.map((pt, ptIdx) => (
                                      <li
                                        key={ptIdx}
                                        style={{ marginBottom: "4px" }}
                                      >
                                        {pt}
                                      </li>
                                    ))}
                                  </ul>
                                )}
                              </div>
                            ))}
                          </div>
                        ) : (
                          /* Render markdown for short, detailed, and key_points */
                          <div style={{ marginTop: "12px" }}>
                            <FormattedAnswer text={summaryResult.summary} />
                          </div>
                        )}

                        {/* Key Points standalone card list if key_points returned and not topic_wise */}
                        {!summaryResult.topic_sections && summaryResult.key_points && summaryResult.key_points.length > 0 && summaryMode !== "key_points" && (
                          <div
                            style={{
                              marginTop: "20px",
                              padding: "16px",
                              background: "rgba(247, 244, 253, 0.65)",
                              border: "1px solid #eee5f7",
                              borderRadius: "8px",
                            }}
                          >
                            <h4
                              style={{
                                margin: "0 0 10px 0",
                                fontSize: "12px",
                                fontWeight: 700,
                                color: "var(--indigo)",
                              }}
                            >
                              Core Bullet Takeaways
                            </h4>
                            <ul
                              style={{
                                margin: 0,
                                paddingLeft: "18px",
                                fontSize: "11.5px",
                                color: "#544665",
                                lineHeight: "1.7",
                              }}
                            >
                              {summaryResult.key_points.map((kp, kpIdx) => (
                                <li
                                  key={kpIdx}
                                  style={{ marginBottom: "5px" }}
                                >
                                  {kp}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        <div className="summary-takeaway">
                          <Icon name="spark" size={20} />
                          <p>
                            Summary strictly grounded in <strong>{summaryResult.document_name}</strong> across {summaryResult.cited_pages.length} source page(s). All facts and terms originate directly from your notes.
                          </p>
                        </div>
                      </>
                    ) : (
                      <div className="empty-state">
                        <span className="large-spark">
                          <Icon name="book" size={30} />
                        </span>
                        <h2>The big ideas belong here.</h2>
                        <p>
                          Select an uploaded PDF and choose a summary mode
                          <br />
                          to generate a grounded, hallucination-free breakdown.
                        </p>
                      </div>
                    )}
                  </section>
                </div>
              ))}
              {page === "Revision Questions" && (
                docs.length === 0 ? (
                  <section className="panel empty-state" style={{ padding: "56px 24px", textAlign: "center", minHeight: "380px", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
                    <span className="icon-box green" style={{ width: "56px", height: "56px", margin: "0 auto 16px", borderRadius: "14px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <Icon name="cards" size={30} />
                    </span>
                    <h2 style={{ fontSize: "19px", fontWeight: 700, margin: "0 0 8px", color: "#f8fafc" }}>
                      Upload notes to generate revision quizzes
                    </h2>
                    <p style={{ fontSize: "13.5px", color: "#94a3b8", margin: "0 auto 22px", maxWidth: "440px", lineHeight: "1.6" }}>
                      Automatically generate MCQs, True/False, short-answer, and long-answer questions grounded directly in your uploaded notes.
                    </p>
                    <Button
                      onClick={() => {
                        setSelectedFile(null)
                        setUploadName("")
                        setUploadError("")
                        setModal("upload")
                      }}
                    >
                      <Icon name="upload" size={17} /> Upload Notes
                    </Button>
                  </section>
                ) : (
                <>
                  {/* ── Configuration panel ─────────────────────────────── */}
                  <section className="panel revision-config">
                    <div>
                      {documentSelect}
                      {/* Question type chips */}
                      <label className="field">
                        Question types
                        <div style={{ display: "flex", gap: "7px", flexWrap: "wrap", marginTop: "2px" }}>
                          {(
                            [
                              { id: "mcq" as QuestionType, label: "Multiple Choice" },
                              { id: "true_false" as QuestionType, label: "True / False" },
                              { id: "short_answer" as QuestionType, label: "Short Answer" },
                              { id: "long_answer" as QuestionType, label: "Long Answer" },
                            ] as const
                          ).map((qt) => {
                            const active = selectedQTypes.includes(qt.id)
                            return (
                              <button
                                key={qt.id}
                                type="button"
                                onClick={() => {
                                  setSelectedQTypes((prev) =>
                                    active
                                      ? prev.length > 1
                                        ? prev.filter((t) => t !== qt.id)
                                        : prev
                                      : [...prev, qt.id]
                                  )
                                  setGenerated(false)
                                  setQuizQuestions([])
                                  setQuizSubmitted(false)
                                  setQuizEvaluation(null)
                                }}
                                style={{
                                  fontSize: "10px",
                                  padding: "6px 11px",
                                  borderRadius: "6px",
                                  border: `1px solid ${active ? "var(--indigo)" : "#e0d9ec"}`,
                                  background: active ? "var(--indigo)" : "#faf7fd",
                                  color: active ? "#fff" : "#8a7aa3",
                                  fontWeight: active ? 600 : 400,
                                  cursor: "pointer",
                                  transition: "all .16s",
                                }}
                              >
                                {qt.label}
                              </button>
                            )
                          })}
                        </div>
                      </label>
                      <label className="field">
                        Questions
                        <select
                          value={count}
                          onChange={(e) => {
                            setCount(e.target.value)
                            setGenerated(false)
                          }}
                        >
                          <option>5</option>
                          <option>10</option>
                          <option>15</option>
                        </select>
                      </label>
                      <label className="field">
                        Difficulty
                        <select
                          value={difficulty}
                          onChange={(e) => {
                            setDifficulty(e.target.value)
                            setGenerated(false)
                          }}
                        >
                          <option>Easy</option>
                          <option>Medium</option>
                          <option>Challenging</option>
                        </select>
                      </label>
                    </div>
                    <Button
                      disabled={questionsBusy || !doc}
                      onClick={handleGenerateQuestions}
                    >
                      {questionsBusy ? (
                        <span className="spinner" />
                      ) : (
                        <Icon name="spark" size={17} />
                      )}{" "}
                      {questionsBusy ? "Generating quiz…" : "Generate Quiz"}
                    </Button>
                    {questionsError && (
                      <p
                        style={{
                          fontSize: "10px",
                          color: "#b46f7a",
                          marginTop: "10px",
                          lineHeight: "1.6",
                        }}
                      >
                        {questionsError}
                      </p>
                    )}
                    {!doc?.backendDocId && (
                      <p
                        style={{
                          fontSize: "9px",
                          color: "#9f8eb2",
                          marginTop: "10px",
                          lineHeight: "1.5",
                        }}
                      >
                        Select an uploaded PDF with indexed chunks for real question generation.
                      </p>
                    )}
                  </section>

                  {/* ── Score card (after submission) ──────────────────── */}
                  {quizSubmitted && quizEvaluation && (
                    <section
                      className="panel"
                      style={{ padding: "24px", marginBottom: "0" }}
                    >
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "22px",
                          flexWrap: "wrap",
                        }}
                      >
                        {/* Big score circle */}
                        <div
                          style={{
                            width: "88px",
                            height: "88px",
                            borderRadius: "50%",
                            background:
                              quizEvaluation.percentage >= 70
                                ? "linear-gradient(135deg,#6fba8c,#4fa87a)"
                                : quizEvaluation.percentage >= 40
                                ? "linear-gradient(135deg,#e8a254,#d68538)"
                                : "linear-gradient(135deg,#c97070,#b85555)",
                            display: "grid",
                            placeItems: "center",
                            color: "white",
                            flexShrink: 0,
                            boxShadow: "0 4px 14px rgba(0,0,0,.12)",
                          }}
                        >
                          <span
                            style={{
                              fontFamily: "'Inter',sans-serif",
                              fontSize: "22px",
                              fontWeight: 700,
                              lineHeight: 1,
                            }}
                          >
                            {Math.round(quizEvaluation.percentage)}%
                          </span>
                        </div>
                        <div style={{ flex: 1 }}>
                          <div
                            style={{
                              fontFamily: "'Inter',sans-serif",
                              fontSize: "18px",
                              fontWeight: 700,
                              letterSpacing: "-.4px",
                              color: "#3d3348",
                              marginBottom: "4px",
                            }}
                          >
                            {quizEvaluation.percentage >= 80
                              ? "Excellent work! 🎉"
                              : quizEvaluation.percentage >= 60
                              ? "Good effort — keep going!"
                              : "Needs more revision — you've got this!"}
                          </div>
                          <p style={{ fontSize: "11px", color: "#9880aa", margin: 0 }}>
                            {quizEvaluation.correct_count} / {quizEvaluation.total_questions} correct
                            &nbsp;·&nbsp; Score: {quizEvaluation.total_score.toFixed(1)} / {quizEvaluation.max_score.toFixed(1)}
                            &nbsp;·&nbsp; Grading: {quizEvaluation.grading_method.replace(/_/g, " ")}
                          </p>
                        </div>
                        <div style={{ display: "flex", gap: "10px", flexShrink: 0 }}>
                          <button
                            className="button secondary"
                            style={{ fontSize: "10px" }}
                            onClick={() => {
                              setUserAnswers({})
                              setQuizSubmitted(false)
                              setQuizEvaluation(null)
                            }}
                          >
                            <Icon name="clock" size={15} /> Retake
                          </button>
                          <button
                            className="button"
                            style={{ fontSize: "10px" }}
                            onClick={() => {
                              setQuizQuestions([])
                              setUserAnswers({})
                              setQuizSubmitted(false)
                              setQuizEvaluation(null)
                              setQuizSessionId("")
                              setGenerated(false)
                            }}
                          >
                            <Icon name="spark" size={15} /> New Quiz
                          </button>
                        </div>
                      </div>
                    </section>
                  )}

                  {/* ── Questions list ──────────────────────────────────── */}
                  {quizQuestions.length > 0 ? (
                    <div className="questions-list">
                      <div className="questions-caption">
                        <span className="sample-tag">
                          <Icon name="spark" size={12} />
                          &nbsp;Grounded questions · offline NLP
                        </span>
                        <span>
                          {difficulty} · {quizQuestions.length} questions · {doc?.name}
                        </span>
                      </div>

                      {quizQuestions.map((q, idx) => {
                        const evalResult = quizEvaluation?.evaluations.find(
                          (e) => e.question_id === q.question_id
                        )
                        const isCorrect = evalResult ? evalResult.score >= 0.5 : null
                        const typeLabel =
                          q.question_type === "mcq"
                            ? "MULTIPLE CHOICE"
                            : q.question_type === "true_false"
                            ? "TRUE / FALSE"
                            : q.question_type === "short_answer"
                            ? "SHORT ANSWER"
                            : "LONG ANSWER"

                        return (
                          <section
                            className="panel question-card"
                            key={q.question_id}
                            style={{
                              borderLeft: quizSubmitted
                                ? `3px solid ${
                                    isCorrect === true
                                      ? "#6fba8c"
                                      : isCorrect === false
                                      ? "#c97070"
                                      : "#e6e0ed"
                                  }`
                                : undefined,
                            }}
                          >
                            {/* Header */}
                            <div className="question-card-header">
                              <span>
                                QUESTION {idx + 1} <i>·</i> {typeLabel}
                                &nbsp;
                                <span
                                  style={{
                                    fontSize: "7px",
                                    padding: "2px 5px",
                                    borderRadius: "3px",
                                    background:
                                      q.difficulty === "easy"
                                        ? "#edf5ef"
                                        : q.difficulty === "hard"
                                        ? "#f9eaea"
                                        : "#f2edfb",
                                    color:
                                      q.difficulty === "easy"
                                        ? "#5a9571"
                                        : q.difficulty === "hard"
                                        ? "#b06060"
                                        : "#7860a8",
                                    fontWeight: 600,
                                    textTransform: "uppercase",
                                  }}
                                >
                                  {q.difficulty}
                                </span>
                              </span>
                              <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                {q.source_page_numbers.length > 0 && (
                                  <span
                                    style={{
                                      fontSize: "8px",
                                      color: "#a08cbb",
                                      background: "#f4eef9",
                                      padding: "3px 6px",
                                      borderRadius: "4px",
                                      border: "1px solid #ebe0f3",
                                    }}
                                  >
                                    Page {q.source_page_numbers.join(", ")}
                                  </span>
                                )}
                                {quizSubmitted && evalResult && (
                                  <span
                                    style={{
                                      fontSize: "9px",
                                      fontWeight: 600,
                                      color: isCorrect ? "#5a9571" : "#b06060",
                                      display: "flex",
                                      alignItems: "center",
                                      gap: "4px",
                                    }}
                                  >
                                    <Icon name={isCorrect ? "check" : "close"} size={13} />
                                    {evalResult.score === 1.0
                                      ? "Full credit"
                                      : evalResult.score === 0.5
                                      ? "Partial credit"
                                      : "No credit"}
                                  </span>
                                )}
                              </span>
                            </div>

                            {/* Question prompt */}
                            <h3>{q.question_text}</h3>

                            {/* MCQ options */}
                            {(q.question_type === "mcq" || q.question_type === "true_false") &&
                              q.options && (
                                <div className="answer-options">
                                  {q.options.map((opt, j) => {
                                    const selected = userAnswers[q.question_id] === opt
                                    const isCorrectOpt =
                                      quizSubmitted &&
                                      evalResult &&
                                      opt.toLowerCase() === evalResult.correct_answer.toLowerCase()
                                    const isWrongSelected =
                                      quizSubmitted &&
                                      evalResult &&
                                      selected &&
                                      opt.toLowerCase() !== evalResult.correct_answer.toLowerCase()
                                    return (
                                      <div
                                        key={opt}
                                        role="button"
                                        tabIndex={quizSubmitted ? -1 : 0}
                                        onClick={() => {
                                          if (quizSubmitted) return
                                          setUserAnswers((prev) => ({
                                            ...prev,
                                            [q.question_id]: opt,
                                          }))
                                        }}
                                        onKeyDown={(e) => {
                                          if (e.key === "Enter" || e.key === " ") {
                                            if (quizSubmitted) return
                                            setUserAnswers((prev) => ({
                                              ...prev,
                                              [q.question_id]: opt,
                                            }))
                                          }
                                        }}
                                        style={{
                                          cursor: quizSubmitted ? "default" : "pointer",
                                          border: `1px solid ${
                                            isCorrectOpt
                                              ? "#6fba8c"
                                              : isWrongSelected
                                              ? "#c97070"
                                              : selected
                                              ? "var(--indigo)"
                                              : "#eee7f3"
                                          }`,
                                          background: isCorrectOpt
                                            ? "rgba(111,186,140,.1)"
                                            : isWrongSelected
                                            ? "rgba(201,112,112,.08)"
                                            : selected
                                            ? "rgba(115,97,190,.07)"
                                            : undefined,
                                          transition: "border .15s, background .15s",
                                        }}
                                      >
                                        <span
                                          style={{
                                            background: isCorrectOpt
                                              ? "#6fba8c"
                                              : isWrongSelected
                                              ? "#c97070"
                                              : selected
                                              ? "var(--indigo)"
                                              : undefined,
                                            color:
                                              isCorrectOpt || isWrongSelected || selected
                                                ? "#fff"
                                                : undefined,
                                          }}
                                        >
                                          {String.fromCharCode(65 + j)}
                                        </span>
                                        {opt}
                                        {isCorrectOpt && (
                                          <Icon
                                            name="check"
                                            size={14}
                                            className=""
                                          />
                                        )}
                                      </div>
                                    )
                                  })}
                                </div>
                              )}

                            {/* Short Answer textarea */}
                            {q.question_type === "short_answer" && (
                              <textarea
                                disabled={quizSubmitted}
                                placeholder="Write your answer here…"
                                value={userAnswers[q.question_id] ?? ""}
                                onChange={(e) =>
                                  setUserAnswers((prev) => ({
                                    ...prev,
                                    [q.question_id]: e.target.value,
                                  }))
                                }
                                style={{
                                  width: "100%",
                                  minHeight: "70px",
                                  border: "1px solid #e2d9ee",
                                  borderRadius: "7px",
                                  padding: "10px 12px",
                                  fontSize: "12px",
                                  color: "#7a668f",
                                  background: quizSubmitted ? "#faf8fc" : "#fdfcfe",
                                  resize: "vertical",
                                  outline: "none",
                                  lineHeight: "1.7",
                                  marginTop: "4px",
                                }}
                              />
                            )}

                            {/* Long Answer textarea */}
                            {q.question_type === "long_answer" && (
                              <textarea
                                disabled={quizSubmitted}
                                placeholder="Write your detailed answer here…"
                                value={userAnswers[q.question_id] ?? ""}
                                onChange={(e) =>
                                  setUserAnswers((prev) => ({
                                    ...prev,
                                    [q.question_id]: e.target.value,
                                  }))
                                }
                                style={{
                                  width: "100%",
                                  minHeight: "110px",
                                  border: "1px solid #e2d9ee",
                                  borderRadius: "7px",
                                  padding: "10px 12px",
                                  fontSize: "12px",
                                  color: "#7a668f",
                                  background: quizSubmitted ? "#faf8fc" : "#fdfcfe",
                                  resize: "vertical",
                                  outline: "none",
                                  lineHeight: "1.7",
                                  marginTop: "4px",
                                }}
                              />
                            )}

                            {/* Post-submission feedback */}
                            {quizSubmitted && evalResult && (
                              <div
                                className="expanded-answer"
                                style={{
                                  marginTop: "14px",
                                  background: isCorrect
                                    ? "rgba(111,186,140,.08)"
                                    : "rgba(201,112,112,.06)",
                                  border: `1px solid ${isCorrect ? "#c5e4cf" : "#e8c0c0"}`,
                                }}
                              >
                                <strong style={{ color: isCorrect ? "#4d8f65" : "#a85555" }}>
                                  {isCorrect ? "✓ Correct" : "✗ Review needed"}
                                </strong>
                                <p style={{ marginTop: "6px", color: "#7a668f" }}>
                                  <strong>Your answer:</strong>{" "}
                                  {evalResult.user_answer || <em>No answer provided</em>}
                                </p>
                                <p style={{ color: "#7a668f" }}>
                                  <strong>Correct answer:</strong> {evalResult.correct_answer}
                                </p>
                                <p style={{ color: "#856fa5", marginTop: "6px" }}>
                                  {evalResult.feedback}
                                </p>
                                <p
                                  style={{
                                    fontSize: "10px",
                                    color: "#9a84b0",
                                    borderTop: "1px solid #ede5f3",
                                    paddingTop: "8px",
                                    marginTop: "10px",
                                  }}
                                >
                                  <Icon name="file" size={12} />
                                  &nbsp;{evalResult.explanation}
                                </p>
                              </div>
                            )}
                          </section>
                        )
                      })}

                      {/* Submit / progress bar */}
                      {!quizSubmitted && (
                        <div
                          style={{
                            padding: "20px",
                            background: "#fff",
                            border: "1px solid var(--border)",
                            borderRadius: "9px",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            gap: "16px",
                            flexWrap: "wrap",
                          }}
                        >
                          <span style={{ fontSize: "10px", color: "#a294ae" }}>
                            {Object.keys(userAnswers).length} / {quizQuestions.length} answered
                          </span>
                          <Button
                            disabled={questionsBusy}
                            onClick={handleSubmitQuiz}
                          >
                            {questionsBusy ? (
                              <span className="spinner" />
                            ) : (
                              <Icon name="check" size={16} />
                            )}{" "}
                            {questionsBusy ? "Evaluating…" : "Submit Quiz"}
                          </Button>
                        </div>
                      )}
                    </div>
                  ) : generated && !questionsBusy ? (
                    <div className="empty-state panel revision-empty">
                      <div className="large-spark">
                        <Icon name="cards" size={30} />
                      </div>
                      <h2>No questions could be extracted.</h2>
                      <p>
                        The document may not have enough structured text.
                        <br />
                        Try a different PDF with clear definitions and headings.
                      </p>
                    </div>
                  ) : !questionsBusy ? (
                    <div className="empty-state panel revision-empty">
                      <div className="large-spark">
                        <Icon name="cards" size={30} />
                      </div>
                      <h2>A little practice. A lot more confidence.</h2>
                      <p>
                        Upload a PDF, choose your question types and difficulty,
                        <br />
                        then hit <strong>Generate Quiz</strong> for real grounded questions.
                      </p>
                    </div>
                  ) : (
                    <div className="loading-state" style={{ justifyContent: "center", padding: "48px" }}>
                      <span className="spinner" />
                      Extracting academic concepts and building your quiz…
                    </div>
                  )}
                </>
              ))}
            </>
          )}
        </main>
      </div>
      {toast && (
        <div className="toast" role="status">
          <span className="toast-icon">
            <Icon name="check" size={17} />
          </span>
          {toast}
          <button
            className="icon-button"
            aria-label="Dismiss notification"
            onClick={() => setToast("")}
          >
            <Icon name="close" size={16} />
          </button>
        </div>
      )}
      {modal && (
        <div
          className="modal-backdrop"
          onClick={() => !uploading && setModal(null)}
        >
          <section
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="modal-title"
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => {
              if (e.key === "Escape" && !uploading) setModal(null)
              if (e.key === "Tab") {
                const controls = Array.from(
                  e.currentTarget.querySelectorAll<HTMLElement>(
                    'button, input, select, [tabindex="0"]',
                  ),
                ).filter((el) => !el.hasAttribute("disabled"))
                const first = controls[0]
                const last = controls[controls.length - 1]
                if (e.shiftKey && document.activeElement === first) {
                  e.preventDefault()
                  last?.focus()
                } else if (!e.shiftKey && document.activeElement === last) {
                  e.preventDefault()
                  first?.focus()
                }
              }
            }}
          >
            <button
              className="icon-button modal-close"
              aria-label="Close dialog"
              disabled={uploading}
              autoFocus
              onClick={() => {
                if (!uploading) {
                  setModal(null)
                  setSelectedFile(null)
                  setUploadName("")
                  setUploadError("")
                }
              }}
            >
              <Icon name="close" />
            </button>
            <span className="icon-box purple">
              <Icon
                name={
                  modal === "upload"
                    ? "upload"
                    : modal === "help"
                      ? "help"
                      : modal === "settings"
                        ? "settings"
                        : "file"
                }
              />
            </span>
            <h2 id="modal-title">
              {modal === "upload"
                ? "A new chapter starts here."
                : modal === "rename"
                  ? "Rename document"
                  : modal === "delete"
                    ? "Remove this document?"
                    : modal === "help"
                      ? "A little guidance, whenever you need it."
                      : "Your workspace"}
            </h2>
            {modal === "upload" && (
              <>
                <p>
                  Upload your study notes in any format: PDF (.pdf), PowerPoint (.pptx),
                  Word (.docx), Text notes (.txt, .md), or Photos/Images (.png, .jpg) with automated OCR.
                  Content is chunked, embedded, and stored persistently in SQLite.
                </p>
                <input
                  type="file"
                  accept=".pdf,.pptx,.ppt,.docx,.doc,.txt,.md,.markdown,.png,.jpg,.jpeg,.webp,.bmp"
                  ref={fileInput}
                  hidden
                  onChange={(e) => selectFile(e.target.files?.[0])}
                />
                <button
                  className="dropzone modal-dropzone"
                  disabled={uploading}
                  onClick={() => fileInput.current?.click()}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault()
                    selectFile(e.dataTransfer.files[0])
                  }}
                >
                  <Icon name={uploadName ? "file" : "upload"} size={25} />
                  <strong>
                    {uploadName || "Choose any note file or photo, or drop it here"}
                  </strong>
                  <small>
                    {selectedFile
                      ? `${(selectedFile.size / 1024).toFixed(1)} KB · Ready to index`
                      : "PDF, PPT, DOCX, TXT, or Photos (OCR) · Max 50 MB"}
                  </small>
                </button>
                {uploadError && (
                  <p className="error-message" role="alert">
                    {uploadError}
                  </p>
                )}
                <div className="modal-actions">
                  <Button
                    variant="secondary"
                    disabled={uploading}
                    onClick={() => {
                      setModal(null)
                      setSelectedFile(null)
                      setUploadName("")
                      setUploadError("")
                    }}
                  >
                    Cancel
                  </Button>
                  <Button
                    disabled={!selectedFile || uploading}
                    onClick={handleUpload}
                  >
                    {uploading ? (
                      <>
                        <span className="spinner" /> Indexing notes…
                      </>
                    ) : (
                      <>
                        <Icon name="upload" size={16} /> Upload & Index Notes
                      </>
                    )}
                  </Button>
                </div>
              </>
            )}
            {modal === "rename" && (
              <>
                <p>Give your notes a name that’s easy to find.</p>
                <label className="field">
                  Document name
                  <GlassInput
                    value={rename}
                    onChange={(e) => setRename(e.target.value)}
                  />
                </label>
                <div className="modal-actions">
                  <Button variant="secondary" onClick={() => setModal(null)}>
                    Cancel
                  </Button>
                  <Button
                    disabled={!rename.trim()}
                    onClick={() => {
                      setDocs((d) =>
                        d.map((item) =>
                          item.id === activeDoc
                            ? { ...item, name: rename.trim() }
                            : item,
                        ),
                      )
                      setModal(null)
                      notify("Document renamed for this session.")
                    }}
                  >
                    Save name
                  </Button>
                </div>
              </>
            )}
            {modal === "delete" && (
              <>
                <p>
                  “{doc?.name}” will be permanently removed from your SQLite database
                  and vector store.
                </p>
                <div className="modal-actions">
                  <Button variant="secondary" onClick={() => setModal(null)}>
                    Keep document
                  </Button>
                  <Button
                    variant="danger"
                    onClick={async () => {
                      const docToDelete = doc
                      if (docToDelete?.backendDocId) {
                        try {
                          await deleteDocument(docToDelete.backendDocId)
                          notify(`"${docToDelete.name}" permanently deleted from database.`)
                        } catch {
                          notify("Could not delete from backend database.")
                        }
                      }
                      setDocs((d) => d.filter((item) => item.id !== activeDoc))
                      setMessages([])
                      setSummaryResult(null)
                      setSummaryError("")
                      setGenerated(false)
                      setModal(null)
                    }}
                  >
                    Delete document
                  </Button>
                </div>
              </>
            )}
            {modal === "help" && (
              <>
                <p>
                  StudySphere AI is an intelligent workspace that extracts text from any note format
                  (PDF, PowerPoint, Word, text, and photos via RapidOCR) and allows grounded Q&A,
                  summarization, and revision quiz generation.
                </p>
                <div className="help-note">
                  <strong>Persistent Local SQLite & Vector Search</strong>
                  <p>
                    Uploaded documents are embedded with sentence-transformers and saved directly into
                    SQLite. Everything is processed offline/locally without requiring cloud API keys.
                  </p>
                </div>
                <Button onClick={() => setModal(null)}>
                  Got it, let’s learn
                </Button>
              </>
            )}
            {modal === "settings" && (
              <>
                <p>You’re exploring the StudySphere AI demo workspace.</p>
                <div className="settings-profile">
                  <span className="avatar">AJ</span>
                  <div>
                    <strong>Alex Johnson</strong>
                    <p>Sample student profile · No authentication</p>
                  </div>
                </div>
                <div className="help-note">
                  <strong>Local demo preferences</strong>
                  <p>
                    Documents, saved questions, and conversations reset when the
                    page reloads. No personal information is stored.
                  </p>
                </div>
                <Button onClick={() => setModal(null)}>
                  Back to my workspace
                </Button>
              </>
            )}
          </section>
        </div>
      )}
    </div>
  )
}
