import { useRef, useState, type ReactNode } from "react"
import GradientBackground from "./components/GradientBackground"
import GlassNavbar from "./components/GlassNavbar"
import Hero from "./components/Hero"
import Features from "./components/Features"
import Pricing from "./components/Pricing"
import Footer from "./components/Footer"
import GlassInput from "./components/GlassInput"
import { glassTheme } from "./components/glassTheme"

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
const navigation: { name: string icon: IconName }[] = [
  { name: "Dashboard", icon: "grid" },
  { name: "My Documents", icon: "file" },
  { name: "Ask Your Notes", icon: "chat" },
  { name: "AI Summarizer", icon: "spark" },
  { name: "Revision Questions", icon: "cards" },
]
type Doc = {
  id: number
  name: string
  subject: string
  size: string
  pages: number
  date: string
  color: string
}
const initialDocs: Doc[] = [
  {
    id: 1,
    name: "DBMS – Unit 3: Normalization",
    subject: "Database Management",
    size: "2.4 MB",
    pages: 24,
    date: "Today",
    color: "purple",
  },
  {
    id: 2,
    name: "Data Structures & Algorithms",
    subject: "Data Structures",
    size: "3.8 MB",
    pages: 42,
    date: "Yesterday",
    color: "blue",
  },
  {
    id: 3,
    name: "Introduction to Artificial Intelligence",
    subject: "Artificial Intelligence",
    size: "1.6 MB",
    pages: 18,
    date: "Oct 22, 2024",
    color: "peach",
  },
  {
    id: 4,
    name: "Operating Systems – Process Management",
    subject: "Operating Systems",
    size: "2.1 MB",
    pages: 32,
    date: "Oct 20, 2024",
    color: "green",
  },
]
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
export default function App() {
  const [page, setPage] = useState("Dashboard")
  const [mobileOpen, setMobileOpen] = useState(false)
  const [docs, setDocs] = useState(initialDocs)
  const [modal, setModal] =
    useState<"upload" | "rename" | "delete" | "help" | "settings" | null>(null)
  const [activeDoc, setActiveDoc] = useState(1)
  const [rename, setRename] = useState("")
  const [search, setSearch] = useState("")
  const [subject, setSubject] = useState("All subjects")
  const [toast, setToast] = useState("")
  const [uploadName, setUploadName] = useState("")
  const [uploadError, setUploadError] = useState("")
  const [question, setQuestion] = useState("")
  const [messages, setMessages] = useState<{
    question: string
    answer: string
  }[]>([])
  const [busy, setBusy] = useState(false)
  const [summary, setSummary] = useState(false)
  const [length, setLength] = useState("Medium")
  const [tab, setTab] = useState("Multiple Choice")
  const [count, setCount] = useState("5")
  const [difficulty, setDifficulty] = useState("Medium")
  const [generated, setGenerated] = useState(false)
  const [expanded, setExpanded] = useState<number[]>([])
  const [saved, setSaved] = useState<number[]>([])
  const [savedOnly, setSavedOnly] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)
  const doc = docs.find((d) => d.id === activeDoc) || docs[0]
  const notify = (message: string) => {
    setToast(message)
    window.setTimeout(() => setToast(""), 4500)
  }
  const go = (destination: string) => {
    setPage(destination)
    setMobileOpen(false)
  }
  const simulate = (done: () => void) => {
    setBusy(true)
    window.setTimeout(() => {
      done()
      setBusy(false)
    }, 1100)
  }
  const ask = (text = question) => {
    if (!text.trim() || busy || !doc) return
    const q = text.trim()
    setQuestion("")
    simulate(() =>
      setMessages((m) => [
        ...m,
        {
          question: q,
          answer: `Here’s an example of how StudySphere could explain a concept from ${doc.name}. Normalization organizes data into related tables to reduce duplication and improve consistency. Think of a student database: instead of repeating a department’s details for every student, store departments in a separate table and connect them with a key. This makes updates simpler and helps prevent inconsistent records.`,
        },
      ]),
    )
  }
  const selectFile = (file?: File) => {
    setUploadError("")
    if (!file) return
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Please choose a PDF file.")
      return
    }
    if (file.size > 20 * 1024 * 1024) {
      setUploadError("Please choose a PDF smaller than 20 MB.")
      return
    }
    setUploadName(file.name)
  }
  const documentSelect = (
    <label className="field">
      Study material
      <select
        value={doc?.id || ""}
        onChange={(e) => {
          setActiveDoc(Number(e.target.value))
          setSummary(false)
          setMessages([])
          setGenerated(false)
          setSaved([])
        }}
      >
        {docs.length ? (
          docs.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}
            </option>
          ))
        ) : (
          <option value="">No documents — add a demo PDF first</option>
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
                setUploadName("")
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
                    setUploadName("")
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
                    label: "Documents uploaded",
                    value: String(docs.length).padStart(2, "0"),
                    icon: "file" as IconName,
                    color: "purple",
                    detail: "Your knowledge, all in one place",
                    extra: "+2 this week",
                  },
                  {
                    label: "Questions asked",
                    value: "28",
                    icon: "chat" as IconName,
                    color: "blue",
                    detail: "One question closer to clarity",
                    extra: "+8 this week",
                  },
                  {
                    label: "Study sessions",
                    value: "12",
                    icon: "clock" as IconName,
                    color: "peach",
                    detail: "Small steps. Real progress.",
                    extra: "Keep it going",
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
                            <small>PDF</small>
                          </span>
                          <span className="document-info">
                            <strong>{d.name}</strong>
                            <span>
                              {d.subject}
                              <i /> {d.pages} pages <i /> {d.size}
                            </span>
                          </span>
                          <span className="document-end">
                            <span className="ready-badge">
                              <span />
                              Demo ready
                            </span>
                            <small>{d.date}</small>
                          </span>
                          <Icon name="chevron" size={16} />
                        </button>
                      ))}
                      {!docs.length && (
                        <div className="empty-state">
                          <Icon name="file" />
                          <p>No documents yet. Add your first demo PDF.</p>
                          <Button onClick={() => setModal("upload")}>
                            Upload Notes
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
                    {[
                      {
                        icon: "chat" as IconName,
                        color: "purple",
                        title: "A concept, made clearer",
                        description: "Asked about normalization in DBMS",
                        time: "25 minutes ago",
                      },
                      {
                        icon: "spark" as IconName,
                        color: "blue",
                        title: "The essentials, at a glance",
                        description: "Summarized Data Structures notes",
                        time: "2 hours ago",
                      },
                      {
                        icon: "file" as IconName,
                        color: "peach",
                        title: "New knowledge added",
                        description: "Uploaded DBMS – Unit 3",
                        time: "3 hours ago",
                      },
                      {
                        icon: "cards" as IconName,
                        color: "green",
                        title: "A little revision, a lot of confidence",
                        description: "Created 10 practice questions",
                        time: "Yesterday",
                      },
                    ].map((a, i) => (
                      <div className="activity-item" key={a.title}>
                        <div className="timeline-icon">
                          <span className={`icon-box ${a.color}`}>
                            <Icon name={a.icon} size={16} />
                          </span>
                          {i < 3 && <i />}
                        </div>
                        <div>
                          <strong>{a.title}</strong>
                          <p>{a.description}</p>
                          <span>{a.time}</span>
                        </div>
                      </div>
                    ))}
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
              <Pricing
                onExplore={(plan) =>
                  notify(
                    `${plan} is a concept plan. No subscription or payment is activated.`,
                  )
                }
              />
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
                      setUploadName("")
                      setModal("upload")
                    }}
                  >
                    <Icon name="upload" size={18} />
                    Upload PDF
                  </Button>
                )}
              </div>
              <div className="demo-notice">
                <Icon name="spark" size={17} />
                <span>
                  Interactive prototype · Documents and AI content are samples.
                  No files are processed or stored.
                </span>
              </div>
              {page === "My Documents" && (
                <>
                  <button
                    className="dropzone"
                    onClick={() => {
                      setUploadName("")
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
                      Drag and drop a PDF, or <span>browse files</span>
                    </p>
                    <small>PDF files only · Up to 20 MB · Demo upload</small>
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
                      <option>PDF files</option>
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
                              <small>PDF</small>
                            </span>
                            <span className="ready-badge">
                              <span />
                              Demo ready
                            </span>
                          </div>
                          <h2>{d.name}</h2>
                          <span className={`subject-badge ${d.color}`}>
                            {d.subject}
                          </span>
                          <p>
                            {d.pages ? `${d.pages} pages · ` : ""}
                            {d.size}
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
                  {!docs.filter(
                    (d) =>
                      d.name.toLowerCase().includes(search.toLowerCase()) &&
                      (subject === "All subjects" || d.subject === subject),
                  ).length && (
                    <div className="empty-state panel">
                      <Icon name="search" size={28} />
                      <h2>No documents found</h2>
                      <p>Try a different search or add a demo PDF.</p>
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
                  )}
                </>
              )}
              {page === "Ask Your Notes" && (
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
                                <span className="sample-tag">
                                  Sample response
                                </span>
                              </strong>
                              <p>{m.answer}</p>
                              <div className="sources">
                                <span>
                                  SOURCES · ILLUSTRATIVE, NOT VERIFIED
                                </span>
                                <p>
                                  <Icon name="file" size={15} />
                                  {doc?.name} · Example: page 8
                                </p>
                                <small>
                                  This is a prewritten demo answer, not a
                                  response retrieved from your PDF.
                                </small>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                    {busy && (
                      <div className="loading-state">
                        <span className="spinner" />
                        Preparing a sample explanation…
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
                            ? "What would you like to understand about your notes?"
                            : "Add a demo document to get started"
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
                      <Icon name="help" size={13} /> Demo answers may not match
                      your document. Always check your original notes.
                    </p>
                  </form>
                </section>
              )}
              {page === "AI Summarizer" && (
                <div className="tool-layout">
                  <section className="panel configuration-panel">
                    <span className="icon-box blue">
                      <Icon name="spark" />
                    </span>
                    <h2>Your notes, distilled.</h2>
                    <p>Choose your material and how much detail you need.</p>
                    {documentSelect}
                    <label className="field">Summary length</label>
                    <div className="segmented">
                      {["Short", "Medium", "Detailed"].map((l) => (
                        <button
                          key={l}
                          className={length === l ? "selected" : ""}
                          onClick={() => {
                            setLength(l)
                            setSummary(false)
                          }}
                        >
                          {l}
                        </button>
                      ))}
                    </div>
                    <p className="field-hint">
                      {length === "Short"
                        ? "A quick overview of the essentials."
                        : length === "Medium"
                          ? "Key concepts with a little context."
                          : "A deeper breakdown with examples."}
                    </p>
                    <Button
                      disabled={busy || !doc}
                      onClick={() => simulate(() => setSummary(true))}
                    >
                      {busy ? (
                        <span className="spinner" />
                      ) : (
                        <Icon name="spark" size={17} />
                      )}{" "}
                      {busy ? "Preparing sample…" : "Generate sample summary"}
                    </Button>
                  </section>
                  <section className="panel output-panel">
                    {summary ? (
                      <>
                        <div className="section-heading">
                          <span className="sample-tag">
                            {length} · Sample summary
                          </span>
                          <button
                            className="text-button"
                            onClick={() => {
                              navigator.clipboard
                                ?.writeText(
                                  "Database normalization: organize data to reduce redundancy. 1NF: atomic values. 2NF: no partial dependencies. 3NF: no transitive dependencies.",
                                )
                                .then(() => notify("Sample summary copied."))
                                .catch(() =>
                                  notify(
                                    "Copy is unavailable in this browser.",
                                  ),
                                )
                            }}
                          >
                            Copy summary
                          </button>
                        </div>
                        <h2>{doc?.name}</h2>
                        <p className="output-intro">
                          This illustrative summary demonstrates the output
                          layout; it is not extracted from the selected
                          document.
                        </p>
                        <h3>The big picture</h3>
                        <p>
                          Database normalization is a systematic approach to
                          organizing relational data. It reduces redundancy and
                          prevents insertion, update, and deletion anomalies.
                        </p>
                        <h3>Key takeaways</h3>
                        <ul>
                          <li>
                            <strong>First Normal Form (1NF):</strong> Each field
                            contains a single, atomic value.
                          </li>
                          {length !== "Short" && (
                            <>
                              <li>
                                <strong>Second Normal Form (2NF):</strong> Every
                                non-key attribute depends on the entire primary
                                key.
                              </li>
                              <li>
                                <strong>Third Normal Form (3NF):</strong> Remove
                                transitive dependencies between non-key
                                attributes.
                              </li>
                            </>
                          )}
                        </ul>
                        {length === "Detailed" && (
                          <>
                            <h3>A simple example</h3>
                            <p>
                              Separate student information and department
                              information into two tables. A department ID
                              connects them, so department details only need to
                              be updated once.
                            </p>
                            <h3>Remember this</h3>
                            <p>
                              Normalization is about giving each fact one clear
                              home. Balance structure with your application’s
                              query needs.
                            </p>
                          </>
                        )}
                        <div className="summary-takeaway">
                          <Icon name="spark" size={20} />
                          <p>
                            One fact, one place. That’s the heart of good
                            database design.
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
                          Select your notes and generate a sample summary
                          <br />
                          to see a clearer, simpler view.
                        </p>
                      </div>
                    )}
                  </section>
                </div>
              )}
              {page === "Revision Questions" && (
                <>
                  <section className="panel revision-config">
                    <div>
                      {documentSelect}
                      <label className="field">
                        Chapter
                        <select>
                          <option>All chapters</option>
                          <option>Chapter 1 · Foundations</option>
                          <option>Chapter 3 · Normalization</option>
                        </select>
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
                      disabled={busy || !doc}
                      onClick={() => {
                        setExpanded([])
                        setSaved([])
                        setSavedOnly(false)
                        simulate(() => setGenerated(true))
                      }}
                    >
                      {busy ? (
                        <span className="spinner" />
                      ) : (
                        <Icon name="spark" size={17} />
                      )}{" "}
                      {busy ? "Creating samples…" : "Generate sample questions"}
                    </Button>
                  </section>
                  <div className="revision-toolbar">
                    <div className="tabs">
                      {["Multiple Choice", "Short Answer", "Long Answer"].map(
                        (t) => (
                          <button
                            key={t}
                            className={tab === t ? "selected" : ""}
                            onClick={() => {
                              setTab(t)
                              setExpanded([])
                            }}
                          >
                            {t}
                          </button>
                        ),
                      )}
                    </div>
                    <button
                      className={`text-button ${savedOnly ? "selected" : ""}`}
                      onClick={() => setSavedOnly(!savedOnly)}
                    >
                      {savedOnly
                        ? "Show all samples"
                        : `Saved this session (${saved.length})`}
                    </button>
                  </div>
                  {generated ? (
                    <div className="questions-list">
                      <div className="questions-caption">
                        <span className="sample-tag">
                          Generated sample questions
                        </span>
                        <span>
                          {difficulty} difficulty · {doc?.name}
                        </span>
                      </div>
                      {Array.from({ length: Number(count) }, (_, i) => i)
                        .filter((i) => !savedOnly || saved.includes(i))
                        .map((i) => (
                          <section className="panel question-card" key={i}>
                            <div className="question-card-header">
                              <span>
                                QUESTION {i + 1} <i>·</i> {tab.toUpperCase()}
                              </span>
                              <button
                                className="text-button"
                                onClick={() =>
                                  setSaved((s) =>
                                    s.includes(i)
                                      ? s.filter((n) => n !== i)
                                      : [...s, i],
                                  )
                                }
                              >
                                {saved.includes(i) ? (
                                  <>
                                    <Icon name="check" size={15} />
                                    Saved
                                  </>
                                ) : (
                                  <>
                                    <Icon name="plus" size={15} />
                                    Save question
                                  </>
                                )}
                              </button>
                            </div>
                            <h3>
                              {tab === "Long Answer"
                                ? [
                                    "Explain the three normal forms and illustrate each with a database example.",
                                    "Discuss the trade-offs between normalization and query performance.",
                                    "Describe insertion, update, and deletion anomalies with examples.",
                                  ][i % 3]
                                : [
                                    "What is the primary purpose of database normalization?",
                                    "Which normal form removes partial dependencies?",
                                    "What is a transitive dependency?",
                                    "How does a primary key identify a record?",
                                    "Why should repeating groups be avoided?",
                                  ][i % 5]}
                            </h3>
                            {tab === "Multiple Choice" && (
                              <div className="answer-options">
                                {(i % 5 === 0
                                  ? [
                                      "Increase data duplication",
                                      "Reduce redundancy and improve consistency",
                                      "Remove all primary keys",
                                      "Store every fact in a single table",
                                    ]
                                  : i % 5 === 1
                                    ? [
                                        "First Normal Form",
                                        "Second Normal Form",
                                        "Third Normal Form",
                                        "No normal form",
                                      ]
                                    : i % 5 === 2
                                      ? [
                                          "An attribute depends on a non-key attribute",
                                          "Every attribute is a key",
                                          "A table has no rows",
                                          "A field contains a single value",
                                        ]
                                      : i % 5 === 3
                                        ? [
                                            "By its position in the table",
                                            "By a unique, non-null value",
                                            "By its display color",
                                            "By its file size",
                                          ]
                                        : [
                                            "They always improve consistency",
                                            "They make data harder to query and update",
                                            "They eliminate dependencies",
                                            "They automatically create keys",
                                          ]
                                ).map((o, j) => (
                                  <div key={o}>
                                    <span>{String.fromCharCode(65 + j)}</span>
                                    {o}
                                  </div>
                                ))}
                              </div>
                            )}
                            <button
                              className="text-button"
                              onClick={() =>
                                setExpanded((s) =>
                                  s.includes(i)
                                    ? s.filter((n) => n !== i)
                                    : [...s, i],
                                )
                              }
                            >
                              {expanded.includes(i)
                                ? "Hide sample answer"
                                : "Show sample answer"}
                              <Icon name="chevron" size={15} />
                            </button>
                            {expanded.includes(i) && (
                              <div className="expanded-answer">
                                <strong>Sample answer</strong>
                                <p>
                                  {tab === "Long Answer"
                                    ? "1NF ensures atomic values. 2NF removes partial dependencies on a composite key. 3NF removes transitive dependencies. For example, separate departments from students so department details are stored once, and link the tables with a department ID."
                                    : [
                                        "Normalization reduces redundancy and improves consistency by organizing related facts into separate tables.",
                                        "Second Normal Form (2NF) removes partial dependencies. Each non-key attribute must depend on the whole primary key.",
                                        "A transitive dependency occurs when a non-key attribute depends on another non-key attribute rather than directly on the key.",
                                        "A primary key uses a unique, non-null value to identify each record.",
                                        "Repeating groups make querying and updating more difficult and violate First Normal Form.",
                                      ][i % 5]}
                                </p>
                              </div>
                            )}
                          </section>
                        ))}
                      {savedOnly && !saved.length && (
                        <div className="empty-state panel">
                          <Icon name="cards" size={28} />
                          <h2>Your saved set starts here.</h2>
                          <p>
                            Save a sample question to revisit it during this
                            session.
                          </p>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="empty-state panel revision-empty">
                      <div className="large-spark">
                        <Icon name="cards" size={30} />
                      </div>
                      <h2>A little practice. A lot more confidence.</h2>
                      <p>
                        Choose your material and generate sample questions.
                        <br />
                        Saved questions stay available during this session only.
                      </p>
                    </div>
                  )}
                </>
              )}
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
        <div className="modal-backdrop" onClick={() => setModal(null)}>
          <section
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="modal-title"
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => {
              if (e.key === "Escape") setModal(null)
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
              autoFocus
              onClick={() => setModal(null)}
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
                  Add a PDF to try the prototype. Only its filename is used
                  locally; your file is not uploaded, read, or stored.
                </p>
                <input
                  type="file"
                  accept=".pdf,application/pdf"
                  ref={fileInput}
                  hidden
                  onChange={(e) => selectFile(e.target.files?.[0])}
                />
                <button
                  className="dropzone modal-dropzone"
                  onClick={() => fileInput.current?.click()}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault()
                    selectFile(e.dataTransfer.files[0])
                  }}
                >
                  <Icon name={uploadName ? "file" : "upload"} size={25} />
                  <strong>
                    {uploadName || "Choose a PDF or drop it here"}
                  </strong>
                  <small>PDF only · Maximum 20 MB</small>
                </button>
                {uploadError && (
                  <p className="error-message" role="alert">
                    {uploadError}
                  </p>
                )}
                <div className="modal-actions">
                  <Button variant="secondary" onClick={() => setModal(null)}>
                    Cancel
                  </Button>
                  <Button
                    disabled={!uploadName}
                    onClick={() => {
                      setDocs((d) => [
                        ...d,
                        {
                          id: Date.now(),
                          name: uploadName.replace(/\.pdf$/i, ""),
                          subject: "Uncategorized",
                          size: "Demo file",
                          pages: 0,
                          date: "Just now",
                          color: "purple",
                        },
                      ])
                      setUploadName("")
                      setModal(null)
                      notify(
                        "Demo document added. No PDF processing was performed.",
                      )
                    }}
                  >
                    Add demo document
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
                  “{doc?.name}” will be removed from this demo workspace. Your
                  original PDF will not be affected.
                </p>
                <div className="modal-actions">
                  <Button variant="secondary" onClick={() => setModal(null)}>
                    Keep document
                  </Button>
                  <Button
                    variant="danger"
                    onClick={() => {
                      setDocs((d) => d.filter((item) => item.id !== activeDoc))
                      setMessages([])
                      setSummary(false)
                      setGenerated(false)
                      setModal(null)
                      notify("Demo document removed.")
                    }}
                  >
                    Remove document
                  </Button>
                </div>
              </>
            )}
            {modal === "help" && (
              <>
                <p>
                  Start with the sample documents, ask a question, explore a
                  summary, or build a revision set using the sidebar.
                </p>
                <div className="help-note">
                  <strong>A prototype, not a connected AI service.</strong>
                  <p>
                    All AI outputs are prewritten examples. Files are not
                    uploaded or processed. Changes and saved questions are kept
                    only in this browser session. No account is required.
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
