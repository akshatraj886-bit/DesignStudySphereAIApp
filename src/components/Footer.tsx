export default function Footer({
  onNavigate,
  onHelp,
}: {
  onNavigate: (page: string) => void
  onHelp: () => void
}) {
  return (
    <footer className="site-footer glass-card glass-light">
      <div className="footer-top">
        <div className="footer-brand">
          <strong>
            StudySphere<span> AI</span>
          </strong>
          <p>
            Your notes. Your questions.
            <br />
            Instant, grounded understanding.
          </p>
          <span className="footer-caption">100% Free &amp; Offline Academic Assistant</span>
        </div>
        <nav aria-label="Footer workspace">
          <h3>WORKSPACE</h3>
          <button onClick={() => onNavigate("My Documents")}>
            My documents
          </button>
          <button onClick={() => onNavigate("Ask Your Notes")}>
            Ask your notes
          </button>
          <button onClick={() => onNavigate("Revision Questions")}>
            Revision questions
          </button>
        </nav>
        <nav aria-label="Footer resources">
          <h3>EXPLORE</h3>
          <button onClick={() => onNavigate("AI Summarizer")}>
            AI summarizer
          </button>
          <button onClick={() => onNavigate("Dashboard")}>Dashboard</button>
          <button onClick={onHelp}>Help &amp; support</button>
        </nav>
        <nav aria-label="Community links">
          <h3>RESOURCES</h3>
          <button onClick={() => onNavigate("Ask Your Notes")}>
            Concept Q&amp;A
          </button>
          <button onClick={() => onNavigate("AI Summarizer")}>
            Quick Summaries
          </button>
          <span>Grounded in your notes</span>
        </nav>
      </div>
      <div className="footer-bottom">
        <span>StudySphere AI · Offline-First RAG &amp; Revision Companion</span>
        <span>Final Year Capstone Project · Computer Science &amp; Engineering</span>
      </div>
    </footer>
  )
}
