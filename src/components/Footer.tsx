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
            Instant understanding.
          </p>
          <span className="footer-caption">Built for curious minds.</span>
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
          <a href="#study-plans">Study plans</a>
          <button onClick={onHelp}>Help & support</button>
        </nav>
        <nav aria-label="Community links">
          <h3>STAY CURIOUS</h3>
          <a
            href="https://github.com/topics/study-tools"
            target="_blank"
            rel="noreferrer"
          >
            GitHub study tools ↗
          </a>
          <a
            href="https://www.reddit.com/r/GetStudying/"
            target="_blank"
            rel="noreferrer"
          >
            Reddit study community ↗
          </a>
          <span>Independent learning resources</span>
        </nav>
      </div>
      <div className="footer-bottom">
        <span>StudySphere AI · A space for your next breakthrough.</span>
        <span>Interactive prototype · Sample data only</span>
      </div>
    </footer>
  )
}
