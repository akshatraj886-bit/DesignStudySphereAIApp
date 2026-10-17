import GlassCard from "./GlassCard"
import type { ReactNode } from "react"

export interface StudyFeature {
  title: string
  description: string
  icon: ReactNode
  color: string
  onClick: () => void
}

export default function Features({ features }: { features: StudyFeature[] }) {
  return (
    <section className="quick-section">
      <div className="section-heading">
        <div>
          <h2>A little help, a big difference</h2>
          <p>Make your next study session a little smarter.</p>
        </div>
      </div>
      <div className="quick-grid">
        {features.map((feature) => (
          <GlassCard className="feature-glass" key={feature.title}>
            <button className="quick-card" onClick={feature.onClick}>
              <span className={`icon-box ${feature.color}`}>
                {feature.icon}
              </span>
              <strong>{feature.title}</strong>
              <p>{feature.description}</p>
              <svg
                className="feature-arrow"
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                aria-hidden="true"
              >
                <path d="M5 12h14m-5-5 5 5-5 5" />
              </svg>
            </button>
          </GlassCard>
        ))}
      </div>
    </section>
  )
}
