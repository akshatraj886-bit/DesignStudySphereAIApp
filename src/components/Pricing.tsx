import GlassCard from "./GlassCard"

const plans = [
  {
    name: "Curious",
    price: "0",
    description: "For your first lightbulb moments.",
    features: [
      "5 study documents",
      "25 questions per month",
      "Short & medium summaries",
    ],
    cta: "Explore the free plan",
    featured: false,
  },
  {
    name: "Focused",
    price: "6",
    description: "A little more room to go deeper.",
    features: [
      "50 study documents",
      "250 questions per month",
      "Detailed summaries & revision sets",
    ],
    cta: "Explore Focused",
    featured: true,
  },
  {
    name: "Limitless",
    price: "12",
    description: "For every chapter of your journey.",
    features: [
      "Unlimited study documents",
      "1,000 questions per month",
      "All study tools & priority processing",
    ],
    cta: "Explore Limitless",
    featured: false,
  },
]

export default function Pricing({
  onExplore,
}: {
  onExplore: (plan: string) => void
}) {
  return (
    <section
      className="pricing-section"
      id="study-plans"
      aria-labelledby="pricing-title"
    >
      <div className="pricing-heading">
        <span className="eyebrow">ROOM TO GROW</span>
        <h2 id="pricing-title">Find your study rhythm.</h2>
        <p>
          A plan for every kind of curious. Sample pricing for demonstration
          only.
        </p>
      </div>
      <div className="pricing-grid">
        {plans.map((plan) => (
          <GlassCard
            key={plan.name}
            variant={plan.featured ? "heavy" : "medium"}
            luminous={plan.featured}
            className={`pricing-card ${
              plan.featured ? "recommended-plan" : ""
            }`}
          >
            <div className="plan-heading">
              <h3>{plan.name}</h3>
              {plan.featured && (
                <span className="plan-badge">THE STUDENT FAVORITE</span>
              )}
            </div>
            <p className="plan-description">{plan.description}</p>
            <div className="plan-price">
              ${plan.price}
              <span>/ month</span>
            </div>
            <ul>
              {plan.features.map((feature) => (
                <li key={feature}>
                  <svg
                    width="15"
                    height="15"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    aria-hidden="true"
                  >
                    <path d="m5 12 4 4L19 6" />
                  </svg>
                  {feature}
                </li>
              ))}
            </ul>
            <button
              className={`button ${plan.featured ? "primary" : "secondary"}`}
              onClick={() => onExplore(plan.name)}
            >
              {plan.cta}
              <span aria-hidden="true">↗</span>
            </button>
          </GlassCard>
        ))}
      </div>
      <p className="pricing-disclaimer">
        Concept plans only. No subscriptions, payments, or plan limits are
        active in this prototype.
      </p>
    </section>
  )
}
