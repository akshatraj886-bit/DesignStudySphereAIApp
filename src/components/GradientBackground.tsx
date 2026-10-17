export default function GradientBackground({
  scheme = "aurora",
}: {
  scheme?: "aurora" | "dusk"
}) {
  return (
    <div
      className={`gradient-background gradient-${scheme}`}
      aria-hidden="true"
    >
      <div className="gradient-mesh" />
      <div className="gradient-orb orb-violet" />
      <div className="gradient-orb orb-blue" />
      <div className="gradient-orb orb-teal" />
      <div className="gradient-grain" />
    </div>
  )
}
