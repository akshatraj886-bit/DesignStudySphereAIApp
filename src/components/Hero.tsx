import type { ReactNode } from "react"

export default function Hero({ children }: { children: ReactNode }) {
  return (
    <section className="hero glass-card glass-light glass-luminous glass-entrance">
      {children}
    </section>
  )
}
