import { useEffect, useState, type ReactNode } from "react"

export default function GlassNavbar({ children }: { children: ReactNode }) {
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => {
    const update = () => setScrolled(window.scrollY > 12)
    update()
    window.addEventListener("scroll", update, { passive: true })
    return () => window.removeEventListener("scroll", update)
  }, [])
  return (
    <header className={`topbar glass-navbar ${scrolled ? "is-scrolled" : ""}`}>
      {children}
    </header>
  )
}
