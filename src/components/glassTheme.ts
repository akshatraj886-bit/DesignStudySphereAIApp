import type { CSSProperties } from "react"

export type GlassVariant = "light" | "medium" | "heavy"

// Deeper glass scatters more light; elevated glass is denser and sharper.
export const glassStyles = {
  light: {
    blur: "32px",
    fill: "rgba(255,255,255,.08)",
    border: "rgba(255,255,255,.18)",
  },
  medium: {
    blur: "20px",
    fill: "rgba(16,23,53,.56)",
    border: "rgba(255,255,255,.22)",
  },
  heavy: {
    blur: "12px",
    fill: "rgba(22,28,58,.94)",
    border: "rgba(255,255,255,.35)",
  },
} satisfies Record<GlassVariant, { blur: string fill: string border: string }>

export const glassTheme = Object.fromEntries(
  Object.entries(glassStyles).flatMap(([tier, values]) =>
    Object.entries(values).map(([key, value]) => [
      `--glass-${tier}-${key}`,
      value,
    ]),
  ),
) as CSSProperties
