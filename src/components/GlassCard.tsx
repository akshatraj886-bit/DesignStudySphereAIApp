import type { HTMLAttributes, ReactNode } from "react"
import type { GlassVariant } from "./glassTheme"

export interface GlassCardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode
  variant?: GlassVariant
  luminous?: boolean
}

export default function GlassCard({
  children,
  variant = "medium",
  luminous = false,
  className = "",
  ...props
}: GlassCardProps) {
  return (
    <div
      className={`glass-card glass-${variant} ${
        luminous ? "glass-luminous" : ""
      } ${className}`}
      {...props}
    >
      {children}
    </div>
  )
}
