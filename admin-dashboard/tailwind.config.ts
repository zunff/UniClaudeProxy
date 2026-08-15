import type { Config } from "tailwindcss";

export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx,js,jsx}"],
  theme: {
    container: {
      center: true,
      padding: "1rem",
      screens: { "2xl": "1400px" },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        // Clean Tech Palette
        brand: {
          cyan: "#06b6d4",
          cyanLight: "#22d3ee",
          cyanDim: "#0891b2",
          blue: "#3b82f6",
          violet: "#8b5cf6",
          green: "#10b981",
          emerald: "#059669",
          amber: "#f59e0b",
          rose: "#f43f5e",
          // Clean solid surfaces
          bg: "#090d16",
          panel: "#0f1629",
          panel2: "#141d36",
          panel3: "#1a2544",
          borderSubtle: "#1e293b",
          borderHover: "#334155",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(6,182,212,0.3), 0 4px 20px -2px rgba(6,182,212,0.15)",
        glowSm: "0 0 0 1px rgba(6,182,212,0.25)",
        card: "0 4px 20px -2px rgba(0, 0, 0, 0.4)",
        cardHover: "0 8px 30px -4px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(6, 182, 212, 0.25)",
      },
      keyframes: {
        shimmer: {
          "100%": { transform: "translateX(100%)" },
        },
        pulseSubtle: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.5" },
        },
        fadeIn: {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        shimmer: "shimmer 2s infinite",
        pulseSubtle: "pulseSubtle 2s ease-in-out infinite",
        fadeIn: "fadeIn 0.2s ease-out",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
} satisfies Config;
