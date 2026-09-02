/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        cyber: {
          bg: "#0B0E14",
          surface: "#121722",
          card: "#182030",
          border: "#26334D",
          accent: "#00E5FF",
          accentGlow: "#00E5FF33",
          green: "#00FF88",
          warning: "#FFB300",
          danger: "#FF3366",
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
