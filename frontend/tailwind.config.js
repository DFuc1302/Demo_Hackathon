/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: { ink: '#10201c', paper: '#f5f3ed', mint: '#bdebd2', coral: '#b33b26', amber: '#f4bd5f', slate: { 500: '#5b6a80' } },
      fontFamily: { display: ['Georgia', 'serif'], sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'] },
    },
  },
  plugins: [],
}
