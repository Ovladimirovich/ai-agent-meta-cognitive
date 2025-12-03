/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // Основные цвета
        primary: {
          50: '#eff6ff',
          500: '#3b82f6',
          900: '#1e3a8a'
        },
        // Цвета для когнитивных процессов
        cognitive: {
          reflection: {
            50: '#ecfdf5',
            500: '#10b981',
            900: '#064e3b'
          },
          memory: {
            50: '#f3e8ff',
            500: '#8b5cf6',
            900: '#4c1d95'
          },
          learning: {
            50: '#fffbeb',
            500: '#f59e0b',
            900: '#78350f'
          },
          meta: {
            50: '#fdf2f8',
            500: '#ef4444',
            900: '#7f1d1d'
          }
        },
        // Состояния
        state: {
          success: '#22c55e',
          warning: '#eab308',
          error: '#ef4444',
          info: '#3b82f6'
        }
      },
      fontFamily: {
        heading: ['Inter', 'system-ui', 'sans-serif'],
        body: ['Inter', 'system-ui', 'sans-serif']
      }
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}