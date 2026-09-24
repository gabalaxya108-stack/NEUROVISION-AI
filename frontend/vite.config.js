import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/model-info': 'http://127.0.0.1:8000',
      '/predict': 'http://127.0.0.1:8000',
      '/explain': 'http://127.0.0.1:8000',
      '/results': 'http://127.0.0.1:8000',
      '/samples': 'http://127.0.0.1:8000',
    },
  },
})
