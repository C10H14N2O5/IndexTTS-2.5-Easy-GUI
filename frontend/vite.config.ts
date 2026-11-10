/// <reference types="vitest/config" />
import { fileURLToPath, URL } from 'node:url';
import { defineConfig, type ProxyOptions } from 'vite';
import react from '@vitejs/plugin-react';

const backend = process.env.INDEXTTS_DEV_BACKEND ?? 'http://127.0.0.1:7860';

// The FastAPI app rejects cross-origin writes. Rewrite Origin so requests that
// arrive through the Vite dev proxy look like they came from the backend page.
const proxy: ProxyOptions = {
  target: backend,
  changeOrigin: true,
  configure: (server) => {
    server.on('proxyReq', (request) => {
      if (request.getHeader('origin')) request.setHeader('origin', backend);
      request.removeHeader('sec-fetch-site');
    });
  },
};

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  build: {
    outDir: '../tools/gui_web',
    emptyOutDir: true,
    assetsDir: 'assets',
    sourcemap: false,
    chunkSizeWarningLimit: 900,
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: { '/api': proxy, '/proofread/api': proxy, '/exports': proxy },
  },
  test: {
    environment: 'jsdom',
    globals: false,
    setupFiles: ['./src/test/setup.ts'],
    css: { modules: { classNameStrategy: 'non-scoped' } },
    include: ['src/**/*.test.{ts,tsx}', 'scripts/**/*.test.{ts,mjs}'],
  },
});
