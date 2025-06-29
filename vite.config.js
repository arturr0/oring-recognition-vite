import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Custom middleware to add cross-origin isolation headers for dev server
function crossOriginIsolation() {
  return {
    name: 'vite-plugin-cross-origin-isolation',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
        res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
        next();
      });
    }
  };
}

export default defineConfig({
  plugins: [react(), crossOriginIsolation()],
  build: {
    rollupOptions: {
      output: {
        // Ensure worker files are in separate chunks
        manualChunks(id) {
          if (id.includes('onnxWorker.js')) {
            return 'onnx-worker';
          }
        }
      }
    }
  },
  preview: {
    host: '0.0.0.0',                 // listen on all interfaces
    port: process.env.PORT || 4173, // use Render assigned port or default
    allowedHosts: ['oring-recognition-vite.onrender.com']  // add your Render hostname here
  }
});
