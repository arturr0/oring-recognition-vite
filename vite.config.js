import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * Custom middleware to add cross-origin isolation headers for dev server
 * Required for SharedArrayBuffer and other advanced browser features
 */
function crossOriginIsolation() {
  return {
    name: 'vite-plugin-cross-origin-isolation',
    configureServer(server) {
      server.middlewares.use((_req, res, next) => {
        res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
        res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
        next();
      });
    },
  };
}

export default defineConfig({
  plugins: [
    react(),
    crossOriginIsolation()
  ],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          // Separate ONNX worker into its own chunk
          if (id.includes('onnxWorker.js')) {
            return 'onnx-worker';
          }
        }
      }
    },
    // Enable modern browser targets
    target: 'es2020'
  },
  server: {
    // Development server configuration
    host: '0.0.0.0',
    port: 3000,
    strictPort: true,
    // Enable CORS for development
    cors: true
  },
  preview: {
    // Production preview server configuration
    host: '0.0.0.0',
    port: process.env.PORT || 4173,
    strictPort: true,
    // Security: only allow requests from your domain
    allowedHosts: [
      'oring-recognition-vite.onrender.com',
      'localhost' // For local testing
    ]
  },
  // Optimize dependencies
  optimizeDeps: {
    esbuildOptions: {
      // Enable top-level await
      target: 'es2020'
    }
  }
});
