import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:10388",
        changeOrigin: true,
      },
      "/stats": {
        target: "http://127.0.0.1:10388",
        changeOrigin: true,
      },
      "/health": {
        target: "http://127.0.0.1:10388",
        changeOrigin: true,
      },
    },
  },
});
