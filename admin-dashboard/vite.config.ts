import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const adminDir = path.dirname(fileURLToPath(import.meta.url));
const defaultTarget = "http://127.0.0.1:9223";

function proxyTargetFromGlobal(): string {
  if (process.env.UNICLAUDE_PROXY) return process.env.UNICLAUDE_PROXY;
  try {
    const raw = JSON.parse(
      fs.readFileSync(path.resolve(adminDir, "../global.json"), "utf-8"),
    );
    const host = raw?.server?.host || "127.0.0.1";
    const port = raw?.server?.port ?? 9223;
    return `http://${host}:${port}`;
  } catch {
    return defaultTarget;
  }
}

const proxyTarget = proxyTargetFromGlobal();

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(adminDir, "./src"),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": { target: proxyTarget, changeOrigin: true },
      "/stats": { target: proxyTarget, changeOrigin: true },
      "/health": { target: proxyTarget, changeOrigin: true },
    },
  },
});
