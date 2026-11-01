import { createServer } from "vite";
import react from "@vitejs/plugin-react";

const vite = await createServer({
  configFile: false,
  server: { middlewareMode: true },
  appType: "custom",
  logLevel: "error",
  plugins: [react()],
});
try {
  await vite.ssrLoadModule("/src/lib/dnai/probe-once.ts");
} finally {
  await vite.close();
}
