import { createRootRoute, HeadContent, Outlet, Scripts } from "@tanstack/react-router";
import { AuthProvider } from "@/lib/auth/provider";
import { PreviewHostBridge } from "@/components/preview-host-bridge";
import appCss from "../styles.css?url";

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1, viewport-fit=cover" },
      { title: "DN-AI — Think Smarter. Solve Anything." },
      { name: "description", content: "DN-AI — trợ lý AI tiếng Việt do Đức Nhân phát triển." },
      { name: "author", content: "Đức Nhân" },
      { name: "theme-color", content: "#0b1020" },
    ],
    links: [
      { rel: "icon", type: "image/svg+xml", href: "/favicon.svg" },
      { rel: "stylesheet", href: appCss },
      { rel: "manifest", href: "/__grok/manifest.webmanifest" },
      { rel: "apple-touch-icon", href: "/__grok/icon-180.png" },
    ],
  }),
  component: () => (
    <html lang="vi" suppressHydrationWarning>
      <head><HeadContent /></head>
      <body>
        <PreviewHostBridge />
        <AuthProvider><Outlet /></AuthProvider>
        <Scripts />
      </body>
    </html>
  ),
});
