import { createFileRoute } from "@tanstack/react-router";
import { modelStatus, reply, type GenRequest } from "@/lib/dnai/engine.server";

const MODES: GenRequest["mode"][] = ["chat", "academic", "writing", "coding", "data", "study"];

function boundedNumber(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = typeof value === "number" || typeof value === "string" ? Number(value) : Number.NaN;
  return Number.isFinite(parsed) ? Math.min(max, Math.max(min, parsed)) : fallback;
}

function safeText(value: string, max: number): string {
  return value.normalize("NFC").replaceAll("\u0000", "").slice(0, max);
}

export const Route = createFileRoute("/api/chat")({
  server: {
    handlers: {
      GET: async () => Response.json({ ok: true, model: modelStatus() }),
      POST: async ({ request }) => {
        try {
          const contentType = request.headers.get("content-type") ?? "";
          if (!contentType.toLowerCase().includes("application/json")) {
            return Response.json({ ok: false, error: "Yêu cầu phải được gửi dưới dạng JSON." }, { status: 415 });
          }

          const raw: unknown = await request.json();
          if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
            return Response.json({ ok: false, error: "Dữ liệu gửi lên không hợp lệ." }, { status: 400 });
          }
          const body = raw as Partial<GenRequest>;
          if (!Array.isArray(body.messages) || body.messages.length === 0) {
            return Response.json({ ok: false, error: "Vui lòng nhập tin nhắn." }, { status: 400 });
          }

          const messages = body.messages
            .filter((m): m is { role: "user" | "assistant"; content: string } =>
              !!m && (m.role === "user" || m.role === "assistant") && typeof m.content === "string")
            .slice(-8)
            .map((m) => ({ role: m.role, content: safeText(m.content, 1200) }));
          if (!messages.some((message) => message.role === "user" && message.content.trim())) {
            return Response.json({ ok: false, error: "Tin nhắn không hợp lệ hoặc đang trống." }, { status: 400 });
          }

          const requestedMode = String(body.mode ?? "chat") as GenRequest["mode"];
          const result = reply({
            messages,
            mode: MODES.includes(requestedMode) ? requestedMode : "chat",
            temperature: boundedNumber(body.temperature, 0.65, 0.1, 1.2),
            topK: boundedNumber(body.topK, 32, 1, 80),
            topP: boundedNumber(body.topP, 0.9, 0.1, 1),
            maxTokens: boundedNumber(body.maxTokens, 64, 8, 80),
            seed: boundedNumber(body.seed, Date.now(), 0, Number.MAX_SAFE_INTEGER),
            document: typeof body.document === "string" ? safeText(body.document, 12_000) : undefined,
            attachment:
              body.attachment === "text" || body.attachment === "table" || body.attachment === "image"
                ? body.attachment
                : "none",
          });
          const status = result.ok ? 200 : result.error.startsWith("Quá nhiều lượt") ? 429 : 400;
          return Response.json(result, { status });
        } catch {
          return Response.json({ ok: false, error: "Không thể đọc yêu cầu. Hãy thử gửi lại nội dung ngắn hơn." }, { status: 400 });
        }
      },
    },
  },
});
