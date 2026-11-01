import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";

type Message = { role: "user" | "assistant"; content: string; meta?: string };
type ApiStatus = "checking" | "ready" | "offline";

const HISTORY_KEY = "dnai-chat-v2";

function isMessageList(value: unknown): value is Message[] {
  return Array.isArray(value) && value.every((item) =>
    !!item && typeof item === "object" &&
    ((item as Message).role === "user" || (item as Message).role === "assistant") &&
    typeof (item as Message).content === "string",
  );
}

export const Route = createFileRoute("/")({ component: Home });

function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [modelName, setModelName] = useState("DN-AI Micro");
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [generationEnabled, setGenerationEnabled] = useState(false);
  const [document, setDocument] = useState("");
  const [fileName, setFileName] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const hasMessages = messages.length > 0;

  useEffect(() => {
    try {
      const saved = localStorage.getItem(HISTORY_KEY);
      if (saved) {
        const parsed: unknown = JSON.parse(saved);
        if (isMessageList(parsed)) setMessages(parsed.slice(-60));
      }
      // Old sessions may contain corrupted model output; keep the new history clean.
      localStorage.removeItem("dnai-chat-v1");
    } catch {
      localStorage.removeItem(HISTORY_KEY);
    }

    let active = true;
    fetch("/api/chat", { headers: { accept: "application/json" } })
      .then(async (response) => {
        if (!response.ok) throw new Error("API unavailable");
        return response.json();
      })
      .then((data) => {
        if (!active) return;
        if (data?.ok && data?.model) {
          setModelName(data.model.generationEnabled ? "DN-AI Micro" : "Kho kiến thức + công cụ nội bộ");
          setGenerationEnabled(Boolean(data.model.generationEnabled));
          setApiStatus("ready");
        } else {
          setApiStatus("offline");
        }
      })
      .catch(() => {
        if (active) {
          setApiStatus("offline");
          setModelName("Chưa kết nối máy chủ");
        }
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(messages.slice(-60))); }
    catch { /* browser storage can be disabled or full */ }
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  useEffect(() => {
    const field = inputRef.current;
    if (!field) return;
    field.style.height = "auto";
    field.style.height = `${Math.min(field.scrollHeight, 120)}px`;
  }, [input]);

  async function sendMessage(value = input) {
    const content = value.trim();
    if (!content || busy) return;
    const next: Message[] = [...messages, { role: "user", content }];
    setMessages(next);
    setInput("");
    setError("");
    setBusy(true);
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json", accept: "application/json" },
        body: JSON.stringify({
          messages: next.slice(-8).map(({ role, content: text }) => ({ role, content: text })),
          mode: "chat",
          temperature: 0.55,
          topK: 24,
          topP: 0.88,
          maxTokens: 64,
          document: document || undefined,
          attachment: fileName.toLowerCase().endsWith(".csv") ? "table" : document ? "text" : "none",
        }),
      });
      const result = await response.json();
      if (!response.ok || !result?.ok || typeof result.content !== "string") {
        throw new Error(typeof result?.error === "string" ? result.error : "DN-AI chưa thể trả lời.");
      }
      setMessages((current) => [...current, {
        role: "assistant",
        content: result.content,
        meta: `${result.meta?.model || modelName}${result.meta?.ms ? ` · ${result.meta.ms} ms` : ""}${result.meta?.lowConfidence ? " · cần kiểm tra lại" : ""}`,
      }]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Có lỗi kết nối. Hãy thử lại.");
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  }

  async function attachFile(file?: File) {
    if (!file) return;
    if (file.size > 12_000) {
      setError("Tệp tối đa 12 KB trong phiên bản này để nội dung không bị cắt khi gửi.");
      return;
    }
    if (!/\.(txt|md|csv|json)$/i.test(file.name)) {
      setError("Hiện hỗ trợ TXT, MD, CSV và JSON. PDF/DOCX chưa được bật.");
      return;
    }
    try {
      const content = await file.text();
      if (!content.trim()) throw new Error("Tệp đang trống.");
      setDocument(content);
      setFileName(file.name);
      setError("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Không đọc được tệp này.");
    }
  }

  function resetChat() {
    setMessages([]);
    setDocument("");
    setFileName("");
    setInput("");
    setError("");
    try { localStorage.removeItem(HISTORY_KEY); } catch { /* ignore */ }
    inputRef.current?.focus();
  }

  const statusLabel = apiStatus === "checking" ? "Đang kết nối" : apiStatus === "ready" ? "Đang hoạt động" : "Mất kết nối";

  return (
    <div className="shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="DN-AI trang chủ">
          <span className="brand-mark" aria-hidden="true">✦</span>
          <span>DN-AI</span>
        </a>
        <div className="top-actions">
          <span className={`status-pill ${apiStatus}`}><i aria-hidden="true" />{statusLabel}</span>
          <button className="new-chat" onClick={resetChat} type="button"><span aria-hidden="true">＋</span> Cuộc trò chuyện mới</button>
        </div>
      </header>

      <main className="main">
        {!hasMessages && (
          <section className="hero">
            <div className="eyebrow">Trợ lý AI do Đức Nhân phát triển</div>
            <h1>Xin chào, đây là <span className="gradient">DN-AI</span></h1>
            <p>Hỏi đáp tiếng Việt, học tập, tính toán và tra cứu tài liệu trong một giao diện gọn nhẹ.</p>
            <div className="chips">
              {[
                "Trí tuệ nhân tạo là gì? Nêu 3 ứng dụng thực tế.",
                "Giải thích định lý Pythagoras.",
                "1+1 bằng mấy?",
              ].map((item) => (
                <button className="chip" key={item} onClick={() => { setInput(item); inputRef.current?.focus(); }}>{item}</button>
              ))}
            </div>
          </section>
        )}

        <section className="chat-card" aria-label="Khung trò chuyện DN-AI">
          <div className="chat-head">
            <div className="chat-title">
              <strong>Trợ lý DN-AI</strong>
              <small>{modelName}</small>
            </div>
            <span className="private-label">{generationEnabled ? "Mô hình nội bộ" : "Kho kiến thức nội bộ"}</span>
          </div>

          <div className="messages" aria-live="polite" aria-relevant="additions text">
            {!hasMessages && (
              <div className="empty">
                <b>✦ Bắt đầu cuộc trò chuyện</b>
                Chọn một gợi ý hoặc nhập câu hỏi bên dưới.
              </div>
            )}
            {messages.map((message, index) => (
              <div className={`message ${message.role}`} key={`${index}-${message.role}`}>
                <div className="message-content">{message.content}</div>
                {message.meta && <span className="meta">{message.meta}</span>}
              </div>
            ))}
            {busy && <div className="message assistant pending"><span className="typing-dot" /> DN-AI đang xử lý…</div>}
            <div ref={endRef} />
          </div>

          <div className="composer">
            {fileName && (
              <div className="attachment-chip">
                <span>📎 {fileName}</span>
                <button type="button" onClick={() => { setFileName(""); setDocument(""); }} aria-label="Bỏ tệp đính kèm">×</button>
              </div>
            )}
            <form className="input-row" onSubmit={(event) => { event.preventDefault(); void sendMessage(); }}>
              <label className="file-label" title="Đính kèm tệp TXT, MD, CSV hoặc JSON">
                <span aria-hidden="true">＋</span>
                <input type="file" accept=".txt,.md,.csv,.json" onChange={(event) => { void attachFile(event.target.files?.[0]); event.currentTarget.value = ""; }} />
              </label>
              <textarea
                ref={inputRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                    event.preventDefault();
                    void sendMessage();
                  }
                }}
                placeholder="Nhập câu hỏi của bạn…"
                rows={1}
                aria-label="Tin nhắn"
                maxLength={1200}
              />
              <button className="send" type="submit" disabled={busy || !input.trim() || apiStatus === "offline"}>{busy ? "Đang xử lý" : "Gửi ↑"}</button>
            </form>
            {error && <div className="error" role="alert">{error}</div>}
            {apiStatus === "offline" && <div className="error" role="status">Không kết nối được máy chủ DN-AI. Hãy tải lại trang sau.</div>}
            <div className="footnote">{generationEnabled ? "Mô hình nội bộ đang hoạt động; câu trả lời vẫn cần được kiểm tra." : "Bản Micro đang dùng kho kiến thức và công cụ nội bộ; câu hỏi ngoài phạm vi sẽ được báo rõ."}</div>
          </div>
        </section>
      </main>
      <footer className="footer">© 2026 DN-AI <span>·</span> Người phát triển: <strong>Đức Nhân</strong> <span>·</span> Không sử dụng API chatbot bên thứ ba.</footer>
    </div>
  );
}
