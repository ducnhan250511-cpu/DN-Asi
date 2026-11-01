import checkpointJson from "./checkpoint.json";
import knowledgeJson from "./knowledge.json";
import reportJson from "./report.json";
import { analyzeTable, parseCsv, round4, type TableReport } from "./analyze";
import { MicroNet, generateIds, unpackF16, type Arch, type Weights } from "./model";
import { bm25, searchTokens, type Doc } from "./retrieve";
import { trySolve } from "./solver";
import { buildPrompt, decode, tokenizerFromVocab, type Tokenizer } from "./tokenizer";

type Packed = { s: number[]; d: string };
type Checkpoint = {
  name: string;
  arch: Arch;
  vocab: string[];
  train: {
    steps: number;
    train_loss: number;
    val_loss: number;
    val_perplexity: number;
    train_rows: number;
    val_rows: number;
    seconds: number;
  };
  tensors: Record<string, Packed>;
};

export type Citation = { source: string; text: string };
export type ToolTrace = { kind: "solver" | "stats" | "lint" | "vision" | "extract"; title: string; lines: string[] };
export type GenMeta = {
  source: "micro" | "solver" | "stats" | "vision" | "knowledge" | "document" | "unavailable";
  logprob?: number;
  lowConfidence?: boolean;
  promptTokens?: number;
  completionTokens?: number;
  ms?: number;
  model: string;
};

export type EngineReply = {
  ok: true;
  content: string;
  citations: Citation[];
  tool?: ToolTrace;
  stats?: TableReport;
  meta: GenMeta;
};

const checkpoint = checkpointJson as Checkpoint;
const knowledge = knowledgeJson as {
  id: string;
  body: string;
  qa?: { question: string; answer: string }[];
}[];
const report = reportJson as {
  model: string;
  status: string;
  note: string;
  train: Checkpoint["train"];
  arch: Arch;
  vocab_size: number;
  param_count: number;
  samples: { prompt: string; greedy: string; sample: string }[];
};

let net: MicroNet | null = null;
let tok: Tokenizer | null = null;

function runtime() {
  if (net && tok) return { net, tok };
  const weights: Weights = {};
  for (const [name, packed] of Object.entries(checkpoint.tensors)) {
    const arr = unpackF16(packed.d);
    if (arr.length !== packed.s.reduce((a, b) => a * b, 1)) {
      throw new Error(`bad tensor ${name}`);
    }
    weights[name] = arr;
  }
  net = new MicroNet(checkpoint.arch, weights);
  tok = tokenizerFromVocab(checkpoint.vocab);
  return { net, tok };
}

const hits: number[] = [];

function allow(): boolean {
  const now = Date.now();
  while (hits.length && now - (hits[0] ?? 0) > 60_000) hits.shift();
  if (hits.length >= 24) return false;
  hits.push(now);
  return true;
}

const GENERATION_PERPLEXITY_LIMIT = 50;

function canUseFreeGeneration(): boolean {
  return Number.isFinite(checkpoint.train.val_perplexity)
    && checkpoint.train.val_perplexity <= GENERATION_PERPLEXITY_LIMIT
    && checkpoint.train.val_rows >= 20;
}

export function modelStatus() {
  const generationEnabled = canUseFreeGeneration();
  return {
    name: checkpoint.name,
    status: generationEnabled ? report.status : "experimental-limited",
    arch: checkpoint.arch,
    train: checkpoint.train,
    paramCount: report.param_count,
    vocabSize: checkpoint.vocab.length,
    samples: report.samples,
    vision: false,
    thirdPartyApi: false,
    generationEnabled,
    responseMode: generationEnabled ? "model-and-knowledge" : "knowledge-and-tools",
    creator: "Đức Nhân",
    note: generationEnabled
      ? report.note
      : "Chế độ sinh tự do đang tắt vì checkpoint chưa đạt ngưỡng đánh giá. DN-AI chỉ trả lời từ kho kiến thức và công cụ nội bộ để tránh tạo văn bản sai.",
  };
}

export type GenRequest = {
  messages: { role: "user" | "assistant"; content: string }[];
  mode: "chat" | "academic" | "writing" | "coding" | "data" | "study";
  temperature: number;
  topK: number;
  topP: number;
  maxTokens: number;
  seed: number;
  document?: string;
  attachment?: "image" | "text" | "table" | "none";
};

export function reply(req: GenRequest): EngineReply | { ok: false; error: string } {
  if (!allow()) {
    return { ok: false, error: "Quá nhiều lượt trong một phút. Đợi một chút rồi gửi lại." };
  }
  const last = req.messages.filter((m) => m.role === "user").at(-1);
  if (!last?.content.trim()) return { ok: false, error: "Tin nhắn trống." };
  const question = last.content.trim();
  if (req.attachment === "image") {
    return {
      ok: true,
      content:
        "Module thị giác của DN-AI chưa được huấn luyện. Mình không nhận diện nội dung ảnh và không đoán những gì có trong đó.",
      citations: [],
      tool: {
        kind: "vision",
        title: "Vision chưa sẵn sàng",
        lines: ["Checkpoint hiện tại chỉ là mô hình ngôn ngữ chữ.", "Ảnh không được đưa vào inference."],
      },
      meta: { source: "vision", model: checkpoint.name, lowConfidence: false },
    };
  }
  const solved = req.mode === "writing" ? null : trySolve(question);
  if (solved) {
    return {
      ok: true,
      content: solved.answer,
      citations: [],
      tool: { kind: "solver", title: "Công cụ ký hiệu", lines: solved.lines },
      meta: { source: "solver", model: "dnai-solver", lowConfidence: false },
    };
  }
  const direct = findKnownAnswer(question);
  if (direct) {
    return {
      ok: true,
      content: direct.answer,
      citations: [{ source: `Kho kiến thức: ${direct.id}`, text: direct.question }],
      tool: { kind: "extract", title: "Tra cứu kho kiến thức nội bộ", lines: ["Câu trả lời được lấy từ bộ câu hỏi và kiến thức đã biên soạn; không phải văn bản sinh tự do."] },
      meta: { source: "knowledge", model: "dnai-knowledge", lowConfidence: false },
    };
  }

  if (req.document && req.attachment !== "table") {
    const fileHits = bm25(question, chunk(req.document), 1);
    const best = fileHits[0];
    if (best && best.score >= 0.45) {
      return {
        ok: true,
        content: `Mình tìm thấy đoạn liên quan trong tài liệu${req.document ? " bạn gửi" : ""}:\n\n“${best.text}”\n\nĐây là phần trích từ tệp, chưa phải bản phân tích đầy đủ.`,
        citations: [{ source: best.id, text: best.text }],
        tool: { kind: "extract", title: "Trích đoạn từ tài liệu", lines: ["Đoạn được chọn bằng tìm kiếm BM25 trong nội dung tệp."] },
        meta: { source: "document", model: "dnai-retrieval", lowConfidence: false },
      };
    }
  }

  const table = req.document && (req.attachment === "table" || looksTabular(req.document))
    ? analyzeTable(parseCsv(req.document))
    : null;
  if (table && (req.mode === "data" || /phân tích|thống kê|trung bình|biểu đồ|csv/i.test(question))) {
    return {
      ok: true,
      content: narrate(table),
      citations: [],
      stats: table,
      tool: {
        kind: "stats",
        title: "Công cụ thống kê",
        lines: ["Số liệu tính trực tiếp từ bảng, không phải token mô hình sinh ra."],
      },
      meta: { source: "stats", model: "dnai-stats", lowConfidence: false },
    };
  }

  if (!canUseFreeGeneration()) {
    const related = bm25(question, knowledge.map((a) => ({ id: a.id, body: a.body })), 1)[0];
    const terms = searchTokens(question);
    const relatedTerms = related ? searchTokens(related.text) : [];
    const coverage = terms.length
      ? terms.filter((term) => relatedTerms.includes(term)).length / terms.length
      : 0;
    if (related && related.score >= 0.65 && coverage >= 0.3) {
      return {
        ok: true,
        content: `Thông tin gần nhất trong kho kiến thức nội bộ:\n\n${related.text}\n\nMình chưa có đủ cơ sở để diễn giải sâu hơn câu hỏi này.`,
        citations: [{ source: `Kho kiến thức: ${related.id}`, text: related.text }],
        tool: { kind: "extract", title: "Tra cứu kiến thức liên quan", lines: ["Đã trả về nội dung nguồn thay vì để checkpoint thử nghiệm tự đoán."] },
        meta: { source: "knowledge", model: "dnai-knowledge", lowConfidence: false },
      };
    }
    return {
      ok: true,
      content: "Mình chưa có đủ dữ liệu đáng tin cậy để trả lời câu này. DN-AI Micro hiện đang ở chế độ thử nghiệm nên chức năng tự sinh câu trả lời được tạm khóa để tránh tạo nội dung sai. Bạn có thể hỏi về kiến thức trong kho, phép tính, phương trình hoặc gửi tệp TXT, MD, CSV hay JSON để mình tra cứu.",
      citations: [],
      tool: { kind: "extract", title: "Chưa có dữ liệu phù hợp", lines: ["Không tìm thấy câu trả lời đủ gần trong kho kiến thức.", "Chức năng sinh tự do đang tắt cho đến khi mô hình vượt qua kiểm định chất lượng."] },
      meta: { source: "unavailable", model: checkpoint.name, lowConfidence: true },
    };
  }

  const started = Date.now();
  try {
    const { net: model, tok: vocab } = runtime();
    const articles: Doc[] = knowledge.map((a) => ({ id: a.id, body: a.body }));
    const articleHits = bm25(question, articles, 1);
    const docHits = req.document ? bm25(question, chunk(req.document), 2) : [];
    const knowledgeText = [articleHits[0]?.text, docHits[0]?.text].filter(Boolean).join("\n");
    const history = req.messages.slice(0, -1).slice(-4).map((m) => ({
      role: m.role,
      content: m.content.slice(0, 500),
    }));
    const prompt = buildPrompt(vocab, question.slice(0, 1200), knowledgeText.slice(0, 700), history, model.arch.max_seq);
    const sampled = generateIds(model, prompt, {
      maxNew: Math.min(80, req.maxTokens),
      temperature: req.temperature,
      topK: req.topK,
      topP: req.topP,
      seed: req.seed >>> 0,
      eos: vocab.eos,
    });
    const text = decode(vocab, sampled.ids).trim();
    if (!text || text.includes("\uFFFD") || /<\/?(?:pad|bos|eos|unk|know|user|asst)>/.test(text) || unstableText(text)) {
      return {
        ok: true,
        content: "Mình vừa tạo được một câu trả lời không đủ rõ ràng nên đã bỏ kết quả đó. Bạn hãy diễn đạt câu hỏi ngắn và cụ thể hơn; mình không muốn hiển thị câu trả lời đoán mò.",
        citations: [],
        tool: { kind: "extract", title: "Đã chặn đầu ra không đạt chất lượng", lines: ["Bộ lọc phát hiện văn bản lỗi hoặc lặp bất thường."] },
        meta: { source: "unavailable", model: checkpoint.name, lowConfidence: true, ms: Date.now() - started },
      };
    }
    const lint = req.mode === "coding" ? lintBrackets(question) : [];
    const unstable = unstableText(text);
    const citations: Citation[] = [
      ...articleHits.map((h) => ({ source: `Kho ${h.id}`, text: h.text })),
      ...docHits.map((h) => ({ source: h.id, text: h.text })),
    ];
    return {
      ok: true,
      content: text || "Mô hình không sinh thêm token nào. Thử giảm temperature hoặc hỏi ngắn hơn.",
      citations,
      tool: lint.length
        ? { kind: "lint", title: "Kiểm tra ngoặc (không thực thi code)", lines: lint }
        : undefined,
      meta: {
        source: "micro",
        model: checkpoint.name,
        logprob: round4(sampled.meanLogprob),
        lowConfidence: unstable || sampled.meanLogprob < -2.4,
        promptTokens: prompt.length,
        completionTokens: sampled.ids.length,
        ms: Date.now() - started,
      },
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "inference failed";
    if (message.includes("missing") || message.includes("bad tensor")) {
      return { ok: false, error: "Checkpoint DN-AI chưa sẵn sàng hoặc bị hỏng. Không có câu trả lời thay thế." };
    }
    return { ok: false, error: "Inference lỗi. Không dùng API bên ngoài để thay thế." };
  }
}

function findKnownAnswer(question: string): { id: string; question: string; answer: string } | null {
  const entries = knowledge.flatMap((article) =>
    (article.qa ?? []).map((qa, index) => ({
      id: `${article.id}:${index + 1}`,
      question: qa.question,
      answer: qa.answer,
    })),
  );
  if (!entries.length) return null;
  const hits = bm25(question, entries.map((entry) => ({ id: entry.id, body: entry.question })), 3);
  const terms = searchTokens(question);
  for (const hit of hits) {
    const entry = entries.find((item) => item.id === hit.id);
    if (!entry) continue;
    const matched = searchTokens(entry.question);
    const coverage = terms.length
      ? terms.filter((term) => matched.includes(term)).length / terms.length
      : 0;
    if (hit.score >= 0.45 && coverage >= 0.34) return entry;
  }
  return null;
}

function looksTabular(text: string) {
  const lines = text.split(/\n/).filter((l) => l.trim());
  return lines.length >= 2 && lines.filter((l) => l.includes(",") || l.includes(";") || l.includes("\t")).length >= 2;
}

function chunk(text: string): Doc[] {
  const parts = text.split(/\n{2,}|(?<=\.)\s+/);
  const docs: Doc[] = [];
  let buf = "";
  for (const part of parts) {
    if ((buf + " " + part).length > 420) {
      if (buf.trim()) docs.push({ id: `Đoạn ${docs.length + 1}`, body: buf.trim() });
      buf = part;
    } else buf = buf ? `${buf} ${part}` : part;
    if (docs.length >= 48) break;
  }
  if (buf.trim() && docs.length < 48) docs.push({ id: `Đoạn ${docs.length + 1}`, body: buf.trim() });
  return docs;
}

function narrate(table: TableReport): string {
  const lines = table.columns
    .filter((c) => c.numeric && c.mean !== undefined)
    .map(
      (c) =>
        `${c.name}: trung bình ${round4(c.mean ?? 0)}, trung vị ${round4(c.median ?? 0)}, nhỏ nhất ${round4(c.min ?? 0)}, lớn nhất ${round4(c.max ?? 0)}, độ lệch chuẩn ${round4(c.stdev ?? 0)} trên ${c.count} dòng.`,
    );
  const extra: string[] = [`Bảng có ${table.rows} dòng và ${table.columns.length} cột.`];
  if (table.correlation) {
    extra.push(
      `Tương quan Pearson giữa ${table.correlation.a} và ${table.correlation.b} là ${round4(table.correlation.r)}. Tương quan không có nghĩa là nguyên nhân.`,
    );
  }
  if (table.trend) {
    const dir = table.trend.slope > 0 ? "tăng" : table.trend.slope < 0 ? "giảm" : "đi ngang";
    extra.push(
      `Theo thứ tự dòng, ${table.trend.name} đang ${dir} (hệ số góc ${round4(table.trend.slope)}).`,
    );
  }
  return [...extra, ...lines].join(" ");
}

function lintBrackets(code: string): string[] {
  const open = new Map<string, string>([
    ["(", ")"],
    ["[", "]"],
    ["{", "}"],
  ]);
  const close = new Map<string, string>([
    [")", "("],
    ["]", "["],
    ["}", "{"],
  ]);
  const stack: string[] = [];
  for (const ch of code) {
    if (open.has(ch)) stack.push(ch);
    const want = close.get(ch);
    if (want) {
      if (stack.pop() !== want) return [`Dấu '${ch}' đóng không khớp.`];
    }
  }
  if (!stack.length) return ["Các cặp (), [] và {} trong tin nhắn đang cân."];
  return [`Còn ${stack.length} dấu mở chưa đóng: ${stack.join(" ")}.`];
}

function unstableText(text: string): boolean {
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length < 10) return false;
  const unique = new Set(words.map((w) => w.toLowerCase())).size;
  if (unique / words.length < 0.38) return true;
  const tail = words.slice(-4).join(" ");
  const prev = words.slice(-8, -4).join(" ");
  return tail.length > 0 && tail === prev;
}
