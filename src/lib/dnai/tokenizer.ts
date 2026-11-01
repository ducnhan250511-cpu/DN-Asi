/** Unicode word tokenizer with UTF-8 byte fallback. Must match ai/dnai/tokenizer.py. */

const TOKEN_RE = /[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]/gu;

export const SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<know>", "<user>", "<asst>"] as const;

const NO_SPACE_BEFORE = new Set([...".,!?;:%)]}>»”’…-"]);
const NO_SPACE_AFTER = new Set([..."([{<«“‘-/"]);

export type Tokenizer = {
  tokenToId: Map<string, number>;
  idToToken: string[];
  pad: number;
  bos: number;
  eos: number;
  unk: number;
  know: number;
  user: number;
  asst: number;
};

export function tokenizerFromVocab(vocab: string[]): Tokenizer {
  const tokenToId = new Map<string, number>();
  vocab.forEach((t, i) => tokenToId.set(t, i));
  const id = (t: string) => {
    const v = tokenToId.get(t);
    if (v === undefined) throw new Error(`missing special ${t}`);
    return v;
  };
  return {
    tokenToId,
    idToToken: vocab,
    pad: id("<pad>"),
    bos: id("<bos>"),
    eos: id("<eos>"),
    unk: id("<unk>"),
    know: id("<know>"),
    user: id("<user>"),
    asst: id("<asst>"),
  };
}

function splitTokens(text: string): string[] {
  const norm = text.normalize("NFC").replaceAll("\u00a0", " ");
  return norm.match(TOKEN_RE) ?? [];
}

function byteToken(b: number): string {
  return `<b${b.toString(16).padStart(2, "0")}>`;
}

export function encode(tok: Tokenizer, text: string): number[] {
  const ids: number[] = [];
  for (const piece of splitTokens(text)) {
    const known = tok.tokenToId.get(piece);
    if (known !== undefined) {
      ids.push(known);
      continue;
    }
    const bytes = new TextEncoder().encode(piece);
    for (const b of bytes) {
      const id = tok.tokenToId.get(byteToken(b));
      ids.push(id ?? tok.unk);
    }
  }
  return ids;
}

export function decode(tok: Tokenizer, ids: number[]): string {
  const words: string[] = [];
  const raw: number[] = [];
  const specials = new Set<string>(SPECIALS);
  const flush = () => {
    if (!raw.length) return;
    words.push(new TextDecoder().decode(new Uint8Array(raw)));
    raw.length = 0;
  };
  for (const id of ids) {
    const token = tok.idToToken[id];
    if (!token) continue;
    if (token.startsWith("<b") && token.endsWith(">") && token.length === 5) {
      const n = Number.parseInt(token.slice(2, 4), 16);
      if (!Number.isNaN(n)) {
        raw.push(n);
        continue;
      }
    }
    flush();
    if (specials.has(token)) continue;
    words.push(token);
  }
  flush();
  let out = "";
  for (const token of words) {
    if (!token) continue;
    if (!out) {
      out = token;
      continue;
    }
    const prev = out[out.length - 1] ?? "";
    if (NO_SPACE_BEFORE.has(token) || NO_SPACE_AFTER.has(prev) || token.startsWith("'")) out += token;
    else out += ` ${token}`;
  }
  return out;
}

export function buildPrompt(
  tok: Tokenizer,
  user: string,
  knowledge: string,
  history: { role: "user" | "assistant"; content: string }[],
  maxSeq: number,
): number[] {
  const ids = [tok.bos];
  if (knowledge.trim()) {
    ids.push(tok.know, ...encode(tok, knowledge));
  }
  for (const turn of history) {
    ids.push(turn.role === "user" ? tok.user : tok.asst, ...encode(tok, turn.content));
  }
  ids.push(tok.user, ...encode(tok, user), tok.asst);
  if (ids.length <= maxSeq) return ids;
  const cut = ids.slice(-maxSeq);
  cut[cut.length - 1] = tok.asst;
  return cut;
}
