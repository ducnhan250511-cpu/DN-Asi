/** BM25 over short original articles or uploaded chunks. */

export type Doc = { id: string; body: string };

export type Hit = { id: string; score: number; text: string };

const WORD = /[\p{L}\p{N}_]+/gu;
const STOP_WORDS = new Set([
  "la", "là", "gi", "gì", "co", "có", "va", "và", "cua", "của", "cho", "mot", "một",
  "nhung", "những", "cac", "các", "the", "thế", "nao", "nào", "hay", "hãy", "nêu",
  "giup", "giúp", "minh", "mình", "ban", "bạn", "toi", "tôi", "voi", "với", "duoc", "được",
  "nay", "này", "do", "đó", "ve", "về", "ra", "sao", "la gi", "please", "what", "is", "are",
  "the", "a", "an", "and", "of", "to", "for", "how", "do", "does", "you", "your", "me",
]);

export function searchTokens(text: string): string[] {
  return (text.normalize("NFC").toLowerCase().match(WORD) ?? [])
    .filter((w) => w.length > 1 && !STOP_WORDS.has(w));
}

export function bm25(query: string, docs: Doc[], k = 2): Hit[] {
  const q = searchTokens(query);
  if (!q.length || !docs.length) return [];
  const tokenized = docs.map((d) => searchTokens(d.body));
  const df = new Map<string, number>();
  for (const words of tokenized) {
    for (const w of new Set(words)) df.set(w, (df.get(w) ?? 0) + 1);
  }
  const avg = tokenized.reduce((s, w) => s + w.length, 0) / tokenized.length;
  const n = docs.length;
  const k1 = 1.2;
  const b = 0.75;
  const scored = docs.map((doc, i) => {
    const words = tokenized[i] ?? [];
    const tf = new Map<string, number>();
    for (const w of words) tf.set(w, (tf.get(w) ?? 0) + 1);
    let score = 0;
    for (const term of q) {
      const f = tf.get(term) ?? 0;
      if (!f) continue;
      const dfi = df.get(term) ?? 0;
      const idf = Math.log(1 + (n - dfi + 0.5) / (dfi + 0.5));
      score += (idf * (f * (k1 + 1))) / (f + k1 * (1 - b + b * (words.length / (avg || 1))));
    }
    return { id: doc.id, score, text: doc.body };
  });
  return scored
    .filter((h) => h.score > 0)
    .sort((a, c) => c.score - a.score)
    .slice(0, k);
}
