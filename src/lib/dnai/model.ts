/** Causal Transformer inference. Numerics match ai/dnai/model.py (RoPE, RMSNorm, GELU-tanh). */

export type Arch = {
  vocab_size: number;
  d_model: number;
  n_layer: number;
  n_head: number;
  d_ff: number;
  max_seq: number;
  rope_theta: number;
};

export type Weights = Record<string, Float32Array>;

const GELU_K = 0.7978845608028654;

export function float16ToFloat32(bits: number): number {
  const sign = (bits & 0x8000) >> 15;
  const exp = (bits & 0x7c00) >> 10;
  const frac = bits & 0x03ff;
  if (exp === 0) {
    return (sign ? -1 : 1) * 2 ** -14 * (frac / 1024);
  }
  if (exp === 31) {
    return frac ? Number.NaN : sign ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;
  }
  return (sign ? -1 : 1) * 2 ** (exp - 15) * (1 + frac / 1024);
}

export function unpackF16(b64: string): Float32Array {
  const bytes = Buffer.from(b64, "base64");
  const out = new Float32Array(bytes.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = float16ToFloat32(bytes.readUInt16LE(i * 2));
  }
  return out;
}

function lin(h: Float32Array, w: Float32Array, inD: number, outD: number, dst: Float32Array) {
  dst.fill(0);
  for (let k = 0; k < inD; k++) {
    const hk = h[k] ?? 0;
    const row = k * outD;
    for (let j = 0; j < outD; j++) dst[j] = (dst[j] ?? 0) + hk * (w[row + j] ?? 0);
  }
}

function rms(x: Float32Array, weight: Float32Array, dst: Float32Array) {
  let ms = 0;
  for (let i = 0; i < x.length; i++) ms += (x[i] ?? 0) * (x[i] ?? 0);
  ms /= x.length;
  const inv = 1 / Math.sqrt(ms + 1e-5);
  for (let i = 0; i < x.length; i++) dst[i] = (x[i] ?? 0) * inv * (weight[i] ?? 0);
}

function geluInto(x: Float32Array) {
  for (let i = 0; i < x.length; i++) {
    const v = x[i] ?? 0;
    const u = GELU_K * (v + 0.044715 * v * v * v);
    x[i] = 0.5 * v * (1 + Math.tanh(u));
  }
}

function ropeHead(vec: Float32Array, offset: number, dh: number, pos: number, theta: number) {
  const half = dh / 2;
  const x1 = vec.slice(offset, offset + half);
  const x2 = vec.slice(offset + half, offset + dh);
  for (let i = 0; i < half; i++) {
    const ang = pos / theta ** (i / half);
    const c = Math.cos(ang);
    const s = Math.sin(ang);
    vec[offset + i] = (x1[i] ?? 0) * c - (x2[i] ?? 0) * s;
    vec[offset + half + i] = (x1[i] ?? 0) * s + (x2[i] ?? 0) * c;
  }
}

export class MicroNet {
  readonly dHead: number;
  private readonly posLimit: number;
  private pos = 0;
  private readonly k: Float32Array[];
  private readonly v: Float32Array[];

  constructor(
    readonly arch: Arch,
    readonly weights: Weights,
  ) {
    if (arch.d_model % arch.n_head !== 0) throw new Error("bad head split");
    this.dHead = arch.d_model / arch.n_head;
    this.posLimit = arch.max_seq;
    this.k = [];
    this.v = [];
    for (let i = 0; i < arch.n_layer; i++) {
      this.k.push(new Float32Array(this.posLimit * arch.d_model));
      this.v.push(new Float32Array(this.posLimit * arch.d_model));
    }
  }

  reset() {
    this.pos = 0;
  }

  /** Push one token. Returns logits over the vocabulary. */
  push(id: number): Float32Array {
    const { arch, weights } = this;
    const d = arch.d_model;
    if (this.pos >= this.posLimit) {
      throw new Error("context-full");
    }
    const embed = weights.embed;
    if (!embed) throw new Error("missing embed");
    let x = embed.slice(id * d, id * d + d);
    if (x.length !== d) throw new Error("token id out of range");
    const h = new Float32Array(d);
    for (let layer = 0; layer < arch.n_layer; layer++) {
      const p = `${layer}.`;
      rms(x, must(weights, p + "norm1"), h);
      const q = new Float32Array(d);
      const kk = new Float32Array(d);
      const vv = new Float32Array(d);
      lin(h, must(weights, p + "wq"), d, d, q);
      lin(h, must(weights, p + "wk"), d, d, kk);
      lin(h, must(weights, p + "wv"), d, d, vv);
      for (let head = 0; head < arch.n_head; head++) {
        const off = head * this.dHead;
        ropeHead(q, off, this.dHead, this.pos, arch.rope_theta);
        ropeHead(kk, off, this.dHead, this.pos, arch.rope_theta);
      }
      const cacheK = this.k[layer];
      const cacheV = this.v[layer];
      if (!cacheK || !cacheV) throw new Error("cache");
      cacheK.set(kk, this.pos * d);
      cacheV.set(vv, this.pos * d);
      const attn = this.attend(q, cacheK, cacheV, this.pos);
      const proj = new Float32Array(d);
      lin(attn, must(weights, p + "wo"), d, d, proj);
      for (let i = 0; i < d; i++) x[i] = (x[i] ?? 0) + (proj[i] ?? 0);
      rms(x, must(weights, p + "norm2"), h);
      const ff = new Float32Array(arch.d_ff);
      lin(h, must(weights, p + "w1"), d, arch.d_ff, ff);
      geluInto(ff);
      const ffOut = new Float32Array(d);
      lin(ff, must(weights, p + "w2"), arch.d_ff, d, ffOut);
      for (let i = 0; i < d; i++) x[i] = (x[i] ?? 0) + (ffOut[i] ?? 0);
    }
    rms(x, must(weights, "final_norm"), h);
    const logits = new Float32Array(arch.vocab_size);
    for (let v = 0; v < arch.vocab_size; v++) {
      let s = 0;
      const row = v * d;
      for (let i = 0; i < d; i++) s += (h[i] ?? 0) * (embed[row + i] ?? 0);
      logits[v] = s;
    }
    this.pos += 1;
    return logits;
  }

  private attend(q: Float32Array, k: Float32Array, v: Float32Array, pos: number): Float32Array {
    const d = this.arch.d_model;
    const dh = this.dHead;
    const heads = this.arch.n_head;
    const scale = 1 / Math.sqrt(dh);
    const out = new Float32Array(d);
    const scores = new Float32Array(pos + 1);
    for (let head = 0; head < heads; head++) {
      const off = head * dh;
      let max = -1e30;
      for (let s = 0; s <= pos; s++) {
        let dot = 0;
        const base = s * d + off;
        for (let i = 0; i < dh; i++) dot += (q[off + i] ?? 0) * (k[base + i] ?? 0);
        const sc = dot * scale;
        scores[s] = sc;
        if (sc > max) max = sc;
      }
      let sum = 0;
      for (let s = 0; s <= pos; s++) {
        const e = Math.exp((scores[s] ?? 0) - max);
        scores[s] = e;
        sum += e;
      }
      for (let i = 0; i < dh; i++) {
        let acc = 0;
        for (let s = 0; s <= pos; s++) {
          acc += ((scores[s] ?? 0) / sum) * (v[s * d + off + i] ?? 0);
        }
        out[off + i] = acc;
      }
    }
    return out;
  }
}

function must(w: Weights, key: string): Float32Array {
  const t = w[key];
  if (!t) throw new Error(`missing tensor ${key}`);
  return t;
}

export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function sampleId(
  logits: Float32Array,
  rand: () => number,
  temperature: number,
  topK: number,
  topP: number,
  recent: number[],
  repetitionPenalty = 1.12,
): { id: number; logprob: number } {
  const x = Float32Array.from(logits);
  if (repetitionPenalty !== 1 && recent.length) {
    const seen = new Set(recent.slice(-24));
    for (const id of seen) {
      const v = x[id];
      if (v === undefined) continue;
      x[id] = v > 0 ? v / repetitionPenalty : v * repetitionPenalty;
    }
  }
  const temp = Math.max(temperature, 1e-4);
  for (let i = 0; i < x.length; i++) x[i] = (x[i] ?? 0) / temp;
  if (topK > 0 && topK < x.length) {
    const copy = Array.from(x);
    copy.sort((a, b) => b - a);
    const cut = copy[topK - 1] ?? -1e9;
    for (let i = 0; i < x.length; i++) if ((x[i] ?? 0) < cut) x[i] = -1e9;
  }
  if (topP > 0 && topP < 1) {
    const order = x.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v);
    const probs = softmax(Float32Array.from(order.map((o) => o.v)));
    let cum = 0;
    let keep = 1;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i] ?? 0;
      keep = i + 1;
      if (cum >= topP) break;
    }
    const allowed = new Set(order.slice(0, keep).map((o) => o.i));
    for (let i = 0; i < x.length; i++) if (!allowed.has(i)) x[i] = -1e9;
  }
  const probs = softmax(x);
  let id = 0;
  if (temperature < 1e-3) {
    let best = -1e30;
    for (let i = 0; i < probs.length; i++) {
      if ((probs[i] ?? 0) > best) {
        best = probs[i] ?? 0;
        id = i;
      }
    }
  } else {
    let r = rand();
    let acc = 0;
    id = probs.length - 1;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i] ?? 0;
      if (r <= acc) {
        id = i;
        break;
      }
    }
  }
  return { id, logprob: Math.log((probs[id] ?? 0) + 1e-9) };
}

function softmax(logits: Float32Array): Float32Array {
  let max = -1e30;
  for (const v of logits) if (v > max) max = v;
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp((logits[i] ?? 0) - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) / sum;
  return out;
}

export function generateIds(
  net: MicroNet,
  prompt: number[],
  opts: {
    maxNew: number;
    temperature: number;
    topK: number;
    topP: number;
    seed: number;
    eos: number;
  },
): { ids: number[]; meanLogprob: number } {
  net.reset();
  let logits = new Float32Array(net.arch.vocab_size);
  for (const id of prompt) logits = net.push(id);
  const rand = mulberry32(opts.seed);
  const ids: number[] = [];
  let sum = 0;
  const recent = prompt.slice();
  for (let n = 0; n < opts.maxNew; n++) {
    const picked = sampleId(logits, rand, opts.temperature, opts.topK, opts.topP, recent);
    ids.push(picked.id);
    recent.push(picked.id);
    sum += picked.logprob;
    if (picked.id === opts.eos) break;
    if (prompt.length + ids.length >= net.arch.max_seq) break;
    logits = net.push(picked.id);
  }
  return { ids, meanLogprob: ids.length ? sum / ids.length : 0 };
}
