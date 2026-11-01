/** Exact arithmetic and small algebra. Not the language model. */

export type SolveResult = {
  kind: "value" | "linear" | "quadratic" | "percent";
  lines: string[];
  answer: string;
};

const FN = new Set(["sqrt", "abs"]);

type Tok =
  | { t: "n"; v: number }
  | { t: "id"; v: string }
  | { t: "op"; v: string }
  | { t: "lp" }
  | { t: "rp" };

export function trySolve(raw: string): SolveResult | null {
  const percent = raw.match(/(-?\d+(?:[.,]\d+)?)\s*%\s*(?:của|cua|of)\s*(-?\d+(?:[.,]\d+)?)/i);
  if (percent) {
    const p = num(percent[1] ?? "0");
    const n = num(percent[2] ?? "0");
    const v = (n * p) / 100;
    return {
      kind: "percent",
      answer: fmt(v),
      lines: [
        `${fmt(p)}% của ${fmt(n)} = ${fmt(n)} × ${fmt(p)} / 100`,
        `= ${fmt(v)}`,
      ],
    };
  }
  let text = raw
    .normalize("NFC")
    .toLowerCase()
    .replaceAll("π", "pi")
    .replaceAll("×", "*")
    .replaceAll("·", "*")
    .replaceAll("−", "-")
    .replaceAll("²", "^2")
    .replaceAll(":", " / ")
    .replace(/căn\s*bậc\s*hai\s*(của|cua)?/g, "sqrt")
    .replace(/căn\s+/g, "sqrt ");
  text = text.replace(/^(hãy\s+)?(tính|giai|giải|calc|calculate|compute|solve)\s+/i, "");
  text = text.replace(/\s*\?+\s*$/g, "");
  text = text.replace(/\s*(bằng bao nhiêu|bằng mấy|là bao nhiêu|ra sao|giúp mình|giúp tôi|nhé|please)\s*$/gi, "");
  text = text.replace(/\s+/g, "");
  if (!/[0-9x]/.test(text)) return null;
  if (!/^[0-9xpieabsqrt+\-*/^().,=]+$/.test(text)) return null;
  if ((text.match(/=/g) ?? []).length > 1) return null;

  if (text.includes("=")) {
    const [l, r] = text.split("=");
    if (!l || r === undefined) return null;
    if (!l.includes("x") && !r.includes("x")) {
      const v = diffValue(l, r, 0);
      if (v === null) return null;
      return { kind: "value", answer: fmt(v), lines: [`${pretty(l)} = ${pretty(r)}`, `Hiệu hai vế = ${fmt(v)}`, "Hai vế không bằng nhau trừ khi hiệu bằng 0."] };
    }
    return solveEquation(`${l}-(${r})`);
  }
  if (text.includes("x")) return null;
  const v = evalAt(text, 0);
  if (v === null) return null;
  return { kind: "value", answer: fmt(v), lines: [pretty(text), `= ${fmt(v)}`] };
}

function solveEquation(expr: string): SolveResult | null {
  const f0 = evalAt(expr, 0);
  const f1 = evalAt(expr, 1);
  const f2 = evalAt(expr, 2);
  const f3 = evalAt(expr, 3);
  if (f0 === null || f1 === null || f2 === null || f3 === null) return null;
  const d2 = f2 - 2 * f1 + f0;
  const d3 = f3 - 2 * f2 + f1;
  const nonlinear = Math.abs(d2 - d3) > 1e-6 || Math.abs(d2) > 1e-8;
  if (!nonlinear) {
    const b = f0;
    const a = f1 - f0;
    if (Math.abs(a) < 1e-9) {
      return {
        kind: "linear",
        answer: Math.abs(b) < 1e-8 ? "vô số nghiệm" : "vô nghiệm",
        lines: [
          "Phương trình bậc nhất sau khi rút gọn.",
          Math.abs(b) < 1e-8 ? "Hai vế trùng nhau." : "Hệ số của x bằng 0 nhưng hai vế khác nhau.",
        ],
      };
    }
    const x = -b / a;
    return {
      kind: "linear",
      answer: `x = ${fmt(x)}`,
      lines: [
        "Đưa về ax + b = 0.",
        `a = ${fmt(a)}, b = ${fmt(b)}`,
        `x = -b / a = ${fmt(x)}`,
      ],
    };
  }
  const a = d2 / 2;
  const b = f1 - f0 - a;
  const c = f0;
  const disc = b * b - 4 * a * c;
  const lines = [
    "Đưa về ax² + bx + c = 0.",
    `a = ${fmt(a)}, b = ${fmt(b)}, c = ${fmt(c)}`,
    `Δ = b² - 4ac = ${fmt(disc)}`,
  ];
  if (disc < -1e-8) {
    return { kind: "quadratic", answer: "vô nghiệm thực", lines: [...lines, "Delta âm, không có nghiệm thực."] };
  }
  if (Math.abs(disc) <= 1e-8) {
    const x = -b / (2 * a);
    return { kind: "quadratic", answer: `x = ${fmt(x)}`, lines: [...lines, `Nghiệm kép x = -b / (2a) = ${fmt(x)}`] };
  }
  const s = Math.sqrt(disc);
  const x1 = (-b - s) / (2 * a);
  const x2 = (-b + s) / (2 * a);
  const lo = Math.min(x1, x2);
  const hi = Math.max(x1, x2);
  return {
    kind: "quadratic",
    answer: `x = ${fmt(lo)} hoặc x = ${fmt(hi)}`,
    lines: [...lines, `x = (-b ± √Δ) / (2a)`, `x₁ = ${fmt(lo)}, x₂ = ${fmt(hi)}`],
  };
}

function diffValue(l: string, r: string, x: number): number | null {
  const a = evalAt(l, x);
  const b = evalAt(r, x);
  if (a === null || b === null) return null;
  return a - b;
}

function evalAt(expr: string, x: number): number | null {
  try {
    const tokens = tokenize(expr);
    const p = new Parser(tokens);
    const v = p.parse(x);
    if (!p.done() || !Number.isFinite(v)) return null;
    return v;
  } catch {
    return null;
  }
}

function tokenize(expr: string): Tok[] {
  const raw: Tok[] = [];
  let i = 0;
  while (i < expr.length) {
    const c = expr[i] ?? "";
    if ((c >= "0" && c <= "9") || c === ".") {
      let j = i + 1;
      while (j < expr.length && (((expr[j] ?? "") >= "0" && (expr[j] ?? "") <= "9") || expr[j] === ".")) j++;
      raw.push({ t: "n", v: num(expr.slice(i, j)) });
      i = j;
      continue;
    }
    if (c >= "a" && c <= "z") {
      let j = i + 1;
      while (j < expr.length && (expr[j] ?? "") >= "a" && (expr[j] ?? "") <= "z") j++;
      raw.push({ t: "id", v: expr.slice(i, j) });
      i = j;
      continue;
    }
    if ("+-*/^".includes(c)) raw.push({ t: "op", v: c });
    else if (c === "(") raw.push({ t: "lp" });
    else if (c === ")") raw.push({ t: "rp" });
    else throw new Error("bad");
    i++;
  }
  const out: Tok[] = [];
  for (let k = 0; k < raw.length; k++) {
    const cur = raw[k];
    const prev = out[out.length - 1];
    if (cur && prev && implicit(prev, cur)) out.push({ t: "op", v: "*" });
    if (cur) out.push(cur);
  }
  return out;
}

function implicit(prev: Tok, cur: Tok): boolean {
  const prevAtom = prev.t === "n" || prev.t === "rp" || (prev.t === "id" && !FN.has(prev.v));
  const curAtom = cur.t === "n" || cur.t === "lp" || cur.t === "id";
  if (!prevAtom || !curAtom) return false;
  if (prev.t === "id" && FN.has(prev.v) && cur.t === "lp") return false;
  return true;
}

class Parser {
  private i = 0;
  private readonly toks: Tok[];
  constructor(toks: Tok[]) {
    this.toks = toks;
  }
  done() {
    return this.i >= this.toks.length;
  }
  parse(x: number) {
    return this.expr(x);
  }
  private peek() {
    return this.toks[this.i];
  }
  private eat() {
    return this.toks[this.i++];
  }
  private expr(x: number): number {
    let v = this.term(x);
    for (;;) {
      const p = this.peek();
      if (p?.t === "op" && (p.v === "+" || p.v === "-")) {
        this.eat();
        const r = this.term(x);
        v = p.v === "+" ? v + r : v - r;
      } else break;
    }
    return v;
  }
  private term(x: number): number {
    let v = this.pow(x);
    for (;;) {
      const p = this.peek();
      if (p?.t === "op" && (p.v === "*" || p.v === "/")) {
        this.eat();
        const r = this.pow(x);
        v = p.v === "*" ? v * r : v / r;
      } else break;
    }
    return v;
  }
  private pow(x: number): number {
    const base = this.unary(x);
    const p = this.peek();
    if (p?.t === "op" && p.v === "^") {
      this.eat();
      return base ** this.pow(x);
    }
    return base;
  }
  private unary(x: number): number {
    const p = this.peek();
    if (p?.t === "op" && p.v === "-") {
      this.eat();
      return -this.unary(x);
    }
    if (p?.t === "op" && p.v === "+") {
      this.eat();
      return this.unary(x);
    }
    return this.primary(x);
  }
  private primary(x: number): number {
    const p = this.eat();
    if (!p) throw new Error("end");
    if (p.t === "n") return p.v;
    if (p.t === "lp") {
      const v = this.expr(x);
      const rp = this.eat();
      if (rp?.t !== "rp") throw new Error("paren");
      return v;
    }
    if (p.t === "id") {
      if (p.v === "pi") return Math.PI;
      if (p.v === "e") return Math.E;
      if (p.v === "x") return x;
      if (FN.has(p.v)) {
        const lp = this.eat();
        if (lp?.t !== "lp") throw new Error("call");
        const v = this.expr(x);
        const rp = this.eat();
        if (rp?.t !== "rp") throw new Error("call");
        if (p.v === "sqrt") return Math.sqrt(v);
        return Math.abs(v);
      }
    }
    throw new Error("primary");
  }
}

function num(s: string): number {
  return Number(s.replace(",", "."));
}

function fmt(n: number): string {
  if (!Number.isFinite(n)) return "không xác định";
  const r = Math.round(n * 1e8) / 1e8;
  if (Object.is(r, -0)) return "0";
  return String(r);
}

function pretty(expr: string): string {
  return expr.replaceAll("*", "×").replaceAll("/", "÷").replaceAll("sqrt", "√");
}
