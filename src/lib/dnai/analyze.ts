/** Deterministic table stats for CSV / JSON rows. */

export type ColumnStat = {
  name: string;
  numeric: boolean;
  count: number;
  mean?: number;
  median?: number;
  min?: number;
  max?: number;
  stdev?: number;
};

export type TableReport = {
  rows: number;
  columns: ColumnStat[];
  correlation?: { a: string; b: string; r: number };
  trend?: { name: string; slope: number; intercept: number };
  labels: string[];
  series: { name: string; data: number[] }[];
};

export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let q = false;
  const src = text.replace(/^\uFEFF/, "");
  for (let i = 0; i < src.length; i++) {
    const c = src[i];
    if (q) {
      if (c === '"') {
        if (src[i + 1] === '"') {
          cell += '"';
          i++;
        } else q = false;
      } else cell += c;
      continue;
    }
    if (c === '"') {
      q = true;
      continue;
    }
    if (c === "," || c === ";" || c === "\t") {
      row.push(cell.trim());
      cell = "";
      continue;
    }
    if (c === "\n") {
      row.push(cell.trim());
      if (row.some((x) => x)) rows.push(row);
      row = [];
      cell = "";
      continue;
    }
    if (c !== "\r") cell += c;
  }
  row.push(cell.trim());
  if (row.some((x) => x)) rows.push(row);
  return rows;
}

export function analyzeTable(matrix: string[][]): TableReport | null {
  if (matrix.length < 2) return null;
  const header = matrix[0] ?? [];
  const body = matrix.slice(1).filter((r) => r.some((c) => c));
  if (!body.length) return null;
  const columns: ColumnStat[] = header.map((name, col) => {
    const nums = body
      .map((r) => Number((r[col] ?? "").replace(",", ".")))
      .filter((n) => Number.isFinite(n));
    const numeric = nums.length >= Math.max(2, Math.floor(body.length * 0.7));
    if (!numeric) return { name: name || `Cột ${col + 1}`, numeric: false, count: body.length };
    const mean = nums.reduce((s, n) => s + n, 0) / nums.length;
    const sorted = [...nums].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 ? (sorted[mid] ?? 0) : ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2;
    const variance = nums.reduce((s, n) => s + (n - mean) ** 2, 0) / nums.length;
    return {
      name: name || `Cột ${col + 1}`,
      numeric: true,
      count: nums.length,
      mean,
      median,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      stdev: Math.sqrt(variance),
    };
  });
  const numericCols = columns
    .map((c, i) => ({ c, i }))
    .filter((x) => x.c.numeric);
  let correlation: TableReport["correlation"];
  if (numericCols.length >= 2) {
    const a = numericCols[0];
    const b = numericCols[1];
    if (a && b) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (const r of body) {
        const x = Number((r[a.i] ?? "").replace(",", "."));
        const y = Number((r[b.i] ?? "").replace(",", "."));
        if (Number.isFinite(x) && Number.isFinite(y)) {
          xs.push(x);
          ys.push(y);
        }
      }
      const r = pearson(xs, ys);
      if (r !== null) correlation = { a: a.c.name, b: b.c.name, r };
    }
  }
  let trend: TableReport["trend"];
  const first = numericCols[0];
  if (first) {
    const ys = body
      .map((r) => Number((r[first.i] ?? "").replace(",", ".")))
      .filter((n) => Number.isFinite(n));
    const xs = ys.map((_, i) => i + 1);
    const fit = linreg(xs, ys);
    if (fit) trend = { name: first.c.name, ...fit };
  }
  const labels = body.slice(0, 24).map((_, i) => String(i + 1));
  const series = numericCols.slice(0, 3).map(({ c, i }) => ({
    name: c.name,
    data: body.slice(0, 24).map((r) => {
      const n = Number((r[i] ?? "").replace(",", "."));
      return Number.isFinite(n) ? n : 0;
    }),
  }));
  return { rows: body.length, columns, correlation, trend, labels, series };
}

function pearson(xs: number[], ys: number[]): number | null {
  if (xs.length < 3 || xs.length !== ys.length) return null;
  const mx = avg(xs);
  const my = avg(ys);
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < xs.length; i++) {
    const a = (xs[i] ?? 0) - mx;
    const b = (ys[i] ?? 0) - my;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  if (dx === 0 || dy === 0) return null;
  return num / Math.sqrt(dx * dy);
}

function linreg(xs: number[], ys: number[]): { slope: number; intercept: number } | null {
  if (xs.length < 2) return null;
  const mx = avg(xs);
  const my = avg(ys);
  let num = 0;
  let den = 0;
  for (let i = 0; i < xs.length; i++) {
    num += ((xs[i] ?? 0) - mx) * ((ys[i] ?? 0) - my);
    den += ((xs[i] ?? 0) - mx) ** 2;
  }
  if (den === 0) return null;
  const slope = num / den;
  return { slope, intercept: my - slope * mx };
}

function avg(xs: number[]) {
  return xs.reduce((s, n) => s + n, 0) / xs.length;
}

export function round4(n: number): number {
  return Math.round(n * 10000) / 10000;
}
