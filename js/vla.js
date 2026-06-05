// The "model": a deterministic, architecture-faithful stand-in for a VLA.
// Parsing/attention/layers are illustrative; the flow-matching head is real
// (it integrates the conditional flow-matching ODE from noise to an action).
import { OBJECTS, objectById, CHUNK_T, LIMITS } from './world.js';

// ---------- tiny seeded RNG ----------
function hashStr(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}
function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randn(rng) {
  const u = 1 - rng(), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = t => t * t * (3 - 2 * t);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// ---------- vocabulary ----------
const COLORS = ['red', 'blue', 'green', 'yellow'];
const NOUNS = { cube: 1, block: 1, box: 1, brick: 1 };
const VERBS = {
  pick: 'grasp', grab: 'grasp', grasp: 'grasp', lift: 'grasp', take: 'grasp',
  get: 'grasp', collect: 'grasp', retrieve: 'grasp', hold: 'grasp',
  point: 'point', touch: 'point', reach: 'point', indicate: 'point',
  push: 'push', nudge: 'push', shove: 'push', slide: 'push',
  move: 'move', go: 'move', shift: 'move',
  open: 'open', release: 'open', drop: 'open',
  close: 'close', grip: 'close', squeeze: 'close',
};
const DIRS = {
  left: [-1, 0, 0], right: [1, 0, 0], up: [0, 1, 0], down: [0, -1, 0],
  forward: [0, 0, 1], back: [0, 0, -1], backward: [0, 0, -1],
};
const STOP = new Set(['the', 'a', 'an', 'to', 'at', 'on', 'of', 'please', 'then',
  'and', 'its', 'it', 'your', 'my', 'that', 'this']);

// ---------- prompt parsing ----------
export function parsePrompt(prompt) {
  const words = prompt.toLowerCase().match(/[a-z]+/g) || [];
  let color = null, noun = null, verb = null, mode = null, dir = null;
  words.forEach((w, i) => {
    if (COLORS.includes(w) && !color) color = w;
    if (NOUNS[w]) noun = w;
    if (VERBS[w] && !verb) { verb = w; mode = VERBS[w]; }
    if (DIRS[w] && !(w === 'up' && VERBS[words[i - 1]] === 'grasp')) dir = w;
  });

  const target = color ? objectById(color) : (noun ? OBJECTS[0] : null);
  if (!mode) mode = target ? 'grasp' : (dir ? 'move' : 'point');
  if ((mode === 'move' || mode === 'point' || mode === 'grasp') && dir && !color && !noun) mode = 'move';

  const gripEnd = mode === 'grasp' || mode === 'close' ? 0
                : mode === 'open' ? 1 : (mode === 'push' ? 0 : 1);

  let conf = 0.4;
  if (verb) conf += 0.3;
  if (target) conf += 0.2;
  if (dir && mode === 'move') conf += 0.1;
  conf = clamp(conf, 0.25, 0.99);

  const label = mode === 'move' && dir ? `move ${dir}`
    : target ? `${mode} → ${target.label}` : mode;

  return { words, color, noun, verb, mode, dir, target, gripEnd, conf, label };
}

// ---------- language tokens ----------
export function tokenize(prompt, intent) {
  const words = prompt.match(/[A-Za-z]+/g) || [];
  return words.map((raw, i) => {
    const w = raw.toLowerCase();
    let type = 'other', act = 0.25;
    if (COLORS.includes(w)) { type = 'color'; act = w === intent.color ? 1.0 : 0.85; }
    else if (NOUNS[w]) { type = 'noun'; act = 0.66; }
    else if (VERBS[w]) { type = 'verb'; act = 0.95; }
    else if (DIRS[w]) {
      const particle = w === 'up' && VERBS[(words[i - 1] || '').toLowerCase()] === 'grasp';
      type = particle ? 'stop' : 'dir';
      act = particle ? 0.18 : (intent.mode === 'move' ? 0.92 : 0.45);
    } else if (STOP.has(w)) { type = 'stop'; act = 0.08; }
    return { text: raw, type, act: act * (0.6 + 0.4 * intent.conf) };
  });
}

// ---------- vision patch grid (8x8) ----------
export const GRID = 8;
export function patchGrid(intent, proj) {
  const n = GRID, g = new Float32Array(n * n);
  const splat = (u, v, amp, sig) => {
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const cu = (c + 0.5) / n, cv = (r + 0.5) / n;
      const d2 = ((cu - u) ** 2 + (cv - v) ** 2) / (sig * sig);
      g[r * n + c] += amp * Math.exp(-d2);
    }
  };
  // bottom-up saliency: every visible object glows a little
  for (const o of OBJECTS) {
    const p = proj[o.id];
    if (p && p.vis) splat(p.u, p.vv, 0.28, 0.10);
  }
  // top-down task attention: the instruction's target lights up strongly
  if (intent.target) {
    const p = proj[intent.target.id];
    if (p && p.vis) splat(p.u, p.vv, 1.0 * intent.conf, 0.085);
  }
  let mx = 1e-3; for (const v of g) mx = Math.max(mx, v);
  for (let i = 0; i < g.length; i++) g[i] = Math.min(1, g[i] / mx);
  return g;
}

// ---------- attention (action queries -> context) ----------
export const N_QUERY = 5;
export function attention(intent, tokens) {
  const content = tokens
    .map((t, i) => ({ ...t, i }))
    .filter(t => t.type !== 'stop' && t.type !== 'other')
    .sort((a, b) => b.act - a.act)
    .slice(0, 4);

  const context = [
    { label: 'img▸target', type: 'vis', base: 0.62 * intent.conf + 0.05 },
    { label: 'img▸scene', type: 'vis', base: 0.14 },
    ...content.map(t => ({ label: t.text, type: t.type, base: 0.5 * t.act })),
    { label: 'state', type: 'state', base: 0.34 },
  ];

  const rng = mulberry32(hashStr(intent.label + '|attn'));
  const W = [];
  for (let q = 0; q < N_QUERY; q++) {
    const f = q / (N_QUERY - 1);               // 0 (first action) .. 1 (last)
    const row = context.map(ctx => {
      let w = ctx.base;
      if (ctx.type === 'state') w *= (1.1 - 0.7 * f);  // early actions lean on current state
      if (ctx.label === 'img▸target') w *= (0.6 + 0.7 * f); // later actions lean on the goal
      w *= 0.8 + 0.4 * rng();
      return Math.max(0.001, w);
    });
    const s = row.reduce((a, b) => a + b, 0);
    W.push(row.map(v => v / s));
  }
  return { context, nQuery: N_QUERY, W };
}

// ---------- layer activations ----------
export const N_LAYER = 12;
export function layerActs(intent) {
  const rng = mulberry32(hashStr(intent.label + '|layers'));
  const out = [];
  for (let l = 0; l < N_LAYER; l++) {
    const x = l / (N_LAYER - 1);
    const bell = Math.exp(-((x - 0.55) ** 2) / 0.10);   // mid-layers most active
    const v = clamp(bell * (0.55 + 0.45 * intent.conf) + (rng() - 0.5) * 0.22, 0.08, 1);
    out.push(v);
  }
  return out;
}

// ---------- task target & action chunk ----------
export function taskTarget(intent, currentTip) {
  if (intent.mode === 'open' || intent.mode === 'close') {
    return { point: { x: currentTip.x, y: currentTip.y, z: currentTip.z },
             gripEnd: intent.gripEnd, highlightId: null, ik: false };
  }
  if (intent.mode === 'move' && intent.dir) {
    const d = DIRS[intent.dir];
    return {
      point: { x: currentTip.x + d[0] * 0.08, y: clamp(currentTip.y + d[1] * 0.08, 0.03, 0.32),
               z: currentTip.z + d[2] * 0.08 },
      gripEnd: 1, highlightId: null, ik: true,
    };
  }
  const o = intent.target || OBJECTS[0];
  const yOff = intent.mode === 'point' ? 0.075 : (intent.mode === 'push' ? 0.004 : 0.012);
  return {
    point: { x: o.pos[0], y: o.pos[1] + yOff, z: o.pos[2] },
    gripEnd: intent.gripEnd, highlightId: o.id, ik: true,
  };
}

// 8-step action chunk: smooth joint trajectory + gripper schedule. Flattened t*6+j.
export function buildChunk(qFrom, qTo, gripEnd, mode) {
  const T = CHUNK_T, out = new Float32Array(T * 6);
  for (let t = 0; t < T; t++) {
    const a = smooth(t / (T - 1));
    for (let j = 0; j < 5; j++) out[t * 6 + j] = lerp(qFrom[j], qTo[j], a);
    // gripper: ease to final state, closing late for a grasp
    const gA = mode === 'grasp' ? smooth(clamp((t / (T - 1) - 0.55) / 0.45, 0, 1)) : a;
    out[t * 6 + 5] = lerp(qFrom[5], gripEnd, gA);
  }
  return out;
}

// ---------- flow matching (the real part) ----------
export const N_PART = 30;
const SIGMA = [0.95, 0.95, 0.95, 0.95, 0.95, 0.4];  // per-joint noise scale

export function makeFlow(chunkTarget, seed) {
  const D = chunkTarget.length;                       // 48
  const rng = mulberry32(hashStr(seed + '|flow'));
  const x0 = [], tgt = [];
  for (let p = 0; p < N_PART; p++) {
    const a = new Float32Array(D), b = new Float32Array(D);
    for (let i = 0; i < D; i++) {
      a[i] = randn(rng) * SIGMA[i % 6];                          // x0 ~ N(0, σ²)
      b[i] = chunkTarget[i] + randn(rng) * 0.035;               // per-sample target (action spread)
    }
    x0.push(a); tgt.push(b);
  }
  // fixed random 2-D projection for the plot
  const P0 = new Float32Array(D), P1 = new Float32Array(D);
  for (let i = 0; i < D; i++) { P0[i] = randn(rng); P1[i] = randn(rng); }
  norm(P0); orth(P1, P0); norm(P1);

  // bounds across the whole trajectory for stable axes
  let xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9;
  for (let p = 0; p < N_PART; p++) for (const tau of [0, 1]) {
    const [px, py] = projAt(x0[p], tgt[p], tau, P0, P1);
    xmin = Math.min(xmin, px); xmax = Math.max(xmax, px);
    ymin = Math.min(ymin, py); ymax = Math.max(ymax, py);
  }
  return { D, x0, tgt, P0, P1, target: chunkTarget, bounds: { xmin, xmax, ymin, ymax } };
}
function norm(v) { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; for (let i = 0; i < v.length; i++) v[i] /= s; }
function orth(v, u) { let d = 0; for (let i = 0; i < v.length; i++) d += v[i] * u[i]; for (let i = 0; i < v.length; i++) v[i] -= d * u[i]; }
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function projAt(x0, tgt, tau, P0, P1) {
  // rectified-flow path x_τ = (1-τ)·x0 + τ·target
  let a = 0, b = 0;
  for (let i = 0; i < x0.length; i++) { const x = (1 - tau) * x0[i] + tau * tgt[i]; a += x * P0[i]; b += x * P1[i]; }
  return [a, b];
}

// particle screen positions + per-particle velocity direction at τ
export function flowAt(flow, tau) {
  const pts = [];
  for (let p = 0; p < N_PART; p++) {
    const [x, y] = projAt(flow.x0[p], flow.tgt[p], tau, flow.P0, flow.P1);
    // velocity v = target - x0  (constant for rectified flow) -> project
    let vx = 0, vy = 0;
    for (let i = 0; i < flow.D; i++) { const v = flow.tgt[p][i] - flow.x0[p][i]; vx += v * flow.P0[i]; vy += v * flow.P1[i]; }
    pts.push({ x, y, vx, vy });
  }
  const [tx, ty] = projAt(flow.target, flow.target, 1, flow.P0, flow.P1);
  return { pts, target: { x: tx, y: ty } };
}

// mean denoised chunk at τ -> [T][6]
export function meanChunkAt(flow, tau) {
  const D = flow.D, mean = new Float32Array(D);
  for (let p = 0; p < N_PART; p++)
    for (let i = 0; i < D; i++) mean[i] += ((1 - tau) * flow.x0[p][i] + tau * flow.tgt[p][i]) / N_PART;
  const T = CHUNK_T, out = [];
  for (let t = 0; t < T; t++) out.push(Array.from(mean.subarray(t * 6, t * 6 + 6)));
  return out;
}

// one sampled trajectory's denoised chunk at τ -> [T][6]
// (a single noise seed integrated to an action; dramatic at τ=0, exact at τ=1)
export function sampleChunkAt(flow, tau, p = 0) {
  const out = [];
  for (let t = 0; t < CHUNK_T; t++) {
    const row = [];
    for (let j = 0; j < 6; j++) { const i = t * 6 + j; row.push((1 - tau) * flow.x0[p][i] + tau * flow.tgt[p][i]); }
    out.push(row);
  }
  return out;
}

export { LIMITS };
