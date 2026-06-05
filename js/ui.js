// Panel rendering: tokens, patch overlay, attention, layers, flow plot, chunk.
import { GRID, N_LAYER, flowAt, LIMITS } from './vla.js';
import { JOINTS } from './world.js';

const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const palette = () => ({
  vision: cssVar('--accent-vision') || '#7aa2ff',
  language: cssVar('--accent-language') || '#d6a85c',
  state: cssVar('--accent-state') || '#7bc7aa',
  backbone: cssVar('--accent-backbone') || '#b89cff',
  activation: cssVar('--accent-activation') || '#f0849b',
  flow: cssVar('--accent-flow') || '#70c8e8',
  action: cssVar('--accent-action') || '#ead66b',
  line: cssVar('--line') || '#151515',
});
const typeColor = type => {
  const p = palette();
  return {
    verb: p.language, color: p.action, noun: p.vision, dir: p.flow,
    vis: p.vision, state: p.state, stop: '#555555', other: '#888888',
  }[type] || '#888888';
};

// ---- canvas helper: logical size = CSS layout size, backing = ×dpr.
// We read clientWidth/Height (layout-driven), never the width attribute, so
// resizing the backing store can't feed back into the logical size.
// Requires every 2-D canvas to have an explicit CSS width AND height.
function ctx2d(canvas) {
  const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
  if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh; }
  const c = canvas.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);
  return { c, w, h };
}
const round = (c, x, y, w, h, r) => {
  c.beginPath();
  c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r); c.closePath();
};

// ---- tooltip singleton
let tipEl;
function tip(html, x, y) {
  if (!tipEl) { tipEl = document.createElement('div'); tipEl.className = 'c-tooltip tooltip'; document.body.appendChild(tipEl); }
  if (!html) { tipEl.style.opacity = 0; return; }
  tipEl.innerHTML = html;
  tipEl.style.left = Math.min(x + 14, innerWidth - 240) + 'px';
  tipEl.style.top = (y + 14) + 'px';
  tipEl.style.opacity = 1;
}

// ---- active-panel glow
export function setActive(stage) {
  document.querySelectorAll('.panel').forEach(p =>
    p.classList.toggle('active', stage === 'all' || p.dataset.stage === stage));
}

// ---- language tokens
export function renderTokens(tokens) {
  const host = document.getElementById('lang-tokens');
  host.innerHTML = '';
  for (const t of tokens) {
    const col = typeColor(t.type);
    const el = document.createElement('div');
    el.className = 'tok' + (t.act > 0.5 ? ' hot' : '');
    el.style.borderColor = t.act > 0.5 ? col : '';
    el.innerHTML =
      `<span class="tok-text" style="color:${t.act > 0.3 ? col : '#777777'}">${t.text}</span>
       <div class="tok-bar"><i style="width:${(t.act * 100).toFixed(0)}%;background:${col}"></i></div>`;
    el.addEventListener('mousemove', e =>
      tip(`<b>${t.text}</b> · ${t.type}<br>drive = ${(t.act).toFixed(2)}`, e.clientX, e.clientY));
    el.addEventListener('mouseleave', () => tip(null));
    host.appendChild(el);
  }
}

// ---- patch overlay over the camera image
export function renderPatches(grid) {
  const cv = document.getElementById('patch-overlay');
  const { c, w, h } = ctx2d(cv);
  const cw = w / GRID, ch = h / GRID;
  for (let r = 0; r < GRID; r++) for (let col = 0; col < GRID; col++) {
    const a = grid[r * GRID + col];
    c.fillStyle = hexA(palette().vision, a * 0.42);
    c.fillRect(col * cw, r * ch, cw, ch);
    c.strokeStyle = hexA(palette().vision, 0.08 + a * 0.34);
    c.lineWidth = 1; c.strokeRect(col * cw + .5, r * ch + .5, cw - 1, ch - 1);
    if (a > 0.6) {
      c.strokeStyle = hexA(palette().action, (a - 0.6) * 1.2);
      c.lineWidth = 1.6; c.strokeRect(col * cw + 1, r * ch + 1, cw - 2, ch - 2);
    }
  }
  cv.onmousemove = e => {
    const rect = cv.getBoundingClientRect();
    const col = Math.floor((e.clientX - rect.left) / rect.width * GRID);
    const r = Math.floor((e.clientY - rect.top) / rect.height * GRID);
    if (col < 0 || r < 0 || col >= GRID || r >= GRID) return;
    tip(`patch [${r},${col}]<br>activation = <b>${grid[r * GRID + col].toFixed(2)}</b>`, e.clientX, e.clientY);
  };
  cv.onmouseleave = () => tip(null);
}

// ---- attention: action queries (bottom) -> context (top)
export function renderAttention(attn) {
  const cv = document.getElementById('attn-canvas');
  const { c, w, h } = ctx2d(cv);
  const ctxN = attn.context, nq = attn.nQuery;
  const topY = 26, botY = h - 22;
  const xs = (n, i) => 22 + (w - 44) * (n === 1 ? 0.5 : i / (n - 1));
  const ctxX = i => xs(ctxN.length, i);
  const qX = i => xs(nq, i);

  // links
  for (let q = 0; q < nq; q++) for (let k = 0; k < ctxN.length; k++) {
    const wt = attn.W[q][k];
    if (wt < 0.04) continue;
    const col = typeColor(ctxN[k].type);
    c.strokeStyle = hexA(col, 0.12 + wt * 0.7);
    c.lineWidth = 0.5 + wt * 5;
    const x1 = qX(q), x2 = ctxX(k), my = (topY + botY) / 2;
    c.beginPath(); c.moveTo(x1, botY - 8);
    c.bezierCurveTo(x1, my, x2, my, x2, topY + 8);
    c.stroke();
  }
  // context nodes
  c.font = '600 9.5px ui-monospace, monospace'; c.textAlign = 'center';
  for (let k = 0; k < ctxN.length; k++) {
    const x = ctxX(k), col = typeColor(ctxN[k].type);
    nodeChip(c, x, topY, ctxN[k].label, col);
  }
  // query nodes
  for (let q = 0; q < nq; q++) {
    nodeChip(c, qX(q), botY, `a${q}`, palette().action);
  }
  c.fillStyle = palette().backbone; c.textAlign = 'left'; c.font = '9px ui-monospace, monospace';
  c.fillText('context tokens', 4, 11);
  c.textAlign = 'right'; c.fillStyle = palette().action; c.fillText('action queries', w - 4, h - 4);
}
function nodeChip(c, x, y, label, col) {
  const tw = c.measureText(label).width + 12, hh = 16;
  round(c, x - tw / 2, y - hh / 2, tw, hh, 5);
  c.fillStyle = hexA(col, 0.16); c.fill();
  c.strokeStyle = hexA(col, 0.8); c.lineWidth = 1; c.stroke();
  c.fillStyle = col; c.textAlign = 'center'; c.fillText(label, x, y + 3.3);
}

// ---- layer activation strips
export function renderLayers(acts) {
  const host = document.getElementById('layers');
  host.innerHTML = '';
  for (let l = 0; l < N_LAYER; l++) {
    const col = document.createElement('div');
    col.className = 'layer-col';
    col.innerHTML = `<i style="height:${(acts[l] * 100).toFixed(0)}%"></i><span>${l}</span>`;
    col.addEventListener('mousemove', e =>
      tip(`layer ${l}<br>activation = <b>${acts[l].toFixed(2)}</b>`, e.clientX, e.clientY));
    col.addEventListener('mouseleave', () => tip(null));
    host.appendChild(col);
  }
}

// ---- flow-matching particle plot: a Gaussian cloud flowing to the action a*
export function renderFlow(flow, tau, sel = -1) {
  const cv = document.getElementById('flow-canvas');
  const { c, w, h } = ctx2d(cv);
  const b = flow.bounds;
  const ex = (b.xmax - b.xmin) * 0.14 || 1, ey = (b.ymax - b.ymin) * 0.14 || 1;
  const xmin = b.xmin - ex, xmax = b.xmax + ex, ymin = b.ymin - ey, ymax = b.ymax + ey;
  const pad = 12;
  const sx = v => pad + (v - xmin) / (xmax - xmin) * (w - 2 * pad);
  const sy = v => pad + (v - ymin) / (ymax - ymin) * (h - 2 * pad);

  // backing + faint grid
  const pal = palette();
  c.fillStyle = '#050505'; round(c, 3, 3, w - 6, h - 6, 3); c.fill();
  c.strokeStyle = pal.line; c.lineWidth = 1;
  for (let i = 1; i < 6; i++) {
    const gx = pad + (w - 2 * pad) * i / 6, gy = pad + (h - 2 * pad) * i / 6;
    c.beginPath(); c.moveTo(gx, pad); c.lineTo(gx, h - pad); c.stroke();
    c.beginPath(); c.moveTo(pad, gy); c.lineTo(w - pad, gy); c.stroke();
  }

  const start = flowAt(flow, 0).pts;
  const now = flowAt(flow, tau);
  const tgt = now.target, tsx = sx(tgt.x), tsy = sy(tgt.y);

  // learned velocity field: faint arrows converging on the action point
  for (let i = 0; i <= 6; i++) for (let j = 0; j <= 6; j++) {
    const ax = pad + (w - 2 * pad) * i / 6, ay = pad + (h - 2 * pad) * j / 6;
    let dx = tsx - ax, dy = tsy - ay; const L = Math.hypot(dx, dy) || 1;
    arrow(c, ax, ay, ax + dx / L * 8, ay + dy / L * 8, hexA(pal.flow, 0.11));
  }

  // source distribution (τ=0 Gaussian cloud), faint
  for (const p of start) {
    c.fillStyle = hexA(pal.flow, 0.20);
    c.beginPath(); c.arc(sx(p.x), sy(p.y), 1.7, 0, 7); c.fill();
  }

  // trails fading noise -> action colour
  for (let p = 0; p < now.pts.length; p++) {
    const hot = p === sel;
    const g = c.createLinearGradient(sx(start[p].x), sy(start[p].y), sx(now.pts[p].x), sy(now.pts[p].y));
    g.addColorStop(0, hexA('#444444', 0.10));
    g.addColorStop(1, hot ? hexA(pal.action, 0.66) : hexA(pal.flow, 0.12 + 0.25 * tau));
    c.strokeStyle = g; c.lineWidth = hot ? 2 : 1.1;
    c.beginPath(); c.moveTo(sx(start[p].x), sy(start[p].y)); c.lineTo(sx(now.pts[p].x), sy(now.pts[p].y)); c.stroke();
  }
  // selected sample: discrete Euler integration steps along its path
  if (sel >= 0 && flow.x0[sel]) {
    const projP = (tk) => {
      let a = 0, b = 0; const x0 = flow.x0[sel], tg = flow.tgt[sel];
      for (let i = 0; i < flow.D; i++) { const x = (1 - tk) * x0[i] + tk * tg[i]; a += x * flow.P0[i]; b += x * flow.P1[i]; }
      return [a, b];
    };
    const K = 10;
    for (let k = 0; k <= K; k++) {
      const tk = k / K; if (tk > tau + 1e-6) break;
      const [px, py] = projP(tk);
      c.fillStyle = mix('#555555', pal.action, tk);
      c.beginPath(); c.arc(sx(px), sy(py), 1.9, 0, 7); c.fill();
    }
  }
  // current particles
  for (let p = 0; p < now.pts.length; p++) {
    const pt = now.pts[p], hot = p === sel;
    c.fillStyle = hot ? pal.action : mix('#555555', pal.flow, tau);
    c.beginPath(); c.arc(sx(pt.x), sy(pt.y), hot ? 3.8 : 2.4, 0, 7); c.fill();
    if (hot) { c.strokeStyle = '#dddddd'; c.lineWidth = 1.2; c.stroke(); }
  }

  // action point a*
  star(c, tsx, tsy, 7, pal.action);
  c.fillStyle = pal.action; c.font = '600 9px ui-monospace, monospace'; c.textAlign = 'left';
  c.fillText('a*', tsx + 9, tsy + 3.5);

  // labels
  c.fillStyle = '#555555'; c.font = '8.5px ui-monospace, monospace'; c.textAlign = 'left';
  c.fillText('noise  →  action', 10, 16);
  c.textAlign = 'right'; c.fillStyle = '#777777';
  c.fillText(`τ = ${tau.toFixed(2)}`, w - 10, h - 10);
}
function arrow(c, x1, y1, x2, y2, col) {
  c.strokeStyle = col; c.lineWidth = 1;
  c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
  const a = Math.atan2(y2 - y1, x2 - x1);
  c.beginPath();
  c.moveTo(x2, y2); c.lineTo(x2 - 4 * Math.cos(a - 0.4), y2 - 4 * Math.sin(a - 0.4));
  c.moveTo(x2, y2); c.lineTo(x2 - 4 * Math.cos(a + 0.4), y2 - 4 * Math.sin(a + 0.4));
  c.stroke();
}
function star(c, x, y, r, col) {
  c.save(); c.translate(x, y); c.fillStyle = col; c.shadowColor = col; c.shadowBlur = 10;
  c.beginPath();
  for (let i = 0; i < 10; i++) {
    const ang = Math.PI / 5 * i - Math.PI / 2, rr = i % 2 ? r * 0.45 : r;
    c.lineTo(Math.cos(ang) * rr, Math.sin(ang) * rr);
  }
  c.closePath(); c.fill(); c.restore();
}

// ---- action chunk heatmap (8 timesteps x 6 joints)
export function renderChunk(chunk) {
  const cv = document.getElementById('chunk-canvas');
  const { c, w, h } = ctx2d(cv);
  const T = chunk.length, J = 6;
  const x0 = 26, y0 = 20, gw = w - x0 - 8, gh = h - y0 - 16;
  const cw = gw / J, ch = gh / T;
  c.font = '8px ui-monospace, monospace'; c.textAlign = 'center';
  for (let j = 0; j < J; j++) { c.fillStyle = '#555555'; c.fillText('J' + (j + 1), x0 + cw * (j + .5), 13); }
  c.textAlign = 'right';
  for (let t = 0; t < T; t++) { c.fillStyle = '#555555'; c.fillText('t' + t, x0 - 4, y0 + ch * (t + .65)); }
  for (let t = 0; t < T; t++) for (let j = 0; j < J; j++) {
    const lo = LIMITS[j][0], hi = LIMITS[j][1];
    const n = ((chunk[t][j] - lo) / (hi - lo)) * 2 - 1;   // -> [-1,1]
    c.fillStyle = diverge(Math.max(-1, Math.min(1, n)));
    c.fillRect(x0 + j * cw + 1, y0 + t * ch + 1, cw - 2, ch - 2);
  }
}

// ---- color utils
function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
}
function mix(h1, h2, t) {
  const a = parseInt(h1.slice(1), 16), b = parseInt(h2.slice(1), 16);
  const r = Math.round((a >> 16 & 255) + ((b >> 16 & 255) - (a >> 16 & 255)) * t);
  const g = Math.round((a >> 8 & 255) + ((b >> 8 & 255) - (a >> 8 & 255)) * t);
  const bl = Math.round((a & 255) + ((b & 255) - (a & 255)) * t);
  return `rgb(${r},${g},${bl})`;
}
function diverge(n) {
  const p = palette();
  if (n >= 0) return mix('#151515', p.action, n);
  return mix('#151515', p.flow, -n);
}
