import { Scene } from './scene.js';
import { HOME, CHUNK_T } from './world.js';
import {
  parsePrompt, tokenize, patchGrid, attention, layerActs,
  taskTarget, buildChunk, makeFlow, sampleChunkAt,
} from './vla.js';
import {
  setActive, renderTokens, renderPatches, renderAttention,
  renderLayers, renderFlow, renderChunk,
} from './ui.js';

const scene = new Scene(
  document.getElementById('scene-canvas'),
  document.getElementById('camera-canvas'),
);

const els = {
  prompt: document.getElementById('prompt'),
  run: document.getElementById('run'),
  presets: document.getElementById('presets'),
  tau: document.getElementById('tau'),
  tauVal: document.getElementById('tau-val'),
  playFlow: document.getElementById('play-flow'),
  execute: document.getElementById('execute'),
  badge: document.getElementById('scene-badge'),
};

const PRESETS = [
  'pick up the red cube', 'grasp the blue block', 'point to the green cube',
  'push the yellow cube', 'move the gripper up', 'move left',
  'open the gripper', 'close the gripper',
];

let committedQ = HOME.slice();   // the arm's real joint state
let current = null;              // last computed VLA result
let runId = 0;

// ---- animation helper (cancellable) ----
let animTok = 0;
const cancelAnim = () => { animTok++; };
function animate(dur, fn) {
  const id = ++animTok, t0 = performance.now();
  return new Promise(res => {
    const step = now => {
      if (id !== animTok) return res(false);
      const t = Math.min(1, (now - t0) / dur);
      fn(t);
      t < 1 ? requestAnimationFrame(step) : res(true);
    };
    requestAnimationFrame(step);
  });
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---- the main inference pass ----
async function run() {
  const id = ++runId;
  cancelAnim();
  const prompt = els.prompt.value.trim() || 'pick up the red cube';

  const intent = parsePrompt(prompt);
  scene.setPose(committedQ);

  const tokens = tokenize(prompt, intent);
  const proj = scene.projectObjects();
  const grid = patchGrid(intent, proj);
  const attn = attention(intent, tokens);
  const layers = layerActs(intent);

  const tip0 = scene.fkTip(committedQ);
  const task = taskTarget(intent, tip0);
  let qTarget = task.ik ? scene.solveIK(task.point) : committedQ.slice();
  qTarget[5] = task.gripEnd;
  const chunkTarget = buildChunk(committedQ, qTarget, task.gripEnd, intent.mode);
  const flow = makeFlow(chunkTarget, prompt + '|' + intent.label);

  current = { intent, qTarget, flow, task };
  scene.highlight(task.highlightId);
  els.badge.textContent = intent.label;
  els.execute.disabled = els.playFlow.disabled = false;

  // staged reveal of the pipeline
  setActive('perception');
  renderTokens(tokens);
  renderPatches(grid);
  await sleep(420); if (id !== runId) return;

  setActive('transformer');
  renderAttention(attn);
  renderLayers(layers);
  await sleep(420); if (id !== runId) return;

  setActive('flow');
  await denoise(id);
  if (id !== runId) return;

  setActive('scene');
}

// ---- flow-matching denoising animation (τ: 0 -> 1) ----
async function denoise(id) {
  await animate(1500, t => {
    const tau = t;
    setTau(tau, false);
  });
  if (id === runId) setTau(1, false);
}

// set flow time τ: redraw plot/chunk and preview the forming action on the arm
function setTau(tau, fromUser) {
  if (!current) return;
  if (fromUser) cancelAnim();
  els.tau.value = tau;
  els.tauVal.textContent = tau.toFixed(2);
  const chunk = sampleChunkAt(current.flow, tau, 0);   // one sampled trajectory
  renderFlow(current.flow, tau, 0);
  renderChunk(chunk);
  scene.setPose(chunk[CHUNK_T - 1]);   // arm shows the destination pose forming from noise
  els.badge.textContent = tau < 0.999
    ? `denoising · τ=${tau.toFixed(2)}` : current.intent.label;
}

// ---- roll out the predicted chunk on the arm over time ----
async function execute() {
  if (!current) return;
  cancelAnim();
  const final = sampleChunkAt(current.flow, 1, 0);
  setActive('scene');
  await animate(1200, s => {
    const idx = s * (final.length - 1), i = Math.floor(idx), f = idx - i;
    const pose = final[i].map((v, j) =>
      i < final.length - 1 ? v + (final[i + 1][j] - v) * f : v);
    scene.setPose(pose);
    els.badge.textContent = `executing · t=${Math.min(final.length - 1, Math.round(idx))}`;
  });
  committedQ = final[final.length - 1].slice();
  scene.setPose(committedQ);
  els.badge.textContent = current.intent.label + ' ✓';
}

// ---- wiring ----
els.run.addEventListener('click', run);
els.prompt.addEventListener('keydown', e => { if (e.key === 'Enter') run(); });
els.tau.addEventListener('input', e => setTau(parseFloat(e.target.value), true));
els.playFlow.addEventListener('click', () => { if (current) denoise(runId); });
els.execute.addEventListener('click', execute);

for (const p of PRESETS) {
  const b = document.createElement('button');
  b.className = 'c-preset preset'; b.textContent = p;
  b.addEventListener('click', () => { els.prompt.value = p; run(); });
  els.presets.appendChild(b);
}

// ---- render loop ----
(function loop() { scene.render(); requestAnimationFrame(loop); })();

// debug handle
window.__vla = { run, execute, setTau, scene, get current() { return current; } };

// first pass
setActive('all');
run();
