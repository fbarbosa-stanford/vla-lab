// Langton's Ant generative background — carried over from felipebarbosa.co
// for visual continuity. Draws onto a fixed full-viewport <canvas id="bg">.
(() => {
  const canvas = document.getElementById('bg');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const IS_IOS = /iPad|iPhone|iPod/.test(navigator.userAgent)
    || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);

  const CELL = IS_IOS ? 6 : 4;
  const NUM_ANTS = IS_IOS ? 2 : 5;
  const STEPS_PER_FRAME = IS_IOS ? 12 : 60;
  const DX = [0, 1, 0, -1];
  const DY = [-1, 0, 1, 0];

  let W, H, cols, rows, cells, ants, imgData;

  function init() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
    cols = Math.ceil(W / CELL);
    rows = Math.ceil(H / CELL);
    cells = new Uint8Array(cols * rows);
    imgData = null;
    ants = [];
    for (let i = 0; i < NUM_ANTS; i++) {
      ants.push({
        x: Math.floor(cols / 2) + Math.floor(Math.random() * 40 - 20),
        y: Math.floor(rows / 2) + Math.floor(Math.random() * 40 - 20),
        d: Math.floor(Math.random() * 4),
      });
    }
  }

  function step() {
    for (const a of ants) {
      const idx = a.y * cols + a.x;
      if (cells[idx] === 0) a.d = (a.d + 1) & 3;
      else a.d = (a.d + 3) & 3;
      cells[idx] ^= 1;
      a.x += DX[a.d]; a.y += DY[a.d];
      if (a.x < 0) a.x = cols - 1; else if (a.x >= cols) a.x = 0;
      if (a.y < 0) a.y = rows - 1; else if (a.y >= rows) a.y = 0;
    }
  }

  function render() {
    if (!imgData || imgData.width !== W || imgData.height !== H)
      imgData = ctx.createImageData(W, H);
    const d = imgData.data;
    d.fill(0);
    for (let cy = 0; cy < rows; cy++) {
      for (let cx = 0; cx < cols; cx++) {
        if (cells[cy * cols + cx] === 0) continue;
        const px = cx * CELL, py = cy * CELL, v = 22;
        for (let dy = 0; dy < CELL - 1; dy++) {
          const ry = py + dy; if (ry >= H) break;
          for (let dx = 0; dx < CELL - 1; dx++) {
            const rx = px + dx; if (rx >= W) break;
            const i = (ry * W + rx) * 4;
            d[i] = d[i + 1] = d[i + 2] = v; d[i + 3] = 175;
          }
        }
      }
    }
    for (const a of ants) {
      const px = a.x * CELL, py = a.y * CELL;
      for (let dy = 0; dy < CELL; dy++) {
        const ry = py + dy; if (ry >= H) break;
        for (let dx = 0; dx < CELL; dx++) {
          const rx = px + dx; if (rx >= W) break;
          const i = (ry * W + rx) * 4;
          d[i] = d[i + 1] = d[i + 2] = 120; d[i + 3] = 220;
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  function frame() {
    for (let i = 0; i < STEPS_PER_FRAME; i++) step();
    render();
    requestAnimationFrame(frame);
  }

  init();
  window.addEventListener('resize', () => {
    const nextW = window.innerWidth, nextH = window.innerHeight;
    const smallIOSViewportShift = IS_IOS && nextW === W && Math.abs(nextH - H) < 120;
    if (!smallIOSViewportShift) init();
  });
  requestAnimationFrame(frame);
})();
