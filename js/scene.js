import * as THREE from 'three';
import { OBJECTS, CUBE, ARM, HOME, LIMITS } from './world.js';

const { baseH, L1, L2, L3 } = ARM;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// ---------------------------------------------------------------------------
// Scene: god-view render + model-camera render + SO-101 arm + kinematics.
// ---------------------------------------------------------------------------
export class Scene {
  constructor(godCanvas, camCanvas) {
    this.godCanvas = godCanvas;
    this.camCanvas = camCanvas;
    this.q = HOME.slice();
    this.cubes = {};
    this._initRenderers();
    this._initScene();
    this._initArm();
    this._initCameras();
    this._initOrbit();
    this.setPose(this.q);
    this._resize();
    window.addEventListener('resize', () => this._resize());
    if (window.ResizeObserver) new ResizeObserver(() => this._resize()).observe(this.godCanvas);
  }

  _initRenderers() {
    this.renderer = new THREE.WebGLRenderer({ canvas: this.godCanvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.camRenderer = new THREE.WebGLRenderer({ canvas: this.camCanvas, antialias: true });
    this.camRenderer.setPixelRatio(1);
    this.camRenderer.setSize(320, 240, false);
  }

  _initScene() {
    const s = new THREE.Scene();
    s.background = new THREE.Color(0x06080d);
    s.fog = new THREE.Fog(0x06080d, 0.9, 2.2);
    this.scene = s;

    const hemi = new THREE.HemisphereLight(0x9fb4ff, 0x141018, 0.75);
    s.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(0.35, 0.7, 0.5);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 0.1; key.shadow.camera.far = 2;
    key.shadow.camera.left = -0.5; key.shadow.camera.right = 0.5;
    key.shadow.camera.top = 0.5; key.shadow.camera.bottom = -0.5;
    s.add(key);
    const rim = new THREE.DirectionalLight(0x5ad1ff, 0.5);
    rim.position.set(-0.5, 0.3, -0.4);
    s.add(rim);

    // table
    const table = new THREE.Mesh(
      new THREE.BoxGeometry(0.9, 0.02, 0.7),
      new THREE.MeshStandardMaterial({ color: 0x10141d, roughness: 0.85, metalness: 0.1 })
    );
    table.position.set(0, -0.01, 0.18);
    table.receiveShadow = true;
    s.add(table);

    const grid = new THREE.GridHelper(0.9, 18, 0x2a3346, 0x1a2030);
    grid.position.set(0, 0.0005, 0.18);
    s.add(grid);

    // cubes
    for (const o of OBJECTS) {
      const mat = new THREE.MeshStandardMaterial({
        color: o.color, roughness: 0.35, metalness: 0.15,
        emissive: o.color, emissiveIntensity: 0.12,
      });
      const cube = new THREE.Mesh(new THREE.BoxGeometry(CUBE, CUBE, CUBE), mat);
      cube.position.set(...o.pos);
      cube.castShadow = true; cube.receiveShadow = true;
      cube.userData.id = o.id;
      s.add(cube);
      this.cubes[o.id] = cube;
    }

    // target highlight ring
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(CUBE * 0.85, CUBE * 1.05, 32),
      new THREE.MeshBasicMaterial({ color: 0xffb454, side: THREE.DoubleSide, transparent: true, opacity: 0 })
    );
    ring.rotation.x = -Math.PI / 2;
    s.add(ring);
    this.ring = ring;
  }

  _link(len, radTop, radBot, color, rough = 0.5) {
    const g = new THREE.CylinderGeometry(radTop, radBot, len, 20);
    g.translate(0, len / 2, 0); // base at origin, extends +Y
    const m = new THREE.MeshStandardMaterial({ color, roughness: rough, metalness: 0.35 });
    const mesh = new THREE.Mesh(g, m);
    mesh.castShadow = true;
    return mesh;
  }

  _joint(rad, color = 0x2b3550) {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(rad, 20, 16),
      new THREE.MeshStandardMaterial({ color, roughness: 0.4, metalness: 0.5 })
    );
    m.castShadow = true;
    return m;
  }

  _initArm() {
    const SHELL = 0xc9d3e6, ACCENT = 0x394760;
    const root = new THREE.Group();
    this.scene.add(root);

    // base plate
    const base = new THREE.Mesh(
      new THREE.CylinderGeometry(0.045, 0.052, 0.03, 28),
      new THREE.MeshStandardMaterial({ color: 0x222a3a, roughness: 0.6, metalness: 0.4 })
    );
    base.position.y = 0.015; base.castShadow = true; base.receiveShadow = true;
    root.add(base);

    this.pan = new THREE.Group();           // shoulder_pan (Y)
    root.add(this.pan);

    this.lift = new THREE.Group();          // shoulder_lift (X)
    this.lift.position.y = baseH;
    this.pan.add(this.lift);
    this.lift.add(this._joint(0.028, ACCENT));
    this.lift.add(this._link(L1, 0.017, 0.021, SHELL));

    this.elbow = new THREE.Group();         // elbow_flex (X)
    this.elbow.position.y = L1;
    this.lift.add(this.elbow);
    this.elbow.add(this._joint(0.023, ACCENT));
    this.elbow.add(this._link(L2, 0.014, 0.017, SHELL));

    this.wrist = new THREE.Group();         // wrist_flex (X)
    this.wrist.position.y = L2;
    this.elbow.add(this.wrist);
    this.wrist.add(this._joint(0.019, ACCENT));

    this.roll = new THREE.Group();          // wrist_roll (Y / link axis)
    this.wrist.add(this.roll);
    const palm = this._link(L3 * 0.45, 0.015, 0.016, ACCENT, 0.45);
    this.roll.add(palm);

    // gripper fingers
    this.fingerL = this._makeFinger();
    this.fingerR = this._makeFinger();
    this.fingerL.position.set(0, L3 * 0.45, 0);
    this.fingerR.position.set(0, L3 * 0.45, 0);
    this.roll.add(this.fingerL);
    this.roll.add(this.fingerR);

    // fingertip marker (for FK readout)
    this.tip = new THREE.Object3D();
    this.tip.position.set(0, L3, 0);
    this.roll.add(this.tip);
  }

  _makeFinger() {
    const g = new THREE.Group();
    const len = L3 * 0.6;
    const m = new THREE.Mesh(
      new THREE.BoxGeometry(0.008, len, 0.016),
      new THREE.MeshStandardMaterial({ color: 0x46566f, roughness: 0.4, metalness: 0.4 })
    );
    m.geometry.translate(0, len / 2, 0);
    m.castShadow = true;
    g.add(m);
    return g;
  }

  _initCameras() {
    this.godCam = new THREE.PerspectiveCamera(42, 1, 0.05, 10);
    this.orbit = { az: 0.62, el: 0.42, r: 0.74, target: new THREE.Vector3(0, 0.12, 0.16) };

    // model's onboard camera looking at the table
    this.modelCam = new THREE.PerspectiveCamera(46, 320 / 240, 0.05, 10);
    this.modelCam.position.set(0.0, 0.36, 0.62);
    this.modelCam.lookAt(0, 0.02, 0.16);
  }

  _applyOrbit() {
    const { az, el, r, target } = this.orbit;
    this.godCam.position.set(
      target.x + r * Math.cos(el) * Math.sin(az),
      target.y + r * Math.sin(el),
      target.z + r * Math.cos(el) * Math.cos(az)
    );
    this.godCam.lookAt(target);
  }

  _initOrbit() {
    const c = this.godCanvas;
    let dragging = false, px = 0, py = 0;
    c.addEventListener('pointerdown', e => { dragging = true; px = e.clientX; py = e.clientY; });
    window.addEventListener('pointerup', () => dragging = false);
    window.addEventListener('pointermove', e => {
      if (!dragging) return;
      this.orbit.az -= (e.clientX - px) * 0.008;
      this.orbit.el = clamp(this.orbit.el + (e.clientY - py) * 0.008, -0.2, 1.45);
      px = e.clientX; py = e.clientY;
    });
    c.addEventListener('wheel', e => {
      e.preventDefault();
      this.orbit.r = clamp(this.orbit.r * (1 + Math.sign(e.deltaY) * 0.08), 0.4, 1.6);
    }, { passive: false });
  }

  _resize() {
    const w = Math.min(this.godCanvas.clientWidth, 4096);
    const h = Math.min(this.godCanvas.clientHeight, 4096);
    if (w < 2 || h < 2) return;
    this.renderer.setSize(w, h, false);
    this.godCam.aspect = w / h;
    this.godCam.updateProjectionMatrix();
  }

  // -------------------------------------------------------------------------
  // kinematics
  // -------------------------------------------------------------------------
  setPose(q) {
    this.q = q.slice();
    this.pan.rotation.y = q[0];
    this.lift.rotation.x = q[1];
    this.elbow.rotation.x = q[2];
    this.wrist.rotation.x = q[3];
    this.roll.rotation.y = q[4];
    const open = clamp(q[5], 0, 1);
    const spread = 0.006 + open * 0.026;
    this.fingerL.position.x = -spread;
    this.fingerR.position.x = spread;
  }

  // forward kinematics: fingertip world position for a joint vector.
  fkTip(q) {
    const t1 = q[1], t12 = q[1] + q[2], t123 = q[1] + q[2] + q[3];
    const radial = L1 * Math.sin(t1) + L2 * Math.sin(t12) + L3 * Math.sin(t123);
    const up = baseH + L1 * Math.cos(t1) + L2 * Math.cos(t12) + L3 * Math.cos(t123);
    return new THREE.Vector3(radial * Math.sin(q[0]), up, radial * Math.cos(q[0]));
  }

  // analytic IK: reach fingertip to world point `T`, gripper pointing down.
  // returns a 6-vector (gripper value left to caller).
  solveIK(T, { gripDown = true } = {}) {
    const pan = Math.atan2(T.x, T.z);
    const rh = Math.hypot(T.x, T.z);

    // wrist point sits L3 above the target (gripper points straight down)
    let a = rh;                       // radial, from shoulder axis
    let b = (T.y + (gripDown ? L3 : 0)) - baseH; // vertical, from shoulder pivot

    // clamp to reachable annulus
    const reach = (L1 + L2) * 0.995;
    const dmin = Math.abs(L1 - L2) + 0.005;
    let d = Math.hypot(a, b);
    if (d > reach) { a *= reach / d; b *= reach / d; d = reach; }
    if (d < dmin)  { const k = dmin / (d || 1e-6); a *= k; b *= k; d = dmin; }

    // standard 2R IK (x=a horizontal, y=b vertical), elbow chosen "up/back"
    const c2 = clamp((a * a + b * b - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1, 1);
    const q2std = -Math.acos(c2);                                   // elbow-up branch
    const q1std = Math.atan2(b, a) - Math.atan2(L2 * Math.sin(q2std), L1 + L2 * Math.cos(q2std));

    // convert angles (measured from +horizontal) to our from-vertical joints
    const lift  = Math.PI / 2 - q1std;
    const elbow = -q2std;

    // wrist_flex so the L3 link points straight down (absolute tilt = PI from up)
    let wristFlex = Math.PI - (lift + elbow);
    // wrap to nearest equivalent within limits
    while (wristFlex > Math.PI) wristFlex -= 2 * Math.PI;
    while (wristFlex < -Math.PI) wristFlex += 2 * Math.PI;

    const q = [pan, lift, elbow, wristFlex, 0, this.q[5]];
    for (let i = 0; i < 5; i++) q[i] = clamp(q[i], LIMITS[i][0], LIMITS[i][1]);
    return q;
  }

  // project every object into the model camera; returns {id: {u,v,vis}} in [0,1].
  projectObjects() {
    const out = {};
    const v = new THREE.Vector3();
    for (const o of OBJECTS) {
      this.cubes[o.id].getWorldPosition(v);
      v.project(this.modelCam);
      out[o.id] = {
        u: (v.x + 1) / 2,
        vv: (1 - v.y) / 2,
        vis: v.z < 1, // in front of camera
      };
    }
    return out;
  }

  highlight(id) {
    if (!id || !this.cubes[id]) { this.ring.material.opacity = 0; return; }
    const p = this.cubes[id].position;
    this.ring.position.set(p.x, 0.002, p.z);
    this.ring.material.opacity = 0.9;
  }

  render() {
    this._applyOrbit();
    this.renderer.render(this.scene, this.godCam);
    this.camRenderer.render(this.scene, this.modelCam);
  }
}
