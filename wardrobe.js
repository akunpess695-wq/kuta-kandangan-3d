// wardrobe.js — "Lemari Baju": memakaikan pakaian adat Banjar ke mannequin 3D.
// Pakaian dibuat sebagai salinan "SkinnedMesh" dari badan mannequin (bobot tulang sama),
// jadi ikut bergerak persis saat jalan, lari, lompat. Tidak bergantung pada nama tulang.
import * as THREE from "three";
import { analyzeRig, CLS_ARM, CLS_LEG, CLS_TORSO } from "./wardrobe-core.js";

const BASE = "assets/clothes/";
const COL = {
  skin: 0xfbc6a2, hair: 0x392c2b, gold: 0xe9b349,
  jacketSleeve: 0x272325, whiteSleeve: 0xf2edf2, purpleSleeve: 0x785483, blackSleeve: 0x262325,
  pants: 0x2a2629, shorts: 0x373135, shoe: 0x1c1a1b, sandal: 0x232021,
  belt: 0x161415, syal: 0x80242c, sole: 0xf4f1ea, pecihat: 0x151314
};

const ATASAN = {
  jaket: { label: "Jaket Adat", tex: "jacket", sleeve: COL.jacketSleeve, long: true, cuff: COL.gold, hem: 0.465 },
  putih: { label: "Kemeja Putih", tex: "shirt_white", sleeve: COL.whiteSleeve, long: true, hem: 0.50 },
  ungu: { label: "Batik Ungu", tex: "shirt_purple", sleeve: COL.purpleSleeve, long: false, hem: 0.50 },
  hitam: { label: "Batik Hitam", tex: "shirt_black", sleeve: COL.blackSleeve, long: false, hem: 0.50 }
};
const KAIN = { songket: { label: "Songket Emas", tex: "songket" }, hitam: { label: "Kain Hitam", tex: "kain_hitam" } };
const TOPI = { laung: { label: "Laung Adat" }, peci: { label: "Peci Hitam" } };
const SEPATU = { pantofel: { label: "Pantofel" }, sneakers: { label: "Sneakers" }, sandal: { label: "Sandal" } };

const PRESETS = {
  adat: { label: "Adat Banjar", sel: { atasan: "jaket", kain: "songket", celana: "hitam", topi: "laung", sepatu: "pantofel", sabuk: true, syal: false } },
  santai: { label: "Santai", sel: { atasan: "ungu", kain: "none", celana: "hitam", topi: "none", sepatu: "sneakers", sabuk: false, syal: false } },
  salat: { label: "Baju Koko", sel: { atasan: "putih", kain: "hitam", celana: "hitam", topi: "peci", sepatu: "sandal", sabuk: false, syal: false } }
};

const W = {
  root: null, A: null, ents: [], skel: null, bindRootInv: [], made: [],
  sel: { ...PRESETS.adat.sel },
  fit: { infl: 1, flip: false },
  tex: new Map(), loader: new THREE.TextureLoader(), skinMat: null
};

function getTex(name, rx = 1, ry = 1) {
  const key = `${name}|${rx}|${ry}`;
  let t = W.tex.get(key);
  if (t) return t;
  t = W.loader.load(BASE + name + ".jpg");
  t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(rx, ry);
  t.anisotropy = 4;
  W.tex.set(key, t);
  return t;
}

// ---------- analisis ----------
function smoothNormals(geo) {
  if (!geo.attributes.normal) geo.computeVertexNormals();
  const p = geo.attributes.position, n = geo.attributes.normal;
  const map = new Map(), keys = new Array(p.count);
  for (let i = 0; i < p.count; i++) {
    const k = p.getX(i).toFixed(4) + "," + p.getY(i).toFixed(4) + "," + p.getZ(i).toFixed(4);
    keys[i] = k;
    let a = map.get(k);
    if (!a) { a = [0, 0, 0]; map.set(k, a); }
    a[0] += n.getX(i); a[1] += n.getY(i); a[2] += n.getZ(i);
  }
  const out = new Float32Array(p.count * 3);
  for (let i = 0; i < p.count; i++) {
    const a = map.get(keys[i]);
    const l = Math.hypot(a[0], a[1], a[2]) || 1;
    out[i * 3] = a[0] / l; out[i * 3 + 1] = a[1] / l; out[i * 3 + 2] = a[2] / l;
  }
  return out;
}

function analyze(manRoot) {
  manRoot.updateMatrixWorld(true);
  const rootInv = new THREE.Matrix4().copy(manRoot.matrixWorld).invert();
  const meshes = [];
  manRoot.traverse((n) => { if (n.isSkinnedMesh && n.skeleton && n.geometry.attributes.skinIndex) meshes.push(n); });
  if (!meshes.length) return false;
  const skel = meshes[0].skeleton;
  const v = new THREE.Vector3();
  const bones = skel.bones.map((b) => {
    v.setFromMatrixPosition(b.matrixWorld).applyMatrix4(rootInv);
    return { x: v.x, y: v.y, z: v.z };
  });
  W.skel = skel;
  W.bindRootInv = skel.bones.map((b) => new THREE.Matrix4().multiplyMatrices(rootInv, b.matrixWorld).invert());

  const ents = [];
  for (const mesh of meshes) {
    if (mesh.skeleton !== skel && mesh.skeleton.bones.length !== skel.bones.length) continue;
    const g = mesh.geometry, pos = g.attributes.position, n = pos.count;
    const M = new THREE.Matrix4().multiplyMatrices(rootInv, mesh.matrixWorld);
    const rp = new Float32Array(n * 3), dom = new Uint16Array(n);
    const si = g.attributes.skinIndex, sw = g.attributes.skinWeight;
    for (let i = 0; i < n; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(M);
      rp[i * 3] = v.x; rp[i * 3 + 1] = v.y; rp[i * 3 + 2] = v.z;
      let best = 0, bw = -1;
      for (let k = 0; k < 4; k++) {
        const w = sw.getComponent(i, k);
        if (w > bw) { bw = w; best = si.getComponent(i, k); }
      }
      dom[i] = best;
    }
    const sn = smoothNormals(g);
    const nr = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      v.set(sn[i * 3], sn[i * 3 + 1], sn[i * 3 + 2]).transformDirection(M);
      nr[i * 3] = v.x; nr[i * 3 + 1] = v.y; nr[i * 3 + 2] = v.z;
    }
    const scale = new THREE.Vector3().setFromMatrixScale(M).x || 1;
    ents.push({ mesh, n, rp, dom, sn, nr, scale });
  }
  if (!ents.length) return false;
  const A = analyzeRig({ entries: ents, bones });
  ents.forEach((e, k) => { e.per = A.per[k]; });
  W.ents = ents;
  W.A = A;
  console.info("[Lemari Baju] rangka:", {
    tinggi: +A.H.toFixed(3), tulang: A.bones, depan: A.frontSign > 0 ? "+z" : "-z",
    bahu: !!(A.S[-1] && A.S[1]), tulangKepala: A.headIdx
  });
  return true;
}

// ---------- pembuat potongan ----------
function mk(ent, test, mode, inflFrac, o = {}) {
  const A = W.A, g0 = ent.mesh.geometry;
  const index = g0.index ? g0.index.array : null;
  const tris = (index ? index.length : g0.attributes.position.count) / 3;
  const cache = new Int8Array(ent.n).fill(-1);
  const t = (i) => { let c = cache[i]; if (c < 0) { c = test(i) ? 1 : 0; cache[i] = c; } return c === 1; };
  const remap = new Map(), list = [], idx = [];
  for (let f = 0; f < tris; f++) {
    const a = index ? index[f * 3] : f * 3, b = index ? index[f * 3 + 1] : f * 3 + 1, c = index ? index[f * 3 + 2] : f * 3 + 2;
    const ok = mode === "all" ? (t(a) && t(b) && t(c)) : (t(a) || t(b) || t(c));
    if (!ok) continue;
    for (const vtx of [a, b, c]) {
      let m = remap.get(vtx);
      if (m === undefined) { m = list.length; remap.set(vtx, m); list.push(vtx); }
      idx.push(m);
    }
  }
  if (!idx.length) return null;
  const infl = (inflFrac * A.H * W.fit.infl) / ent.scale;
  const n = list.length;
  const pos = new Float32Array(n * 3), nor = new Float32Array(n * 3);
  const si = new Uint16Array(n * 4), sw = new Float32Array(n * 4);
  const P = g0.attributes.position, N = g0.attributes.normal, SI = g0.attributes.skinIndex, SW = g0.attributes.skinWeight;
  for (let k = 0; k < n; k++) {
    const i = list[k];
    pos[k * 3] = P.getX(i) + ent.sn[i * 3] * infl;
    pos[k * 3 + 1] = P.getY(i) + ent.sn[i * 3 + 1] * infl;
    pos[k * 3 + 2] = P.getZ(i) + ent.sn[i * 3 + 2] * infl;
    nor[k * 3] = N.getX(i); nor[k * 3 + 1] = N.getY(i); nor[k * 3 + 2] = N.getZ(i);
    for (let c = 0; c < 4; c++) { si[k * 4 + c] = SI.getComponent(i, c); sw[k * 4 + c] = SW.getComponent(i, c); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
  g.setAttribute("skinIndex", new THREE.BufferAttribute(si, 4));
  g.setAttribute("skinWeight", new THREE.BufferAttribute(sw, 4));
  g.setIndex(new THREE.BufferAttribute(new Uint32Array(idx), 1));
  const uv = new Float32Array(n * 2);
  if (o.uv) o.uv(list, uv);
  g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  let vc = false;
  if (o.colorFn) {
    const col = new Float32Array(n * 3), c = new THREE.Color();
    for (let k = 0; k < n; k++) { c.set(o.colorFn(list[k])); col[k * 3] = c.r; col[k * 3 + 1] = c.g; col[k * 3 + 2] = c.b; }
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    vc = true;
  }
  const mat = new THREE.MeshStandardMaterial({
    color: o.map ? 0xffffff : (o.color ?? 0xffffff), map: o.map || null,
    roughness: o.rough ?? 0.85, metalness: 0, vertexColors: vc
  });
  const sm = new THREE.SkinnedMesh(g, mat);
  sm.position.copy(ent.mesh.position); sm.quaternion.copy(ent.mesh.quaternion); sm.scale.copy(ent.mesh.scale);
  sm.bind(ent.mesh.skeleton, ent.mesh.bindMatrix);
  sm.frustumCulled = false; sm.castShadow = true; sm.receiveShadow = true;
  ent.mesh.parent.add(sm);
  W.made.push(sm);
  return sm;
}

function clearMade() {
  for (const m of W.made) { m.parent?.remove(m); m.geometry.dispose(); m.material.dispose(); }
  W.made = [];
}

function attachToBone(mesh, boneIdx) {
  mesh.updateMatrix();
  const local = new THREE.Matrix4().multiplyMatrices(W.bindRootInv[boneIdx], mesh.matrix);
  local.decompose(mesh.position, mesh.quaternion, mesh.scale);
  W.skel.bones[boneIdx].add(mesh);
  W.made.push(mesh);
}

function headBoneIndex() {
  const A = W.A;
  if (A.headIdx >= 0) return A.headIdx;
  let best = 0, bd = Infinity;
  W.skel.bones.forEach((b, i) => {
    const p = new THREE.Vector3().setFromMatrixPosition(b.matrixWorld);
    const d = Math.abs(p.y - A.headY);
    if (d < bd) { bd = d; best = i; }
  });
  return best;
}

// ---------- terapkan pilihan ----------
function apply() {
  if (!W.A) return;
  clearMade();
  const A = W.A, H = A.H, y0 = A.y0, F = A.F;
  const fs = A.frontSign * (W.fit.flip ? -1 : 1);
  const R = (e, i) => (e.rp[i * 3 + 1] - y0) / H;
  const nz = (e, i) => e.nr[i * 3 + 2] * fs;
  const s = W.sel;

  const planarUV = (e, xc, xw, yb, yt) => (list, uv) => {
    for (let k = 0; k < list.length; k++) {
      const i = list[k];
      uv[k * 2] = 0.5 + fs * (e.rp[i * 3] - xc) / xw;
      uv[k * 2 + 1] = (e.rp[i * 3 + 1] - yb) / (yt - yb);
    }
  };
  const extentX = (e, list) => {
    let a = Infinity, b = -Infinity;
    for (const i of list) {
      if (e.per.cls[i] !== CLS_TORSO) continue;
      const x = e.rp[i * 3]; if (x < a) a = x; if (x > b) b = x;
    }
    return a < b ? [(a + b) / 2, b - a] : [A.cx, 0.3 * H];
  };
  // UV planar yang lebarnya mengikuti sebaran vertex yang terpilih
  const fitUV = (e, test, yb, yt) => (list, uv) => {
    const [xc, xw] = extentX(e, list);
    planarUV(e, xc, xw, yb, yt)(list, uv);
  };

  for (const e of W.ents) {
    const cls = e.per.cls, dn = e.per.dn;
    const shoulderF = (A.shoulderY - y0) / H;

    // --- celana / celana pendek ---
    if (s.celana === "hitam") {
      mk(e, (i) => cls[i] !== CLS_ARM && R(e, i) < F.waist && R(e, i) > F.footTop, "any", 0.008, { color: COL.pants });
    } else {
      mk(e, (i) => cls[i] !== CLS_ARM && R(e, i) < F.waist && R(e, i) > 0.36, "any", 0.008, { color: COL.shorts });
    }

    // --- kain / sarung ---
    if (s.kain !== "none" && KAIN[s.kain]) {
      const yb = y0 + F.knee * H, yt = y0 + (F.waist + 0.012) * H;
      const tex = getTex(KAIN[s.kain].tex);
      mk(e, (i) => cls[i] !== CLS_ARM && R(e, i) > F.knee && R(e, i) < F.waist + 0.012, "any", 0.022, {
        map: tex, uv: (list, uv) => {
          const [xc, xw] = extentX(e, list);
          const texW = 0.46 * (yt - yb); // jaga proporsi motif
          for (let k = 0; k < list.length; k++) {
            const i = list[k];
            uv[k * 2] = 0.5 + fs * (e.rp[i * 3] - xc) / Math.max(texW, 1e-4);
            uv[k * 2 + 1] = (e.rp[i * 3 + 1] - yb) / (yt - yb);
          }
        }
      });
    }

    // --- sepatu ---
    if (s.sepatu && s.sepatu !== "none") {
      if (s.sepatu === "sandal") {
        mk(e, (i) => cls[i] === CLS_LEG && R(e, i) < 0.02, "any", 0.010, { color: COL.sandal });
      } else {
        mk(e, (i) => cls[i] === CLS_LEG && R(e, i) < F.footTop, "any", 0.013, {
          color: COL.shoe,
          colorFn: s.sepatu === "sneakers" ? (i) => (R(e, i) < 0.016 ? COL.sole : COL.shoe) : null
        });
      }
    }

    // --- atasan ---
    const at = ATASAN[s.atasan];
    if (at) {
      const yTopF = shoulderF + 0.025, yb = y0 + at.hem * H, yt = y0 + yTopF * H;
      const tex = getTex(at.tex);
      const torsoTest = (i) => cls[i] !== CLS_ARM && R(e, i) >= at.hem && R(e, i) <= yTopF;
      mk(e, torsoTest, "any", 0.014, { map: tex, uv: (list, uv) => fitUV(e, torsoTest, yb, yt)(list, uv) });
      const cut = at.long ? 0.80 : 0.42;
      mk(e, (i) => cls[i] === CLS_ARM && dn[i] <= cut, "any", 0.016, {
        color: at.sleeve,
        colorFn: at.cuff ? (i) => (dn[i] > cut - 0.07 ? at.cuff : at.sleeve) : null
      });
    }

    // --- sabuk ---
    if (s.sabuk) {
      const bt = (i) => cls[i] !== CLS_ARM && R(e, i) >= F.waist - 0.012 && R(e, i) <= F.waist + 0.022;
      mk(e, bt, "any", 0.030, { color: COL.belt });
      mk(e, (i) => bt(i) && nz(e, i) > 0.45 && Math.abs(e.rp[i * 3] - A.cx) < 0.03 * H, "any", 0.036, { color: COL.gold, rough: 0.35 });
    }

    // --- syal ---
    if (s.syal) {
      const st = (i) => cls[i] !== CLS_ARM && R(e, i) >= shoulderF - 0.005 && R(e, i) <= shoulderF + 0.04;
      const tile = 0.09 * H;
      mk(e, st, "any", 0.03, {
        map: getTex("syal"), uv: (list, uv) => {
          for (let k = 0; k < list.length; k++) {
            const i = list[k];
            uv[k * 2] = fs * (e.rp[i * 3] - A.cx) / tile + 0.5;
            uv[k * 2 + 1] = (e.rp[i * 3 + 1] - y0) / tile;
          }
        }
      });
    }

    // --- wajah & rambut (kepala mannequin diberi wajah) ---
    const hd = A.head;
    if (hd.w > 0 && hd.h > 0) {
      const front = (i) => cls[i] === CLS_TORSO && e.rp[i * 3 + 1] >= hd.chinY - 0.012 * H && nz(e, i) > 0.25;
      const vBot = hd.chinY - 0.05 * hd.h;
      mk(e, front, "any", 0.012, {
        map: getTex("face"), rough: 0.9,
        uv: (list, uv) => {
          for (let k = 0; k < list.length; k++) {
            const i = list[k];
            uv[k * 2] = 0.5 + fs * (e.rp[i * 3] - hd.cx) / hd.w;
            uv[k * 2 + 1] = (e.rp[i * 3 + 1] - vBot) / (hd.topY - vBot);
          }
        }
      });
      const earY = hd.chinY + 0.46 * hd.h;
      mk(e, (i) => cls[i] === CLS_TORSO && e.rp[i * 3 + 1] >= earY && !(nz(e, i) > 0.25), "any", 0.009, { color: COL.hair, rough: 0.95 });
    }
  }

  // --- topi (menempel ke tulang kepala) ---
  if (s.topi && s.topi !== "none") addHat(s.topi);
  try { localStorage.setItem("kk_wardrobe", JSON.stringify({ sel: W.sel, fit: W.fit })); } catch (_) { /* abaikan */ }
}

function addHat(kind) {
  const A = W.A, hd = A.head;
  const hbi = headBoneIndex();
  const rimY = hd.chinY + 0.66 * hd.h;
  const rx = hd.w / 2, rz = hd.d / 2;
  let mesh;
  if (kind === "laung") {
    const domeH = (hd.topY - rimY) + 0.14 * hd.h;
    const geo = new THREE.SphereGeometry(1, 32, 18, 0, Math.PI * 2, 0, Math.PI * 0.56);
    const mat = new THREE.MeshStandardMaterial({ map: getTex("laung", 2, 1), roughness: 0.8, side: THREE.DoubleSide });
    mesh = new THREE.Mesh(geo, mat);
    mesh.scale.set(rx * 1.13, domeH, rz * 1.13);
    mesh.position.set(hd.cx, rimY, hd.cz);
  } else {
    const h = 0.30 * hd.h;
    const geo = new THREE.CylinderGeometry(0.86, 1.0, 1, 28, 1, false);
    const mat = new THREE.MeshStandardMaterial({ color: COL.pecihat, roughness: 0.9 });
    mesh = new THREE.Mesh(geo, mat);
    mesh.scale.set(rx * 1.1, h, rz * 1.1);
    mesh.position.set(hd.cx, rimY - 0.02 * hd.h + h / 2, hd.cz);
    const band = new THREE.Mesh(
      new THREE.CylinderGeometry(1.0, 1.0, 0.30, 28, 1, true),
      new THREE.MeshStandardMaterial({ map: getTex("peci_band", 3, 1), roughness: 0.7, side: THREE.DoubleSide })
    );
    band.scale.set(1.012, 1, 1.012);
    band.position.set(0, -0.36, 0);
    mesh.add(band);
  }
  mesh.castShadow = true;
  attachToBone(mesh, hbi);
}

// ---------- antarmuka ----------
function buildUI() {
  const hud = document.getElementById("jelajahHud");
  const exitBtn = document.getElementById("btnJelajahExit");
  if (!hud || !exitBtn || document.getElementById("wardrobePanel")) return;

  const openBtn = document.createElement("button");
  openBtn.id = "btnWardrobe"; openBtn.type = "button";
  openBtn.className = "comic-btn rounded-full bg-[#fffdf6] px-3 py-1.5 text-xs font-extrabold text-[#16392b]";
  openBtn.textContent = "👕 Lemari Baju";
  exitBtn.parentElement.insertBefore(openBtn, exitBtn);

  const panel = document.createElement("div");
  panel.id = "wardrobePanel";
  panel.className = "comic-card-solid pointer-events-auto hidden";
  panel.style.cssText = "position:fixed;left:50%;transform:translateX(-50%);bottom:158px;width:min(440px,calc(100vw - 20px));max-height:46vh;overflow:auto;padding:12px;border-radius:16px;z-index:45;";
  hud.appendChild(panel);

  const chip = (label, active, on) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = label;
    b.className = "comic-pill rounded-full px-2.5 py-1 text-[11px] font-extrabold text-[#16392b]";
    b.style.background = active ? "#e9b349" : "#fffdf6";
    b.style.margin = "2px";
    b.onclick = on;
    return b;
  };
  const row = (title, nodes) => {
    const d = document.createElement("div"); d.style.marginTop = "8px";
    const k = document.createElement("div"); k.className = "kicker"; k.textContent = title;
    const w = document.createElement("div"); w.style.cssText = "display:flex;flex-wrap:wrap;margin-top:3px;";
    nodes.forEach((n) => w.appendChild(n));
    d.append(k, w);
    return d;
  };

  const render = () => {
    panel.innerHTML = "";
    const head = document.createElement("div");
    head.style.cssText = "display:flex;justify-content:space-between;align-items:center;";
    head.innerHTML = '<span class="font-serif-editorial" style="font-size:16px;font-weight:800;">Lemari Baju Banjar</span>';
    const close = chip("✕", false, () => panel.classList.add("hidden"));
    head.appendChild(close);
    panel.appendChild(head);

    const set = (k, v) => () => { W.sel[k] = v; apply(); render(); };
    panel.appendChild(row("Set Lengkap", Object.entries(PRESETS).map(([k, p]) =>
      chip(p.label, false, () => { W.sel = { ...p.sel }; apply(); render(); }))));
    panel.appendChild(row("Atasan", [chip("Tanpa", W.sel.atasan === "none", set("atasan", "none")),
      ...Object.entries(ATASAN).map(([k, a]) => chip(a.label, W.sel.atasan === k, set("atasan", k)))]));
    panel.appendChild(row("Kain / Sarung", [chip("Tanpa", W.sel.kain === "none", set("kain", "none")),
      ...Object.entries(KAIN).map(([k, a]) => chip(a.label, W.sel.kain === k, set("kain", k)))]));
    panel.appendChild(row("Celana", [chip("Pendek", W.sel.celana === "none", set("celana", "none")),
      chip("Celana Hitam", W.sel.celana === "hitam", set("celana", "hitam"))]));
    panel.appendChild(row("Topi", [chip("Tanpa", W.sel.topi === "none", set("topi", "none")),
      ...Object.entries(TOPI).map(([k, a]) => chip(a.label, W.sel.topi === k, set("topi", k)))]));
    panel.appendChild(row("Sepatu", [chip("Kaki Ayam", W.sel.sepatu === "none", set("sepatu", "none")),
      ...Object.entries(SEPATU).map(([k, a]) => chip(a.label, W.sel.sepatu === k, set("sepatu", k)))]));
    panel.appendChild(row("Aksesori", [
      chip("Sabuk Emas", W.sel.sabuk, () => { W.sel.sabuk = !W.sel.sabuk; apply(); render(); }),
      chip("Syal Merah", W.sel.syal, () => { W.sel.syal = !W.sel.syal; apply(); render(); })]));

    // setelan pas
    const tune = document.createElement("details"); tune.style.marginTop = "10px";
    tune.innerHTML = '<summary class="kicker" style="cursor:pointer;">Setelan Pas Baju</summary>';
    const lab = document.createElement("label");
    lab.style.cssText = "display:block;font-size:11px;font-weight:700;margin-top:6px;";
    lab.textContent = "Ketebalan baju";
    const rng = document.createElement("input");
    rng.type = "range"; rng.min = "0.4"; rng.max = "3"; rng.step = "0.1"; rng.value = String(W.fit.infl);
    rng.style.width = "100%";
    rng.oninput = () => { W.fit.infl = parseFloat(rng.value); apply(); };
    lab.appendChild(rng);
    const flip = document.createElement("label");
    flip.style.cssText = "display:flex;gap:6px;align-items:center;font-size:11px;font-weight:700;margin-top:6px;";
    const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = W.fit.flip;
    cb.onchange = () => { W.fit.flip = cb.checked; apply(); };
    flip.append(cb, document.createTextNode("Balik arah depan (jika wajah/motif terbalik)"));
    tune.append(lab, flip);
    panel.appendChild(tune);
  };
  render();
  openBtn.onclick = () => { render(); panel.classList.toggle("hidden"); };
}

/**
 * Dipanggil sekali setelah GLB mannequin dimuat, SEBELUM animasi dimainkan
 * (supaya analisis memakai pose awal / bind pose).
 */
export function initWardrobe(manRoot) {
  try {
    W.root = manRoot;
    W.skinMat = new THREE.MeshStandardMaterial({ color: COL.skin, roughness: 0.75, metalness: 0 });
    manRoot.traverse((n) => {
      if (n.isMesh) { n.material = W.skinMat; n.castShadow = true; n.receiveShadow = true; }
    });
    if (!analyze(manRoot)) { console.warn("[Lemari Baju] SkinnedMesh tidak ditemukan; dilewati."); return; }
    try {
      const saved = JSON.parse(localStorage.getItem("kk_wardrobe") || "null");
      if (saved?.sel) W.sel = { ...W.sel, ...saved.sel };
      if (saved?.fit) W.fit = { ...W.fit, ...saved.fit };
    } catch (_) { /* abaikan */ }
    apply();
    buildUI();
    window.__wardrobe = { W, apply };
  } catch (err) {
    console.warn("[Lemari Baju] gagal:", err);
  }
}
