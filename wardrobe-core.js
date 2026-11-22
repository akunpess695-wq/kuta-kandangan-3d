// wardrobe-core.js — analisis rangka mannequin tanpa bergantung pada nama tulang.
// Semua koordinat berada di ruang lokal manRoot (pose bind / pose awal).
// Kelas bagian tubuh: 0 = badan/kepala, 1 = lengan, 2 = kaki.

export const CLS_TORSO = 0;
export const CLS_ARM = 1;
export const CLS_LEG = 2;

export const FIT_DEFAULTS = {
  armAx: 0.085,   // jarak tulang dari sumbu tengah (x H) untuk dianggap lengan
  legBy: 0.51,    // tulang di bawah tinggi ini (x H) dianggap kaki
  footTop: 0.058, // tinggi telapak kaki (x H)
  waist: 0.565,   // pinggang (x H)
  knee: 0.27      // ujung bawah kain/sarung (x H)
};

/**
 * @param {Object} p
 * @param {{rp:Float32Array, dom:Uint16Array}[]} p.entries  posisi vertex (root space) + tulang dominan per vertex
 * @param {{x:number,y:number,z:number}[]} p.bones          posisi tulang (root space)
 */
export function analyzeRig({ entries, bones }, opt = {}) {
  const F = { ...FIT_DEFAULTS, ...opt };
  let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
  for (const e of entries) {
    const a = e.rp;
    for (let i = 0; i < a.length; i += 3) {
      if (a[i] < minx) minx = a[i]; if (a[i] > maxx) maxx = a[i];
      if (a[i + 1] < miny) miny = a[i + 1]; if (a[i + 1] > maxy) maxy = a[i + 1];
    }
  }
  const H = maxy - miny;
  const cx = (minx + maxx) / 2;
  const y0 = miny;

  // --- klasifikasi tulang berdasar posisi ---
  const boneCls = new Uint8Array(bones.length);
  for (let i = 0; i < bones.length; i++) {
    const ax = Math.abs(bones[i].x - cx) / H;
    const by = (bones[i].y - y0) / H;
    if (ax > F.armAx && by > 0.36) boneCls[i] = CLS_ARM;
    else if (by < F.legBy && ax > 0.012) boneCls[i] = CLS_LEG;
    else boneCls[i] = CLS_TORSO;
  }

  // --- bahu & panjang lengan per sisi ---
  const S = { '-1': null, '1': null };
  const armLen = { '-1': 0, '1': 0 };
  for (const sd of [-1, 1]) {
    let best = null, bestAx = Infinity;
    for (let i = 0; i < bones.length; i++) {
      if (boneCls[i] !== CLS_ARM) continue;
      if (Math.sign(bones[i].x - cx) !== sd) continue;
      const ax = Math.abs(bones[i].x - cx);
      if (ax < bestAx) { bestAx = ax; best = bones[i]; }
    }
    S[sd] = best;
    if (best) {
      for (let i = 0; i < bones.length; i++) {
        if (boneCls[i] !== CLS_ARM || Math.sign(bones[i].x - cx) !== sd) continue;
        const d = Math.hypot(bones[i].x - best.x, bones[i].y - best.y, bones[i].z - best.z);
        if (d > armLen[sd]) armLen[sd] = d;
      }
    }
  }
  const shoulderY = (S[-1] && S[1]) ? (S[-1].y + S[1].y) / 2
    : (S[-1] || S[1]) ? (S[-1] || S[1]).y : y0 + 0.82 * H;

  // --- tulang kepala: tulang tengah tertinggi di antara 0.80H..0.93H ---
  let headIdx = -1, headY = -Infinity;
  for (let i = 0; i < bones.length; i++) {
    const ax = Math.abs(bones[i].x - cx) / H;
    const by = (bones[i].y - y0) / H;
    if (ax < 0.03 && by >= 0.80 && by <= 0.93 && bones[i].y > headY) { headY = bones[i].y; headIdx = i; }
  }
  if (headIdx < 0) headY = y0 + 0.865 * H;

  // --- klasifikasi vertex ---
  const per = entries.map((e) => {
    const n = e.rp.length / 3;
    const cls = new Uint8Array(n), side = new Int8Array(n), dn = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const x = e.rp[i * 3], y = e.rp[i * 3 + 1], z = e.rp[i * 3 + 2];
      cls[i] = boneCls[e.dom[i]] ?? 0;
      const sd = x >= cx ? 1 : -1;
      side[i] = sd;
      const s = S[sd];
      dn[i] = (s && armLen[sd] > 0) ? Math.hypot(x - s.x, y - s.y, z - s.z) / armLen[sd] : 0;
    }
    return { cls, side, dn };
  });

  // --- kepala: lebar/dalam & posisi dagu ---
  const headTop = maxy;
  let hx0 = Infinity, hx1 = -Infinity, hz0 = Infinity, hz1 = -Infinity, hzSum = 0, hN = 0;
  const bins = new Map(); // lebar per lapisan y
  const BIN = 0.004 * H;
  entries.forEach((e, k) => {
    const c = per[k].cls;
    for (let i = 0; i < c.length; i++) {
      if (c[i] !== CLS_TORSO) continue;
      const y = e.rp[i * 3 + 1];
      if (y < headY) continue;
      const x = e.rp[i * 3];
      const b = Math.floor((y - headY) / BIN);
      const r = bins.get(b) || [Infinity, -Infinity];
      if (x < r[0]) r[0] = x; if (x > r[1]) r[1] = x;
      bins.set(b, r);
    }
  });
  let maxW = 0;
  for (const r of bins.values()) maxW = Math.max(maxW, r[1] - r[0]);
  let chinY = headY + 0.02 * H;
  const keys = [...bins.keys()].sort((a, b) => a - b);
  for (const b of keys) {
    const r = bins.get(b);
    if (r[1] - r[0] >= 0.55 * maxW) { chinY = headY + b * BIN; break; }
  }
  entries.forEach((e, k) => {
    const c = per[k].cls;
    for (let i = 0; i < c.length; i++) {
      if (c[i] !== CLS_TORSO) continue;
      const y = e.rp[i * 3 + 1];
      if (y < chinY) continue;
      const x = e.rp[i * 3], z = e.rp[i * 3 + 2];
      if (x < hx0) hx0 = x; if (x > hx1) hx1 = x;
      if (z < hz0) hz0 = z; if (z > hz1) hz1 = z;
      hzSum += z; hN++;
    }
  });
  const head = {
    cx: (hx0 + hx1) / 2, cz: (hz0 + hz1) / 2, w: hx1 - hx0, d: hz1 - hz0,
    chinY, topY: headTop, h: headTop - chinY
  };

  // --- arah depan: telapak kaki menjorok ke depan dari pergelangan ---
  let ankleZ = null, nA = 0, azs = 0;
  for (let i = 0; i < bones.length; i++) {
    const by = (bones[i].y - y0) / H;
    if (boneCls[i] === CLS_LEG && by >= 0.03 && by <= 0.10) { azs += bones[i].z; nA++; }
  }
  if (nA) ankleZ = azs / nA;
  let fz = 0, fn = 0;
  entries.forEach((e, k) => {
    const c = per[k].cls;
    for (let i = 0; i < c.length; i++) {
      if (c[i] === CLS_LEG && (e.rp[i * 3 + 1] - y0) / H < 0.03) { fz += e.rp[i * 3 + 2]; fn++; }
    }
  });
  let frontSign = 1;
  if (ankleZ !== null && fn > 0) frontSign = (fz / fn - ankleZ) >= 0 ? 1 : -1;

  return { H, y0, cx, minx, maxx, miny, maxy, bones: bones.length, boneCls, S, armLen,
    shoulderY, headIdx, headY, head, frontSign, F, per };
}
