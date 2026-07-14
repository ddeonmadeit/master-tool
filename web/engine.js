/* Hip-Hop Mix & Master — browser engine (Web Worker).
 *
 * A JavaScript port of the Python pipeline's signature chain: LR4 band splits,
 * RMS-relative soft-knee multiband glue, transient restoration, asymmetric
 * tape-style saturation, dynamic harsh-taming, M/S width with mono lows,
 * auto-release bus glue, then a clipper -> two-stage lookahead true-peak
 * limiter loop converging on the target LUFS (BS.1770 metering), TPDF dither,
 * 24-bit WAV out. Everything runs locally in the browser — audio never leaves
 * the machine.
 */
"use strict";

const SR = 44100;

/* ------------------------------------------------------------------ *
 * Small DSP utilities                                                 *
 * ------------------------------------------------------------------ */
function db2lin(db) { return Math.pow(10, db / 20); }
function lin2db(x) { return 20 * Math.log10(Math.max(x, 1e-12)); }

/* RBJ biquad coefficient makers (normalized, a0 divided out). */
function bqLowpass(fc, Q, fs) {
  const w = 2 * Math.PI * fc / fs, c = Math.cos(w), a = Math.sin(w) / (2 * Q);
  const a0 = 1 + a;
  return [(1 - c) / 2 / a0, (1 - c) / a0, (1 - c) / 2 / a0, (-2 * c) / a0, (1 - a) / a0];
}
function bqHighpass(fc, Q, fs) {
  const w = 2 * Math.PI * fc / fs, c = Math.cos(w), a = Math.sin(w) / (2 * Q);
  const a0 = 1 + a;
  return [(1 + c) / 2 / a0, -(1 + c) / a0, (1 + c) / 2 / a0, (-2 * c) / a0, (1 - a) / a0];
}
function bqPeak(fc, Q, dB, fs) {
  const A = Math.pow(10, dB / 40), w = 2 * Math.PI * fc / fs;
  const c = Math.cos(w), al = Math.sin(w) / (2 * Q);
  const a0 = 1 + al / A;
  return [(1 + al * A) / a0, (-2 * c) / a0, (1 - al * A) / a0, (-2 * c) / a0, (1 - al / A) / a0];
}
function bqLowShelf(fc, Q, dB, fs) {
  const A = Math.pow(10, dB / 40), w = 2 * Math.PI * fc / fs;
  const c = Math.cos(w), al = Math.sin(w) / (2 * Q), sq = 2 * Math.sqrt(A) * al;
  const a0 = (A + 1) + (A - 1) * c + sq;
  return [A * ((A + 1) - (A - 1) * c + sq) / a0, 2 * A * ((A - 1) - (A + 1) * c) / a0,
          A * ((A + 1) - (A - 1) * c - sq) / a0, -2 * ((A - 1) + (A + 1) * c) / a0,
          ((A + 1) + (A - 1) * c - sq) / a0];
}
function bqHighShelf(fc, Q, dB, fs) {
  const A = Math.pow(10, dB / 40), w = 2 * Math.PI * fc / fs;
  const c = Math.cos(w), al = Math.sin(w) / (2 * Q), sq = 2 * Math.sqrt(A) * al;
  const a0 = (A + 1) - (A - 1) * c + sq;
  return [A * ((A + 1) + (A - 1) * c + sq) / a0, -2 * A * ((A - 1) + (A + 1) * c) / a0,
          A * ((A + 1) + (A - 1) * c - sq) / a0, 2 * ((A - 1) - (A + 1) * c) / a0,
          ((A + 1) - (A - 1) * c - sq) / a0];
}

/* Apply one biquad in place (returns same array). */
function biquad(x, k) {
  const [b0, b1, b2, a1, a2] = k;
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    const y = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = xi; y2 = y1; y1 = y;
    x[i] = y;
  }
  return x;
}
function biquadST(ch, k) { biquad(ch[0], k); biquad(ch[1], k); return ch; }

function copyST(ch) { return [ch[0].slice(), ch[1].slice()]; }

/* Linkwitz-Riley 4th order split -> [low, high]; low+high reconstructs flat. */
function lr4Split(ch, fc) {
  const Q = Math.SQRT1_2;
  const lp = bqLowpass(fc, Q, SR), hp = bqHighpass(fc, Q, SR);
  const low = copyST(ch), high = copyST(ch);
  biquadST(low, lp); biquadST(low, lp);
  biquadST(high, hp); biquadST(high, hp);
  return [low, high];
}
/* Phase-coherent band isolate: returns [band(lo..hi), rest]. */
function splitBand(ch, lo, hi) {
  const [belowHi, aboveHi] = lr4Split(ch, hi);
  const [belowLo, band] = lr4Split(belowHi, lo);
  const rest = belowLo;
  for (let c = 0; c < 2; c++)
    for (let i = 0; i < rest[c].length; i++) rest[c][i] += aboveHi[c][i];
  return [band, rest];
}

function onepole(x, ms) {
  const a = Math.exp(-1 / (Math.max(ms, 1e-3) * 1e-3 * SR)), b = 1 - a;
  const y = new Float32Array(x.length);
  let s = 0;
  for (let i = 0; i < x.length; i++) { s = b * x[i] + a * s; y[i] = s; }
  return y;
}
/* Fast-attack / slow-release peak envelope over a stereo pair (mono env out). */
function envFollower(ch, atkMs, relMs) {
  const n = ch[0].length, rect = new Float32Array(n);
  for (let i = 0; i < n; i++) rect[i] = Math.max(Math.abs(ch[0][i]), Math.abs(ch[1][i]));
  const fast = onepole(rect, atkMs), slow = onepole(rect, relMs);
  for (let i = 0; i < n; i++) fast[i] = Math.max(fast[i], slow[i]);
  return fast;
}

/* Soft-knee downward-compressor gain reduction (dB >= 0). */
function softkneeGR(envDb, i, thr, ratio, knee) {
  const slope = 1 - 1 / Math.max(ratio, 1);
  const x = envDb - thr;
  if (x >= knee / 2) return x * slope;
  if (x > -knee / 2) { const kx = x + knee / 2; return slope * kx * kx / (2 * knee); }
  return 0;
}

function gainST(ch, g) {
  for (let c = 0; c < 2; c++) { const a = ch[c]; for (let i = 0; i < a.length; i++) a[i] *= g; }
  return ch;
}

/* ------------------------------------------------------------------ *
 * 2x oversampling (halfband FIR; near-zero taps skipped)              *
 * ------------------------------------------------------------------ */
const HALFBAND = new Float32Array([
  -8.22099723e-04, 4.71362655e-18, 9.82233068e-04, -1.97201761e-18, -1.38373907e-03, -3.31038850e-18,
  2.07001148e-03, -3.72516133e-18, -3.08972408e-03, 1.83357693e-17, 4.50049353e-03, -6.58182301e-18,
  -6.37528983e-03, 8.28436370e-18, 8.81377942e-03, -1.00789824e-17, -1.19630973e-02, 1.18922073e-17,
  1.60580621e-02, -1.36498047e-17, -2.15054006e-02, 1.52798183e-17, 2.90802149e-02, -1.67155152e-17,
  -4.04588904e-02, 1.78981179e-17, 6.00300973e-02, -1.87792103e-17, -1.03947747e-01, 1.93227206e-17,
  3.17811797e-01, 5.00398599e-01, 3.17811797e-01, 1.93227206e-17, -1.03947747e-01, -1.87792103e-17,
  6.00300973e-02, 1.78981179e-17, -4.04588904e-02, -1.67155152e-17, 2.90802149e-02, 1.52798183e-17,
  -2.15054006e-02, -1.36498047e-17, 1.60580621e-02, 1.18922073e-17, -1.19630973e-02, -1.00789824e-17,
  8.81377942e-03, 8.28436370e-18, -6.37528983e-03, -6.58182301e-18, 4.50049353e-03, 1.83357693e-17,
  -3.08972408e-03, -3.72516133e-18, 2.07001148e-03, -3.31038850e-18, -1.38373907e-03, -1.97201761e-18,
  9.82233068e-04, 4.71362655e-18, -8.22099723e-04,
]);
const HB_IDX = [], HB_VAL = [];
for (let i = 0; i < HALFBAND.length; i++)
  if (Math.abs(HALFBAND[i]) > 1e-9) { HB_IDX.push(i); HB_VAL.push(HALFBAND[i]); }
const HB_DELAY = (HALFBAND.length - 1) >> 1;

function firSparse(x) {
  const n = x.length, y = new Float32Array(n);
  const m = HB_IDX.length;
  for (let i = 0; i < n; i++) {
    let acc = 0;
    for (let t = 0; t < m; t++) {
      const j = i - HB_IDX[t] + HB_DELAY;
      if (j >= 0 && j < n) acc += HB_VAL[t] * x[j];
    }
    y[i] = acc;
  }
  return y;
}
function upsample2(x) {
  const up = new Float32Array(x.length * 2);
  for (let i = 0; i < x.length; i++) up[2 * i] = 2 * x[i];
  return firSparse(up);
}
function downsample2(x, outLen) {
  const f = firSparse(x), y = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) y[i] = f[2 * i] || 0;
  return y;
}

/* ------------------------------------------------------------------ *
 * BS.1770 loudness + true peak                                        *
 * ------------------------------------------------------------------ */
function kWeightCoeffs() {
  return [bqHighShelf(1681.9744509555319, 0.7071752369, 3.99984385397, SR),
          bqHighpass(38.13547087602444, 0.5003270373238773, SR)];
}
function integratedLUFS(ch) {
  const [shelf, hp] = kWeightCoeffs();
  const w = copyST(ch);
  biquadST(w, shelf); biquadST(w, hp);
  const block = Math.round(0.4 * SR), hop = Math.round(0.1 * SR);
  const n = w[0].length, blocks = [];
  for (let s = 0; s + block <= n; s += hop) {
    let z = 0;
    for (let c = 0; c < 2; c++) { const a = w[c]; for (let i = s; i < s + block; i++) z += a[i] * a[i]; }
    blocks.push(z / block);
  }
  if (!blocks.length) return -Infinity;
  const loud = blocks.map(z => -0.691 + 10 * Math.log10(Math.max(z, 1e-12)));
  let keep = blocks.filter((_, i) => loud[i] > -70);
  if (!keep.length) return -Infinity;
  let mean = keep.reduce((a, b) => a + b, 0) / keep.length;
  const gate = -0.691 + 10 * Math.log10(mean) - 10;
  keep = blocks.filter((_, i) => loud[i] > -70 && (-0.691 + 10 * Math.log10(Math.max(blocks[i], 1e-12))) > gate);
  if (!keep.length) return -Infinity;
  mean = keep.reduce((a, b) => a + b, 0) / keep.length;
  return -0.691 + 10 * Math.log10(mean);
}
function truePeakDb(ch) {
  let p = 0;
  for (let c = 0; c < 2; c++) {
    const up = upsample2(ch[c]);
    for (let i = 0; i < up.length; i++) { const a = Math.abs(up[i]); if (a > p) p = a; }
  }
  return lin2db(p);
}

/* ------------------------------------------------------------------ *
 * Processors (ports of pipeline/dsp.py)                               *
 * ------------------------------------------------------------------ */
function compressBand(band, p) {
  let rms = 0; const n = band[0].length;
  for (let c = 0; c < 2; c++) for (let i = 0; i < n; i++) rms += band[c][i] * band[c][i];
  rms = Math.sqrt(rms / (2 * n));
  if (rms < 1e-9) return band;
  const thr = lin2db(rms) + p.thrOff;
  const env = envFollower(band, p.atk, p.rel);
  for (let i = 0; i < n; i++) {
    const gr = Math.min(softkneeGR(lin2db(env[i]), i, thr, p.ratio, 8), p.maxGR);
    const g = db2lin(p.makeup - gr);
    band[0][i] *= g; band[1][i] *= g;
  }
  return band;
}
function multibandCompress(ch) {
  const [belowHi, high] = lr4Split(ch, 3000);
  const [low, mid] = lr4Split(belowHi, 110);
  compressBand(low,  { ratio: 2.5, atk: 22, rel: 200, thrOff: 4, makeup: 1.2, maxGR: 4 });
  compressBand(mid,  { ratio: 2.0, atk: 25, rel: 160, thrOff: 6, makeup: 0.3, maxGR: 3 });
  compressBand(high, { ratio: 2.0, atk: 6,  rel: 90,  thrOff: 6, makeup: 0.4, maxGR: 3 });
  const out = low;
  for (let c = 0; c < 2; c++)
    for (let i = 0; i < out[c].length; i++) out[c][i] += mid[c][i] + high[c][i];
  return out;
}

function transientEnhance(ch, amount, maxBoost) {
  const fast = envFollower(ch, 1, 45), slow = envFollower(ch, 20, 45);
  const n = ch[0].length;
  for (let i = 0; i < n; i++) {
    const d = 20 * Math.log10(Math.max(fast[i], 1e-9) / Math.max(slow[i], 1e-9));
    const b = Math.min(Math.max(d * amount, 0), maxBoost);
    const g = db2lin(b);
    ch[0][i] *= g; ch[1][i] *= g;
  }
  return ch;
}

function asymSat(ch, drive, asym) {
  if (drive <= 1e-4) return ch;
  const k = 1 + 4 * drive, b = asym * drive;
  const tb = Math.tanh(b), tk = Math.tanh(k);
  const mix = Math.min(Math.max(0.5 + 0.5 * Math.min(drive, 1), 0.5), 1);
  for (let c = 0; c < 2; c++) {
    const x = ch[c], up = upsample2(x);
    for (let i = 0; i < up.length; i++) up[i] = (Math.tanh(up[i] * k + b) - tb) / tk;
    const dn = downsample2(up, x.length);
    for (let i = 0; i < x.length; i++) x[i] = mix * dn[i] + (1 - mix) * x[i];
  }
  return ch;
}
function tapeSaturate(ch, drive) {
  if (drive <= 1e-4) return ch;
  const [low, high] = lr4Split(ch, 3800);
  asymSat(low, drive, 0.35);
  asymSat(high, drive * 0.4, 0.15);
  for (let c = 0; c < 2; c++)
    for (let i = 0; i < low[c].length; i++) ch[c][i] = low[c][i] + high[c][i];
  return ch;
}

function dynamicBandReduction(ch, lo, hi, thrDb, ratio, maxRed, atk, rel) {
  const [band, rest] = splitBand(ch, lo, hi);
  const env = envFollower(band, atk, rel);
  const n = ch[0].length;
  for (let i = 0; i < n; i++) {
    const gr = Math.min(softkneeGR(lin2db(env[i]), i, thrDb, ratio, 6), maxRed);
    const g = db2lin(-gr);
    ch[0][i] = rest[0][i] + band[0][i] * g;
    ch[1][i] = rest[1][i] + band[1][i] * g;
  }
  return ch;
}

function hfExciter(ch, freq, amount) {
  if (amount <= 1e-4) return ch;
  const hp = copyST(ch);
  const k = bqHighpass(freq, Math.SQRT1_2, SR);
  biquadST(hp, k);
  let rms = 0; const n = hp[0].length;
  for (let c = 0; c < 2; c++) for (let i = 0; i < n; i++) rms += hp[c][i] * hp[c][i];
  rms = Math.sqrt(rms / (2 * n));
  if (rms < 1e-9) return ch;
  const g = db2lin(-12) / rms, drive = 0.6, kk = 1 + 4 * drive, tk = Math.tanh(kk);
  for (let c = 0; c < 2; c++)
    for (let i = 0; i < n; i++) {
      const v = hp[c][i] * g;
      const sat = Math.tanh(v * kk) / tk / g;
      ch[c][i] += amount * (sat - hp[c][i]);
    }
  return ch;
}

function applyWidth(ch, width, monoBelow, widthAbove) {
  const n = ch[0].length;
  const mid = new Float32Array(n), side = new Float32Array(n);
  for (let i = 0; i < n; i++) { mid[i] = (ch[0][i] + ch[1][i]) / 2; side[i] = (ch[0][i] - ch[1][i]) / 2; }
  const hp = bqHighpass(monoBelow, Math.SQRT1_2, SR);
  biquad(side, hp);
  const sPair = [side, side.slice()];
  const [sLow, sHigh] = lr4Split([side, new Float32Array(n)], widthAbove);
  const wGain = 1 + 0.8 * width;
  for (let i = 0; i < n; i++) side[i] = sLow[0][i] + sHigh[0][i] * wGain;
  for (let i = 0; i < n; i++) { ch[0][i] = mid[i] + side[i]; ch[1][i] = mid[i] - side[i]; }
  return ch;
}

function busGlue(ch, thrDb, ratio) {
  const env = envFollower(ch, 25, 120);
  const n = ch[0].length;
  const gr = new Float32Array(n);
  for (let i = 0; i < n; i++)
    gr[i] = Math.min(softkneeGR(lin2db(env[i]), i, thrDb, ratio, 8), 3);
  const fast = onepole(gr, 90), slow = onepole(gr, 550);
  for (let i = 0; i < n; i++) {
    const g = db2lin(0.4 - Math.max(fast[i], slow[i]));
    ch[0][i] *= g; ch[1][i] *= g;
  }
  return ch;
}

/* Lookahead limiter with sliding-min anticipation and dual (program-dependent)
 * release. Operates at the CURRENT rate of `ch` (caller oversamples). */
function lookaheadLimit(ch, fs, ceilDb, lookMs, relMs) {
  const ceil = db2lin(ceilDb), n = ch[0].length;
  const desired = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = Math.max(Math.abs(ch[0][i]), Math.abs(ch[1][i]));
    desired[i] = p > ceil ? ceil / p : 1;
  }
  const look = Math.max(1, Math.round(lookMs * 1e-3 * fs));
  const W = 2 * look + 1;
  const g = new Float32Array(n);
  // sliding-window minimum (monotonic deque)
  const qi = new Int32Array(n); let head = 0, tail = 0;
  for (let i = 0; i < n + look; i++) {
    if (i < n) {
      while (tail > head && desired[qi[tail - 1]] >= desired[i]) tail--;
      qi[tail++] = i;
    }
    const lo = i - W + 1;
    while (qi[head] < lo) head++;
    const out = i - look;
    if (out >= 0 && out < n) g[out] = desired[qi[head]];
  }
  const red = new Float32Array(n);
  for (let i = 0; i < n; i++) red[i] = Math.min(Math.max(1 - g[i], 0), 1);
  const smoothOne = (arr, atkMs, rMs) => {
    const fa = onepole(arr, atkMs), so = onepole(arr, rMs);
    for (let i = 0; i < n; i++) fa[i] = Math.max(fa[i], so[i]);
    return fa;
  };
  const rf = smoothOne(red, 0.2, relMs), rs = smoothOne(red, 0.2, relMs * 3.5);
  for (let i = 0; i < n; i++) {
    const r = Math.max(rf[i], rs[i] * 0.9);
    const gg = 1 - r;
    let l = ch[0][i] * gg, rr2 = ch[1][i] * gg;
    ch[0][i] = Math.min(Math.max(l, -ceil), ceil);
    ch[1][i] = Math.min(Math.max(rr2, -ceil), ceil);
  }
  return ch;
}

/* Soft clip toward ceiling (knee below it), blended by `amount`. In place. */
function softClipInPlace(ch, ceilDb, kneeDb, amount) {
  if (amount <= 1e-6) return ch;
  const c = db2lin(ceilDb), knee = db2lin(-kneeDb);
  for (let cc = 0; cc < 2; cc++) {
    const x = ch[cc];
    for (let i = 0; i < x.length; i++) {
      const y = x[i] / c, a = Math.abs(y);
      if (a > knee) {
        const t = (a - knee) / (1 - knee);
        const shaped = (Math.sign(y) * (knee + (1 - knee) * Math.tanh(t))) * c;
        x[i] = amount * shaped + (1 - amount) * x[i];
      }
    }
  }
  return ch;
}

/* Clip + two-stage limit sharing one 2x-oversampled pass. */
function clipAndLimit(ch, ceilDb, clipAmount) {
  const n = ch[0].length, fs2 = SR * 2;
  const up = [upsample2(ch[0]), upsample2(ch[1])];
  softClipInPlace(up, ceilDb, 3, clipAmount);
  lookaheadLimit(up, fs2, ceilDb - 0.1, 2.0, 180);
  lookaheadLimit(up, fs2, ceilDb - 0.1, 0.8, 40);
  const ceil = db2lin(ceilDb);
  for (let c = 0; c < 2; c++) {
    const dn = downsample2(up[c], n);
    for (let i = 0; i < n; i++) ch[c][i] = Math.min(Math.max(dn[i], -ceil), ceil);
  }
  return ch;
}

/* ------------------------------------------------------------------ *
 * Sounds (voicing) + signature chain + finalize                       *
 * ------------------------------------------------------------------ */
const SOUNDS = {
  trap:    { lowW: 2.6, satM: 1.35, widM: 1.15, pres: 0.7, air: 1.0, dyn: 1.0,
             voicing: [["ls", 80, 0.7, 1.2], ["pk", 3500, 0.8, -0.5]] },
  boombap: { lowW: 0.4, satM: 1.2, widM: 0.65, pres: 2.3, air: 0.4, dyn: 1.35,
             voicing: [["pk", 250, 0.9, 0.8], ["pk", 2800, 0.8, 0.6], ["hs", 9000, 0.7, -0.8]] },
  melodic: { lowW: 1.0, satM: 0.7, widM: 1.35, pres: 1.2, air: 2.8, dyn: 1.05,
             voicing: [["ls", 70, 0.7, 0.6], ["hs", 10000, 0.7, 1.0]] },
};

function signatureChain(ch, st, prog) {
  const prof = SOUNDS[st.sound] || SOUNDS.trap;
  // 1. clarity
  biquadST(ch, bqHighpass(26, Math.SQRT1_2, SR));
  biquadST(ch, bqPeak(300, 1.0, -0.8, SR));
  // 1b. fixed voicing (Lite stand-in for the spectrum-matching EQ)
  for (const [kind, f, q, g] of prof.voicing) {
    const mk = kind === "ls" ? bqLowShelf : kind === "hs" ? bqHighShelf : bqPeak;
    biquadST(ch, mk(f, q, g, SR));
  }
  prog(0.15);
  // 2. multiband glue + transient restore
  ch = multibandCompress(ch);
  prog(0.4);
  transientEnhance(ch, 0.3, 2.0);
  prog(0.5);
  // 3. Kanye house character: tape saturation + head bump + weight + body
  const drive = 0.5 * st.warmth * prof.satM;
  tapeSaturate(ch, drive);
  biquadST(ch, bqPeak(62, 1.1, 0.5 + 0.6 * drive, SR));
  biquadST(ch, bqLowShelf(90, 0.7, 1.2 + 1.0 * st.warmth + 0.7 * prof.lowW, SR));
  biquadST(ch, bqPeak(220, 0.9, 1.0, SR));
  biquadST(ch, bqHighpass(18, 0.5, SR));       // DC block
  prog(0.72);
  // 4. top end: tame harsh, presence, air (+untiljapan baseline), exciter
  dynamicBandReduction(ch, 5000, 9000, -16, 2.0, 2.0, 1, 80);
  biquadST(ch, bqPeak(2800, 0.6, prof.pres, SR));
  biquadST(ch, bqHighShelf(11000, 0.6, prof.air + 0.4, SR));
  hfExciter(ch, 9500, 0.12);
  prog(0.88);
  // 5. width (lows mono)
  const effW = Math.min(Math.max(st.width * prof.widM + 0.05, 0), 1);
  applyWidth(ch, effW, 120, 300);
  // 6. auto-release bus glue
  const thr = -14 + (prof.dyn - 1) * 16;
  const ratio = Math.max(1.3, 1.8 - (prof.dyn - 1) * 1.6);
  busGlue(ch, thr, ratio);
  prog(1.0);
  return ch;
}

function finalize(ch, target, prog) {
  const ceil = -1.0;
  let prevGap = null;
  for (let i = 0; i < 6; i++) {
    prog(Math.min(i / 3, 0.95));
    const cur = integratedLUFS(ch);
    if (!isFinite(cur)) break;
    const diff = target - cur;
    if (Math.abs(diff) <= 0.15) break;
    if (prevGap !== null && Math.abs(diff) > prevGap - 0.08) break;
    prevGap = Math.abs(diff);
    const step = i === 0 ? diff : Math.min(Math.max(diff, -3), 2);
    gainST(ch, db2lin(step));
    clipAndLimit(ch, ceil, 0.65);
  }
  const tp = truePeakDb(ch);
  if (tp > ceil) gainST(ch, db2lin(ceil - tp - 0.1));
  // TPDF dither @24-bit
  const lsb = Math.pow(2, -23);
  for (let c = 0; c < 2; c++) {
    const x = ch[c];
    for (let i = 0; i < x.length; i++) {
      x[i] += (Math.random() - Math.random()) * lsb;
      x[i] = Math.min(Math.max(x[i], -1), 1);
    }
  }
  return ch;
}

/* ------------------------------------------------------------------ *
 * Mix (stems) + WAV encode + report                                   *
 * ------------------------------------------------------------------ */
function mixStems(voc, instr, vocalLevelDb) {
  // vocal -> mono, HPF 90, de-ess, LUFS-match to instr +1 LU + user offset, sum
  const n = Math.max(voc[0].length, instr[0].length);
  const pad = (a) => { const o = new Float32Array(n); o.set(a); return o; };
  const vm = new Float32Array(n);
  const v0 = pad(voc[0]), v1 = pad(voc[1]);
  for (let i = 0; i < n; i++) vm[i] = (v0[i] + v1[i]) / 2;
  const vch = [vm, vm.slice()];
  biquadST(vch, bqHighpass(90, Math.SQRT1_2, SR));
  dynamicBandReduction(vch, 6000, 8500, -26, 3.0, 2.5, 0.5, 50);
  const ich = [pad(instr[0]), pad(instr[1])];
  const vL = integratedLUFS(vch), iL = integratedLUFS(ich);
  if (isFinite(vL) && isFinite(iL)) gainST(vch, db2lin(iL + 1 + vocalLevelDb - vL));
  const bus = [new Float32Array(n), new Float32Array(n)];
  for (let c = 0; c < 2; c++)
    for (let i = 0; i < n; i++) bus[c][i] = vch[c][i] + ich[c][i];
  // -6 dBFS headroom
  let peak = 0;
  for (let c = 0; c < 2; c++) for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(bus[c][i]));
  if (peak > 0) gainST(bus, db2lin(-6) / peak);
  return bus;
}

function encodeWav24(ch) {
  const n = ch[0].length, bytes = 44 + n * 2 * 3;
  const buf = new ArrayBuffer(bytes), v = new DataView(buf);
  const wstr = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  wstr(0, "RIFF"); v.setUint32(4, bytes - 8, true); wstr(8, "WAVE");
  wstr(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 2, true);
  v.setUint32(24, SR, true); v.setUint32(28, SR * 6, true); v.setUint16(32, 6, true); v.setUint16(34, 24, true);
  wstr(36, "data"); v.setUint32(40, n * 6, true);
  let o = 44;
  for (let i = 0; i < n; i++)
    for (let c = 0; c < 2; c++) {
      let s = Math.round(Math.min(Math.max(ch[c][i], -1), 1) * 8388607);
      v.setUint8(o++, s & 255); v.setUint8(o++, (s >> 8) & 255); v.setUint8(o++, (s >> 16) & 255);
    }
  return buf;
}

function makeReport(ch, target) {
  const lufs = integratedLUFS(ch), tp = truePeakDb(ch);
  const n = ch[0].length;
  let sl = 0, sr_ = 0, slr = 0, sll = 0, srr = 0;
  for (let i = 0; i < n; i++) { sl += ch[0][i]; sr_ += ch[1][i]; }
  const ml = sl / n, mr = sr_ / n;
  for (let i = 0; i < n; i++) {
    const a = ch[0][i] - ml, b = ch[1][i] - mr;
    slr += a * b; sll += a * a; srr += b * b;
  }
  const corr = slr / Math.max(Math.sqrt(sll * srr), 1e-12);
  return {
    integrated_lufs: Math.round(lufs * 100) / 100,
    true_peak_dbtp: Math.round(tp * 100) / 100,
    stereo_correlation: Math.round(corr * 1000) / 1000,
    checks: {
      loudness_on_target: Math.abs(lufs - target) <= 0.5,
      true_peak_safe: tp <= -0.95,
      correlation_non_negative: corr >= 0,
    },
  };
}

/* ------------------------------------------------------------------ *
 * Worker entry                                                        *
 * ------------------------------------------------------------------ */
function runMaster(msg, post) {
  const st = msg.settings;
  const prog = (stage, pct) => post({ type: "progress", stage, pct: Math.round(pct) });
  let bus;
  prog("mix", 2);
  if (msg.mode === "stems") {
    bus = mixStems([msg.vocal.L, msg.vocal.R], [msg.instr.L, msg.instr.R], st.vocalLevelDb || 0);
  } else {
    const n = msg.track.L.length;
    bus = [msg.track.L.slice(), msg.track.R.slice()];
    let peak = 0;
    for (let c = 0; c < 2; c++) for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(bus[c][i]));
    if (peak > 0) gainST(bus, db2lin(-6) / peak);
  }
  prog("master", 12);
  const pre = [bus[0].slice(), bus[1].slice()];
  bus = signatureChain(bus, st, f => prog("master", 12 + 58 * f));
  prog("loudness", 72);
  finalize(bus, st.target, f => prog("loudness", 72 + 22 * f));
  prog("report", 95);
  const report = makeReport(bus, st.target);
  const wav = encodeWav24(bus);
  const preWav = encodeWav24(pre);
  prog("done", 100);
  post({ type: "done", wav, preWav, report }, [wav, preWav]);
}

if (typeof self !== "undefined" && typeof self.postMessage === "function") {
  self.onmessage = (e) => {
    try {
      runMaster(e.data, (m, tr) => self.postMessage(m, tr || []));
    } catch (err) {
      self.postMessage({ type: "error", message: String(err && err.message || err) });
    }
  };
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = { runMaster, integratedLUFS, truePeakDb, SR };
}
