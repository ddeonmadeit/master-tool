"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const files = { track: null, vocal: null, instrumental: null };
let inputKind = "track";
let sound = "trap";
let worker = null;

/* ---------------- drops ---------------- */
function wireDrop(el) {
  const kind = el.dataset.kind;
  const input = el.querySelector("input[type=file]");
  const show = (f) => {
    files[kind] = f;
    el.querySelector(".drop-file").textContent = f ? f.name : "";
    el.classList.toggle("loaded", !!f);
    refresh();
  };
  el.addEventListener("click", () => input.click());
  input.addEventListener("change", () => show(input.files[0] || null));
  ["dragenter", "dragover"].forEach((e) => el.addEventListener(e, (ev) => { ev.preventDefault(); el.classList.add("over"); }));
  ["dragleave", "drop"].forEach((e) => el.addEventListener(e, (ev) => { ev.preventDefault(); el.classList.remove("over"); }));
  el.addEventListener("drop", (ev) => { const f = ev.dataTransfer.files[0]; if (f) show(f); });
}
$$(".drop").forEach(wireDrop);

/* ---------------- segmented controls ---------------- */
$$("#input-mode .seg-btn").forEach((b) => b.addEventListener("click", () => {
  $$("#input-mode .seg-btn").forEach((x) => x.classList.remove("active"));
  b.classList.add("active");
  inputKind = b.dataset.input;
  $("#drops-track").hidden = inputKind !== "track";
  $("#drops-stems").hidden = inputKind !== "stems";
  $("#row-vocal").hidden = inputKind !== "stems";
  refresh();
}));
$$("#sound .seg-btn").forEach((b) => b.addEventListener("click", () => {
  $$("#sound .seg-btn").forEach((x) => x.classList.remove("active"));
  b.classList.add("active");
  sound = b.dataset.sound;
}));

/* ---------------- sliders ---------------- */
const widthWord = (v) => v < 0.33 ? "subtle" : v < 0.66 ? "medium" : "wide";
function bindSlider(id, valId, fmt) {
  const el = $(id);
  const upd = () => ($(valId).textContent = fmt(parseFloat(el.value)));
  el.addEventListener("input", upd); upd();
}
bindSlider("#s-vocal", "#v-vocal", (v) => (v >= 0 ? "+" : "") + v.toFixed(1) + " dB");
bindSlider("#s-width", "#v-width", widthWord);
bindSlider("#s-warmth", "#v-warmth", (v) => v.toFixed(2));
bindSlider("#s-loud", "#v-loud", (v) => v.toFixed(1) + " LUFS");

function refresh() {
  const ok = inputKind === "track" ? !!files.track : !!(files.vocal && files.instrumental);
  $("#master-btn").disabled = !ok;
}

/* ---------------- decode ---------------- */
async function decodeTo44k(file) {
  const buf = await file.arrayBuffer();
  const ctx = new OfflineAudioContext(2, 44100, 44100);
  const ab = await ctx.decodeAudioData(buf);
  const L = ab.getChannelData(0).slice();
  const R = (ab.numberOfChannels > 1 ? ab.getChannelData(1) : ab.getChannelData(0)).slice();
  if (ab.sampleRate !== 44100) {
    // decodeAudioData resampled to the context rate (44100) in modern browsers;
    // if a browser kept the native rate, resample linearly as a fallback.
    const ratio = ab.sampleRate / 44100;
    const n = Math.floor(L.length / ratio);
    const rl = new Float32Array(n), rr = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const p = i * ratio, j = Math.floor(p), f = p - j;
      rl[i] = L[j] * (1 - f) + (L[j + 1] || 0) * f;
      rr[i] = R[j] * (1 - f) + (R[j + 1] || 0) * f;
    }
    return { L: rl, R: rr };
  }
  return { L, R };
}

/* ---------------- progress ---------------- */
const STAGE_LABELS = {
  decode: "Decoding files", mix: "Preparing mix", master: "Mastering",
  loudness: "Loudness & limiting", report: "Finishing", done: "Done",
};
let startT = 0;
function showProgress(stage, pct) {
  $("#prog-wrap").hidden = false;
  $("#prog-bar").style.width = pct + "%";
  const el = Math.round((performance.now() - startT) / 1000);
  let txt = `${STAGE_LABELS[stage] || stage} · ${pct}% · ${el}s`;
  if (pct >= 15 && pct < 100 && el > 2) {
    txt += ` · ~${Math.max(1, Math.round(el * (100 - pct) / pct))}s left`;
  }
  $("#prog-label").textContent = txt;
}

/* ---------------- master ---------------- */
$("#master-btn").addEventListener("click", async () => {
  const status = $("#status");
  status.className = "status"; status.textContent = "";
  $("#master-btn").disabled = true;
  $("#result").hidden = true;
  startT = performance.now();
  showProgress("decode", 2);

  try {
    const settings = {
      sound,
      width: parseFloat($("#s-width").value),
      warmth: parseFloat($("#s-warmth").value),
      target: parseFloat($("#s-loud").value),
      vocalLevelDb: parseFloat($("#s-vocal").value),
    };
    let msg;
    if (inputKind === "track") {
      const t = await decodeTo44k(files.track);
      msg = { mode: "track", track: t, settings };
    } else {
      const [v, i] = await Promise.all([decodeTo44k(files.vocal), decodeTo44k(files.instrumental)]);
      msg = { mode: "stems", vocal: v, instr: i, settings };
    }
    showProgress("mix", 6);

    if (worker) worker.terminate();
    worker = new Worker("engine.js");
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") showProgress(m.stage, m.pct);
      else if (m.type === "error") {
        status.className = "status err";
        status.textContent = "Error: " + m.message;
        $("#prog-wrap").hidden = true;
        refresh();
      } else if (m.type === "done") {
        $("#prog-wrap").hidden = true;
        renderResult(m);
        refresh();
      }
    };
    const transfers = [];
    if (msg.track) transfers.push(msg.track.L.buffer, msg.track.R.buffer);
    if (msg.vocal) transfers.push(msg.vocal.L.buffer, msg.vocal.R.buffer, msg.instr.L.buffer, msg.instr.R.buffer);
    worker.postMessage(msg, transfers);
  } catch (e) {
    status.className = "status err";
    status.textContent = "Error: " + (e.message || e);
    $("#prog-wrap").hidden = true;
    refresh();
  }
});

/* ---------------- result ---------------- */
let urls = { pre: null, master: null };
function renderResult(m) {
  if (urls.pre) URL.revokeObjectURL(urls.pre);
  if (urls.master) URL.revokeObjectURL(urls.master);
  urls.pre = URL.createObjectURL(new Blob([m.preWav], { type: "audio/wav" }));
  urls.master = URL.createObjectURL(new Blob([m.wav], { type: "audio/wav" }));
  $("#result").hidden = false;
  const player = $("#player");
  const setAB = (w) => {
    const playing = !player.paused, tpos = player.currentTime || 0;
    player.src = urls[w]; player.load();
    try { player.currentTime = tpos; } catch (_) {}
    if (playing) player.play();
  };
  $$("#ab-toggle .seg-btn").forEach((b) => (b.onclick = () => {
    $$("#ab-toggle .seg-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active"); setAB(b.dataset.ab);
  }));
  setAB("master");
  const r = m.report;
  const metric = (k, v, u) => `<div class="metric"><div class="k">${k}</div><div class="vv">${v ?? "—"}${u || ""}</div></div>`;
  const chip = (ok, label) => `<span class="chip ${ok ? "ok" : "no"}">${ok ? "✓" : "✗"} ${label}</span>`;
  $("#report").innerHTML =
    metric("Integrated", r.integrated_lufs, " LUFS") +
    metric("True peak", r.true_peak_dbtp, " dBTP") +
    metric("Correlation", r.stereo_correlation, "") +
    chip(r.checks.loudness_on_target, "loudness on target") +
    chip(r.checks.true_peak_safe, "≤ −1 dBTP") +
    chip(r.checks.correlation_non_negative, "phase-safe");
  $("#download").href = urls.master;
  $("#result").scrollIntoView({ behavior: "smooth" });
}
