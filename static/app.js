"use strict";

// ----------------------------------------------------------------------------
// State
// ----------------------------------------------------------------------------
const files = { vocal: null, instrumental: null, reference: null };
let mode = "genre";
let lastJob = null;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// ----------------------------------------------------------------------------
// Tabs
// ----------------------------------------------------------------------------
$$(".tab").forEach((t) => t.addEventListener("click", () => {
  $$(".tab").forEach((x) => x.classList.remove("active"));
  $$(".panel").forEach((x) => x.classList.remove("active"));
  t.classList.add("active");
  $("#" + t.dataset.tab).classList.add("active");
}));

// ----------------------------------------------------------------------------
// Drop zones
// ----------------------------------------------------------------------------
function wireDrop(el) {
  const kind = el.dataset.kind;
  const input = el.querySelector("input[type=file]");
  const show = (f) => {
    files[kind] = f;
    el.querySelector(".drop-file").textContent = f ? f.name : "";
    el.classList.toggle("loaded", !!f);
    refreshMasterBtn();
  };
  el.addEventListener("click", () => input.click());
  input.addEventListener("change", () => show(input.files[0] || null));
  ["dragenter", "dragover"].forEach((e) => el.addEventListener(e, (ev) => {
    ev.preventDefault(); el.classList.add("over");
  }));
  ["dragleave", "drop"].forEach((e) => el.addEventListener(e, (ev) => {
    ev.preventDefault(); el.classList.remove("over");
  }));
  el.addEventListener("drop", (ev) => {
    const f = ev.dataTransfer.files[0];
    if (f) show(f);
  });
}
$$(".drop").forEach(wireDrop);

// ----------------------------------------------------------------------------
// Mode selector
// ----------------------------------------------------------------------------
$$("#mode .seg-btn").forEach((b) => b.addEventListener("click", () => {
  $$("#mode .seg-btn").forEach((x) => x.classList.remove("active"));
  b.classList.add("active");
  mode = b.dataset.mode;
  $("#drop-reference").hidden = mode !== "my_reference";
  refreshMasterBtn();
}));

// ----------------------------------------------------------------------------
// Sliders (live labels)
// ----------------------------------------------------------------------------
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
bindSlider("#s-offset", "#v-offset", (v) => v.toFixed(0) + " ms");

function currentSettings() {
  return {
    mode,
    vocal_level_db: parseFloat($("#s-vocal").value),
    width: parseFloat($("#s-width").value),
    warmth: parseFloat($("#s-warmth").value),
    loudness_target: parseFloat($("#s-loud").value),
    offset_ms: parseFloat($("#s-offset").value),
    glue_reverb: $("#t-verb").checked,
    deesser: $("#t-deess").checked,
    ducking: $("#t-duck").checked,
    mixbus_glue: $("#t-glue").checked,
  };
}

function refreshMasterBtn() {
  const ready = files.vocal && files.instrumental &&
    (mode !== "my_reference" || files.reference);
  $("#master-btn").disabled = !ready;
}

// ----------------------------------------------------------------------------
// Master — async job with progress polling
// ----------------------------------------------------------------------------
const STAGE_LABELS = {
  ingest: "Decoding files",
  vocal: "Vocal conditioning",
  balance: "Balancing levels",
  mixbus: "Summing mix bus",
  master: "Mastering",
  loudness: "Loudness & limiting",
  report: "Generating report",
  done: "Done",
};

let _pollTimer = null;

function stopPolling() {
  if (_pollTimer !== null) { clearInterval(_pollTimer); _pollTimer = null; }
}

function showProgress(pct, stage, elapsed) {
  $("#prog-wrap").hidden = false;
  $("#prog-bar").style.width = pct + "%";
  const label = STAGE_LABELS[stage] || stage;
  $("#prog-label").textContent = `${label} · ${pct}% · ${elapsed}s elapsed`;
}

function hideProgress() {
  $("#prog-wrap").hidden = true;
  $("#prog-bar").style.width = "0";
}

$("#master-btn").addEventListener("click", async () => {
  const status = $("#status");
  status.className = "status";
  status.textContent = "";
  $("#master-btn").disabled = true;
  hideProgress();
  stopPolling();

  const fd = new FormData();
  fd.append("vocal", files.vocal);
  fd.append("instrumental", files.instrumental);
  if (mode === "my_reference" && files.reference) fd.append("reference", files.reference);
  fd.append("settings", JSON.stringify(currentSettings()));

  let jobId;
  try {
    const r = await fetch("/api/master", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const data = await r.json();
    jobId = data.job_id;
  } catch (e) {
    status.className = "status err";
    status.textContent = "Error: " + e.message;
    refreshMasterBtn();
    return;
  }

  showProgress(0, "ingest", 0);

  _pollTimer = setInterval(async () => {
    let st;
    try {
      const r = await fetch("/api/status/" + jobId);
      if (!r.ok) { stopPolling(); status.className="status err"; status.textContent="Status check failed."; refreshMasterBtn(); return; }
      st = await r.json();
    } catch (_) { return; }

    showProgress(st.pct, st.stage, st.elapsed_s);

    if (st.error) {
      stopPolling();
      hideProgress();
      status.className = "status err";
      status.textContent = "Error: " + st.error;
      refreshMasterBtn();
    } else if (st.done) {
      stopPolling();
      hideProgress();
      lastJob = st.result;
      renderResult(st.result);
      status.textContent = "";
      refreshMasterBtn();
    }
  }, 500);
});

function renderResult(data) {
  $("#result").hidden = false;
  // notices
  const nz = $("#notices");
  nz.innerHTML = "";
  (data.notices || []).forEach((n) => {
    const d = document.createElement("div"); d.textContent = "⚠ " + n; nz.appendChild(d);
  });
  // A/B player
  const player = $("#player");
  const urls = { pre: data.ab_premaster_url, master: data.ab_master_url };
  const setAB = (which) => {
    const wasPlaying = !player.paused;
    const tpos = player.currentTime || 0;
    player.src = urls[which];
    player.load();
    player.currentTime = tpos;
    if (wasPlaying) player.play();
  };
  $$("#ab-toggle .seg-btn").forEach((b) => b.addEventListener("click", () => {
    $$("#ab-toggle .seg-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    setAB(b.dataset.ab);
  }));
  setAB("master");
  // report
  $("#report").innerHTML = reportHTML(data.report);
  // download
  const dl = $("#download");
  dl.href = data.download_url;
  $("#result").scrollIntoView({ behavior: "smooth" });
}

function reportHTML(rep) {
  const m = (k, v, unit) =>
    `<div class="metric"><div class="k">${k}</div><div class="vv">${v ?? "—"}${unit || ""}</div></div>`;
  const labels = {
    loudness_on_target: "−9 LUFS ±0.5",
    true_peak_safe: "≤ −1 dBTP",
    no_clipping: "no clipping",
    correlation_non_negative: "correlation ≥ 0",
    lows_mostly_mono: "lows mono",
  };
  const chips = Object.entries(rep.checks || {}).map(([k, ok]) =>
    `<span class="chip ${ok ? "ok" : "no"}">${ok ? "✓" : "✗"} ${labels[k] || k}</span>`).join("");
  return [
    m("Integrated", rep.integrated_lufs, " LUFS"),
    m("True peak", rep.true_peak_dbtp, " dBTP"),
    m("Loudness range", rep.loudness_range_lu, " LU"),
    m("Correlation", rep.stereo_correlation, ""),
    m("Sample rate", rep.sample_rate, " Hz"),
    `<div class="checks">${chips}</div>`,
  ].join("");
}

// ----------------------------------------------------------------------------
// Batch / Album
// ----------------------------------------------------------------------------
let pairId = 0;
const batchPairs = {}; // id -> {vocal, instrumental, name}

$("#add-pair").addEventListener("click", addPair);
addPair(); // start with one row

function addPair() {
  const id = pairId++;
  batchPairs[id] = { vocal: null, instrumental: null, name: "" };
  const row = document.createElement("div");
  row.className = "pair";
  row.innerHTML = `
    <input type="text" placeholder="Track name" data-f="name" />
    <div class="filebtn" data-f="vocal">Vocal…</div>
    <div class="filebtn" data-f="instrumental">Instrumental…</div>
    <button class="rm" title="remove">×</button>`;
  row.querySelector('[data-f="name"]').addEventListener("input", (e) => {
    batchPairs[id].name = e.target.value;
  });
  ["vocal", "instrumental"].forEach((kind) => {
    const btn = row.querySelector(`[data-f="${kind}"]`);
    btn.addEventListener("click", () => {
      const inp = document.createElement("input");
      inp.type = "file";
      inp.accept = ".wav,.aif,.aiff,.flac,.mp3,.ogg";
      inp.onchange = () => {
        const f = inp.files[0];
        if (f) { batchPairs[id][kind] = f; btn.textContent = f.name; btn.classList.add("set"); }
        refreshBatchBtn();
      };
      inp.click();
    });
  });
  row.querySelector(".rm").addEventListener("click", () => {
    delete batchPairs[id]; row.remove(); refreshBatchBtn();
  });
  $("#pairs").appendChild(row);
  refreshBatchBtn();
}

function refreshBatchBtn() {
  const vals = Object.values(batchPairs);
  const ok = vals.length > 0 && vals.every((p) => p.vocal && p.instrumental);
  $("#batch-btn").disabled = !ok;
}

$("#batch-btn").addEventListener("click", async () => {
  const status = $("#batch-status");
  status.className = "status";
  status.textContent = "Processing album…";
  $("#batch-btn").disabled = true;

  const fd = new FormData();
  const names = [];
  Object.values(batchPairs).forEach((p, i) => {
    fd.append("files", p.vocal);
    fd.append("files", p.instrumental);
    names.push(p.name || `track${i + 1}`);
  });
  fd.append("names", JSON.stringify(names));
  fd.append("settings", JSON.stringify(currentSettings()));

  try {
    const r = await fetch("/api/batch", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    renderBatch(await r.json());
    status.textContent = "Album done.";
  } catch (e) {
    status.className = "status err";
    status.textContent = "Error: " + e.message;
  } finally {
    refreshBatchBtn();
  }
});

function renderBatch(data) {
  const a = data.album_report;
  const el = $("#batch-result");
  let html = `<div class="album-summary">
      <b>Album:</b> ${a.track_count} tracks · target ${a.album_target_lufs} LUFS ·
      spread ${a.loudness_spread_lu} LU ·
      <span class="chip ${a.consistent ? "ok" : "no"}">${a.consistent ? "✓ consistent" : "✗ uneven"}</span>
    </div>`;
  data.tracks.forEach((t) => {
    html += `<div class="trow">
      <span>${t.name}</span>
      <span>${t.report.integrated_lufs} LUFS</span>
      <span>${t.report.true_peak_dbtp} dBTP</span>
      <a href="${t.download_url}" download>Download</a>
    </div>`;
  });
  el.innerHTML = html;
}
