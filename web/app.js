"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtRs = (x) => "Rs " + Math.round(x).toLocaleString("en-IN");
const pct = (x, d = 1) => (x >= 0 ? "+" : "") + (x * 100).toFixed(d) + "%";

let map, flowLayer;
let mandis = [];
const markers = {};   // name -> marker
const pos = {};       // name -> [lat, lon] as drawn
let current = null;   // { home, date, quintals } of the last successful advice
let splitSeq = 0;
let splitTimer = null;

async function api(path, params = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([k, v]) => {
    if (v !== null && v !== undefined && v !== "") url.searchParams.set(k, v);
  });
  const res = await fetch(url);
  let body = null;
  try { body = await res.json(); } catch (e) { /* not JSON */ }
  if (!res.ok) {
    const detail = body && typeof body.detail === "string" ? body.detail : "The request was not accepted.";
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return body;
}

function setStatus(msg, isError = false) {
  const el = $("status");
  el.textContent = msg || "";
  el.className = "status" + (isError ? " error" : "");
}

/* ------------------------------------------------------------------ map */
function initMap() {
  map = L.map("map", { scrollWheelZoom: false }).setView([17.5, 76.5], 6);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors", maxZoom: 12,
  }).addTo(map);
  flowLayer = L.layerGroup().addTo(map);
}

function drawMarkers() {
  const seen = {};
  const bounds = [];
  mandis.forEach((m) => {
    if (m.lat == null || m.lon == null) return;
    let lat = m.lat, lon = m.lon;
    const key = lat.toFixed(2) + "," + lon.toFixed(2);
    if (seen[key]) lon += 0.12 * seen[key];   // nudge mandis that share a coordinate so both stay visible
    seen[key] = (seen[key] || 0) + 1;
    pos[m.name] = [lat, lon];
    const mk = L.circleMarker([lat, lon], { radius: 9, weight: 2, color: "#34d399", fillColor: "#0b1220", fillOpacity: 0.95 });
    mk.bindTooltip(m.name);
    mk.on("click", () => { $("home").value = m.name; highlightHome(); });
    mk.addTo(map);
    markers[m.name] = mk;
    bounds.push([lat, lon]);
  });
  if (bounds.length) map.fitBounds(bounds, { padding: [40, 40] });
  highlightHome();
}

function highlightHome() {
  const home = $("home").value;
  Object.entries(markers).forEach(([name, mk]) => {
    const isHome = name === home;
    mk.setStyle({ radius: isHome ? 13 : 9, fillColor: isHome ? "#34d399" : "#0b1220" });
  });
}

function drawFlows(split) {
  flowLayer.clearLayers();
  const from = pos[split.home];
  if (!from) return;
  const total = split.tonnes * 10;
  Object.entries(split.best_plan.allocation).forEach(([name, qtl]) => {
    if (name === split.home || !pos[name]) return;
    const line = L.polyline([from, pos[name]], {
      color: "#fbbf24", weight: 2 + (qtl / total) * 10, opacity: 0.9, className: "flow",
    });
    line.bindTooltip(qtl + " q", { permanent: true, direction: "center", className: "flow-label" });
    line.addTo(flowLayer);
  });
}

/* ------------------------------------------------------------------ advice */
function renderAdvice(r) {
  const isWait = r.action === "WAIT";
  const f = r.forecast;
  const bar = ["down", "flat", "up"].map((k) =>
    `<div class="p ${k}" style="width:${(f[k] * 100).toFixed(1)}%">${k[0].toUpperCase() + k.slice(1)} ${Math.round(f[k] * 100)}%</div>`
  ).join("");
  const options = r.options.map((o) =>
    `<tr><td>${esc(o.action.replace("_", " "))}</td><td>${fmtRs(o.net_rs)}</td><td>${fmtRs(o.cost_rs)}</td><td>${esc(o.note)}</td></tr>`
  ).join("");
  const a = r.assumptions;
  const pct = (k) => Math.round(f[k] * 100);
  const lean = f.down >= f.up ? `price is more likely to fall (${pct("down")}%) than rise (${pct("up")}%)`
                              : `price is more likely to rise (${pct("up")}%) than fall (${pct("down")}%)`;
  const plain = isWait
    ? `Over the next week the ${lean}. Waiting is expected to earn more, even after storage losses.`
    : `Over the next week the ${lean}. Waiting is not expected to beat selling now, so selling today is the safer choice. Confidence: ${r.confidence}.`;
  $("advice").innerHTML = `
    <h2>Advice for ${esc(r.mandi)} on ${esc(r.date)}</h2>
    <p class="sub">${r.quintals} quintals (${r.quintals / 10} tonnes) at ${fmtRs(r.price_today_rs_qtl)} per quintal</p>
    <div class="verdict ${isWait ? "wait" : "sell"}">
      <span class="chip">${isWait ? "WAIT ABOUT A WEEK" : "SELL TODAY"}</span>
    </div>
    <p class="plain">${esc(plain)}</p>
    <details class="more"><summary>Why? Show the numbers</summary>
      <p class="sub">Forecast confidence: ${esc(r.confidence)}. Model's 7-day outlook ("flat" = within 3% of today's price):</p>
      <div class="probbar">${bar}</div>
      <ul class="reasons">${r.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
      <table><tr><th>Option</th><th>Expected money in hand</th><th>Costs</th><th></th></tr>${options}</table>
    </details>
    <div class="row buttons"><button id="reveal" class="primary">Reveal what happened next</button></div>
    <div id="replay" class="replay hidden"></div>
    <details><summary>Assumptions used (all adjustable in the code)</summary>
      <pre>${esc(JSON.stringify(a, null, 2))}</pre>
    </details>`;
  $("advice").classList.remove("hidden");
  $("reveal").addEventListener("click", () => {
    $("replay").innerHTML = replayHtml(r);
    $("replay").classList.remove("hidden");
    $("reveal").classList.add("hidden");
  });
}

function replayHtml(r) {
  const p = r.replay;
  if (p.price_7_days_later == null) return "<p>No price was recorded about 7 days later at this mandi.</p>";
  const other = r.action === "WAIT" ? p.net_if_sold_now : p.net_if_waited;
  const diff = p.net_following_advice - other;
  const verdict = diff >= 0
    ? `<p class="good"><b>The advice earned ${fmtRs(diff)} more than the alternative.</b></p>`
    : `<p class="bad"><b>The advice earned ${fmtRs(-diff)} less than the alternative this time.</b> No forecast is right every time; that is why we show the odds.</p>`;
  return `
    <p>7 days later the price was <b>${fmtRs(p.price_7_days_later)}</b> per quintal (${pct(p.change_pct)}).</p>
    <p>Selling that day: ${fmtRs(p.net_if_sold_now)}<br>Waiting 7 days (after holding costs): ${fmtRs(p.net_if_waited)}<br>
    Following the advice (${esc(r.action.replace("_", " "))}): <b>${fmtRs(p.net_following_advice)}</b></p>
    ${verdict}`;
}

/* ------------------------------------------------------------------ compare */
async function loadCompare(home, date) {
  const el = $("compare");
  try {
    const c = await api("/api/v1/compare", { home, date });
    const rows = c.rows.map((r) =>
      `<tr><td>${esc(r.mandi)}</td><td>${fmtRs(r.price_3day_avg_rs_qtl)}</td><td>${r.usual_price_vs_home.toFixed(2)}x</td>
       <td>${pct(r.vs_usual_pct)}</td><td>${r.road_km == null ? "-" : Math.round(r.road_km) + " km"}</td>
       <td>${r.freight_rs_per_qtl == null ? "-" : fmtRs(r.freight_rs_per_qtl)}</td></tr>`).join("");
    el.innerHTML = `
      <h2>Price comparison with other mandis</h2>
      <p class="sub">Information only. Reported prices; grades can differ between mandis.</p>
      <table><tr><th>Mandi</th><th>Price (3-day avg)</th><th>Usually vs home</th><th>Today vs usual</th><th>Distance</th><th>Freight / quintal</th></tr>${rows}</table>
      <ul class="findings">${c.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>`;
    el.classList.remove("hidden");
  } catch (e) {
    el.classList.add("hidden");
  }
}

/* ------------------------------------------------------------------ split */
function fmtAlloc(a) {
  return Object.entries(a).map(([k, v]) => `${esc(k)} ${v} q`).join(" + ");
}

function renderSplit(s) {
  const sens = s.sensitivity.map((x) =>
    `<tr><td>${x.flexibility.toFixed(2)}</td><td>${fmtRs(x.gain_rs)}</td><td>${pct(x.gain_pct)}</td><td>${fmtAlloc(x.best_plan)}</td></tr>`).join("");
  $("splitBody").innerHTML = `
    <div class="splitgrid">
      <div class="card">Best single mandi<b class="big">${fmtAlloc(s.best_single.allocation)}</b>
        <span class="sub">Money in hand ${fmtRs(s.best_single.net_rs)} (transport ${fmtRs(s.best_single.transport_rs)})</span></div>
      <div class="card">Best plan (shown on the map)<b class="big">${fmtAlloc(s.best_plan.allocation)}</b>
        <span class="sub">Money in hand ${fmtRs(s.best_plan.net_rs)} (transport ${fmtRs(s.best_plan.transport_rs)})</span></div>
    </div>
    <p>Gain from splitting: <span class="gain">${fmtRs(s.gain_rs)} (${pct(s.gain_pct)})</span></p>
    <p class="sub">Price effect used: ${s.flexibility_used.toFixed(2)} (${esc(s.flexibility_note)}).</p>
    <h3>How much does the answer depend on that assumption?</h3>
    <table><tr><th>Price effect</th><th>Gain</th><th>Gain %</th><th>Best plan</th></tr>${sens}</table>
    <ul class="findings">${s.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>`;
  drawFlows(s);
}

async function loadSplit() {
  if (!current) return;
  const seq = ++splitSeq;
  $("splitBody").innerHTML = `<p class="sub">Working out the best split...</p>`;
  try {
    const s = await api("/api/v1/split", {
      home: current.home, date: current.date,
      tonnes: $("tonnes").value, flexibility: $("flex").value,
    });
    if (seq === splitSeq) renderSplit(s);
  } catch (e) {
    if (seq === splitSeq) $("splitBody").innerHTML = `<p class="error">${esc(e.message)}</p>`;
  }
}

function scheduleSplit() {
  clearTimeout(splitTimer);
  splitTimer = setTimeout(loadSplit, 450);
}

function updateSliderLabels() {
  $("tonnesVal").textContent = $("tonnes").value + " tonnes";
  const f = Number($("flex").value);
  $("flexVal").textContent = f.toFixed(2) + (f === 0 ? " (no price effect)" : " (assumed)");
}

/* ------------------------------------------------------------------ evidence */
async function loadEvidence() {
  try {
    const b = await api("/api/v1/backtest");
    const rows = [["All test years", b.all_folds_pooled]].concat(
      Object.entries(b.folds).map(([y, v]) => ["Tested on " + y, v]));
    const body = rows.map(([label, v]) => {
      const best = Math.max(...Object.values(v.true_class_share));
      const c = v.confident_predictions;
      const conf = c.accuracy == null ? "-" : `${(c.accuracy * 100).toFixed(0)}% <small>(${(c.share_of_rows * 100).toFixed(0)}% of days)</small>`;
      return `<tr><td>${esc(label)}</td><td>${v.rows}</td><td><b>${(v.model.accuracy * 100).toFixed(1)}%</b></td>
        <td>${(best * 100).toFixed(1)}%</td><td>${(v.baseline_trend_continues.accuracy * 100).toFixed(1)}%</td><td>${conf}</td></tr>`;
    }).join("");
    $("evidenceBody").innerHTML = `
      <p class="sub">Each year was predicted by a model trained only on earlier years. Three-way guess: down, flat or up in 7 days.</p>
      <table><tr><th>Test</th><th>Days</th><th>Model</th><th>Best constant guess</th><th>"Trend continues"</th><th>Model when 60%+ sure</th></tr>${body}</table>`;
  } catch (e) {
    $("evidenceBody").innerHTML = `<p class="sub">Backtest results are not available.</p>`;
  }
}

/* ------------------------------------------------------------------ actions */
async function getAdvice(home, date, quintals) {
  setStatus("Asking the model (the first request can take a few seconds while the database wakes up)...");
  try {
    const rec = await api("/api/v1/recommend", { home, date, quintals });
    setStatus("");
    current = { home, date, quintals };
    renderAdvice(rec);
    $("split").classList.remove("hidden");
    loadCompare(home, date);
    loadSplit();
    return true;
  } catch (e) {
    setStatus(e.message, true);
    return false;
  }
}

function readControls() {
  return { home: $("home").value, date: $("date").value, quintals: Number($("quintals").value) };
}

async function randomDay() {
  const start = new Date("2022-01-03").getTime();
  const end = new Date("2024-12-27").getTime();
  for (let i = 0; i < 12; i++) {
    const d = new Date(start + Math.random() * (end - start)).toISOString().slice(0, 10);
    $("date").value = d;
    const { home, quintals } = readControls();
    setStatus("Picking a random day...");
    if (await getAdvice(home, d, quintals)) return;
  }
  setStatus("Could not find a day with data. Try again.", true);
}

async function init() {
  initMap();
  try {
    mandis = await api("/api/v1/mandis");
  } catch (e) {
    setStatus("Could not load the mandis: " + e.message, true);
    return;
  }
  $("home").innerHTML = mandis.map((m) => `<option value="${esc(m.name)}">${esc(m.name)} (${esc(m.state)})</option>`).join("");
  $("home").value = "Gudimalkapur";
  drawMarkers();
  loadEvidence();
  updateSliderLabels();

  $("home").addEventListener("change", highlightHome);
  $("go").addEventListener("click", () => { const c = readControls(); getAdvice(c.home, c.date, c.quintals); });
  $("random").addEventListener("click", randomDay);
  document.querySelectorAll(".chipbtn").forEach((b) => b.addEventListener("click", () => {
    $("home").value = b.dataset.home; $("date").value = b.dataset.date; $("quintals").value = b.dataset.q;
    highlightHome();
    getAdvice(b.dataset.home, b.dataset.date, Number(b.dataset.q));
  }));
  $("tonnes").addEventListener("input", () => { updateSliderLabels(); scheduleSplit(); });
  $("flex").addEventListener("input", () => { updateSliderLabels(); scheduleSplit(); });
  $("flexZero").addEventListener("click", () => { $("flex").value = 0; updateSliderLabels(); scheduleSplit(); });
}

init();
