let allRows = [];
let sortKey = "prob_1";
let sortDir = "desc";

function fmt(v, d = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toFixed(d);
}
function fmtPct(v, d = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return (Number(v) * 100).toFixed(d) + "%";
}
function fmtProb(v) {
  return fmtPct(v, 1);
}
function riskBadge(risk) {
  const x = String(risk || "OK");
  let cls = "bg-slate-100 text-slate-800";
  if (x === "OK") cls = "bg-emerald-100 text-emerald-800";
  else if (x === "CAUTION") cls = "bg-amber-100 text-amber-800";
  else if (x === "HIGH") cls = "bg-rose-100 text-rose-800";
  else if (x === "BLOCKED") cls = "bg-rose-200 text-rose-900";
  return `<span class="px-2 py-1 rounded-full text-xs font-medium ${cls}">${x}</span>`;
}
function uncertaintyBadge(u) {
  const x = String(u || "LOW");
  let cls = "bg-emerald-100 text-emerald-800";
  if (x === "MED") cls = "bg-amber-100 text-amber-800";
  if (x === "HIGH") cls = "bg-rose-100 text-rose-800";
  return `<span class="px-2 py-1 rounded-full text-xs font-medium ${cls}">${x}</span>`;
}
function verdictBadge(v) {
  const x = String(v || "—");
  let cls = "bg-slate-100 text-slate-800";
  if (x === "CLEAN_TOUCH") cls = "bg-emerald-100 text-emerald-800";
  else if (x === "BOUNCY_TOUCH") cls = "bg-amber-100 text-amber-800";
  else if (x === "UGLY_TOUCH") cls = "bg-rose-100 text-rose-800";
  else if (x === "NO_TOUCH") cls = "bg-slate-100 text-slate-700";
  return `<span class="px-2 py-1 rounded-full text-xs font-medium ${cls}">${x}</span>`;
}
function bucketBadge(v) {
  const x = String(v || "—");
  let cls = "bg-slate-100 text-slate-800";
  if (x === "WORTH_REVIEW") cls = "bg-emerald-100 text-emerald-800";
  else if (x === "WATCHLIST_ONLY") cls = "bg-amber-100 text-amber-800";
  else if (x === "REJECT") cls = "bg-rose-100 text-rose-800";
  return `<span class="px-2 py-1 rounded-full text-xs font-medium ${cls}">${x}</span>`;
}
function fmtSkipReasons(obj) {
  if (!obj) return "—";
  const pairs = Object.entries(obj).filter(([, v]) => v > 0);
  if (!pairs.length) return "—";
  return pairs.map(([k, v]) => `${k}: ${v}`).join(" · ");
}
function sortRows(rows) {
  rows.sort((a, b) => {
    const va = a[sortKey];
    const vb = b[sortKey];
    if (va === vb) return 0;
    if (sortDir === "asc") return (va > vb) ? 1 : -1;
    return (va < vb) ? 1 : -1;
  });
}
function applyFiltersLocal() {
  const minProb = parseFloat(document.getElementById("minProb").value || "0");
  const sector = document.getElementById("sector").value || "";
  const pMin = document.getElementById("priceMin").value ? parseFloat(document.getElementById("priceMin").value) : null;
  const pMax = document.getElementById("priceMax").value ? parseFloat(document.getElementById("priceMax").value) : null;
  let rows = allRows.slice();
  rows = rows.filter(r => (r.prob_1 ?? 0) >= minProb);
  if (sector) rows = rows.filter(r => (r.sector || "") === sector);
  if (pMin !== null) rows = rows.filter(r => (r.price ?? 0) >= pMin);
  if (pMax !== null) rows = rows.filter(r => (r.price ?? 0) <= pMax);
  sortRows(rows);
  renderRows(rows);
}
function renderRows(rows) {
  const tbody = document.getElementById("rows");
  tbody.innerHTML = "";
  const empty = document.getElementById("emptyState");
  if (!rows || rows.length === 0) {
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="py-2 pr-4 font-semibold">${r.symbol}</td>
      <td class="py-2 pr-4">${fmt(r.price, 2)}</td>
      <td class="py-2 pr-4">${fmt(r.vwap, 2)}</td>
      <td class="py-2 pr-4 font-semibold">${fmtProb(r.prob_1)}</td>
      <td class="py-2 pr-4">${riskBadge(r.risk)}</td>
      <td class="py-2 pr-4 text-slate-600">${r.risk_reasons || "—"}</td>
      <td class="py-2 pr-4">${fmt(r.downside_risk ?? null, 2)}</td>
      <td class="py-2 pr-4">${uncertaintyBadge(r.uncertainty || "LOW")}</td>
      <td class="py-2 pr-4">${r.sector || ""}</td>
      <td class="py-2 pr-4 text-slate-600">${r.reasons || ""}</td>`;
    tbody.appendChild(tr);
  }
}
function renderTracked(rows) {
  const tbody = document.getElementById("diagTrackedRows");
  tbody.innerHTML = "";
  for (const r of (rows || [])) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="py-2 pr-4 font-semibold">${r.symbol}</td>
      <td class="py-2 pr-4">${fmtProb(r.max_prob_1)}</td>
      <td class="py-2 pr-4">${fmtProb(r.max_prob_1_raw)}</td>
      <td class="py-2 pr-4">${r.cap_bound_seen ? "Yes" : "No"}</td>
      <td class="py-2 pr-4">${r.seen_count || 0}</td>
      <td class="py-2 pr-4 text-slate-600">${(r.risk_set || []).join(", ") || "—"}</td>
      <td class="py-2 pr-4">${fmt(r.max_downside_risk, 2)}</td>
      <td class="py-2 pr-4 text-slate-600">${(r.uncertainty_set || []).join(", ") || "—"}</td>`;
    tbody.appendChild(tr);
  }
}
function renderEvaluation(rows) {
  const tbody = document.getElementById("diagEvaluationRows");
  tbody.innerHTML = "";
  const empty = document.getElementById("diagEmpty");
  if (!rows || rows.length === 0) {
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="py-2 pr-4 font-semibold">${r.symbol}</td>
      <td class="py-2 pr-4">${fmtProb(r.best_prob_1)}</td>
      <td class="py-2 pr-4">${r.touch_1pct ? "Yes" : "No"}</td>
      <td class="py-2 pr-4">${verdictBadge(r.path_verdict)}</td>
      <td class="py-2 pr-4">${fmtPct(r.mae_before_touch_pct, 2)}</td>
      <td class="py-2 pr-4">${r.held_above_scan_10m === null || r.held_above_scan_10m === undefined ? "—" : (r.held_above_scan_10m ? "Yes" : "No")}</td>
      <td class="py-2 pr-4">${fmtPct(r.close_vs_scan_pct, 2)}</td>
      <td class="py-2 pr-4">${bucketBadge(r.review_bucket)}</td>`;
    tbody.appendChild(tr);
  }
}
async function refreshScores() {
  const resp = await fetch("/api/scores");
  const data = await resp.json();
  allRows = data.rows || [];
  document.getElementById("lastRun").textContent = data.last_run_utc || "—";
  const sectors = [...new Set(allRows.map(r => r.sector).filter(Boolean))].sort();
  const sel = document.getElementById("sector");
  const current = sel.value;
  sel.innerHTML = `<option value="">All</option>` + sectors.map(s => `<option value="${s}">${s}</option>`).join("");
  sel.value = current;
  applyFiltersLocal();
}
async function refreshTraining() {
  const resp = await fetch("/api/training/status");
  const t = await resp.json();
  document.getElementById("trainRunning").textContent = t.running ? "Running" : "Idle";
  document.getElementById("trainStarted").textContent = t.started_at_utc || "—";
  document.getElementById("trainFinished").textContent = t.finished_at_utc || "—";
  document.getElementById("trainError").textContent = t.last_error || "";
}
async function refreshDiagnostics() {
  const resp = await fetch("/api/diagnostics");
  const d = await resp.json();
  const summary = d.summary || {};
  document.getElementById("diagTradeDate").textContent = d.trade_date || "—";
  document.getElementById("diagSnapshots").textContent = summary.snapshots_count || 0;
  document.getElementById("diagTracked").textContent = summary.tracked_count || 0;
  document.getElementById("diagCapBound").textContent = summary.cap_bound_count || 0;
  document.getElementById("diagEvaluated").textContent = summary.evaluated ? "Yes" : "No";
  document.getElementById("diagClean").textContent = summary.clean_touch_count || 0;
  document.getElementById("diagUgly").textContent = summary.ugly_touch_count || 0;
  document.getElementById("diagNoTouch").textContent = summary.no_touch_count || 0;
  document.getElementById("diagWorth").textContent = summary.worthy_count || 0;
  document.getElementById("diagLatestSnapshot").textContent = summary.latest_snapshot_utc || "—";
  renderTracked(d.tracked || []);
  renderEvaluation((d.evaluation && d.evaluation.rows) ? d.evaluation.rows : []);
}
async function refreshStatus() {
  const resp = await fetch("/api/status");
  const s = await resp.json();
  document.getElementById("clock").textContent = (new Date()).toLocaleString();
  const marketOpen = !!s.market_open;
  const badge = document.getElementById("marketBadge");
  badge.textContent = marketOpen ? "Market open" : "Market closed";
  badge.className = "px-2 py-1 rounded-full text-xs " + (marketOpen ? "bg-emerald-100 text-emerald-800" : "bg-slate-100 text-slate-700");
  const alp = s.alpaca || {};
  document.getElementById("alpacaStatus").textContent = alp.ok ? "OK" : (alp.message || "Not connected");
  document.getElementById("feedStatus").textContent = (alp.feed || "sip").toUpperCase();
  document.getElementById("lastBar").textContent = alp.last_bar_timestamp || "—";
  document.getElementById("rateLimit").textContent = alp.rate_limit_warn || "";
  const ttc = s.time_to_close_seconds || 0;
  document.getElementById("ttc").textContent = marketOpen ? (Math.floor(ttc / 60) + " min") : "—";
  const m = (s.model && s.model.pt1) ? s.model.pt1 : {};
  let m1 = m.trained ? "pt1 ✅" : "pt1 ⏳";
  if (m.trained && m.tail_ready_75 === false) m1 += " · tail cap";
  document.getElementById("modelStatus").textContent = m1;
  const uni = s.constituents || {};
  document.getElementById("universeStatus").textContent = (uni.count || 0) + " (" + (uni.source || "fallback") + ")";
  const cov = s.coverage || {};
  document.getElementById("coverageStatus").textContent = `Stage1 ${cov.stage1_candidate_count || 0} → Stage2 ${cov.stage2_scored_count || 0}`;
  document.getElementById("skipReasons").textContent = fmtSkipReasons(cov.top_skip_reasons);
  document.getElementById("profileNote").textContent = m.selection_warning || cov.profile_note || "";
  document.getElementById("lastError").textContent = s.last_error || "";
  const tc = cov.threshold_counts || {};
  const atc = cov.acceptable_threshold_counts || {};
  document.getElementById("tailCounts").textContent = `All: 0.60 ${tc.ge_0_60 || 0} · 0.70 ${tc.ge_0_70 || 0} · 0.75 ${tc.ge_0_75 || 0} · 0.80 ${tc.ge_0_80 || 0}`;
  document.getElementById("acceptableTailCounts").textContent = `Acceptable: 0.60 ${atc.ge_0_60 || 0} · 0.70 ${atc.ge_0_70 || 0} · 0.75 ${atc.ge_0_75 || 0} · 0.80 ${atc.ge_0_80 || 0}`;
  const gs = cov.guardrail_stats || {};
  document.getElementById("guardrailStats").textContent = `Blocked ${gs.blocked_in_universe || 0} · Event ${gs.event_in_universe || 0} · High downside ${gs.high_downside_in_candidates || 0} · High uncertainty ${gs.high_uncertainty_in_candidates || 0}`;
  await refreshTraining();
}
async function startTraining() {
  const pwd = document.getElementById("adminPassword").value || "";
  const form = new FormData();
  form.append("admin_password", pwd);
  const resp = await fetch("/train", { method: "POST", body: form });
  const js = await resp.json();
  if (!resp.ok) alert(js.error || "Training failed");
  await refreshTraining();
}

window.addEventListener("DOMContentLoaded", async () => {
  document.getElementById("applyFilters").addEventListener("click", applyFiltersLocal);
  document.getElementById("resetFilters").addEventListener("click", () => {
    document.getElementById("minProb").value = "0.00";
    document.getElementById("sector").value = "";
    document.getElementById("priceMin").value = "";
    document.getElementById("priceMax").value = "";
    applyFiltersLocal();
  });
  document.getElementById("startTraining").addEventListener("click", startTraining);
  document.querySelectorAll("th[data-sort]").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.getAttribute("data-sort");
      if (sortKey === key) sortDir = sortDir === "desc" ? "asc" : "desc";
      else { sortKey = key; sortDir = "desc"; }
      applyFiltersLocal();
    });
  });
  await refreshStatus();
  await refreshScores();
  await refreshDiagnostics();
  setInterval(refreshStatus, 10000);
  setInterval(refreshScores, 10000);
  setInterval(refreshDiagnostics, 15000);
});
