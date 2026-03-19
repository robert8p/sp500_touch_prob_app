let allRows = [];
let watchlistRescueRows = [];
let sortKey = "relative_strength_rank";
let sortDir = "asc";
let latestDiagnosticsSummary = {};
let latestModelMeta = {};
let latestAIStrategy = {};

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
function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
function el(id) {
  return document.getElementById(id);
}
function setText(id, value) {
  const node = el(id);
  if (node) node.textContent = value;
}
function setHtml(id, value) {
  const node = el(id);
  if (node) node.innerHTML = value;
}
function setVisible(id, isVisible) {
  const node = el(id);
  if (!node) return;
  node.classList.toggle("hidden", !isVisible);
}
function listHtml(items, emptyLabel = '—') {
  const arr = Array.isArray(items) ? items.filter(Boolean) : [];
  if (!arr.length) return escapeHtml(emptyLabel);
  return `<ul class=\"space-y-1\">${arr.map(x => `<li class=\"flex gap-2\"><span class=\"mt-[7px] inline-block h-1.5 w-1.5 rounded-full bg-slate-400\"></span><span>${escapeHtml(x)}</span></li>`).join('')}</ul>`;
}
function postureBadge(value) {
  const x = String(value || 'stand_aside');
  const tone = x === 'normal_size' ? 'emerald' : x === 'pilot_size' ? 'cyan' : x === 'watchlist_only' ? 'amber' : 'rose';
  return statePill(x.replaceAll('_', ' '), tone);
}
function candidateHandlingHtml(items) {
  const arr = Array.isArray(items) ? items : [];
  if (!arr.length) return '—';
  return arr.map(item => {
    const tone = item.stance === 'candidate' ? 'emerald' : item.stance === 'watchlist' ? 'amber' : 'rose';
    return `<div class=\"rounded-2xl border border-slate-200 bg-slate-50 p-3\"><div class=\"flex items-center justify-between gap-3\"><div class=\"font-semibold text-slate-950\">${escapeHtml(item.symbol || '')}</div>${statePill(String(item.stance || 'watchlist'), tone)}</div><div class=\"mt-2 text-xs leading-5 text-slate-600\">${escapeHtml(item.rationale || '—')}</div><div class=\"mt-2 text-[11px] uppercase tracking-[0.16em] text-slate-500\">Trigger</div><div class=\"text-xs leading-5 text-slate-700\">${escapeHtml(item.entry_trigger || '—')}</div><div class=\"mt-2 text-[11px] uppercase tracking-[0.16em] text-slate-500\">Invalidation</div><div class=\"text-xs leading-5 text-slate-700\">${escapeHtml(item.invalidation || '—')}</div></div>`;
  }).join('');
}
function statePill(text, tone = "slate") {
  const tones = {
    slate: "border-slate-200 bg-slate-100 text-slate-700",
    emerald: "border-emerald-200 bg-emerald-100 text-emerald-800",
    amber: "border-amber-200 bg-amber-100 text-amber-800",
    rose: "border-rose-200 bg-rose-100 text-rose-800",
    cyan: "border-cyan-200 bg-cyan-100 text-cyan-800",
    indigo: "border-indigo-200 bg-indigo-100 text-indigo-800"
  };
  return `<span class="inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-semibold ${tones[tone] || tones.slate}">${escapeHtml(text)}</span>`;
}
function regimeBadge(state) {
  const x = String(state || "NOT_EVALUATED").toUpperCase();
  let tone = "slate";
  if (x === "GREEN") tone = "emerald";
  else if (x === "AMBER") tone = "amber";
  else if (x === "RED") tone = "rose";
  else if (x === "CLOSED") tone = "indigo";
  return statePill(x, tone);
}
function riskBadge(risk) {
  const x = String(risk || "OK");
  let tone = "emerald";
  if (x === "CAUTION") tone = "amber";
  else if (x === "HIGH" || x === "BLOCKED") tone = "rose";
  return statePill(x, tone);
}
function uncertaintyBadge(u) {
  const x = String(u || "LOW");
  let tone = "emerald";
  if (x === "MED") tone = "amber";
  if (x === "HIGH") tone = "rose";
  return statePill(x, tone);
}
function verdictBadge(v) {
  const x = String(v || "—");
  let tone = "slate";
  if (x === "CLEAN_TOUCH") tone = "emerald";
  else if (x === "BOUNCY_TOUCH") tone = "amber";
  else if (x === "UGLY_TOUCH") tone = "rose";
  return statePill(x, tone);
}
function bucketBadge(v) {
  const x = String(v || "—");
  let tone = "slate";
  if (x === "WORTH_REVIEW") tone = "cyan";
  else if (x === "WATCHLIST_ONLY") tone = "amber";
  else if (x === "REJECT") tone = "rose";
  return statePill(x, tone);
}
function probBadge(v) {
  const n = Number(v ?? 0);
  const ct = _adaptiveThreshold * _pathQualityMin;
  let tone = "slate";
  if (n >= ct) tone = "emerald";
  else if (n >= ct * 0.60) tone = "cyan";
  else if (n >= ct * 0.30) tone = "amber";
  return `<span class="inline-flex min-w-[76px] items-center justify-center rounded-full border px-3 py-1 text-xs font-bold ${tone === 'emerald' ? 'border-emerald-200 bg-emerald-100 text-emerald-900' : tone === 'cyan' ? 'border-cyan-200 bg-cyan-100 text-cyan-900' : tone === 'amber' ? 'border-amber-200 bg-amber-100 text-amber-900' : 'border-slate-200 bg-slate-100 text-slate-700'}">${fmtProb(n)}</span>`;
}
let _adaptiveThreshold = 0.10; // updated from model status
let _pathQualityMin = 0.65; // updated from training_selection
function touchBadge(v) {
  const n = Number(v ?? 0);
  const at = _adaptiveThreshold || 0.10;
  let tone = "slate";
  if (n >= at) tone = "emerald";
  else if (n >= at * 0.65) tone = "cyan";
  else if (n >= at * 0.35) tone = "amber";
  return `<span class="inline-flex min-w-[76px] items-center justify-center rounded-full border px-3 py-1 text-xs font-bold ${tone === 'emerald' ? 'border-emerald-200 bg-emerald-100 text-emerald-900' : tone === 'cyan' ? 'border-cyan-200 bg-cyan-100 text-cyan-900' : tone === 'amber' ? 'border-amber-200 bg-amber-100 text-amber-900' : 'border-slate-200 bg-slate-100 text-slate-700'}">${fmtProb(n)}</span>`;
}
function pathBadge(v) {
  const n = Number(v ?? 0);
  let tone = "slate";
  if (n >= 0.80) tone = "emerald";
  else if (n >= 0.65) tone = "cyan";
  else if (n >= 0.50) tone = "amber";
  return `<span class="inline-flex min-w-[76px] items-center justify-center rounded-full border px-3 py-1 text-xs font-bold ${tone === 'emerald' ? 'border-emerald-200 bg-emerald-100 text-emerald-900' : tone === 'cyan' ? 'border-cyan-200 bg-cyan-100 text-cyan-900' : tone === 'amber' ? 'border-amber-200 bg-amber-100 text-amber-900' : 'border-slate-200 bg-slate-100 text-slate-700'}">${fmtProb(n)}</span>`;
}
function signalBadge(s) {
  if (s === "ACTIONABLE") return `<span class="inline-flex items-center rounded-full border border-emerald-300 bg-emerald-100 px-3 py-1 text-xs font-bold text-emerald-900 shadow-sm shadow-emerald-200">ACTIONABLE</span>`;
  if (s === "CANDIDATE") return `<span class="inline-flex items-center rounded-full border border-cyan-300 bg-cyan-100 px-3 py-1 text-xs font-bold text-cyan-900">CANDIDATE</span>`;
  return `<span class="text-xs text-slate-400">—</span>`;
}
function rescueBadge() {
  return `<span class="inline-flex items-center rounded-full border border-amber-300 bg-amber-100 px-3 py-1 text-xs font-bold text-amber-900">WATCHLIST</span>`;
}
function fmtSkipReasons(obj) {
  if (!obj) return "—";
  const pairs = Object.entries(obj).filter(([, v]) => v > 0);
  if (!pairs.length) return "—";
  return pairs.map(([k, v]) => `${k}: ${v}`).join(" · ");
}

function signalPriorityValue(row) {
  const signal = String(row?.signal || '');
  if (signal === 'ACTIONABLE') return 3;
  if (signal === 'CANDIDATE') return 2;
  if (row?.watchlist_rescue) return 1;
  return 0;
}
function riskPenaltyValue(risk) {
  const x = String(risk || 'OK').toUpperCase();
  if (x === 'BLOCKED') return 3;
  if (x === 'HIGH') return 2;
  if (x === 'CAUTION') return 1;
  return 0;
}
function uncertaintyPenaltyValue(level) {
  const x = String(level || 'LOW').toUpperCase();
  if (x === 'HIGH') return 2;
  if (x === 'MED' || x === 'MEDIUM') return 1;
  return 0;
}
function computePromiseScore(row) {
  const provided = Number(row?.relative_strength_score);
  if (Number.isFinite(provided)) return provided;
  const prob1 = Number(row?.prob_1 || 0);
  const touch = Number(row?.prob_touch || 0);
  const path = Number(row?.prob_path || 0);
  const stage1 = Number(row?.stage1_score || 0);
  const downside = Number(row?.downside_risk || 0);
  const signalBoost = signalPriorityValue(row) * 1000;
  const acceptableBoost = row?.acceptable ? 120 : 0;
  const watchlistBoost = row?.watchlist_rescue ? 25 : 0;
  const familyBonus = Number(row?.setup_family_bonus || 0) * 100;
  const penalties =
    riskPenaltyValue(row?.risk) * 45 +
    uncertaintyPenaltyValue(row?.uncertainty) * 25 +
    (row?.event_risk ? 18 : 0) +
    (row?.high_downside ? 30 : (row?.medium_downside ? 12 : 0)) +
    (row?.high_uncertainty ? 22 : 0) +
    (row?.suppression_reason ? 35 : 0) +
    (Number.isFinite(downside) ? downside * 18 : 0);
  return signalBoost + acceptableBoost + watchlistBoost + familyBonus + (prob1 * 320) + (touch * 220) + (path * 110) + (stage1 * 8) - penalties;
}
function applyRelativeStrengthRanks(rows) {
  const ranked = (rows || []).slice().sort((a, b) => {
    const scoreDiff = computePromiseScore(b) - computePromiseScore(a);
    if (scoreDiff !== 0) return scoreDiff;
    const signalDiff = signalPriorityValue(b) - signalPriorityValue(a);
    if (signalDiff !== 0) return signalDiff;
    const probDiff = Number(b?.prob_1 || 0) - Number(a?.prob_1 || 0);
    if (probDiff !== 0) return probDiff;
    const touchDiff = Number(b?.prob_touch || 0) - Number(a?.prob_touch || 0);
    if (touchDiff !== 0) return touchDiff;
    const pathDiff = Number(b?.prob_path || 0) - Number(a?.prob_path || 0);
    if (pathDiff !== 0) return pathDiff;
    return String(a?.symbol || '').localeCompare(String(b?.symbol || ''));
  });
  ranked.forEach((row, idx) => {
    row.relative_strength_rank = idx + 1;
    row.relative_strength_score = computePromiseScore(row);
  });
}
function rankBadge(rank) {
  const n = Number(rank || 0);
  let tone = 'slate';
  if (n > 0 && n <= 3) tone = 'emerald';
  else if (n > 0 && n <= 10) tone = 'cyan';
  else if (n > 0 && n <= 25) tone = 'amber';
  return `<span class="inline-flex min-w-[68px] items-center justify-center rounded-full border px-3 py-1 text-xs font-bold ${tone === 'emerald' ? 'border-emerald-200 bg-emerald-100 text-emerald-900' : tone === 'cyan' ? 'border-cyan-200 bg-cyan-100 text-cyan-900' : tone === 'amber' ? 'border-amber-200 bg-amber-100 text-amber-900' : 'border-slate-200 bg-slate-100 text-slate-700'}">#${n || '—'}</span>`;
}
function riskSummary(row) {
  const parts = [];
  if (row?.risk) parts.push(String(row.risk));
  if (row?.uncertainty) parts.push(`${String(row.uncertainty)} uncertainty`);
  if (row?.risk_reasons) parts.push(String(row.risk_reasons));
  return parts.filter(Boolean).join(' · ') || '—';
}

function sortRows(rows) {
  rows.sort((a, b) => {
    const va = a[sortKey];
    const vb = b[sortKey];
    if (va === vb) return 0;
    if (va === null || va === undefined) return 1;
    if (vb === null || vb === undefined) return -1;
    if (sortDir === "asc") return (va > vb) ? 1 : -1;
    return (va < vb) ? 1 : -1;
  });
}
function applyFiltersLocal() {
  const rows = allRows.slice();
  sortRows(rows);
  renderRows(rows);
  renderRowSummary(rows);
}
function renderRowSummary(rows) {
  setText("kpiRows", rows.length || 0);
  const actionable = rows.filter(r => r.signal === "ACTIONABLE").length;
  const candidate = rows.filter(r => r.signal === "CANDIDATE").length;
  setText("kpiActionable", actionable > 0 ? actionable : candidate);
  const kpiSubEl = document.querySelector('#kpiActionable')?.closest('.panel')?.querySelector('.text-xs');
  if (kpiSubEl) kpiSubEl.textContent = actionable > 0 ? 'Signal = ACTIONABLE' : (candidate > 0 ? 'Signal = CANDIDATE (tail unvalidated)' : 'No signals yet');
  const avg = rows.length ? rows.reduce((acc, r) => acc + Number(r.prob_touch || 0), 0) / rows.length : 0;
  setText("kpiAvgProb", fmtProb(avg));
  const promiseOrdered = rows.slice().sort((a, b) => Number(a.relative_strength_rank || 999999) - Number(b.relative_strength_rank || 999999));
  const top = promiseOrdered[0] || null;
  setText("kpiTopSymbol", top ? (top.symbol || "—") : "—");
  setText("kpiTopProb", top ? `Touch ${fmtProb(top.prob_touch)} · Path ${fmtProb(top.prob_path)} · ${top.sector || 'No sector'}` : "Waiting for scores");
  const clean = Number(latestDiagnosticsSummary.clean_touch_count || 0);
  const ugly = Number(latestDiagnosticsSummary.ugly_touch_count || 0);
  if (!clean && !ugly) {
    setText("kpiDiagHealth", "—");
    setText("kpiDiagNote", "Clean vs ugly touch mix");
  } else {
    const ratio = ugly === 0 ? clean : (clean / ugly);
    setText("kpiDiagHealth", `${fmt(ratio, 1)}x`);
    setText("kpiDiagNote", `Clean ${clean} vs ugly ${ugly}`);
  }
}
function renderRows(rows) {
  const tbody = el("rows");
  tbody.innerHTML = "";
  const empty = el("emptyState");
  if (!rows || rows.length === 0) {
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  for (const r of rows) {
    const priceVsVwap = (r.price !== null && r.price !== undefined && r.vwap !== null && r.vwap !== undefined)
      ? ((Number(r.price) / Number(r.vwap)) - 1)
      : null;
    const tr = document.createElement("tr");
    tr.className = "transition hover:bg-slate-50/90";
    tr.innerHTML = `
      <td class="px-4 py-3 pr-4 align-top">${rankBadge(r.relative_strength_rank)}</td>
      <td class="px-4 py-3 pr-4 align-top">
        <div class="font-semibold text-slate-950">${escapeHtml(r.symbol || "")}</div>
        <div class="mt-1 text-xs text-slate-500">${escapeHtml(r.sector || "") || "—"}</div>
      </td>
      <td class="px-4 py-3 pr-4 align-top font-medium text-slate-900">${fmt(r.price, 2)}</td>
      <td class="px-4 py-3 pr-4 align-top">
        <div class="text-xs ${priceVsVwap !== null && priceVsVwap >= 0 ? 'text-emerald-700' : 'text-rose-700'}">${priceVsVwap === null ? '—' : fmtPct(priceVsVwap, 2)} vs VWAP</div>
      </td>
      <td class="px-4 py-3 pr-4 align-top">${probBadge(r.prob_1)}</td>
      <td class="px-4 py-3 pr-4 align-top">${touchBadge(r.prob_touch)}</td>
      <td class="px-4 py-3 pr-4 align-top">${pathBadge(r.prob_path)}</td>
      <td class="px-4 py-3 pr-4 align-top text-slate-700 whitespace-nowrap">${escapeHtml(r.setup_family || "OTHER")}</td>
      <td class="px-4 py-3 pr-4 align-top">${signalBadge(r.signal)}</td>
      <td class="px-4 py-3 pr-4 align-top font-medium text-slate-900">${fmt(r.downside_risk ?? null, 2)}</td>
      <td class="px-4 py-3 pr-4 align-top text-slate-600 max-w-[240px] whitespace-normal">${escapeHtml(riskSummary(r))}</td>
      <td class="px-4 py-3 pr-4 align-top max-w-[360px] whitespace-normal text-slate-600">${escapeHtml(r.reasons || "") || "—"}</td>`;
    tbody.appendChild(tr);
  }
}
function renderWatchlist(rows) {
  const tbody = el("watchlistRows");
  if (!tbody) return;
  tbody.innerHTML = "";
  const empty = el("watchlistEmpty");
  if (!rows || rows.length === 0) {
    if (empty) empty.classList.remove("hidden");
    return;
  }
  if (empty) empty.classList.add("hidden");
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.className = "transition hover:bg-slate-50/90";
    tr.innerHTML = `
      <td class="px-4 py-3 pr-4 align-top">${rankBadge(r.relative_strength_rank)}</td>
      <td class="px-4 py-3 pr-4 align-top">
        <div class="font-semibold text-slate-950">${escapeHtml(r.symbol || "")}</div>
        <div class="mt-1 text-xs text-slate-500">${escapeHtml(r.sector || "") || "—"}</div>
      </td>
      <td class="px-4 py-3 pr-4 align-top font-medium text-slate-900">${fmt(r.price, 2)}</td>
      <td class="px-4 py-3 pr-4 align-top">${touchBadge(r.prob_touch)}</td>
      <td class="px-4 py-3 pr-4 align-top">${pathBadge(r.prob_path)}</td>
      <td class="px-4 py-3 pr-4 align-top text-slate-700 whitespace-nowrap">${escapeHtml(r.setup_family || "OTHER")}</td>
      <td class="px-4 py-3 pr-4 align-top text-slate-600 max-w-[360px] whitespace-normal">${escapeHtml(r.reasons || r.watchlist_reason || "—")}</td>`;
    tbody.appendChild(tr);
  }
}

function renderTracked(rows) {
  const tbody = el("diagTrackedRows");
  tbody.innerHTML = "";
  for (const r of (rows || [])) {
    const tr = document.createElement("tr");
    tr.className = "transition hover:bg-slate-50/90";
    const touchProb = r.max_prob_touch || r.max_prob_1 || 0;
    const riskNotes = [
      ...((r.risk_set || []).map(x => String(x))),
      ...((r.uncertainty_set || []).map(x => String(x)))
    ].filter(Boolean).join(', ');
    tr.innerHTML = `
      <td class="px-4 py-3 pr-4 font-semibold text-slate-950">${escapeHtml(r.symbol || "")}</td>
      <td class="px-4 py-3 pr-4">${touchBadge(touchProb)}</td>
      <td class="px-4 py-3 pr-4">${probBadge(r.max_prob_1)}</td>
      <td class="px-4 py-3 pr-4 text-slate-700">${r.times_signal || 0}</td>
      <td class="px-4 py-3 pr-4 text-slate-900">${fmt(r.max_downside_risk, 2)}</td>
      <td class="px-4 py-3 pr-4 text-slate-600 max-w-[260px] whitespace-normal">${escapeHtml(riskNotes || "—")}</td>`;
    tbody.appendChild(tr);
  }
}
function renderEvaluation(rows) {
  const tbody = el("diagEvaluationRows");
  tbody.innerHTML = "";
  const empty = el("diagEmpty");
  if (!rows || rows.length === 0) {
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.className = "transition hover:bg-slate-50/90";
    tr.innerHTML = `
      <td class="px-4 py-3 pr-4 font-semibold text-slate-950">${escapeHtml(r.symbol || "")}</td>
      <td class="px-4 py-3 pr-4">${probBadge(r.best_prob_1)}</td>
      <td class="px-4 py-3 pr-4 text-slate-700">${r.touch_1pct ? "Yes" : "No"}</td>
      <td class="px-4 py-3 pr-4">${verdictBadge(r.path_verdict)}</td>
      <td class="px-4 py-3 pr-4 text-slate-900">${fmtPct(r.mae_before_touch_pct, 2)}</td>
      <td class="px-4 py-3 pr-4 text-slate-900">${fmtPct(r.close_vs_scan_pct, 2)}</td>
      <td class="px-4 py-3 pr-4">${bucketBadge(r.review_bucket)}</td>`;
    tbody.appendChild(tr);
  }
}
async function refreshScores() {
  const resp = await fetch("/api/scores");
  const data = await resp.json();
  allRows = data.rows || [];
  applyRelativeStrengthRanks(allRows);
  watchlistRescueRows = data.watchlist_rescue_rows || [];
  setText("lastRun", data.last_run_utc || "—");
  setText("watchlistCount", watchlistRescueRows.length || 0);
  renderWatchlist(watchlistRescueRows);
  applyFiltersLocal();
}
async function refreshTraining() {
  const resp = await fetch("/api/training/status");
  const t = await resp.json();
  setText("trainRunning", t.running ? "Running" : "Idle");
  setText("trainStarted", t.started_at_utc || "—");
  setText("trainFinished", t.finished_at_utc || "—");
  setText("trainBadge", t.running ? "Training live" : "Ready");
  const badge = el("trainBadge");
  badge.className = "rounded-full px-3 py-1 text-xs font-medium " + (t.running ? "bg-amber-100 text-amber-800" : "bg-emerald-100 text-emerald-800");
  const errorNode = el("trainError");
  const err = t.last_error || "";
  errorNode.textContent = err;
  errorNode.classList.toggle("hidden", !err);
}
async function refreshDiagnostics() {
  const resp = await fetch("/api/diagnostics");
  const d = await resp.json();
  const summary = d.summary || {};
  latestDiagnosticsSummary = summary;
  setText("diagTradeDate", d.trade_date || "—");
  setText("diagSnapshots", summary.snapshots_count || 0);
  setText("diagTracked", summary.tracked_count || 0);
  setText("diagCapBound", summary.signaled_count || 0);
  setText("diagEvaluated", summary.evaluated ? "Yes" : "No");
  setText("diagClean", summary.clean_touch_count || 0);
  setText("diagUgly", summary.ugly_touch_count || 0);
  setText("diagNoTouch", summary.no_touch_count || 0);
  setText("diagWorth", summary.worthy_count || 0);
  setText("diagLatestSnapshot", summary.latest_snapshot_utc || "—");
  renderTracked(d.tracked || []);
  renderEvaluation((d.evaluation && d.evaluation.rows) ? d.evaluation.rows : []);
  renderRowSummary(allRows.slice());
}
async function downloadReviewPack() {
  const btn = el("downloadReviewPack");
  const original = btn ? btn.textContent : "Download review pack";
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Preparing…";
    btn.classList.add("opacity-60", "cursor-not-allowed");
  }
  try {
    const tradeDate = (el("diagTradeDate")?.textContent || "").trim();
    const qs = tradeDate && tradeDate !== "—" ? `?trade_date=${encodeURIComponent(tradeDate)}` : "";
    const resp = await fetch(`/api/review-export/download${qs}`);
    if (!resp.ok) throw new Error(`Export failed (${resp.status})`);
    const blob = await resp.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    const fallbackDate = tradeDate && tradeDate !== "—" ? tradeDate : new Date().toISOString().slice(0, 10);
    a.href = url;
    a.download = `post_close_review_${fallbackDate}.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    alert(err?.message || "Export failed");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = original;
      btn.classList.remove("opacity-60", "cursor-not-allowed");
    }
  }
}







async function refreshStatus() {
  const resp = await fetch("/api/status");
  const s = await resp.json();
  setText("clock", (new Date()).toLocaleString());
  const marketOpen = !!s.market_open;
  const badge = el("marketBadge");
  badge.textContent = marketOpen ? "Market open" : "Market closed";
  badge.className = "inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold " + (marketOpen ? "border-emerald-400/30 bg-emerald-400/15 text-emerald-100" : "border-white/10 bg-white/10 text-slate-100");

  const alp = s.alpaca || {};
  setText("alpacaStatus", alp.ok ? "OK" : (alp.message || "Not connected"));
  setText("feedStatus", (alp.feed || "sip").toUpperCase());
  setText("lastBar", alp.last_bar_timestamp || "—");
  setText("rateLimit", alp.rate_limit_warn || "");

  const ttc = s.time_to_close_seconds || 0;
  setText("ttc", marketOpen ? (Math.floor(ttc / 60) + " min") : "Closed / waiting");

  const regime = s.regime || {};
  const regimeConfig = s.regime_config || {};
  const regimeState = String(regime.state || "NOT_EVALUATED").toUpperCase();
  const regimeBadgeNode = el("regimeBadge");
  if (regimeBadgeNode) {
    regimeBadgeNode.textContent = regimeState;
    regimeBadgeNode.className = "inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold " + (regimeState === "RED"
      ? "border-rose-300/30 bg-rose-300/15 text-rose-50"
      : regimeState === "AMBER"
        ? "border-amber-300/30 bg-amber-300/15 text-amber-50"
        : regimeState === "GREEN"
          ? "border-emerald-300/30 bg-emerald-300/15 text-emerald-50"
          : regimeState === "CLOSED"
            ? "border-indigo-300/30 bg-indigo-300/15 text-indigo-50"
            : "border-slate-300/30 bg-slate-300/15 text-slate-100");
  }
  const regimeTitle = regimeState === "RED"
    ? "Headline-sensitive market structure"
    : regimeState === "AMBER"
      ? "Elevated event-risk market structure"
      : regimeState === "GREEN"
        ? "Normal market structure"
        : regimeState === "CLOSED"
          ? "Market closed — showing last live regime"
          : "Regime not yet evaluated";
  setText("regimeTitle", regimeTitle);
  const lastLiveState = regime.last_live_state ? String(regime.last_live_state).toUpperCase() : "";
  let regimeReasons = regime.reasons || regime.note || (regimeState === "GREEN" ? "Proxy regime normal" : "No live proxy evaluation yet");
  if (regime.note && regime.reasons && regime.note !== regime.reasons) regimeReasons += ` · ${regime.note}`;
  if (lastLiveState && regimeState !== lastLiveState) regimeReasons += ` · last live ${lastLiveState}`;
  if (regime.evaluated_at_utc) regimeReasons += ` · evaluated ${regime.evaluated_at_utc}`;
  if (regime.cooldown_until_utc) regimeReasons += ` · until ${regime.cooldown_until_utc}`;
  setText("regimeSummary", regimeReasons);
  setText("regimeMultiplier", regime.live_evaluated ? `${fmt(regime.multiplier ?? 1, 2)}x` : "—");
  setText("regimeCap", regime.live_evaluated && !(regime.prob_cap === null || regime.prob_cap === undefined) ? fmtProb(regime.prob_cap) : "—");
  setText("regimeSignalPolicy", regimeState === "CLOSED" ? "Market closed" : (regime.suppress_new_signals ? "Suppress new signals" : (regime.live_evaluated ? "Normal gating" : "Waiting for live proxy evaluation")));
  const proxyPieces = [];
  const metrics = regime.metrics || {};
  const oilMetric = metrics.oil || {};
  const volMetric = metrics.volatility || {};
  const havenMetric = metrics.safe_haven || {};
  const energyMetric = metrics.energy_vs_spy || {};
  if (regimeConfig.oil_proxy && oilMetric.ret_1h !== undefined && oilMetric.ret_1h !== null) proxyPieces.push(`${regimeConfig.oil_proxy}: ${fmtPct(oilMetric.ret_1h, 2)}`);
  if (regimeConfig.vol_proxy && volMetric.ret_1h !== undefined && volMetric.ret_1h !== null) proxyPieces.push(`${regimeConfig.vol_proxy}: ${fmtPct(volMetric.ret_1h, 2)}`);
  if (regimeConfig.safe_haven_proxy && havenMetric.ret_1h !== undefined && havenMetric.ret_1h !== null) proxyPieces.push(`${regimeConfig.safe_haven_proxy}: ${fmtPct(havenMetric.ret_1h, 2)}`);
  if (energyMetric.ret_1h !== undefined && energyMetric.ret_1h !== null) proxyPieces.push(`XLE-SPY: ${fmtPct(energyMetric.ret_1h, 2)}`);
  setText("regimeProxyNote", proxyPieces.length ? `1h proxy moves — ${proxyPieces.join(" · ")}` : (regime.live_evaluated ? "Proxy set evaluated, but no proxy deltas were available." : "No live proxy move set available yet."));
  const banner = el("regimeBanner");
  if (banner) {
    banner.className = "mb-5 rounded-[24px] px-5 py-4 text-slate-100 " + (regimeState === "RED"
      ? "border border-rose-200/20 bg-rose-400/10"
      : regimeState === "AMBER"
        ? "border border-amber-200/20 bg-amber-400/10"
        : regimeState === "GREEN"
          ? "border border-cyan-200/20 bg-cyan-400/10"
          : regimeState === "CLOSED"
            ? "border border-indigo-200/20 bg-indigo-400/10"
            : "border border-slate-200/20 bg-slate-400/10");
  }

  const m = (s.model && s.model.pt1) ? s.model.pt1 : {};
  latestModelMeta = m;
  let m1 = m.trained ? "pt1 trained" : "pt1 untrained";
  if (m.trained && (m.decision_tail_validated || m.selection_tier === 'decision_tail_validated')) m1 += " · decision tail validated";
  else if (m.trained) m1 += " · decision tail unvalidated";
  if (m.model_b_method) m1 += ` · B:${m.model_b_method}`;
  else if (m.trained) m1 += ' · NO PATH MODEL';
  setText("modelStatus", m1);
  setText("liveCapStatus", m.trained ? `Probabilities: ${m.probability_contract || 'uncapped_decomposed'}` : "No model loaded");
  const tierText = (m.selection_tier === 'decision_tail_validated' || m.decision_tail_validated) ? 'Decision tail validated' : (m.selection_tier === 'touch_tail_validated' ? 'Touch tail validated' : (m.selection_tier || "Waiting for model metadata"));
  setText("selectionTier", tierText);
  setText("selectionWarningTop", m.selection_warning || "No active selection warning");
  // v10: new scanner header indicators
  const ath = m.adaptive_decision_threshold || m.adaptive_touch_threshold;
  const athText = ath ? ` (>=${(ath * 100).toFixed(1)}%)` : '';
  const tailValidated = Boolean(m.decision_tail_validated || m.selection_tier === 'decision_tail_validated');
  setText("touchTailStatus", (m.trained ? (tailValidated ? 'Validated' : 'Unvalidated') : 'No model') + athText);
  if (ath) _adaptiveThreshold = ath; else _adaptiveThreshold = 0.10;
  const pqm = s?.training_selection?.path_quality_action_min;
  if (pqm) _pathQualityMin = pqm;
  const msrc = m.model_source || (m.trained ? (m.model_b_method ? 'trained' : 'trained_no_path') : 'heuristic');
  setText("modelSource", msrc === 'trained' ? `A+B (${m.model_b_method || '?'})` : (msrc === 'trained_no_path' ? 'A only (no path model)' : 'Heuristic'));
  setText("probContract", m.trained ? (m.probability_contract || 'uncapped') : 'heuristic fallback');

  const uni = s.constituents || {};
  setText("universeStatus", `${uni.count || 0} (${uni.source || 'fallback'})`);

  const cov = s.coverage || {};
  setText("coverageStatus", `Stage 1 ${cov.stage1_candidate_count || 0} → Stage 2 ${cov.stage2_scored_count || 0}${cov.stage1_strong_override_count ? ` (${cov.stage1_strong_override_count} override)` : ""}`);
  setText("skipReasons", fmtSkipReasons(cov.top_skip_reasons));
  setText("profileNote", (m.selection_warning || cov.profile_note || ""));
  setText("lastError", s.last_error || "");

  const tc = cov.threshold_counts || {};
  const atc = cov.acceptable_threshold_counts || {};
  setText("tailCounts", `Decision score: low ${tc.ge_low || 0} · mid ${tc.ge_mid || 0} · high ${tc.ge_high || 0} · top ${tc.ge_top || 0}`);
  setText("acceptableTailCounts", `Acceptable decision: low ${atc.ge_low || 0} · mid ${atc.ge_mid || 0} · high ${atc.ge_high || 0} · top ${atc.ge_top || 0}`);
  const gs = cov.guardrail_stats || {};
  setText("guardrailStats", `Blocked ${gs.blocked_in_universe || 0} · Weak structure ${gs.weak_structure_in_universe || 0} · Event ${gs.event_in_universe || 0} · High downside ${gs.high_downside_in_candidates || 0} · High uncertainty ${gs.high_uncertainty_in_candidates || 0} · Rescue ${gs.watchlist_rescue_count || 0} · Regime ${gs.regime_state || "NOT_EVALUATED"}`);

  await refreshTraining();
}
async function startTraining() {
  const pwd = el("adminPassword").value || "";
  const form = new FormData();
  form.append("admin_password", pwd);
  const resp = await fetch("/train", { method: "POST", body: form });
  const js = await resp.json();
  if (!resp.ok) alert(js.error || "Training failed");
  await refreshTraining();
}

window.addEventListener("DOMContentLoaded", async () => {
  el("startTraining").addEventListener("click", startTraining);
  el("downloadReviewPack")?.addEventListener("click", downloadReviewPack);
  document.querySelectorAll("th[data-sort]").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.getAttribute("data-sort");
      if (sortKey === key) sortDir = sortDir === "desc" ? "asc" : "desc";
      else {
        sortKey = key;
        sortDir = key === "relative_strength_rank" ? "asc" : "desc";
      }
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
