let allRows = [];
let sortKey = "prob_2";
let sortDir = "desc";

function fmt(x, d=2) {
  if (x === null || x === undefined) return "";
  if (typeof x !== "number") return String(x);
  return x.toFixed(d);
}
function fmtProb(x) { return fmt(x, 3); }

function applyFiltersLocal() {
  const minProb = parseFloat(document.getElementById("minProb").value || "0");
  const sector = document.getElementById("sector").value || "";
  const pMin = document.getElementById("priceMin").value ? parseFloat(document.getElementById("priceMin").value) : null;
  const pMax = document.getElementById("priceMax").value ? parseFloat(document.getElementById("priceMax").value) : null;

  let rows = allRows.slice();
  rows = rows.filter(r => (r.prob_2 ?? 0) >= minProb);
  if (sector) rows = rows.filter(r => (r.sector || "") === sector);
  if (pMin !== null) rows = rows.filter(r => (r.price ?? 0) >= pMin);
  if (pMax !== null) rows = rows.filter(r => (r.price ?? 0) <= pMax);

  rows.sort((a,b) => {
    const va = a[sortKey]; const vb = b[sortKey];
    if (va === vb) return 0;
    if (sortDir === "asc") return (va > vb) ? 1 : -1;
    return (va < vb) ? 1 : -1;
  });

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
      <td class="py-2 pr-4">${fmtProb(r.prob_1)}</td>
      <td class="py-2 pr-4 font-semibold">${fmtProb(r.prob_2)}</td>
      <td class="py-2 pr-4">${r.sector || ""}</td>
      <td class="py-2 pr-4 text-slate-600">${r.reasons || ""}</td>
    `;
    tbody.appendChild(tr);
  }
}

function populateSectors(rows) {
  const s = new Set();
  for (const r of rows) if (r.sector) s.add(r.sector);
  const sel = document.getElementById("sector");
  const cur = sel.value;
  while (sel.options.length > 1) sel.remove(1);
  Array.from(s).sort().forEach(sec => {
    const opt = document.createElement("option");
    opt.value = sec;
    opt.textContent = sec;
    sel.appendChild(opt);
  });
  sel.value = cur;
}

async function refreshStatus() {
  const resp = await fetch("/api/status");
  const s = await resp.json();
  const now = new Date();
  document.getElementById("clock").textContent = now.toLocaleString();

  const marketOpen = !!s.market_open;
  const badge = document.getElementById("marketBadge");
  badge.textContent = marketOpen ? "Market open" : "Market closed";
  badge.className = "px-2 py-1 rounded-full text-xs " + (marketOpen ? "bg-emerald-100 text-emerald-800" : "bg-slate-100 text-slate-700");

  const alp = s.alpaca || {};
  document.getElementById("alpacaStatus").textContent = alp.ok ? "OK" : (alp.message || "Not connected");
  document.getElementById("feedStatus").textContent = (alp.feed || "sip").toUpperCase();
  document.getElementById("lastBar").textContent = alp.last_bar_timestamp || "—";
  document.getElementById("rateLimit").textContent = alp.rate_limit_warn || "—";

  const ttc = s.time_to_close_seconds || 0;
  document.getElementById("ttc").textContent = marketOpen ? Math.floor(ttc/60) + " min" : "—";

  const m1 = (s.model && s.model.pt1 && s.model.pt1.trained) ? "pt1 ✅" : "pt1 ⏳";
  const m2 = (s.model && s.model.pt2 && s.model.pt2.trained) ? "pt2 ✅" : "pt2 ⏳";
  document.getElementById("modelStatus").textContent = m1 + " " + m2;

  const uni = s.constituents || {};
  const uniText = (uni.count || 0) + " (" + (uni.source || "fallback") + ")";
  document.getElementById("universeStatus").textContent = uniText;

  document.getElementById("lastError").textContent = s.last_error || "";

  await refreshTraining();
}

async function refreshScores() {
  const resp = await fetch("/api/scores");
  const js = await resp.json();
  allRows = js.rows || [];
  document.getElementById("lastRun").textContent = js.last_run_utc || "—";
  populateSectors(allRows);
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

async function startTraining() {
  const pw = document.getElementById("adminPassword").value || "";
  const form = new FormData();
  form.append("admin_password", pw);
  const resp = await fetch("/train", { method: "POST", body: form });
  const js = await resp.json();
  if (!js.ok) {
    document.getElementById("trainError").textContent = js.error || "Training failed to start";
    return;
  }
  document.getElementById("trainError").textContent = "";
  await refreshTraining();
}

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
    if (sortKey === key) sortDir = (sortDir === "asc") ? "desc" : "asc";
    else { sortKey = key; sortDir = (key === "symbol") ? "asc" : "desc"; }
    applyFiltersLocal();
  });
});

async function tick() {
  try { await refreshStatus(); } catch(e) {}
  try { await refreshScores(); } catch(e) {}
}
tick();
setInterval(tick, 10000);
setInterval(refreshTraining, 4000);
