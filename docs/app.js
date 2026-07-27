/* Market Regime Monitor — front-end (dependency-free) */

const SVGNS = "http://www.w3.org/2000/svg";

// Shared regime metadata. `k` maps to a CSS colour variable --c-<k>.
const REGIME = {
  "Risk-On":         { k: "risk-on",  label: "Risk-On",          blurb: "Expansion — broad participation" },
  "Risk-On-Retreat": { k: "retreat",  label: "Risk-On-Retreat",  blurb: "Late-cycle — momentum slowing" },
  "Risk-Off":        { k: "risk-off", label: "Risk-Off",         blurb: "Crisis / bear market" },
  "Risk-Off-Stable": { k: "stable",   label: "Risk-Off-Stable",  blurb: "Recovery / accumulation" },
  "bull":            { k: "risk-on",  label: "Bull",             blurb: "Uptrend, normal volatility" },
  "bull_high_vol":   { k: "risk-on",  label: "Bull · high-vol",  blurb: "Uptrend, elevated volatility" },
  "bear":            { k: "risk-off", label: "Bear",             blurb: "Downtrend" },
  "bear_high_vol":   { k: "risk-off", label: "Bear · high-vol",  blurb: "Downtrend, elevated volatility" },
  "range":           { k: "neutral",  label: "Range",            blurb: "Sideways / non-directional" },
  "high_vol":        { k: "volatile", label: "High Volatility",  blurb: "Directionless, high volatility" },
};

// HMM probability order (fixed — matches the four states, bullish→bearish).
const HMM_ORDER = ["Risk-On", "Risk-On-Retreat", "Risk-Off", "Risk-Off-Stable"];
// Rule-based composite order (bullish→bearish), for the validation panel.
const RULE_ORDER = ["bull", "bull_high_vol", "range", "high_vol", "bear_high_vol", "bear"];

// Legend entries for the ribbon (colour → what it means across both engines).
const LEGEND = [
  ["risk-on", "Risk-On / Bull"],
  ["retreat", "Risk-On-Retreat"],
  ["risk-off", "Risk-Off / Bear"],
  ["stable", "Risk-Off-Stable"],
  ["neutral", "Range"],
  ["volatile", "High Volatility"],
];

const meta = (r) => REGIME[r] || { k: "neutral", label: r, blurb: "" };
const cvar = (k) => `var(--c-${k})`;
const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}) => {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};

// ── Theme toggle ──────────────────────────────────────────────────────────
(function initTheme() {
  const saved = localStorage.getItem("regime-theme");
  if (saved) document.documentElement.setAttribute("data-theme", saved);
  $("theme-toggle").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const isDark = cur === "dark" ||
      (!cur && matchMedia("(prefers-color-scheme: dark)").matches);
    const next = isDark ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("regime-theme", next);
  });
})();

// ── Boot ────────────────────────────────────────────────────────────────────
Promise.all([
  fetch("data/latest.json").then((r) => r.json()),
  fetch("data/history.json").then((r) => r.json()),
  fetch("data/validation.json").then((r) => r.json()),
])
  .then(([latest, history, validation]) => {
    renderLatest(latest);
    renderProbBars(latest.hmm.probabilities);
    renderLegend();
    renderRibbon(history);
    initValidation(validation);
    window.addEventListener("resize", debounce(() => renderRibbon(history), 150));
  })
  .catch((err) => {
    $("app").insertAdjacentHTML("afterbegin",
      `<div class="panel" style="border-color:var(--c-risk-off)">Failed to load data: ${err}</div>`);
  });

// ── Latest read ───────────────────────────────────────────────────────────
function renderLatest(d) {
  $("as-of").textContent = d.as_of;
  const stale = (d.data_freshness?.days_since_last_bar ?? 0) > 4;
  $("freshness").classList.toggle("stale", stale);
  $("freshness-txt").textContent = stale
    ? `data ${d.data_freshness.days_since_last_bar}d old`
    : "data fresh";
  $("generated").textContent = (d.generated_at || "").replace("T", " ").replace("Z", " UTC");

  // Rule-based card
  const rb = d.rule_based;
  const rm = meta(rb.regime);
  const rcard = $("card-rule");
  rcard.style.setProperty("--accent", cvar(rm.k));
  $("rule-name").textContent = rm.label;
  $("rule-blurb").textContent = rm.blurb;
  $("rule-conf").textContent = fmtPct(rb.confidence);
  $("rule-meter").style.width = pct(rb.confidence);
  $("rule-diag").innerHTML = diagRows([
    ["Trend", rb.trend, ""],
    ["EMA stack", rb.ema_stack, ""],
    ["Volatility", rb.vol, ""],
    ["ATR %", rb.atr_percent, "%"],
    ["Chop", rb.chop, ""],
    ["ADX", rb.adx, ""],
  ]);

  // HMM card
  const hm = meta(d.hmm.regime);
  const hcard = $("card-hmm");
  hcard.style.setProperty("--accent", cvar(hm.k));
  $("hmm-name").textContent = hm.label;
  $("hmm-blurb").textContent = hm.blurb;
  $("hmm-conf").textContent = fmtPct(d.hmm.confidence);
  $("hmm-meter").style.width = pct(d.hmm.confidence);
  const tr = d.hmm.transition_risk;
  const trChip = $("hmm-tr");
  trChip.className = "chip tr-" + tr;
  $("hmm-tr-txt").textContent = tr;

  // Agreement
  const agreeBox = $("agreement");
  agreeBox.classList.toggle("agree", d.agreement);
  agreeBox.classList.toggle("disagree", !d.agreement);
  $("agreement-verdict").textContent = d.agreement ? "Engines agree" : "Engines diverge";
  $("agreement-note").textContent = d.agreement ? "" : "read with caution";
}

function diagRows(rows) {
  return rows.map(([k, v, suf]) =>
    `<div><dt>${k}</dt><b>${v ?? "—"}${suf || ""}</b></div>`).join("");
}

// ── Probability bars ────────────────────────────────────────────────────────
function renderProbBars(probs) {
  $("prob-bars").innerHTML = HMM_ORDER.map((name) => {
    const m = meta(name);
    const v = probs[name] ?? 0;
    return `<div class="prob-row" style="--accent:${cvar(m.k)}">
      <span class="lbl"><span class="sw"></span>${m.label}</span>
      <span class="prob-track"><span class="prob-fill" style="width:${v}%"></span></span>
      <span class="prob-val">${v.toFixed(1)}%</span>
    </div>`;
  }).join("");
}

// ── Legend ────────────────────────────────────────────────────────────────
function renderLegend() {
  $("ribbon-legend").innerHTML = LEGEND.map(([k, lbl]) =>
    `<span class="item" style="--accent:${cvar(k)}"><span class="sw"></span>${lbl}</span>`
  ).join("");
}

// ── Validation panel ────────────────────────────────────────────────────────
const valState = { data: null, engine: "hmm", horizon: "21" };

function initValidation(v) {
  valState.data = v;
  const start = v.range?.start?.slice(0, 4), end = v.range?.end?.slice(0, 4);
  $("val-caveat").textContent =
    `Descriptive, in-sample association over ${start}–${end} using overlapping windows — ` +
    `not a forecast. The HMM's parameters were fit on 2022–2026, so most of this span sits ` +
    `outside its training window. "% up" = share of windows with a positive return; n = sample size.`;

  $("val-engine").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    valState.engine = b.dataset.engine;
    setActive($("val-engine"), b);
    renderValidation();
  });
  $("val-horizon").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    valState.horizon = b.dataset.h;
    setActive($("val-horizon"), b);
    renderValidation();
  });
  renderValidation();
}

function setActive(group, btn) {
  [...group.children].forEach((c) => c.classList.toggle("active", c === btn));
}

function renderValidation() {
  const { data, engine, horizon } = valState;
  const rows = (data[engine] && data[engine][horizon]) || [];
  const byRegime = new Map(rows.map((r) => [r.regime, r]));
  const order = engine === "hmm" ? HMM_ORDER : RULE_ORDER;
  const sorted = order.map((r) => byRegime.get(r)).filter(Boolean);
  if (!sorted.length) { $("val-bars").innerHTML = "<p class='sub'>No data.</p>"; return; }

  const vals = sorted.map((r) => r.mean);
  const dMin = Math.min(0, ...vals), dMax = Math.max(0, ...vals);
  const span = (dMax - dMin) || 1;
  const zero = ((0 - dMin) / span) * 100;

  $("val-bars").innerHTML = sorted.map((r) => {
    const m = meta(r.regime);
    const vPos = ((r.mean - dMin) / span) * 100;
    const left = r.mean >= 0 ? zero : vPos;
    const width = Math.max(Math.abs(vPos - zero), 0.6);
    return `<div class="vrow" style="--accent:${cvar(m.k)}">
      <span class="vlbl"><span class="sw"></span>${m.label}</span>
      <span class="vbar-area">
        <span class="vbar-zero" style="left:${zero}%"></span>
        <span class="vbar-fill" style="left:${left}%;width:${width}%"></span>
      </span>
      <span class="vstats"><b>${r.mean >= 0 ? "+" : ""}${r.mean}%</b> · ${r.hit_rate}% up · n=${r.n}</span>
    </div>`;
  }).join("");
}

// ── Historical ribbon (SVG) ─────────────────────────────────────────────────
function renderRibbon(history) {
  const host = $("ribbon");
  host.innerHTML = "";

  const spy = history.spy || [];
  if (spy.length < 2) return;

  // Master timeline = SPY dates. Index regimes by date for alignment + hover.
  const dates = spy.map((p) => p.date);
  const closes = spy.map((p) => p.close);
  const idxByDate = new Map(dates.map((d, i) => [d, i]));
  const hmmByDate = mapRegime(history.hmm, idxByDate, dates.length);
  const ruleByDate = mapRegime(history.rule_based, idxByDate, dates.length);

  // Geometry (viewBox units; CSS scales width to 100%).
  const W = 1000, mL = 46, mR = 12, mT = 8;
  const priceH = 190, gap = 16, rowH = 24, rowGap = 8, axisH = 22;
  const plotW = W - mL - mR;
  const yHMM = mT + priceH + gap;
  const yRule = yHMM + rowH + rowGap;
  const H = yRule + rowH + axisH + 6;

  const n = dates.length;
  const x = (i) => mL + (plotW * i) / (n - 1);
  const stepW = plotW / (n - 1);

  const lo = Math.min(...closes), hi = Math.max(...closes);
  const pad = (hi - lo) * 0.06 || 1;
  const yPrice = (c) => mT + priceH - ((c - (lo - pad)) / ((hi + pad) - (lo - pad))) * priceH;

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none",
    role: "img", "aria-label": "SPY price with regime bands over time" });

  // Price gridlines + labels
  const gridEnd = mL + plotW;
  [hi, (hi + lo) / 2, lo].forEach((val) => {
    const gy = yPrice(val);
    svg.appendChild(el("line", { class: "grid", x1: mL, y1: gy, x2: gridEnd, y2: gy }));
    const t = el("text", { class: "axis-txt", x: mL - 6, y: gy + 3, "text-anchor": "end" });
    t.textContent = Math.round(val);
    svg.appendChild(t);
  });

  // Regime ribbons
  drawRibbonRow(svg, hmmByDate, yHMM, rowH, x, stepW, n);
  drawRibbonRow(svg, ruleByDate, yRule, rowH, x, stepW, n);
  addRowLabel(svg, "HMM", mL - 6, yHMM + rowH / 2);
  addRowLabel(svg, "Rule", mL - 6, yRule + rowH / 2);

  // Price line (drawn over gridlines)
  let dstr = "";
  for (let i = 0; i < n; i++) dstr += (i ? "L" : "M") + x(i).toFixed(2) + "," + yPrice(closes[i]).toFixed(2);
  svg.appendChild(el("path", { class: "price-line", d: dstr }));

  // X axis: year ticks
  let prevYear = null;
  for (let i = 0; i < n; i++) {
    const yr = dates[i].slice(0, 4);
    if (yr !== prevYear) {
      prevYear = yr;
      if (i > 0) {
        svg.appendChild(el("line", { class: "grid", x1: x(i), y1: mT, x2: x(i), y2: yRule + rowH }));
      }
      const t = el("text", { class: "axis-txt", x: x(i) + 3, y: H - 6 });
      t.textContent = yr;
      svg.appendChild(t);
    }
  }

  // Hover layer
  const crosshair = el("line", { class: "crosshair", y1: mT, y2: yRule + rowH });
  const dot = el("circle", { class: "hover-dot", r: 3.5 });
  svg.appendChild(crosshair);
  svg.appendChild(dot);
  const overlay = el("rect", { x: mL, y: mT, width: plotW, height: yRule + rowH - mT,
    fill: "transparent", style: "cursor:crosshair" });
  svg.appendChild(overlay);

  host.appendChild(svg);

  const tip = $("ribbon-tooltip");
  const rectW = () => host.getBoundingClientRect().width || W;
  overlay.addEventListener("mousemove", (ev) => {
    const scale = rectW() / W;
    const rel = host.getBoundingClientRect();
    const svgX = (ev.clientX - rel.left) / scale;
    let i = Math.round(((svgX - mL) / plotW) * (n - 1));
    i = Math.max(0, Math.min(n - 1, i));
    const px = x(i);
    crosshair.setAttribute("x1", px); crosshair.setAttribute("x2", px);
    crosshair.setAttribute("opacity", 1);
    dot.setAttribute("cx", px); dot.setAttribute("cy", yPrice(closes[i]));
    dot.setAttribute("opacity", 1);

    const hm = meta(hmmByDate[i]), rm = meta(ruleByDate[i]);
    tip.innerHTML = `<div class="tt-date">${dates[i]}</div>
      <div class="tt-row"><span class="k">SPY</span><span class="v">${closes[i].toFixed(2)}</span></div>
      <div class="tt-row"><span class="sw" style="background:${cvar(hm.k)}"></span><span class="k">HMM</span><span class="v">${hm.label}</span></div>
      <div class="tt-row"><span class="sw" style="background:${cvar(rm.k)}"></span><span class="k">Rule</span><span class="v">${rm.label}</span></div>`;
    tip.style.opacity = 1;
    const left = px * scale + 14;
    const maxL = rectW() - tip.offsetWidth - 8;
    tip.style.left = Math.min(left, maxL) + "px";
    tip.style.top = "6px";
  });
  overlay.addEventListener("mouseleave", () => {
    tip.style.opacity = 0;
    crosshair.setAttribute("opacity", 0);
    dot.setAttribute("opacity", 0);
  });

  $("ribbon-sub").textContent =
    `SPY with both engines' regime classification, ${dates[0]} → ${dates[n - 1]}.`;
}

// Build an index-aligned array of regime labels from a [{date,regime}] series.
function mapRegime(series, idxByDate, n) {
  const out = new Array(n).fill(null);
  for (const row of series || []) {
    const i = idxByDate.get(row.date);
    if (i !== undefined) out[i] = row.regime;
  }
  // Forward-fill gaps so bands are continuous.
  let last = null;
  for (let i = 0; i < n; i++) {
    if (out[i] == null) out[i] = last;
    else last = out[i];
  }
  return out;
}

// Draw one regime row as runs of same-colour rects (2px gaps between runs).
function drawRibbonRow(svg, labels, y, h, x, stepW, n) {
  let start = 0;
  for (let i = 1; i <= n; i++) {
    const curK = i < n ? meta(labels[i]).k : null;
    const runK = meta(labels[start]).k;
    if (i === n || curK !== runK) {
      if (labels[start] != null) {
        const x0 = x(start);
        const w = Math.max(1, x(i - 1) - x0 + stepW - 1);
        svg.appendChild(el("rect", {
          class: "seg", x: x0.toFixed(2), y, width: w.toFixed(2), height: h,
          rx: 2, style: `fill:${cvar(runK)}`,
        }));
      }
      start = i;
    }
  }
}

function addRowLabel(svg, txt, x, y) {
  const t = el("text", { class: "row-lbl", x, y: y + 4, "text-anchor": "end" });
  t.textContent = txt;
  svg.appendChild(t);
}

// ── utils ───────────────────────────────────────────────────────────────────
const pct = (v) => `${Math.round((v <= 1 ? v * 100 : v))}%`;
const fmtPct = (v) => `${Math.round((v <= 1 ? v * 100 : v))}%`;
function debounce(fn, ms) {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
