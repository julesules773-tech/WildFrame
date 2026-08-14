/* ===================================================================
   WildFrame — Admin Dashboard | Moderation Logic
   =================================================================== */

(function () {
  "use strict";

  // -----------------------------------------------------------------------
  // State
  // -----------------------------------------------------------------------
  const STATE = {
    secret: sessionStorage.getItem("wildframe_admin_secret") || "",
    reports: [],
    autoApproved: [],
    fires: [],
    processing: new Set(),  // report IDs currently being processed
  };

  // -----------------------------------------------------------------------
  // DOM refs
  // -----------------------------------------------------------------------
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  const els = {
    loginOverlay: $("#login-overlay"),
    loginBtn: $("#admin-login-btn"),
    password: $("#admin-password"),
    loginError: $("#login-error"),
    dashboard: $("#admin-dashboard"),
    pendingCount: $("#pending-count"),
    refreshBtn: $("#admin-refresh-btn"),
    acceptAllBtn: $("#admin-accept-all-btn"),
    logoutBtn: $("#admin-logout-btn"),
    container: $("#reports-container"),
    emptyState: $("#empty-state"),
    autoSection: $("#auto-approved-section"),
    autoList: $("#auto-approved-list"),
    autoCount: $("#auto-approved-count"),
    autoToggle: $("#auto-approved-toggle"),
    firesSection: $("#fires-section"),
    firesList: $("#fires-list"),
    firesCount: $("#fires-count"),
    firesToggle: $("#fires-toggle"),
    toastContainer: $("#admin-toast-container"),
  };

  // -----------------------------------------------------------------------
  // Section collapse state (persisted per browser)
  // -----------------------------------------------------------------------
  // Both secondary sections start COLLAPSED so the reports grid — the
  // primary moderation workspace — gets the full height instead of being
  // squeezed by the fires / auto-approved panels. The choice sticks.
  const SECTION_UI_KEY = "wf.admin.sectionsCollapsed";
  let uiState = { firesCollapsed: true, autoCollapsed: true };
  try {
    const saved = JSON.parse(localStorage.getItem(SECTION_UI_KEY) || "{}");
    if (typeof saved.firesCollapsed === "boolean") uiState.firesCollapsed = saved.firesCollapsed;
    if (typeof saved.autoCollapsed === "boolean") uiState.autoCollapsed = saved.autoCollapsed;
  } catch (e) { /* ignore malformed storage */ }

  function applySectionState() {
    const toggle = (btn, collapsed) => {
      if (!btn) return;
      btn.classList.toggle("collapsed", collapsed);
      btn.setAttribute("aria-expanded", String(!collapsed));
      btn.title = collapsed ? "Expand" : "Collapse";
    };
    if (els.firesSection) els.firesSection.classList.toggle("collapsed", uiState.firesCollapsed);
    if (els.autoSection) els.autoSection.classList.toggle("collapsed", uiState.autoCollapsed);
    toggle(els.firesToggle, uiState.firesCollapsed);
    toggle(els.autoToggle, uiState.autoCollapsed);
  }

  function toggleSection(key) {
    uiState[key] = !uiState[key];
    try { localStorage.setItem(SECTION_UI_KEY, JSON.stringify(uiState)); } catch (e) { /* ignore */ }
    applySectionState();
  }

  // -----------------------------------------------------------------------
  // Toast notification
  // -----------------------------------------------------------------------
  function toast(message, type = "info") {
    const icons = { success: "✅", error: "❌", info: "ℹ️" };
    const div = document.createElement("div");
    div.className = `toast ${type}`;
    div.innerHTML = `<span class="toast-icon">${icons[type] || "ℹ️"}</span><span class="toast-msg">${message}</span>`;
    els.toastContainer.appendChild(div);
    setTimeout(() => {
      div.classList.add("out");
      setTimeout(() => div.remove(), 300);
    }, 4000);
  }

  // -----------------------------------------------------------------------
  // Admin API helper
  // -----------------------------------------------------------------------
  async function adminFetch(url, options = {}) {
    const res = await fetch(url, {
      ...options,
      headers: {
        ...(options.headers || {}),
        "X-Admin-Secret": STATE.secret,
      },
    });
    if (res.status === 401) {
      // Session expired or invalid — force re-login
      sessionStorage.removeItem("wildframe_admin_secret");
      STATE.secret = "";
      showLogin();
      throw new Error("Session expired. Please log in again.");
    }
    return res;
  }

  // -----------------------------------------------------------------------
  // Login / Logout
  // -----------------------------------------------------------------------
  function showLogin() {
    els.loginOverlay.classList.remove("hidden");
    els.dashboard.classList.add("hidden");
    els.password.value = "";
    els.loginError.classList.add("hidden");
  }

  function showDashboard() {
    els.loginOverlay.classList.add("hidden");
    els.dashboard.classList.remove("hidden");
  }

  async function attemptLogin() {
    // Fall back to the stored session secret when the password field is
    // empty (the auto-login path from init()) — otherwise a returning
    // moderator with a saved session could never get back in without
    // retyping the password.
    const secret = els.password.value.trim() || STATE.secret;
    if (!secret) return;

    // Quick validation against the server
    try {
      const res = await fetch("/api/admin/pending", {
        headers: { "X-Admin-Secret": secret },
      });
      if (!res.ok) {
        els.loginError.classList.remove("hidden");
        els.password.focus();
        return;
      }
      // Success!
      STATE.secret = secret;
      sessionStorage.setItem("wildframe_admin_secret", secret);
      els.loginError.classList.add("hidden");
      showDashboard();
      await loadPending();
    } catch (err) {
      els.loginError.textContent = "Connection error. Is the server running?";
      els.loginError.classList.remove("hidden");
    }
  }

  function logout() {
    sessionStorage.removeItem("wildframe_admin_secret");
    STATE.secret = "";
    showLogin();
    toast("🔒 Dashboard locked", "info");
  }

  // -----------------------------------------------------------------------
  // Load pending reports
  // -----------------------------------------------------------------------
  async function loadPending() {
    if (!STATE.secret) return;

    try {
      const res = await adminFetch("/api/admin/pending");
      const data = await res.json();
      STATE.reports = data.reports || [];

      els.pendingCount.textContent = `${STATE.reports.length} pending`;
      renderReports();
      loadAutoApproved();
      loadFires();
    } catch (err) {
      if (err.message !== "Session expired. Please log in again.") {
        console.error("Failed to load reports:", err);
        toast("Failed to load reports", "error");
      }
    }
  }

  // -----------------------------------------------------------------------
  // Recently auto-approved (human backstop)
  // -----------------------------------------------------------------------
  async function loadAutoApproved() {
    if (!STATE.secret) return;
    try {
      const res = await adminFetch("/api/admin/auto-approved");
      const data = await res.json();
      STATE.autoApproved = data.reports || [];
      renderAutoApproved();
    } catch (err) {
      if (err.message !== "Session expired. Please log in again.") {
        console.error("Failed to load auto-approved reports:", err);
      }
    }
  }

  function renderAutoApproved() {
    if (!els.autoSection || !els.autoList) return;
    const rows = STATE.autoApproved;
    els.autoList.innerHTML = "";

    if (rows.length === 0) {
      els.autoSection.classList.add("hidden");
      return;
    }

    els.autoSection.classList.remove("hidden");
    els.autoCount.textContent = `${rows.length} — reject any that look wrong`;

    rows.forEach((r) => {
      const src =
        r.approval_source === "satellite" ? "satellite (FIRMS)" :
        r.approval_source === "cluster" ? "nearby confirmed report" :
        "satellite + cluster";
      const clsTag =
        r.approval_class === "flame" ? "🔥" :
        r.approval_class === "smoke" ? "💨" : "🤖";
      const confTxt =
        typeof r.approval_confidence === "number" && isFinite(r.approval_confidence)
          ? ` ${Math.round(Math.min(Math.max(r.approval_confidence, 0), 1) * 100)}%`
          : "";
      const cancelled = r.status === "cancelled"
        ? `<span class="sat-badge sat-none" title="This report was cancelled (agency cancel / fire contained)">❌ Cancelled</span>`
        : "";
      const row = document.createElement("div");
      row.className = "auto-approved-row" + (cancelled ? " cancelled" : "");
      row.id = `auto-${r.id}`;
      row.innerHTML = `
        <div class="auto-thumb">
          ${r.photo_url
            ? `<img src="${r.photo_url}" alt="" loading="lazy" />`
            : `<span class="auto-no-photo">📸</span>`
          }
        </div>
        <div class="auto-meta">
          <div class="auto-title">🤖 Auto-approved via ${src} — ${clsTag}${confTxt}${cancelled}</div>
          <div class="auto-sub">${new Date(r.captured_at).toLocaleString()} · ${r.lat.toFixed(4)}, ${r.lon.toFixed(4)}</div>
        </div>
        <button class="card-btn reject-btn auto-reject-btn" data-id="${r.id}" title="Reject and delete this auto-approved report">
          <span>❌</span> Reject
        </button>
      `;
      els.autoList.appendChild(row);
    });

    els.autoList.querySelectorAll(".auto-reject-btn").forEach((btn) => {
      btn.addEventListener("click", () => rejectAutoApproved(btn.dataset.id));
    });
  }

  async function rejectAutoApproved(id) {
    if (STATE.processing.has(id)) return;
    STATE.processing.add(id);
    const row = document.getElementById(`auto-${id}`);
    if (row) row.style.opacity = "0.4";

    try {
      const res = await adminFetch(`/api/admin/reject/${id}`, { method: "POST" });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || "Failed to reject");
      }
      toast("🗑️ Auto-approved report rejected", "info");
      STATE.autoApproved = STATE.autoApproved.filter((r) => r.id !== id);
      if (row) row.remove();
      renderAutoApproved();
    } catch (err) {
      toast(err.message || "Failed to reject report", "error");
      if (row) row.style.opacity = "";
    } finally {
      STATE.processing.delete(id);
    }
  }

  // -----------------------------------------------------------------------
  // Live fires — active Bayesian grids with EFFIS fuel-moisture context
  // -----------------------------------------------------------------------
  // Display-only tier thresholds (the model's math lives server-side).
  const FWI_TIERS = {
    ffmc: [ // fine-fuel moisture code — lower = damper
      { max: 70, label: "Damp", cls: "tier-low" },
      { max: 80, label: "Moderate", cls: "tier-mid" },
      { max: 88, label: "Dry", cls: "tier-high" },
      { max: 1e9, label: "Very dry", cls: "tier-extreme" },
    ],
    dmc: [ // duff moisture code
      { max: 20, label: "Low", cls: "tier-low" },
      { max: 40, label: "Moderate", cls: "tier-mid" },
      { max: 60, label: "High", cls: "tier-high" },
      { max: 1e9, label: "Very high", cls: "tier-extreme" },
    ],
    isi: [ // initial spread index
      { max: 3, label: "Low", cls: "tier-low" },
      { max: 7, label: "Moderate", cls: "tier-mid" },
      { max: 15, label: "High", cls: "tier-high" },
      { max: 1e9, label: "Extreme", cls: "tier-extreme" },
    ],
  };

  function fwiTier(kind, value) {
    // Keys are lowercase; callers pass display names like "FFMC".
    const tiers = FWI_TIERS[String(kind).toLowerCase()];
    if (!tiers) return { label: "Unknown", cls: "tier-na" };
    for (const t of tiers) if (value <= t.max) return t;
    return tiers[tiers.length - 1];
  }

  function fwiBadge(kind, value) {
    const has = typeof value === "number" && isFinite(value) && value > 0;
    const t = has ? fwiTier(kind, value) : { label: "No data", cls: "tier-na" };
    const title = has
      ? `${kind.toUpperCase()} ${value.toFixed(1)} — ${t.label}`
      : `${kind.toUpperCase()} — outside EFFIS coverage or not fetched yet`;
    return `<span class="fwi-badge ${t.cls}" title="${title}">${kind.toUpperCase()} ${has ? Math.round(value) : "—"}</span>`;
  }

  function compassDir(deg) {
    const dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
    return dirs[Math.round(((deg % 360 + 360) % 360) / 22.5) % 16];
  }

  function evidenceAgeText(ageH) {
    if (ageH == null) return "no evidence yet";
    if (ageH < 1) return `${Math.round(ageH * 60)} min ago`;
    return `${ageH.toFixed(1)} h ago`;
  }

  // Skip re-rendering the fires list when nothing about the fires changed
  // (the 8s admin poll would otherwise rebuild 500 DOM rows every tick;
  // evidence age is deliberately excluded — it only grows, and a stale
  // "0.3 h ago" is harmless until the fire's data actually changes).
  let _firesSig = "";
  function firesSignature(fires) {
    let h = 0;
    for (const f of fires) {
      h = (h * 31 + f.id.length + (f.max_p * 1000 | 0) + (f.ffmc * 10 | 0)
           + (f.dmc * 10 | 0) + (f.isi * 10 | 0) + (f.wind_speed * 10 | 0)
           + (f.wind_dir_deg | 0)) | 0;
    }
    return `${fires.length}:${h}`;
  }

  async function loadFires() {
    if (!STATE.secret) return;
    try {
      const res = await adminFetch("/api/admin/grids");
      const data = await res.json();
      STATE.fires = data.grids || [];
      renderFires();
    } catch (err) {
      if (err.message !== "Session expired. Please log in again.") {
        console.error("Failed to load fires:", err);
      }
    }
  }

  function renderFires() {
    if (!els.firesSection || !els.firesList) return;
    const rows = STATE.fires;

    const sig = firesSignature(rows);
    if (sig === _firesSig && els.firesList.childElementCount > 0) return;
    _firesSig = sig;

    els.firesList.innerHTML = "";

    if (rows.length === 0) {
      els.firesSection.classList.add("hidden");
      return;
    }

    els.firesSection.classList.remove("hidden");
    els.firesCount.textContent = `${rows.length} active fires — EFFIS-data fires first`;

    rows.forEach((f) => {
      const windKmh = (f.wind_speed * 3.6).toFixed(1);
      const windTxt = `${windKmh} km/h → ${compassDir(f.wind_dir_deg)} (${f.wind_dir_deg.toFixed(0)}°)`;
      const stale = f.evidence_age_h != null && f.evidence_age_h > 24;
      const prob = `${(Math.min(Math.max(f.max_p, 0), 1) * 100).toFixed(1)}%`;

      // Deep-link to the live map focused on this fire. The row carries
      // everything the map popup needs (grid id, centroid, probability,
      // wind, fuel moisture) so the map shows real data immediately
      // without another round-trip.
      const hrefParams = new URLSearchParams({
        grid: f.id,
        lat: f.lat.toFixed(5),
        lon: f.lon.toFixed(5),
        max_p: f.max_p.toFixed(4),
        wind_speed: f.wind_speed.toFixed(1),
        wind_dir_deg: f.wind_dir_deg.toFixed(0),
      });
      if (f.ffmc > 0) hrefParams.set("ffmc", f.ffmc.toFixed(1));
      if (f.dmc > 0) hrefParams.set("dmc", f.dmc.toFixed(1));
      if (f.isi > 0) hrefParams.set("isi", f.isi.toFixed(1));
      if (f.moisture_factor > 0) hrefParams.set("mf", f.moisture_factor.toFixed(2));

      const row = document.createElement("a");
      row.className = "fire-row" + (stale ? " stale" : "");
      row.href = `/?${hrefParams.toString()}`;
      row.target = "_blank";
      row.rel = "noopener";
      row.title = "Show this fire on the live map";
      row.innerHTML = `
        <div class="fire-main">
          <span class="fire-id" title="${f.lat.toFixed(4)}, ${f.lon.toFixed(4)}">${f.id}</span>
          <span class="fire-prob" title="Peak grid probability">${prob}</span>
        </div>
        <div class="fire-sub">
          <span>${f.lat.toFixed(3)}, ${f.lon.toFixed(3)}</span>
          <span class="fire-wind" title="Wind the spread model steers by">${windTxt}</span>
          <span class="fire-age">evidence: ${evidenceAgeText(f.evidence_age_h)}</span>
        </div>
        <div class="fire-moist">
          ${fwiBadge("FFMC", f.ffmc)}
          ${fwiBadge("DMC", f.dmc)}
          ${fwiBadge("ISI", f.isi)}
          <span class="mf-badge" title="Spread-rate multiplier the model applies from FFMC (1.00 = wind-only behaviour)">×${f.moisture_factor.toFixed(2)}</span>
        </div>
        <span class="fire-map-chip" aria-hidden="true">🗺️ Show on map</span>
      `;
      els.firesList.appendChild(row);
    });
  }

  // -----------------------------------------------------------------------
  // Render report cards
  // -----------------------------------------------------------------------
  function renderReports() {
    // Clear existing cards (keep empty state as template)
    $$(".report-card").forEach((el) => el.remove());
    els.emptyState.classList.add("hidden");

    // Show/hide Accept All button
    if (STATE.reports.length >= 2) {
      els.acceptAllBtn.classList.remove("hidden");
    } else {
      els.acceptAllBtn.classList.add("hidden");
    }

    if (STATE.reports.length === 0) {
      els.emptyState.classList.remove("hidden");
      return;
    }

    STATE.reports.forEach((r) => {
      const card = document.createElement("div");
      card.className = "report-card";
      card.id = `report-${r.id}`;

      const ts = new Date(r.captured_at).toLocaleString();
      const headingInfo = r.device_heading ? `${r.device_heading}°` : "N/A";

      const satInfo = r.satellite_confirmation;
      let satBadge;
      if (satInfo && satInfo.confirmed) {
        satBadge = `<span class="sat-badge sat-confirmed" title="Nearest FIRMS hotspot ${satInfo.nearest_km}km away">🛰️ Satellite confirmed (${satInfo.hotspot_count})</span>`;
      } else if (satInfo) {
        satBadge = `<span class="sat-badge sat-none" title="No FIRMS hotspot within range at check time">🛰️ No satellite match</span>`;
      } else {
        satBadge = `<span class="sat-badge sat-unchecked">🛰️ Not checked</span>`;
      }

      card.innerHTML = `
        <div class="card-photo">
          ${r.photo_url
            ? `<img src="${r.photo_url}" alt="Report photo" loading="lazy" />`
            : `<div class="card-no-photo"><span>📸</span><p>No photo</p></div>`
          }
          ${aiBadge(r.ai_analysis)}
        </div>
        <div class="card-info">
          <div class="card-info-row">
            <span class="card-label">Location</span>
            <span class="card-value">${r.lat.toFixed(5)}, ${r.lon.toFixed(5)}</span>
          </div>
          <div class="card-info-row">
            <span class="card-label">Heading</span>
            <span class="card-value">${headingInfo}</span>
          </div>
          <div class="card-info-row">
            <span class="card-label">Reported</span>
            <span class="card-value">${ts}</span>
          </div>
          <div class="card-info-row">
            <span class="card-label">Source</span>
            <span class="card-value">${r.source_type || "citizen"}</span>
          </div>
          <div class="card-info-row">
            <span class="card-label">Satellite</span>
            <span class="card-value">${satBadge}</span>
          </div>
        </div>
        <div class="card-actions">
          <button class="card-btn accept-btn" data-id="${r.id}" title="Confirm this report">
            <span>✅</span> Accept
          </button>
          <button class="card-btn reject-btn" data-id="${r.id}" title="Reject and delete this report">
            <span>❌</span> Reject
          </button>
          <button class="card-btn sat-check-btn" data-id="${r.id}" title="Check NASA FIRMS for a corroborating hotspot near this report">
            <span>🛰️</span> Check satellite
          </button>
        </div>
        <div class="card-processing hidden" id="processing-${r.id}">
          <span class="processing-spinner"></span>
          <span>Processing…</span>
        </div>
      `;

      // Insert before empty state or append
      els.container.appendChild(card);
    });

    // Attach event listeners
    $$(".accept-btn").forEach((btn) => {
      btn.addEventListener("click", () => acceptReport(btn.dataset.id));
    });
    $$(".reject-btn").forEach((btn) => {
      btn.addEventListener("click", () => rejectReport(btn.dataset.id));
    });
    $$(".sat-check-btn").forEach((btn) => {
      btn.addEventListener("click", () => checkSatellite(btn.dataset.id));
    });
  }

  // -----------------------------------------------------------------------
  // AI certainty badge
  // -----------------------------------------------------------------------
  function aiBadge(analysis) {
    if (!analysis || typeof analysis !== "object" || !("verdict" in analysis)) {
      return '<span class="ai-badge ai-unknown" title="No AI analysis for this report">🤖 No AI scan</span>';
    }

    const verdict = analysis.verdict || "error";
    const conf = analysis.confidence;
    const confPct =
      typeof conf === "number" && isFinite(conf)
        ? `${Math.round(Math.min(Math.max(conf, 0), 1) * 100)}%`
        : "—";

    const labels = {
      flame: ["🔥 Fire", "ai-fire"],
      smoke: ["💨 Smoke", "ai-smoke"],
      both: ["🔥 Fire + Smoke", "ai-fire"],
      // "nothing" reports are KEPT for human review (the hosted model can
      // miss borderline fires) — neutral styling, not a green all-clear.
      nothing: ["🤔 None — review", "ai-review"],
      error: ["⚠️ Scan error", "ai-error"],
      unknown: ["❔ Unknown", "ai-unknown"],
    };
    const [label, cls] = labels[verdict] || labels.unknown;

    const title =
      verdict === "error"
        ? "AI scan failed (no verdict)"
        : verdict === "nothing"
          ? "AI found nothing — kept for human review"
          : `AI verdict: ${verdict} — certainty ${confPct}`;

    return `<span class="ai-badge ${cls}" title="${title}">${label} ${confPct}</span>`;
  }

  // -----------------------------------------------------------------------
  // Satellite confirmation (on-demand NASA FIRMS check)
  // -----------------------------------------------------------------------
  async function checkSatellite(id) {
    const btn = document.querySelector(`.sat-check-btn[data-id="${id}"]`);
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = '<span class="processing-spinner" style="width:12px;height:12px;border-width:2px"></span> Checking…';
    }
    try {
      const res = await adminFetch(`/api/reports/${id}/check-satellite`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) {
        toast(data.error || "Satellite check failed", "error");
        return;
      }
      const report = STATE.reports.find((r) => r.id === id);
      if (report) report.satellite_confirmation = data.satellite_confirmation;
      renderReports();
      toast(
        data.satellite_confirmation.confirmed
          ? `🛰️ Confirmed — hotspot ${data.satellite_confirmation.nearest_km}km away`
          : "🛰️ No FIRMS hotspot found nearby",
        data.satellite_confirmation.confirmed ? "success" : "info"
      );
    } catch (err) {
      toast("Satellite check request failed", "error");
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<span>🛰️</span> Check satellite';
      }
    }
  }

  // -----------------------------------------------------------------------
  // Accept All
  // -----------------------------------------------------------------------
  async function acceptAll() {
    if (els.acceptAllBtn.disabled) return;
    const count = STATE.reports.length;
    if (count === 0) return;

    // Confirm
    if (!confirm(`Accept all ${count} pending report(s)? This cannot be undone.`)) return;

    els.acceptAllBtn.disabled = true;
    els.acceptAllBtn.innerHTML = '<span class="processing-spinner" style="width:14px;height:14px;border-width:2px"></span> Accepting…';

    try {
      const res = await adminFetch("/api/admin/accept-all", { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to accept all");

      toast(`✅ ${data.message}`, "success");
      await loadPending();
    } catch (err) {
      toast(err.message || "Failed to accept all reports", "error");
    } finally {
      els.acceptAllBtn.disabled = false;
      els.acceptAllBtn.innerHTML = '<span>✅</span> Accept All';
    }
  }

  // -----------------------------------------------------------------------
  // Accept / Reject
  // -----------------------------------------------------------------------
  async function acceptReport(id) {
    if (STATE.processing.has(id)) return;
    STATE.processing.add(id);
    setProcessing(id, true);

    try {
      const res = await adminFetch(`/api/admin/accept/${id}`, { method: "POST" });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || "Failed to accept");
      }
      toast("✅ Report confirmed", "success");
      removeCard(id);
      await loadPending(); // refresh full list
    } catch (err) {
      toast(err.message || "Failed to accept report", "error");
      setProcessing(id, false);
    } finally {
      STATE.processing.delete(id);
    }
  }

  async function rejectReport(id) {
    if (STATE.processing.has(id)) return;
    STATE.processing.add(id);
    setProcessing(id, true);

    try {
      const res = await adminFetch(`/api/admin/reject/${id}`, { method: "POST" });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || "Failed to reject");
      }
      toast("🗑️ Report deleted", "info");
      removeCard(id);
      await loadPending(); // refresh full list
    } catch (err) {
      toast(err.message || "Failed to reject report", "error");
      setProcessing(id, false);
    } finally {
      STATE.processing.delete(id);
    }
  }

  function setProcessing(id, active) {
    const card = document.getElementById(`report-${id}`);
    if (!card) return;
    const actions = card.querySelector(".card-actions");
    const processing = card.querySelector(".card-processing");
    if (actions) actions.classList.toggle("hidden", active);
    if (processing) processing.classList.toggle("hidden", !active);
  }

  function removeCard(id) {
    const card = document.getElementById(`report-${id}`);
    if (card) {
      card.style.transform = "scale(0.95)";
      card.style.opacity = "0";
      setTimeout(() => card.remove(), 300);
    }
  }

  // -----------------------------------------------------------------------
  // Auto-refresh polling
  // -----------------------------------------------------------------------
  let pollInterval = null;

  function startPolling(ms = 8000) {
    stopPolling();
    pollInterval = setInterval(() => {
      if (STATE.secret && els.dashboard.classList.contains("hidden") === false) {
        loadPending();
        loadAutoApproved();
      }
    }, ms);
  }

  function stopPolling() {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  }

  // -----------------------------------------------------------------------
  // Init
  // -----------------------------------------------------------------------
  function init() {
    // If we have a stored secret, try to use it
    if (STATE.secret) {
      // Quick check
      attemptLogin();
    }

    // Login button
    els.loginBtn.addEventListener("click", attemptLogin);
    els.password.addEventListener("keydown", (e) => {
      if (e.key === "Enter") attemptLogin();
    });

    // Refresh button
    els.refreshBtn.addEventListener("click", () => {
      els.refreshBtn.classList.add("spinning");
      loadPending().finally(() => {
        setTimeout(() => els.refreshBtn.classList.remove("spinning"), 400);
      });
    });

    // Accept All button
    els.acceptAllBtn.addEventListener("click", acceptAll);

    // Logout button
    els.logoutBtn.addEventListener("click", logout);

    // Section collapse toggles
    if (els.firesToggle) els.firesToggle.addEventListener("click", () => toggleSection("firesCollapsed"));
    if (els.autoToggle) els.autoToggle.addEventListener("click", () => toggleSection("autoCollapsed"));
    applySectionState();

    // Start polling
    startPolling();
  }

  // -----------------------------------------------------------------------
  // Boot
  // -----------------------------------------------------------------------
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();