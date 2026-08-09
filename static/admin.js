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
    toastContainer: $("#admin-toast-container"),
  };

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
    const secret = els.password.value.trim();
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