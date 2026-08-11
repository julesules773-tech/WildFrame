/* ===================================================================
   WildFrame — Wildfire Detection | Frontend App
   =================================================================== */

(function () {
  "use strict";

  // -----------------------------------------------------------------------
  // State
  // -----------------------------------------------------------------------
  const STATE = {
    map: null,
    mode: "production",         // "production" (live) | "demo"
    currentPos: null,           // { lat, lon }
    heading: null,              // from DeviceOrientation API
    sessionId: "",
    reports: [],
    clusters: [],
    markers: { reports: [], clusters: [], triangulation: [], fireOrigin: [] },
    uploading: false,
    liveDemo: {
      active: false,
      stepInterval: null,
      currentStep: -1,
      totalSteps: 0,
      fireLat: null,
      fireLon: null,
      trueOriginMarker: null,
    },
    bayesian: {
      active: false,
      panelOpen: true,   // control card visible (boot collapses it on mobile)
      heatmapLayer: null,
      cells: [],
      contour: [],
      stats: null,
      threshold: 0.0000000001,
      showHeatmap: true,
      showContour: true,
      pollingInterval: null,
      windLabels: [],   // per-grid wind label markers added to map
      metaDots: [],     // low-zoom (detail=meta) intensity dots
      metaLayer: null,  // Leaflet layerGroup holding those dots
      metaSig: "",      // signature of last-rendered dots (skip DOM rebuild when unchanged)
      satellitePollerActive: false,
      firmsPollerActive: false,
      usersOnly: false,       // hide satellite/Bayesian layers, show user reports only
      roadRiskActive: false,   // road risk overlay toggle state
      roadRiskLayer: null,     // Leaflet GeoJSON layer for road risk
    },
    historicDemo: {
      active: false,
      stepInterval: null,
      currentStep: -1,
      totalSteps: 0,
      fireLat: null,
      fireLon: null,
      hotspotMarkers: [],
      perimeterLayer: null,
      originMarker: null,
      description: "",
      windSpeed: 0,
      windDir: 0,
      hotspotCount: 0,
    },
  };

  // -----------------------------------------------------------------------
  // DOM refs
  // -----------------------------------------------------------------------
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  const els = {
    statusText: $("#status-text"),
    statusDot: $("#status-badge .dot"),
    statReports: $("#stat-reports"),
    statClusters: $("#stat-clusters"),
    statPending: $("#stat-pending"),

    uploadTrigger: $("#upload-trigger"),
    uploadCard: $("#upload-card"),
    cardClose: $("#card-close"),

    photoPreview: $("#photo-preview"),
    previewPlaceholder: $("#preview-placeholder"),
    previewImage: $("#preview-image"),
    photoInput: $("#photo-input"),

    form: $("#upload-form"),
    inputLat: $("#input-lat"),
    inputLon: $("#input-lon"),
    inputHeading: $("#input-heading"),
    inputSession: $("#input-session"),
    inputCapturedAt: $("#input-captured-at"),
    submitBtn: $("#submit-btn"),
    progressBar: $("#upload-progress"),

    locText: $("#loc-text"),
    manualCoordFields: $("#manual-coords"),
    inputLatManual: $("#input-lat-manual"),
    inputLonManual: $("#input-lon-manual"),

    gpsRefreshBtn: $("#gps-refresh-btn"),
    topMenu: $("#top-menu"),
    topMenuBtn: $("#top-menu-btn"),
    toastContainer: $("#toast-container"),
  };

  // -----------------------------------------------------------------------
  // Session
  // -----------------------------------------------------------------------
  function getSessionId() {
    let id = localStorage.getItem("wildframe_session");
    if (!id) {
      id = crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2) + Date.now().toString(36);
      localStorage.setItem("wildframe_session", id);
    }
    return id;
  }

  // -----------------------------------------------------------------------
  // Data mode helpers
  // -----------------------------------------------------------------------

  /**
   * Return the query-string fragment that selects which store the backend
   * should read from. Empty string = production (the safe default).
   */
  function modeQuery() {
    return STATE.mode === "demo" ? "?mode=demo" : "";
  }

  /**
   * Append a mode field to a JSON request body.
   */
  function modeBody(body = {}) {
    return { ...body, mode: STATE.mode };
  }

  /**
   * Switch the displayed data mode. In "demo" mode the map reads/writes the
   * demo store only; in "production" mode it reads/writes live data only.
   * Re-fetches everything and re-renders the map.
   */
  async function setMode(mode) {
    if (mode !== "production" && mode !== "demo") mode = "production";
    const changed = STATE.mode !== mode;

    // Flip the mode first so any teardown below refreshes with the NEW mode.
    STATE.mode = mode;
    const isDemo = mode === "demo";

    // Leaving demo mode: tear down any running demos so their simulated
    // data isn't left on screen under a "Live" label.
    if (changed && !isDemo) {
      if (STATE.liveDemo.active) await cancelLiveDemo();
      if (STATE.historicDemo.active) await cancelHistoricDemo();
    }

    // Update mode switch buttons
    const liveBtn = document.getElementById("mode-live-btn");
    const demoBtn = document.getElementById("mode-demo-btn");
    if (liveBtn) liveBtn.classList.toggle("active", !isDemo);
    if (demoBtn) demoBtn.classList.toggle("active", isDemo);

    // Show/hide the DEMO banner
    const banner = document.getElementById("demo-banner");
    if (banner) banner.classList.toggle("hidden", !isDemo);

    if (!changed) return;

    // Re-fetch everything for the new mode
    await refreshData();
    if (STATE.bayesian.active) await fetchBayesianState();
  }

  // -----------------------------------------------------------------------
  // Toast notifications
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
  // Status bar
  // -----------------------------------------------------------------------
  function setStatus(text, type = "pending") {
    els.statusText.textContent = text;
    els.statusDot.className = "dot " + type;
  }

  // -----------------------------------------------------------------------
  // Geolocation
  // -----------------------------------------------------------------------

  /**
   * Toggle the manual coordinate input fields based on GPS availability.
   * When GPS fails, show the manual fields so users can type coordinates.
   * When GPS succeeds, hide them (GPS is working).
   */
  function _setManualCoordsVisible(visible) {
    if (!els.manualCoordFields) return;
    els.manualCoordFields.classList.toggle("hidden", !visible);
  }

  /**
   * Sync manual coordinate input values into the hidden form fields
   * so they're included when the form is submitted.
   */
  function _syncManualCoordsToHidden() {
    const lat = parseFloat(els.inputLatManual?.value);
    const lon = parseFloat(els.inputLonManual?.value);
    if (!isNaN(lat) && !isNaN(lon)) {
      els.inputLat.value = lat;
      els.inputLon.value = lon;
    }
  }

  /**
   * Convert a GeolocationPositionError into a human-readable string with
   * a clear explanation of WHY it failed and what the user can do about it.
   */
  function _gpsErrorString(err) {
    const code = err && err.code;
    const browserMsg = (err && err.message) || "";

    if (code === 1) {
      // PERMISSION_DENIED
      // Check if the page is served over HTTPS (or localhost), because
      // modern browsers REQUIRE a secure context for geolocation.
      const isSecure = window.location.protocol === "https:" ||
                       window.location.hostname === "localhost" ||
                       window.location.hostname === "127.0.0.1";
      if (!isSecure) {
        return "GPS blocked: browser requires HTTPS for location. " +
               "Access via https:// or localhost, or enter coordinates manually.";
      }
      return "GPS permission denied. Enable location services in your " +
             "browser/device settings, or enter coordinates manually.";
    }

    if (code === 2) {
      // POSITION_UNAVAILABLE
      // err.message is typically "Position unavailable" — the browser
      // doesn't reveal the root cause, so we don't guess.
      if (browserMsg.includes("network") || browserMsg.includes("Network")) {
        return "GPS unavailable: network location failed. Try Wi-Fi or move outdoors.";
      }
      return "GPS unavailable: couldn't determine your location. " +
             "Try moving outdoors or enter coordinates manually.";
    }

    if (code === 3) {
      // TIMEOUT
      return "GPS timed out (15s). Move to a clearer area and try again, " +
             "or enter coordinates manually.";
    }

    // Unknown error code — fall back to the browser's own message
    return browserMsg || "GPS location failed for an unknown reason.";
  }

  /**
   * Attempt to acquire the device's GPS position. Updates the UI with a
   * human-readable status message on success or failure.
   */
  function acquirePosition() {
    if (!navigator.geolocation) {
      els.locText.textContent = "❌ GPS not available";
      setStatus("GPS unavailable", "error");
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        STATE.currentPos = { lat: pos.coords.latitude, lon: pos.coords.longitude };
        STATE.heading = pos.coords.heading; // may be null
        els.locText.textContent = `📍 ${pos.coords.latitude.toFixed(4)}, ${pos.coords.longitude.toFixed(4)}`;
        els.inputLat.value = pos.coords.latitude;
        els.inputLon.value = pos.coords.longitude;
        els.inputHeading.value = pos.coords.heading ?? "";
        setStatus("GPS locked", "active");
        _setManualCoordsVisible(false);

        // Center map on position
        if (STATE.map) {
          STATE.map.setView([pos.coords.latitude, pos.coords.longitude], 14);
        }
      },
      (err) => {
        const msg = _gpsErrorString(err);
        console.warn("Geolocation error:", err.code, err.message);
        els.locText.textContent = `⚠️ ${msg}`;
        setStatus("GPS failed", "error");
        _setManualCoordsVisible(true);
        // Default to Yosemite National Park (forest demo location)
        if (STATE.map) {
          STATE.map.setView([37.745, -119.593], 8);
        }
      },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 60000 }
    );
  }

  // -----------------------------------------------------------------------
  // DeviceOrientation for heading
  // -----------------------------------------------------------------------
  function watchHeading() {
    if (!window.DeviceOrientationEvent) return;
    window.addEventListener("deviceorientation", (e) => {
      // alpha = compass heading (0-360) if available
      if (e.alpha !== null) {
        STATE.heading = e.alpha;
        if (els.inputHeading) els.inputHeading.value = e.alpha;
      }
    }, { passive: true });
  }

  // -----------------------------------------------------------------------
  // Map
  // -----------------------------------------------------------------------
  function initMap() {
    // Default to Yosemite National Park — the forest demo location
    const defaultCenter = STATE.currentPos
      ? [STATE.currentPos.lat, STATE.currentPos.lon]
      : [37.745, -119.593];

    STATE.map = L.map("map", {
      center: defaultCenter,
      zoom: STATE.currentPos ? 14 : 8,
      zoomControl: true,
      attributionControl: true,
    });

    // CartoDB dark tiles for mood
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
      subdomains: "abcd",
      maxZoom: 19,
    }).addTo(STATE.map);

    // Add a locate button
    L.control.locate({
      position: "topleft",
      strings: { title: "Show my location" },
      locateOptions: { enableHighAccuracy: true },
    }).addTo(STATE.map);

    // Re-render clusters when zoom changes (convex hull vs dot)
    STATE.map.on("zoomend", () => {
      renderData(STATE.reports, STATE.clusters);
    });
  }

  // -----------------------------------------------------------------------
  // Convex Hull (Monotone Chain / Andrew's algorithm)
  // -----------------------------------------------------------------------
  function convexHull(points) {
    // points: [[lat, lon], ...]
    // Returns [[lat, lon], ...] in CCW order, or empty array if < 3 points
    if (points.length < 3) return [];

    // Sort by lon, then lat
    const sorted = [...points].sort((a, b) => a[1] - b[1] || a[0] - b[0]);

    const cross = (o, a, b) =>
      (a[1] - o[1]) * (b[0] - o[0]) - (a[0] - o[0]) * (b[1] - o[1]);

    // Build lower hull
    const lower = [];
    for (const p of sorted) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
        lower.pop();
      lower.push(p);
    }

    // Build upper hull
    const upper = [];
    for (const p of sorted.reverse()) {
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
        upper.pop();
      upper.push(p);
    }

    // Remove last point of each list (duplicate of first in the other)
    lower.pop();
    upper.pop();

    return [...lower, ...upper];
  }

  // -----------------------------------------------------------------------
  // Cluster rendering helpers
  // -----------------------------------------------------------------------
  const CLUSTER_POLYGON_ZOOM = 13; // show polygons at this zoom and above

  function clusterLevel(count) {
    if (count >= 4) return 4;
    if (count >= 3) return 3;
    if (count >= 2) return 2;
    return 1;
  }

  function clusterColor(count) {
    const colors = ["#eab308", "#f97316", "#ef4444", "#dc2626"];
    return colors[Math.min(clusterLevel(count), 4) - 1];
  }

  function createClusterIcon(count) {
    const level = clusterLevel(count);
    return L.divIcon({
      className: "cluster-marker",
      html: `<div class="cluster-marker level-${level}">${count < 10 ? count : "9+"}</div>`,
      iconSize: [52, 52],
      iconAnchor: [26, 26],
      popupAnchor: [0, -28],
    });
  }

  // -----------------------------------------------------------------------
  // Triangulation rendering helpers
  // -----------------------------------------------------------------------

  /**
   * Generate vertices of an uncertainty ellipse as [lat, lon] pairs.
   * The ellipse is defined in a local tangent plane at its centre.
   */
  function ellipsePolygon(cLat, cLon, semiMajor, semiMinor, angleDeg, n = 36) {
    if (semiMajor <= 0 || semiMinor <= 0) return [];
    const pts = [];
    const α = (angleDeg || 0) * Math.PI / 180;
    const latRad = cLat * Math.PI / 180;
    const cosLat = Math.cos(latRad);
    const DEG_PER_M_LAT = 1 / 111320;
    const DEG_PER_M_LON = 1 / (111320 * (cosLat || 0.0001));

    for (let i = 0; i < n; i++) {
      const t = (2 * Math.PI * i) / n;
      // Rotated ellipse offset in local (east=x, north=y)
      const ex = semiMajor * Math.cos(t) * Math.cos(α) - semiMinor * Math.sin(t) * Math.sin(α);
      const ey = semiMajor * Math.cos(t) * Math.sin(α) + semiMinor * Math.sin(t) * Math.cos(α);
      pts.push([cLat + ey * DEG_PER_M_LAT, cLon + ex * DEG_PER_M_LON]);
    }
    return pts;
  }

  /**
   * Compute the endpoint of a bearing ray: starting from (lat, lon),
   * extending `distance` metres at compass bearing `headingDeg`.
   */
  function bearingEndpoint(lat, lon, headingDeg, distance) {
    const θ = headingDeg * Math.PI / 180;
    const dx = distance * Math.sin(θ);  // east component
    const dy = distance * Math.cos(θ);  // north component
    const latRad = lat * Math.PI / 180;
    const cosLat = Math.cos(latRad);
    const degPerMLat = 1 / 111320;
    const degPerMLon = 1 / (111320 * (cosLat || 0.0001));
    return [lat + dy * degPerMLat, lon + dx * degPerMLon];
  }

  /**
   * Render the triangulation overlay for a single cluster:
   *   - Bearing rays from each reporter
   *   - Fire origin crosshair
   *   - Uncertainty ellipse
   */
  function renderTriangulation(c, color) {
    const t = c.triangulation;
    if (!t) return;

    const layers = [];
    const fireLat = t.fire_lat;
    const fireLon = t.fire_lon;

    // --- Bearing rays: from each report toward the fire ---
    if (c.report_ids && c.points) {
      // Estimate cluster spread to choose ray length
      const lats = c.points.map(p => p[0]);
      const lons = c.points.map(p => p[1]);
      const latSpread = Math.max(...lats) - Math.min(...lats);
      const lonSpread = Math.max(...lons) - Math.min(...lons);
      const clusterSpan = Math.max(latSpread, lonSpread) * 111320;  // approx metres

      // We don't have the headings per report here from the frontend reports,
      // so we'll draw rays from the cluster API data when available.
      // For now, draw rays from centroid toward the fire origin.
      const rayLen = Math.max(clusterSpan * 2, 500);

      c.points.forEach((p) => {
        // Draw a ray from each report, through the fire origin,
        // extending a bit past it — this visualises the bearing line.
        const end = bearingEndpoint(p[0], p[1],
          // Convert bearing from report toward fire origin to compass
          // atan2(dlat, dlon) gives math angle CCW from east.
          // Compass bearing = (90 - math_deg + 360) % 360
          (() => {
            const dlat = fireLat - p[0];
            const dlon = fireLon - p[1];
            const mathDeg = Math.atan2(dlat, dlon) * 180 / Math.PI;
            return ((90 - mathDeg) + 360) % 360;
          })(),
          rayLen
        );
        const line = L.polyline([p, end], {
          color: color,
          weight: 1,
          opacity: 0.3,
          dashArray: "4, 6",
        }).addTo(STATE.map);
        layers.push(line);
      });
    }

    // --- Fire origin: crosshair marker ---
    const originIcon = L.divIcon({
      className: "fire-origin-marker",
      html: `<div class="fire-origin">
        <svg width="28" height="28" viewBox="0 0 28 28">
          <circle cx="14" cy="14" r="6" fill="none" stroke="${color}" stroke-width="2.5" opacity="0.9"/>
          <line x1="14" y1="2" x2="14" y2="26" stroke="${color}" stroke-width="2" opacity="0.7"/>
          <line x1="2" y1="14" x2="26" y2="14" stroke="${color}" stroke-width="2" opacity="0.7"/>
          <circle cx="14" cy="14" r="2" fill="${color}" opacity="0.9"/>
        </svg>
      </div>`,
      iconSize: [28, 28],
      iconAnchor: [14, 14],
    });

    const origin = L.marker([fireLat, fireLon], { icon: originIcon }).addTo(STATE.map);

    // Popup for the fire origin
    const confColor = t.confidence === "high" ? "#22c55e" : t.confidence === "medium" ? "#eab308" : "#ef4444";
    origin.bindPopup(`
      <div class="popup-title">🔥 Triangulated Fire Origin</div>
      <div><span class="popup-label">Location:</span> ${fireLat.toFixed(5)}, ${fireLon.toFixed(5)}</div>
      <div><span class="popup-label">From:</span> ${t.num_reports} bearing reports</div>
      <div><span class="popup-label">Confidence:</span> <span style="color:${confColor};font-weight:600">${t.confidence.toUpperCase()}</span></div>
      <div><span class="popup-label">Uncertainty:</span> ${t.ellipse_semi_major.toFixed(0)} × ${t.ellipse_semi_minor.toFixed(0)} m</div>
      <div style="margin-top:4px;font-size:11px;color:var(--text-muted)">${t.message}</div>
    `, { maxWidth: 300 });

    layers.push(origin);

    // --- Uncertainty ellipse ---
    if (t.ellipse_semi_major > 0 && t.ellipse_semi_minor > 0) {
      const ellipsePts = ellipsePolygon(
        fireLat, fireLon,
        t.ellipse_semi_major, t.ellipse_semi_minor,
        t.ellipse_angle_deg
      );
      if (ellipsePts.length >= 3) {
        const ellipse = L.polygon(ellipsePts, {
          color: color,
          weight: 1.5,
          opacity: 0.6,
          fillColor: color,
          fillOpacity: 0.1,
        }).addTo(STATE.map);
        layers.push(ellipse);
      }
    }

    STATE.markers.triangulation.push(...layers);
  }

  // -----------------------------------------------------------------------
  // Live Demo True Origin Marker
  // -----------------------------------------------------------------------
  function renderTrueOrigin(lat, lon) {
    // Clear previous
    STATE.markers.fireOrigin.forEach((m) => STATE.map.removeLayer(m));
    STATE.markers.fireOrigin = [];

    if (lat == null || lon == null) return;

    const icon = L.divIcon({
      className: "true-origin-marker",
      html: `<div class="true-origin">
        <svg width="32" height="32" viewBox="0 0 32 32">
          <circle cx="16" cy="16" r="12" fill="none" stroke="#22d3ee" stroke-width="2" opacity="0.6" stroke-dasharray="4,3"/>
          <circle cx="16" cy="16" r="3" fill="#22d3ee" opacity="0.9"/>
          <line x1="16" y1="0" x2="16" y2="32" stroke="#22d3ee" stroke-width="1.5" opacity="0.3" stroke-dasharray="2,4"/>
          <line x1="0" y1="16" x2="32" y2="16" stroke="#22d3ee" stroke-width="1.5" opacity="0.3" stroke-dasharray="2,4"/>
        </svg>
        <div class="true-origin-label">TRUE FIRE</div>
      </div>`,
      iconSize: [32, 44],
      iconAnchor: [16, 16],
    });

    const marker = L.marker([lat, lon], { icon }).addTo(STATE.map);
    marker.bindPopup(`
      <div class="popup-title">🎯 True Fire Origin (Demo)</div>
      <div><span class="popup-label">Location:</span> ${lat.toFixed(5)}, ${lon.toFixed(5)}</div>
      <div style="margin-top:4px;font-size:11px;color:var(--text-muted)">This is the simulated ground truth that triangulation tries to estimate.</div>
    `);
    STATE.markers.fireOrigin.push(marker);
  }


  // -----------------------------------------------------------------------
  // Render reports & clusters
  // -----------------------------------------------------------------------
  function renderData(reports, clusters) {
    // Clear existing markers
    STATE.markers.reports.forEach((m) => STATE.map.removeLayer(m));
    STATE.markers.clusters.forEach((m) => STATE.map.removeLayer(m));
    STATE.markers.triangulation.forEach((m) => STATE.map.removeLayer(m));
    STATE.markers.fireOrigin.forEach((m) => STATE.map.removeLayer(m));
    STATE.markers.reports = [];
    STATE.markers.clusters = [];
    STATE.markers.triangulation = [];
    STATE.markers.fireOrigin = [];

    // Build report_id → cluster color lookup for confirmed reports
    const clusterColorMap = {};
    clusters.forEach((c) => {
      const color = clusterColor(c.count);
      (c.report_ids || []).forEach((rid) => {
        clusterColorMap[rid] = color;
      });
    });

    // --- Individual report markers (ALL reports, including confirmed) ---
    // Confirmed reports show as small dots inside their cluster.
    // Pending/rejected reports show as larger dots.
    reports.forEach((r) => {
      let color, radius, opacity, fillOpacity, weight, isPending = false;

      if (r.status === "confirmed") {
        // Show as a small dot in the cluster color
        color = clusterColorMap[r.id] || "#eab308";
        radius = 4;
        opacity = 0.85;
        fillOpacity = 0.7;
        weight = 1;
      } else if (r.status === "cancelled") {
        // Agency-cancelled (fire contained / alert retracted) — grey, faded
        color = "#94a3b8";
        radius = 5;
        opacity = 0.35;
        fillOpacity = 0.3;
        weight = 1;
      } else if (r.status === "rejected") {
        color = "#64748b";
        radius = 5;
        opacity = 0.6;
        fillOpacity = 0.5;
        weight = 1.5;
      } else {
        // Pending
        color = "#eab308";
        radius = 5;
        opacity = 0.9;
        fillOpacity = 0.8;
        weight = 1.5;
        isPending = true;
      }

      const marker = L.circleMarker([r.lat, r.lon], {
        radius: radius,
        fillColor: color,
        color: "#fff",
        weight: weight,
        opacity: opacity,
        fillOpacity: fillOpacity,
      }).addTo(STATE.map);

      // Popup
      const headingInfo = r.device_heading ? `Heading: ${r.device_heading}°` : "Heading: N/A";

      // Build AI analysis badge
      const ai = r.ai_analysis;
      let aiBadge = "";
      if (ai && ai.verdict && ai.verdict !== "error") {
        // "nothing" verdicts are kept for human review, NOT verified-clear
        // — give them a neutral look so moderators can triage at a glance.
        const aiEmoji = ai.verdict === "flame" || ai.verdict === "both"
          ? "🔥" : ai.verdict === "smoke" ? "💨" : "🤔";
        const aiLabel = ai.verdict === "both" ? "FIRE + SMOKE"
          : ai.verdict === "nothing" ? "NONE — kept for review"
          : ai.verdict.toUpperCase();
        const aiColor = ai.verdict === "flame" || ai.verdict === "both"
          ? "#ef4444" : ai.verdict === "smoke" ? "#a78bfa" : "#94a3b8";
        const aiConf = (ai.confidence * 100).toFixed(0);
        aiBadge = `
          <div style="margin-top:6px;padding:6px 8px;background:rgba(0,0,0,0.3);border-radius:6px;
                      border-left:3px solid ${aiColor};text-align:center">
            <div style="font-size:12px;font-weight:700;color:${aiColor}">${aiEmoji} AI: ${aiLabel}</div>
            <div style="font-size:11px;color:var(--text-muted)">
              ${aiConf}% confidence
              ${ai.fire_confidence ? `&nbsp;·&nbsp;🔥 ${(ai.fire_confidence * 100).toFixed(0)}%` : ""}
              ${ai.smoke_confidence ? `&nbsp;·&nbsp;💨 ${(ai.smoke_confidence * 100).toFixed(0)}%` : ""}
            </div>
          </div>`;
      } else if (ai && ai.verdict === "error" && ai.error && !ai.error.includes("ROBOFLOW_API_KEY")) {
        aiBadge = `<div style="margin-top:6px;padding:4px 8px;background:rgba(0,0,0,0.2);border-radius:4px;
                    font-size:10px;color:var(--text-muted);text-align:center">🤖 AI scan failed: ${ai.error}</div>`;
      }

      const sat = r.satellite_confirmation;
    const satLine = sat && sat.confirmed
      ? `<div><span class="popup-label">Satellite:</span> <span style="color:#38bdf8;font-weight:600">🛰️ Confirmed (${sat.nearest_km}km)</span></div>`
      : sat
        ? `<div><span class="popup-label">Satellite:</span> No FIRMS match</div>`
        : "";

    marker.bindPopup(`
        <div class="popup-title">📸 Fire Report</div>
        <div><span class="popup-label">Status:</span> ${r.status}</div>
        <div><span class="popup-label">Location:</span> ${r.lat.toFixed(4)}, ${r.lon.toFixed(4)}</div>
        <div><span class="popup-label">${headingInfo}</span></div>
        <div><span class="popup-label">Reported:</span> ${new Date(r.captured_at).toLocaleString()}</div>
        <div><span class="popup-label">Source:</span> ${r.source_type || "citizen"}</div>
        ${satLine}
        ${r.photo_url ? `<div style="margin-top:6px"><img src="${r.photo_url}" style="width:100%;max-width:180px;border-radius:6px;" alt="Report photo" /></div>` : ""}
        ${aiBadge}
      `, { maxWidth: 280 });

      STATE.markers.reports.push(marker);

      // Pending reports pulse — they await a moderator's approval, so the
      // ring draws attention without cluttering confirmed fires.
      if (isPending) {
        const pulse = L.circleMarker([r.lat, r.lon], {
          radius: 7,
          color: "#eab308",
          weight: 2,
          fill: false,
          opacity: 0.8,
          className: "report-pulse-ring",
        }).addTo(STATE.map);
        STATE.markers.reports.push(pulse);
      }
    });

    // --- Cluster markers: convex hull (high zoom) or dot (low zoom) ---
    const zoom = STATE.map.getZoom();

    clusters.forEach((c) => {
      const color = clusterColor(c.count);
      const points = (c.points || [[c.centroid_lat, c.centroid_lon]]);

      if (zoom >= CLUSTER_POLYGON_ZOOM && points.length >= 3) {
        // --- Convex hull polygon (detailed at high zoom) ---
        const hull = convexHull(points);
        if (hull.length >= 3) {
          const polygon = L.polygon(hull, {
            color: color,
            weight: 2,
            opacity: 0.7,
            fillColor: color,
            fillOpacity: 0.15,
          }).addTo(STATE.map);
          STATE.markers.clusters.push(polygon);

          // Centroid dot
          const dot = L.circleMarker([c.centroid_lat, c.centroid_lon], {
            radius: 5,
            fillColor: color,
            color: "#fff",
            weight: 1.5,
            opacity: 0.9,
            fillOpacity: 0.8,
          }).addTo(STATE.map);
          STATE.markers.clusters.push(dot);

          // Attach popup to the polygon
          polygon.bindPopup(renderClusterPopup(c, color));
        }
      } else {
        // --- Dot with ring (simple at low zoom) ---
        const size = 10 + c.count * 6;
        const radius = Math.min(Math.max(size, 22), 52);

        const marker = L.circleMarker([c.centroid_lat, c.centroid_lon], {
          radius: radius / 2,
          fillColor: color,
          color: "#fff",
          weight: 2.5,
          opacity: 0.95,
          fillOpacity: 0.75,
        }).addTo(STATE.map);

        // Heat ring
        const ringRadius = Math.min(radius / 2 + 12, 60);
        const ring = L.circleMarker([c.centroid_lat, c.centroid_lon], {
          radius: ringRadius,
          color: color,
          weight: 1.5,
          opacity: 0.2,
          fill: false,
        }).addTo(STATE.map);

        STATE.markers.clusters.push(marker, ring);
        marker.bindPopup(renderClusterPopup(c, color));
      }

      // --- Triangulation overlay (only at high zoom) ---
      if (zoom >= CLUSTER_POLYGON_ZOOM && c.triangulation) {
        renderTriangulation(c, color);
      }
    });

    // --- Re-render true fire origin marker during live demo ---
    // (it gets cleared above, so restore it)
    if (STATE.liveDemo.active && STATE.liveDemo.fireLat != null) {
      renderTrueOrigin(STATE.liveDemo.fireLat, STATE.liveDemo.fireLon);
    }

  function renderClusterPopup(c, color) {
    // Build a popup showing cluster info and individual report locations
    let html = `<div style="text-align:center">
      <div class="popup-report-count" style="color:${color}">${c.count}</div>
      <div class="popup-label">REPORTS IN CLUSTER</div>`;
    if (c.points && c.points.length > 0) {
      html += `<div style="margin-top:4px;font-size:11px;color:var(--text-muted)">`;
      // Show up to 3 sample points
      const showPoints = c.points.slice(0, 3);
      showPoints.forEach(p => {
        html += `<div>📍 ${p[0].toFixed(4)}, ${p[1].toFixed(4)}</div>`;
      });
      if (c.points.length > 3) {
        html += `<div style="margin-top:2px;color:var(--text-muted)">…and ${c.points.length - 3} more</div>`;
      }
      html += `</div>`;
    }
    // --- Triangulation info (if available) ---
    if (c.triangulation) {
      const t = c.triangulation;
      const confSym = t.confidence === "high" ? "🟢" : t.confidence === "medium" ? "🟡" : "🔴";
      html += `<div style="margin-top:6px;padding-top:4px;border-top:1px solid var(--border)">`;
      html += `<div style="font-size:11px;font-weight:600;color:${color}">🔥 Triangulated Origin</div>`;
      html += `<div style="font-size:11px">${confSym} ${t.confidence.toUpperCase()} — from ${t.num_reports} bearings</div>`;
      html += `<div style="font-size:10px;color:var(--text-muted)">${t.fire_lat.toFixed(5)}, ${t.fire_lon.toFixed(5)}</div>`;
      html += `</div>`;
    }
    html += `</div>`;
    return html;
  }
  }

  // -----------------------------------------------------------------------
  // Bayesian Heatmap Layer (Canvas Overlay)
  // -----------------------------------------------------------------------

  /**
   * Custom Leaflet layer that renders the Bayesian probability grid as a
   * translucent heatmap on an HTML5 Canvas overlay.
   * Redraws on every map move/zoom and every state update.
   */
  const BayesianHeatmapLayer = L.Layer.extend({
    initialize: function (options) {
      L.Util.setOptions(this, options);
      this._regions = [];
      this._showHeatmap = true;
      this._showContour = true;
      this._threshold = 0.05;
    },

    onAdd: function (map) {
      this._container = L.DomUtil.create("div", "bayesian-heatmap-container");
      this._canvas = L.DomUtil.create("canvas", "bayesian-heatmap-canvas");
      this._container.appendChild(this._canvas);

      // IMPORTANT: append to the map's root container, NOT an overlay pane.
      // Leaflet applies its own CSS transform to panes (mapPane/overlayPane)
      // during pan/zoom for performance. Our drawing code positions cells
      // using map.latLngToContainerPoint(), which is already screen-absolute.
      // If we sat inside a pane, Leaflet's transform would shift our
      // already-positioned pixels a second time, causing the cells to drift
      // further out of place with every pan/zoom. The root map container is
      // not transformed, so it stays screen-fixed and matches our math.
      map.getContainer().appendChild(this._container);
      this._container.style.zIndex = 400; // sit above tiles, below markers/popups

      this._ctx = this._canvas.getContext("2d");

      // Resize canvas to map size
      this._resize();

      // Redraw on map move end (debounced — avoids jank during continuous pan)
      map.on("moveend", this._redraw, this);
      map.on("resize", this._resize, this);
      // Hide stale canvas during drag/zoom so it never shows content
      // positioned for the previous view.
      map.on("dragstart", this._hideCanvas, this);
      map.on("dragend", this._showAndRedraw, this);
      map.on("zoomstart", this._hideCanvas, this);
      map.on("zoomend", this._showAndRedraw, this);

      this._redraw();
    },

    onRemove: function (map) {
      map.off("moveend", this._redraw, this);
      map.off("resize", this._resize, this);
      map.off("dragstart", this._hideCanvas, this);
      map.off("dragend", this._showAndRedraw, this);
      map.off("zoomstart", this._hideCanvas, this);
      map.off("zoomend", this._showAndRedraw, this);
      if (this._container && this._container.parentNode) {
        this._container.parentNode.removeChild(this._container);
      }
      this._container = null;
      this._canvas = null;
      this._ctx = null;
      this._offscreenCanvas = null;
      this._blurCanvas = null;
    },

    setData: function (regions) {
      this._regions = regions || [];
      this._redraw();
    },

    setOptions: function (opts) {
      if (opts.showHeatmap !== undefined) this._showHeatmap = opts.showHeatmap;
      if (opts.showContour !== undefined) this._showContour = opts.showContour;
      if (opts.threshold !== undefined) this._threshold = opts.threshold;
      this._redraw();
    },

    _resize: function () {
      if (!this._canvas || !this._map) return;
      const size = this._map.getSize();
      this._canvas.width = size.x;
      this._canvas.height = size.y;
      this._canvas.style.width = size.x + "px";
      this._canvas.style.height = size.y + "px";
      if (this._container) {
        this._container.style.width = size.x + "px";
        this._container.style.height = size.y + "px";
      }
      this._redraw();
    },

    /**
     * Hide the canvas during map drag so stale content isn't visible at wrong positions.
     */
    _hideCanvas: function () {
      if (this._container) {
        this._container.style.opacity = '0';
      }
    },

    /**
     * Restore canvas visibility after a drag ends.
     * No redraw here — Leaflet fires moveend synchronously after dragend,
     * so the moveend listener handles the actual redraw without doubling up.
     */
    _showAndRedraw: function () {
      if (this._container) {
        this._container.style.opacity = '1';
      }
    },

    /**
     * Return a reusable offscreen canvas at the requested size.
     */
    _getOffscreenCanvas: function (w, h) {
      if (!this._offscreenCanvas || this._offscreenCanvas.width !== w || this._offscreenCanvas.height !== h) {
        this._offscreenCanvas = document.createElement('canvas');
        this._offscreenCanvas.width = w;
        this._offscreenCanvas.height = h;
      }
      return this._offscreenCanvas;
    },

    /**
     * Compute a region's cell size in pixels at the current map zoom.
     * Uses the region's own reference latitude so sizing stays consistent
     * regardless of how far you pan from that fire's grid center. Each
     * fire cluster has its own grid (and thus potentially its own cell
     * size), so this is computed per-region rather than once globally.
     */
    _cellSizePxFor: function (region) {
      if (!this._map || !region || region.cellSizeM <= 0) return 10;
      const zoom = this._map.getZoom();
      const refLat = region.refLat != null ? region.refLat : this._map.getCenter().lat;
      const latRad = refLat * Math.PI / 180;
      // Approx meters per pixel at this latitude and zoom
      const mPerPx = (40075017 / 256) * Math.cos(latRad) / Math.pow(2, zoom);
      return Math.max(5, Math.round(region.cellSizeM / mPerPx));
    },

    _redraw: function () {
      if (!this._ctx || !this._map || !this._canvas) return;

      const ctx = this._ctx;
      const map = this._map;
      const canvasW = this._canvas.width;
      const canvasH = this._canvas.height;

      ctx.clearRect(0, 0, canvasW, canvasH);

      const regions = this._regions || [];

      // --- Draw organic "lava field" heatmap ---
      // Technique: draw each cell as a soft radial gradient whose brightness
      // IS the cell's absolute probability (t = p), composited with per-
      // channel max ("lighten") so overlapping/adjacent cells MERGE into
      // continuous blobby shapes instead of a grid of squares — and so
      // overlapping cells never sum into a brighter color than their
      // certainty deserves. Then remap the intensity field through a lava
      // color ramp per-pixel: hotter (more certain) regions glow brighter.
      //
      // Cells come from potentially several independent grids (one per
      // fire cluster), each with its own cell size / reference latitude,
      // so pixel radius is computed per-region. The color scale is ABSOLUTE
      // (probability 0..1, matching the "Max prob" stat): yellow is always
      // low probability, red is always ≥0.6, no matter what else is on
      // screen or at any zoom level.
      const anyCells = regions.some((r) => r.cells && r.cells.length > 0);
      if (this._showHeatmap && anyCells) {
        // Pass 1: accumulate a grayscale intensity field via per-cell max.
        const accumCanvas = this._getOffscreenCanvas(canvasW, canvasH);
        const accumCtx = accumCanvas.getContext('2d');
        accumCtx.clearRect(0, 0, canvasW, canvasH);
        accumCtx.globalCompositeOperation = 'lighten';

        for (const region of regions) {
          if (!region.cells || region.cells.length === 0) continue;
          const cellSizePx = this._cellSizePxFor(region);
          // Radius bigger than one cell so neighboring cells overlap and
          // fuse into a single shape rather than staying as separate dots.
          const radius = cellSizePx * 1.7;

          for (const cell of region.cells) {
            const p = cell.p;
            if (p < this._threshold) continue;

            const pt = map.latLngToContainerPoint([cell.lat, cell.lon]);
            if (pt.x < -radius || pt.x > canvasW + radius ||
                pt.y < -radius || pt.y > canvasH + radius) continue;

            // Absolute scale: brightness = the cell's true probability.
            // (Grid probabilities live in 0..1 — bayesian_filter clamps at
            // PROB_MAX=0.9999 — so 1.0 is the natural "hot" end.)
            const t = Math.min(1, Math.max(0, p));
            const v = Math.round(255 * t);
            const grd = accumCtx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, radius);
            grd.addColorStop(0, `rgba(${v},${v},${v},1)`);
            grd.addColorStop(1, 'rgba(0,0,0,1)');
            accumCtx.fillStyle = grd;
            accumCtx.beginPath();
            accumCtx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
            accumCtx.fill();
          }
        }
        accumCtx.globalCompositeOperation = 'source-over';

        // Pass 2: soften the blob edges into smooth, melty contours.
        const blurCanvas = this._blurCanvas || (this._blurCanvas = document.createElement('canvas'));
        blurCanvas.width = canvasW;
        blurCanvas.height = canvasH;
        const blurCtx = blurCanvas.getContext('2d');
        blurCtx.clearRect(0, 0, canvasW, canvasH);
        blurCtx.filter = 'blur(6px)';
        blurCtx.drawImage(accumCanvas, 0, 0);
        blurCtx.filter = 'none';

        // Pass 3: remap accumulated intensity -> lava color per pixel.
        const imgData = blurCtx.getImageData(0, 0, canvasW, canvasH);
        const data = imgData.data;
        for (let i = 0; i < data.length; i += 4) {
          const intensity = data[i] / 255; // gray channel (max of cells, not summed overlap)
          if (intensity < 0.03) {
            data[i + 3] = 0;
            continue;
          }
          const color = this._lavaColor(intensity);
          data[i] = color.r;
          data[i + 1] = color.g;
          data[i + 2] = color.b;
          data[i + 3] = Math.round(color.a * 255);
        }
        blurCtx.putImageData(imgData, 0, 0);

        ctx.save();
        ctx.globalAlpha = 0.92;
        ctx.drawImage(blurCanvas, 0, 0);
        ctx.restore();
      }

      // --- Draw contour as glowing fire perimeter boundary ---
      // Contour segments are absolute lat/lon polylines, so segments from
      // every region can just be concatenated and drawn the same way.
      const allContourSegs = [];
      for (const region of regions) {
        if (region.contour && region.contour.length > 0) {
          allContourSegs.push(...region.contour);
        }
      }
      if (this._showContour && allContourSegs.length > 0) {
        // First pass: glow behind the perimeter
        ctx.save();
        ctx.shadowColor = 'rgba(255, 80, 0, 0.6)';
        ctx.shadowBlur = 20;
        ctx.lineWidth = 4;
        ctx.strokeStyle = 'rgba(255, 150, 20, 0.5)';
        ctx.globalAlpha = 0.6;

        for (const seg of allContourSegs) {
          if (seg.length < 2) continue;
          ctx.beginPath();
          const pt0 = map.latLngToContainerPoint(seg[0]);
          ctx.moveTo(pt0.x, pt0.y);
          for (let k = 1; k < seg.length; k++) {
            const pt = map.latLngToContainerPoint(seg[k]);
            ctx.lineTo(pt.x, pt.y);
          }
          ctx.stroke();
        }
        ctx.restore();

        // Second pass: bright core line
        ctx.save();
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgba(255, 220, 100, 0.9)';
        ctx.shadowColor = 'rgba(255, 200, 50, 0.3)';
        ctx.shadowBlur = 8;

        for (const seg of allContourSegs) {
          if (seg.length < 2) continue;
          ctx.beginPath();
          const pt0 = map.latLngToContainerPoint(seg[0]);
          ctx.moveTo(pt0.x, pt0.y);
          for (let k = 1; k < seg.length; k++) {
            const pt = map.latLngToContainerPoint(seg[k]);
            ctx.lineTo(pt.x, pt.y);
          }
          ctx.stroke();
        }
        ctx.restore();
      }

      ctx.globalAlpha = 1.0;
    },

    /**
     * Certainty color ramp.
     * Maps an ABSOLUTE probability (t = p, 0..1) to a fixed
     * yellow→orange→red→crimson gradient — the same color always means the
     * same probability, independent of what else is on screen:
     *   yellow       = low probability  (< 0.3)
     *   orange       = moderate         (0.3 – 0.6)
     *   red          = high             (0.6 – 0.85)
     *   deep crimson = very high        (≥ 0.85)
     */
    _lavaColor: function (t) {
      t = Math.min(1, Math.max(0, t));
      let r, g, b;
      let a;
      if (t < 0.3) {
        // Yellow → amber (low certainty)
        const s = t / 0.3;
        r = 255;
        g = Math.round(200 - 60 * s);   // 200 → 140
        b = Math.round(40 - 40 * s);     // 40 → 0
        a = 0.25 + 0.25 * s;
      } else if (t < 0.6) {
        // Amber → orange (medium certainty)
        const s = (t - 0.3) / 0.3;
        r = 255;
        g = Math.round(140 - 80 * s);   // 140 → 60
        b = 0;
        a = 0.50 + 0.25 * s;
      } else if (t < 0.85) {
        // Orange → red (high certainty)
        const s = (t - 0.6) / 0.25;
        r = 255;
        g = Math.round(60 - 60 * s);    // 60 → 0
        b = 0;
        a = 0.75 + 0.12 * s;
      } else {
        // Red → deep crimson (very high certainty)
        const s = (t - 0.85) / 0.15;
        r = 255;
        g = 0;
        b = Math.round(40 * s);         // 0 → 40 (deepens to crimson)
        a = 0.87 + 0.08 * s;
      }
      return { r, g, b, a: Math.min(0.95, a) };
    },
  });

  // -----------------------------------------------------------------------
  // Bayesian Grid — Fetch & Render
  // -----------------------------------------------------------------------

  /**
   * Initialize the Bayesian heatmap layer on the map.
   */
  function initBayesianLayer() {
    if (!STATE.map) return;

    STATE.bayesian.heatmapLayer = new BayesianHeatmapLayer({
      pane: "overlayPane",
    });
    STATE.map.addLayer(STATE.bayesian.heatmapLayer);

    // Initial fetch
    if (STATE.bayesian.active) {
      fetchBayesianState();
    }
  }

  /**
   * Fetch the latest Bayesian grid state from the server and render.
   */
  /**
   * Current map viewport as a "west,south,east,north" bbox string, or ""
   * if the map isn't ready. Sending the viewport lets the backend skip
   * serializing fires the user can't see (important with global FIRMS
   * data — 1000+ grids would otherwise make every poll take ~30s).
   */
  function viewportBBox() {
    if (!STATE.map) return "";
    const b = STATE.map.getBounds();
    const sw = b.getSouthWest();
    const ne = b.getNorthEast();
    return `${sw.lng},${sw.lat},${ne.lng},${ne.lat}`;
  }

  // Below this zoom the map renders cheap intensity dots (detail=meta)
  // instead of full heatmap state, so zooming out over thousands of fires
  // stays fast.
  const BAYESIAN_LOD_ZOOM = 7;

  async function fetchBayesianState() {
    if (!STATE.bayesian.active || !STATE.bayesian.heatmapLayer) return;

    try {
      // The server floors the threshold at MIN_STATE_THRESHOLD, so sending
      // the raw slider value is safe (0 is allowed — faint cells just get
      // dropped server-side to keep the payload small).
      const threshold = STATE.bayesian.threshold;
      // Note: `mode` must be joined with `&` here because the URL already
      // has query params — appending `?mode=demo` would corrupt the string
      // and Flask would never see the mode (falling back to production).
      const bbox = viewportBBox();

      // A. Zoom-aware level of detail: far out, full heatmap state is
      // pointless and expensive — fetch cheap intensity dots instead.
      const zoom = STATE.map ? STATE.map.getZoom() : 12;
      const meta = zoom < BAYESIAN_LOD_ZOOM;
      // D. Skip marching-squares contours server-side when the toggle is off.
      const contour = STATE.bayesian.showContour ? 0.6 : 0;
      const url = `/api/bayesian/state?threshold=${threshold}&contour=${contour}&mode=${STATE.mode}&bbox=${encodeURIComponent(bbox)}&detail=${meta ? "meta" : "full"}`;

      const res = await fetch(url);
      if (!res.ok) return;

      const data = await res.json();
      const grids = data.grids || [];

      if (meta) {
        // Low zoom — dots only: clear the heavy heatmap, wind labels and
        // stats (the meta payload carries no cell/statistics data).
        STATE.bayesian.regions = [];
        STATE.bayesian.stats = null;
        STATE.bayesian.heatmapLayer.setData([]);
        renderGridWindLabels([], []);
        renderMetaDots(grids);
        updateBayesianStats();
        return;
      }

      // Full detail — clear any low-zoom dots.
      renderMetaDots([]);

      // Each grid is one physically separate fire (its own cluster), with
      // its own cell size / reference origin. Build one "region" per grid
      // for the heatmap layer to render in a single pass.
      const regions = grids.map((g) => {
        const st = g.state || {};
        return {
          cells: st.cells || [],
          contour: g.contour || [],
          cellSizeM: st.cell_size_m || 100,
          refLat: st.ref_lat,
          refLon: st.ref_lon,
          nx: st.nx || 120,
          ny: st.ny || 120,
        };
      });

      STATE.bayesian.regions = regions;

      // Aggregate stats across all grids for the stats panel
      STATE.bayesian.stats = _aggregateBayesianStats(grids);

      STATE.bayesian.heatmapLayer.setData(regions);

      // Update stats panel
      updateBayesianStats();

      // --- Render per-grid wind labels on the map ---
      renderGridWindLabels(grids, regions);

      // --- Refresh road risk overlay if active ---
      if (STATE.bayesian.roadRiskActive) {
        fetchRoadRisk();
      }
    } catch (err) {
      console.warn("Fire grid state fetch error:", err);
    }
  }

  /**
   * Combine per-grid statistics (one fire cluster per grid) into a single
   * summary for the stats panel: max probability across all fires, and
   * summed area/cell counts.
   */
  function _aggregateBayesianStats(grids) {
    if (!grids.length) return null;
    let maxP = 0;
    let areaSum = 0;
    let cellsSum = 0;
    for (const g of grids) {
      const s = g.statistics || {};
      if ((s.max_p || 0) > maxP) maxP = s.max_p;
      areaSum += s.area_ha_p_above_0_10 || 0;
      cellsSum += s.cells_p_above_0_10 || 0;
    }
    return { max_p: maxP, area_ha_p_above_0_10: areaSum, cells_p_above_0_10: cellsSum };
  }

  /**
   * Update the small satellite status line under the panel controls.
   */
  function _setSatelliteStatus(text, kind) {
    const el = document.getElementById("satellite-status");
    if (!el) return;
    el.textContent = text;
    el.classList.toggle("satellite-active", kind === "active");
    el.classList.toggle("satellite-error", kind === "error");
  }

  /**
   * Trigger a one-shot simulated satellite pass over all active fire grids,
   * then immediately refresh the heatmap so newly-fused hotspots show up
   * without waiting for the next 5s poll.
   */
  async function simulateSatellitePass() {
    // Simulated satellite passes only inject into demo grids.
    if (STATE.mode !== "demo") {
      await setMode("demo");
    }
    const btn = document.getElementById("satellite-pass-btn");
    if (btn) btn.disabled = true;
    _setSatelliteStatus("Satellite: scanning…", "active");

    try {
      const res = await fetch("/api/satellite/simulate-pass", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ probability: 0.6, min_hotspots: 1, max_hotspots: 3 }),
      });
      const data = await res.json();

      if (!res.ok) {
        _setSatelliteStatus(`Satellite: ${data.error || "pass failed"}`, "error");
        return;
      }

      const kind = data.grids_considered === 0 ? null : (data.injected > 0 ? "active" : null);
      _setSatelliteStatus(`Satellite: ${data.message}`, kind);

      // Reflect the new evidence right away rather than waiting for the poll
      await fetchBayesianState();
    } catch (err) {
      console.warn("Satellite pass error:", err);
      _setSatelliteStatus("Satellite: request failed", "error");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  /**
   * Start or stop the backend's continuous simulated satellite poller.
   */
  // -----------------------------------------------------------------------
  // NASA FIRMS — Real Satellite Data
  // -----------------------------------------------------------------------

  /**
   * Manually trigger a real NASA FIRMS data fetch for all active fire grids.
   *
   * The pass (global FIRMS API call + clustering ~100k hotspots + grid
   * injection) runs as a background job in the queue worker, so this only
   * waits for the job to be accepted, then polls
   * /api/satellite/poller/status until the fetch completes and shows the
   * summary. The map's own 5s polling refreshes the heatmap meanwhile.
   */
  async function fetchFirmsData() {
    // Real FIRMS ingestion is a live/operational action — show live data.
    if (STATE.mode !== "production") {
      await setMode("production");
    }
    const btn = document.getElementById("firms-fetch-btn");
    if (btn) btn.disabled = true;
    _setSatelliteStatus("Satellite: fetching FIRMS…", "active");

    try {
      const res = await fetch("/api/satellite/firms-fetch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ day_range: 1, min_confidence: "nominal" }),
      });
      const data = await res.json();

      if (!res.ok) {
        _setSatelliteStatus(`Satellite: ${data.error || "FIRMS fetch failed"}`, "error");
        return;
      }

      let result = data;
      if (data.accepted) {
        // Background job — poll until it finishes (up to ~14 minutes,
        // matching the server's 15-minute stale-flag window).
        _setSatelliteStatus("Satellite: fetching FIRMS… (background job running)", "active");
        const deadline = Date.now() + 14 * 60 * 1000;
        while (Date.now() < deadline) {
          await new Promise((r) => setTimeout(r, 3000));
          let st = null;
          try {
            st = await (await fetch("/api/satellite/poller/status")).json();
          } catch (e) {
            continue; // transient — keep polling
          }
          if (st && !st.firms_fetch_in_progress && st.firms_fetch_last_result) {
            result = st.firms_fetch_last_result;
            break;
          }
        }
      }

      if (result.accepted) {
        // Still running (or timed out) — the map's own polling will pick up
        // the new grids whenever the background job finishes.
        _setSatelliteStatus("Satellite: FIRMS fetch still processing in the background — map refreshes automatically", "active");
        await fetchBayesianState();
        return;
      }

      const kind = result.grids_considered === 0 ? null : (result.injected > 0 ? "active" : null);
      const emoji = result.injected > 0 ? "🛰️" : "ℹ️";
      _setSatelliteStatus(`Satellite: ${emoji} ${result.message || "no data"}`, kind);

      // --- Auto-enable Bayesian overlay if it's not already active ---
      // fetchBayesianState silently returns if STATE.bayesian.active is false,
      // so new FIRMS grids would be created server-side but never rendered.
      if (result.injected > 0 && !STATE.bayesian.active) {
        toggleBayesian(true);
        // Keep the heatmap on but respect the saved card preference (mobile
        // defaults to collapsed so the map stays visible).
        applyBayesianPanelLayout();
      }

      // Refresh the heatmap with the new evidence
      await fetchBayesianState();

      // --- Auto-pan map to show the newly created FIRMS fire grids ---
      if (result.injected > 0 && STATE.bayesian.regions && STATE.bayesian.regions.length > 0) {
        // Collect all cell coordinates from all regions to compute bounds
        const allPoints = [];
        for (const region of STATE.bayesian.regions) {
          if (region.cells) {
            for (const cell of region.cells) {
              allPoints.push([cell.lat, cell.lon]);
            }
          }
        }
        if (allPoints.length > 0) {
          const bounds = L.latLngBounds(allPoints);
          STATE.map.fitBounds(bounds, { padding: [50, 50], maxZoom: 12 });
        }
      }
    } catch (err) {
      console.warn("FIRMS fetch error:", err);
      _setSatelliteStatus("Satellite: FIRMS request failed", "error");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  /**
   * Start or stop the real NASA FIRMS background poller.
   */
  async function toggleFirmsPoller(active) {
    // Real FIRMS polling is a live/operational action.
    if (active && STATE.mode !== "production") {
      await setMode("production");
    }
    const toggle = document.getElementById("firms-poller-toggle");
    try {
      const url = active ? "/api/satellite/firms-poller/start" : "/api/satellite/firms-poller/stop";
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: active ? JSON.stringify({ interval_s: 1800, day_range: 1, min_confidence: "nominal" }) : undefined,
      });
      const data = await res.json();

      if (!res.ok) {
        _setSatelliteStatus(`Satellite: ${data.error || "FIRMS poller error"}`, "error");
        if (toggle) toggle.checked = STATE.bayesian.firmsPollerActive;
        return;
      }

      STATE.bayesian.firmsPollerActive = active;
      _setSatelliteStatus(
        active ? "🛰️ FIRMS live: polling every 10 min" : "Satellite: FIRMS poller stopped",
        active ? "active" : null
      );

      // If starting, do an immediate fetch so the user sees data right away
      if (active) {
        await fetchFirmsData();
      }
    } catch (err) {
      console.warn("FIRMS poller toggle error:", err);
      _setSatelliteStatus("Satellite: FIRMS poller request failed", "error");
      if (toggle) toggle.checked = STATE.bayesian.firmsPollerActive; // revert
    }
  }

  /**
   * FIRMS Live is the default: on boot, sync the toggle with the server's
   * durable poller state, and start the poller if it isn't already running.
   */
  async function syncFirmsPollerDefault() {
    try {
      const res = await fetch("/api/satellite/poller/status");
      const st = await res.json();
      const toggle = document.getElementById("firms-poller-toggle");
      if (st && st.firms_poller_active) {
        STATE.bayesian.firmsPollerActive = true;
        if (toggle) toggle.checked = true;
        _setSatelliteStatus("🛰️ FIRMS live: polling every 10 min", "active");
      } else {
        // Nothing running yet — FIRMS Live is the default.
        await toggleFirmsPoller(true);
      }
    } catch (err) {
      console.warn("FIRMS default sync failed:", err);
    }
  }

  async function toggleSatellitePoller(active) {
    // The simulated poller only injects into demo grids.
    if (active && STATE.mode !== "demo") {
      await setMode("demo");
    }
    const toggle = document.getElementById("satellite-poller-toggle");
    try {
      const url = active ? "/api/satellite/poller/start" : "/api/satellite/poller/stop";
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: active ? JSON.stringify({ interval_s: 20, probability: 0.5 }) : undefined,
      });
      const data = await res.json();

      if (!res.ok) {
        _setSatelliteStatus(`Satellite: ${data.error || "poller error"}`, "error");
        if (toggle) toggle.checked = STATE.bayesian.satellitePollerActive; // revert
        return;
      }

      STATE.bayesian.satellitePollerActive = active;
      _setSatelliteStatus(
        active ? "Satellite: live poller running (every 20s)" : "Satellite: poller stopped",
        active ? "active" : null
      );
    } catch (err) {
      console.warn("Satellite poller toggle error:", err);
      _setSatelliteStatus("Satellite: request failed", "error");
      if (toggle) toggle.checked = STATE.bayesian.satellitePollerActive; // revert
    }
  }

  // Throttle road risk re-fetches — the backend has a 10-minute OSM cache,
  // so hammering it every 5 seconds on the Bayesian poll is wasteful.
  // 15 seconds is enough to pick up contour changes without hitting Overpass.
  let _lastRoadRiskFetch = 0;
  const ROAD_RISK_THROTTLE_MS = 15000;

  /**
   * Fetch road risk data from the backend and render on the map.
   * Calls POST /api/bayesian/road-risk which fetches OSM roads near each fire
   * contour and assesses fire spread risk using the head/back/flank ellipse.
   * Renders as a GeoJSON layer on the map, colored by risk tier.
   * Does NOT require STATE.bayesian.active — grids exist independently.
   *
   * **Throttled**: will not make a new request more often than every
   * ``ROAD_RISK_THROTTLE_MS`` (15 s) to avoid hammering the Overpass API.
   */
  async function fetchRoadRisk() {
    if (!STATE.map || !STATE.bayesian.roadRiskActive) return;

    // Throttle: skip this call if we attempted recently. Updated for BOTH
    // successes and failures so a down Overpass can't turn into a retry
    // firehose (each retry would miss the OSM cache and hammer the API).
    const now = Date.now();
    if (now - _lastRoadRiskFetch < ROAD_RISK_THROTTLE_MS) return;
    _lastRoadRiskFetch = now;

    const statusEl = document.getElementById("roadrisk-status");
    if (statusEl) statusEl.textContent = "🛣️ fetching…";

    try {
      const res = await fetch("/api/bayesian/road-risk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(modeBody({
          grid_id: "all",
          contour_level: 0.3,
          radius_km: 5.0,
          bbox: viewportBBox(),
        })),
      });

      const data = await res.json();

      if (!res.ok) {
        const msg = data.error || "request failed";
        if (statusEl) statusEl.textContent = `🛣️ ${msg}`;
        return;
      }

      // Remove old road risk layer
      if (STATE.bayesian.roadRiskLayer && STATE.map) {
        STATE.map.removeLayer(STATE.bayesian.roadRiskLayer);
        STATE.bayesian.roadRiskLayer = null;
      }

      if (!data.features || data.features.length === 0) {
        const reason = data.metadata && data.metadata.empty_reason;
        if (statusEl) statusEl.textContent = reason ? `🛣️ ${reason}` : "🛣️ no roads found";
        return;
      }

      // Risk tier colors
      const riskColors = {
        critical: "#ef4444",   // bright red
        high: "#f97316",       // orange
        moderate: "#eab308",   // yellow
        low: "#22c55e",        // green
      };

      // Build GeoJSON layer with risk-tier styling
      const layer = L.geoJSON(data, {
        style: function (feature) {
          const tier = feature.properties.risk_tier || "low";
          const opacity = tier === "critical" ? 0.95 : tier === "high" ? 0.85 : 0.65;
          return {
            color: riskColors[tier] || "#64748b",
            weight: tier === "critical" ? 5 : tier === "high" ? 4 : 3,
            opacity: opacity,
          };
        },
        onEachFeature: function (feature, layer) {
          const p = feature.properties;
          if (!p) return;

          const tier = p.risk_tier || "low";
          const arrival = p.t_arrival_min != null ? `${p.t_arrival_min} min` : "—";
          const dist = p.nearest_distance_m != null ? `${p.nearest_distance_m} m` : "—";
          const rate = p.effective_spread_rate_m_min != null ? `${p.effective_spread_rate_m_min} m/min` : "—";
          const prob = p.probability_at_contour != null ? `${(p.probability_at_contour * 100).toFixed(1)}%` : "—";
          const bearing = p.bearing_from_wind_deg != null ? `${p.bearing_from_wind_deg}°` : "—";

          const tierEmoji = tier === "critical" ? "🔴" : tier === "high" ? "🟠" : tier === "moderate" ? "🟡" : "🟢";
          layer.bindPopup(`
            <div style="text-align:center">
              <div style="font-size:15px;font-weight:700;color:${riskColors[tier] || "#64748b"}">${tierEmoji} ${tier.toUpperCase()} RISK</div>
              <table style="margin:6px 0;font-size:12px;border-collapse:collapse">
                <tr><td style="padding:2px 6px;color:var(--text-muted)">⏱ Arrival:</td><td style="padding:2px 6px;font-weight:600">${arrival}</td></tr>
                <tr><td style="padding:2px 6px;color:var(--text-muted)">📏 Distance:</td><td style="padding:2px 6px;font-weight:600">${dist}</td></tr>
                <tr><td style="padding:2px 6px;color:var(--text-muted)">💨 Spread rate:</td><td style="padding:2px 6px;font-weight:600">${rate}</td></tr>
                <tr><td style="padding:2px 6px;color:var(--text-muted)">🧭 Bearing Φ:</td><td style="padding:2px 6px;font-weight:600">${bearing}</td></tr>
                <tr><td style="padding:2px 6px;color:var(--text-muted)">🔥 Edge prob:</td><td style="padding:2px 6px;font-weight:600">${prob}</td></tr>
              </table>
            </div>
          `, { maxWidth: 240, className: "roadrisk-popup" });
        },
      }).addTo(STATE.map);

      STATE.bayesian.roadRiskLayer = layer;

      // Update status with tier counts
      const counts = {};
      for (const f of data.features) {
        const t = f.properties.risk_tier || "low";
        counts[t] = (counts[t] || 0) + 1;
      }
      const parts = [];
      if (counts.critical) parts.push(`🔴${counts.critical}`);
      if (counts.high) parts.push(`🟠${counts.high}`);
      if (counts.moderate) parts.push(`🟡${counts.moderate}`);
      if (counts.low) parts.push(`🟢${counts.low}`);
      if (statusEl) statusEl.textContent = parts.length ? `🛣️ ${parts.join(" ")}` : "🛣️ no risk data";
    } catch (err) {
      console.warn("Road risk fetch error:", err);
      const statusEl = document.getElementById("roadrisk-status");
      if (statusEl) statusEl.textContent = "🛣️ error";
    }
  }

  /**
   * Toggle the road risk overlay on/off.
   * Does NOT require Bayesian overlay to be active.
   */
  function toggleRoadRisk(active) {
    STATE.bayesian.roadRiskActive = active;

    const statusEl = document.getElementById("roadrisk-status");

    if (active) {
      if (statusEl) statusEl.textContent = "🛣️ fetching…";
      fetchRoadRisk();
    } else {
      // Remove the road risk layer from the map
      if (STATE.bayesian.roadRiskLayer && STATE.map) {
        STATE.map.removeLayer(STATE.bayesian.roadRiskLayer);
        STATE.bayesian.roadRiskLayer = null;
      }
      if (statusEl) statusEl.textContent = "🛣️ idle";
    }
  }

  /**
   * Update the Bayesian stats panel with latest statistics.
   */
  function updateBayesianStats() {
    const s = STATE.bayesian.stats;
    if (!s) return;

    const els = {
      maxP: document.getElementById("bayesian-stat-maxp"),
      area: document.getElementById("bayesian-stat-area"),
      cells: document.getElementById("bayesian-stat-cells"),
    };

    if (els.maxP) els.maxP.textContent = s.max_p?.toFixed(3) || "—";
    if (els.area) els.area.textContent = `${s.area_ha_p_above_0_10 || 0} ha`;
    if (els.cells) els.cells.textContent = s.cells_p_above_0_10 || 0;
  }

  /**
   * Compass direction label from degrees.
   */
  function _windDirLabel(deg) {
    const dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
    const idx = Math.round(((deg + 360) % 360) / 45) % 8;
    return dirs[idx];
  }

  /**
   * Render per-grid wind labels on the map.
   *
   * Creates/updates a small floating badge near each grid's reference
   * center showing wind speed and direction, with an animated arrow.
   * The badge is styled to match the existing panel design language.
   */
  /**
   * Low-zoom (detail=meta) renderer: colored circles sized/colored by peak
   * probability. Cheaper than the full heatmap and all that matters at
   * world scale. Refreshed wholesale on each poll.
   */
  function renderMetaDots(dots) {
    if (!STATE.map) return;
    // Skip the DOM rebuild when nothing changed (most 5s polls) — avoids
    // churning hundreds of markers every poll.
    // Users-only filter hides all satellite-derived overlays — do not even
    // create the layer, otherwise it would be added straight to the map.
    if (STATE.bayesian.usersOnly) return;

    const sig = (dots || [])
      .map((d) => `${d.id}:${(d.max_p || 0).toFixed(4)}`)
      .join("|");
    if (sig === STATE.bayesian.metaSig) return;
    STATE.bayesian.metaSig = sig;

    if (!STATE.bayesian.metaLayer) {
      STATE.bayesian.metaLayer = L.layerGroup().addTo(STATE.map);
    }
    STATE.bayesian.metaLayer.clearLayers();
    // Respect the threshold slider so it keeps behaving at low zoom.
    const threshold = STATE.bayesian.threshold || 0;
    const capped = (dots || []).slice(0, 600);
    for (const d of capped) {
      const p = d.max_p || 0;
      if (p < threshold) continue;
      // Clickable: beta users expected the dots to be buttons — they now
      // are. Clicking zooms into the fire and opens a summary popup.
      const dot = L.circleMarker([d.lat, d.lon], {
        radius: 3 + Math.min(15, p * 130),
        color: "#ffffff",
        weight: 1,
        fillColor: _metaDotColor(p),
        fillOpacity: 0.85,
        className: "fire-meta-dot",
        // Don't bubble the click to the map — the map's closePopupOnClick
        // would close the just-opened popup before the user sees it.
        bubblingMouseEvents: false,
      });
      dot.on("click", () => openFireDotPopup(d));
      dot.addTo(STATE.bayesian.metaLayer);
    }
  }

  /**
   * Approximate great-circle distance in kilometres (Haversine).
   */
  function _latLonKm(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLon = ((lon2 - lon1) * Math.PI) / 180;
    const a =
      Math.sin(dLat / 2) ** 2 +
      Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  /**
   * Open a summary popup for a low-zoom fire dot and zoom the map into it.
   *
   * The meta payload only carries grid metadata (peak probability, wind),
   * so the popup enriches it with what the client already knows: reports
   * near the fire (count + source types) corroborate whether this is a
   * citizen/agency-reported fire or a purely satellite (FIRMS) detection.
   *
   * The popup is a STANDALONE map popup (not bound to the dot) because
   * zooming to full detail immediately clears the meta-dot layer — a bound
   * popup would be destroyed with it.
   */
  function openFireDotPopup(d) {
    if (!STATE.map) return;

    const p = Math.min(1, Math.max(0, d.max_p || 0));
    const pct = Math.round(p * 100);
    const color = _metaDotColor(p);

    // Reports within ~3 km (matches the FIRMS corroboration radius) tell us
    // the fire's sources and how many people reported it. Only CONFIRMED
    // reports corroborate a fire — pending ones are still in review.
    const nearby = (STATE.reports || []).filter((r) =>
      r.status === "confirmed" && _latLonKm(d.lat, d.lon, r.lat, r.lon) <= 3
    );
    const sources = [...new Set(nearby.map((r) => (r.source_type || "citizen")))];

    const windSpeed = d.wind_speed != null ? d.wind_speed : 3.0;
    const windDir = d.wind_dir_deg != null ? d.wind_dir_deg : 270.0;

    const html = `
      <div class="popup-title">🔥 Active Fire</div>
      <div style="margin-top:4px"><span class="popup-label">Probability:</span>
        <span style="color:${color};font-weight:700">${pct}%</span></div>
      <div><span class="popup-label">Source:</span> ${sources.length ? sources.join(", ") : "Satellite (FIRMS)"}</div>
      <div><span class="popup-label">Reports:</span> ${nearby.length} confirmed nearby</div>
      <div><span class="popup-label">Wind:</span> ${windSpeed.toFixed(1)} m/s ${_windDirLabel(windDir)}</div>
      <div style="margin-top:4px;font-size:11px;color:var(--text-muted)">📍 ${d.lat.toFixed(4)}, ${d.lon.toFixed(4)}</div>
      <div style="margin-top:6px;font-size:11px;color:var(--accent);font-weight:600">🔍 Zooming in…</div>
    `;

    // autoPan: false — we're already flying to the fire; auto-panning on
    // open would fight the flyTo animation.
    L.popup({ maxWidth: 280, autoPan: false })
      .setLatLng([d.lat, d.lon])
      .setContent(html)
      .openOn(STATE.map);

    // Fly in so the full heatmap + reports render (above the meta-dot LOD).
    const targetZoom = Math.max(STATE.map.getZoom(), 10);
    STATE.map.flyTo([d.lat, d.lon], targetZoom, { duration: 0.8 });

    // The map only re-fetches grid state on its 5s poll; nudge it so the
    // full-detail heatmap appears as soon as the flyTo lands.
    setTimeout(() => {
      if (STATE.bayesian.active) fetchBayesianState();
    }, 900);
  }

  function _metaDotColor(p) {
    // Same absolute stops as the heatmap lava ramp (0.85 / 0.6 / 0.3) so the
    // low-zoom dots and the full heatmap agree on what a color means.
    if (p >= 0.85) return "#ff1a1a";
    if (p >= 0.6) return "#ff6600";
    if (p >= 0.3) return "#ffaa00";
    return "#ffd633";
  }

  function renderGridWindLabels(grids, regions) {
    // Remove old wind label markers
    for (const m of STATE.bayesian.windLabels) {
      if (STATE.map) STATE.map.removeLayer(m);
    }
    STATE.bayesian.windLabels = [];

    if (!STATE.map) return;

    // Users-only filter hides all satellite-derived overlays.
    if (STATE.bayesian.usersOnly) return;

    for (let i = 0; i < grids.length; i++) {
      const g = grids[i];
      const region = regions[i];

      // Wind data from the backend (now included in grid response)
      const windSpd = g.wind_speed != null ? g.wind_speed : 3.0;
      const windDir = g.wind_dir_deg != null ? g.wind_dir_deg : 270.0;

      // Position badge at the grid's reference origin (center of that fire's grid)
      const refLat = region.refLat;
      const refLon = region.refLon;
      if (refLat == null || refLon == null) continue;

      const dirLabel = _windDirLabel(windDir);
      const spdStr = windSpd.toFixed(1);

      const icon = L.divIcon({
        className: "grid-wind-marker",
        html: `
          <div class="grid-wind-badge">
            <span class="wind-label-speed">${spdStr}</span>
            <span class="wind-label-arrow" style="transform:rotate(${windDir - 90}deg)">→</span>
            <span class="wind-label-dir">${dirLabel}</span>
          </div>
        `,
        iconSize: [80, 22],
        iconAnchor: [40, 11],
      });

      const marker = L.marker([refLat, refLon], { icon }).addTo(STATE.map);
      STATE.bayesian.windLabels.push(marker);
    }
  }

  /**
   * Add/remove the satellite-derived grid layers (heatmap canvas, low-zoom
   * meta dots) from the map based on the Bayesian toggle AND the users-only
   * filter. Wind labels are handled separately (renderGridWindLabels).
   */
  function _applyGridLayerVisibility() {
    if (!STATE.map) return;
    const showGrid = STATE.bayesian.active && !STATE.bayesian.usersOnly;

    if (STATE.bayesian.heatmapLayer) {
      if (showGrid) {
        if (!STATE.map.hasLayer(STATE.bayesian.heatmapLayer)) {
          STATE.map.addLayer(STATE.bayesian.heatmapLayer);
        }
      } else if (STATE.map.hasLayer(STATE.bayesian.heatmapLayer)) {
        STATE.map.removeLayer(STATE.bayesian.heatmapLayer);
      }
    }

    if (STATE.bayesian.metaLayer && STATE.map) {
      if (showGrid) {
        if (!STATE.map.hasLayer(STATE.bayesian.metaLayer)) {
          STATE.map.addLayer(STATE.bayesian.metaLayer);
        }
      } else if (STATE.map.hasLayer(STATE.bayesian.metaLayer)) {
        STATE.map.removeLayer(STATE.bayesian.metaLayer);
      }
    }
  }

  /**
   * Toggle the "Users only" filter: hides the satellite/FIRMS-derived
   * Bayesian layers so the map shows just the fires reported by app users
   * (report markers + clusters).
   */
  function toggleUsersOnly(active) {
    STATE.bayesian.usersOnly = active;
    const btn = document.getElementById("users-only-toggle");
    if (btn) btn.classList.toggle("active", active);

    _applyGridLayerVisibility();

    // Wind labels: remove now, and renderGridWindLabels skips re-adding
    // them while the filter is active.
    if (active && STATE.map) {
      for (const m of STATE.bayesian.windLabels) {
        STATE.map.removeLayer(m);
      }
      STATE.bayesian.windLabels = [];
    }
  }

  /**
   * Activate/deactivate the Bayesian grid overlay.
   */
  function toggleBayesian(active) {
    STATE.bayesian.active = active;
    STATE.bayesian.panelOpen = active;  // turning the grid on shows the card

    const panel = document.getElementById("bayesian-panel");
    const btn = document.getElementById("bayesian-toggle");

    if (panel) panel.classList.toggle("hidden", !active);
    if (btn) btn.classList.toggle("active", active);
    _syncBayesianPill();  // never leave an orphan pill when the grid is off

    if (active) {
      _applyGridLayerVisibility();
      fetchBayesianState();

      // Start polling Bayesian state every 5 seconds
      STATE.bayesian.pollingInterval = setInterval(fetchBayesianState, 5000);
    } else {
      // Remove from map
      if (STATE.bayesian.heatmapLayer && STATE.map) {
        STATE.map.removeLayer(STATE.bayesian.heatmapLayer);
      }
      // Remove low-zoom meta dots too
      if (STATE.bayesian.metaLayer && STATE.map) {
        STATE.map.removeLayer(STATE.bayesian.metaLayer);
      }

      // Remove wind labels
      for (const m of STATE.bayesian.windLabels) {
        if (STATE.map) STATE.map.removeLayer(m);
      }
      STATE.bayesian.windLabels = [];

      // Stop polling
      if (STATE.bayesian.pollingInterval) {
        clearInterval(STATE.bayesian.pollingInterval);
        STATE.bayesian.pollingInterval = null;
      }

      // Remove road risk layer when Bayesian is turned off
      if (STATE.bayesian.roadRiskLayer && STATE.map) {
        STATE.map.removeLayer(STATE.bayesian.roadRiskLayer);
        STATE.bayesian.roadRiskLayer = null;
      }
      STATE.bayesian.roadRiskActive = false;
      const roadRiskToggle = document.getElementById("roadrisk-toggle");
      if (roadRiskToggle) roadRiskToggle.checked = false;
      const roadriskStatus = document.getElementById("roadrisk-status");
      if (roadriskStatus) roadriskStatus.textContent = "🛣️ idle";

      // Stop the live satellite poller too, so it doesn't keep injecting
      // evidence into fires nobody's watching anymore
      if (STATE.bayesian.satellitePollerActive) {
        const satelliteToggle = document.getElementById("satellite-poller-toggle");
        if (satelliteToggle) satelliteToggle.checked = false;
        toggleSatellitePoller(false);
      }
    }
  }

  /**
   * Show/hide only the Fire Grid control card, leaving the heatmap layer and
   * polling untouched. Used on mobile, where the fixed card would otherwise
   * cover most of the map (the Fire Grid menu button reopens it).
   */
  // User's explicit collapse choice, persisted across visits. When unset,
  // the viewport decides: mobile defaults collapsed, desktop expanded.
  const PANEL_COLLAPSE_KEY = "wf.bayesian.panelCollapsed";

  function _panelCollapsedPref() {
    try { return localStorage.getItem(PANEL_COLLAPSE_KEY); } catch (err) { return null; }
  }

  // The reopen pill shows only while the grid is on and the card is hidden.
  // Kept in one place so every path (toggle, collapse, boot) agrees on it.
  function _syncBayesianPill() {
    const pill = document.getElementById("bayesian-panel-pill");
    if (pill) {
      pill.classList.toggle("hidden", !STATE.bayesian.active || STATE.bayesian.panelOpen);
    }
  }

  function setBayesianPanelVisible(show) {
    STATE.bayesian.panelOpen = !!show;
    const panel = document.getElementById("bayesian-panel");
    if (panel) panel.classList.toggle("hidden", !show);
    _syncBayesianPill();
  }

  // Collapse/expand with persistence (header ▾ button, reopen pill, and the
  // mobile Fire Grid menu button when the grid is already active).
  function setBayesianPanelCollapsed(collapsed) {
    setBayesianPanelVisible(!collapsed);
    try { localStorage.setItem(PANEL_COLLAPSE_KEY, collapsed ? "1" : "0"); }
    catch (err) { /* private mode — in-memory only */ }
  }

  // Apply the correct panel state for the current viewport + saved preference.
  // Kept separate from toggleBayesian so turning the grid on/off never fights
  // the collapse preference.
  function applyBayesianPanelLayout() {
    if (!STATE.bayesian.active) return;
    const pref = _panelCollapsedPref();
    const collapsed = pref === null ? MOBILE_MENU_QUERY.matches : pref === "1";
    setBayesianPanelVisible(!collapsed);
  }

  // -----------------------------------------------------------------------
  // Fetch & refresh data
  // -----------------------------------------------------------------------
  async function refreshData() {
    try {
      const [reportsRes, clustersRes] = await Promise.all([
        fetch(`/api/reports${modeQuery()}`),
        fetch(`/api/clusters${modeQuery()}`),
      ]);
      const reportsData = await reportsRes.json();
      const clustersData = await clustersRes.json();

      STATE.reports = reportsData.reports || [];
      STATE.clusters = clustersData.clusters || [];

      // Update stats
      els.statReports.textContent = STATE.reports.length;
      els.statClusters.textContent = STATE.clusters.length;
      els.statPending.textContent = STATE.reports.filter((r) => r.status === "pending").length;

      renderData(STATE.reports, STATE.clusters);

      // Also refresh Bayesian state if active
      if (STATE.bayesian.active) {
        fetchBayesianState();
      }
    } catch (err) {
      console.error("Failed to refresh data:", err);
      setStatus("Connection error", "error");
    }
  }

  // -----------------------------------------------------------------------
  // Photo upload
  // -----------------------------------------------------------------------
  function setupUpload() {
    // ---- Trigger card open/close ----
    els.uploadTrigger.addEventListener("click", () => {
      els.uploadCard.classList.toggle("hidden");
      els.uploadTrigger.classList.toggle("hidden");
    });

    els.cardClose.addEventListener("click", () => {
      els.uploadCard.classList.add("hidden");
      els.uploadTrigger.classList.remove("hidden");
      resetForm();
    });

    // ---- Click preview to open file picker ----
    els.photoPreview.addEventListener("click", () => {
      els.photoInput.click();
    });

    // ---- Photo selected ----
    els.photoInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (ev) => {
        els.previewPlaceholder.classList.add("hidden");
        els.previewImage.classList.remove("hidden");
        els.previewImage.src = ev.target.result;
        els.submitBtn.disabled = false;
      };
      reader.readAsDataURL(file);

      // Set captured_at to now
      els.inputCapturedAt.value = new Date().toISOString();
    });

    // ---- Form submit ----
    els.form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (STATE.uploading) return;

      const file = els.photoInput.files[0];
      if (!file) {
        toast("Please select a photo first", "error");
        return;
      }

      // If GPS failed, check if user typed coordinates manually
      _syncManualCoordsToHidden();
      if (!els.inputLat.value || !els.inputLon.value) {
        toast("GPS position not available. Enter coordinates manually above, or enable location services.", "error");
        return;
      }

      STATE.uploading = true;
      els.submitBtn.disabled = true;
      els.progressBar.classList.remove("hidden");
      els.progressBar.classList.add("active");

      const formData = new FormData(els.form);
      formData.set("session_id", STATE.sessionId);

      try {
        const res = await fetch("/api/reports", { method: "POST", body: formData });
        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.error || "Upload failed");
        }

        // NOTE: the server no longer returns accepted:false — even a
        // "nothing" AI verdict is kept as a pending report for human review
        // (the hosted model can miss borderline fires), so every successful
        // upload follows the success path below.
        if (STATE.mode === "demo") {
          // Uploads are real citizen reports and always go to the LIVE
          // (production) store — they won't show on the demo map.
          toast("✅ Report submitted to LIVE data (switch to Live mode to see it)", "info");
        } else if (data.report?.ai_analysis?.verdict === "nothing") {
          // The AI found nothing, but the photo is kept for a human
          // moderator (the hosted model can miss borderline fires) —
          // never silently discard a real fire.
          toast("ℹ️ No fire detected by AI — photo kept for human review", "info");
        } else {
          toast("✅ Report submitted successfully!", "success");
        }
        resetForm();
        els.uploadCard.classList.add("hidden");
        els.uploadTrigger.classList.remove("hidden");

        // Refresh map data
        await refreshData();

        // Fly to the submitted report location
        if (data.report) {
          STATE.map.setView([data.report.lat, data.report.lon], 15);
        }
      } catch (err) {
        toast(err.message || "Upload failed. Check your connection.", "error");
        console.error("Upload error:", err);
      } finally {
        STATE.uploading = false;
        els.submitBtn.disabled = false;
        els.progressBar.classList.remove("active");
        els.progressBar.classList.add("hidden");
      }
    });
  }

  function resetForm() {
    els.form.reset();
    els.previewPlaceholder.classList.remove("hidden");
    els.previewImage.classList.add("hidden");
    els.previewImage.src = "";
    els.submitBtn.disabled = true;
    els.progressBar.classList.remove("active");
    els.progressBar.classList.add("hidden");
    // Clear manual coordinate fields
    if (els.inputLatManual) els.inputLatManual.value = "";
    if (els.inputLonManual) els.inputLonManual.value = "";
    // Re-populate hidden fields from GPS if available
    if (STATE.currentPos) {
      els.inputLat.value = STATE.currentPos.lat;
      els.inputLon.value = STATE.currentPos.lon;
    }
  }

  // -----------------------------------------------------------------------
  // GPS Refresh button
  // -----------------------------------------------------------------------
  function setupGpsRefresh() {
    if (!els.gpsRefreshBtn) return;

    els.gpsRefreshBtn.addEventListener("click", (e) => {
      e.stopPropagation();

      // Visual feedback: spin the icon
      els.gpsRefreshBtn.classList.add("spinning");
      els.locText.textContent = "🔄 Re-acquiring GPS…";

      // Force re-acquire by busting the cache
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          STATE.currentPos = { lat: pos.coords.latitude, lon: pos.coords.longitude };
          STATE.heading = pos.coords.heading;
          els.locText.textContent = `📍 ${pos.coords.latitude.toFixed(4)}, ${pos.coords.longitude.toFixed(4)}`;
          els.inputLat.value = pos.coords.latitude;
          els.inputLon.value = pos.coords.longitude;
          els.inputHeading.value = pos.coords.heading ?? "";
          setStatus("GPS locked", "active");
          _setManualCoordsVisible(false);

          // Clear manual fields so stale values don't linger
          if (els.inputLatManual) els.inputLatManual.value = "";
          if (els.inputLonManual) els.inputLonManual.value = "";

          // Center map on new position
          if (STATE.map) {
            STATE.map.flyTo([pos.coords.latitude, pos.coords.longitude], 14);
          }

          els.gpsRefreshBtn.classList.remove("spinning");
          toast("📍 GPS position updated", "success");
        },
        (err) => {
          const msg = _gpsErrorString(err);
          console.warn("Re-acquire error:", err.code, err.message);
          els.locText.textContent = `⚠️ ${msg}`;
          els.gpsRefreshBtn.classList.remove("spinning");
          _setManualCoordsVisible(true);
          toast("📍 " + msg, "info");
        },
        { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 } // maximumAge: 0 forces fresh reading
      );
    });
  }

  // -----------------------------------------------------------------------
  // Seed Data (Greece)
  // -----------------------------------------------------------------------
  async function seedGreeceData() {
    if (els.seedBtn.disabled) return;  // prevent double-clicks

    // Seeding is a demo action — show the demo store before/after.
    if (STATE.mode !== "demo") {
      await setMode("demo");
    }

    try {
      els.seedBtn.disabled = true;
      els.seedBtn.innerHTML = '<span class="seed-icon">⏳</span><span class="seed-text">Seeding…</span>';
      setStatus("Seeding test data…", "pending");

      const res = await fetch("/api/seed", { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Seed failed");

      toast(`🇬🇷 ${data.message} — ${data.cluster_count} clusters formed!`, "success");
      await refreshData();

      // Fly to Greece to show the clusters
      if (STATE.map && data.clusters && data.clusters.length > 0) {
        const first = data.clusters[0];
        STATE.map.flyTo([first.centroid_lat, first.centroid_lon], 9);
      } else {
        STATE.map.flyTo([38.1723, 23.7171], 8);
      }
      setStatus("Test data loaded", "active");
    } catch (err) {
      console.error("Seed error:", err);
      toast("❌ " + err.message, "error");
      setStatus("Seed failed", "error");
    } finally {
      els.seedBtn.disabled = false;
      els.seedBtn.innerHTML = '<span class="seed-icon">🇬🇷</span><span class="seed-text">Greece</span>';
    }
  }

  // -----------------------------------------------------------------------
  // Seed Data (Forest — Yosemite)
  // -----------------------------------------------------------------------
  async function seedForestData() {
    if (els.seedForestBtn.disabled) return;

    // Seeding is a demo action — show the demo store before/after.
    if (STATE.mode !== "demo") {
      await setMode("demo");
    }

    try {
      els.seedForestBtn.disabled = true;
      els.seedForestBtn.innerHTML = '<span class="seed-forest-icon">⏳</span><span class="seed-forest-text">Seeding…</span>';
      setStatus("Seeding forest data…", "pending");

      const res = await fetch("/api/seed/forest", { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Seed failed");

      toast(`🌲 ${data.message} — ${data.cluster_count} clusters formed!`, "success");
      await refreshData();

      // Fly to Yosemite
      if (STATE.map && data.clusters && data.clusters.length > 0) {
        const first = data.clusters[0];
        STATE.map.flyTo([first.centroid_lat, first.centroid_lon], 10);
      } else {
        STATE.map.flyTo([37.745, -119.593], 9);
      }
      setStatus("Forest data loaded", "active");
    } catch (err) {
      console.error("Forest seed error:", err);
      toast("❌ " + err.message, "error");
      setStatus("Seed failed", "error");
    } finally {
      els.seedForestBtn.disabled = false;
      els.seedForestBtn.innerHTML = '<span class="seed-forest-icon">🌲</span><span class="seed-forest-text">Forest</span>';
    }
  }

  // -----------------------------------------------------------------------
  // Live Demo — Progressive Triangulation Simulation
  // -----------------------------------------------------------------------

  /**
   * Start the live demo: calls the server to initialize the simulation,
   * then begins stepping through reports every ~6 seconds to simulate
   * real-time wildfire discovery and show triangulation convergence.
   */
  async function startLiveDemo() {
    if (STATE.liveDemo.active) return;

    // The live demo runs entirely in the demo store.
    if (STATE.mode !== "demo") {
      await setMode("demo");
    }

    els.liveDemoBtn.disabled = true;
    els.liveDemoBtn.innerHTML = '<span class="live-icon">⏳</span><span class="live-text">Starting…</span>';
    setStatus("Starting live demo…", "pending");

    try {
      // Reset any previous demo data first
      await fetch("/api/live-demo/reset", { method: "POST" });

      // Start the demo
      const startRes = await fetch("/api/live-demo/start", { method: "POST" });
      const startData = await startRes.json();
      if (!startRes.ok) throw new Error(startData.error || "Failed to start demo");

      STATE.liveDemo.active = true;
      STATE.liveDemo.currentStep = -1;
      STATE.liveDemo.totalSteps = startData.total_steps;
      STATE.liveDemo.fireLat = startData.fire_lat;
      STATE.liveDemo.fireLon = startData.fire_lon;

      toast("▶️ Live demo started! Reports will appear progressively.", "info");

      // Show live status panel
      els.liveStatus.classList.remove("hidden");
      els.liveTriResult.classList.add("hidden");
      els.liveProgressFill.style.width = "0%";
      els.liveStepInfo.textContent = `Step 0 / ${startData.total_steps}`;
      els.liveReportCount.textContent = "0 reports";

      // Render the true fire origin on the map
      renderTrueOrigin(startData.fire_lat, startData.fire_lon);

      // Fly to Yosemite
      if (STATE.map) {
        STATE.map.flyTo([startData.fire_lat, startData.fire_lon], 14);
      }

      // Disable seed buttons during live demo
      if (els.seedBtn) els.seedBtn.disabled = true;
      if (els.seedForestBtn) els.seedForestBtn.disabled = true;

      // Do the first step immediately
      await liveDemoStep();

      // Then set interval (every 7 seconds)
      STATE.liveDemo.stepInterval = setInterval(liveDemoStep, 7000);

      setStatus("Live demo running", "active");
    } catch (err) {
      console.error("Live demo error:", err);
      toast("❌ " + err.message, "error");
      setStatus("Demo failed", "error");
      cancelLiveDemo();
    } finally {
      els.liveDemoBtn.disabled = false;
      els.liveDemoBtn.innerHTML = '<span class="live-icon">▶️</span><span class="live-text">Live Demo</span>';
    }
  }

  /**
   * Advance the live demo by one step (called by the interval).
   */
  async function liveDemoStep() {
    if (!STATE.liveDemo.active) return;

    try {
      const stepRes = await fetch("/api/live-demo/step", { method: "POST" });
      const stepData = await stepRes.json();
      if (!stepRes.ok) throw new Error(stepData.error || "Step failed");

      STATE.liveDemo.currentStep = stepData.step;

      // Update UI
      const progress = ((stepData.step + 1) / stepData.total_steps) * 100;
      els.liveProgressFill.style.width = `${progress}%`;
      els.liveStepInfo.textContent = `Step ${stepData.step + 1} / ${stepData.total_steps}`;
      els.liveReportCount.textContent = `${stepData.total_seeded} reports`;

      // Refresh map data to show new clusters + triangulation
      await refreshData();

      // Check if we have a fire cluster with triangulation
      const fireCluster = stepData.fire_cluster;
      if (fireCluster && fireCluster.triangulation) {
        const t = fireCluster.triangulation;
        els.liveTriResult.classList.remove("hidden");
        els.liveTriCoords.textContent = `${t.fire_lat.toFixed(5)}, ${t.fire_lon.toFixed(5)}`;

        // Confidence with emoji
        const confEmoji = t.confidence === "high" ? "🟢" : t.confidence === "medium" ? "🟡" : "🔴";
        els.liveTriConfidence.textContent = `${confEmoji} ${t.confidence.toUpperCase()}`;
        els.liveTriConfidence.style.color = t.confidence === "high" ? "#22c55e" : t.confidence === "medium" ? "#eab308" : "#ef4444";
        els.liveTriUncertainty.textContent = `${t.ellipse_semi_major.toFixed(0)} × ${t.ellipse_semi_minor.toFixed(0)} m`;

        // Compute error from true fire origin
        if (STATE.liveDemo.fireLat != null && STATE.liveDemo.fireLon != null) {
          const dlat = t.fire_lat - STATE.liveDemo.fireLat;
          const dlon = t.fire_lon - STATE.liveDemo.fireLon;
          // Rough meters per degree
          const latM = dlat * 111320;
          const lonM = dlon * 111320 * Math.cos(t.fire_lat * Math.PI / 180);
          const errorM = Math.sqrt(latM * latM + lonM * lonM);
          els.liveTriError.textContent = `${errorM.toFixed(1)} m`;
          els.liveTriError.style.color = errorM < 100 ? "#22c55e" : errorM < 500 ? "#eab308" : "#ef4444";
        }
      }

      // If all steps done, wrap up
      if (stepData.all_revealed) {
        toast("✅ Live demo complete! Triangulation converged.", "success");
        setStatus("Demo complete", "active");
        stopLiveDemoInterval();
        els.liveProgressFill.style.width = "100%";

        // Fly to show the full scene
        if (STATE.map && STATE.liveDemo.fireLat && STATE.liveDemo.fireLon) {
          STATE.map.flyTo([STATE.liveDemo.fireLat, STATE.liveDemo.fireLon], 14);
        }
      }
    } catch (err) {
      console.error("Live demo step error:", err);
      // Don't stop on error, keep trying
    }
  }

  /**
   * Cancel the live demo, stop stepping, and clean up.
   */
  async function cancelLiveDemo() {
    stopLiveDemoInterval();

    if (STATE.liveDemo.active) {
      // Remove demo data from server
      try {
        await fetch("/api/live-demo/reset", { method: "POST" });
      } catch (err) {
        console.warn("Failed to reset demo:", err);
      }
    }

    STATE.liveDemo.active = false;
    STATE.liveDemo.currentStep = -1;

    // Hide live status panel
    els.liveStatus.classList.add("hidden");

    // Re-enable seed buttons
    if (els.seedBtn) els.seedBtn.disabled = false;
    if (els.seedForestBtn) els.seedForestBtn.disabled = false;

    // Refresh map to clear demo data
    await refreshData();

    // Remove true origin marker
    STATE.markers.fireOrigin.forEach((m) => {
      if (STATE.map) STATE.map.removeLayer(m);
    });
    STATE.markers.fireOrigin = [];

    toast("⏹️ Live demo stopped", "info");
  }

  function stopLiveDemoInterval() {
    if (STATE.liveDemo.stepInterval) {
      clearInterval(STATE.liveDemo.stepInterval);
      STATE.liveDemo.stepInterval = null;
    }
  }

  // -----------------------------------------------------------------------
  // Auto-refresh polling
  // -----------------------------------------------------------------------
  function startPolling(intervalMs = 15000) {
    refreshData();
    setInterval(refreshData, intervalMs);
  }

  // -----------------------------------------------------------------------
  // Mobile top-bar overflow menu
  // -----------------------------------------------------------------------

  /**
   * The secondary controls that live in the top bar on desktop but collapse
   * into the ⋯ overflow menu on narrow screens. Order matters — it defines
   * both the menu order (mobile) and the restored order (desktop).
   */
  function _mobileMenuItems() {
    return [
      document.querySelector(".admin-link"),
      document.querySelector(".contact-link"),
      document.querySelector(".privacy-link"),
      document.getElementById("bayesian-toggle"),
      document.getElementById("users-only-toggle"),
      document.getElementById("status-badge"),
    ];
  }

  // Guarded so an ancient browser without matchMedia degrades to the
  // desktop layout (all controls stay in the top bar) instead of crashing.
  const MOBILE_MENU_QUERY =
    typeof window.matchMedia === "function"
      ? window.matchMedia("(max-width: 600px)")
      : { matches: false, addEventListener() {}, addListener() {} };

  /**
   * Move the secondary controls between the top bar and the ⋯ menu based on
   * viewport width. Re-parenting is safe here: all listeners are attached to
   * the elements themselves in init(), so they keep working wherever they
   * live, and any active/disabled state travels with them.
   */
  function _layoutTopBar() {
    const menu = els.topMenu;
    const actions = document.querySelector(".top-actions");
    const searchBox = document.getElementById("search-box");
    if (!menu || !actions) return;

    const mobile = MOBILE_MENU_QUERY.matches;
    _mobileMenuItems().forEach((el) => {
      if (!el) return;
      if (mobile && el.parentNode !== menu) menu.appendChild(el);
      else if (!mobile && el.parentNode !== actions) actions.appendChild(el);
    });

    // Search: desktop keeps its slot between the brand and the actions; on
    // mobile it lives at the top of the ⋯ overflow menu so the top bar stays
    // a single row.
    if (searchBox) {
      if (mobile) {
        if (searchBox.parentNode !== menu) menu.prepend(searchBox);
      } else if (searchBox.parentNode === menu) {
        actions.parentNode.insertBefore(searchBox, actions);
      }
    }

    // Never leave a stray open menu when switching back to desktop.
    if (!mobile && menu.classList.contains("open")) {
      _closeTopMenu();
    }
  }

  function _toggleTopMenu() {
    if (!els.topMenu) return;
    const open = els.topMenu.classList.toggle("open");
    if (els.topMenuBtn) {
      els.topMenuBtn.classList.toggle("active", open);
      els.topMenuBtn.setAttribute("aria-expanded", String(open));
    }
    // The search dropdown lives inside the menu on mobile — never leave it
    // open behind a closed menu.
    if (!open) _closeSearchResults();
  }

  function _closeTopMenu() {
    if (els.topMenu) els.topMenu.classList.remove("open");
    if (els.topMenuBtn) {
      els.topMenuBtn.classList.remove("active");
      els.topMenuBtn.setAttribute("aria-expanded", "false");
    }
    _closeSearchResults();
  }

  function _setupTopBarMenu() {
    _layoutTopBar();

    // ⋯ button toggles the menu.
    if (els.topMenuBtn) {
      els.topMenuBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        _toggleTopMenu();
      });
    }

    // Picking an action closes the menu.
    if (els.topMenu) {
      els.topMenu.addEventListener("click", (e) => {
        if (e.target.closest("button, a")) _closeTopMenu();
      });
    }

    // Close on outside click or Escape.
    document.addEventListener("click", (e) => {
      if (!els.topMenu || !els.topMenu.classList.contains("open")) return;
      if (els.topMenuBtn && els.topMenuBtn.contains(e.target)) return;
      if (els.topMenu.contains(e.target)) return;
      _closeTopMenu();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") _closeTopMenu();
    });

    // Re-layout whenever the viewport crosses the mobile breakpoint.
    if (MOBILE_MENU_QUERY.addEventListener) {
      MOBILE_MENU_QUERY.addEventListener("change", (e) => {
        _layoutTopBar();
        // Only auto-collapse/expand when the user hasn't explicitly chosen
        // (the header ▾ button / reopen pill persist their preference).
        if (_panelCollapsedPref() === null && STATE.bayesian.active) {
          setBayesianPanelVisible(!e.matches);
        }
      });
    } else if (MOBILE_MENU_QUERY.addListener) {
      MOBILE_MENU_QUERY.addListener(() => {
        _layoutTopBar();
        if (_panelCollapsedPref() === null && STATE.bayesian.active) {
          setBayesianPanelVisible(!MOBILE_MENU_QUERY.matches);
        }
      });
    }
  }

  // -----------------------------------------------------------------------
  // Search (top-bar geocoding → Nominatim via /api/geocode)
  // -----------------------------------------------------------------------

  function _escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  let _searchTimer = null;
  let _searchSeq = 0; // stale-response guard (only the latest query renders)
  let _searchActiveIdx = -1;
  let _lastResults = [];

  function _searchResultIcon(res) {
    const cls = res.class || "";
    if (cls === "highway") return "🛣️";
    if (cls === "building") return "🏢";
    if (cls === "amenity" || cls === "shop" || cls === "tourism") return "⭐";
    if (cls === "boundary" || cls === "place" || cls === "city" || cls === "town" || cls === "village" || cls === "hamlet") return "📍";
    return "🔍";
  }

  function _searchZoom(res) {
    const t = (res.type || "") + "|" + (res.class || "");
    if (t.includes("building")) return 17;
    if (t.includes("road") || t.includes("highway") || t.includes("street")) return 16;
    if (t.includes("city") || t.includes("town") || t.includes("village") || t.includes("hamlet")) return 12;
    if (t.includes("county") || t.includes("state") || t.includes("region")) return 10;
    if (t.includes("country")) return 6;
    return 13;
  }

  function _renderSearchResults(results) {
    const box = document.getElementById("search-results");
    if (!box) return;
    box.innerHTML = "";
    _lastResults = results || [];
    _searchActiveIdx = -1;

    if (!_lastResults.length) {
      const row = document.createElement("div");
      row.className = "search-result-empty";
      row.textContent = "No places found";
      box.appendChild(row);
      box.classList.add("open");
      return;
    }

    _lastResults.forEach((res, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "search-result";
      btn.dataset.index = String(i);

      const icon = document.createElement("span");
      icon.className = "search-result-icon";
      icon.textContent = _searchResultIcon(res);

      const text = document.createElement("span");
      const name = document.createElement("div");
      name.className = "search-result-name";
      name.textContent = res.name || res.label;
      const sub = document.createElement("div");
      sub.className = "search-result-sub";
      // The label's leading name is already shown bold — render the rest.
      sub.textContent = (res.label || "").split(",").slice(1).join(",").trim();
      text.appendChild(name);
      text.appendChild(sub);

      btn.appendChild(icon);
      btn.appendChild(text);
      box.appendChild(btn);
    });
    box.classList.add("open");
  }

  function _closeSearchResults() {
    _searchSeq++; // discard any in-flight geocode response (stale-guard)
    const box = document.getElementById("search-results");
    if (box) box.classList.remove("open");
    const input = document.getElementById("search-input");
    if (input) input.setAttribute("aria-expanded", "false");
    _searchActiveIdx = -1;
  }

  function _selectSearchResult(res) {
    if (!STATE.map || !res) return;
    STATE.map.flyTo([res.lat, res.lon], _searchZoom(res), { duration: 0.8 });

    const html =
      `<div style="font-size:13px;font-weight:600;margin-bottom:2px">${_escapeHtml(res.name || res.label)}</div>` +
      `<div style="font-size:11px;color:var(--text-muted)">📍 ${res.lat.toFixed(4)}, ${res.lon.toFixed(4)}</div>`;
    L.popup({ maxWidth: 280 })
      .setLatLng([res.lat, res.lon])
      .setContent(html)
      .openOn(STATE.map);

    _closeSearchResults();
    const input = document.getElementById("search-input");
    if (input) input.blur();
  }

  function _runSearch() {
    const input = document.getElementById("search-input");
    const q = (input.value || "").trim();
    if (q.length < 2) {
      _closeSearchResults();
      return;
    }
    const seq = ++_searchSeq;
    fetch(`/api/geocode?q=${encodeURIComponent(q)}&limit=5`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("geocode " + r.status))))
      .then((data) => {
        if (seq !== _searchSeq) return; // stale
        const box = document.getElementById("search-results");
        if (!box) return;
        if (data.degraded) {
          box.innerHTML = '<div class="search-result-error">Search temporarily unavailable</div>';
          box.classList.add("open");
          return;
        }
        _renderSearchResults(data.results || []);
        const input2 = document.getElementById("search-input");
        if (input2) input2.setAttribute("aria-expanded", "true");
      })
      .catch(() => {
        if (seq !== _searchSeq) return;
        const box = document.getElementById("search-results");
        if (box) {
          box.innerHTML = '<div class="search-result-error">Search unavailable — try again later</div>';
          box.classList.add("open");
        }
      });
  }

  function _setupSearch() {
    const input = document.getElementById("search-input");
    const box = document.getElementById("search-results");
    if (!input || !box) return;

    input.addEventListener("input", () => {
      clearTimeout(_searchTimer);
      _searchTimer = setTimeout(_runSearch, 280);
    });

    input.addEventListener("keydown", (e) => {
      const items = box.querySelectorAll(".search-result");
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        if (!items.length) return;
        const dir = e.key === "ArrowDown" ? 1 : -1;
        _searchActiveIdx = (_searchActiveIdx + dir + items.length) % items.length;
        items.forEach((el, i) => el.classList.toggle("active", i === _searchActiveIdx));
        items[_searchActiveIdx].scrollIntoView({ block: "nearest" });
      } else if (e.key === "Enter") {
        const idx = _searchActiveIdx >= 0 ? _searchActiveIdx : 0;
        if (_lastResults[idx]) {
          e.preventDefault();
          _selectSearchResult(_lastResults[idx]);
        }
      } else if (e.key === "Escape") {
        _closeSearchResults();
        input.blur();
      }
    });

    box.addEventListener("click", (e) => {
      const btn = e.target.closest(".search-result");
      if (btn) _selectSearchResult(_lastResults[Number(btn.dataset.index)]);
    });

    // Close the dropdown when clicking anywhere outside the search box.
    document.addEventListener("click", (e) => {
      if (e.target.closest("#search-box")) return;
      _closeSearchResults();
    });
  }

  // -----------------------------------------------------------------------
  // Init
  // -----------------------------------------------------------------------
  // Show the Admin button only when this browser already proved the admin
  // key (the server sets an HttpOnly cookie after /admin?key=...).
  // Fail-closed: the link starts hidden (class in the HTML) and is only
  // revealed here, so a fetch failure keeps it hidden for everyone.
  async function _syncAdminVisibility() {
    try {
      const res = await fetch("/api/admin/status");
      const st = await res.json();
      const link = document.querySelector(".admin-link");
      if (link) link.classList.toggle("admin-link--hidden", !(st && st.authed));
    } catch (err) {
      console.warn("Admin status check failed — keeping Admin button hidden:", err);
    }
  }

  async function init() {
    STATE.sessionId = getSessionId();
    els.inputSession.value = STATE.sessionId;

    _setupTopBarMenu();
    _setupSearch();

    // Show the Admin button only if this browser holds the admin cookie.
    _syncAdminVisibility();

    setStatus("Acquiring GPS…", "pending");

    // Get GPS
    acquirePosition();

    // Watch device heading
    watchHeading();

    // Init map
    initMap();

    // Init Bayesian heatmap layer and turn the Fire Grid on by default —
    // the overlay + panel + 5s polling start immediately, no click needed.
    initBayesianLayer();
    toggleBayesian(true);
    // Default: mobile collapses the control card (heatmap stays on), desktop
    // keeps it open — unless the user has chosen otherwise; that choice
    // persists across visits.
    applyBayesianPanelLayout();

    // Setup upload UI
    setupUpload();

    // Load data
    await refreshData();

    // Start polling
    startPolling();

    // Set captured_at default
    els.inputCapturedAt.value = new Date().toISOString();

    // GPS refresh button
    setupGpsRefresh();

    // Manual coordinate inputs: sync to hidden fields on keystroke
    if (els.inputLatManual) {
      els.inputLatManual.addEventListener("input", _syncManualCoordsToHidden);
    }
    if (els.inputLonManual) {
      els.inputLonManual.addEventListener("input", _syncManualCoordsToHidden);
    }

    // --- Refetch Bayesian grids + road risk when the map moves, so only
    // fires in the current viewport are loaded (global FIRMS = 1000+ grids).
    // Debounced so a pan that fires several moveend events doesn't hammer
    // the backend with overlapping requests.
    if (STATE.map) {
      let _moveTimer = null;
      STATE.map.on("moveend", () => {
        if (!STATE.bayesian.active) return;
        clearTimeout(_moveTimer);
        _moveTimer = setTimeout(() => {
          fetchBayesianState();
          if (STATE.bayesian.roadRiskActive) fetchRoadRisk();
        }, 250);
      });
    }

    // --- Bayesian Toggle ---
    const bayesianToggle = document.getElementById("bayesian-toggle");
    if (bayesianToggle) {
      bayesianToggle.addEventListener("click", () => {
        // Mobile: once the grid is on, the button toggles just the control
        // card so the map stays visible. Desktop keeps the full on/off.
        if (MOBILE_MENU_QUERY.matches && STATE.bayesian.active) {
          setBayesianPanelCollapsed(STATE.bayesian.panelOpen);
        } else {
          toggleBayesian(!STATE.bayesian.active);
        }
      });
    }

    // --- Bayesian Collapse (header ▾ button + reopen pill) ---
    const bayesianCollapseBtn = document.getElementById("bayesian-collapse-btn");
    if (bayesianCollapseBtn) {
      bayesianCollapseBtn.addEventListener("click", () => setBayesianPanelCollapsed(true));
    }
    const bayesianPanelPill = document.getElementById("bayesian-panel-pill");
    if (bayesianPanelPill) {
      bayesianPanelPill.addEventListener("click", () => setBayesianPanelCollapsed(false));
    }

    // --- Users-only filter ---
    const usersOnlyToggle = document.getElementById("users-only-toggle");
    if (usersOnlyToggle) {
      usersOnlyToggle.addEventListener("click", () => {
        toggleUsersOnly(!STATE.bayesian.usersOnly);
      });
    }

    // --- Bayesian Layer Controls ---
    const heatmapToggle = document.getElementById("bayesian-heatmap-toggle");
    if (heatmapToggle) {
      heatmapToggle.addEventListener("change", (e) => {
        STATE.bayesian.showHeatmap = e.target.checked;
        if (STATE.bayesian.heatmapLayer) {
          STATE.bayesian.heatmapLayer.setOptions({ showHeatmap: e.target.checked });
        }
      });
    }

    const contourToggle = document.getElementById("bayesian-contour-toggle");
    if (contourToggle) {
      contourToggle.addEventListener("change", (e) => {
        STATE.bayesian.showContour = e.target.checked;
        if (STATE.bayesian.heatmapLayer) {
          STATE.bayesian.heatmapLayer.setOptions({ showContour: e.target.checked });
        }
        // Refetch so the server skips contour extraction entirely.
        fetchBayesianState();
      });
    }

    const thresholdSlider = document.getElementById("bayesian-threshold");
    if (thresholdSlider) {
      thresholdSlider.addEventListener("input", (e) => {
        STATE.bayesian.threshold = parseFloat(e.target.value);
        if (STATE.bayesian.heatmapLayer) {
          STATE.bayesian.heatmapLayer.setOptions({ threshold: STATE.bayesian.threshold });
        }
      });
    }

    // FIRMS fetch button
    const firmsFetchBtn = document.getElementById("firms-fetch-btn");
    if (firmsFetchBtn) {
      firmsFetchBtn.addEventListener("click", fetchFirmsData);
    }

    // FIRMS poller toggle
    const firmsPollerToggle = document.getElementById("firms-poller-toggle");
    if (firmsPollerToggle) {
      firmsPollerToggle.addEventListener("change", (e) => {
        toggleFirmsPoller(e.target.checked);
      });
    }

    // FIRMS Live is the default — sync/start the poller on boot.
    syncFirmsPollerDefault();

    // --- Road Risk Toggle ---
    const roadRiskToggle = document.getElementById("roadrisk-toggle");
    if (roadRiskToggle) {
      roadRiskToggle.addEventListener("change", (e) => {
        toggleRoadRisk(e.target.checked);
      });
    }
  }

  // -----------------------------------------------------------------------
  // Historic Demo — 2020 Creek Fire Replay
  // -----------------------------------------------------------------------

  /**
   * Start the Bayesian historic demo (2020 Creek Fire replay).
   * Activates the Bayesian grid, fetches initial state, and begins stepping
   * through the scenario, showing satellite hotspots, fire perimeters, and
   * wind data evolving over time.
   */
  async function startHistoricDemo() {
    if (STATE.historicDemo.active) return;

    // The historic replay runs entirely in the demo store.
    if (STATE.mode !== "demo") {
      await setMode("demo");
    }

    const btn = document.getElementById("historic-demo-btn");
    if (btn) btn.disabled = true;

    setStatus("Starting Creek Fire replay…", "pending");

    try {
      // Activate Bayesian heatmap without starting the 5-second polling
      // (the demo step function handles refreshes at 6-second intervals)
      if (!STATE.bayesian.active) {
        STATE.bayesian.active = true;
        const panel = document.getElementById("bayesian-panel");
        const btn = document.getElementById("bayesian-toggle");
        if (panel) panel.classList.remove("hidden");
        if (btn) btn.classList.add("active");
        if (STATE.bayesian.heatmapLayer && STATE.map) {
          STATE.map.addLayer(STATE.bayesian.heatmapLayer);
        }
      }

      // Reset any previous demo
      await fetch("/api/bayesian-demo/reset", { method: "POST" });

      // Start the demo
      const startRes = await fetch("/api/bayesian-demo/start", { method: "POST" });
      const startData = await startRes.json();
      if (!startRes.ok) throw new Error(startData.error || "Failed to start historic demo");

      STATE.historicDemo.active = true;
      STATE.historicDemo.currentStep = 0;
      STATE.historicDemo.totalSteps = startData.total_steps;
      STATE.historicDemo.fireLat = startData.fire_lat;
      STATE.historicDemo.fireLon = startData.fire_lon;

      toast("🌲 Creek Fire replay started! Watch the fire evolve with satellite data.", "info");

      // Show historic demo panel
      const panel = document.getElementById("historic-panel");
      if (panel) panel.classList.remove("hidden");

      // Update panel with step 0 data
      updateHistoricPanel(startData);

      // Render initial hotspot markers
      renderHistoricHotspots(startData.hotspots || []);

      // Render initial perimeter
      renderHistoricPerimeter(startData.perimeter || []);

      // Render origin marker
      renderHistoricOrigin(startData.fire_lat, startData.fire_lon);

      // Fly to the fire
      if (STATE.map) {
        STATE.map.flyTo([startData.fire_lat, startData.fire_lon], 12);
      }

      // Fetch initial Bayesian state
      await fetchBayesianState();

      // Disable seed & demo buttons during replay
      if (els.seedBtn) els.seedBtn.disabled = true;
      if (els.seedForestBtn) els.seedForestBtn.disabled = true;
      if (els.liveDemoBtn) els.liveDemoBtn.disabled = true;

      // Start stepping (every 6 seconds for dramatic pacing)
      STATE.historicDemo.stepInterval = setInterval(historicDemoStep, 6000);

      setStatus("Creek Fire replay running", "active");
    } catch (err) {
      console.error("Historic demo error:", err);
      toast("❌ " + err.message, "error");
      cancelHistoricDemo();
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  /**
   * Advance the historic demo by one step.
   */
  async function historicDemoStep() {
    if (!STATE.historicDemo.active) return;

    try {
      const stepRes = await fetch("/api/bayesian-demo/step", { method: "POST" });
      const stepData = await stepRes.json();
      if (!stepRes.ok) throw new Error(stepData.error || "Step failed");

      STATE.historicDemo.currentStep = stepData.step;

      // Update panel
      updateHistoricPanel(stepData);

      // Update hotspot markers
      renderHistoricHotspots(stepData.hotspots || []);

      // Update perimeter
      renderHistoricPerimeter(stepData.perimeter || []);

      // Fetch Bayesian state to update heatmap & contour
      await fetchBayesianState();

      // If complete, wrap up
      if (stepData.all_complete || stepData.status === "complete") {
        toast("✅ Creek Fire replay complete! 380,000 acres burned.", "success");
        setStatus("Replay complete", "active");
        stopHistoricDemoInterval();

        const fill = document.getElementById("historic-progress-fill");
        if (fill) fill.style.width = "100%";

        // Fly out to see full extent
        if (STATE.map && STATE.historicDemo.fireLat) {
          STATE.map.flyTo([STATE.historicDemo.fireLat - 0.05, STATE.historicDemo.fireLon - 0.05], 10);
        }
      }
    } catch (err) {
      console.error("Historic demo step error:", err);
    }
  }

  /**
   * Cancel the historic demo and clean up.
   */
  async function cancelHistoricDemo() {
    stopHistoricDemoInterval();

    if (STATE.historicDemo.active) {
      try {
        await fetch("/api/bayesian-demo/reset", { method: "POST" });
      } catch (err) {
        console.warn("Failed to reset historic demo:", err);
      }
    }

    STATE.historicDemo.currentStep = -1;

    // Hide panel
    const panel = document.getElementById("historic-panel");
    if (panel) panel.classList.add("hidden");

    // Clear hotspot markers
    clearHistoricOverlays();

    // Re-enable buttons
    if (els.seedBtn) els.seedBtn.disabled = false;
    if (els.seedForestBtn) els.seedForestBtn.disabled = false;
    if (els.liveDemoBtn) els.liveDemoBtn.disabled = false;

    // Reset the DEMO Bayesian grid (the historic replay lives in the demo
    // registry, so always target demo regardless of the current display mode)
    await fetch("/api/bayesian/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "demo" }),
    });
    await fetchBayesianState();

    // Mark inactive AFTER all cleanup is done (prevents race with re-start)
    STATE.historicDemo.active = false;

    toast("⏹️ Creek Fire replay stopped", "info");
  }

  function stopHistoricDemoInterval() {
    if (STATE.historicDemo.stepInterval) {
      clearInterval(STATE.historicDemo.stepInterval);
      STATE.historicDemo.stepInterval = null;
    }
  }

  /**
   * Update the historic demo panel with current step data.
   */
  function updateHistoricPanel(data) {
    const id = (el) => document.getElementById(el);

    // Step label
    const stepLabel = id("historic-step-label");
    if (stepLabel) {
      stepLabel.textContent = data.step_label || `Step ${data.step}`;
    }

    // Description
    const desc = id("historic-description");
    if (desc) {
      desc.textContent = data.step_description || "";
    }

    // Wind: speed + direction arrow
    const windEl = id("historic-wind");
    if (windEl && data.wind_speed !== undefined) {
      const dirDeg = data.wind_dir_deg || 0;
      // Map compass direction to arrow symbol
      const dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
      const idx = Math.round(((dirDeg + 360) % 360) / 45) % 8;
      const dirLabel = dirs[idx];
      windEl.innerHTML = `${data.wind_speed.toFixed(1)} m/s <span class="wind-arrow" style="transform:rotate(${dirDeg - 90}deg)">→</span> ${dirLabel}`;
    }

    // Hotspot count
    const hsCount = id("historic-hotspot-count");
    if (hsCount && data.hotspots) {
      hsCount.textContent = data.hotspots.length;
    }

    // Fire area from stats (hectares)
    const areaEl = id("historic-fire-area");
    if (areaEl && data.statistics) {
      const areaHa = data.statistics.area_ha_p_above_0_10 || 0;
      areaEl.textContent = `${areaHa} ha`;
    }

    // Progress bar
    const fill = id("historic-progress-fill");
    if (fill && data.total_steps) {
      const pct = ((data.step + 1) / data.total_steps) * 100;
      fill.style.width = `${pct}%`;
    }
  }

  /**
   * Render satellite hotspot markers on the map.
   */
  function renderHistoricHotspots(hotspots) {
    // Clear previous hotspot markers
    STATE.historicDemo.hotspotMarkers.forEach((m) => {
      if (STATE.map) STATE.map.removeLayer(m);
    });
    STATE.historicDemo.hotspotMarkers = [];

    if (!STATE.map || !hotspots || hotspots.length === 0) return;

    const icon = L.divIcon({
      className: "satellite-hotspot-marker",
      html: '<div class="satellite-hotspot-dot"></div>',
      iconSize: [12, 12],
      iconAnchor: [6, 6],
    });

    hotspots.forEach((hs) => {
      const marker = L.marker([hs[0], hs[1]], { icon }).addTo(STATE.map);

      // Popup with VIIRS info
      marker.bindPopup(`
        <div class="popup-title">🛰️ VIIRS Hotspot</div>
        <div><span class="popup-label">Location:</span> ${hs[0].toFixed(4)}, ${hs[1].toFixed(4)}</div>
        <div><span class="popup-label">Confidence:</span> <span style="color:#22c55e;font-weight:600">HIGH</span></div>
        <div><span class="popup-label">Source:</span> VIIRS 375m · Fire Thermal Anomaly</div>
      `, { maxWidth: 260, className: "hotspot-popup" });

      STATE.historicDemo.hotspotMarkers.push(marker);
    });

    STATE.historicDemo.hotspotCount = hotspots.length;
  }

  /**
   * Render the true fire perimeter as a dashed polygon.
   */
  function renderHistoricPerimeter(perimeterCoords) {
    // Clear previous perimeter
    if (STATE.historicDemo.perimeterLayer && STATE.map) {
      STATE.map.removeLayer(STATE.historicDemo.perimeterLayer);
      STATE.historicDemo.perimeterLayer = null;
    }

    if (!STATE.map || !perimeterCoords || perimeterCoords.length < 3) return;

    // Build Leaflet polygon
    const polygon = L.polygon(perimeterCoords, {
      color: "#fbbf24",
      weight: 2,
      opacity: 0.7,
      fillColor: "rgba(251, 191, 36, 0.06)",
      fillOpacity: 0.3,
      dashArray: "6, 6",
      className: "historic-perimeter",
    }).addTo(STATE.map);

    // Popup with area info
    const areaKm2 = (perimeterCoords.length * 10).toFixed(0);  // rough visual estimate
    polygon.bindPopup(`
      <div class="popup-title">🔥 Fire Perimeter (True)</div>
      <div><span class="popup-label">Fire:</span> 2020 Creek Fire</div>
      <div><span class="popup-label">Source:</span> CalFire historical data</div>
      <div style="margin-top:4px;font-size:10px;color:var(--text-muted)">Dashed line shows the approximate true fire boundary at this stage.</div>
    `, { maxWidth: 260 });

    STATE.historicDemo.perimeterLayer = polygon;
  }

  /**
   * Render the fire origin marker.
   */
  function renderHistoricOrigin(lat, lon) {
    // Clear previous
    if (STATE.historicDemo.originMarker && STATE.map) {
      STATE.map.removeLayer(STATE.historicDemo.originMarker);
      STATE.historicDemo.originMarker = null;
    }

    if (!STATE.map || lat == null || lon == null) return;

    const icon = L.divIcon({
      className: "historic-origin-marker",
      html: `<div class="historic-origin">
        <div class="historic-origin-icon"></div>
        <div class="historic-origin-label">ORIGIN</div>
      </div>`,
      iconSize: [14, 24],
      iconAnchor: [7, 14],
    });

    const marker = L.marker([lat, lon], { icon }).addTo(STATE.map);
    marker.bindPopup(`
      <div class="popup-title">🔥 Creek Fire Origin</div>
      <div><span class="popup-label">Location:</span> ${lat.toFixed(5)}, ${lon.toFixed(5)}</div>
      <div><span class="popup-label">Started:</span> September 4, 2020</div>
      <div><span class="popup-label">Cause:</span> Power line ignition (suspected)</div>
      <div style="margin-top:4px;font-size:11px;color:var(--text-muted)">Near Shaver Lake, Sierra National Forest</div>
    `);

    STATE.historicDemo.originMarker = marker;
  }

  /**
   * Clear all historic demo overlays from the map.
   */
  function clearHistoricOverlays() {
    // Hotspot markers
    STATE.historicDemo.hotspotMarkers.forEach((m) => {
      if (STATE.map) STATE.map.removeLayer(m);
    });
    STATE.historicDemo.hotspotMarkers = [];

    // Perimeter
    if (STATE.historicDemo.perimeterLayer && STATE.map) {
      STATE.map.removeLayer(STATE.historicDemo.perimeterLayer);
      STATE.historicDemo.perimeterLayer = null;
    }

    // Origin marker
    if (STATE.historicDemo.originMarker && STATE.map) {
      STATE.map.removeLayer(STATE.historicDemo.originMarker);
      STATE.historicDemo.originMarker = null;
    }
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