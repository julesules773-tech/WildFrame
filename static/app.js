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
    dataPollingInterval: null, // reports/clusters poll handle (pause on hidden tab)
    dataPollingMs: 15000,
    reports: [],
    clusters: [],
    _dataSig: 0,     // signature of last renderData payload
    _dataFullSig: 0, // includes zoom-level style change (hull vs dot)
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
      pollMs: 5000,     // adaptive: 5s while fires change, backs off to 30s when idle
      windLabels: [],   // per-grid wind label markers added to map
      metaDots: [],     // low-zoom (detail=meta) intensity dots
      metaLayer: null,  // Leaflet layerGroup holding those dots
      metaSig: "",      // signature of last-rendered dots (skip DOM rebuild when unchanged)
      fullSig: "",      // signature of last full-detail render (skip redraw when unchanged)
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
    photoStatus: $("#photo-status"),
    fireGateNotice: $("#fire-gate-notice"),

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
  // Admin deep-link — focus a specific fire grid on the map
  // -----------------------------------------------------------------------

  /**
   * Parse the admin-dashboard deep-link query params
   * (?grid=…&lat=…&lon=…&max_p=…&wind_speed=…&wind_dir_deg=…
   * &ffmc=…&dmc=…&isi=…&mf=…). Returns null when the URL doesn't request
   * a specific fire. Function declarations hoist, so this is safe to call
   * at module scope below.
   */
  function _deepLinkedFire() {
    const params = new URLSearchParams(window.location.search);
    // Whitelist the grid id — it is interpolated into popup HTML, so a
    // crafted ?grid=<img onerror=...> link must never survive into markup.
    // Real grid ids are server-generated ("grid-<alnum>").
    const grid = (params.get("grid") || "").replace(/[^a-zA-Z0-9_-]/g, "");
    const lat = parseFloat(params.get("lat"));
    const lon = parseFloat(params.get("lon"));
    if (!grid || !isFinite(lat) || !isFinite(lon)) return null;
    // Range-check so a hand-edited URL can't fly the map to nonsense.
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return null;
    const num = (k) => {
      const v = parseFloat(params.get(k));
      return isFinite(v) ? v : null;
    };
    return {
      id: grid,
      lat,
      lon,
      max_p: num("max_p"),
      wind_speed: num("wind_speed"),
      wind_dir_deg: num("wind_dir_deg"),
      ffmc: num("ffmc"),
      dmc: num("dmc"),
      isi: num("isi"),
      moisture_factor: num("mf"),
    };
  }

  // Parsed once at boot — GPS must not override the deep-link destination.
  const _DEEP_LINK_FIRE = _deepLinkedFire();

  /**
   * Fly the map to the fire grid named in the URL (opened from the admin
   * dashboard's "Live fires" list) and open a popup with its data:
   * probability, wind, EFFIS fuel moisture, and nearby report sources.
   */
  function focusDeepLinkedFire() {
    const f = _DEEP_LINK_FIRE;
    if (!f || !STATE.map) return;

    // The Fire Grid overlay + control panel are on by default; if the
    // visitor had turned them off, the deep link still needs the heatmap.
    if (!STATE.bayesian.active) toggleBayesian(true);

    const p = Math.min(1, Math.max(0, f.max_p || 0));
    const pct = Math.round(p * 100);
    const color = _metaDotColor(p);

    // Confirmed reports within ~3 km (matches the FIRMS corroboration
    // radius) tell us the fire's sources and how many people reported it.
    const nearby = (STATE.reports || []).filter((r) =>
      r.status === "confirmed" && _latLonKm(f.lat, f.lon, r.lat, r.lon) <= 3
    );
    const sources = [...new Set(nearby.map((r) => (r.source_type || "citizen")))];

    const windRow = f.wind_speed != null && f.wind_dir_deg != null
      ? `<div><span class="popup-label">Wind:</span> ${f.wind_speed.toFixed(1)} m/s ${_windDirLabel(f.wind_dir_deg)}</div>`
      : `<div><span class="popup-label">Wind:</span> <span style="color:var(--text-muted)">N/A</span></div>`;

    const hasFuel = f.ffmc != null && f.ffmc > 0;
    const fuelRow = hasFuel
      ? `<div><span class="popup-label">Fuel:</span> FFMC ${Math.round(f.ffmc)} · DMC ${f.dmc != null ? Math.round(f.dmc) : "—"} · ISI ${f.isi != null ? Math.round(f.isi) : "—"} <span style="color:var(--text-muted)">(×${f.moisture_factor != null ? f.moisture_factor.toFixed(2) : "1.00"})</span></div>`
      : `<div><span class="popup-label">Fuel:</span> <span style="color:var(--text-muted)">Outside EFFIS coverage</span></div>`;

    // f.id is sanitized to [a-zA-Z0-9_-] in _deepLinkedFire, so it is safe
    // to interpolate into popup markup.
    const html = `
      <div class="popup-title">🔥 Active Fire</div>
      <div style="margin-top:4px;font-size:11px;color:var(--text-muted)">${f.id}</div>
      <div style="margin-top:4px"><span class="popup-label">Probability:</span>
        <span style="color:${color};font-weight:700">${pct}%</span></div>
      <div><span class="popup-label">Source:</span> ${sources.length ? sources.join(", ") : "Satellite (FIRMS)"}</div>
      <div><span class="popup-label">Reports:</span> ${nearby.length} confirmed nearby</div>
      ${windRow}
      ${fuelRow}
      <div style="margin-top:4px;font-size:11px;color:var(--text-muted)">📍 ${f.lat.toFixed(4)}, ${f.lon.toFixed(4)}</div>
      <div style="margin-top:6px;font-size:11px;color:var(--accent);font-weight:600">Opened from admin dashboard</div>
    `;

    // Fly in so the full heatmap + reports render (above the meta-dot LOD).
    // The map may boot on another continent, so open the popup only once
    // the camera has arrived — an off-screen popup would otherwise sit at
    // a nonsense projected position during the flight.
    const targetZoom = Math.max(STATE.map.getZoom(), 10);
    let popupOpened = false;
    const openPopup = () => {
      if (popupOpened) return;
      popupOpened = true;
      L.popup({ maxWidth: 300, autoPan: false })
        .setLatLng([f.lat, f.lon])
        .setContent(html)
        .openOn(STATE.map);
    };
    STATE.map.flyTo([f.lat, f.lon], targetZoom, { duration: 0.8 });
    STATE.map.once("moveend", openPopup);
    // Fallback: flyTo to the current position would never fire moveend.
    setTimeout(openPopup, 1500);

    // The map only re-fetches grid state on its 5s poll; nudge it so the
    // full-detail heatmap appears as soon as the flyTo lands.
    setTimeout(() => {
      if (STATE.bayesian.active) fetchBayesianState();
    }, 900);
  }

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
    // Success confirmations (e.g. a submitted report) deserve longer on
    // screen than transient info/error notes.
    const duration = type === "success" ? 6500 : 4000;
    setTimeout(() => {
      div.classList.add("out");
      setTimeout(() => div.remove(), 300);
    }, duration);
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

        // Center map on position (unless a deep link already aims the map
        // at a specific fire from the admin dashboard).
        if (STATE.map && !_DEEP_LINK_FIRE) {
          STATE.map.setView([pos.coords.latitude, pos.coords.longitude], 14);
        }
      },
      (err) => {
        const msg = _gpsErrorString(err);
        console.warn("Geolocation error:", err.code, err.message);
        els.locText.textContent = `⚠️ ${msg}`;
        setStatus("GPS failed", "error");
        _setManualCoordsVisible(true);
        // Default to Yosemite National Park (forest demo location) unless
        // a deep link already aims the map at a specific fire.
        if (STATE.map && !_DEEP_LINK_FIRE) {
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
    const darkLayer = L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
      subdomains: "abcd",
      maxZoom: 19,
    }).addTo(STATE.map);

    // Esri World Imagery — free satellite tiles, no API key
    const satelliteLayer = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      attribution: '&copy; <a href="https://www.esri.com/">Esri</a> &mdash; Esri, DeLorme, NAVTEQ',
      maxZoom: 18,
    });

    // CartoDB Positron — light basemap (same provider as the dark tiles)
    const lightLayer = L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
      subdomains: "abcd",
      maxZoom: 19,
    });

    // Layer switcher (bottom-left, above the cluster severity legend)
    L.control.layers({ "Dark": darkLayer, "Light": lightLayer, "Satellite": satelliteLayer }, null, { position: "bottomleft" }).addTo(STATE.map);

    // Add a locate button
    L.control.locate({
      position: "topleft",
      strings: { title: "Show my location" },
      locateOptions: { enableHighAccuracy: true },
    }).addTo(STATE.map);

    // Re-render clusters when zoom changes (convex hull vs dot).
    // force=true: the zoom-level style changed but data didn't, so the
    // signature alone would skip the redraw.  The zoom is folded into
    // _dataFullSig, but zoomend always fires so we force a redraw to
    // catch the hull↔dot transition at CLUSTER_POLYGON_ZOOM.
    STATE.map.on("zoomend", () => {
      renderData(STATE.reports, STATE.clusters, /*force=*/ true);
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

  /**
   * Fast hash of the reports + clusters payload.  Catches any change
   * that would affect rendering: report count, status, location, AI
   * analysis, satellite confirmation, cluster membership, and contour
   * geometry.  Intentionally excludes fields that don't change rendering
   * (e.g. created_at granularity beyond seconds) to avoid false misses.
   */
  function _reportsSig(reports, clusters) {
    let h = 0;
    // Reports: id + status + lat/lon (rounded) + ai verdict + photo presence
    for (const r of reports) {
      h = (h * 31 + r.id) | 0;
      h = (h * 31 + r.status.charCodeAt(0)) | 0;
      h = (h * 31 + ((r.lat * 1e4) | 0)) | 0;
      h = (h * 31 + ((r.lon * 1e4) | 0)) | 0;
      const ai = r.ai_analysis;
      if (ai && ai.verdict) {
        h = (h * 31 + ai.verdict.charCodeAt(0)) | 0;
        h = (h * 31 + ((ai.confidence * 1000) | 0)) | 0;
      } else {
        h = (h * 31) | 0;
      }
      h = (h * 31 + (r.photo_url ? 1 : 0)) | 0;
      h = (h * 31 + (r.satellite_confirmation ? 1 : 0)) | 0;
    }
    // Clusters: count + centroid + report_ids length + triangulation presence
    for (const c of clusters) {
      h = (h * 31 + c.count) | 0;
      h = (h * 31 + ((c.centroid_lat * 1e4) | 0)) | 0;
      h = (h * 31 + ((c.centroid_lon * 1e4) | 0)) | 0;
      h = (h * 31 + (c.report_ids ? c.report_ids.length : 0)) | 0;
      h = (h * 31 + (c.triangulation ? 1 : 0)) | 0;
      if (c.points) {
        for (const p of c.points) {
          h = (h * 31 + ((p[0] * 1e4) | 0)) | 0;
          h = (h * 31 + ((p[1] * 1e4) | 0)) | 0;
        }
      }
    }
    return h | 0;
  }

  function renderData(reports, clusters, force) {
    // --- Change detection: skip the expensive DOM rebuild when the
    //     payload is identical to the last render.  The common case
    //     (15 s poll, nothing changed) used to destroy and recreate
    //     every marker, pulse ring, cluster polygon, and triangulation
    //     overlay — hundreds of Leaflet DOM operations for zero benefit.
    const sig = _reportsSig(reports, clusters);
    // Full sig includes zoom level (cluster hull vs dot changes at
    // CLUSTER_POLYGON_ZOOM) so a zoom-only change still redraws.
    const fullSig = (sig * 31 + STATE.map.getZoom()) | 0;
    if (!force && sig === STATE._dataSig && fullSig === STATE._dataFullSig) {
      return;  // nothing changed
    }
    STATE._dataSig = sig;
    STATE._dataFullSig = fullSig;

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

    // Photos are gated on approval: the server strips photo_url from
    // unapproved reports, and the UI shows a placeholder instead so a
    // pending report never leaks its photo on the public map.
    const photoBlock = r.status === "confirmed" && r.photo_url
      ? `<div style="margin-top:6px"><img src="${r.photo_url}" style="width:100%;max-width:180px;border-radius:6px;" alt="Report photo" /></div>`
      : r.status === "pending"
        ? `<div style="margin-top:6px;font-size:11px;color:var(--text-muted)">📷 Photo visible after approval</div>`
        : "";

    marker.bindPopup(`
        <div class="popup-title">📸 Fire Report</div>
        <div><span class="popup-label">Status:</span> ${r.status}</div>
        <div><span class="popup-label">Location:</span> ${r.lat.toFixed(4)}, ${r.lon.toFixed(4)}</div>
        <div><span class="popup-label">${headingInfo}</span></div>
        <div><span class="popup-label">Reported:</span> ${new Date(r.captured_at).toLocaleString()}</div>
        <div><span class="popup-label">Source:</span> ${r.source_type || "citizen"}</div>
        ${satLine}
        ${photoBlock}
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

        // Zoomed-out clusters are just dots — clicking one should fly into
        // the fire and show the full heatmap blob (not just the cluster
        // popup). Target CLUSTER_POLYGON_ZOOM so the hull + triangulation
        // also appear; aim at the triangulated origin when we have one.
        marker.on("click", () => {
          const t = c.triangulation;
          _flyToFire(
            t ? t.fire_lat : c.centroid_lat,
            t ? t.fire_lon : c.centroid_lon,
            Math.max(STATE.map.getZoom(), CLUSTER_POLYGON_ZOOM)
          );
        });
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
   * Separable centered sliding-window max (dilate, isMax=true) or min
   * (erode, isMax=false) filter on a grayscale ImageData buffer, in place.
   *
   * A monotonic deque gives O(w*h) per pass regardless of radius; the
   * window is CENTERED on each pixel (max/min over [i-r, i+r]) via a
   * forward + backward half-window (radius r+1) per axis, so morphology is
   * symmetric — a causal-only window would erode away the trailing edge of
   * every narrow blob. The accumulated heatmap field is gray (R=G=B), so
   * only the R channel is filtered and mirrored back to G/B. Used for the
   * display-level merge: dilate then erode (a grayscale closing) bridges
   * ~1 km gaps between hot cells so nearby fires fuse into one blob.
   */
  function _grayDilateErode(data, w, h, r, isMax) {
    const win = r + 1;
    const better = isMax ? (a, b) => a >= b : (a, b) => a <= b;
    const tmp = new Uint8ClampedArray(w * h);
    const f = new Uint8ClampedArray(Math.max(w, h));
    const b = new Uint8ClampedArray(Math.max(w, h));
    const deque = new Int32Array(Math.max(w, h));

    // Horizontal pass: centered window over x in [x-r, x+r]
    for (let y = 0; y < h; y++) {
      const base = y * w * 4;
      let head = 0, tail = 0;
      for (let x = 0; x < w; x++) {
        const v = data[base + x * 4];
        while (head < tail && better(v, data[base + deque[tail - 1] * 4])) tail--;
        deque[tail++] = x;
        while (deque[head] <= x - win) head++;
        f[x] = data[base + deque[head] * 4];
      }
      head = 0; tail = 0;
      for (let x = w - 1; x >= 0; x--) {
        const v = data[base + x * 4];
        while (head < tail && better(v, data[base + deque[tail - 1] * 4])) tail--;
        deque[tail++] = x;
        while (deque[head] >= x + win) head++;
        b[x] = data[base + deque[head] * 4];
      }
      for (let x = 0; x < w; x++) {
        tmp[y * w + x] = better(f[x], b[x]) ? f[x] : b[x];
      }
    }

    // Vertical pass: centered window over y in [y-r, y+r]
    for (let x = 0; x < w; x++) {
      let head = 0, tail = 0;
      for (let y = 0; y < h; y++) {
        const idx = y * w + x;
        const v = tmp[idx];
        while (head < tail && better(v, tmp[deque[tail - 1] * w + x])) tail--;
        deque[tail++] = y;
        while (deque[head] <= y - win) head++;
        f[y] = tmp[deque[head] * w + x];
      }
      head = 0; tail = 0;
      for (let y = h - 1; y >= 0; y--) {
        const idx = y * w + x;
        const v = tmp[idx];
        while (head < tail && better(v, tmp[deque[tail - 1] * w + x])) tail--;
        deque[tail++] = y;
        while (deque[head] >= y + win) head++;
        b[y] = tmp[deque[head] * w + x];
      }
      for (let y = 0; y < h; y++) {
        data[(y * w + x) * 4] = better(f[y], b[y]) ? f[y] : b[y];
      }
    }

    // Mirror R back to G/B (gray field).
    for (let i = 0; i < w * h; i++) {
      const v = data[i * 4];
      data[i * 4 + 1] = v;
      data[i * 4 + 2] = v;
    }
  }

  /**
   * Bounding-box-limited grayscale closing (dilate then erode) on a
   * canvas-sized RGBA ImageData buffer, in place.
   *
   * The full-canvas morphology is O(w·h) per pass, which at the 4x
   * downscale cap is ~79k px × 4 sweeps × 2 calls — several tens of ms
   * per redraw. But the hot pixels (the only cells above the visibility
   * floor) usually cover a small fraction of the canvas: fires are
   * scattered dots on black. This computes the bbox of the non-zero
   * (R channel) pixels, crops that region padded by the kernel radius,
   * runs the identical separable morphology on the crop, and writes it
   * back. The pad (2×r) is generous enough that the closing at every
   * original bbox pixel has its full [±r, ±r] context inside the crop,
   * so the result is bit-identical to running on the whole canvas.
   */
  /**
   * Chaikin corner-cutting (1974) on a contour polyline — display-only
   * rounding of the sharp ~90° corners that marching squares leaves on
   * the fire boundary (the probability field is a per-cell step function
   * on a 100 m lattice, so contour segments run along cell edges). Each
   * vertex is replaced by two points at 25%/75% along its adjacent
   * edges; repeated iterations round the corners further. Closed rings
   * are smoothed cyclically (the seam corner is cut too); open chains (a
   * fire touching the grid edge) keep their endpoints. The grid
   * probabilities are untouched — this only changes what gets stroked.
   */
  function _chaikinSmooth(seg, iterations) {
    const it = iterations || 2;
    if (seg.length < 3) return seg;
    const closed = seg.length >= 4 &&
      Math.abs(seg[0][0] - seg[seg.length - 1][0]) < 1e-6 &&
      Math.abs(seg[0][1] - seg[seg.length - 1][1]) < 1e-6;
    let pts = closed ? seg.slice(0, -1) : seg.slice();
    if (closed && pts.length < 3) return seg;
    for (let n = 0; n < it; n++) {
      const out = [];
      if (closed) {
        for (let i = 0; i < pts.length; i++) {
          const a = pts[i];
          const b = pts[(i + 1) % pts.length];
          out.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]]);
          out.push([0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]]);
        }
      } else {
        out.push(pts[0]);
        for (let i = 0; i < pts.length - 1; i++) {
          const a = pts[i];
          const b = pts[i + 1];
          out.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]]);
          out.push([0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]]);
        }
        out.push(pts[pts.length - 1]);
      }
      pts = out;
    }
    if (closed) pts.push(pts[0]);
    return pts;
  }

  function _grayDilateErodeBBox(data, w, h, r, isMax) {
    // 1. Find the bounding box of non-zero (hot) pixels.
    let x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (let y = 0; y < h; y++) {
      const base = y * w * 4;
      for (let x = 0; x < w; x++) {
        if (data[base + x * 4] > 0) {
          if (x < x0) x0 = x;
          if (x > x1) x1 = x;
          if (y < y0) y0 = y;
          if (y > y1) y1 = y;
        }
      }
    }
    if (x1 < 0) return; // empty field — nothing to merge

    // 2. Pad the crop by 2× the kernel radius so every original pixel's
    //    centered window stays inside the crop.
    const pad = 2 * r;
    const X0 = Math.max(0, x0 - pad), Y0 = Math.max(0, y0 - pad);
    const X1 = Math.min(w - 1, x1 + pad), Y1 = Math.min(h - 1, y1 + pad);
    const cw = X1 - X0 + 1, ch = Y1 - Y0 + 1;

    // If the crop covers most of the canvas the copy costs more than it
    // saves — just run the plain full-canvas filter.
    if (cw * ch >= w * h * 0.75) {
      _grayDilateErode(data, w, h, r, isMax);
      return;
    }

    // 3. Copy the crop (R channel only) into a compact 1-channel buffer.
    const tmp = new Uint8ClampedArray(cw * ch);
    for (let y = 0; y < ch; y++) {
      const sBase = (Y0 + y) * w * 4 + X0 * 4;
      const dBase = y * cw;
      for (let x = 0; x < cw; x++) {
        tmp[dBase + x] = data[sBase + x * 4];
      }
    }

    // 4. Run the same separable morphology on the compact crop.
    const win = r + 1;
    const better = isMax ? (a, b) => a >= b : (a, b) => a <= b;
    const f = new Uint8ClampedArray(Math.max(cw, ch));
    const bb = new Uint8ClampedArray(Math.max(cw, ch));
    const deque = new Int32Array(Math.max(cw, ch));
    const out = new Uint8ClampedArray(cw * ch);

    // Horizontal pass
    for (let y = 0; y < ch; y++) {
      const base = y * cw;
      let head = 0, tail = 0;
      for (let x = 0; x < cw; x++) {
        const v = tmp[base + x];
        while (head < tail && better(v, tmp[base + deque[tail - 1]])) tail--;
        deque[tail++] = x;
        while (deque[head] <= x - win) head++;
        f[x] = tmp[base + deque[head]];
      }
      head = 0; tail = 0;
      for (let x = cw - 1; x >= 0; x--) {
        const v = tmp[base + x];
        while (head < tail && better(v, tmp[base + deque[tail - 1]])) tail--;
        deque[tail++] = x;
        while (deque[head] >= x + win) head++;
        bb[x] = tmp[base + deque[head]];
      }
      for (let x = 0; x < cw; x++) {
        out[base + x] = better(f[x], bb[x]) ? f[x] : bb[x];
      }
    }

    // Vertical pass
    for (let x = 0; x < cw; x++) {
      let head = 0, tail = 0;
      for (let y = 0; y < ch; y++) {
        const idx = y * cw + x;
        const v = out[idx];
        while (head < tail && better(v, out[deque[tail - 1] * cw + x])) tail--;
        deque[tail++] = y;
        while (deque[head] <= y - win) head++;
        f[y] = out[deque[head] * cw + x];
      }
      head = 0; tail = 0;
      for (let y = ch - 1; y >= 0; y--) {
        const idx = y * cw + x;
        const v = out[idx];
        while (head < tail && better(v, out[deque[tail - 1] * cw + x])) tail--;
        deque[tail++] = y;
        while (deque[head] >= y + win) head++;
        bb[y] = out[deque[head] * cw + x];
      }
      for (let y = 0; y < ch; y++) {
        tmp[y * cw + x] = better(f[y], bb[y]) ? f[y] : bb[y];
      }
    }

    // 5. Write the crop back (R channel), mirroring G/B like the full
    //    filter does so the blurred/LUT passes see an identical field.
    for (let y = 0; y < ch; y++) {
      const dBase = (Y0 + y) * w * 4 + X0 * 4;
      const sBase = y * cw;
      for (let x = 0; x < cw; x++) {
        const v = tmp[sBase + x];
        const idx = dBase + x * 4;
        data[idx] = v;
        data[idx + 1] = v;
        data[idx + 2] = v;
      }
    }
  }

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

      // Append to the OVERLAY pane, not the map root. The overlay pane
      // stacks above the tiles but below the shadow/marker/popup panes, so
      // the heatmap glow sits over the map but never covers the wind
      // badges, report markers or popups. Drawing uses
      // map.latLngToLayerPoint() (pane-relative): Leaflet shifts the pane
      // itself during pan/zoom (translate3d of _mapPanePos), so pane-local
      // coordinates stay correct without double-offsetting. The canvas is
      // hidden during drag/zoom and redrawn on moveend/zoomend.
      map.getPane('overlayPane').appendChild(this._container);

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
      // Technique: sample the cells onto a SMALL offscreen canvas, then
      // upscale that field onto the map canvas with bilinear smoothing.
      // Sampling each cell at ~1/3 of its pixel size averages neighboring
      // cells together, and the upscale interpolates between them — so
      // adjacent cells MERGE into one continuous gradient instead of a
      // visible grid of boxes. Compositing is per-channel max ("lighten")
      // so overlapping cells never sum into a brighter color than their
      // certainty deserves. The color scale is ABSOLUTE (probability 0..1,
      // matching the "Max prob" stat): yellow is always low probability,
      // red is always ≥0.6, no matter what else is on screen or at any
      // zoom level.
      const anyCells = regions.some((r) => r.cells && r.cells.length > 0);
      if (this._showHeatmap && anyCells) {
        // Pick a downscale so low-res cells are ~3px — small enough that
        // the upscale interpolation reads as one smooth field, but big
        // enough that the low-res canvas stays cheap to remap/blur.
        const maxCellPx = Math.max(5, ...regions
          .filter((r) => r.cells && r.cells.length > 0)
          .map((r) => this._cellSizePxFor(r)));
        // Cap the upscale at 4x: the upscale's smoothing kernel spreads
        // a hard edge over ~1.5–2 source pixels, so at 8–16x the whole
        // field melted into a blurry mush. 4x keeps the edge transition
        // to ~6–8 full-res px and the low-res canvas at ~350×225 (79k
        // pixels) — still cheap for the LUT remap loop.
        const downscale = Math.min(4, Math.max(2, maxCellPx / 3));
        const loW = Math.max(2, Math.ceil(canvasW / downscale));
        const loH = Math.max(2, Math.ceil(canvasH / downscale));

        // Pass 1: accumulate a grayscale intensity field via per-cell max.
        const accumCanvas = this._getOffscreenCanvas(loW, loH);
        const accumCtx = accumCanvas.getContext('2d');
        accumCtx.clearRect(0, 0, loW, loH);
        accumCtx.globalCompositeOperation = 'lighten';

        for (const region of regions) {
          if (!region.cells || region.cells.length === 0) continue;
          const cellSizeLo = Math.max(1.2, this._cellSizePxFor(region) / downscale);
          // Radius just past one cell so neighboring cells overlap and
          // fuse into a single shape. The plateau always extends past the
          // neighbor's center (1 cell) so there is no dip between adjacent
          // cells; the fade band targets a constant ~6px on the full-res
          // canvas — soft when zoomed out (melty, hides cell dots) but
          // crisp when zoomed in (the old wide fade grew with the cell
          // size, so zoomed-in blobs melted into a wide blurry mush).
          const radius = cellSizeLo * 1.55;
          const fadeLo = Math.min(radius * 0.5, Math.max(0.6, 6 / downscale));
          const plateau = Math.min(0.97, Math.max(1 - fadeLo / radius, (1.05 * cellSizeLo) / radius));

          for (const cell of region.cells) {
            const p = cell.p;
            if (p < this._threshold) continue;

            const pt = map.latLngToLayerPoint([cell.lat, cell.lon]);
            const x = pt.x / downscale;
            const y = pt.y / downscale;
            if (x < -radius || x > loW + radius ||
                y < -radius || y > loH + radius) continue;

            // Absolute scale: brightness = the cell's true probability.
            // (Grid probabilities live in 0..1 — bayesian_filter clamps at
            // PROB_MAX=0.9999 — so 1.0 is the natural "hot" end.)
            const t = Math.min(1, Math.max(0, p));
            const v = Math.round(255 * t);
            // Flat-topped falloff: intensity holds at the cell's value over
            // most of the radius and only fades near the edge.
            const grd = accumCtx.createRadialGradient(x, y, 0, x, y, radius);
            grd.addColorStop(0, `rgba(${v},${v},${v},1)`);
            grd.addColorStop(plateau, `rgba(${v},${v},${v},1)`);
            grd.addColorStop(1, 'rgba(0,0,0,1)');
            accumCtx.fillStyle = grd;
            accumCtx.beginPath();
            accumCtx.arc(x, y, radius, 0, Math.PI * 2);
            accumCtx.fill();
          }
        }
        accumCtx.globalCompositeOperation = 'source-over';

        // Display-level merge — the same ~1 km logic as the contour layer:
        // a grayscale CLOSING (dilate then erode) on the accumulated field
        // with a ~5-cell kernel. Dilation bridges the dark gap between hot
        // cells whose fires are within ~1 km of each other, so they fuse
        // into one connected blob instead of rendering as separate islands;
        // the erosion then restores the outer boundary so every fire doesn't
        // grow ~500 m bigger. Display-only: the grid probabilities are
        // untouched, so "Max prob" and the absolute color scale stay honest.
        // 5 cells of lo-res px = the ~1 km merge (cells are ~200 m). The
        // old 20px cap silently shrank this to ~300 m once the downscale
        // cap dropped to 4x; the deque filter is O(w·h) regardless of
        // kernel size, so a wide window is free.
        const cellLo = Math.max(1.2, maxCellPx / downscale);
        const kernelR = Math.max(2, Math.min(64, Math.round(5 * cellLo)));
        const grayImg = accumCtx.getImageData(0, 0, loW, loH);
        // Bbox-limited morphology: the full-canvas sweep was ~45ms of the
        // ~57ms redraw on dense viewports even though hot pixels cover a
        // fraction of the canvas. The crop is bit-identical to the full
        // pass (2×r padding keeps every window inside the crop).
        _grayDilateErodeBBox(grayImg.data, loW, loH, kernelR, true);   // dilate — bridge the gaps
        _grayDilateErodeBBox(grayImg.data, loW, loH, kernelR, false);  // erode — restore the boundary
        accumCtx.putImageData(grayImg, 0, 0);

        // Pass 2: soften blob edges at low resolution (cheap) so the
        // upscaled field is melty rather than grainy.
        const blurCanvas = this._blurCanvas || (this._blurCanvas = document.createElement('canvas'));
        blurCanvas.width = loW;
        blurCanvas.height = loH;
        const blurCtx = blurCanvas.getContext('2d');
        blurCtx.clearRect(0, 0, loW, loH);
        // Fixed 0.3px blur at low resolution: it hides per-cell ringing
        // and softens the upscale just enough to stay melty, but never
        // grows with the cell size — the old 1.5px fixed blur (and the
        // adaptive variant) smeared the whole field by up to ~10px after
        // the upscale.
        const blurR = 0.3;
        blurCtx.filter = `blur(${blurR.toFixed(2)}px)`;
        blurCtx.drawImage(accumCanvas, 0, 0);
        blurCtx.filter = 'none';

        // Pass 3: remap accumulated intensity -> lava color per pixel.
        // Pure array reads from the precomputed LUT — no per-pixel
        // function calls or branchy Math. Runs at low resolution, so it's
        // up to ~100x cheaper than remapping the full-size canvas.
        const imgData = blurCtx.getImageData(0, 0, loW, loH);
        const data = imgData.data;
        const lut = this._ensureLavaLut();
        const lutR = lut.r, lutG = lut.g, lutB = lut.b, lutA = lut.a;
        for (let i = 0; i < data.length; i += 4) {
          const v = data[i]; // gray channel (max of cells, not summed overlap)
          data[i] = lutR[v];
          data[i + 1] = lutG[v];
          data[i + 2] = lutB[v];
          data[i + 3] = lutA[v];
        }
        blurCtx.putImageData(imgData, 0, 0);

        // Pass 4: upscale the low-res colored field to the map canvas with
        // bilinear smoothing — this interpolation is what turns the
        // per-cell samples into one continuous, box-free heatmap.
        ctx.save();
        ctx.globalAlpha = 0.92;
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(blurCanvas, 0, 0, canvasW, canvasH);
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
          const pt0 = map.latLngToLayerPoint(seg[0]);
          ctx.moveTo(pt0.x, pt0.y);
          for (let k = 1; k < seg.length; k++) {
            const pt = map.latLngToLayerPoint(seg[k]);
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
          const pt0 = map.latLngToLayerPoint(seg[0]);
          ctx.moveTo(pt0.x, pt0.y);
          for (let k = 1; k < seg.length; k++) {
            const pt = map.latLngToLayerPoint(seg[k]);
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
     * yellow→orange→red→ember gradient — the same color always means the
     * same probability, independent of what else is on screen:
     *   yellow      = low probability  (< 0.3)
     *   orange      = moderate         (0.3 – 0.6)
     *   warm red    = high             (0.6 – 0.85)
     *   bright red  = very high        (0.85 – 0.92)
     *   ember       = certain          (≥ 0.92)
     *
     * The top tier was raised from 0.85 → 0.92 and the deepest tone changed
     * from near-black crimson to a brighter coral "ember": with the FIRMS
     * fleet saturated at max_p ≥ 0.85, the old ramp rendered the whole map
     * as a wall of dark maroon. Now the saturated zone differentiates
     * (0.85–0.92 = bright red, ≥0.92 = ember) and the hottest color is
     * vivid instead of ominous.
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
        // Orange → warm red (high certainty) — softer than before: never
        // quite reaches pure red, and the alpha stays lower so blobs read
        // less heavy.
        const s = (t - 0.6) / 0.25;
        r = 255;
        g = Math.round(60 - 45 * s);    // 60 → 15
        b = 0;
        a = 0.75 + 0.07 * s;            // 0.75 → 0.82
      } else if (t < 0.92) {
        // Warm red → bright red (very high certainty) — new tier: the old
        // ramp lumped everything ≥ 0.85 into one dark crimson.
        const s = (t - 0.85) / 0.07;
        r = 255;
        g = Math.round(15 - 10 * s);    // 15 → 5
        b = 0;
        a = 0.82 + 0.03 * s;            // 0.82 → 0.85
      } else {
        // Bright red → ember (certain): a vivid, lighter coral-red instead
        // of the old near-black crimson, so the hottest cells glow rather
        // than go dark.
        const s = (t - 0.92) / 0.08;
        r = 255;
        g = Math.round(5 + 25 * s);     // 5 → 30
        b = Math.round(15 + 25 * s);    // 15 → 40
        a = 0.85 + 0.03 * s;            // 0.85 → 0.88
      }
      return { r, g, b, a: Math.min(0.95, a) };
    },

    // Precomputed RGBA lookup tables for the per-pixel remap, built once
    // lazily from _lavaColor. The hot loop in Pass 3 then does four array
    // reads per pixel instead of a branchy function call with Math per
    // pixel — the old path was the biggest single jank source when
    // zooming/panning over dense fire regions.
    _lavaLut: null,
    _ensureLavaLut: function () {
      if (this._lavaLut) return this._lavaLut;
      const r = new Uint8Array(256);
      const g = new Uint8Array(256);
      const b = new Uint8Array(256);
      const a = new Uint8Array(256);
      for (let v = 0; v < 256; v++) {
        const c = this._lavaColor(v / 255);
        r[v] = c.r;
        g[v] = c.g;
        b[v] = c.b;
        // The old loop forced intensity < 0.03 transparent; fold that
        // into the table so the hot loop needs no branch at all.
        a[v] = v < 8 ? 0 : Math.round(c.a * 255);
      }
      this._lavaLut = { r, g, b, a };
      return this._lavaLut;
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

  // A grid must reach this peak probability before it counts as a *visible*
  // fire for the wind badge / low-zoom dots. The server exports any cell
  // above 0.02 (cheap payload), but a grid that decayed to max_p 0.03 paints
  // only a near-invisible yellow fleck — yet still arrives with wind data,
  // which produced orphaned badges/dots over "no fire" ground. Production
  // grids are bimodal: real fires sit ≥ 0.3, decayed ghosts ≤ 0.03, so 0.1
  // cleanly separates them (matches the legend's "low" band boundary).
  const MIN_FIRE_VISIBLE_P = 0.1;

  /**
   * Cheap change detector for the full-detail grid payload. The worker
   * checkpoints grids on ~15s buckets and the export cache serves the same
   * state between changes, so most 5s polls return identical data. Hash
   * per-grid (id, cell count, state version, sampled cells, wind) — a few
   * thousand integer ops, far cheaper than re-rendering ~100k cells.
   */
  function _gridSignature(grids) {
    let h = 0;
    for (const g of grids) {
      const st = g.state || {};
      const cells = st.cells || [];
      h = (h * 31 + g.id.length + cells.length + ((st.last_predict_time || 0) | 0)) | 0;
      // Grid shape is part of the signature too: a fire can grow/shrink
      // its footprint without the cell count or version changing.
      h = (h * 31 + ((st.nx || 0) | 0) + ((st.ny || 0) | 0) + ((st.count || 0) | 0)) | 0;
      // Hash EVERY cell on small grids (a localized hot-spot shift must
      // never be missed); stride only on big payloads (100k+ cells) where
      // sampling + the count/shape above still catches any real change.
      if (cells.length < 512) {
        for (const c of cells) {
          h = (h * 31 + ((c.p * 10000) | 0) + ((c.lat * 10000) | 0) + ((c.lon * 10000) | 0)) | 0;
        }
      } else {
        const step = Math.max(1, (cells.length / 64) | 0);
        for (let i = 0; i < cells.length; i += step) {
          const c = cells[i];
          h = (h * 31 + ((c.p * 10000) | 0) + ((c.lat * 10000) | 0) + ((c.lon * 10000) | 0)) | 0;
        }
      }
      if (g.wind_speed != null) h = (h * 31 + ((g.wind_speed * 10) | 0)) | 0;
      if (g.wind_dir_deg != null) h = (h * 31 + (g.wind_dir_deg | 0)) | 0;
    }
    return h | 0;
  }

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
      // Level 0.3 = the heatmap's fire-edge threshold (legend: orange starts
      // at 0.3) — a 0.6 contour only rings the hot core and renders as
      // scattered segments. Same level the road-risk feature uses.
      const contour = STATE.bayesian.showContour ? 0.3 : 0;
      const countryParam = location.pathname === "/map/poland" ? "&country=pl" : "";
      const url = `/api/bayesian/state?threshold=${threshold}&contour=${contour}&mode=${STATE.mode}&bbox=${encodeURIComponent(bbox)}&detail=${meta ? "meta" : "full"}${countryParam}`;

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
        // Meta payloads are tiny (~3KB) — keep the cadence fast here so
        // low-zoom dots stay current; backoff only applies to the heavy
        // full-detail path.
        STATE.bayesian.pollMs = 5000;
        return;
      }

      // Full detail — show any overflow dots: fires beyond the server's
      // full+coarse serialization budget (the strongest N ship as blobs,
      // the rest as cheap clickable dots) so a dense viewport never
      // renders with holes. Cleared automatically when empty.
      renderMetaDots(data.overflow || []);

      // Skip the entire render pass when the payload is unchanged since
      // the last poll (the common case between worker checkpoints).
      // Rebuilding ~100k cells, wind badges and the road-risk fetch on an
      // identical 5s poll was a large part of the zoom/pan jank. Also
      // back off the poll interval: when nothing on the map changed, the
      // next poll happens later (up to 30s), so an idle browser stops
      // hammering the server with identical 300KB payloads. Any change
      // resets the cadence back to 5s.
      // Overflow dots are part of the payload too — fold them into the
      // change detection so a fire drifting in/out of the overflow tier
      // re-renders instead of showing stale dots.
      let oh = 0;
      for (const d of data.overflow || []) {
        oh = (oh * 31 + d.id.length + ((d.max_p * 10000) | 0)) | 0;
      }
      const sig = (_gridSignature(grids) * 31 + oh) | 0;
      if (sig === STATE.bayesian.fullSig) {
        STATE.bayesian.pollMs = Math.min((STATE.bayesian.pollMs || 5000) * 2, 30000);
        return;
      }
      STATE.bayesian.fullSig = sig;
      STATE.bayesian.pollMs = 5000;
      STATE.bayesian.rawGrids = grids;  // for simulate panel button

      // Each grid is one physically separate fire (its own cluster), with
      // its own cell size / reference origin. Build one "region" per grid
      // for the heatmap layer to render in a single pass. Contours are
      // corner-rounded here (once per payload) so the render loop strokes
      // the smoothed polylines on every redraw without re-smoothing.
      const regions = grids.map((g) => {
        const st = g.state || {};
        return {
          cells: st.cells || [],
          contour: (g.contour || []).map((seg) => _chaikinSmooth(seg, 2)),
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
      // Transient error — back off so a flapping endpoint doesn't get
      // hammered on a 5s loop; next successful change resets the cadence.
      STATE.bayesian.pollMs = Math.min((STATE.bayesian.pollMs || 5000) * 2, 30000);
    }
  }

  /**
   * Self-rescheduling Bayesian poll loop. Instead of a fixed 5s interval,
   * each tick reschedules itself at STATE.bayesian.pollMs — which
   * fetchBayesianState backs off (up to 30s) when the payload is
   * unchanged and resets to 5s when a fire changes. Idle tabs therefore
   * stop requesting identical ~300KB payloads, cutting server + client
   * load dramatically; the map still snaps to fresh data the moment
   * anything moves.
   */
  async function _bayesianPollTick() {
    if (!STATE.bayesian.active || !STATE.bayesian.heatmapLayer) return;
    await fetchBayesianState();
    // Re-arm AFTER the fetch so a backoff decided by this poll applies to
    // the next delay (otherwise the cadence lags one tick behind).
    STATE.bayesian.pollingInterval = setTimeout(
      _bayesianPollTick,
      STATE.bayesian.pollMs || 5000
    );
  }

  /**
   * Combine per-grid statistics (one fire cluster per grid) into a single
   * summary for the stats panel: max probability across all fires, and
   * summed area/cell counts.
   */
  function _aggregateBayesianStats(grids) {
    if (!grids.length) return null;
    let maxP = 0;
    let burnedAreaSum = 0;   // p > 0.25 — high-confidence fire zone
    let riskAreaSum = 0;     // p > 0.05 — broad risk / spread zone
    let cellsSum = 0;
    for (const g of grids) {
      const s = g.statistics || {};
      if ((s.max_p || 0) > maxP) maxP = s.max_p;
      burnedAreaSum += s.area_ha_p_above_0_25 || 0;
      riskAreaSum += s.area_ha_p_above_0_05 || 0;
      cellsSum += s.cells_p_above_0_10 || 0;
    }
    return {
      max_p: maxP,
      area_ha_burned: burnedAreaSum,
      area_ha_risk: riskAreaSum,
      cells_p_above_0_10: cellsSum,
    };
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
      burned: document.getElementById("bayesian-stat-burned"),
      risk: document.getElementById("bayesian-stat-risk"),
      cells: document.getElementById("bayesian-stat-cells"),
    };

    if (els.maxP) els.maxP.textContent = s.max_p?.toFixed(3) || "—";
    if (els.burned) els.burned.textContent = `${s.area_ha_burned || 0} ha`;
    if (els.risk) els.risk.textContent = `${s.area_ha_risk || 0} ha`;
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
      // Both the slider AND the visibility floor gate a dot: a decayed
      // grid (max_p ~0.03) must not render a clickable "fire" dot — the
      // same rule the wind badge uses, so low-zoom dots never outlive
      // the fire they represent.
      if (p < threshold || p < MIN_FIRE_VISIBLE_P) continue;
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
  /**
   * Fly the map to a fire and nudge the full-detail grid fetch so the
   * heatmap blob renders as soon as the flyTo lands. The poll loop alone
   * could wait up to the backed-off interval (5-30s), so the explicit
   * 900ms nudge (just past the 0.8s flyTo) is what makes the blob appear
   * "immediately" instead of on the next poll.
   */
  function _flyToFire(lat, lon, targetZoom) {
    if (!STATE.map) return;
    STATE.map.flyTo([lat, lon], targetZoom, { duration: 0.8 });
    setTimeout(() => {
      if (STATE.bayesian.active) fetchBayesianState();
    }, 900);
  }

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

    // No real weather for this fire yet (backend sends null) → honest N/A,
    // never the old fake "3.0 m/s West".
    const windRow = d.wind_speed != null && d.wind_dir_deg != null
      ? `<div><span class="popup-label">Wind:</span> ${d.wind_speed.toFixed(1)} m/s ${_windDirLabel(d.wind_dir_deg)}</div>`
      : `<div><span class="popup-label">Wind:</span> <span style="color:var(--text-muted)">N/A</span></div>`;

    const html = `
      <div class="popup-title">🔥 Active Fire</div>
      <div style="margin-top:4px"><span class="popup-label">Probability:</span>
        <span style="color:${color};font-weight:700">${pct}%</span></div>
      <div><span class="popup-label">Source:</span> ${sources.length ? sources.join(", ") : "Satellite (FIRMS)"}</div>
      <div><span class="popup-label">Reports:</span> ${nearby.length} confirmed nearby</div>
      ${windRow}
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
    _flyToFire(d.lat, d.lon, Math.max(STATE.map.getZoom(), 10));
  }

  function _metaDotColor(p) {
    // Same absolute stops as the heatmap lava ramp (0.92 / 0.85 / 0.6 / 0.3)
    // so the low-zoom dots and the full heatmap agree on what a color means.
    if (p >= 0.92) return "#ff1e28";  // ember
    if (p >= 0.85) return "#ff0f00";  // bright red
    if (p >= 0.6) return "#ff3c00";   // warm red
    if (p >= 0.3) return "#ff8c00";   // orange
    return "#ffc828";                  // amber
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

      // No visible fire → no badge. A decayed grid (FIRMS stopped
      // corroborating) still arrives in the payload with wind data but no
      // cells above the render threshold — the heatmap shows nothing, so
      // an orphaned wind badge would linger over empty ground.
      if (!region.cells || region.cells.length === 0) continue;

      // Peak probability floor: a grid that decayed to a few near-invisible
      // cells still ships cells above the export floor, but isn't a fire
      // worth a badge. Tie the badge to the same visibility the heatmap
      // color scale implies (legend: "low" < 0.3) so a faded fire loses
      // its wind badge when its blob fades.
      const maxP = (g.statistics || {}).max_p;
      if (maxP == null || maxP < MIN_FIRE_VISIBLE_P) continue;

      // No real weather for this grid yet → no badge (never fake a value).
      if (g.wind_speed == null || g.wind_dir_deg == null) continue;
      const windSpd = g.wind_speed;
      const windDir = g.wind_dir_deg;

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

      // Start the adaptive Bayesian poll loop (5s while active, backs off
      // to 30s when the payload is unchanged).
      _bayesianPollTick();
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
        clearTimeout(STATE.bayesian.pollingInterval);
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
  // -----------------------------------------------------------------------
  // Client-side fire gate (MobileNetV3 via TF.js) — instant UX feedback
  // only. It NEVER blocks or discards: a low-probability photo gets a
  // soft-reject notice with a submit-anyway path. The server-side scan
  // and auto-approval stay authoritative.
  // Hoisted to IIFE scope: setupUpload, resetForm and the submit flow all
  // touch this state, so it must not stay trapped inside setupUpload.
  // -----------------------------------------------------------------------
  const FIRE_GATE = { ready: false, model: null, minProb: 0.4, loading: false, warned: false, lastProb: null, pending: null };

  /**
   * Lazily inject the TF.js runtime (<script> tag) if it isn't loaded yet.
   * Returns immediately when already present.  The 1.4 MB download only
   * happens on first upload-card open — visitors who never upload save the
   * bandwidth and main-thread parse cost.
   */
  let _tfLoadPromise = null;
  function _ensureTfJs() {
    if (window.tf) return Promise.resolve();
    if (_tfLoadPromise) return _tfLoadPromise;
    _tfLoadPromise = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = "/vendor/tf.min.js?v=20260814";
      s.onload = resolve;
      s.onerror = () => reject(new Error("Failed to load TF.js runtime"));
      document.head.appendChild(s);
    });
    return _tfLoadPromise;
  }

  async function _loadFireGate() {
      if (FIRE_GATE.loading || FIRE_GATE.ready) return;
      FIRE_GATE.loading = true;
      try {
        // Load TF.js runtime on demand (1.4 MB) — saves bandwidth for
        // every visitor who never opens the upload card.
        await _ensureTfJs();
        const cfg = await fetch("/model/gate/config.json").then((r) => r.json());
        FIRE_GATE.minProb = cfg.min_prob ?? 0.4;
        // Graph model format: MobileNetV3 contains TFOpLambda ops that the
        // layers format can't deserialize ('Unknown layer: TFOpLambda').
        FIRE_GATE.model = await window.tf.loadGraphModel("/model/gate/model.json");
        FIRE_GATE.ready = true;
        console.info(`[fire-gate] ready (min_prob=${FIRE_GATE.minProb})`);
      } catch (err) {
        // The gate is a progressive enhancement — the app works without it.
        console.warn("[fire-gate] unavailable, proceeding without gate:", err.message);
        // Clear the promise so a retry on next card open is possible
        _tfLoadPromise = null;
      } finally {
        FIRE_GATE.loading = false;
      }
    }

    // Open an image for the gate. createImageBitmap is fastest, but Safari
    // can reject HEIC via ImageBitmap — fall back to <img> + object URL so
    // iPhone photos still get gated.
    async function _openGateImage(file) {
      try {
        const bmp = await createImageBitmap(file);
        return { src: bmp, width: bmp.width, height: bmp.height, close: () => bmp.close() };
      } catch (err) {
        const url = URL.createObjectURL(file);
        const img = new Image();
        try {
          await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = url; });
          return { src: img, width: img.naturalWidth, height: img.naturalHeight, close: () => URL.revokeObjectURL(url) };
        } catch (e) {
          URL.revokeObjectURL(url);
          throw e;
        }
      }
    }

  function _gatePhoto(file) {
    // The notice is only shown at SUBMIT time — never while the photo is
    // being picked. Reset per-photo state here so a stale warning from a
    // previous photo can never resurface.
    _setGateNotice("", false);
    _setSubmitLabel();
    FIRE_GATE.warned = false;
    FIRE_GATE.lastProb = null;
    if (!FIRE_GATE.ready || !file || !file.type.startsWith("image/")) return;
    // Track the in-flight inference so the submit handler can await it —
    // a fast submit must not silently skip the warning because the score
    // wasn't ready yet.
    FIRE_GATE.pending = (async () => {
      let opened = null;
      try {
        opened = await _openGateImage(file);
        const size = 224;
        const canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext("2d");
        // cover-crop to a square — must mirror train_gate.py exactly
        const scale = Math.max(size / opened.width, size / opened.height);
        const w = size / scale;
        const h = size / scale;
        ctx.drawImage(opened.src, (opened.width - w) / 2, (opened.height - h) / 2, w, h, 0, 0, size, size);
        const pixels = window.tf.browser.fromPixels(canvas);
        // Feed raw [0, 255] — MobileNetV3 embeds its own Rescaling first
        // layer, so normalizing here too would crush the inputs (squashed
        // outputs ~0.3-0.6). Must match training exactly.
        const batched = pixels.expandDims(0).toFloat();
        const out = FIRE_GATE.model.predict(batched);
        // Graph models can return {name: tensor}; unwrap to the tensor.
        const tensor = out && out.data ? out : Object.values(out)[0];
        const fireProb = (await tensor.data())[0];
        window.tf.dispose([pixels, batched, tensor]);
        FIRE_GATE.lastProb = fireProb;
      } catch (err) {
        console.warn("[fire-gate] inference failed, skipping gate:", err);
      } finally {
        if (opened) opened.close();
        FIRE_GATE.pending = null;
      }
    })();
  }

  function _setGateNotice(text, show) {
    if (!els.fireGateNotice) return;
    els.fireGateNotice.textContent = text;
    els.fireGateNotice.classList.toggle("hidden", !show);
  }

  /**
   * Photo attach status — an explicit "photo made it into the report"
   * signal shown at the attach stage (before submit), independent of the
   * client-side scan result.
   */
  function _setPhotoStatus(text, kind) {
    if (!els.photoStatus) return;
    els.photoStatus.textContent = text || "";
    els.photoStatus.dataset.kind = kind || "";
    els.photoStatus.classList.toggle("hidden", !text);
  }

  /**
   * Swap the Submit button label for the gate's confirm state ("Submit
   * anyway") and restore the original label once the user proceeds or the
   * form resets.
   */
  function _setSubmitLabel(text) {
    if (!els.submitBtn) return;
    const label = els.submitBtn.querySelector("span:not(.btn-icon)");
    if (!label) return;
    if (text) {
      if (!els.submitBtn.dataset.wfLabel) els.submitBtn.dataset.wfLabel = label.textContent;
      label.textContent = text;
    } else if (els.submitBtn.dataset.wfLabel) {
      label.textContent = els.submitBtn.dataset.wfLabel;
      delete els.submitBtn.dataset.wfLabel;
    }
  }

  function setupUpload() {
    // ---- Trigger card open/close ----
    els.uploadTrigger.addEventListener("click", () => {
      els.uploadCard.classList.toggle("hidden");
      els.uploadTrigger.classList.toggle("hidden");
      // Lazy-load the client-side fire gate (MobileNetV3 + TF.js) on
      // first card open instead of at boot.  The model is ~5 MB and the
      // TF.js runtime is ~1 MB — most visitors never upload a photo, so
      // loading these eagerly wastes bandwidth and delays first paint.
      _loadFireGate();
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

      _setPhotoStatus("Reading photo…", "pending");

      const reader = new FileReader();
      reader.onload = (ev) => {
        els.previewPlaceholder.classList.add("hidden");
        els.previewImage.classList.remove("hidden");
        els.previewImage.src = ev.target.result;
        els.submitBtn.disabled = false;
        // Explicit confirmation that the photo made it into the report,
        // shown regardless of what the client-side scan says.
        _setPhotoStatus("✓ Photo uploaded — ready to submit", "ok");
      };
      reader.onerror = () => {
        _setPhotoStatus("Couldn't read this photo — try another one", "error");
        els.submitBtn.disabled = true;
      };
      reader.readAsDataURL(file);

      // Set captured_at to now
      els.inputCapturedAt.value = new Date().toISOString();

      // Compute the gate score silently — the notice is shown at submit
      // time, never on selection.
      _gatePhoto(file);
    });

    // ---- Form submit ----
    els.form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (STATE.uploading) return;

      // The photo-attach confirmation has served its purpose once the user
      // commits to submitting — dismiss it on any submit click (the gate
      // warning, if shown, replaces it).
      _setPhotoStatus("", "");

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

      // Client-side fire gate — warn at submit time, not when the photo is
      // picked. The first submit click on a photo the model thinks is clean
      // shows the notice; a second click proceeds. Soft-reject, never blocks.
      // Await an in-flight inference so a fast submit can't silently skip
      // the warning because the score wasn't ready yet.
      if (FIRE_GATE.pending) {
        try { await FIRE_GATE.pending; } catch (e) { /* gate degrades gracefully */ }
      }
      // If the model finished loading AFTER the photo was picked, the gate
      // never ran (FIRE_GATE.ready was false at pick time).  Re-run it now
      // so the warning still fires for non-fire photos.
      if (FIRE_GATE.ready && FIRE_GATE.lastProb === null && file) {
        _gatePhoto(file);
        if (FIRE_GATE.pending) {
          try { await FIRE_GATE.pending; } catch (e) { /* gate degrades */ }
        }
      }
      if (els.fireGateNotice && FIRE_GATE.ready && FIRE_GATE.lastProb != null &&
          FIRE_GATE.lastProb < FIRE_GATE.minProb && !FIRE_GATE.warned) {
        FIRE_GATE.warned = true;
        _setGateNotice(
          "We didn't detect fire or smoke in this photo. You can still submit it — " +
          "distant smoke or early-stage fires are easy to miss.",
          true
        );
        // Make the confirm flow unmistakable: the button itself becomes
        // the "proceed anyway" action.
        _setSubmitLabel("Submit anyway");
        return;
      }
      // Confirmed — hide the notice so the confirm copy doesn't linger
      // while the upload runs (or if it fails for an unrelated reason).
      _setGateNotice("", false);
      _setSubmitLabel();

      STATE.uploading = true;
      els.submitBtn.disabled = true;
      els.progressBar.classList.remove("hidden");
      els.progressBar.classList.add("active");

      const formData = new FormData(els.form);
      formData.set("session_id", STATE.sessionId);
      // Pass the pre-filter gate's verdict to the server — it raises the
      // auto-approval flame floor when it found no fire (soft veto). Only
      // sent when the gate actually ran; absent = no veto.
      if (FIRE_GATE.ready && FIRE_GATE.lastProb != null) {
        formData.set("gate_prob", String(FIRE_GATE.lastProb));
      }

      try {
        const res = await fetch("/api/reports", { method: "POST", body: formData });
        const data = await res.json();

        if (!res.ok) {
          // IP blocked for repeated non-fire uploads — show a prominent
          // message so the user understands why they can't upload.
          if (res.status === 403 && data.blocked_until) {
            const until = new Date(data.blocked_until);
            const hours = Math.max(1, Math.round((until - Date.now()) / 3600000));
            toast(
              `Upload blocked: too many photos without fire detected. ` +
              `Try again in ${hours}h.`,
              "error",
            );
            // Keep the card open so the user sees the toast, but stop uploading.
            STATE.uploading = false;
            els.submitBtn.disabled = false;
            els.progressBar.classList.remove("active");
            els.progressBar.classList.add("hidden");
            return;
          }
          throw new Error(data.error || "Upload failed");
        }

        // NOTE: the server no longer returns accepted:false — even a
        // "nothing" AI verdict is kept as a pending report for human review
        // (the hosted model can miss borderline fires), so every successful
        // upload follows the success path below.
        // Every successful upload must end with a clear success signal —
        // regardless of the client-side gate or the AI scan result (a
        // 'nothing' verdict still means the photo was accepted and kept
        // for human review, never silently discarded).
        if (STATE.mode === "demo") {
          // Uploads are real citizen reports and always go to the LIVE
          // (production) store — they won't show on the demo map.
          toast("Report submitted to LIVE data — switch to Live mode to see it", "success");
        } else if (data.report?.ai_analysis?.verdict === "nothing") {
          toast("Report submitted — AI found no fire/smoke, kept for human review", "success");
        } else {
          toast("Report submitted successfully!", "success");
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

    // Fire gate is lazy-loaded on first card open (see uploadTrigger
    // click handler above) instead of eagerly at boot — the ~6 MB
    // TF.js + MobileNetV3 payload is wasted for visitors who never
    // upload a photo.
  }

  function resetForm() {
    els.form.reset();
    els.previewPlaceholder.classList.remove("hidden");
    els.previewImage.classList.add("hidden");
    els.previewImage.src = "";
    els.submitBtn.disabled = true;
    _setGateNotice("", false);
    _setPhotoStatus("", "");
    _setSubmitLabel();
    FIRE_GATE.warned = false;
    FIRE_GATE.lastProb = null;
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
  /**
   * Self-rescheduling data poll loop.  Each tick awaits the fetch before
   * scheduling the next one, so a slow /api/reports response can never
   * overlap with the next poll — the old setInterval fired every N ms
   * regardless of how long the previous fetch took, causing two in-flight
   * requests on slow connections.
   */
  async function _dataPollTick() {
    if (!STATE.dataPollingInterval) return;  // stopped while awaiting
    await refreshData();
    // Re-arm AFTER the await so a stop (clearTimeout) during the fetch
    // doesn't immediately respawn the timer.
    if (STATE.dataPollingInterval) {
      STATE.dataPollingInterval = setTimeout(
        _dataPollTick,
        STATE.dataPollingMs || 15000
      );
    }
  }

  function startPolling(intervalMs = 15000) {
    // Keep the handle (and the period) so the tab-hidden pause/resume can
    // stop and restart polling without leaking timers.
    if (STATE.dataPollingInterval) clearTimeout(STATE.dataPollingInterval);
    STATE.dataPollingMs = intervalMs;
    refreshData();
    STATE.dataPollingInterval = setTimeout(_dataPollTick, intervalMs);
  }

  /**
   * Pause background polling while the tab is hidden and resume (with an
   * immediate refresh) when it comes back into view. Browsers throttle
   * background timers anyway, but being explicit stops wasted heavy
   * /api/bayesian/state polls on an invisible tab and leaves the map
   * current the instant the user looks at it again.
   */
  function _setupVisibilityPause() {
    if (!document.addEventListener) return;
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        if (STATE.bayesian.pollingInterval) {
          clearTimeout(STATE.bayesian.pollingInterval);
          STATE.bayesian.pollingInterval = null;
        }
        if (STATE.dataPollingInterval) {
          clearTimeout(STATE.dataPollingInterval);
          STATE.dataPollingInterval = null;
        }
      } else {
        if (STATE.bayesian.active && !STATE.bayesian.pollingInterval) {
          // Resume the adaptive poll loop (pollMs remembers the backoff
          // from before the tab was hidden — no reason to re-arm at 5s
          // if fires were idle).
          _bayesianPollTick();
          fetchBayesianState();
        }
        if (!STATE.dataPollingInterval) {
          refreshData();
          STATE.dataPollingInterval = setTimeout(_dataPollTick, STATE.dataPollingMs || 15000);
        }
      }
    });
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
      document.querySelector(".about-link"),
      document.querySelector(".faq-link"),
      document.querySelector(".poland-map-link"),
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

    // Close on Escape (click-outside is handled by _setupClickOutside).
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

  }

  // -----------------------------------------------------------------------
  // Single document-level click-outside handler
  // -----------------------------------------------------------------------
  // Multiple dropdowns (top-bar overflow menu, search results) each need
  // to close when the user clicks outside.  Instead of N separate
  // document.addEventListener("click") calls that all fire on every
  // click, one consolidated handler routes to the relevant close logic.
  function _setupClickOutside() {
    document.addEventListener("click", (e) => {
      // --- Top-bar overflow menu (mobile ⋯) ---
      if (els.topMenu && els.topMenu.classList.contains("open")) {
        if (els.topMenuBtn && els.topMenuBtn.contains(e.target)) return;
        if (els.topMenu.contains(e.target)) return;
        _closeTopMenu();
        return;
      }
      // --- Search results dropdown ---
      if (!e.target.closest("#search-box")) {
        _closeSearchResults();
      }
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
    _setupClickOutside();

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

    // Deep link from the admin dashboard: focus the requested fire.
    focusDeepLinkedFire();

    // Start polling
    startPolling();

    // Pause/resume the polls when the tab is hidden/visible
    _setupVisibilityPause();

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

    // Load conflict zone overlays (non-blocking)
    _loadConflictZones();
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
    if (windEl && data.wind_speed != null) {
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

    // Fire area from stats (hectares) — burned area for demo replay
    const areaEl = id("historic-fire-area");
    if (areaEl && data.statistics) {
      const areaHa = data.statistics.area_ha_p_above_0_25 || 0;
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
  // Fire Spread Simulation Player
  // -----------------------------------------------------------------------
  // On-demand forecast: user clicks "Simulate Spread" → fetch N future
  // timesteps → animate heatmap, road risk, and containment probability.

  const SIM = { active: false, frames: [], meta: null, timer: null, idx: 0, roadLayer: null, contourLayer: null };

  // Find the nearest grid to the viewport center from the raw grid data
  function _nearestGridToViewport() {
    const grids = STATE.bayesian && STATE.bayesian.rawGrids;
    if (!grids || !grids.length || !STATE.map) return null;
    const c = STATE.map.getCenter();
    let best = null, bestDist = Infinity;
    for (const g of grids) {
      const lat = g.centroid_lat || (g.state && g.state.ref_lat);
      const lon = g.centroid_lon || (g.state && g.state.ref_lon);
      if (lat == null || lon == null) continue;
      const dlat = lat - c.lat, dlon = (lon - c.lng) * Math.cos(c.lat * Math.PI / 180);
      const dist = dlat * dlat + dlon * dlon;
      if (dist < bestDist) { bestDist = dist; best = g; }
    }
    return best;
  }

  async function _wfStartSim() {
    if (SIM.active) { _wfStopSim(); return; }
    const btn = document.getElementById("wf-sim-panel-btn");
    const grid = _nearestGridToViewport();
    if (!grid) { toast("No fires visible — zoom into a fire first.", "warn"); return; }
    const gridId = grid.id;
    if (btn) { btn.textContent = "⏳ Computing…"; btn.disabled = true; }
    try {
      const res = await fetch(`/api/simulate/${encodeURIComponent(gridId)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ steps: 24 }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      SIM.frames = data.frames;
      SIM.meta = data.metadata;
      SIM.active = true;
      SIM.idx = 0;
      SIM.timer = null;

      // Create overlay layers (below popups, above heatmap)
      SIM.roadLayer = L.layerGroup().addTo(STATE.map);
      SIM.contourLayer = L.layerGroup().addTo(STATE.map);

      // Show the player UI
      _renderSimPlayer(data);
      _showSimFrame(0);
    } catch (e) {
      console.error("[simulate] error:", e);
      toast(`Simulation failed: ${e.message}`, "error");
      if (btn) { btn.textContent = "▶ Simulate Spread (Beta)"; btn.disabled = false; }
    }
  }

  function _wfStopSim() {
    SIM.active = false;
    if (SIM.timer) { clearInterval(SIM.timer); SIM.timer = null; }
    try {
      if (SIM.roadLayer && STATE.map) { STATE.map.removeLayer(SIM.roadLayer); }
      if (SIM.contourLayer && STATE.map) { STATE.map.removeLayer(SIM.contourLayer); }
    } catch (_) { /* ignore */ }
    SIM.roadLayer = null;
    SIM.contourLayer = null;
    const el = document.getElementById("wf-sim-player");
    if (el) el.remove();
    const btn = document.getElementById("wf-sim-panel-btn");
    if (btn) { btn.textContent = "▶ Simulate Spread (Beta)"; btn.disabled = false; }
  }
  window._wfStopSim = _wfStopSim;

  // Wire up the panel simulate button (after _wfStartSim is defined)
  document.getElementById("wf-sim-panel-btn").addEventListener("click", _wfStartSim);

  function _renderSimPlayer(data) {
    // Remove old player if any
    const old = document.getElementById("wf-sim-player");
    if (old) old.remove();

    const total = data.frames.length;
    const m = data.metadata;
    const html = `
      <div id="wf-sim-player" style="position:fixed;bottom:20px;left:50%;transform:translateX(-50%);z-index:10000;background:var(--card-bg,#1a1a2e);border:1px solid var(--border,#333);border-radius:12px;padding:12px 16px;min-width:340px;max-width:400px;box-shadow:0 4px 20px rgba(0,0,0,0.5);font-family:inherit;color:var(--text,#e0e0e0);">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <div style="font-weight:700;font-size:14px;">🔥 Spread Forecast</div>
          <button onclick="_wfStopSim()" style="background:none;border:none;color:var(--text-muted,#888);cursor:pointer;font-size:18px;padding:0 4px;">✕</button>
        </div>
        <div id="wf-sim-info" style="font-size:11px;color:var(--text-muted,#888);margin-bottom:8px;">
          Wind: ${m.wind_speed.toFixed(1)} m/s from ${Math.round(m.wind_dir_deg)}° · Moisture ×${m.moisture_factor}
        </div>
        <div id="wf-sim-frame" style="font-size:12px;margin-bottom:8px;"></div>
        <div style="display:flex;align-items:center;gap:8px;">
          <button id="wf-sim-play" onclick="window._wfSimTogglePlay()" style="background:var(--accent,#ff3c00);border:none;border-radius:6px;color:#fff;width:32px;height:32px;cursor:pointer;font-size:14px;">▶</button>
          <input id="wf-sim-slider" type="range" min="0" max="${total - 1}" value="0" style="flex:1;accent-color:var(--accent,#ff3c00);" oninput="window._wfSimGo(+this.value)">
          <span id="wf-sim-label" style="font-size:11px;min-width:50px;text-align:right;">+15 min</span>
        </div>
      </div>
    `;
    document.body.insertAdjacentHTML("beforeend", html);
  }

  function _showSimFrame(idx) {
    if (!SIM.active || !SIM.contourLayer || !SIM.roadLayer || idx < 0 || idx >= SIM.frames.length) return;
    SIM.idx = idx;
    const f = SIM.frames[idx];

    // Update frame info
    const infoEl = document.getElementById("wf-sim-frame");
    if (infoEl) {
      const roadCount = f.road_risk ? f.road_risk.length : 0;
      const critCount = f.road_risk ? f.road_risk.filter(r => r.risk_tier === "critical").length : 0;
      const highCount = f.road_risk ? f.road_risk.filter(r => r.risk_tier === "high").length : 0;
      let roadText = roadCount > 0
        ? `· 🛣️ ${roadCount} road(s) at risk` + (critCount ? ` (${critCount} critical)` : "") + (highCount ? ` (${highCount} high)` : "")
        : "· No roads at risk";
      infoEl.innerHTML = `
        <span style="color:var(--text,#e0e0e0);font-weight:600">${f.t_label}</span>
        <span style="margin-left:8px;">${f.cell_count} cells · max ${Math.round(f.max_p * 100)}%</span>
        <span style="margin-left:8px;">🛡️ ${Math.round(f.containment * 100)}%</span>
        <div style="margin-top:2px;font-size:11px;color:var(--text-muted,#888)">${roadText}</div>
      `;
    }

    // Update slider
    const slider = document.getElementById("wf-sim-slider");
    if (slider) slider.value = idx;
    const label = document.getElementById("wf-sim-label");
    if (label) label.textContent = f.t_label;

    // Clear old overlays
    if (SIM.roadLayer) SIM.roadLayer.clearLayers();
    if (SIM.contourLayer) SIM.contourLayer.clearLayers();

    // --- Ghost perimeters: dashed outlines of where the fire was
    //     at previous timesteps.  Earlier steps are dimmer.
    for (let gi = 0; gi < idx; gi++) {
      const gf = SIM.frames[gi];
      const opacity = 0.06 + 0.1 * (gi / Math.max(idx, 1));
      const segs = gf.contour_perimeter || [];
      for (const seg of segs) {
        if (seg.length < 2) continue;
        L.polyline(seg, {
          color: "#ff6600", weight: 1.5, opacity, dashArray: "4 4",
        }).addTo(SIM.contourLayer);
      }
    }

    // --- Single current perimeter: the outer spread boundary (0.05)
    //     one clean advancing white line with a light fill.
    //     After hour 2 (step 8), the line goes dashed to indicate
    //     lower forecast confidence.
    const hour2Step = 8; // 8 × 15 min = 2 h
    if (f.contour_perimeter && f.contour_perimeter.length > 0) {
      for (const seg of f.contour_perimeter) {
        if (seg.length < 3) continue;
        const dashed = idx >= hour2Step;
        L.polygon(seg, {
          fillColor: "#ff3c00",
          fillOpacity: 0.15 + 0.1 * idx / Math.max(SIM.frames.length - 1, 1),
          color: "#ffffff",
          weight: dashed ? 2 : 3,
          opacity: dashed ? 0.6 : 0.95,
          dashArray: dashed ? "8 6" : null,
        }).addTo(SIM.contourLayer);
      }
    }

    // --- Road risk lines (colored by tier) ---
    if (f.road_risk && f.road_risk.length > 0) {
      const tierColors = { critical: "#ff0000", high: "#ff6600", moderate: "#ffaa00" };
      for (const r of f.road_risk) {
        if (!r.segment || r.segment.length < 2) continue;
        const color = tierColors[r.risk_tier] || "#ffaa00";
        const weight = r.risk_tier === "critical" ? 4 : r.risk_tier === "high" ? 3 : 2;
        L.polyline(r.segment, { color, weight, opacity: 0.85 }).addTo(SIM.roadLayer);
      }
    }
  }

  window._wfSimGo = function(idx) { _showSimFrame(idx); };

  window._wfSimTogglePlay = function() {
    if (SIM.timer) {
      clearInterval(SIM.timer);
      SIM.timer = null;
      const btn = document.getElementById("wf-sim-play");
      if (btn) btn.textContent = "▶";
    } else {
      const btn = document.getElementById("wf-sim-play");
      if (btn) btn.textContent = "⏸";
      SIM.timer = setInterval(() => {
        let next = SIM.idx + 1;
        if (next >= SIM.frames.length) {
          next = 0;  // loop
        }
        _showSimFrame(next);
      }, 1500);  // 1.5s per frame
    }
  };

  // -----------------------------------------------------------------------
  // Conflict-zone hover overlays
  // -----------------------------------------------------------------------
  // Render semi-transparent overlays for conflict zones with a hover
  // popup explaining why satellite fires are hidden there.
  async function _loadConflictZones() {
    try {
      const res = await fetch("/api/conflict-zones");
      if (!res.ok) return;
      const { zones } = await res.json();
      if (!zones || !zones.length) return;

      const conflictLayer = L.layerGroup().addTo(STATE.map);
      const msg = "Satellite fire data is not shown in this region. " +
        "Active conflict makes it impossible to distinguish wildfires from " +
        "non-fire thermal sources (strikes, explosions, settlements).";

      for (const z of zones) {
        let layer;
        if (z.type === "bbox") {
          const b = z.bounds; // [south, west, north, east]
          layer = L.rectangle([[b[0], b[1]], [b[2], b[3]]], {
            color: "#ff4444", weight: 1, opacity: 0.3,
            fillColor: "#ff4444", fillOpacity: 0.06,
            interactive: true,
          });
        } else if (z.type === "polygon" && z.points.length >= 3) {
          layer = L.polygon(z.points, {
            color: "#ff4444", weight: 1, opacity: 0.3,
            fillColor: "#ff4444", fillOpacity: 0.06,
            interactive: true,
          });
        }
        if (layer) {
          layer.bindPopup(
            `<div style="max-width:220px;font-size:12px;line-height:1.4;">` +
            `<div style="font-weight:700;margin-bottom:4px;">⚠️ ${z.label}</div>` +
            `<div>${msg}</div></div>`,
            { closeButton: false, autoPan: false }
          );
          layer.on("mouseover", function(e) { this.openPopup(); });
          layer.on("mouseout", function(e) { this.closePopup(); });
          layer.addTo(conflictLayer);
        }
      }
    } catch (_) { /* non-critical */ }
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