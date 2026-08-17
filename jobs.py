#!/usr/bin/env python3
"""
jobs.py — Procrastinate job-queue app for WildFrame
====================================================

Replaces the process-local background threads (simulated satellite poller
and FIRMS poller) with durable, Postgres-backed periodic jobs. Because the
jobs live in the queue, they survive restarts and run exactly once even
with multiple web workers (the queue is the single source of truth for
scheduling; workers only pick up jobs).

Each periodic job checks a flag in the ``kv_store`` table before doing any
work — the API's start/stop endpoints just flip that flag, so no schedules
need to be added/removed at runtime.

Scheduling notes
----------------
Periodic jobs use a 5-field cron (``* * * * *`` = once a minute) and then
self-throttle against the configured ``interval_s`` using their last-run
heartbeat. 6-field (second-resolution) crons are NOT used because the
installed croniter (6.x) misparses them without ``second_at_beginning``
and Procrastinate constructs its croniter without that flag — a 6-field
expression fires every second instead of every N seconds.

Run the worker (handles both job execution and periodic deferral):
    python3 worker.py
"""

import logging
import os
import time

import cap_adapter
import db
import effis_fwi
import weather

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("wildframe.jobs")


def _make_app():
    from procrastinate import App, PsycopgConnector

    return App(
        connector=PsycopgConnector(conninfo=db.DATABASE_URL),
        periodic_defaults={"max_delay": 300.0},
    )


app = _make_app()


def _as_bool(value, default: bool = False) -> bool:
    """Parse a poller-config flag robustly (bools or common string spellings),
    so e.g. ``"include_test": "false"`` from a UI/JSON config never becomes
    ``True`` via Python's string truthiness."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return default


def _should_run(kind: str, default_interval_s: float, cfg: dict) -> tuple[bool, str]:
    """Apply the poller flag + interval throttle. Returns (run, skip_reason)."""
    if not cfg["active"]:
        return False, "poller inactive"
    interval_s = float(cfg.get("interval_s", default_interval_s))
    hb = db.kv_get(f"{kind}_poller_heartbeat") or {}
    last_run = hb.get("at", 0)
    if time.time() - last_run < interval_s:
        return False, "within interval"
    return True, ""


@app.periodic(cron="* * * * *")
@app.task(name="satellite.simulated_pass")
def simulated_satellite_pass_job(**kwargs):
    """Periodic simulated satellite pass over demo grids.

    Runs at most once a minute (cron granularity); self-throttles to the
    configured ``interval_s`` (default 20 s) so a longer interval is
    honored exactly, and anything shorter just runs every minute.
    """
    cfg = db.get_poller_config("satellite", {})
    run, reason = _should_run("satellite", 20.0, cfg)
    if not run:
        return {"skipped": True, "reason": reason}

    import server

    try:
        result = server._simulate_satellite_pass(
            probability=float(cfg.get("probability", 0.6)),
            min_hotspots=int(cfg.get("min_hotspots", 1)),
            max_hotspots=int(cfg.get("max_hotspots", 3)),
        )
        db.kv_set("satellite_poller_heartbeat", {"at": time.time()})
        if result.get("injected", 0) > 0:
            logger.info(
                "[satellite-poller] Injected %d hotspot(s) across %d grid(s).",
                result["injected"], result["grids_hit"],
            )
        return result
    except Exception as exc:
        logger.error("[satellite-poller] Error: %s", exc)
        return {"error": str(exc)}


@app.periodic(cron="* * * * *")
@app.task(name="grids.expire_stale")
def expire_stale_grids_job(**kwargs):
    """Periodic expiry sweep: delete production fires whose newest evidence
    is older than 24 hours.

    The map's grid list comes straight from Postgres, so deleting the rows
    makes old fires disappear from the heatmap and the DB. Also runs at the
    end of every FIRMS fetch, but this job keeps the map current even when
    the poller is idle.

    Self-throttles to once an hour (cron is once a minute) via the same
    heartbeat pattern as the pollers.
    """
    hb = db.kv_get("grids_expire_heartbeat") or {}
    last_run = hb.get("at", 0)
    if time.time() - last_run < 3600.0:
        return {"skipped": True, "reason": "within interval"}

    try:
        # Claim the hour FIRST so a transient error doesn't retry every
        # minute (and hammer the DB with the same sweep).
        db.kv_set("grids_expire_heartbeat", {"at": time.time()})
        n = db.purge_stale_grids("production", max_age_hours=24.0)
        if n:
            logger.info(
                "[grids-expire] Removed %d stale production grid(s) "
                "(no evidence in 24h).", n,
            )
        return {"purged": n}
    except Exception as exc:
        logger.error("[grids-expire] Error: %s", exc)
        return {"error": str(exc)}


@app.periodic(cron="* * * * *")
@app.task(name="grids.advance")
def advance_grids_job(**kwargs):
    """Periodic checkpoint: advance + persist a rotating slice of
    production grids by their elapsed time (see db.advance_grids).

    The /api/bayesian/state read path is read-only now — it extrapolates
    fires in-memory between checkpoints but never writes. This job is what
    actually persists progress, on a rolling basis (a bounded slice per
    run) so a global dataset of thousands of fires never gets one
    expensive sweep.

    Self-throttles to once a minute (cron is once a minute anyway).
    """
    hb = db.kv_get("grids_advance_heartbeat") or {}
    last_run = hb.get("at", 0)
    if time.time() - last_run < 60.0:
        return {"skipped": True, "reason": "within interval"}

    try:
        # Claim the interval FIRST so a transient error doesn't retry
        # every minute.
        # limit=1000 keeps the sweep (~13k grids → ~13 min) faster than the
        # read path's 10-min extrapolation cap, so fires keep animating
        # between checkpoints instead of freezing at max extrapolation.
        db.kv_set("grids_advance_heartbeat", {"at": time.time()})
        n = db.advance_grids("production", limit=1000, min_age_s=60.0)
        if n:
            logger.info("[grids-advance] Checkpointed %d production grid(s).", n)
        # Keep long-lived fires on real, current wind: refresh a bounded
        # slice of grids whose wind is stale. Self-budgeting + ~55 km cell
        # caching inside weather.py keep this under the API free tier.
        try:
            # Wall-clock capped: a degraded weather API must never hold the
            # single worker queue (it starves firms.fetch / cap_poll). The
            # sweep resumes next run from the still-stale grids.
            refreshed = weather.refresh_grids_wind(
                "production", limit=200, max_age_s=24 * 60 * 60,
                max_wall_s=15.0,
            )
        except Exception as exc:
            logger.error("[grids-advance] Wind refresh error: %s", exc)
            refreshed = 0
        # Daily fuel-moisture / fire-weather indices from EFFIS (FFMC, DMC,
        # ISI) — feeds the spread kernel's moisture factor + decay scale.
        # Cached per ~55 km cell per day; grids outside EFFIS coverage are
        # stamped once and skipped.
        try:
            # Wall-clock capped (see refresh_grids_fwi): the EFFIS WMS is
            # intermittently down and each hang used to stretch this job to
            # ~250s, blocking FIRMS/cap/expire behind it.
            refreshed_fwi = effis_fwi.refresh_grids_fwi(
                "production", limit=200, max_age_s=12 * 3600,
                max_wall_s=25.0,
            )
        except Exception as exc:
            logger.error("[grids-advance] EFFIS refresh error: %s", exc)
            refreshed_fwi = 0
        return {"advanced": n, "wind_refreshed": refreshed, "fwi_refreshed": refreshed_fwi}
    except Exception as exc:
        logger.error("[grids-advance] Error: %s", exc)
        return {"error": str(exc)}


@app.periodic(cron="* * * * *")
@app.task(name="firms.fetch")
def firms_fetch_job(**kwargs):
    """Real NASA FIRMS fetch — periodic and on-demand.

    Periodic runs self-throttle to the configured ``interval_s`` (default
    600 s / 10 min). A manual "Fetch FIRMS" click defers this same job
    with ``force=True`` to skip the throttle; it then runs exactly once in
    the worker and the result summary is stored in ``kv_store`` so the UI
    can show it without blocking the request.
    """
    force = bool(kwargs.pop("force", False))

    cfg = db.get_poller_config("firms", {})
    if not force:
        run, reason = _should_run("firms", 600.0, cfg)
        if not run:
            return {"skipped": True, "reason": reason}
        # A manual fetch owns the pass — never run the scheduled poll on top
        # of it (both would inject the same hotspots, double-counting
        # evidence). Only the periodic run is blocked; the forced job that
        # the route queued is the one doing the fetching.
        in_prog = db.kv_get("firms_fetch_in_progress") or {}
        if in_prog.get("at") and time.time() - in_prog.get("at") < 900:
            return {"skipped": True, "reason": "manual fetch in progress"}

    # Claim the pass (both manual and periodic runs): this makes the manual
    # route's 409 catch a scheduled poll that's already running, and lets a
    # periodic run in another worker see this one's in-flight pass. Cleared
    # on completion or error below.
    db.kv_set("firms_fetch_in_progress", {"at": time.time()})

    import server

    try:
        # Look-back window defaults to 2 days (48h, matching NASA's map
        # view) inside _fetch_nasa_firms_pass; the poller config can
        # override it via day_range (1-5, clamped there).
        result = server._fetch_nasa_firms_pass(
            min_confidence=str(kwargs.get("min_confidence", cfg.get("min_confidence", "nominal"))),
            day_range=int(kwargs.get("day_range", cfg.get("day_range", 2))),
        )
        db.kv_set("firms_poller_heartbeat", {"at": time.time()})
        db.kv_set("firms_fetch_in_progress", False)
        db.kv_set("firms_fetch_last_result", {
            "finished_at": time.time(),
            "message": server._firms_fetch_message(result),
            "api_error": result.get("api_error"),
            "injected": result.get("injected", 0),
            "grids_hit": result.get("grids_hit", 0),
            "grids_considered": result.get("grids_considered", 0),
            "firms_hotspots": result.get("firms_hotspots", 0),
            "new_grids": result.get("new_grids", 0),
            "reports_confirmed": result.get("reports_confirmed", 0),
        })
        if result.get("api_error"):
            logger.warning("[firms-poller] API error: %s", result["api_error"])
        elif result.get("injected", 0) > 0:
            logger.info(
                "[firms-poller] Injected %d hotspot(s) across %d grid(s) "
                "(%d new grids).",
                result["injected"], result["grids_hit"], result.get("new_grids", 0),
            )
        else:
            logger.info("[firms-poller] No FIRMS hotspots found.")
        return result
    except Exception as exc:
        logger.error("[firms-poller] Error: %s", exc)
        db.kv_set("firms_fetch_in_progress", False)
        db.kv_set("firms_fetch_last_result", {
            "finished_at": time.time(),
            "error": str(exc),
        })
        return {"error": str(exc)}


@app.periodic(cron="* * * * *")
@app.task(name="agencies.cap_poll")
def cap_poll_job(**kwargs):
    """Pull a government CAP feed and push every alert through the agency
    ingest endpoint — the pull path of the government-incident pipeline.

    Periodic runs self-throttle to the configured ``interval_s`` (default
    300 s / 5 min). Reads ``db.get_poller_config("cap", ...)``:

      - ``feed_url``     : CAP feed URL (required — the job is a no-op when
                           unset, so the machinery ships dormant)
      - ``base_url``     : WildFrame server to POST incidents to (defaults
                           to the ``WILDFRAME_BASE_URL`` env var, else
                           http://localhost:4141)
      - ``mode``         : "production" | "demo" (default production)
      - ``include_test`` : include drill/Test CAP alerts (sandbox testing)
      - ``api_key``      : X-Agency-Key sent to the ingest endpoint
                           (defaults to the ``WILDFRAME_AGENCY_API_KEY`` env)

    Configure with::

        db.set_poller_active("cap", True, {
            "feed_url": "https://.../cap.xml",
            "mode": "demo",
        })

    Results land in ``kv_store["cap_poller_last_result"]`` and the
    ``cap_poller_heartbeat`` drives the ``_should_run`` throttle. Note the
    heartbeat is stamped BEFORE the fetch: a failed or slow pass backs off
    the throttle for the full ``interval_s`` (unlike ``firms.fetch``, which
    only stamps on success) — a deliberate failure-backoff so a flaky feed
    can't turn into a per-minute retry firehose.
    """
    cfg = db.get_poller_config("cap", {})
    run, reason = _should_run("cap", 300.0, cfg)
    if not run:
        return {"skipped": True, "reason": reason}

    feed_url = (cfg.get("feed_url") or "").strip()
    if not feed_url:
        # Warn + record ONCE per state change, not every minute: the job
        # runs every 60 s via the periodic deferrer, and "active but no
        # feed_url" is a config mistake, not a condition to log 1440×/day.
        msg = "No CAP feed_url configured — see agencies.cap_poll docstring"
        last = db.kv_get("cap_poller_last_result") or {}
        if last.get("error") != "feed_url not configured":
            logger.warning("[cap-poller] %s", msg)
            db.kv_set("cap_poller_last_result", {
                "finished_at": time.time(),
                "message": msg,
                "error": "feed_url not configured",
            })
        return {"skipped": True, "reason": "feed_url not configured"}

    base_url = (
        cfg.get("base_url")
        or os.environ.get("WILDFRAME_BASE_URL")
        or "http://localhost:4141"
    ).rstrip("/")
    mode = cfg.get("mode", "production")
    include_test = _as_bool(cfg.get("include_test", False))
    api_key = cfg.get("api_key") or os.environ.get("WILDFRAME_AGENCY_API_KEY") or ""

    db.kv_set("cap_poller_heartbeat", {"at": time.time()})
    try:
        result = cap_adapter.consume_cap_feed(
            feed_url, base_url, mode=mode, include_test=include_test,
            api_key=api_key or None,
        )
        db.kv_set("cap_poller_last_result", {
            "finished_at": time.time(),
            "fetched": result.get("fetched", 0),
            "message": "CAP poll: %d alert(s) processed" % result.get("fetched", 0),
            "results": result.get("results", []),
        })
        if result.get("fetched", 0):
            logger.info(
                "[cap-poller] Processed %d CAP alert(s) via %s",
                result["fetched"], base_url,
            )
        return result
    except Exception as exc:
        logger.error("[cap-poller] Error: %s", exc)
        db.kv_set("cap_poller_last_result", {
            "finished_at": time.time(),
            "error": str(exc),
        })
        return {"error": str(exc)}
