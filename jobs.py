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
Both jobs use a 5-field cron (``* * * * *`` = once a minute) and then
self-throttle against the configured ``interval_s`` using their last-run
heartbeat. 6-field (second-resolution) crons are NOT used because the
installed croniter (6.x) misparses them without ``second_at_beginning``
and Procrastinate constructs its croniter without that flag — a 6-field
expression fires every second instead of every N seconds.

Run the worker (handles both job execution and periodic deferral):
    python3 worker.py
"""

import logging
import time

import db

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
        return {"advanced": n}
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
        # The fetch is always the past 24 hours — day_range is hard-wired
        # to 1 inside _fetch_nasa_firms_pass itself.
        result = server._fetch_nasa_firms_pass(
            min_confidence=str(kwargs.get("min_confidence", cfg.get("min_confidence", "nominal"))),
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
