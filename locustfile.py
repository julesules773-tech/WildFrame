"""WildFrame / Pyrae — map-reader load test.

Simulates beta users browsing the live fire map. Each virtual user loads the
app shell once, then keeps polling the three endpoints the frontend calls on
every map tick:

    GET /api/reports
    GET /api/clusters
    GET /api/bayesian/state?threshold=1e-10&contour=0&mode=production&bbox=…&detail=full

Load ramps through ``RampShape.stages`` (5 → 200 users), so latency/RPS climb
gradually and we can pinpoint the concurrency ceiling — the "breaking point".

Run it headless against a production-shaped server:

    .venv/bin/locust -f locustfile.py --host http://127.0.0.1:4143 \
        --headless --csv /tmp/wf_stress

Results land in /tmp/wf_stress_stats.csv (per-endpoint percentiles) and
/tmp/wf_stress_stats_history.csv (per-second RPS + response times across the
ramp, for spotting exactly where the curve bends).
"""

import random

from locust import HttpUser, between, task
from locust import LoadTestShape


class MapReader(HttpUser):
    """A beta user browsing the fire map."""

    # Think time while panning / reading between poll cycles.
    wait_time = between(1, 4)

    def on_start(self):
        # Browser shell load — happens once per visit.
        self.client.get("/")
        self.client.get("/app.js")
        self.client.get("/style.css")
        self.client.get("/api/admin/status", name="/api/admin/status")

    @task
    def map_tick(self):
        # One map poll cycle, mirroring the frontend's API calls.
        self.client.get("/api/reports", name="/api/reports")
        self.client.get("/api/clusters", name="/api/clusters")
        bbox = random.choice(BBOXES)
        self.client.get(
            "/api/bayesian/state?threshold=1e-10&contour=0&mode=production"
            f"&bbox={bbox}&detail=full",
            name="/api/bayesian/state",
        )


# Realistic Poland-area viewports the map actually opens (from live logs).
BBOXES = [
    "16.803588867187504,50.43651601698633,25.938720703125004,53.79740645735382",
    "16.776123046875004,51.382066781130604,25.911254882812504,54.67383096593116",
    "16.199340820312504,51.51216124955517,25.334472656250004,54.7943516039205",
]


class RampShape(LoadTestShape):
    """Staged ramp through realistic beta peaks, then past them until it breaks.

    Durations are cumulative end-times (locust convention): stage N runs from
    the previous stage's duration up to this one's.
    """

    stages = [
        {"duration": 30, "users": 50, "spawn_rate": 20},
        {"duration": 90, "users": 100, "spawn_rate": 50},
        {"duration": 150, "users": 200, "spawn_rate": 100},
        {"duration": 210, "users": 300, "spawn_rate": 100},
        {"duration": 270, "users": 400, "spawn_rate": 100},
        {"duration": 330, "users": 500, "spawn_rate": 100},
        {"duration": 390, "users": 600, "spawn_rate": 100},
        {"duration": 450, "users": 800, "spawn_rate": 100},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])
        # All stages done — end the run.
        return None
