"""WildFrame / Pyrae — upload-flow load test with the AI scan failing.

Simulates beta users submitting photos while the Roboflow AI scan is DOWN
(the throwaway server is launched with a garbage ``ROBOFLOW_API_KEY``).
Every upload must still return 201 with a *pending* report and the scan
error recorded — the fallback under load.

Run it headless against the poisoned instance:

    .venv/bin/locust -f locustfile_upload.py --host http://127.0.0.1:4147 \
        --headless --csv /tmp/wf_ul_stress

Each upload tags ``session_id=chaos-load-*`` so all test reports (and their
photos) can be bulk-deleted afterwards.
"""

import uuid

from locust import HttpUser, LoadTestShape, between, task

PHOTO = "/tmp/wf_chaos_photo.jpg"


class Uploader(HttpUser):
    """A beta user submitting a fire report while the AI scanner is down."""

    wait_time = between(2, 6)

    def on_start(self):
        # Unique per-user session so cleanup can target the whole run.
        self.session = f"chaos-load-{uuid.uuid4().hex[:8]}"

    @task
    def upload_report(self):
        with open(PHOTO, "rb") as f:
            self.client.post(
                "/api/reports",
                files={"photo": ("report.jpg", f, "image/jpeg")},
                data={
                    "lat": "54.40",
                    "lon": "18.60",
                    "session_id": self.session,
                },
                name="/api/reports (upload)",
            )


class Flat100(LoadTestShape):
    """Hold 100 concurrent uploaders for 90s, then stop."""

    def tick(self):
        if self.get_run_time() < 90:
            return (100, 10)
        return None
