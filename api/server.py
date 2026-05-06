"""Local HTTP API server for FamilyGPU Orchestrator.

Provides REST endpoints for AI agents to request GPU compute,
check job status, and cancel jobs. When auto loop is enabled,
also provides endpoints for monitoring and controlling the
auto-scheduling daemon.

Agent endpoints:
  POST /jobs              — Submit a training job
  GET  /jobs/{id}         — Check job status
  POST /jobs/{id}/cancel  — Cancel a job

Auto loop endpoints:
  GET  /autoloop          — Get auto loop status
  POST /autoloop/start    — Start the auto loop daemon
  POST /autoloop/stop     — Stop the auto loop daemon
  POST /autoloop/failover — Force failover for a job

Administrative endpoints:
  GET  /accounts          — List accounts (no credentials exposed)
  GET  /leases            — List active leases
  GET  /usage             — Get usage summary
  GET  /capacity          — Get available capacity
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
from urllib.parse import urlparse, parse_qs

from api import GPUSchedulerAPI
from scheduler.request import JobRequest
from scheduler.autoloop import AutoLoop, AutoLoopConfig

logger = logging.getLogger("fgt.api.http")

# Global API instance
_api: Optional[GPUSchedulerAPI] = None

# Global AutoLoop instance
_autoloop: Optional[AutoLoop] = None


class AgentAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Agent API."""

    def _set_headers(self, status_code: int = 200, content_type: str = "application/json"):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _read_body(self) -> dict:
        """Read and parse JSON body from request."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {}

    def _send_json(self, data, status_code: int = 200):
        """Send a JSON response."""
        self._set_headers(status_code)
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())

    def _send_error(self, message: str, status_code: int = 400):
        """Send an error response."""
        self._send_json({"error": message}, status_code)

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self._set_headers(204)
        self.wfile.write(b"")

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query)

        if path == "/jobs":
            status = query.get("status", [None])[0]
            jobs = _api.list_jobs(status=status)
            self._send_json({"jobs": jobs})

        elif path.startswith("/jobs/"):
            job_id = path.split("/jobs/")[1]
            if job_id == "capacity":
                capacity = _api.get_available_capacity()
                self._send_json(capacity)
            else:
                status = _api.get_job_status(job_id)
                if status:
                    self._send_json(status)
                else:
                    self._send_error(f"Job {job_id} not found", 404)

        elif path == "/autoloop":
            self._handle_autoloop_status()

        elif path == "/accounts":
            accounts = _api.account_repo.list_all()
            # Remove credential_ref from response
            safe = [
                {k: v for k, v in a.items() if k != "credential_ref"}
                for a in accounts
            ]
            self._send_json({"accounts": safe})

        elif path == "/leases":
            active = query.get("active", ["false"])[0].lower() == "true"
            if active:
                leases = _api.lease_repo.list_active()
            else:
                leases = _api.lease_repo.list_all()
            self._send_json({"leases": leases})

        elif path == "/usage":
            group_by = query.get("group_by", ["provider"])[0]
            summary = _api.quota_repo.get_usage_summary(group_by=group_by)
            self._send_json({"usage": summary})

        elif path == "/capacity":
            capacity = _api.get_available_capacity()
            self._send_json(capacity)

        else:
            self._send_error(f"Unknown endpoint: {path}", 404)

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/jobs":
            # Submit a new job
            body = self._read_body()
            try:
                job_request = JobRequest.from_dict(body)

                # Use autoloop if available, otherwise use direct API
                if _autoloop and _autoloop.is_running:
                    result = _autoloop.submit_job(job_request)
                    self._send_json(
                        result.to_dict(),
                        status_code=201 if result.status == "accepted" else 200,
                    )
                else:
                    result = _api.request_gpu(job_request)
                    self._send_json(
                        result.to_dict(),
                        status_code=201 if result.status == "accepted" else 200,
                    )
            except Exception as e:
                self._send_error(f"Invalid request: {e}", 400)

        elif path.startswith("/jobs/") and path.endswith("/cancel"):
            # Cancel a job
            job_id = path.split("/jobs/")[1].replace("/cancel", "")
            success = _api.cancel_job(job_id)
            if success:
                self._send_json({"status": "cancelled", "job_id": job_id})
            else:
                self._send_error(f"Cannot cancel job {job_id}", 400)

        elif path == "/autoloop/start":
            self._handle_autoloop_start()

        elif path == "/autoloop/stop":
            self._handle_autoloop_stop()

        elif path == "/autoloop/failover":
            self._handle_autoloop_failover()

        else:
            self._send_error(f"Unknown endpoint: {path}", 404)

    # ── Auto Loop Endpoints ────────────────────────────────────────

    def _handle_autoloop_status(self):
        """GET /autoloop — Get auto loop status."""
        global _autoloop
        if _autoloop:
            self._send_json(_autoloop.get_status())
        else:
            self._send_json({
                "auto_loop": {"is_running": False},
                "message": "AutoLoop not initialized. Start with POST /autoloop/start",
            })

    def _handle_autoloop_start(self):
        """POST /autoloop/start — Start the auto loop daemon."""
        global _autoloop
        body = self._read_body()

        if _autoloop and _autoloop.is_running:
            self._send_json({"status": "already_running", "message": "AutoLoop is already running"})
            return

        # Create AutoLoop with optional config from body
        config = AutoLoopConfig()
        if body:
            if "lease_check_interval" in body:
                config.lease_check_interval = float(body["lease_check_interval"])
            if "queue_check_interval" in body:
                config.queue_check_interval = float(body["queue_check_interval"])
            if "health_check_interval" in body:
                config.health_check_interval = float(body["health_check_interval"])
            if "checkpoint_before_expiry_minutes" in body:
                config.checkpoint_before_expiry_minutes = int(body["checkpoint_before_expiry_minutes"])
            if "auto_failover" in body:
                config.auto_failover = bool(body["auto_failover"])
            if "auto_start_queued" in body:
                config.auto_start_queued = bool(body["auto_start_queued"])
            if "auto_health_check" in body:
                config.auto_health_check = bool(body["auto_health_check"])

        _autoloop = AutoLoop(config=config)
        _autoloop.start()

        self._send_json({
            "status": "started",
            "message": "AutoLoop daemon started",
            "config": {
                "lease_check_interval": config.lease_check_interval,
                "queue_check_interval": config.queue_check_interval,
                "auto_failover": config.auto_failover,
                "auto_start_queued": config.auto_start_queued,
            },
        })

    def _handle_autoloop_stop(self):
        """POST /autoloop/stop — Stop the auto loop daemon."""
        global _autoloop
        if not _autoloop or not _autoloop.is_running:
            self._send_json({"status": "not_running", "message": "AutoLoop is not running"})
            return

        stats = _autoloop.stats.to_dict()
        _autoloop.stop()

        self._send_json({
            "status": "stopped",
            "message": "AutoLoop daemon stopped",
            "final_stats": stats,
        })

    def _handle_autoloop_failover(self):
        """POST /autoloop/failover — Force failover for a job."""
        global _autoloop
        body = self._read_body()
        job_id = body.get("job_id", "")

        if not job_id:
            self._send_error("job_id is required", 400)
            return

        if not _autoloop:
            self._send_error("AutoLoop not initialized", 400)
            return

        result = _autoloop.force_failover(job_id)
        if result:
            self._send_json(result.to_dict())
        else:
            self._send_error(f"No active lease found for job {job_id}", 404)

    def log_message(self, format, *args):
        """Override to use our logger instead of stderr."""
        logger.debug(f"HTTP: {format % args}")


def start_api_server(host: str = "127.0.0.1", port: int = 8420,
                     db_path: Optional[str] = None,
                     autoloop: Optional[AutoLoop] = None):
    """Start the HTTP API server.

    Args:
        host: Bind address (default: localhost only)
        port: Port number (default: 8420)
        db_path: Path to SQLite database
        autoloop: Optional AutoLoop instance for auto-scheduling
    """
    global _api, _autoloop
    _api = GPUSchedulerAPI(db_path=db_path)
    _autoloop = autoloop

    server = HTTPServer((host, port), AgentAPIHandler)
    logger.info(f"Agent API server starting on http://{host}:{port}")
    logger.info("Agent endpoints: POST /jobs, GET /jobs/{id}, POST /jobs/{id}/cancel")

    if _autoloop and _autoloop.is_running:
        logger.info("Auto loop daemon is active — continuous scheduling enabled")
        logger.info("Auto endpoints: GET /autoloop, POST /autoloop/start, POST /autoloop/stop, POST /autoloop/failover")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("API server shutting down")
        if _autoloop and _autoloop.is_running:
            _autoloop.stop()
        server.server_close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_api_server()
