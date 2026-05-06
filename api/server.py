"""Local HTTP API server for FamilyGPU Orchestrator.

Provides REST endpoints for AI agents to request GPU compute,
check job status, and cancel jobs. When auto loop is enabled,
also provides endpoints for monitoring and controlling the
auto-scheduling daemon.

All endpoints (except /health) require Bearer token authentication.
The token is auto-generated on first startup and saved to
~/.familygpu/api_token. Use --no-auth flag to disable auth for
development.

Rate limiting: 60 requests per minute per IP.

Health endpoint (no auth required):
  GET  /health            — Health check

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
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
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

# Authentication
_api_token: Optional[str] = None
_no_auth: bool = False


def _load_or_create_token() -> str:
    """Load or create API authentication token."""
    import secrets
    token_path = Path(os.path.expanduser("~")) / ".familygpu" / "api_token"
    token_path.parent.mkdir(parents=True, exist_ok=True)

    if token_path.exists():
        return token_path.read_text().strip()

    token = secrets.token_urlsafe(32)
    token_path.write_text(token)
    os.chmod(str(token_path), 0o600)
    logger.info(f"Generated new API token: {token[:8]}...")
    logger.info(f"Full token saved to {token_path}")
    return token


class AgentAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Agent API."""

    # Rate limiter (class-level)
    _rate_limits: dict[str, list[float]] = {}  # IP -> list of request timestamps
    RATE_LIMIT_PER_MINUTE = 60

    def _check_rate_limit(self) -> bool:
        """Check if the request is within rate limits. Returns True if allowed."""
        client_ip = self.client_address[0]
        now = time.time()

        if client_ip not in AgentAPIHandler._rate_limits:
            AgentAPIHandler._rate_limits[client_ip] = []

        # Remove timestamps older than 60 seconds
        AgentAPIHandler._rate_limits[client_ip] = [
            t for t in AgentAPIHandler._rate_limits[client_ip] if now - t < 60
        ]

        if len(AgentAPIHandler._rate_limits[client_ip]) >= AgentAPIHandler.RATE_LIMIT_PER_MINUTE:
            return False

        AgentAPIHandler._rate_limits[client_ip].append(now)
        return True

    def _check_auth(self) -> bool:
        """Check bearer token authentication. Returns True if authorized."""
        global _api_token, _no_auth
        if _no_auth:
            return True
        if not _api_token:
            return True  # No token configured = no auth required

        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            return token == _api_token
        return False

    def _rate_limit_remaining(self) -> int:
        """Return number of remaining requests for the client IP this minute."""
        client_ip = self.client_address[0]
        now = time.time()
        timestamps = AgentAPIHandler._rate_limits.get(client_ip, [])
        current = [t for t in timestamps if now - t < 60]
        return max(0, AgentAPIHandler.RATE_LIMIT_PER_MINUTE - len(current))

    def _set_headers(self, status_code: int = 200, content_type: str = "application/json"):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("X-RateLimit-Remaining", str(self._rate_limit_remaining()))
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
        # Rate limit check
        if not self._check_rate_limit():
            self._send_error("Rate limit exceeded", 429)
            return

        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query)

        # Health endpoint — no auth required
        if path == "/health":
            self._send_json({"status": "ok", "version": "0.1.0"})
            return

        # Auth check (all other endpoints)
        if not self._check_auth():
            self._send_error("Unauthorized — provide Bearer token", 401)
            return

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
        # Rate limit check
        if not self._check_rate_limit():
            self._send_error("Rate limit exceeded", 429)
            return

        # Auth check (all POST endpoints require auth)
        if not self._check_auth():
            self._send_error("Unauthorized — provide Bearer token", 401)
            return

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
                     autoloop: Optional[AutoLoop] = None,
                     no_auth: bool = False):
    """Start the HTTP API server.

    Args:
        host: Bind address (default: localhost only)
        port: Port number (default: 8420)
        db_path: Path to SQLite database
        autoloop: Optional AutoLoop instance for auto-scheduling
        no_auth: Disable Bearer token authentication (for development)
    """
    global _api, _autoloop, _api_token, _no_auth
    _api = GPUSchedulerAPI(db_path=db_path)
    _autoloop = autoloop
    _no_auth = no_auth

    # Load or create API token
    if not no_auth:
        _api_token = _load_or_create_token()
    else:
        logger.warning("API authentication is DISABLED (--no-auth flag)")

    server = HTTPServer((host, port), AgentAPIHandler)
    logger.info(f"Agent API server starting on http://{host}:{port}")
    logger.info("Health endpoint: GET /health (no auth required)")
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
