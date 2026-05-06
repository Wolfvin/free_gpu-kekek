"""Local HTTP API server for FamilyGPU Orchestrator.

Provides REST endpoints for AI agents to request GPU compute,
check job status, and cancel jobs.

Agents should only use:
  POST /jobs          — Submit a training job
  GET  /jobs/{id}     — Check job status
  POST /jobs/{id}/cancel — Cancel a job

Other endpoints are for the TUI and administrative use:
  GET  /accounts      — List accounts (no credentials exposed)
  GET  /leases        — List active leases
  GET  /usage         — Get usage summary
  GET  /capacity      — Get available capacity
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
from urllib.parse import urlparse, parse_qs

from api import GPUSchedulerAPI
from scheduler.request import JobRequest

logger = logging.getLogger("fgt.api.http")

# Global API instance
_api: Optional[GPUSchedulerAPI] = None


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
                result = _api.request_gpu(job_request)
                self._send_json(result.to_dict(), status_code=201 if result.status == "accepted" else 200)
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

        else:
            self._send_error(f"Unknown endpoint: {path}", 404)

    def log_message(self, format, *args):
        """Override to use our logger instead of stderr."""
        logger.debug(f"HTTP: {format % args}")


def start_api_server(host: str = "127.0.0.1", port: int = 8420,
                     db_path: Optional[str] = None):
    """Start the HTTP API server.

    Args:
        host: Bind address (default: localhost only)
        port: Port number (default: 8420)
        db_path: Path to SQLite database
    """
    global _api
    _api = GPUSchedulerAPI(db_path=db_path)

    server = HTTPServer((host, port), AgentAPIHandler)
    logger.info(f"Agent API server starting on http://{host}:{port}")
    logger.info("Agent endpoints: POST /jobs, GET /jobs/{id}, POST /jobs/{id}/cancel")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("API server shutting down")
        server.server_close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_api_server()
