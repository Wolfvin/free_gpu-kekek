# FamilyGPU Orchestrator

> **Quota-Aware Multi-Account GPU Scheduler with Auto Loop**

A TUI + local scheduler for managing authorized GPU/compute accounts
across 12 free-tier providers. Designed for families who want to pool
their legitimate account quotas for AI training workloads.

> ⚠️ **This project is for authorized accounts only.**
> It does NOT create accounts, bypass limits, use CAPTCHA solvers,
> or perform any anti-detection/evasion. All accounts must be owned
> or explicitly authorized by their owners.

## Key Principles

| ✅ This Project Does | ❌ This Project Does NOT |
|---|---|
| Pool authorized account quotas | Create accounts automatically |
| Schedule jobs across providers | Bypass provider usage limits |
| Checkpoint and resume training | Use CAPTCHA bypass or fingerprint spoofing |
| Encrypt credentials securely | Share credentials with agents or scripts |
| Track usage per owner/account | Claim "unlimited free GPU" |
| Enforce daily/weekly limits | Perform ban evasion |
| Auto-failover when leases expire | Claim ownership of others' accounts |

## Features

- **Auto Loop Daemon**: Continuous background scheduling — auto-starts queued jobs, auto-fails over on lease expiry, auto-checkpoints before expiry, auto-health-checks accounts
- **12 GPU Providers**: Google Colab, Kaggle, HuggingFace, Paperspace, SageMaker, Lightning AI, Codesphere, Oracle Cloud, GCP, Intel DevCloud, Deepnote, NVIDIA vGPU
- **Account Pooling**: Add accounts from family members (papa, mama, adik) with per-owner quotas
- **Smart Scheduler**: Selects best account based on quota, cooldown, priority, health, and provider capability
- **AI Agent Interface**: Agents request GPU via API without seeing credentials
- **SQLite State**: All state (accounts, jobs, leases, quota, audit) in a local SQLite database
- **Credential Encryption**: OS keychain (preferred) or Fernet encryption — plaintext is DISABLED
- **Checkpoint/Resume**: Training jobs checkpoint before lease expiry and resume on failover
- **Audit Logging**: All important actions are recorded and viewable in TUI
- **Quota Enforcement**: Daily and weekly limits per account, with cooldown tracking

## Auto Mode

Auto mode runs a background daemon that continuously manages the training lifecycle:

```
┌──────────────────────────────────────────────────────────┐
│                    AUTO LOOP DAEMON                       │
├──────────────────────────────────────────────────────────┤
│  Every 30s: Check for expired leases                     │
│             → Auto-failover to new account               │
│             → Preemptive checkpoint before expiry        │
│                                                          │
│  Every 15s: Check queued jobs                            │
│             → Auto-start when capacity available          │
│             → Score-based account selection              │
│                                                          │
│  Every 60s: Heartbeat active leases                      │
│                                                          │
│  Every 5m:  Health check accounts                        │
│             → Auto-disable failing accounts              │
│             → Auto-re-enable after cooldown              │
└──────────────────────────────────────────────────────────┘
```

### Starting Auto Mode

```bash
# Launch TUI with auto loop
python run.py --auto

# Launch TUI with custom intervals
python run.py --auto --lease-check-interval 15 --queue-check-interval 10

# Launch API with auto loop
python run.py --auto --api --api-port 8420

# Disable specific auto features
python run.py --auto --no-failover          # No auto-failover
python run.py --auto --no-auto-start        # No auto-starting queued jobs
python run.py --auto --no-health-check      # No auto health checks
```

### Auto Loop in TUI

When running with `--auto`, the TUI shows an "Auto Loop" tab with:
- Start/Stop controls (also toggle with keyboard `A`)
- Live stats (leases checked, failovers triggered, jobs started)
- Configuration display
- Activity log of auto loop events

### Auto Loop via API

```bash
# Start the auto loop daemon
curl -X POST http://localhost:8420/autoloop/start

# Check auto loop status
curl http://localhost:8420/autoloop

# Stop the auto loop daemon
curl -X POST http://localhost:8420/autoloop/stop

# Force failover for a specific job
curl -X POST http://localhost:8420/autoloop/failover \
  -H "Content-Type: application/json" \
  -d '{"job_id": "job_abc123"}'
```

## Architecture

```text
TUI User ──→ Account Manager ──→ Secret Vault
                  │
                  ▼
            SQLite State DB ←──→ Quota Ledger
                  ▲
            GPU Scheduler ←──→ Auto Loop Daemon
                  ▲                    │
            AI Agent Loop             │
                  │                    │
                  ▼                    ▼
            Training Job Req    Auto-failover
                               Auto-checkpoint
                               Auto-start queued
                               Auto-health-check

Scheduler creates:
  GPU Lease ──→ Provider Adapter Interface
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Colab    Kaggle     GCP ...
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch TUI with auto mode
python run.py --auto

# Or start HTTP API for agents (with auto scheduling)
python run.py --auto --api --api-port 8420

# Check status
python run.py --status
python run.py --status --json
```

## First-Time Setup

1. **Add Owners** — Go to Settings tab, add family members (me, papa, mama, adik)
2. **Add Accounts** — Go to Accounts tab, add GPU accounts for each owner
3. **Submit Jobs** — Go to Jobs tab, submit a training job (or let auto loop handle it)
4. **Monitor** — Check Auto Loop tab for live status, Leases and Usage tabs for details

## Agent API

AI agents can request GPU compute without accessing credentials:

```python
from api import GPUSchedulerAPI
from scheduler.request import JobRequest

api = GPUSchedulerAPI()

request = JobRequest(
    job_name="train-lora-001",
    gpu_profile="medium_gpu",
    max_runtime_minutes=180,
    checkpoint_uri="file:///workspace/checkpoints/train-lora-001",
    entrypoint="train.py",
    priority="normal",
)

result = api.request_gpu(request)
# result.status = "accepted"
# result.provider = "kaggle"
# result.account_owner = "mama"  # Agent sees owner, NOT credentials
```

Or via HTTP with auto loop:

```bash
# Submit job (auto loop will schedule it)
curl -X POST http://localhost:8420/jobs \
  -H "Content-Type: application/json" \
  -d '{"job_name": "train-lora-001", "gpu_profile": "small_gpu", "max_runtime_minutes": 180, "checkpoint_uri": "file:///ckpt/job1"}'

# Check status
curl http://localhost:8420/jobs/job_abc123

# Cancel
curl -X POST http://localhost:8420/jobs/job_abc123/cancel

# Check auto loop
curl http://localhost:8420/autoloop
```

## Provider Classes

| Class | Providers | Automation Level |
|-------|-----------|-----------------|
| **A — API/SSH** | Oracle Cloud, GCP, Paperspace, Lightning AI, Codesphere | Full auto (SSH) or API |
| **B — Notebook** | Google Colab, Kaggle, SageMaker, Deepnote | Partial (Kaggle auto, others manual) |
| **C — Special** | HuggingFace, Intel DevCloud, NVIDIA vGPU | Manual required in MVP |

> **Note**: HuggingFace Spaces (ZeroGPU) is designed for inference/demos,
> NOT for long-running training. Use Kaggle or Oracle Cloud SSH for actual training.

## Security Model

```text
TUI input → Secret Vault encrypt → SQLite stores only credential_ref
                                    ↓
                            Scheduler requests at runtime
                                    ↓
                            Provider adapter receives in memory only
                                    ↓
                            Credential never logged, never in notebook
```

- ✅ OS keychain (keyring) preferred — credentials never on disk
- ✅ Fernet encryption fallback — .master_key file with 600 permissions
- ❌ Plaintext storage is DISABLED — system refuses to save plaintext
- ✅ Secret scanning before embedding scripts in notebooks
- ✅ Log redaction for API keys, tokens, passwords, private keys
- ✅ .master_key and .env are in .gitignore and never committed

## Training Script Contract

Training scripts must implement:

```python
def save_checkpoint(path: str) -> None: ...
def load_checkpoint(path: str) -> None: ...
def train(resume_from: str | None = None) -> None: ...
```

See `train.py` for a complete example.

## Database Schema

The SQLite database contains these tables:
- **owners** — Account owners (me, papa, mama, adik)
- **providers** — 12 GPU provider definitions
- **accounts** — User accounts linked to owners and providers
- **jobs** — Training job requests and their status
- **leases** — GPU lease lifecycle (pending → running → completed/failed/expired)
- **quota_ledger** — Usage tracking per account/day/week
- **provider_health** — Health check results
- **audit_logs** — Append-only audit trail
- **secrets_metadata** — Credential metadata (not the secrets themselves)

## Running Tests

```bash
python -m pytest tests/ -v
# Or individually:
python tests/test_repositories.py
python tests/test_vault.py
python tests/test_scheduler.py
python tests/test_providers.py
python tests/test_autoloop.py
```

## License

MIT — Use responsibly. This tool is designed for managing YOUR OWN authorized accounts only.

## Disclaimer

This project helps manage authorized GPU accounts. It is NOT designed for:
- Creating multiple accounts to circumvent provider limits
- Bypassing CAPTCHAs or security measures
- Evading bans or detection systems
- Any activity that violates provider terms of service

Users are solely responsible for ensuring their use complies with all applicable provider terms and conditions.
