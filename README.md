# FamilyGPU Orchestrator

> **Quota-Aware Multi-Account GPU Scheduler**

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

## Features

- **12 GPU Providers**: Google Colab, Kaggle, HuggingFace, Paperspace, SageMaker, Lightning AI, Codesphere, Oracle Cloud, GCP, Intel DevCloud, Deepnote, NVIDIA vGPU
- **Account Pooling**: Add accounts from family members (papa, mama, adik) with per-owner quotas
- **Smart Scheduler**: Selects best account based on quota, cooldown, priority, health, and provider capability
- **AI Agent Interface**: Agents request GPU via API without seeing credentials
- **SQLite State**: All state (accounts, jobs, leases, quota, audit) in a local SQLite database
- **Credential Encryption**: OS keychain (preferred) or Fernet encryption — plaintext is DISABLED
- **Checkpoint/Resume**: Training jobs checkpoint before lease expiry and resume on failover
- **Audit Logging**: All important actions are recorded and viewable in TUI
- **Quota Enforcement**: Daily and weekly limits per account, with cooldown tracking

## Architecture

```text
TUI User ──→ Account Manager ──→ Secret Vault
                  │
                  ▼
            SQLite State DB ←──→ Quota Ledger
                  ▲
            GPU Scheduler
                  ▲
            AI Agent Loop
                  │
                  ▼
            Training Job Req

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

# Launch TUI
python run.py

# Or start HTTP API for agents
python run.py --api --api-port 8420

# Check status
python run.py --status
python run.py --status --json
```

## First-Time Setup

1. **Add Owners** — Go to Settings tab, add family members (me, papa, mama, adik)
2. **Add Accounts** — Go to Accounts tab, add GPU accounts for each owner
3. **Submit Jobs** — Go to Jobs tab, submit a training job
4. **Monitor** — Check Leases and Usage tabs for real-time status

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

Or via HTTP:

```bash
# Submit job
curl -X POST http://localhost:8420/jobs \
  -H "Content-Type: application/json" \
  -d '{"job_name": "train-lora-001", "gpu_profile": "small_gpu", "max_runtime_minutes": 180, "checkpoint_uri": "file:///ckpt/job1"}'

# Check status
curl http://localhost:8420/jobs/job_abc123

# Cancel
curl -X POST http://localhost:8420/jobs/job_abc123/cancel
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
