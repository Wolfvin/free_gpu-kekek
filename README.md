# GPU Account Scheduler

A terminal-based tool and CLI for scheduling AI training jobs across GPU platform accounts you own. Designed for AI agents and power users who need to maximize their own GPU allocation across platforms like Kaggle, Oracle Cloud, and GCP.

> **⚠️ Compliance Notice:** This tool is designed for scheduling workloads across accounts and platforms **you legitimately own or are authorized to use**. Rotating between multiple accounts on the same platform to circumvent usage limits may violate that platform's Terms of Service. Users are responsible for ensuring their usage complies with each platform's policies. The authors of this tool do not endorse or encourage any violation of platform terms.

## Features

- **12 GPU Platforms** supported: Google Colab, Kaggle, HuggingFace Spaces, Paperspace, SageMaker Studio Lab, Lightning AI, Codesphere, Oracle Cloud, GCP, Intel DevCloud, Deepnote, NVIDIA vGPU
- **AUTO platforms** (Kaggle, Oracle Cloud SSH, GCP SSH) — fully automated push, start, status check, and stop
- **MANUAL platforms** (Colab, notebooks) — generates notebooks for manual upload, requires `/confirm`
- **Account stacking** — add multiple accounts per platform for longer continuous training
- **Auto-rotation** — automatically rotates to the next account when session time is about to expire
- **Checkpoint support** — saves and resumes training checkpoints across sessions
- **Encrypted credentials** — OS keychain (keyring) preferred, Fernet encryption fallback, plaintext storage is **disabled**
- **Headless CLI** — full JSON API for AI agent integration (`--status --json`, `--start`, `--stop`, `--confirm`, `--done`, `--schema`, `--platforms`)
- **Event callbacks** — `on_session_confirmed`, `on_session_expired`, `on_rotation_needed`, `on_no_accounts` for programmatic control

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch interactive TUI
python run.py

# Or use headless CLI
python tui.py --status              # Human-readable status
python tui.py --status --json       # JSON for agents
python tui.py --start               # Start training headlessly
python tui.py --confirm             # Confirm session headlessly
python tui.py --stop                # Stop training
python tui.py --done                # Signal training complete
python tui.py --schema kaggle       # Show credential schema
python tui.py --platforms           # List all platforms
```

## TUI Commands

| Command | Description |
|---------|-------------|
| `/add` | Add platform → enter credentials → stack accounts |
| `/remove` | Remove a platform |
| `/choose` | Force next session to a specific platform |
| `/start` | Start training (AUTO platforms confirm automatically) |
| `/confirm` | Confirm training is running (required for MANUAL platforms) |
| `/stop` | Stop training |
| `/done` | Signal training complete, rotate to next account early |
| `/status` | Show current session + platform status |
| `/save` | Save config to config.yaml |
| `/reset` | Reset weekly counters |
| `/help` | Show all commands |

## Platform Types

- **AUTO** (green) — Push + start + status all automated via API or SSH. No manual intervention needed.
- **MANUAL** (yellow) — Generates notebook for manual upload. Requires `/confirm` after you start it in browser.

## Credential Security

Credentials are encrypted before storage:
1. **OS Keychain** (preferred) — via `keyring` library. Credentials never touch disk as plaintext.
2. **Fernet Encryption** — via `cryptography` library. Encrypted with `.master_key` file.
3. **Plaintext** — **DISABLED**. The tool will refuse to save credentials if neither keyring nor cryptography is installed.

## Important: HuggingFace Spaces

HuggingFace Spaces (ZeroGPU) is designed for **hosting ML demos and inference endpoints**, NOT for long-running training jobs. The ZeroGPU quota is per-request (seconds to minutes), not per-session. Use Kaggle or Oracle Cloud SSH for actual training workloads.

## Legal Disclaimer

This software is provided "as is" without warranty of any kind. Users must:
- Only use accounts and platforms they are authorized to access
- Comply with each platform's Terms of Service
- Not use this tool to circumvent platform usage limits or restrictions
- Be aware that some platforms prohibit automated access or account rotation

The tool's "account stacking" feature is intended for users with **multiple legitimately-owned accounts** (e.g., different projects, team accounts) on the same platform. Using it to create fake accounts or circumvent free tier limits is not endorsed.

## Architecture

```
tui.py          — TUI app + headless CLI entry point
handlers.py     — Platform integrations (API, SSH, notebook generators)
session.py      — Session manager with rotation, events, state persistence
platforms.py    — Platform definitions and credential schemas
vault.py        — Credential encryption (keyring/Fernet)
trainer.py      — Training job management with checkpoint support
train.py        — Sample training script (replace with your own)
run.py          — Quick launcher
```

## License

MIT
