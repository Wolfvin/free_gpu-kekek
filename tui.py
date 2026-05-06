"""FamilyGPU Orchestrator TUI — Textual-based interface.

A quota-aware multi-account GPU scheduler with screens for:
  - Accounts management (add, list, update policy, disable)
  - Jobs management (submit, view queue, view active, stop)
  - Leases view
  - Usage/ledger view
  - Audit logs view
  - System settings
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import (
    Header, Footer, Static, Button, Input, Select, DataTable,
    TabbedContent, TabPane, Label, TextArea,
)
from textual.reactive import reactive
from textual import work

from db.connection import init_db, DB_PATH
from db.repositories import (
    OwnerRepository, AccountRepository, ProviderRepository,
    JobRepository, LeaseRepository, QuotaLedgerRepository,
    AuditLogRepository, HealthRepository,
)
from scheduler.request import JobRequest, GPU_PROFILES
from api import GPUSchedulerAPI
from vault import get_storage_mode, encrypt_credentials, get_credential_status

logger = logging.getLogger("fgt.tui")


# ── Reactive Status Bar ─────────────────────────────────────────

class StatusBar(Static):
    """Bottom status bar showing vault mode and account count."""

    vault_mode: reactive[str] = reactive("")
    account_count: reactive[int] = reactive(0)
    active_leases: reactive[int] = reactive(0)

    def watch_vault_mode(self, mode: str):
        self._update()

    def watch_account_count(self, count: int):
        self._update()

    def watch_active_leases(self, count: int):
        self._update()

    def _update(self):
        self.update(
            f"  Vault: {self.vault_mode}  |  "
            f"Accounts: {self.account_count}  |  "
            f"Active Leases: {self.active_leases}"
        )


# ── Accounts Screen ─────────────────────────────────────────────

class AccountsScreen(VerticalScroll):
    """Screen for managing accounts."""

    def __init__(self, api: GPUSchedulerAPI, **kwargs):
        super().__init__(**kwargs)
        self.api = api

    def compose(self) -> ComposeResult:
        yield Label("## Account Management", classes="title")
        yield Horizontal(
            Button("Add Account", id="add-account-btn", variant="success"),
            Button("Refresh", id="refresh-accounts-btn", variant="primary"),
            Button("Health Check", id="health-check-btn", variant="warning"),
        )
        yield DataTable(id="accounts-table")

    def on_mount(self):
        table = self.query_one("#accounts-table", DataTable)
        table.add_columns("ID", "Owner", "Provider", "Label", "Status",
                          "Priority", "Daily Limit", "Weekly Limit", "Credential")
        self._refresh()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "refresh-accounts-btn":
            self._refresh()
        elif event.button.id == "add-account-btn":
            self.app.push_screen(AddAccountModal(self.api))
        elif event.button.id == "health-check-btn":
            self._health_check()

    def _refresh(self):
        table = self.query_one("#accounts-table", DataTable)
        table.clear()
        accounts = self.api.account_repo.list_all()
        for a in accounts:
            cred_status = get_credential_status(a.get("credential_ref", ""))
            table.add_row(
                a["id"][:12],
                a.get("owner_id", "?"),
                a["provider_key"],
                a["label"],
                a["status"],
                str(a["priority"]),
                f"{a['daily_limit_minutes']}m",
                f"{a['weekly_limit_minutes']}m",
                cred_status,
            )

    def _health_check(self):
        """Run health check on all accounts."""
        from providers.registry import get_adapter
        accounts = self.api.account_repo.list_all(status="active")
        for a in accounts:
            adapter = get_adapter(a["provider_key"])
            if adapter:
                try:
                    result = adapter.health_check(a)
                    self.api.health_repo.record(
                        account_id=a["id"],
                        provider_key=a["provider_key"],
                        status="ok" if result.ok else "down",
                        message=result.message,
                    )
                except Exception as e:
                    self.api.health_repo.record(
                        account_id=a["id"],
                        provider_key=a["provider_key"],
                        status="error",
                        message=str(e),
                    )
        self._refresh()


# ── Add Account Modal ───────────────────────────────────────────

class AddAccountModal(Container):
    """Modal for adding a new account."""

    def __init__(self, api: GPUSchedulerAPI, **kwargs):
        super().__init__(**kwargs)
        self.api = api

    def compose(self) -> ComposeResult:
        yield Label("## Add Account", classes="title")

        owners = self.api.account_repo  # Will use OwnerRepository
        owner_repo = OwnerRepository()
        owner_list = owner_repo.list_all()
        owner_options = [(o["name"], o["id"]) for o in owner_list]

        if not owner_list:
            yield Label("⚠ No owners configured. Add an owner first (System > Owners).")

        yield Label("Owner:")
        yield Select(owner_options, id="add-owner", prompt="Select owner...")

        providers = self.api.account_repo  # Will use ProviderRepository
        provider_repo = ProviderRepository()
        provider_list = provider_repo.list_all(enabled_only=True)
        provider_options = [(p["display_name"], p["key"]) for p in provider_list]

        yield Label("Provider:")
        yield Select(provider_options, id="add-provider", prompt="Select provider...")

        yield Label("Account Label:")
        yield Input(id="add-label", placeholder="e.g. mama-kaggle-1")

        yield Label("Priority (1-10):")
        yield Input(id="add-priority", value="5")

        yield Label("Daily Limit (minutes):")
        yield Input(id="add-daily-limit", value="120")

        yield Label("Weekly Limit (minutes):")
        yield Input(id="add-weekly-limit", value="600")

        yield Horizontal(
            Button("Create", id="create-account-btn", variant="success"),
            Button("Cancel", id="cancel-account-btn", variant="default"),
        )

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "cancel-account-btn":
            self.app.pop_screen()
            return

        if event.button.id == "create-account-btn":
            self._create_account()

    def _create_account(self):
        try:
            owner_id = self.query_one("#add-owner", Select).value
            provider_key = self.query_one("#add-provider", Select).value
            label = self.query_one("#add-label", Input).value
            priority = int(self.query_one("#add-priority", Input).value or "5")
            daily = int(self.query_one("#add-daily-limit", Input).value or "120")
            weekly = int(self.query_one("#add-weekly-limit", Input).value or "600")

            if not owner_id or not provider_key or not label:
                self.app.notify("Owner, provider, and label are required", severity="error")
                return

            # Create account with a credential reference
            # Credentials will be stored via vault when user sets them
            credential_ref = f"keyring:{provider_key}:{label}"

            account = self.api.account_repo.create(
                owner_id=owner_id,
                provider_key=provider_key,
                label=label,
                credential_ref=credential_ref,
                priority=priority,
                daily_limit_minutes=daily,
                weekly_limit_minutes=weekly,
            )

            if account:
                self.api.audit_repo.log(
                    action="add_account",
                    entity_type="account",
                    entity_id=account["id"],
                    message=f"Account added: {label} ({provider_key})",
                )
                self.app.notify(f"Account created: {label}", severity="information")
                self.app.pop_screen()
            else:
                self.app.notify("Failed to create account", severity="error")

        except ValueError as e:
            self.app.notify(f"Invalid input: {e}", severity="error")


# ── Jobs Screen ─────────────────────────────────────────────────

class JobsScreen(VerticalScroll):
    """Screen for managing jobs."""

    def __init__(self, api: GPUSchedulerAPI, **kwargs):
        super().__init__(**kwargs)
        self.api = api

    def compose(self) -> ComposeResult:
        yield Label("## Job Management", classes="title")
        yield Horizontal(
            Button("Submit Job", id="submit-job-btn", variant="success"),
            Button("Refresh", id="refresh-jobs-btn", variant="primary"),
        )
        yield Label("### Queued/Active Jobs")
        yield DataTable(id="active-jobs-table")
        yield Label("### Job History")
        yield DataTable(id="history-jobs-table")

    def on_mount(self):
        # Active jobs table
        active_table = self.query_one("#active-jobs-table", DataTable)
        active_table.add_columns("ID", "Name", "Status", "Profile", "Provider", "Started")

        # History table
        history_table = self.query_one("#history-jobs-table", DataTable)
        history_table.add_columns("ID", "Name", "Status", "Profile", "Completed", "Reason")
        self._refresh()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "refresh-jobs-btn":
            self._refresh()
        elif event.button.id == "submit-job-btn":
            self.app.push_screen(SubmitJobModal(self.api))

    def _refresh(self):
        # Active jobs
        active_table = self.query_one("#active-jobs-table", DataTable)
        active_table.clear()
        for status in ("queued", "starting", "running"):
            for j in self.api.job_repo.list_all(status=status):
                lease = self.api.lease_repo.get_active_for_job(j["id"])
                provider = lease["provider_key"] if lease else "-"
                started = j.get("started_at", "-") or "-"
                active_table.add_row(
                    j["id"][:12], j["name"], j["status"],
                    j["gpu_profile"], provider,
                    started[:16] if started != "-" else "-",
                )

        # History
        history_table = self.query_one("#history-jobs-table", DataTable)
        history_table.clear()
        for status in ("completed", "failed", "cancelled"):
            for j in self.api.job_repo.list_all(status=status):
                completed = j.get("completed_at", "-") or "-"
                reason = j.get("failure_reason", "-") or "-"
                history_table.add_row(
                    j["id"][:12], j["name"], j["status"],
                    j["gpu_profile"],
                    completed[:16] if completed != "-" else "-",
                    reason[:40] if reason != "-" else "-",
                )


# ── Submit Job Modal ────────────────────────────────────────────

class SubmitJobModal(Container):
    """Modal for submitting a new training job."""

    def __init__(self, api: GPUSchedulerAPI, **kwargs):
        super().__init__(**kwargs)
        self.api = api

    def compose(self) -> ComposeResult:
        yield Label("## Submit Training Job", classes="title")

        yield Label("Job Name:")
        yield Input(id="job-name", placeholder="e.g. train-lora-001")

        yield Label("GPU Profile:")
        profile_options = [(f"{k} — {v['description']}", k) for k, v in GPU_PROFILES.items()]
        yield Select(profile_options, id="job-profile", value="small_gpu")

        yield Label("Max Runtime (minutes):")
        yield Input(id="job-runtime", value="180")

        yield Label("Priority:")
        yield Select([("Low", "low"), ("Normal", "normal"), ("High", "high")],
                      id="job-priority", value="normal")

        yield Label("Checkpoint URI:")
        yield Input(id="job-checkpoint", value="file:///workspace/checkpoints/job")

        yield Label("Entrypoint Script:")
        yield Input(id="job-entrypoint", value="train.py")

        yield Horizontal(
            Button("Submit", id="submit-btn", variant="success"),
            Button("Cancel", id="cancel-btn", variant="default"),
        )

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "cancel-btn":
            self.app.pop_screen()
            return

        if event.button.id == "submit-btn":
            self._submit()

    def _submit(self):
        try:
            name = self.query_one("#job-name", Input).value
            profile = self.query_one("#job-profile", Select).value
            runtime = int(self.query_one("#job-runtime", Input).value or "180")
            priority = self.query_one("#job-priority", Select).value
            checkpoint = self.query_one("#job-checkpoint", Input).value
            entrypoint = self.query_one("#job-entrypoint", Input).value

            if not name:
                self.app.notify("Job name is required", severity="error")
                return

            request = JobRequest(
                job_name=name,
                gpu_profile=profile,
                max_runtime_minutes=runtime,
                priority=priority,
                checkpoint_uri=checkpoint,
                entrypoint=entrypoint,
                checkpoint_required=True,
            )

            result = self.api.request_gpu(request)

            if result.status == "accepted":
                self.app.notify(
                    f"Job submitted! Provider: {result.provider}, "
                    f"Owner: {result.account_owner}",
                    severity="information",
                )
            else:
                self.app.notify(
                    f"Job rejected: {result.message}",
                    severity="warning",
                )

            self.app.pop_screen()

        except ValueError as e:
            self.app.notify(f"Invalid input: {e}", severity="error")


# ── Leases Screen ───────────────────────────────────────────────

class LeasesScreen(VerticalScroll):
    """Screen for viewing leases."""

    def __init__(self, api: GPUSchedulerAPI, **kwargs):
        super().__init__(**kwargs)
        self.api = api

    def compose(self) -> ComposeResult:
        yield Label("## GPU Leases", classes="title")
        yield Horizontal(
            Button("Refresh", id="refresh-leases-btn", variant="primary"),
        )
        yield DataTable(id="leases-table")

    def on_mount(self):
        table = self.query_one("#leases-table", DataTable)
        table.add_columns("ID", "Job", "Account", "Provider", "Status",
                          "Started", "Expires", "Runtime")
        self._refresh()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "refresh-leases-btn":
            self._refresh()

    def _refresh(self):
        table = self.query_one("#leases-table", DataTable)
        table.clear()
        leases = self.api.lease_repo.list_all()
        for l in leases:
            started = l.get("started_at", "-") or "-"
            expires = l.get("expires_at", "-") or "-"
            runtime = f"{l.get('runtime_minutes', 0)}m"
            table.add_row(
                l["id"][:12], l["job_id"][:12], l["account_id"][:12],
                l["provider_key"], l["status"],
                started[:16] if started != "-" else "-",
                expires[:16] if expires != "-" else "-",
                runtime,
            )


# ── Usage Screen ────────────────────────────────────────────────

class UsageScreen(VerticalScroll):
    """Screen for viewing quota usage."""

    def __init__(self, api: GPUSchedulerAPI, **kwargs):
        super().__init__(**kwargs)
        self.api = api

    def compose(self) -> ComposeResult:
        yield Label("## Quota Usage", classes="title")
        yield Horizontal(
            Button("By Provider", id="usage-provider-btn", variant="primary"),
            Button("By Owner", id="usage-owner-btn", variant="primary"),
            Button("Refresh", id="refresh-usage-btn", variant="default"),
        )
        yield DataTable(id="usage-table")

    def on_mount(self):
        self._show_by_provider()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "usage-provider-btn":
            self._show_by_provider()
        elif event.button.id == "usage-owner-btn":
            self._show_by_owner()
        elif event.button.id == "refresh-usage-btn":
            self._show_by_provider()

    def _show_by_provider(self):
        table = self.query_one("#usage-table", DataTable)
        table.clear()
        table.add_columns("Provider", "Entries", "Total Minutes", "Total Hours")
        summary = self.api.quota_repo.get_usage_summary(group_by="provider")
        for s in summary:
            table.add_row(
                s["provider_key"],
                str(s["entry_count"]),
                str(s["total_minutes"]),
                f"{s['total_minutes'] / 60:.1f}",
            )

    def _show_by_owner(self):
        table = self.query_one("#usage-table", DataTable)
        table.clear()
        table.add_columns("Owner", "Entries", "Total Minutes", "Total Hours")
        summary = self.api.quota_repo.get_usage_summary(group_by="owner")
        for s in summary:
            table.add_row(
                s["owner_id"],
                str(s["entry_count"]),
                str(s["total_minutes"]),
                f"{s['total_minutes'] / 60:.1f}",
            )


# ── Audit Screen ────────────────────────────────────────────────

class AuditScreen(VerticalScroll):
    """Screen for viewing audit logs."""

    def __init__(self, api: GPUSchedulerAPI, **kwargs):
        super().__init__(**kwargs)
        self.api = api

    def compose(self) -> ComposeResult:
        yield Label("## Audit Logs", classes="title")
        yield Horizontal(
            Button("Refresh", id="refresh-audit-btn", variant="primary"),
        )
        yield DataTable(id="audit-table")

    def on_mount(self):
        table = self.query_one("#audit-table", DataTable)
        table.add_columns("Time", "Actor", "Action", "Entity", "Message")
        self._refresh()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "refresh-audit-btn":
            self._refresh()

    def _refresh(self):
        table = self.query_one("#audit-table", DataTable)
        table.clear()
        logs = self.api.audit_repo.list_recent(limit=50)
        for log in logs:
            table.add_row(
                log["created_at"][:19],
                log.get("actor", "-"),
                log["action"],
                f"{log['entity_type']}:{log['entity_id'][:8]}" if log.get("entity_id") else log["entity_type"],
                (log.get("message") or "")[:60],
            )


# ── Settings Screen ─────────────────────────────────────────────

class SettingsScreen(VerticalScroll):
    """Screen for system settings and owner management."""

    def __init__(self, api: GPUSchedulerAPI, **kwargs):
        super().__init__(**kwargs)
        self.api = api

    def compose(self) -> ComposeResult:
        yield Label("## System Settings", classes="title")

        yield Label("### Vault Status")
        yield Static(f"Storage Mode: {get_storage_mode()}", id="vault-status")

        yield Label("### Owners")
        yield Horizontal(
            Input(id="owner-id", placeholder="Owner ID (e.g. me, papa, mama, adik)"),
            Input(id="owner-name", placeholder="Display Name"),
            Input(id="owner-relationship", placeholder="Relationship (e.g. father, sister)"),
            Button("Add Owner", id="add-owner-btn", variant="success"),
        )
        yield DataTable(id="owners-table")

        yield Label("### Provider Registry")
        yield DataTable(id="providers-table")

    def on_mount(self):
        # Owners table
        owners_table = self.query_one("#owners-table", DataTable)
        owners_table.add_columns("ID", "Name", "Relationship", "Consent Note")
        self._refresh_owners()

        # Providers table
        providers_table = self.query_one("#providers-table", DataTable)
        providers_table.add_columns("Key", "Name", "Class", "Enabled", "Session Limit", "Cooldown")
        provider_repo = ProviderRepository()
        for p in provider_repo.list_all():
            session_min = p.get("default_session_limit_minutes", "-") or "-"
            cooldown = p.get("default_cooldown_minutes", "-") or "-"
            providers_table.add_row(
                p["key"], p["display_name"], p["provider_class"],
                "Yes" if p["enabled"] else "No",
                f"{session_min}m" if session_min != "-" else "-",
                f"{cooldown}m" if cooldown != "-" else "-",
            )

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "add-owner-btn":
            self._add_owner()

    def _add_owner(self):
        owner_id = self.query_one("#owner-id", Input).value.strip()
        name = self.query_one("#owner-name", Input).value.strip()
        relationship = self.query_one("#owner-relationship", Input).value.strip()

        if not owner_id or not name:
            self.app.notify("Owner ID and name are required", severity="error")
            return

        owner_repo = OwnerRepository()
        existing = owner_repo.get_by_id(owner_id)
        if existing:
            self.app.notify(f"Owner '{owner_id}' already exists", severity="error")
            return

        owner = owner_repo.create(
            id=owner_id,
            name=name,
            relationship=relationship,
            consent_note=f"Authorized by {owner_id}",
        )

        if owner:
            self.api.audit_repo.log(
                action="add_owner",
                entity_type="owner",
                entity_id=owner_id,
                message=f"Owner added: {name} ({relationship})",
            )
            self.app.notify(f"Owner added: {name}", severity="information")
            self._refresh_owners()
        else:
            self.app.notify("Failed to add owner", severity="error")

    def _refresh_owners(self):
        owners_table = self.query_one("#owners-table", DataTable)
        owners_table.clear()
        owner_repo = OwnerRepository()
        for o in owner_repo.list_all():
            owners_table.add_row(o["id"], o["name"], o.get("relationship", "-"), o.get("consent_note", "-"))


# ── Main TUI Application ────────────────────────────────────────

class FamilyGPUTUI(App):
    """FamilyGPU Orchestrator — Quota-Aware Multi-Account GPU Scheduler.

    A TUI application for managing family GPU accounts, submitting
    training jobs, and monitoring usage across 12 providers.
    """

    TITLE = "FamilyGPU Orchestrator"
    SUB_TITLE = "Quota-Aware Multi-Account GPU Scheduler"
    CSS_PATH = None  # We'll use inline CSS

    CSS = """
    Screen {
        layout: vertical;
    }
    .title {
        text-style: bold;
        margin: 1 0;
        color: $text;
    }
    Horizontal {
        height: auto;
        margin: 1 0;
    }
    DataTable {
        height: auto;
        max-height: 20;
    }
    TabbedContent {
        height: 1fr;
    }
    StatusBar {
        dock: bottom;
        background: $primary;
        color: $text;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, db_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.db_path = db_path
        self.api: Optional[GPUSchedulerAPI] = None

    def on_mount(self):
        # Initialize database and API
        self.api = GPUSchedulerAPI(db_path=self.db_path)
        self._update_status_bar()

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Accounts", id="tab-accounts"):
                yield AccountsScreen(self.api) if self.api else Static("Loading...")
            with TabPane("Jobs", id="tab-jobs"):
                yield JobsScreen(self.api) if self.api else Static("Loading...")
            with TabPane("Leases", id="tab-leases"):
                yield LeasesScreen(self.api) if self.api else Static("Loading...")
            with TabPane("Usage", id="tab-usage"):
                yield UsageScreen(self.api) if self.api else Static("Loading...")
            with TabPane("Audit", id="tab-audit"):
                yield AuditScreen(self.api) if self.api else Static("Loading...")
            with TabPane("Settings", id="tab-settings"):
                yield SettingsScreen(self.api) if self.api else Static("Loading...")
        yield StatusBar(id="status-bar")
        yield Footer()

    def _update_status_bar(self):
        if not self.api:
            return
        try:
            bar = self.query_one("#status-bar", StatusBar)
            bar.vault_mode = get_storage_mode()
            bar.account_count = len(self.api.account_repo.list_all())
            bar.active_leases = len(self.api.lease_repo.list_active())
        except Exception:
            pass

    def action_refresh(self):
        """Refresh the current tab."""
        self._update_status_bar()
        self.notify("Refreshed", severity="information")


def run_app():
    """Entry point for the TUI application."""
    app = FamilyGPUTUI()
    app.run()


if __name__ == "__main__":
    run_app()
