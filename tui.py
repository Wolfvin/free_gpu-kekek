"""Free GPU Trainer — TUI Application.

A terminal-based tool for managing and rotating free GPU platform accounts
for continuous AI training and fine-tuning.

Session lifecycle:
  /start  → Session created in PENDING state (timer not counting)
  Auto-platforms (Kaggle, SSH): auto-confirmed after successful push
  Manual platforms (Colab, notebooks): require /confirm from user
  /confirm → Session confirmed, countdown timer starts
  Timer expires → Auto-rotate to next account (or end if none)

Usage:
    python tui.py                       # Launch TUI
    python tui.py --status              # Show human-readable status
    python tui.py --status --json       # Show machine-readable JSON status
    python tui.py --start               # Start training headlessly
    python tui.py --confirm             # Confirm current session headlessly
    python tui.py --stop                # Stop training headlessly
    python tui.py --done                # Signal training complete, rotate to next
    python tui.py --schema <platform>   # Print credential schema for a platform
    python tui.py --platforms           # List all platforms + required fields + AUTO/MANUAL

Slash Commands:
    /add         → Pick platform → Enter credentials → See detail + stack accounts
    /remove      → Pick platform → Remove it
    /choose      → Pick platform → Force next session there
    /start       Start training (session created in PENDING — needs /confirm for manual platforms)
    /confirm     Confirm training is running on platform (starts countdown)
    /stop        Stop training
    /save        Save config
    /help        Show commands
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from platforms import (
    PlatformConfig, AccountConfig, PlatformStatus, build_platform,
    PLATFORM_DEFS, CREDENTIAL_SCHEMAS,
)
from session import Session, SessionManager, SessionPhase
from vault import encrypt_credentials, delete_credentials, get_storage_mode
from handlers import validate_account_name, is_auto_platform, is_manual_platform, platform_type_label, get_handler

import time
import threading
import yaml
import logging
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import (
    Header, Footer, Static, Button, Label, ProgressBar,
    DataTable, RichLog, TabbedContent, TabPane, Input, OptionList,
)
from textual.screen import ModalScreen
from textual.coordinate import Coordinate
from rich.text import Text


# ── Helpers ────────────────────────────────────────────────────────

STATUS_ICONS = {
    PlatformStatus.AVAILABLE: "[green]●[/green]",
    PlatformStatus.IN_USE: "[yellow]●[/yellow]",
    PlatformStatus.COOLDOWN: "[cyan]●[/cyan]",
    PlatformStatus.EXHAUSTED: "[red]●[/red]",
    PlatformStatus.DISABLED: "[dim]●[/dim]",
    PlatformStatus.WEEKLY_LIMIT: "[orange3]●[/orange3]",
    PlatformStatus.TRIAL_EXPIRED: "[red]●[/red]",
}

STATUS_COLORS = {
    PlatformStatus.AVAILABLE: "green",
    PlatformStatus.IN_USE: "yellow",
    PlatformStatus.COOLDOWN: "cyan",
    PlatformStatus.EXHAUSTED: "red",
    PlatformStatus.DISABLED: "dim",
    PlatformStatus.WEEKLY_LIMIT: "orange3",
    PlatformStatus.TRIAL_EXPIRED: "red",
}

PHASE_ICONS = {
    SessionPhase.PENDING: "[yellow]⏳[/yellow]",
    SessionPhase.CONFIRMED: "[green]▶[/green]",
    SessionPhase.EXPIRED: "[red]⏹[/red]",
    SessionPhase.ENDED: "[dim]⏹[/dim]",
}


def format_seconds(s: float) -> str:
    if s <= 0:
        return "0:00:00"
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h}:{m:02d}:{sec:02d}"


def load_config(config_path: str = "config.yaml") -> dict:
    p = Path(config_path)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f)
    return {"platforms": {}, "training": {}, "logging": {}}


def save_config(config: dict, config_path: str = "config.yaml"):
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def mask_credential(val: str) -> str:
    """Mask a credential value for display: show first 4 + ***."""
    if not val:
        return ""
    if len(val) <= 8:
        return val[:2] + "***"
    return val[:4] + "***" + val[-2:]


# ── Modal: Platform List Picker ────────────────────────────────────

class PlatformListScreen(ModalScreen[str]):
    """Full-screen platform picker. Returns platform key."""

    def __init__(self, title: str, platforms: list, show_empty: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._title = title
        self._platforms = platforms
        self._show_empty = show_empty

    def compose(self) -> ComposeResult:
        with Container(id="plist-outer"):
            yield Label(f"[bold]{self._title}[/bold]", id="plist-title")
            yield OptionList(id="plist-options")
            yield Horizontal(
                Button("Cancel [dim](Esc)[/dim]", id="plist-cancel", variant="default"),
            )

    def on_mount(self) -> None:
        ol = self.query_one("#plist-options", OptionList)
        if self._show_empty:
            for key, defn in PLATFORM_DEFS.items():
                already = any(p.key == key for p in self._platforms)
                tag = " [dim](added)[/dim]" if already else ""
                cred_count = len(CREDENTIAL_SCHEMAS.get(key, []))
                cred_info = f"  Creds: [yellow]{cred_count} fields[/yellow]" if cred_count else "  [dim]No API needed[/dim]"
                ptype = platform_type_label(key)
                ptype_color = "green" if ptype == "AUTO" else "yellow"
                ol.add_option(
                    Text.from_markup(
                        f"[bold]{defn['name']}[/bold]{tag}\n"
                        f"  GPU: [cyan]{defn['gpu_type']}[/cyan]  "
                        f"Session: [white]{defn['session_limit_hours']}h[/white]  "
                        f"Type: [{ptype_color}]{ptype}[/{ptype_color}]\n"
                        f"  URL: [dim]{defn['url']}[/dim]\n"
                        f"{cred_info}"
                    )
                )
        else:
            for p in self._platforms:
                icon = STATUS_ICONS.get(p.status, "?")
                ol.add_option(
                    Text.from_markup(
                        f"{icon} [bold]{p.name}[/bold]\n"
                        f"  GPU: [cyan]{p.gpu_type}[/cyan]  "
                        f"Stack: [white]{p.total_accounts}x[/white] = "
                        f"[green]{p.max_continuous_hours:.0f}h[/green]  "
                        f"[{STATUS_COLORS.get(p.status, 'white')}]{p.status.value}[/]"
                    )
                )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        idx = event.option_index
        if self._show_empty:
            keys = list(PLATFORM_DEFS.keys())
            if idx < len(keys):
                self.dismiss(keys[idx])
        else:
            if idx < len(self._platforms):
                self.dismiss(self._platforms[idx].key)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "plist-cancel":
            self.dismiss("")

    def key_escape(self) -> None:
        self.dismiss("")


# ── Modal: Credential Input ────────────────────────────────────────

class CredentialInputScreen(ModalScreen[dict]):
    """Multi-field credential input form for a platform."""

    def __init__(self, platform_key: str, account_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self.platform_key = platform_key
        self.platform_name = PLATFORM_DEFS[platform_key]["name"]
        self.default_name = account_name
        self.cred_fields = CREDENTIAL_SCHEMAS.get(platform_key, [])

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="cred-outer"):
            yield Label(
                f"[bold]{self.platform_name}[/bold] — Add Account",
                id="cred-title",
            )
            yield Static(
                f"  GPU: [cyan]{PLATFORM_DEFS[self.platform_key]['gpu_type']}[/cyan]  "
                f"Session: [white]{PLATFORM_DEFS[self.platform_key]['session_limit_hours']}h[/white]",
                id="cred-platform-info",
            )

            yield Label("[bold]Account Name[/bold]", classes="cred-field-label")
            yield Input(
                placeholder="e.g. my-colab-1",
                id="cred-name-input",
                value=self.default_name,
            )

            if self.cred_fields:
                yield Label("[bold]Credentials[/bold]", classes="cred-field-label")
                yield Static(
                    "  [dim]Encrypted before saving. Never sent anywhere else.[/dim]",
                    id="cred-disclaimer",
                )
                for cf in self.cred_fields:
                    yield Label(f"  {cf['label']}", classes="cred-field-label")
                    if cf.get("hint"):
                        yield Static(
                            f"    [dim]{cf['hint']}[/dim]",
                            classes="cred-hint",
                        )
                    yield Input(
                        placeholder=cf.get("hint", ""),
                        id=f"cred-field-{cf['key']}",
                        password=cf.get("secret", False),
                    )

            yield Horizontal(
                Button("Add Account", id="cred-submit", variant="success"),
                Button("Skip Credentials", id="cred-skip", variant="default"),
                Button("Cancel [dim](Esc)[/dim]", id="cred-cancel", variant="error"),
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cred-submit":
            self._submit(skip_creds=False)
        elif event.button.id == "cred-skip":
            self._submit(skip_creds=True)
        elif event.button.id == "cred-cancel":
            self.dismiss({})

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "cred-name-input":
            if self.cred_fields:
                first_key = self.cred_fields[0]["key"]
                try:
                    self.query_one(f"#cred-field-{first_key}", Input).focus()
                except Exception:
                    self._submit(skip_creds=False)
            else:
                self._submit(skip_creds=False)
        else:
            self._advance_or_submit(event.input.id)

    def _advance_or_submit(self, current_id: str):
        field_keys = [cf["key"] for cf in self.cred_fields]
        if current_id.startswith("cred-field-"):
            current_key = current_id[len("cred-field-"):]
            try:
                idx = field_keys.index(current_key)
                if idx + 1 < len(field_keys):
                    next_key = field_keys[idx + 1]
                    self.query_one(f"#cred-field-{next_key}", Input).focus()
                    return
            except (ValueError, Exception):
                pass
        self._submit(skip_creds=False)

    def _submit(self, skip_creds: bool = False):
        name_input = self.query_one("#cred-name-input", Input)
        name = name_input.value.strip()
        if not name:
            name_input.focus()
            return

        is_valid, err_msg = validate_account_name(name)
        if not is_valid:
            name_input.value = ""
            name_input.placeholder = f"X {err_msg}"
            name_input.focus()
            return

        creds = {}
        if not skip_creds:
            for cf in self.cred_fields:
                try:
                    inp = self.query_one(f"#cred-field-{cf['key']}", Input)
                    val = inp.value.strip()
                    if val:
                        creds[cf["key"]] = val
                except Exception:
                    pass

        self.dismiss({"name": name, "credentials": creds})

    def key_escape(self) -> None:
        self.dismiss({})


# ── Modal: Platform Detail + Account Stack ─────────────────────────

class PlatformDetailScreen(ModalScreen[str]):
    """Shows platform detail with stacked accounts."""

    def __init__(self, platform: PlatformConfig, config: dict, **kwargs):
        super().__init__(**kwargs)
        self.platform = platform
        self.config = config

    def compose(self) -> ComposeResult:
        p = self.platform
        icon = STATUS_ICONS.get(p.status, "?")
        cred_schema = CREDENTIAL_SCHEMAS.get(p.key, [])
        cred_info = f"  Creds needed: [yellow]{len(cred_schema)} fields[/yellow]" if cred_schema else ""
        ptype = platform_type_label(p.key)
        ptype_color = "green" if ptype == "AUTO" else "yellow"
        ptype_line = f"  Type: [{ptype_color}]{ptype}[/{ptype_color}]"

        # HF warning
        hf_warning = ""
        if p.key == "huggingface":
            hf_warning = "\n  [bold yellow]WARNING:[/bold yellow] [yellow]ZeroGPU Spaces are for inference/demos, NOT long-running training.[/yellow]"

        with VerticalScroll(id="pdetail-outer"):
            yield Label(
                f"{icon} [bold]{p.name}[/bold]", id="pdetail-title"
            )
            yield Static(
                f"  GPU: [cyan]{p.gpu_type}[/cyan]   "
                f"Session: [white]{p.session_limit_hours}h[/white]   "
                f"Cooldown: [white]{p.cooldown_minutes}m[/white]\n"
                f"  Stack: [white]{p.total_accounts}x[/white] accounts = "
                f"[green]{p.max_continuous_hours:.0f}h[/green] continuous\n"
                f"  URL: [dim]{p.url}[/dim]\n"
                f"{ptype_line}{cred_info}{hf_warning}",
                id="pdetail-info",
            )
            yield Label("[bold]Stacked Accounts[/bold]", id="pdetail-acct-label")

            if p.accounts:
                for acc in p.accounts:
                    acc_icon = STATUS_ICONS.get(acc.status, "?")
                    cred_status = ""
                    if cred_schema:
                        filled = sum(1 for cf in cred_schema if acc.credentials.get(cf["key"]))
                        total = len(cred_schema)
                        if filled == total:
                            cred_status = " [green]creds OK[/green]"
                        elif filled > 0:
                            cred_status = f" [yellow]creds {filled}/{total}[/yellow]"
                        else:
                            cred_status = " [red]no creds[/red]"

                    with Horizontal(classes="acct-row"):
                        yield Static(
                            f"  {acc_icon} {acc.name}   "
                            f"[dim]{acc.total_hours_used:.1f}h used[/dim]{cred_status}",
                            classes="acct-name",
                        )
                        yield Button(
                            "Edit Creds", id=f"edit-{acc.name}", variant="primary",
                            classes="acct-edit-btn",
                        )
                        yield Button(
                            "Remove", id=f"rm-{acc.name}", variant="error",
                            classes="acct-rm-btn",
                        )
            else:
                yield Static("  [dim]No accounts yet[/dim]", id="pdetail-no-acct")

            yield Horizontal(
                Button("+ Add Account (with credentials)", id="pdetail-add-btn", variant="success"),
                Button("Done [dim](Esc)[/dim]", id="pdetail-done", variant="default"),
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pdetail-done":
            self.dismiss("done")
        elif event.button.id == "pdetail-add-btn":
            self.dismiss("add_account")
        elif event.button.id and event.button.id.startswith("edit-"):
            name = event.button.id[5:]
            self.dismiss(f"edit_creds:{name}")
        elif event.button.id and event.button.id.startswith("rm-"):
            name = event.button.id[3:]
            self.dismiss(f"remove:{name}")

    def key_escape(self) -> None:
        self.dismiss("done")


# ── Modal: Edit Credentials ────────────────────────────────────────

class EditCredentialScreen(ModalScreen[dict]):
    """Edit credentials for an existing account."""

    def __init__(self, platform_key: str, account: AccountConfig, **kwargs):
        super().__init__(**kwargs)
        self.platform_key = platform_key
        self.platform_name = PLATFORM_DEFS[platform_key]["name"]
        self.account = account
        self.cred_fields = CREDENTIAL_SCHEMAS.get(platform_key, [])

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="editcred-outer"):
            yield Label(
                f"[bold]{self.platform_name}[/bold] — Edit Credentials: {self.account.name}",
                id="editcred-title",
            )
            yield Static(
                "  [dim]Leave a field blank to keep the existing value.[/dim]\n"
                "  [dim]Enter 'CLEAR' to remove a credential.[/dim]",
                id="editcred-hint",
            )

            if self.cred_fields:
                for cf in self.cred_fields:
                    current = self.account.credentials.get(cf["key"], "")
                    masked = mask_credential(current) if current else "[dim]not set[/dim]"
                    yield Label(f"  {cf['label']}", classes="cred-field-label")
                    yield Static(
                        f"    Current: [yellow]{masked}[/yellow]",
                        classes="cred-hint",
                    )
                    if cf.get("hint"):
                        yield Static(
                            f"    [dim]{cf['hint']}[/dim]",
                            classes="cred-hint",
                        )
                    yield Input(
                        placeholder=f"New value (or leave blank to keep)",
                        id=f"editcred-field-{cf['key']}",
                        password=cf.get("secret", False),
                    )
            else:
                yield Static("  [dim]This platform has no credential fields.[/dim]")

            yield Horizontal(
                Button("Save", id="editcred-save", variant="success"),
                Button("Cancel [dim](Esc)[/dim]", id="editcred-cancel", variant="default"),
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "editcred-save":
            self._save()
        elif event.button.id == "editcred-cancel":
            self.dismiss({})

    def _save(self):
        updates = {}
        for cf in self.cred_fields:
            try:
                inp = self.query_one(f"#editcred-field-{cf['key']}", Input)
                val = inp.value.strip()
                if val == "CLEAR":
                    updates[cf["key"]] = None
                elif val:
                    updates[cf["key"]] = val
            except Exception:
                pass
        self.dismiss({"updates": updates})

    def key_escape(self) -> None:
        self.dismiss({})


# ── Modal: Help ────────────────────────────────────────────────────

class HelpScreen(ModalScreen):
    def compose(self) -> ComposeResult:
        with Container(id="help-dialog"):
            yield Label("[bold]Commands[/bold]\n")
            yield Static(
                "[bold]/add[/bold]      Pick platform → enter credentials → stack accounts\n"
                "[bold]/remove[/bold]   Pick platform → remove it\n"
                "[bold]/choose[/bold]   Pick platform → force next session there\n"
                "[bold]/start[/bold]    Start training (auto-platforms confirm automatically)\n"
                "[bold]/confirm[/bold]  Confirm training is running (required for manual platforms)\n"
                "[bold]/stop[/bold]     Stop training\n"
                "[bold]/status[/bold]   Show current session info + platform status\n"
                "[bold]/save[/bold]     Save config to config.yaml\n"
                "[bold]/reset[/bold]    Reset weekly counters\n"
                "[bold]/help[/bold]     Show this help\n\n"
                "[bold]Platform Types:[/bold]\n"
                "  [green]AUTO[/green] = push + start + status all automated (Kaggle, SSH)\n"
                "  [yellow]MANUAL[/yellow] = notebook upload + /confirm required (Colab, etc.)\n\n"
                "[dim]Press / to focus command bar[/dim]"
            )
            yield Button("Close", id="help-close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "help-close":
            self.dismiss()

    def key_escape(self) -> None:
        self.dismiss()


# ── TUI Widgets (Dashboard) ────────────────────────────────────────

class SessionPanel(Static):
    def __init__(self, session_manager: SessionManager, **kwargs):
        super().__init__(**kwargs)
        self.session_manager = session_manager
        self._refresh()

    def _refresh(self):
        sm = self.session_manager
        if sm.current_session and sm.current_session.is_active:
            s = sm.current_session
            phase_icon = PHASE_ICONS.get(s.phase, "?")

            if s.is_pending:
                ptype = platform_type_label(s.platform.key)
                content = (
                    f"{phase_icon} [bold yellow]PENDING — Awaiting Confirmation[/bold yellow]\n\n"
                    f"  Platform: [cyan]{s.platform.name}[/cyan]  ({ptype})\n"
                    f"  Account:  [white]{s.account.name}[/white]\n"
                    f"  GPU:      [green]{s.platform.gpu_type}[/green]\n"
                    f"  Limit:    [white]{s.platform.session_limit_hours}h[/white]\n\n"
                    f"  [bold yellow]Timer not started yet![/bold yellow]\n"
                )
                if is_auto_platform(s.platform.key):
                    content += "  [green]Auto-platform — confirming after push...[/green]"
                else:
                    content += (
                        "  [yellow]Manual platform — upload notebook, then run:[/yellow]\n"
                        "  [bold]/confirm[/bold] to start countdown timer"
                    )
            else:
                # CONFIRMED — show countdown
                remaining = s.remaining_seconds
                elapsed = s.elapsed_seconds
                progress = s.progress
                bar_len = 30
                filled = int(bar_len * progress)
                empty = bar_len - filled
                bar = f"[green]{'█' * filled}[/green][dim]{'░' * empty}[/dim]"
                # Show last platform status check
                status_str = s._last_status_check or "not checked"
                content = (
                    f"{phase_icon} [bold yellow]TRAINING[/bold yellow]\n\n"
                    f"  Platform: [cyan]{s.platform.name}[/cyan]  "
                    f"({platform_type_label(s.platform.key)})\n"
                    f"  Account:  [white]{s.account.name}[/white]\n"
                    f"  GPU:      [green]{s.platform.gpu_type}[/green]\n\n"
                    f"  Elapsed:   [white]{format_seconds(elapsed)}[/white]  "
                    f"Remaining: [yellow]{format_seconds(remaining)}[/yellow]\n"
                    f"  {bar} {progress*100:.1f}%\n\n"
                    f"  Platform status: [dim]{status_str}[/dim]  "
                    f"Sessions: {s.account.sessions_used}  "
                    f"Hours: {s.account.total_hours_used:.1f}h"
                )
        else:
            total_h = sm.get_total_available_hours()
            next_acct = sm.get_next_account()
            next_info = "None"
            if next_acct:
                plat, acc = next_acct
                ptype = platform_type_label(plat.key)
                next_info = f"{plat.name}/{acc.name} ({plat.gpu_type}, {ptype})"
            content = (
                f"[bold]IDLE[/bold]\n\n"
                f"  Available: [green]{total_h:.1f}h[/green]  "
                f"Next: [cyan]{next_info}[/cyan]\n\n"
                f"  [bold]/add[/bold] to add platform  |  [bold]/start[/bold] to train"
            )
        self.update(content)


class PlatformCard(Static):
    def __init__(self, platform: PlatformConfig, **kwargs):
        super().__init__(**kwargs)
        self.platform = platform
        self._refresh()

    def _refresh(self):
        p = self.platform
        icon = STATUS_ICONS.get(p.status, "?")
        color = STATUS_COLORS.get(p.status, "white")
        cred_schema = CREDENTIAL_SCHEMAS.get(p.key, [])
        ptype = platform_type_label(p.key)
        ptype_color = "green" if ptype == "AUTO" else "yellow"
        acct_lines = []
        for acc in p.accounts:
            ai = STATUS_ICONS.get(acc.status, "?")
            cred_status = ""
            if cred_schema:
                filled = sum(1 for cf in cred_schema if acc.credentials.get(cf["key"]))
                total = len(cred_schema)
                if filled == total:
                    cred_status = " [green]OK[/green]"
                elif filled > 0:
                    cred_status = f" [yellow]{filled}/{total}[/yellow]"
                else:
                    cred_status = " [red]X[/red]"
            acct_lines.append(f"  {ai} {acc.name}  [dim]{acc.total_hours_used:.1f}h[/dim]{cred_status}")
        accts = "\n".join(acct_lines) if acct_lines else "  [dim]no accounts[/dim]"
        content = (
            f"[bold]{icon} {p.name}[/bold]  "
            f"[cyan]{p.gpu_type}[/cyan]  "
            f"{p.session_limit_hours}h/sess  "
            f"[white]{p.total_accounts}x[/white]=[green]{p.max_continuous_hours:.0f}h[/green]  "
            f"[{ptype_color}]{ptype}[/{ptype_color}]\n"
            f"{accts}"
        )
        self.update(content)
        self.border_title = p.name
        self.border_subtitle = f"[{color}]{p.status.value}[/{color}]"


class DashboardView(VerticalScroll):
    def __init__(self, platforms, session_manager, **kwargs):
        super().__init__(**kwargs)
        self.platforms = platforms
        self.session_manager = session_manager

    def compose(self) -> ComposeResult:
        yield Label("[bold]Session[/bold]", classes="section-header")
        yield SessionPanel(self.session_manager, classes="session-panel")
        yield Label("[bold]Platforms[/bold]", classes="section-header")
        for p in self.platforms:
            yield PlatformCard(p, classes="platform-card")

    def refresh_data(self):
        for w in self.query(PlatformCard):
            w._refresh()
        for w in self.query(SessionPanel):
            w._refresh()


class ScheduleView(Static):
    def __init__(self, platforms, session_manager, **kwargs):
        super().__init__(**kwargs)
        self.platforms = platforms
        self.session_manager = session_manager
        self._refresh()

    def _refresh(self):
        lines = []
        cum = 0.0
        for p in sorted(self.platforms, key=lambda x: x.session_limit_hours, reverse=True):
            if not p.enabled:
                continue
            for acc in p.accounts:
                if acc.status in (PlatformStatus.AVAILABLE, PlatformStatus.IN_USE, PlatformStatus.COOLDOWN):
                    end = cum + p.session_limit_hours
                    ico = STATUS_ICONS.get(acc.status, "?")
                    if self.session_manager.current_session:
                        cs = self.session_manager.current_session
                        if cs.platform.key == p.key and cs.account.name == acc.name:
                            ico = "[bold yellow]▶[/bold yellow]"
                    # Show auto/manual label
                    ptype = platform_type_label(p.key)
                    ptype_color = "green" if ptype == "AUTO" else "yellow"
                    lines.append(
                        f"  {ico} [{cum:6.1f}h-{end:6.1f}h]  "
                        f"[cyan]{p.name}[/cyan]  [white]{acc.name}[/white]  "
                        f"[green]{p.gpu_type}[/green]  "
                        f"[{ptype_color}]{ptype}[/{ptype_color}]"
                    )
                    cum = end + p.cooldown_minutes / 60
        total = self.session_manager.get_total_available_hours()
        self.update(
            f"[bold]Rotation Schedule[/bold]\n\n"
            f"  Total: [bold green]{total:.1f}h[/bold green]   "
            f"[green]AUTO[/green] = fully automated  [yellow]MANUAL[/yellow] = needs /confirm\n\n"
            + "\n".join(lines)
        )


class LogView(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield RichLog(id="training-log", highlight=True, markup=True)


# ── Main App ───────────────────────────────────────────────────────

class FreeGPUTrainerApp(App):
    TITLE = "Free GPU Trainer"
    SUB_TITLE = "/add to add platform  /start to train  /confirm for manual platforms"

    CSS = """
    Screen { background: $surface; }

    .section-header {
        text-align: center; padding: 1 0;
        color: $text; text-style: bold;
    }
    .session-panel {
        height: auto; min-height: 10; padding: 1 2;
        margin: 0 1 1 1; background: $surface-darken-1;
        border: round $primary;
    }
    .platform-card {
        height: auto; min-height: 4; padding: 1 2;
        margin: 0 1 1 1;
    }
    TabbedContent { height: 1fr; }
    DataTable { height: 1fr; }
    RichLog { height: 1fr; border: solid $primary; }

    #command-bar {
        dock: bottom; height: 3; padding: 0 1;
        background: $primary-darken-2; border-top: solid $primary;
    }
    #command-input { width: 1fr; }
    #command-hint { width: auto; color: $text-muted; padding: 1 1 0 0; }

    /* Platform List Modal */
    #plist-outer {
        padding: 1 2; width: 72; height: 28;
        background: $surface; border: thick $primary;
    }
    #plist-title { padding: 0 0 1 0; }
    #plist-options { height: 1fr; border: solid $primary; }

    /* Credential Input Modal */
    #cred-outer {
        padding: 1 2; width: 72; height: 30;
        background: $surface; border: thick $success;
    }
    #cred-title { padding: 0 0 1 0; }
    #cred-platform-info { padding: 0 0 1 0; }
    .cred-field-label { padding: 0 0 0 0; margin: 0 0 0 0; }
    .cred-hint { padding: 0 0 0 0; margin: 0 0 0 0; color: $text-muted; }
    #cred-disclaimer { padding: 0 0 1 0; }

    /* Platform Detail Modal */
    #pdetail-outer {
        padding: 1 2; width: 72; height: 28;
        background: $surface; border: thick $accent;
    }
    #pdetail-title { padding: 0 0 0 0; }
    #pdetail-info { padding: 0 0 1 0; }
    #pdetail-acct-label { padding: 1 0 0 0; }
    .acct-row { height: 3; padding: 0 1; }
    .acct-name { width: 1fr; }
    .acct-edit-btn { width: auto; }
    .acct-rm-btn { width: auto; }

    /* Edit Credential Modal */
    #editcred-outer {
        padding: 1 2; width: 72; height: 30;
        background: $surface; border: thick $warning;
    }
    #editcred-title { padding: 0 0 1 0; }
    #editcred-hint { padding: 0 0 1 0; }

    /* Help Modal */
    #help-dialog {
        padding: 2 4; width: 60; height: auto;
        background: $surface; border: thick $accent;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "start_training", "Start"),
        Binding("c", "confirm_session", "Confirm"),
        Binding("x", "stop_training", "Stop"),
        Binding("slash", "focus_command", "/", key_display="/"),
        Binding("1", "tab_dashboard", "Dash"),
        Binding("2", "tab_schedule", "Sched"),
        Binding("3", "tab_logs", "Logs"),
    ]

    training_active: reactive[bool] = reactive(False)

    # Status check interval (seconds) — polls check_status() on current session
    STATUS_CHECK_INTERVAL = 300  # 5 minutes

    def __init__(self, config_path="config.yaml", **kwargs):
        super().__init__(**kwargs)
        self.config_path = config_path
        self.config = load_config(config_path)
        self.platforms: list[PlatformConfig] = []
        self.session_manager: Optional[SessionManager] = None
        self._tick_timer: Optional[Timer] = None
        self._status_check_counter: int = 0
        self._status_check_running: bool = False  # True while background status check is in progress
        self._current_job = None  # TrainingJob for save_state on rotate
        self._setup_logging()
        self._build_platforms()

    def _setup_logging(self):
        log_cfg = self.config.get("logging", {})
        level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
        self.logger = logging.getLogger("fgt")

    def _build_platforms(self):
        self.platforms = []
        config_dir = os.path.dirname(os.path.abspath(self.config_path))
        tc = self.config.get("training", {})
        entry_script = tc.get("entry_script", "train.py")
        for key, cfg in self.config.get("platforms", {}).items():
            if key in PLATFORM_DEFS:
                self.platforms.append(build_platform(key, cfg, config_dir))
        self.session_manager = SessionManager(
            platforms=self.platforms,
            auto_rotate=tc.get("auto_rotate", True),
            rotate_buffer_minutes=tc.get("rotate_buffer_minutes", 10),
            checkpoint_before_rotate=tc.get("checkpoint_before_rotate", True),
            entry_script=entry_script,
        )

    def _rebuild(self):
        tc = self.config.get("training", {})
        entry_script = tc.get("entry_script", "train.py")
        self.session_manager = SessionManager(
            platforms=self.platforms,
            auto_rotate=tc.get("auto_rotate", True),
            rotate_buffer_minutes=tc.get("rotate_buffer_minutes", 10),
            checkpoint_before_rotate=tc.get("checkpoint_before_rotate", True),
            entry_script=entry_script,
        )
        self._rebuild_dashboard()

    def _rebuild_dashboard(self):
        try:
            d = self.query_one(DashboardView)
            d.remove_children()
            d.mount(Label("[bold]Session[/bold]", classes="section-header"))
            d.mount(SessionPanel(self.session_manager, classes="session-panel"))
            d.mount(Label("[bold]Platforms[/bold]", classes="section-header"))
            for p in self.platforms:
                d.mount(PlatformCard(p, classes="platform-card"))
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="dashboard"):
            with TabPane("1:Dash", id="dashboard"):
                yield DashboardView(self.platforms, self.session_manager)
            with TabPane("2:Schedule", id="schedule"):
                yield ScheduleView(self.platforms, self.session_manager)
            with TabPane("3:Logs", id="logs"):
                yield LogView()
        with Horizontal(id="command-bar"):
            yield Label("/ ", id="command-hint")
            yield Input(placeholder="type a command... (/add /start /confirm /help)", id="command-input")
        yield Footer()

    def on_mount(self) -> None:
        self._tick_timer = self.set_interval(1.0, self._tick)
        self._log("Free GPU Trainer started")
        self._log(f"{len(self.platforms)} platforms, "
                  f"{sum(p.total_accounts for p in self.platforms)} accounts, "
                  f"{self.session_manager.get_total_available_hours():.0f}h total")
        self._log(f"[dim]Credential storage: {get_storage_mode()}[/dim]")
        self._log("[bold]/add[/bold] add platform  [bold]/start[/bold] train  "
                  "[bold]/confirm[/bold] confirm manual  [bold]/help[/bold] commands")

    def _tick(self):
        try:
            d = self.query_one(DashboardView)
            d.refresh_data()
            for sv in self.query(ScheduleView):
                sv._refresh()
            if self.session_manager:
                self.training_active = self.session_manager.is_training

                # Auto-reset weekly counters if a new week has started
                self.session_manager.auto_reset_weekly_if_needed()

                # Periodic status check (every STATUS_CHECK_INTERVAL seconds)
                # Run in background thread to avoid blocking the TUI event loop
                self._status_check_counter += 1
                if (self._status_check_counter >= self.STATUS_CHECK_INTERVAL and
                        self.session_manager.is_training and
                        self.session_manager.current_session.is_confirmed and
                        not getattr(self, '_status_check_running', False)):
                    self._status_check_counter = 0
                    self._status_check_running = True
                    threading.Thread(
                        target=self._run_status_check, daemon=True
                    ).start()
        except Exception:
            pass

    def _run_status_check(self):
        """Run status check in a background thread, then update UI via call_from_thread.

        This prevents blocking the TUI event loop while waiting for
        subprocess calls (SSH, Kaggle API) that can take 10-30 seconds.
        """
        try:
            status = self.session_manager.check_current_session_status()
            if status:
                self.call_from_thread(self._log, f"[dim]Platform status: {status}[/dim]")
        except Exception as e:
            self.logger.debug(f"Background status check error: {e}")
        finally:
            self._status_check_running = False

    # ── Command Bar ─────────────────────────────────────────────

    def action_focus_command(self):
        self.query_one("#command-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "command-input":
            return
        raw = event.value.strip()
        event.input.value = ""
        if not raw:
            return
        if not raw.startswith("/"):
            raw = "/" + raw

        cmd = raw.split()[0].lower().lstrip("/")
        dispatch = {
            "add": self._cmd_add,
            "remove": self._cmd_remove,
            "choose": self._cmd_choose,
            "start": self._cmd_start,
            "confirm": self._cmd_confirm,
            "stop": self._cmd_stop,
            "done": self._cmd_done,
            "status": self._cmd_status,
            "save": self._cmd_save,
            "reset": self._cmd_reset,
            "help": self._cmd_help,
        }
        handler = dispatch.get(cmd)
        if handler:
            handler()
        else:
            self._log(f"[red]Unknown:[/red] /{cmd}  try /help")

    # ── Commands ────────────────────────────────────────────────

    def _cmd_add(self):
        self.push_screen(
            PlatformListScreen("Pick Platform", self.platforms, show_empty=True),
            self._on_add_pick,
        )

    def _on_add_pick(self, key: str):
        if not key:
            return

        existing = None
        for p in self.platforms:
            if p.key == key:
                existing = p
                break

        if existing:
            self._open_platform_detail(existing)
        else:
            defn = PLATFORM_DEFS[key]
            cfg = {"enabled": True, "accounts": []}
            new_p = build_platform(key, cfg)
            self.platforms.append(new_p)
            if "platforms" not in self.config:
                self.config["platforms"] = {}
            self.config["platforms"][key] = cfg
            self._rebuild()
            self._log(f"[green]Added:[/green] {new_p.name} ({platform_type_label(key)})")
            self._prompt_add_account(new_p)

    def _prompt_add_account(self, platform: PlatformConfig):
        self.push_screen(
            CredentialInputScreen(platform.key),
            lambda result, p=platform: self._on_credential_input(p, result),
        )

    def _on_credential_input(self, platform: PlatformConfig, result: dict):
        if not result or not result.get("name"):
            if platform.accounts:
                self._open_platform_detail(platform)
            return

        name = result["name"]
        creds = result.get("credentials", {})

        for acc in platform.accounts:
            if acc.name == name:
                self._log(f"[yellow]Already exists:[/yellow] {name}")
                self._prompt_add_account(platform)
                return

        config_dir = os.path.dirname(os.path.abspath(self.config_path))
        encrypted_creds = encrypt_credentials(platform.key, name, creds, config_dir)

        new_acc = AccountConfig(name=name, credentials=creds)
        self._sync_legacy_fields(new_acc)

        platform.accounts.append(new_acc)
        self._save_account_to_config(platform, name, encrypted_creds)
        self._rebuild()

        cred_count = len(creds)
        total_creds = len(CREDENTIAL_SCHEMAS.get(platform.key, []))
        cred_msg = f" ({cred_count}/{total_creds} creds)" if total_creds > 0 else ""
        storage_mode = get_storage_mode()
        self._log(f"[green]Stacked:[/green] {name} on {platform.name} "
                  f"(now {platform.total_accounts}x = {platform.max_continuous_hours:.0f}h){cred_msg}")
        self._log(f"[dim]Storage: {storage_mode}[/dim]")

        self._open_platform_detail(platform)

    def _sync_legacy_fields(self, acc: AccountConfig):
        creds = acc.credentials
        if "kaggle_username" in creds or "kaggle_key" in creds:
            import json as _json
            acc.token = _json.dumps({"username": creds.get("kaggle_username", ""), "key": creds.get("kaggle_key", "")})
            acc.api_key = creds.get("kaggle_key")
        if "hf_token" in creds:
            acc.token = creds["hf_token"]
        if "oci_vm_host" in creds:
            acc.token = creds.get("oci_vm_host")
        if "gcp_vm_host" in creds:
            acc.token = creds.get("gcp_vm_host")
        if "token" in creds:
            acc.token = creds["token"]
        if "api_key" in creds:
            acc.api_key = creds["api_key"]

    def _save_account_to_config(self, platform: PlatformConfig, name: str, creds: dict):
        if platform.key not in self.config.get("platforms", {}):
            self.config["platforms"][platform.key] = {"enabled": True, "accounts": []}

        acct_cfg = {"name": name}
        if creds:
            acct_cfg["credentials"] = creds
        self.config["platforms"][platform.key].setdefault("accounts", []).append(acct_cfg)

    def _open_platform_detail(self, platform: PlatformConfig):
        self.push_screen(
            PlatformDetailScreen(platform, self.config),
            lambda result, p=platform: self._on_detail_result(p, result),
        )

    def _on_detail_result(self, platform: PlatformConfig, result: str):
        if not result or result == "done":
            self._rebuild()
            return

        if result == "add_account":
            self._prompt_add_account(platform)

        elif result.startswith("edit_creds:"):
            name = result[11:]
            acc = None
            for a in platform.accounts:
                if a.name == name:
                    acc = a
                    break
            if acc:
                self.push_screen(
                    EditCredentialScreen(platform.key, acc),
                    lambda r, p=platform, a=acc: self._on_edit_creds(p, a, r),
                )
            else:
                self._open_platform_detail(platform)

        elif result.startswith("add:"):
            name = result[4:]
            self.push_screen(
                CredentialInputScreen(platform.key, account_name=name),
                lambda r, p=platform: self._on_credential_input(p, r),
            )

        elif result.startswith("remove:"):
            name = result[7:]
            found = None
            for acc in platform.accounts:
                if acc.name == name:
                    found = acc
                    break
            if found:
                if found.status == PlatformStatus.IN_USE:
                    self._log("[red]Can't remove account in use![/red]")
                else:
                    platform.accounts.remove(found)
                    if platform.key in self.config.get("platforms", {}):
                        accts = self.config["platforms"][platform.key].get("accounts", [])
                        self.config["platforms"][platform.key]["accounts"] = [
                            a for a in accts if a.get("name") != name
                        ]
                    self._log(f"[yellow]Removed:[/yellow] {name} from {platform.name}")
                    delete_credentials(platform.key, name)
                    self._rebuild()
            self._open_platform_detail(platform)

    def _on_edit_creds(self, platform: PlatformConfig, account: AccountConfig, result: dict):
        if not result or "updates" not in result:
            self._open_platform_detail(platform)
            return

        updates = result["updates"]
        changed = 0
        for key, val in updates.items():
            if val is None:
                account.credentials.pop(key, None)
                changed += 1
            else:
                account.credentials[key] = val
                changed += 1

        if changed > 0:
            self._sync_legacy_fields(account)
            config_dir = os.path.dirname(os.path.abspath(self.config_path))
            encrypted = encrypt_credentials(platform.key, account.name, account.credentials, config_dir)
            self._update_account_creds_in_config(platform, account, encrypted)
            self._rebuild()
            self._log(f"[green]Updated {changed} credentials[/green] for {account.name}")
            self._log(f"[dim]Storage: {get_storage_mode()}[/dim]")

        self._open_platform_detail(platform)

    def _update_account_creds_in_config(self, platform: PlatformConfig, account: AccountConfig,
                                         encrypted_creds: dict = None):
        if platform.key not in self.config.get("platforms", {}):
            return
        accts = self.config["platforms"][platform.key].get("accounts", [])
        for a in accts:
            if a.get("name") == account.name:
                a["credentials"] = encrypted_creds if encrypted_creds else account.credentials
                break

    def _cmd_remove(self):
        self.push_screen(
            PlatformListScreen("Remove Platform", self.platforms),
            self._on_remove_pick,
        )

    def _on_remove_pick(self, key: str):
        if not key:
            return
        p = None
        for plat in self.platforms:
            if plat.key == key:
                p = plat
                break
        if p:
            self.platforms.remove(p)
            if p.key in self.config.get("platforms", {}):
                del self.config["platforms"][p.key]
            self._rebuild()
            self._log(f"[yellow]Removed:[/yellow] {p.name}")

    def _cmd_choose(self):
        self.push_screen(
            PlatformListScreen("Choose Platform for Next Session", self.platforms),
            self._on_choose_pick,
        )

    def _on_choose_pick(self, key: str):
        if not key:
            return
        p = None
        for plat in self.platforms:
            if plat.key == key:
                p = plat
                break
        if p:
            self.platforms.sort(key=lambda x: 0 if x.key == p.key else 1)
            self._rebuild()
            self._log(f"[cyan]Next session ->[/cyan] {p.name} ({p.gpu_type}, {platform_type_label(p.key)})")

    def _cmd_start(self):
        if self.session_manager.is_training:
            self._log("Already training")
            return
        self._log("Starting...")

        def on_rotate(old, new):
            # Save training state before rotating
            if self._current_job:
                try:
                    self._current_job.save_state()
                    self._log("[dim]Training state saved before rotation[/dim]")
                except Exception as e:
                    self._log(f"[yellow]Failed to save state: {e}[/yellow]")

            if new:
                self._log(f"[green]ROTATED:[/green] {old.platform.name}/{old.account.name} -> "
                          f"{new.platform.name}/{new.account.name}")
                self._gen_script(new)
                # Auto-confirm for auto-platforms (Kaggle uses delayed polling)
                if is_auto_platform(new.platform.key):
                    self.session_manager._auto_confirm_with_polling(new)
                    if new.platform.key == "kaggle":
                        self._log("[green]Auto-confirming[/green] — polling Kaggle until running...")
                    else:
                        self._log("[green]Auto-confirmed[/green] (API/SSH platform)")
            else:
                self._log("[red]No account for rotation![/red] Stopped.")

        session = self.session_manager.start_session(on_rotate=on_rotate)
        if session:
            ptype = platform_type_label(session.platform.key)
            self._log(f"[green]SESSION CREATED:[/green] {session.platform.name}/{session.account.name} "
                      f"({session.platform.gpu_type}, {session.platform.session_limit_hours}h, {ptype})")

            # Push code via handler
            self._gen_script(session)

            # Auto-confirm for auto-platforms (Kaggle uses delayed polling, SSH immediate)
            if is_auto_platform(session.platform.key):
                self.session_manager._auto_confirm_with_polling(session)
                if session.platform.key == "kaggle":
                    self._log("[green]Auto-confirming[/green] — polling Kaggle until kernel is running...")
                else:
                    self._log("[green]Auto-confirmed[/green] — countdown timer started")
            elif session.platform.key == "huggingface":
                self._log("[yellow]HuggingFace Spaces are for inference/demos, NOT long training.[/yellow]")
                self._log("[yellow]Run /confirm once your Space is running.[/yellow]")
            else:
                self._log("[yellow]Manual platform — upload notebook, then run /confirm[/yellow]")
                self._log("[yellow]Timer will NOT start until you /confirm[/yellow]")

            self.training_active = True
        else:
            self._log("[red]No accounts![/red] Use /add then stack accounts with credentials")

    def _cmd_confirm(self):
        """Confirm the current session is running on the platform."""
        if not self.session_manager or not self.session_manager.current_session:
            self._log("[yellow]No active session to confirm[/yellow]")
            return

        session = self.session_manager.current_session
        if not session.is_pending:
            self._log(f"[dim]Session already confirmed (phase: {session.phase.value})[/dim]")
            return

        if self.session_manager.confirm_session():
            self._log(f"[green]CONFIRMED:[/green] {session.platform.name}/{session.account.name} — countdown started!")
            self._log(f"[dim]Session limit: {session.platform.session_limit_hours}h, "
                      f"auto-rotate: {self.session_manager.auto_rotate}[/dim]")
        else:
            self._log("[red]Failed to confirm session[/red]")

    def _cmd_done(self):
        """Signal training complete — end session early and rotate."""
        if not self.session_manager or not self.session_manager.is_training:
            self._log("[yellow]No active training session[/yellow]")
            return

        session = self.session_manager.current_session
        self._log(f"[green]Training complete:[/green] ending {session.platform.name}/{session.account.name} early")
        rotated = self.session_manager.mark_training_complete()
        if rotated:
            self._log("[green]Rotated to next account[/green]")
        else:
            self._log("[yellow]No more accounts to rotate to[/yellow]")
            self.training_active = False

    def _cmd_stop(self):
        if self.session_manager and self.session_manager.current_session:
            from handlers import get_handler as _get_handler
            s = self.session_manager.current_session
            handler = _get_handler(s.platform.key)
            if handler:
                entry_script = self.session_manager.entry_script
                result = handler.stop_session(s.account, entry_script)
                if not result.get("ok"):
                    self._log(f"[dim]Handler stop: {result.get('message', '')}[/dim]")
            self.session_manager.stop()
            self._log("[yellow]Stopped.[/yellow]")
            self.training_active = False
        elif self.session_manager:
            self.session_manager.stop()
            self._log("[yellow]Stopped.[/yellow]")
            self.training_active = False

    def _cmd_status(self):
        sm = self.session_manager
        if sm.current_session and sm.current_session.is_active:
            s = sm.current_session
            phase_str = s.phase.value.upper()
            ptype = platform_type_label(s.platform.key)
            status_msg = (
                f"[{phase_str}] {s.platform.name}/{s.account.name} "
                f"{s.platform.gpu_type} ({ptype})"
            )
            if s.is_confirmed:
                status_msg += (
                    f" elapsed={format_seconds(s.elapsed_seconds)} "
                    f"remaining={format_seconds(s.remaining_seconds)}"
                )
            elif s.is_pending:
                status_msg += " [yellow]TIMER NOT STARTED — run /confirm[/yellow]"

            self._log(status_msg)

            # Also check real platform status if available
            real_status = sm.check_current_session_status()
            if real_status:
                self._log(f"[dim]Platform reports: {real_status}[/dim]")
        else:
            nxt = sm.get_next_account()
            if nxt:
                plat, acc = nxt
                self._log(f"Idle. Next: {plat.name}/{acc.name} ({plat.gpu_type}, {platform_type_label(plat.key)})")
            else:
                self._log("Idle. No accounts.")
            self._log(f"Total: {sm.get_total_available_hours():.1f}h")

    def _cmd_save(self):
        save_config(self.config, self.config_path)
        self._log(f"[green]Saved -> {self.config_path}[/green]")

    def _cmd_reset(self):
        self.session_manager.reset_weekly()
        self._log("[green]Weekly counters reset[/green]")

    def _cmd_help(self):
        self.push_screen(HelpScreen())

    def _cmd_save_state(self):
        """Manually save runtime state to disk."""
        if self.session_manager:
            self.session_manager.save_runtime_state()
            self._log("[green]Runtime state saved to state.json[/green]")

    # ── Training Actions ────────────────────────────────────────

    def action_start_training(self):
        self._cmd_start()

    def action_confirm_session(self):
        self._cmd_confirm()

    def action_stop_training(self):
        self._cmd_stop()

    def action_tab_dashboard(self):
        self.query_one(TabbedContent).active = "dashboard"

    def action_tab_schedule(self):
        self.query_one(TabbedContent).active = "schedule"

    def action_tab_logs(self):
        self.query_one(TabbedContent).active = "logs"

    def _log(self, msg: str):
        try:
            log = self.query_one("#training-log", RichLog)
            log.write(f"[dim][{time.strftime('%H:%M:%S')}][/dim] {msg}")
        except Exception:
            pass

    def _gen_script(self, session):
        """Generate scripts AND push code via real platform handlers."""
        from trainer import TrainingJob
        from handlers import get_handler as _get_handler
        tc = self.config.get("training", {})
        script_path = tc.get("entry_script", "train.py")
        checkpoint_dir = tc.get("checkpoint_dir", "./checkpoints")
        resume = tc.get("resume_from_checkpoint", True)

        # Create and store TrainingJob for save_state on rotate
        job = TrainingJob(script_path=script_path, checkpoint_dir=checkpoint_dir, resume=resume)
        self._current_job = job

        # Generate local scripts
        Path("./run_session.sh").write_text(job.generate_run_command(session.platform))
        self._log(f"[cyan]Script ->[/cyan] run_session.sh")

        # Use real handler to push code to platform
        handler = _get_handler(session.platform.key)
        if handler:
            self._log(f"[dim]Pushing code via {handler.name} handler...[/dim]")
            result = handler.push_code(session.account, script_path, checkpoint_dir)
            if result.get("ok"):
                msg = result.get("message", "OK")
                self._log(f"[green]Handler:[/green] {msg}")
                if result.get("manual"):
                    self._log(f"[yellow]Manual step:[/yellow] upload notebook to {session.platform.name}")
                    if result.get("notebook_path"):
                        self._log(f"  Notebook: {result['notebook_path']}")
                    if result.get("url"):
                        self._log(f"  URL: {result['url']}")
                if result.get("warning"):
                    self._log(f"[yellow]Warning:[/yellow] {result['warning']}")
                if result.get("notebook_path"):
                    self._log(f"[cyan]Notebook ->[/cyan] {result['notebook_path']}")
            else:
                self._log(f"[red]Handler failed:[/red] {result.get('message', 'unknown error')}")
        else:
            self._log(f"[dim]No handler for {session.platform.key} — using local scripts only[/dim]")


def run_app():
    """Main entry point — supports both TUI and headless CLI modes."""
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(
        description="Free GPU Trainer — continuous AI training across free GPU platforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python tui.py                       # Launch interactive TUI
  python tui.py --status              # Human-readable status
  python tui.py --status --json       # Machine-readable JSON (for agents)
  python tui.py --start               # Start training headlessly
  python tui.py --confirm             # Confirm current session headlessly
  python tui.py --stop                # Stop training headlessly
  python tui.py --done                # Signal training complete, rotate to next
  python tui.py --schema kaggle       # Print credential schema for a platform
  python tui.py --platforms           # List all platforms + required fields + AUTO/MANUAL
"""
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--json", action="store_true", help="Output as JSON (use with --status)")
    parser.add_argument("--start", action="store_true", help="Start training headlessly")
    parser.add_argument("--confirm", action="store_true", help="Confirm current session headlessly")
    parser.add_argument("--stop", action="store_true", help="Stop training headlessly")
    parser.add_argument("--done", action="store_true", help="Signal training complete, rotate to next")
    parser.add_argument("--schema", type=str, default=None, metavar="PLATFORM",
                        help="Print credential schema for a platform (e.g. kaggle, oracle_cloud)")
    parser.add_argument("--platforms", action="store_true",
                        help="List all platforms + required fields + AUTO/MANUAL type")

    args = parser.parse_args()
    config_path = args.config

    # ── --schema: Print credential schema for a platform ─────────
    if args.schema:
        key = args.schema
        if key not in PLATFORM_DEFS:
            print(f"Error: Unknown platform '{key}'. Available: {', '.join(PLATFORM_DEFS.keys())}")
            sys.exit(1)
        schema = {
            "platform": key,
            "name": PLATFORM_DEFS[key]["name"],
            "type": "AUTO" if key in {"kaggle", "oracle_cloud", "gcp"} else "MANUAL",
            "url": PLATFORM_DEFS[key]["url"],
            "gpu_type": PLATFORM_DEFS[key]["gpu_type"],
            "session_limit_hours": PLATFORM_DEFS[key]["session_limit_hours"],
            "credentials": CREDENTIAL_SCHEMAS.get(key, []),
        }
        if args.json:
            print(_json.dumps(schema, indent=2))
        else:
            print(f"\n  Platform: {schema['name']} ({key})")
            print(f"  Type: {schema['type']}")
            print(f"  GPU: {schema['gpu_type']}")
            print(f"  Session limit: {schema['session_limit_hours']}h")
            print(f"  URL: {schema['url']}")
            if schema["credentials"]:
                print(f"\n  Required credentials:")
                for cf in schema["credentials"]:
                    req = "required" if cf.get("required", True) else "optional"
                    secret = " (secret)" if cf.get("secret", False) else ""
                    print(f"    {cf['key']:25s} — {cf['label']} [{req}{secret}]")
                    if cf.get("hint"):
                        print(f"      Hint: {cf['hint']}")
            else:
                print(f"\n  No credentials required")
            print()
        return

    # ── --platforms: List all platforms ──────────────────────────
    if args.platforms:
        if args.json:
            platforms_list = []
            for key, defn in PLATFORM_DEFS.items():
                ptype = "AUTO" if key in {"kaggle", "oracle_cloud", "gcp"} else "MANUAL"
                platforms_list.append({
                    "key": key,
                    "name": defn["name"],
                    "type": ptype,
                    "url": defn["url"],
                    "gpu_type": defn["gpu_type"],
                    "session_limit_hours": defn["session_limit_hours"],
                    "weekly_limit_hours": defn.get("weekly_limit_hours"),
                    "credentials": [cf["key"] for cf in CREDENTIAL_SCHEMAS.get(key, [])],
                })
            print(_json.dumps(platforms_list, indent=2))
        else:
            print(f"\n  {'Platform':35s} {'Type':8s} {'GPU':20s} {'Session':8s} {'Creds':6s}")
            print(f"  {'─'*35} {'─'*8} {'─'*20} {'─'*8} {'─'*6}")
            for key, defn in PLATFORM_DEFS.items():
                ptype = "AUTO" if key in {"kaggle", "oracle_cloud", "gcp"} else "MANUAL"
                cred_count = len(CREDENTIAL_SCHEMAS.get(key, []))
                cred_str = str(cred_count) if cred_count else "-"
                print(f"  {defn['name']:35s} {ptype:8s} {defn['gpu_type']:20s} {defn['session_limit_hours']:5.0f}h   {cred_str:6s}")
            print()
        return

    # ── Load config and build SessionManager for CLI commands ────
    def _build_sm(config_path):
        config = load_config(config_path)
        platforms = []
        config_dir = os.path.dirname(os.path.abspath(config_path))
        for key, cfg in config.get("platforms", {}).items():
            if key in PLATFORM_DEFS:
                platforms.append(build_platform(key, cfg, config_dir))
        tc = config.get("training", {})
        sm = SessionManager(
            platforms=platforms,
            auto_rotate=tc.get("auto_rotate", True),
            rotate_buffer_minutes=tc.get("rotate_buffer_minutes", 10),
            checkpoint_before_rotate=tc.get("checkpoint_before_rotate", True),
            entry_script=tc.get("entry_script", "train.py"),
        )
        return sm, config

    # ── --status: Show status ────────────────────────────────────
    if args.status:
        sm, config = _build_sm(config_path)
        if args.json:
            print(_json.dumps(sm.get_status_json(), indent=2))
        else:
            print(f"\n  Free GPU Trainer — Status\n")
            print(f"  Total: {sm.get_total_available_hours():.1f}h  Platforms: {len(sm.platforms)}")
            for p in sm.platforms:
                ptype = platform_type_label(p.key)
                print(f"    {p.name:35s} ({p.gpu_type:20s}) {p.total_accounts}x = {p.max_continuous_hours:.0f}h  [{ptype}]")
            if sm.current_session and sm.current_session.is_active:
                s = sm.current_session
                phase = s.phase.value.upper()
                print(f"\n  Current: [{phase}] {s.platform.name}/{s.account.name} {s.platform.gpu_type}")
                if s.is_confirmed:
                    print(f"           Elapsed: {format_seconds(s.elapsed_seconds)}  Remaining: {format_seconds(s.remaining_seconds)}")
            print()
        return

    # ── --start: Start training headlessly ───────────────────────
    if args.start:
        sm, config = _build_sm(config_path)
        if sm.is_training:
            s = sm.current_session
            result = sm.get_status_json()
            if args.json:
                print(_json.dumps({"ok": False, "error": "already_training", "session": result.get("current_session")}, indent=2))
            else:
                print(f"Already training: {s.platform.name}/{s.account.name} ({s.phase.value})")
            sys.exit(1)

        session = sm.start_session()
        if not session:
            if args.json:
                print(_json.dumps({"ok": False, "error": "no_accounts"}, indent=2))
            else:
                print("No accounts available. Add platforms and accounts to config.yaml first.")
            sys.exit(1)

        # Push code via handler
        from handlers import get_handler as _get_handler
        handler = _get_handler(session.platform.key)
        push_result = {"ok": False, "message": "No handler"}
        if handler:
            tc = config.get("training", {})
            script_path = tc.get("entry_script", "train.py")
            checkpoint_dir = tc.get("checkpoint_dir", "./checkpoints")
            push_result = handler.push_code(session.account, script_path, checkpoint_dir)

        # Auto-confirm for AUTO platforms
        if is_auto_platform(session.platform.key):
            sm._auto_confirm_with_polling(session)

        result = {
            "ok": True,
            "session": {
                "platform": session.platform.key,
                "platform_name": session.platform.name,
                "account": session.account.name,
                "gpu_type": session.platform.gpu_type,
                "phase": session.phase.value,
                "type": "AUTO" if is_auto_platform(session.platform.key) else "MANUAL",
            },
            "push": {"ok": push_result.get("ok", False), "message": push_result.get("message", "")},
        }
        # Save state
        sm.save_runtime_state()

        if args.json:
            print(_json.dumps(result, indent=2))
        else:
            ptype = "AUTO" if is_auto_platform(session.platform.key) else "MANUAL"
            print(f"Session created: {session.platform.name}/{session.account.name} ({session.platform.gpu_type}, {ptype})")
            print(f"  Phase: {session.phase.value}")
            if is_auto_platform(session.platform.key):
                if session.platform.key == "kaggle":
                    print(f"  Status: Polling Kaggle until kernel is running...")
                else:
                    print(f"  Status: Auto-confirmed — countdown started")
            else:
                print(f"  Status: PENDING — run 'python tui.py --confirm' after uploading notebook")
            print(f"  Push: {push_result.get('message', 'N/A')}")
        return

    # ── --confirm: Confirm current session headlessly ────────────
    if args.confirm:
        sm, config = _build_sm(config_path)
        if not sm.current_session or not sm.current_session.is_pending:
            if args.json:
                print(_json.dumps({"ok": False, "error": "no_pending_session"}, indent=2))
            else:
                print("No pending session to confirm")
            sys.exit(1)

        if sm.confirm_session():
            s = sm.current_session
            sm.save_runtime_state()
            result = {
                "ok": True,
                "session": {
                    "platform": s.platform.key,
                    "account": s.account.name,
                    "phase": s.phase.value,
                    "remaining_seconds": round(s.remaining_seconds, 1),
                }
            }
            if args.json:
                print(_json.dumps(result, indent=2))
            else:
                print(f"Confirmed: {s.platform.name}/{s.account.name} — countdown started ({s.platform.session_limit_hours}h)")
        else:
            if args.json:
                print(_json.dumps({"ok": False, "error": "confirm_failed"}, indent=2))
            else:
                print("Failed to confirm session")
            sys.exit(1)
        return

    # ── --stop: Stop training headlessly ─────────────────────────
    if args.stop:
        sm, config = _build_sm(config_path)
        if not sm.is_training:
            if args.json:
                print(_json.dumps({"ok": False, "error": "not_training"}, indent=2))
            else:
                print("Not currently training")
            sys.exit(1)

        s = sm.current_session
        info = {
            "platform": s.platform.key,
            "account": s.account.name,
            "elapsed_seconds": round(s.elapsed_seconds, 1),
        }

        # Try handler stop
        from handlers import get_handler as _get_handler
        handler = _get_handler(s.platform.key)
        if handler:
            handler.stop_session(s.account, sm.entry_script)

        sm.stop()
        sm.save_runtime_state()

        result = {"ok": True, "stopped_session": info}
        if args.json:
            print(_json.dumps(result, indent=2))
        else:
            print(f"Stopped: {info['platform']}/{info['account']} (elapsed: {format_seconds(info['elapsed_seconds'])})")
        return

    # ── --done: Signal training complete, rotate ─────────────────
    if args.done:
        sm, config = _build_sm(config_path)
        if not sm.is_training:
            if args.json:
                print(_json.dumps({"ok": False, "error": "not_training"}, indent=2))
            else:
                print("Not currently training")
            sys.exit(1)

        rotated = sm.mark_training_complete()
        sm.save_runtime_state()

        if rotated:
            new_s = sm.current_session
            result = {
                "ok": True,
                "rotated": True,
                "new_session": {
                    "platform": new_s.platform.key,
                    "account": new_s.account.name,
                    "phase": new_s.phase.value,
                    "type": "AUTO" if is_auto_platform(new_s.platform.key) else "MANUAL",
                } if new_s and new_s.is_active else None,
            }
        else:
            result = {"ok": True, "rotated": False, "message": "No more accounts to rotate to"}

        if args.json:
            print(_json.dumps(result, indent=2))
        else:
            if rotated:
                if sm.current_session and sm.current_session.is_active:
                    ns = sm.current_session
                    print(f"Training complete — rotated to {ns.platform.name}/{ns.account.name}")
                else:
                    print(f"Training complete — no more accounts")
            else:
                print(f"Training complete — no accounts available for rotation")
        return

    # ── Default: Launch TUI ──────────────────────────────────────
    app = FreeGPUTrainerApp(config_path=config_path)
    app.run()


if __name__ == "__main__":
    run_app()
