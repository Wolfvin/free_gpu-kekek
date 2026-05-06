"""Free GPU Trainer — TUI Application.

A terminal-based tool for managing and rotating free GPU platform accounts
for continuous AI training and fine-tuning.

Usage:
    python tui.py                # Launch TUI
    python tui.py --status       # Show status and exit

Slash Commands:
    /add         → Pick platform → Enter credentials → See detail + stack accounts
    /remove      → Pick platform → Remove it
    /choose      → Pick platform → Force next session there
    /start       Start training
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
from session import Session, SessionManager
from vault import encrypt_credentials, delete_credentials, get_storage_mode
from handlers import validate_account_name

import time
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
            # Show ALL platform definitions (for /add new)
            for key, defn in PLATFORM_DEFS.items():
                # Check if already in active platforms
                already = any(p.key == key for p in self._platforms)
                tag = " [dim](added)[/dim]" if already else ""
                cred_count = len(CREDENTIAL_SCHEMAS.get(key, []))
                cred_info = f"  Creds: [yellow]{cred_count} fields[/yellow]" if cred_count else "  [dim]No API needed[/dim]"
                ol.add_option(
                    Text.from_markup(
                        f"[bold]{defn['name']}[/bold]{tag}\n"
                        f"  GPU: [cyan]{defn['gpu_type']}[/cyan]  "
                        f"Session: [white]{defn['session_limit_hours']}h[/white]  "
                        f"URL: [dim]{defn['url']}[/dim]\n"
                        f"{cred_info}"
                    )
                )
        else:
            # Show only active platforms
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
    """Multi-field credential input form for a platform.

    Returns dict with:
      - "name": account name
      - "credentials": {key: value, ...}
    Or empty dict if cancelled.
    """

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
                    "  [dim]These are saved locally in config.yaml and never sent anywhere else.[/dim]",
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
            # Focus first cred field or submit
            if self.cred_fields:
                first_key = self.cred_fields[0]["key"]
                try:
                    self.query_one(f"#cred-field-{first_key}", Input).focus()
                except Exception:
                    self._submit(skip_creds=False)
            else:
                self._submit(skip_creds=False)
        else:
            # Advance to next field or submit
            self._advance_or_submit(event.input.id)

    def _advance_or_submit(self, current_id: str):
        """After a credential field is submitted, go to next field or submit."""
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
        # Last field or unknown — submit
        self._submit(skip_creds=False)

    def _submit(self, skip_creds: bool = False):
        name_input = self.query_one("#cred-name-input", Input)
        name = name_input.value.strip()
        if not name:
            name_input.focus()
            return

        # Validate account name
        is_valid, err_msg = validate_account_name(name)
        if not is_valid:
            # Show error in the input placeholder and refocus
            name_input.value = ""
            name_input.placeholder = f"✗ {err_msg}"
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
    """Shows platform detail with stacked accounts. Can add/remove accounts."""

    def __init__(self, platform: PlatformConfig, config: dict, **kwargs):
        super().__init__(**kwargs)
        self.platform = platform
        self.config = config

    def compose(self) -> ComposeResult:
        p = self.platform
        icon = STATUS_ICONS.get(p.status, "?")
        cred_schema = CREDENTIAL_SCHEMAS.get(p.key, [])
        cred_info = f"  Creds needed: [yellow]{len(cred_schema)} fields[/yellow]" if cred_schema else ""

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
                f"{cred_info}",
                id="pdetail-info",
            )
            yield Label("[bold]Stacked Accounts[/bold]", id="pdetail-acct-label")

            if p.accounts:
                for acc in p.accounts:
                    acc_icon = STATUS_ICONS.get(acc.status, "?")
                    # Show credential status
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
                    updates[cf["key"]] = None  # Signal to remove
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
                "[bold]/start[/bold]    Start training with auto-rotation\n"
                "[bold]/stop[/bold]     Stop training\n"
                "[bold]/status[/bold]   Show current session info\n"
                "[bold]/save[/bold]     Save config to config.yaml\n"
                "[bold]/reset[/bold]    Reset weekly counters\n"
                "[bold]/help[/bold]     Show this help\n\n"
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
            remaining = s.remaining_seconds
            elapsed = s.elapsed_seconds
            progress = s.progress
            bar_len = 30
            filled = int(bar_len * progress)
            empty = bar_len - filled
            bar = f"[green]{'█' * filled}[/green][dim]{'░' * empty}[/dim]"
            content = (
                f"[bold yellow]▶ TRAINING[/bold yellow]\n\n"
                f"  Platform: [cyan]{s.platform.name}[/cyan]\n"
                f"  Account:  [white]{s.account.name}[/white]\n"
                f"  GPU:      [green]{s.platform.gpu_type}[/green]\n\n"
                f"  Elapsed:   [white]{format_seconds(elapsed)}[/white]  "
                f"Remaining: [yellow]{format_seconds(remaining)}[/yellow]\n"
                f"  {bar} {progress*100:.1f}%\n\n"
                f"  Sessions: {s.account.sessions_used}  "
                f"Hours: {s.account.total_hours_used:.1f}h"
            )
        else:
            total_h = sm.get_total_available_hours()
            next_acct = sm.get_next_account()
            next_info = "None"
            if next_acct:
                plat, acc = next_acct
                next_info = f"{plat.name}/{acc.name} ({plat.gpu_type})"
            content = (
                f"[bold]⏸ IDLE[/bold]\n\n"
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
        acct_lines = []
        for acc in p.accounts:
            ai = STATUS_ICONS.get(acc.status, "?")
            cred_status = ""
            if cred_schema:
                filled = sum(1 for cf in cred_schema if acc.credentials.get(cf["key"]))
                total = len(cred_schema)
                if filled == total:
                    cred_status = " [green]✓[/green]"
                elif filled > 0:
                    cred_status = f" [yellow]{filled}/{total}[/yellow]"
                else:
                    cred_status = " [red]✗[/red]"
            acct_lines.append(f"  {ai} {acc.name}  [dim]{acc.total_hours_used:.1f}h[/dim]{cred_status}")
        accts = "\n".join(acct_lines) if acct_lines else "  [dim]no accounts[/dim]"
        content = (
            f"[bold]{icon} {p.name}[/bold]  "
            f"[cyan]{p.gpu_type}[/cyan]  "
            f"{p.session_limit_hours}h/sess  "
            f"[white]{p.total_accounts}x[/white]=[green]{p.max_continuous_hours:.0f}h[/green]\n"
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
        yield Label("[bold]═══ Session ═══[/bold]", classes="section-header")
        yield SessionPanel(self.session_manager, classes="session-panel")
        yield Label("[bold]═══ Platforms ═══[/bold]", classes="section-header")
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
                    lines.append(
                        f"  {ico} [{cum:6.1f}h-{end:6.1f}h]  "
                        f"[cyan]{p.name}[/cyan]  [white]{acc.name}[/white]  "
                        f"[green]{p.gpu_type}[/green]"
                    )
                    cum = end + p.cooldown_minutes / 60
        total = self.session_manager.get_total_available_hours()
        self.update(
            f"[bold]Rotation Schedule[/bold]\n\n"
            f"  Total: [bold green]{total:.1f}h[/bold green]\n\n"
            + "\n".join(lines)
        )


class LogView(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield RichLog(id="training-log", highlight=True, markup=True)


# ── Main App ───────────────────────────────────────────────────────

class FreeGPUTrainerApp(App):
    TITLE = "Free GPU Trainer"
    SUB_TITLE = "/add to add platform  /start to train"

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
        Binding("x", "stop_training", "Stop"),
        Binding("slash", "focus_command", "/", key_display="/"),
        Binding("1", "tab_dashboard", "Dash"),
        Binding("2", "tab_schedule", "Sched"),
        Binding("3", "tab_logs", "Logs"),
    ]

    training_active: reactive[bool] = reactive(False)

    def __init__(self, config_path="config.yaml", **kwargs):
        super().__init__(**kwargs)
        self.config_path = config_path
        self.config = load_config(config_path)
        self.platforms: list[PlatformConfig] = []
        self.session_manager: Optional[SessionManager] = None
        self._tick_timer: Optional[Timer] = None
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
        for key, cfg in self.config.get("platforms", {}).items():
            if key in PLATFORM_DEFS:
                self.platforms.append(build_platform(key, cfg, config_dir))
        tc = self.config.get("training", {})
        self.session_manager = SessionManager(
            platforms=self.platforms,
            auto_rotate=tc.get("auto_rotate", True),
            rotate_buffer_minutes=tc.get("rotate_buffer_minutes", 10),
            checkpoint_before_rotate=tc.get("checkpoint_before_rotate", True),
        )

    def _rebuild(self):
        tc = self.config.get("training", {})
        self.session_manager = SessionManager(
            platforms=self.platforms,
            auto_rotate=tc.get("auto_rotate", True),
            rotate_buffer_minutes=tc.get("rotate_buffer_minutes", 10),
            checkpoint_before_rotate=tc.get("checkpoint_before_rotate", True),
        )
        self._rebuild_dashboard()

    def _rebuild_dashboard(self):
        try:
            d = self.query_one(DashboardView)
            d.remove_children()
            d.mount(Label("[bold]═══ Session ═══[/bold]", classes="section-header"))
            d.mount(SessionPanel(self.session_manager, classes="session-panel"))
            d.mount(Label("[bold]═══ Platforms ═══[/bold]", classes="section-header"))
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
            yield Input(placeholder="type a command... (/add /start /help)", id="command-input")
        yield Footer()

    def on_mount(self) -> None:
        self._tick_timer = self.set_interval(1.0, self._tick)
        self._log("Free GPU Trainer started")
        self._log(f"{len(self.platforms)} platforms, "
                  f"{sum(p.total_accounts for p in self.platforms)} accounts, "
                  f"{self.session_manager.get_total_available_hours():.0f}h total")
        self._log(f"[dim]Credential storage: {get_storage_mode()}[/dim]")
        self._log("[bold]/add[/bold] add platform + credentials  [bold]/start[/bold] train  [bold]/help[/bold] commands")

    def _tick(self):
        try:
            d = self.query_one(DashboardView)
            d.refresh_data()
            for sv in self.query(ScheduleView):
                sv._refresh()
            if self.session_manager:
                self.training_active = self.session_manager.is_training
        except Exception:
            pass

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
            "stop": self._cmd_stop,
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
        """Show platform list → pick → enter credentials → show detail."""
        self.push_screen(
            PlatformListScreen("Pick Platform", self.platforms, show_empty=True),
            self._on_add_pick,
        )

    def _on_add_pick(self, key: str):
        if not key:
            return

        # If platform not yet added, add it first
        existing = None
        for p in self.platforms:
            if p.key == key:
                existing = p
                break

        if existing:
            # Platform already added — go to detail
            self._open_platform_detail(existing)
        else:
            # New platform — add it, then go straight to credential input
            defn = PLATFORM_DEFS[key]
            cfg = {"enabled": True, "accounts": []}
            new_p = build_platform(key, cfg)
            self.platforms.append(new_p)
            if "platforms" not in self.config:
                self.config["platforms"] = {}
            self.config["platforms"][key] = cfg
            self._rebuild()
            self._log(f"[green]Added:[/green] {new_p.name}")
            # Now prompt for first account + credentials
            self._prompt_add_account(new_p)

    def _prompt_add_account(self, platform: PlatformConfig):
        """Open credential input screen for adding a new account."""
        self.push_screen(
            CredentialInputScreen(platform.key),
            lambda result, p=platform: self._on_credential_input(p, result),
        )

    def _on_credential_input(self, platform: PlatformConfig, result: dict):
        """Handle result from CredentialInputScreen."""
        if not result or not result.get("name"):
            if platform.accounts:
                self._open_platform_detail(platform)
            return

        name = result["name"]
        creds = result.get("credentials", {})

        # Check duplicate
        for acc in platform.accounts:
            if acc.name == name:
                self._log(f"[yellow]Already exists:[/yellow] {name}")
                self._prompt_add_account(platform)
                return

        # Encrypt credentials for storage
        config_dir = os.path.dirname(os.path.abspath(self.config_path))
        encrypted_creds = encrypt_credentials(platform.key, name, creds, config_dir)

        # Store decrypted creds in memory for runtime use
        new_acc = AccountConfig(name=name, credentials=creds)
        self._sync_legacy_fields(new_acc)

        platform.accounts.append(new_acc)
        # Save encrypted creds to config dict
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
        """Sync credentials dict to legacy token/api_key fields for handler compatibility."""
        creds = acc.credentials
        # Kaggle
        if "kaggle_username" in creds or "kaggle_key" in creds:
            import json as _json
            acc.token = _json.dumps({"username": creds.get("kaggle_username", ""), "key": creds.get("kaggle_key", "")})
            acc.api_key = creds.get("kaggle_key")
        # HuggingFace
        if "hf_token" in creds:
            acc.token = creds["hf_token"]
        # Oracle Cloud
        if "oci_vm_host" in creds:
            acc.token = creds.get("oci_vm_host")
        # GCP
        if "gcp_vm_host" in creds:
            acc.token = creds.get("gcp_vm_host")
        # Generic: if there's a token field, set it
        if "token" in creds:
            acc.token = creds["token"]
        if "api_key" in creds:
            acc.api_key = creds["api_key"]

    def _save_account_to_config(self, platform: PlatformConfig, name: str, creds: dict):
        """Save account with credentials to config dict."""
        if platform.key not in self.config.get("platforms", {}):
            self.config["platforms"][platform.key] = {"enabled": True, "accounts": []}

        acct_cfg = {"name": name}
        if creds:
            acct_cfg["credentials"] = creds
        self.config["platforms"][platform.key].setdefault("accounts", []).append(acct_cfg)

    def _open_platform_detail(self, platform: PlatformConfig):
        """Open the detail screen for a platform."""
        self.push_screen(
            PlatformDetailScreen(platform, self.config),
            lambda result, p=platform: self._on_detail_result(p, result),
        )

    def _on_detail_result(self, platform: PlatformConfig, result: str):
        if not result or result == "done":
            self._rebuild()
            return

        if result == "add_account":
            # Open credential input for new account
            self._prompt_add_account(platform)

        elif result.startswith("edit_creds:"):
            name = result[11:]
            # Find account
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
            # Legacy add without creds — redirect to credential input
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
                    # Delete from keyring too
                    delete_credentials(platform.key, name)
                    self._rebuild()
            self._open_platform_detail(platform)

    def _on_edit_creds(self, platform: PlatformConfig, account: AccountConfig, result: dict):
        """Handle credential edit results."""
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
            # Re-encrypt and update config
            config_dir = os.path.dirname(os.path.abspath(self.config_path))
            encrypted = encrypt_credentials(platform.key, account.name, account.credentials, config_dir)
            self._update_account_creds_in_config(platform, account, encrypted)
            self._rebuild()
            self._log(f"[green]Updated {changed} credentials[/green] for {account.name}")
            self._log(f"[dim]Storage: {get_storage_mode()}[/dim]")

        self._open_platform_detail(platform)

    def _update_account_creds_in_config(self, platform: PlatformConfig, account: AccountConfig,
                                         encrypted_creds: dict = None):
        """Update credentials in config.yaml for a specific account."""
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
            self._log(f"[cyan]Next session →[/cyan] {p.name} ({p.gpu_type})")

    def _cmd_start(self):
        if self.session_manager.is_training:
            self._log("Already training")
            return
        self._log("Starting...")

        def on_rotate(old, new):
            if new:
                self._log(f"[green]ROTATED:[/green] {old.platform.name}/{old.account.name} → "
                          f"{new.platform.name}/{new.account.name}")
                self._gen_script(new)
            else:
                self._log("[red]No account for rotation![/red] Stopped.")

        session = self.session_manager.start_session(on_rotate=on_rotate)
        if session:
            self._log(f"[green]STARTED:[/green] {session.platform.name}/{session.account.name} "
                      f"({session.platform.gpu_type}, {session.platform.session_limit_hours}h)")
            self._gen_script(session)
            self.training_active = True
        else:
            self._log("[red]No accounts![/red] Use /add then stack accounts with credentials")

    def _cmd_stop(self):
        if self.session_manager and self.session_manager.current_session:
            # Use real handler to stop session
            from handlers import get_handler
            s = self.session_manager.current_session
            handler = get_handler(s.platform.key)
            if handler:
                result = handler.stop_session(s.account)
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
            self._log(f"[yellow]Active:[/yellow] {s.platform.name}/{s.account.name} "
                      f"{s.platform.gpu_type} "
                      f"elapsed={format_seconds(s.elapsed_seconds)} "
                      f"remaining={format_seconds(s.remaining_seconds)}")
        else:
            nxt = sm.get_next_account()
            if nxt:
                plat, acc = nxt
                self._log(f"Idle. Next: {plat.name}/{acc.name} ({plat.gpu_type})")
            else:
                self._log("Idle. No accounts.")
            self._log(f"Total: {sm.get_total_available_hours():.1f}h")

    def _cmd_save(self):
        save_config(self.config, self.config_path)
        self._log(f"[green]Saved → {self.config_path}[/green]")

    def _cmd_reset(self):
        self.session_manager.reset_weekly()
        self._log("[green]Weekly counters reset[/green]")

    def _cmd_help(self):
        self.push_screen(HelpScreen())

    # ── Training Actions ────────────────────────────────────────

    def action_start_training(self):
        self._cmd_start()

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
        from handlers import get_handler
        tc = self.config.get("training", {})
        script_path = tc.get("entry_script", "train.py")
        checkpoint_dir = tc.get("checkpoint_dir", "./checkpoints")
        resume = tc.get("resume_from_checkpoint", True)

        # Generate local scripts
        job = TrainingJob(script_path=script_path, checkpoint_dir=checkpoint_dir, resume=resume)
        Path("./run_session.sh").write_text(job.generate_run_command(session.platform))
        self._log(f"[cyan]Script →[/cyan] run_session.sh")

        # Use real handler to push code to platform
        handler = get_handler(session.platform.key)
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
                if result.get("notebook_path"):
                    self._log(f"[cyan]Notebook →[/cyan] {result['notebook_path']}")
            else:
                self._log(f"[red]Handler failed:[/red] {result.get('message', 'unknown error')}")
        else:
            self._log(f"[dim]No handler for {session.platform.key} — using local scripts only[/dim]")


def run_app():
    config_path = "config.yaml"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--status":
            config = load_config(config_path)
            platforms = []
            for key, cfg in config.get("platforms", {}).items():
                if key in PLATFORM_DEFS:
                    platforms.append(build_platform(key, cfg))
            sm = SessionManager(platforms)
            print(f"\n  Free GPU Trainer — Status\n")
            print(f"  Total: {sm.get_total_available_hours():.1f}h  Platforms: {len(platforms)}")
            for p in platforms:
                print(f"    {p.name:35s} ({p.gpu_type:20s}) {p.total_accounts}x = {p.max_continuous_hours:.0f}h")
            print()
            return
        elif sys.argv[1] == "--config":
            config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    app = FreeGPUTrainerApp(config_path=config_path)
    app.run()


if __name__ == "__main__":
    run_app()
