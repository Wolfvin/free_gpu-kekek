"""Free GPU Trainer — TUI Application.

A terminal-based tool for managing and rotating free GPU platform accounts
for continuous AI training and fine-tuning.

Usage:
    python tui.py                # Launch TUI
    python tui.py --status       # Show status and exit

Slash Commands (type in command bar):
    /help                        Show all commands
    /add <platform>              Add platform to active list
    /remove <platform>           Remove platform from active list
    /stack <platform> <name>     Stack a new account on a platform
    /unstack <platform> <name>   Remove an account from a platform
    /choose <platform>           Force next session on this platform
    /accounts [platform]         List accounts (all or for one platform)
    /platforms                   List all platforms with status
    /start                       Start training with auto-rotation
    /stop                        Stop training
    /status                      Show current session status
    /save                        Save config to config.yaml
    /reset                       Reset weekly counters
"""

import sys
import os
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from platforms import (
    PlatformConfig, AccountConfig, PlatformStatus, build_platform, PLATFORM_DEFS,
)
from session import Session, SessionManager

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
    DataTable, Tree, RichLog, TabbedContent, TabPane, Input,
)
from textual.screen import ModalScreen
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box


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
    """Save config back to YAML."""
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ── TUI Widgets ────────────────────────────────────────────────────

class PlatformCard(Static):
    """A card showing platform status and accounts."""

    def __init__(self, platform: PlatformConfig, **kwargs):
        super().__init__(**kwargs)
        self.platform = platform
        self._refresh()

    def _refresh(self):
        p = self.platform
        icon = STATUS_ICONS.get(p.status, "?")
        color = STATUS_COLORS.get(p.status, "white")

        account_lines = []
        for acc in p.accounts:
            acc_icon = STATUS_ICONS.get(acc.status, "?")
            acc_color = STATUS_COLORS.get(acc.status, "white")
            hours_info = f"{acc.total_hours_used:.1f}h used"
            if p.weekly_limit_hours:
                hours_info += f" / {p.weekly_limit_hours}h wk"
            account_lines.append(
                f"  {acc_icon} [{acc_color}]{acc.name}[/{acc_color}]  {hours_info}"
            )

        accounts_text = "\n".join(account_lines) if account_lines else "  [dim]No accounts — use /stack to add[/dim]"

        content = (
            f"[bold]{icon} {p.name}[/bold]\n"
            f"  GPU: [cyan]{p.gpu_type}[/cyan]  "
            f"Session: [white]{p.session_limit_hours}h[/white]  "
            f"Stack: [white]{p.total_accounts}x[/white] = "
            f"[green]{p.max_continuous_hours:.0f}h[/green]\n"
            f"{accounts_text}"
        )

        self.update(content)
        self.border_title = p.name
        self.border_subtitle = f"[{color}]{p.status.value}[/{color}]"


class SessionPanel(Static):
    """Panel showing current session info."""

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
                f"[bold yellow]▶ TRAINING ACTIVE[/bold yellow]\n\n"
                f"  Platform: [cyan]{s.platform.name}[/cyan]\n"
                f"  Account:  [white]{s.account.name}[/white]\n"
                f"  GPU:      [green]{s.platform.gpu_type}[/green]\n\n"
                f"  Elapsed:   [white]{format_seconds(elapsed)}[/white]\n"
                f"  Remaining: [yellow]{format_seconds(remaining)}[/yellow]\n"
                f"  {bar} {progress*100:.1f}%\n\n"
                f"  Sessions used: {s.account.sessions_used}  "
                f"Total hours: {s.account.total_hours_used:.1f}h"
            )
        else:
            total_h = sm.get_total_available_hours()
            next_acct = sm.get_next_account()
            next_info = "None"
            if next_acct:
                plat, acc = next_acct
                next_info = f"{plat.name}/{acc.name} ({plat.gpu_type}, {plat.session_limit_hours}h)"

            content = (
                f"[bold]⏸ IDLE[/bold]\n\n"
                f"  Total available: [green]{total_h:.1f}h[/green] across all accounts\n"
                f"  Next session: [cyan]{next_info}[/cyan]\n\n"
                f"  Type [bold]/start[/bold] to begin training\n"
                f"  Type [bold]/help[/bold] for all commands"
            )

        self.update(content)


class DashboardView(VerticalScroll):
    """Main dashboard view."""

    def __init__(self, platforms: list[PlatformConfig],
                 session_manager: SessionManager, **kwargs):
        super().__init__(**kwargs)
        self.platforms = platforms
        self.session_manager = session_manager

    def compose(self) -> ComposeResult:
        yield Label("[bold]═══ Current Session ═══[/bold]", classes="section-header")
        yield SessionPanel(self.session_manager, classes="session-panel")

        yield Label("[bold]═══ Platforms ═══[/bold]", classes="section-header")
        for platform in self.platforms:
            yield PlatformCard(platform, classes="platform-card")

    def refresh_data(self):
        for widget in self.query(PlatformCard):
            widget._refresh()
        for widget in self.query(SessionPanel):
            widget._refresh()


class AccountsView(VerticalScroll):
    """Detailed accounts view with a table."""

    def __init__(self, platforms: list[PlatformConfig], **kwargs):
        super().__init__(**kwargs)
        self.platforms = platforms

    def compose(self) -> ComposeResult:
        table = DataTable(id="accounts-table")
        table.add_columns("Platform", "Account", "Status", "Sessions", "Hours", "Weekly", "GPU")
        for p in self.platforms:
            for acc in p.accounts:
                table.add_row(
                    p.name,
                    acc.name,
                    acc.status.value,
                    str(acc.sessions_used),
                    f"{acc.total_hours_used:.1f}h",
                    f"{acc.weekly_hours_used:.1f}h",
                    p.gpu_type,
                )
        yield table

    def rebuild(self):
        """Rebuild the table data."""
        table = self.query_one("#accounts-table", DataTable)
        table.clear()
        for p in self.platforms:
            for acc in p.accounts:
                table.add_row(
                    p.name,
                    acc.name,
                    acc.status.value,
                    str(acc.sessions_used),
                    f"{acc.total_hours_used:.1f}h",
                    f"{acc.weekly_hours_used:.1f}h",
                    p.gpu_type,
                )


class ScheduleView(Static):
    """Shows the training schedule — rotation order and timing."""

    def __init__(self, platforms: list[PlatformConfig],
                 session_manager: SessionManager, **kwargs):
        super().__init__(**kwargs)
        self.platforms = platforms
        self.session_manager = session_manager
        self._refresh()

    def _refresh(self):
        schedule_lines = []
        cumulative_hours = 0.0

        for p in sorted(self.platforms, key=lambda x: x.session_limit_hours, reverse=True):
            if not p.enabled:
                continue
            for acc in p.accounts:
                if acc.status in (PlatformStatus.AVAILABLE, PlatformStatus.IN_USE, PlatformStatus.COOLDOWN):
                    start_h = cumulative_hours
                    end_h = cumulative_hours + p.session_limit_hours
                    status_icon = STATUS_ICONS.get(acc.status, "?")

                    if self.session_manager.current_session:
                        cs = self.session_manager.current_session
                        if cs.platform.key == p.key and cs.account.name == acc.name:
                            status_icon = "[bold yellow]▶[/bold yellow]"

                    schedule_lines.append(
                        f"  {status_icon} [{start_h:6.1f}h - {end_h:6.1f}h]  "
                        f"[cyan]{p.name:25s}[/cyan] "
                        f"[white]{acc.name:20s}[/white] "
                        f"[green]{p.gpu_type}[/green]"
                    )
                    cumulative_hours = end_h + p.cooldown_minutes / 60

        total = self.session_manager.get_total_available_hours()
        content = (
            f"[bold]═══ Rotation Schedule ═══[/bold]\n\n"
            f"  Total continuous training: [bold green]{total:.1f}h[/bold green]\n\n"
            + "\n".join(schedule_lines)
        )
        self.update(content)


class LogView(VerticalScroll):
    """Log output view."""

    def compose(self) -> ComposeResult:
        yield RichLog(id="training-log", highlight=True, markup=True)


# ── Modal Screens ──────────────────────────────────────────────────

class PlatformPickerScreen(ModalScreen[str]):
    """Modal to pick a platform from a list."""

    def __init__(self, title: str = "Pick Platform", platforms: list[PlatformConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self._title = title
        self._platforms = platforms or []

    def compose(self) -> ComposeResult:
        with Container(id="picker-dialog"):
            yield Label(f"[bold]{self._title}[/bold]")
            for p in self._platforms:
                icon = STATUS_ICONS.get(p.status, "?")
                yield Button(
                    f"{icon} {p.name} ({p.gpu_type}, {p.total_accounts} accts)",
                    id=f"pick-{p.key}",
                    variant="primary",
                )
            yield Button("Cancel", id="pick-cancel", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pick-cancel":
            self.dismiss("")
        elif event.button.id and event.button.id.startswith("pick-"):
            self.dismiss(event.button.id.replace("pick-", ""))


class HelpScreen(ModalScreen):
    """Modal showing all slash commands."""

    def compose(self) -> ComposeResult:
        with Container(id="help-dialog"):
            yield Label("[bold]Slash Commands[/bold]\n")
            yield Static(
                "[bold]/add[/bold] [dim]<platform>[/dim]        Add platform to active list\n"
                "[bold]/remove[/bold] [dim]<platform>[/dim]     Remove platform from active list\n"
                "[bold]/stack[/bold] [dim]<plat> <name>[/dim]   Stack new account on platform\n"
                "[bold]/unstack[/bold] [dim]<plat> <name>[/dim] Remove account from platform\n"
                "[bold]/choose[/bold] [dim]<platform>[/dim]     Force next session on platform\n"
                "[bold]/accounts[/bold] [dim>[plat][/dim]       List accounts (all or filtered)\n"
                "[bold]/platforms[/bold]              List all platforms with status\n"
                "[bold]/start[/bold]                  Start training with auto-rotation\n"
                "[bold]/stop[/bold]                   Stop training\n"
                "[bold]/status[/bold]                 Show current session status\n"
                "[bold]/save[/bold]                   Save config to config.yaml\n"
                "[bold]/reset[/bold]                  Reset weekly counters\n"
                "[bold]/help[/bold]                   Show this help\n\n"
                "[dim]Platform keys:[/dim] " + ", ".join(PLATFORM_DEFS.keys())
            )
            yield Button("Close", id="help-close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "help-close":
            self.dismiss()


# ── Main App ───────────────────────────────────────────────────────

class FreeGPUTrainerApp(App):
    """Free GPU Trainer — Continuous AI training with free tier rotation."""

    TITLE = "Free GPU Trainer"
    SUB_TITLE = "Type /help for commands"

    CSS = """
    Screen {
        background: $surface;
    }

    .section-header {
        text-align: center;
        padding: 1 0;
        color: $text;
        text-style: bold;
    }

    .session-panel {
        height: auto;
        min-height: 12;
        padding: 1 2;
        margin: 0 1 1 1;
        background: $surface-darken-1;
        border: round $primary;
    }

    .platform-card {
        height: auto;
        min-height: 5;
        padding: 1 2;
        margin: 0 1 1 1;
    }

    #picker-dialog {
        padding: 1 2;
        width: 70;
        height: auto;
        max-height: 25;
        background: $surface;
        border: thick $primary;
    }

    #help-dialog {
        padding: 2 4;
        width: 70;
        height: auto;
        background: $surface;
        border: thick $accent;
    }

    .hint {
        color: $text-muted;
        padding: 0 0 1 0;
    }

    TabbedContent {
        height: 1fr;
    }

    DataTable {
        height: 1fr;
    }

    RichLog {
        height: 1fr;
        border: solid $primary;
    }

    #command-bar {
        dock: bottom;
        height: 3;
        padding: 0 1;
        background: $primary-darken-2;
        border-top: solid $primary;
    }

    #command-input {
        width: 1fr;
    }

    #command-hint {
        width: auto;
        color: $text-muted;
        padding: 1 1 0 0;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "start_training", "Start"),
        Binding("x", "stop_training", "Stop"),
        Binding("r", "refresh", "Refresh"),
        Binding("slash", "focus_command", "Command", key_display="/"),
        Binding("t", "toggle_tab", "Next Tab"),
        Binding("1", "tab_dashboard", "Dashboard"),
        Binding("2", "tab_accounts", "Accounts"),
        Binding("3", "tab_schedule", "Schedule"),
        Binding("4", "tab_logs", "Logs"),
    ]

    training_active: reactive[bool] = reactive(False)

    def __init__(self, config_path: str = "config.yaml", **kwargs):
        super().__init__(**kwargs)
        self.config_path = config_path
        self.config = load_config(config_path)
        self.platforms: list[PlatformConfig] = []
        self.session_manager: Optional[SessionManager] = None
        self._tick_timer: Optional[Timer] = None
        self._forced_platform: Optional[str] = None
        self._setup_logging()
        self._build_platforms()

    def _setup_logging(self):
        log_cfg = self.config.get("logging", {})
        level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        self.logger = logging.getLogger("fgt")

    def _build_platforms(self):
        self.platforms = []
        platforms_cfg = self.config.get("platforms", {})
        for key, cfg in platforms_cfg.items():
            if key in PLATFORM_DEFS:
                self.platforms.append(build_platform(key, cfg))

        training_cfg = self.config.get("training", {})
        self.session_manager = SessionManager(
            platforms=self.platforms,
            auto_rotate=training_cfg.get("auto_rotate", True),
            rotate_buffer_minutes=training_cfg.get("rotate_buffer_minutes", 10),
            checkpoint_before_rotate=training_cfg.get("checkpoint_before_rotate", True),
        )

    def _rebuild_session_manager(self):
        """Rebuild session manager after platform/account changes."""
        training_cfg = self.config.get("training", {})
        was_training = self.session_manager.is_training if self.session_manager else False
        self.session_manager = SessionManager(
            platforms=self.platforms,
            auto_rotate=training_cfg.get("auto_rotate", True),
            rotate_buffer_minutes=training_cfg.get("rotate_buffer_minutes", 10),
            checkpoint_before_rotate=training_cfg.get("checkpoint_before_rotate", True),
        )

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="dashboard"):
            with TabPane("1:Dashboard", id="dashboard"):
                yield DashboardView(self.platforms, self.session_manager)
            with TabPane("2:Accounts", id="accounts"):
                yield AccountsView(self.platforms)
            with TabPane("3:Schedule", id="schedule"):
                yield ScheduleView(self.platforms, self.session_manager)
            with TabPane("4:Logs", id="logs"):
                yield LogView()
        with Horizontal(id="command-bar"):
            yield Label("/ ", id="command-hint")
            yield Input(placeholder="Type a command... (/help for list)", id="command-input")
        yield Footer()

    def on_mount(self) -> None:
        self._tick_timer = self.set_interval(1.0, self._tick)
        self._log("Free GPU Trainer started")
        self._log(f"Loaded {len(self.platforms)} platforms, "
                  f"{sum(p.total_accounts for p in self.platforms)} accounts")
        self._log(f"Total stackable: {self.session_manager.get_total_available_hours():.0f}h")
        self._log("Type [bold]/help[/bold] for slash commands  |  Press [bold]/[/bold] to focus command bar")

    def _tick(self):
        try:
            dashboard = self.query_one(DashboardView)
            dashboard.refresh_data()
            for sv in self.query(ScheduleView):
                sv._refresh()
            if self.session_manager:
                self.training_active = self.session_manager.is_training
        except Exception:
            pass

    # ── Slash Command Engine ────────────────────────────────────

    def action_focus_command(self):
        """Focus the command input bar."""
        self.query_one("#command-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle slash command input."""
        if event.input.id != "command-input":
            return
        raw = event.value.strip()
        if not raw:
            return
        # Clear input
        event.input.value = ""

        # If it doesn't start with /, treat as /<first_word>
        if not raw.startswith("/"):
            raw = "/" + raw

        self._execute_command(raw)

    def _execute_command(self, raw: str):
        """Parse and execute a slash command."""
        parts = raw.split()
        cmd = parts[0].lower().lstrip("/")
        args = parts[1:]

        dispatch = {
            "help": self._cmd_help,
            "add": self._cmd_add,
            "remove": self._cmd_remove,
            "stack": self._cmd_stack,
            "unstack": self._cmd_unstack,
            "choose": self._cmd_choose,
            "accounts": self._cmd_accounts,
            "platforms": self._cmd_platforms,
            "start": self._cmd_start,
            "stop": self._cmd_stop,
            "status": self._cmd_status,
            "save": self._cmd_save,
            "reset": self._cmd_reset,
        }

        handler = dispatch.get(cmd)
        if handler:
            handler(args)
        else:
            self._log(f"[red]Unknown command:[/red] /{cmd}  Type /help for list")

    def _find_platform(self, key: str) -> Optional[PlatformConfig]:
        """Find platform by key or partial name match."""
        # Exact key match
        for p in self.platforms:
            if p.key == key:
                return p
        # Partial name match
        key_lower = key.lower()
        for p in self.platforms:
            if key_lower in p.name.lower() or key_lower in p.key:
                return p
        return None

    # ── Command Implementations ─────────────────────────────────

    def _cmd_help(self, args):
        self.push_screen(HelpScreen())

    def _cmd_add(self, args):
        """Add a platform to the active list."""
        if not args:
            self.push_screen(PlatformPickerScreen(
                "Add Platform", self.platforms
            ), self._on_add_platform_pick)
            return

        key = args[0]
        # Check if already added
        if self._find_platform(key):
            self._log(f"[yellow]{key} already in active platforms[/yellow]")
            return

        if key not in PLATFORM_DEFS:
            # Try partial match
            matches = [k for k in PLATFORM_DEFS if key in k or key in PLATFORM_DEFS[k]["name"].lower()]
            if len(matches) == 1:
                key = matches[0]
            elif len(matches) > 1:
                self._log(f"[yellow]Ambiguous:[/yellow] matches: {', '.join(matches)}")
                return
            else:
                self._log(f"[red]Unknown platform:[/red] {args[0]}")
                self._log(f"[dim]Available: {', '.join(PLATFORM_DEFS.keys())}[/dim]")
                return

        defn = PLATFORM_DEFS[key]
        cfg = {"enabled": True, "accounts": []}
        new_platform = build_platform(key, cfg)
        self.platforms.append(new_platform)

        # Update config
        if "platforms" not in self.config:
            self.config["platforms"] = {}
        self.config["platforms"][key] = cfg

        self._rebuild_session_manager()
        self._rebuild_dashboard()
        self._log(f"[green]Added:[/green] {new_platform.name} (GPU: {new_platform.gpu_type})")
        self._log(f"[dim]Use /stack {key} <name> to add accounts[/dim]")

    def _on_add_platform_pick(self, key: str):
        if key:
            self._cmd_add([key])

    def _cmd_remove(self, args):
        """Remove a platform from active list."""
        if not args:
            self._log("[red]Usage:[/red] /remove <platform>")
            return

        p = self._find_platform(args[0])
        if not p:
            self._log(f"[red]Platform not found:[/red] {args[0]}")
            return

        self.platforms.remove(p)
        if p.key in self.config.get("platforms", {}):
            del self.config["platforms"][p.key]
        self._rebuild_session_manager()
        self._rebuild_dashboard()
        self._log(f"[yellow]Removed:[/yellow] {p.name}")

    def _cmd_stack(self, args):
        """Stack a new account onto a platform."""
        if len(args) < 2:
            self._log("[red]Usage:[/red] /stack <platform> <account_name>")
            self._log("[dim]Example: /stack google_colab my-alt-account[/dim]")
            return

        p = self._find_platform(args[0])
        if not p:
            self._log(f"[red]Platform not found:[/red] {args[0]}")
            self._log(f"[dim]Available: {', '.join(p.key for p in self.platforms)}[/dim]")
            return

        account_name = args[1]
        # Check if account name already exists
        for acc in p.accounts:
            if acc.name == account_name:
                self._log(f"[yellow]Account already exists:[/yellow] {account_name}")
                return

        new_account = AccountConfig(name=account_name)
        p.accounts.append(new_account)

        # Update config
        if p.key in self.config.get("platforms", {}):
            self.config["platforms"][p.key].setdefault("accounts", []).append({"name": account_name})

        self._rebuild_session_manager()
        self._rebuild_dashboard()
        self._log(
            f"[green]Stacked:[/green] {account_name} on {p.name} "
            f"(now {p.total_accounts} accounts = {p.max_continuous_hours:.0f}h)"
        )

    def _cmd_unstack(self, args):
        """Remove an account from a platform."""
        if len(args) < 2:
            self._log("[red]Usage:[/red] /unstack <platform> <account_name>")
            return

        p = self._find_platform(args[0])
        if not p:
            self._log(f"[red]Platform not found:[/red] {args[0]}")
            return

        account_name = args[1]
        found = None
        for acc in p.accounts:
            if acc.name == account_name:
                found = acc
                break

        if not found:
            self._log(f"[red]Account not found:[/red] {account_name} on {p.name}")
            return

        if found.status == PlatformStatus.IN_USE:
            self._log("[red]Cannot remove account currently in use![/red]")
            return

        p.accounts.remove(found)

        # Update config
        if p.key in self.config.get("platforms", {}):
            accounts_cfg = self.config["platforms"][p.key].get("accounts", [])
            self.config["platforms"][p.key]["accounts"] = [
                a for a in accounts_cfg if a.get("name") != account_name
            ]

        self._rebuild_session_manager()
        self._rebuild_dashboard()
        self._log(
            f"[yellow]Unstacked:[/yellow] {account_name} from {p.name} "
            f"(now {p.total_accounts} accounts = {p.max_continuous_hours:.0f}h)"
        )

    def _cmd_choose(self, args):
        """Force the next session to use a specific platform."""
        if not args:
            # Show platform picker
            self.push_screen(PlatformPickerScreen(
                "Choose Platform for Next Session", self.platforms
            ), self._on_choose_platform_pick)
            return

        p = self._find_platform(args[0])
        if not p:
            self._log(f"[red]Platform not found:[/red] {args[0]}")
            return

        if not p.available_accounts:
            self._log(f"[red]No available accounts[/red] on {p.name}")
            return

        self._forced_platform = p.key
        # Reorder platforms so chosen one is first
        self.platforms.sort(key=lambda x: 0 if x.key == p.key else 1)
        self._rebuild_session_manager()
        self._log(
            f"[cyan]Next session will use:[/cyan] {p.name} "
            f"({p.gpu_type}, {len(p.available_accounts)} available)"
        )

    def _on_choose_platform_pick(self, key: str):
        if key:
            self._cmd_choose([key])

    def _cmd_accounts(self, args):
        """List accounts, optionally filtered by platform."""
        if args:
            p = self._find_platform(args[0])
            if not p:
                self._log(f"[red]Platform not found:[/red] {args[0]}")
                return
            self._log(f"[bold]{p.name}[/bold] — {p.total_accounts} accounts:")
            for acc in p.accounts:
                icon = STATUS_ICONS.get(acc.status, "?")
                self._log(
                    f"  {icon} {acc.name}  status={acc.status.value}  "
                    f"sessions={acc.sessions_used}  hours={acc.total_hours_used:.1f}h"
                )
        else:
            for p in self.platforms:
                self._log(f"[bold]{p.name}[/bold] — {p.total_accounts} accounts, {p.max_continuous_hours:.0f}h stack:")
                for acc in p.accounts:
                    icon = STATUS_ICONS.get(acc.status, "?")
                    self._log(
                        f"  {icon} {acc.name}  status={acc.status.value}  "
                        f"sessions={acc.sessions_used}  hours={acc.total_hours_used:.1f}h"
                    )

    def _cmd_platforms(self, args):
        """List all platforms with status."""
        for p in self.platforms:
            icon = STATUS_ICONS.get(p.status, "?")
            self._log(
                f"  {icon} {p.key:20s} {p.name:30s} "
                f"GPU: {p.gpu_type:20s} "
                f"{p.total_accounts} accts = {p.max_continuous_hours:.0f}h  "
                f"[{p.status.value}]"
            )
        total = self.session_manager.get_total_available_hours()
        self._log(f"  [bold green]Total stackable: {total:.0f}h[/bold green]")

    def _cmd_start(self, args):
        """Start training with auto-rotation."""
        self.action_start_training()

    def _cmd_stop(self, args):
        """Stop training."""
        self.action_stop_training()

    def _cmd_status(self, args):
        """Show current session status."""
        sm = self.session_manager
        if sm.current_session and sm.current_session.is_active:
            s = sm.current_session
            self._log(
                f"[yellow]Training active:[/yellow] {s.platform.name}/{s.account.name} "
                f"GPU: {s.platform.gpu_type} "
                f"Elapsed: {format_seconds(s.elapsed_seconds)} "
                f"Remaining: {format_seconds(s.remaining_seconds)}"
            )
        else:
            total = sm.get_total_available_hours()
            next_acct = sm.get_next_account()
            if next_acct:
                plat, acc = next_acct
                self._log(
                    f"Idle. Next: {plat.name}/{acc.name} ({plat.gpu_type}, {plat.session_limit_hours}h)"
                )
            else:
                self._log("Idle. No available accounts.")
            self._log(f"Total available: {total:.1f}h")

    def _cmd_save(self, args):
        """Save current config to config.yaml."""
        save_config(self.config, self.config_path)
        self._log(f"[green]Config saved to {self.config_path}[/green]")

    def _cmd_reset(self, args):
        """Reset weekly counters."""
        self.session_manager.reset_weekly()
        self._log("[green]Weekly counters reset[/green]")

    # ── Rebuild helpers ─────────────────────────────────────────

    def _rebuild_dashboard(self):
        """Rebuild the dashboard with current platform list."""
        try:
            dashboard = self.query_one(DashboardView)
            # Remove old platform cards and re-add
            dashboard.remove_children()
            dashboard._platform_widgets = []
            from textual.widgets import Label
            dashboard.mount(Label("[bold]═══ Current Session ═══[/bold]", classes="section-header"))
            dashboard.mount(SessionPanel(self.session_manager, classes="session-panel"))
            dashboard.mount(Label("[bold]═══ Platforms ═══[/bold]", classes="section-header"))
            for platform in self.platforms:
                dashboard.mount(PlatformCard(platform, classes="platform-card"))

            # Also rebuild accounts table
            try:
                accounts_view = self.query_one(AccountsView)
                accounts_view.platforms = self.platforms
                accounts_view.rebuild()
            except Exception:
                pass
        except Exception as e:
            self._log(f"[dim]Dashboard refresh: {e}[/dim]")

    # ── Training Actions ────────────────────────────────────────

    def action_start_training(self):
        if self.session_manager.is_training:
            self._log("Training already in progress")
            return

        self._log("Starting training session...")

        def on_rotate(old_session, new_session):
            if new_session:
                self._log(
                    f"[bold green]ROTATED:[/bold green] "
                    f"{old_session.platform.name}/{old_session.account.name} -> "
                    f"{new_session.platform.name}/{new_session.account.name}"
                )
                self._generate_run_script(new_session)
            else:
                self._log("[bold red]No account available for rotation![/bold red] Training stopped.")

        session = self.session_manager.start_session(on_rotate=on_rotate)
        if session:
            self._log(
                f"[bold green]STARTED:[/bold green] "
                f"{session.platform.name}/{session.account.name} "
                f"(GPU: {session.platform.gpu_type}, "
                f"limit: {session.platform.session_limit_hours}h)"
            )
            self._generate_run_script(session)
            self.training_active = True
        else:
            self._log("[bold red]No available accounts![/bold red] Use /stack to add accounts")

    def action_stop_training(self):
        if self.session_manager:
            self.session_manager.stop()
            self._log("[bold yellow]Training stopped.[/bold yellow]")
            self.training_active = False

    def action_refresh(self):
        self._tick()

    def action_toggle_tab(self):
        tc = self.query_one(TabbedContent)
        tabs = list(tc.tab_ids)
        if tabs:
            current = tc.active
            try:
                idx = tabs.index(current)
            except ValueError:
                idx = -1
            next_idx = (idx + 1) % len(tabs)
            tc.active = tabs[next_idx]

    def action_tab_dashboard(self):
        self.query_one(TabbedContent).active = "dashboard"

    def action_tab_accounts(self):
        self.query_one(TabbedContent).active = "accounts"

    def action_tab_schedule(self):
        self.query_one(TabbedContent).active = "schedule"

    def action_tab_logs(self):
        self.query_one(TabbedContent).active = "logs"

    def _log(self, message: str):
        try:
            log = self.query_one("#training-log", RichLog)
            timestamp = time.strftime("%H:%M:%S")
            log.write(f"[dim][{timestamp}][/dim] {message}")
        except Exception:
            pass

    def _generate_run_script(self, session: Session):
        from trainer import TrainingJob
        training_cfg = self.config.get("training", {})
        job = TrainingJob(
            script_path=training_cfg.get("entry_script", "train.py"),
            checkpoint_dir=training_cfg.get("checkpoint_dir", "./checkpoints"),
            resume=training_cfg.get("resume_from_checkpoint", True),
        )

        bash_script = job.generate_run_command(session.platform)
        script_path = Path("./run_session.sh")
        script_path.write_text(bash_script)

        if session.platform.key in ("google_colab", "kaggle", "paperspace", "sagemaker", "deepnote"):
            notebook_code = job.generate_notebook_code(session.platform)
            nb_path = Path("./run_session_notebook.py")
            nb_path.write_text(notebook_code)
            self._log(
                f"[cyan]Notebook code generated:[/cyan] run_session_notebook.py — "
                f"paste into {session.platform.name} cell"
            )
        else:
            self._log(
                f"[cyan]Run script generated:[/cyan] run_session.sh — run: bash run_session.sh"
            )


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
            print(f"  Total available: {sm.get_total_available_hours():.1f}h")
            print(f"  Platforms: {len(platforms)}")
            for p in platforms:
                total_accts = p.total_accounts
                stack_h = p.max_continuous_hours
                print(f"    {p.name:35s} ({p.gpu_type:20s}) {total_accts} accts = {stack_h:.0f}h")
            print()
            return
        elif sys.argv[1] == "--config":
            config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"

    app = FreeGPUTrainerApp(config_path=config_path)
    app.run()


if __name__ == "__main__":
    run_app()
