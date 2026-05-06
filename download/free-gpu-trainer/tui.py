"""Free GPU Trainer — TUI Application.

A terminal-based tool for managing and rotating free GPU platform accounts
for continuous AI training and fine-tuning.

Usage:
    python tui.py                # Launch TUI
    python tui.py --status       # Show status and exit
"""

import sys
import os

# Ensure we can import sibling modules
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
    DataTable, Tree, RichLog, TabbedContent, TabPane,
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

        accounts_text = "\n".join(account_lines) if account_lines else "  [dim]No accounts configured[/dim]"

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
                f"  Press [bold reverse]S[/bold reverse] to start training\n"
                f"  Press [bold reverse]A[/bold reverse] to add accounts"
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
        table = DataTable()
        table.add_columns("Platform", "Account", "Status", "Sessions", "Hours", "Weekly", "GPU")
        for p in self.platforms:
            for acc in p.accounts:
                icon = STATUS_ICONS.get(acc.status, "?")
                table.add_row(
                    p.name,
                    acc.name,
                    f"{acc.status.value}",
                    str(acc.sessions_used),
                    f"{acc.total_hours_used:.1f}h",
                    f"{acc.weekly_hours_used:.1f}h",
                    p.gpu_type,
                )
        yield table


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

class AddAccountScreen(ModalScreen[str]):
    """Modal for adding a new account."""

    def compose(self) -> ComposeResult:
        with Container(id="add-account-dialog"):
            yield Label("[bold]Add Account[/bold]")
            yield Label("Platform keys:")
            yield Static(", ".join(PLATFORM_DEFS.keys()), classes="hint")
            yield Label("Edit config.yaml to add account tokens")
            yield Horizontal(
                Button("Cancel", variant="default", id="cancel-btn"),
                Button("OK", variant="success", id="add-btn"),
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add-btn":
            self.dismiss("added")
        else:
            self.dismiss("")


# ── Main App ───────────────────────────────────────────────────────

class FreeGPUTrainerApp(App):
    """Free GPU Trainer — Continuous AI training with free tier rotation."""

    TITLE = "Free GPU Trainer"
    SUB_TITLE = "Continuous AI Training with Free GPU Rotation"

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

    #add-account-dialog {
        padding: 2 4;
        width: 60;
        height: auto;
        background: $surface;
        border: thick $primary;
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
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "start_training", "Start"),
        Binding("x", "stop_training", "Stop"),
        Binding("r", "refresh", "Refresh"),
        Binding("a", "add_account", "Add Acct"),
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
        yield Footer()

    def on_mount(self) -> None:
        self._tick_timer = self.set_interval(1.0, self._tick)
        self._log("Free GPU Trainer started")
        self._log(f"Loaded {len(self.platforms)} platforms, "
                  f"{sum(p.total_accounts for p in self.platforms)} accounts")
        self._log(f"Total stackable: {self.session_manager.get_total_available_hours():.0f}h")
        self._log("Press [bold]S[/bold] to start training")

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
            self._log("[bold red]No available accounts![/bold red] Add accounts in config.yaml")

    def action_stop_training(self):
        if self.session_manager:
            self.session_manager.stop()
            self._log("[bold yellow]Training stopped.[/bold yellow]")
            self.training_active = False

    def action_refresh(self):
        self._tick()

    def action_add_account(self):
        self.push_screen(AddAccountScreen(), self._on_add_account)

    def _on_add_account(self, result: str):
        if result:
            self._log("Edit config.yaml to add account tokens")

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
