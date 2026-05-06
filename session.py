"""Session manager: handles rotation, cooldowns, and scheduling across accounts.

Key design decisions:
  - Sessions start in PENDING state — timer doesn't count down until confirmed
  - Auto-platforms (Kaggle API, SSH) are auto-confirmed after push_code succeeds
  - Manual platforms (Colab, notebooks) require /confirm from user
  - check_status() is polled periodically to detect real platform state
  - If check_status() reports stopped while session is active, session ends
  - AUTO platforms are prioritized over MANUAL in scheduling (agent-friendly)
  - Event callbacks allow agents to react to session lifecycle changes
  - Runtime state is persisted to state.json for crash recovery
"""

import time
import json
import threading
import logging
from pathlib import Path
from typing import Optional, Callable
from enum import Enum

from platforms import (
    PlatformConfig, AccountConfig, PlatformStatus,
)

logger = logging.getLogger("free-gpu-trainer")


def _format_seconds(s: float) -> str:
    """Format seconds as H:MM:SS for logging."""
    if s <= 0:
        return "0:00:00"
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h}:{m:02d}:{sec:02d}"


class SessionPhase(Enum):
    """Lifecycle phases of a session."""
    PENDING = "pending"       # Created but not yet confirmed running on platform
    CONFIRMED = "confirmed"   # Training confirmed running — timer counting down
    EXPIRED = "expired"       # Session time limit reached
    ENDED = "ended"           # Session manually ended or platform reported stopped


class Session:
    """Represents an active training session on a specific account."""

    def __init__(self, platform: PlatformConfig, account: AccountConfig,
                 on_expire: Optional[Callable] = None):
        self.platform = platform
        self.account = account
        self._phase: SessionPhase = SessionPhase.PENDING
        self._confirmed_at: Optional[float] = None  # When user confirmed training is running
        self.limit_seconds = platform.session_limit_hours * 3600
        self._on_expire = on_expire
        self._timer: Optional[threading.Timer] = None
        self._rotation_timer: Optional[threading.Timer] = None
        self._active = True
        self._last_status_check: Optional[str] = None  # Last known platform status

        # Mark account as in use (but phase is still PENDING)
        account.status = PlatformStatus.IN_USE
        account.sessions_used += 1

    @property
    def phase(self) -> SessionPhase:
        return self._phase

    @property
    def is_pending(self) -> bool:
        """Session created but not yet confirmed as running on the platform."""
        return self._phase == SessionPhase.PENDING

    @property
    def is_confirmed(self) -> bool:
        """Training confirmed running — countdown is active."""
        return self._phase == SessionPhase.CONFIRMED

    def confirm(self):
        """Confirm that training is actually running on the platform.

        This starts the countdown timer. For auto-platforms, this is called
        automatically after successful push_code. For manual platforms, user
        must call /confirm.
        """
        if self._phase != SessionPhase.PENDING:
            return
        self._phase = SessionPhase.CONFIRMED
        self._confirmed_at = time.time()
        self.account.current_session_start = self._confirmed_at
        logger.info(
            f"Session confirmed: {self.platform.name}/{self.account.name} "
            f"(limit: {self.platform.session_limit_hours}h)"
        )

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since confirmation (0 if still pending)."""
        if self._confirmed_at is None:
            return 0.0
        return time.time() - self._confirmed_at

    @property
    def remaining_seconds(self) -> float:
        """Seconds remaining before session limit (full limit if pending)."""
        if self._confirmed_at is None:
            return self.limit_seconds
        return max(0, self.limit_seconds - self.elapsed_seconds)

    @property
    def progress(self) -> float:
        """Progress through session (0 if pending)."""
        if self._confirmed_at is None:
            return 0.0
        return min(1.0, self.elapsed_seconds / self.limit_seconds)

    @property
    def is_active(self) -> bool:
        return self._active and self._phase in (SessionPhase.PENDING, SessionPhase.CONFIRMED)

    def update_platform_status(self, status: str):
        """Update the last known status from the platform's check_status()."""
        self._last_status_check = status
        # If platform reports training is stopped/complete while we think it's active,
        # end the session
        if self._phase == SessionPhase.CONFIRMED and status in ("stopped", "complete", "error"):
            logger.warning(
                f"Platform reports status '{status}' for "
                f"{self.platform.name}/{self.account.name} — ending session"
            )
            self.end()

    def end(self):
        """Manually end this session."""
        self._active = False
        self._phase = SessionPhase.ENDED
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if self._rotation_timer:
            self._rotation_timer.cancel()
            self._rotation_timer = None
        elapsed_h = self.elapsed_seconds / 3600
        self.account.total_hours_used += elapsed_h
        self.account.weekly_hours_used += elapsed_h
        self.account.current_session_start = None
        # Put account into cooldown
        cooldown_s = self.platform.cooldown_minutes * 60
        if cooldown_s > 0:
            self.account.status = PlatformStatus.COOLDOWN
            self.account.cooldown_until = time.time() + cooldown_s
            cooldown_timer = threading.Timer(cooldown_s, self._end_cooldown, args=[self.account])
            cooldown_timer.daemon = True
            cooldown_timer.start()
        else:
            self.account.status = PlatformStatus.AVAILABLE
        logger.info(
            f"Session ended: {self.platform.name}/{self.account.name} "
            f"({elapsed_h:.1f}h used)"
        )

    def _end_cooldown(self, account: AccountConfig):
        if account.status == PlatformStatus.COOLDOWN:
            # Check weekly limit
            if (self.platform.weekly_limit_hours and
                    account.weekly_hours_used >= self.platform.weekly_limit_hours):
                account.status = PlatformStatus.WEEKLY_LIMIT
            else:
                account.status = PlatformStatus.AVAILABLE
            logger.info(f"Cooldown ended: {self.platform.name}/{account.name}")


class SessionManager:
    """Manages sessions across all platforms with auto-rotation.

    Scheduling priority (AUTO-first for agent friendliness):
    1. Same platform (stack accounts) — if current session's platform
    2. AUTO platforms with available accounts (Kaggle, SSH)
    3. MANUAL platforms with available accounts (Colab, notebooks)

    Event callbacks (for AI agents and TUI integration):
      sm.on_session_confirmed = callback(session)
      sm.on_session_expired = callback(session)
      sm.on_rotation_needed = callback(old_session)  # before rotate
      sm.on_no_accounts = callback()                  # no accounts left

    State persistence:
      sm.save_runtime_state() / sm.load_runtime_state() for crash recovery
      Persists weekly_hours_used, sessions_used, cooldown_until per account.
    """

    # Platforms with real API push — can be auto-confirmed
    AUTO_PLATFORMS = {"kaggle", "oracle_cloud", "gcp"}
    # HuggingFace is semi-auto — push works but it's NOT for training
    # Manual platforms need /confirm from user

    def __init__(self, platforms: list[PlatformConfig],
                 auto_rotate: bool = True,
                 rotate_buffer_minutes: int = 10,
                 checkpoint_before_rotate: bool = True,
                 entry_script: str = "train.py",
                 state_path: str = "state.json"):
        self.platforms = platforms
        self.auto_rotate = auto_rotate
        self.rotate_buffer_seconds = rotate_buffer_minutes * 60
        self.checkpoint_before_rotate = checkpoint_before_rotate
        self.entry_script = entry_script
        self.state_path = state_path
        self.current_session: Optional[Session] = None
        self.session_history: list[dict] = []
        self._rotation_timer: Optional[threading.Timer] = None
        self._on_rotate: Optional[Callable] = None
        self._lock = threading.Lock()
        self._running = False

        # ── Event callbacks (for agents / TUI) ─────────────────
        self.on_session_confirmed: Optional[Callable] = None
        self.on_session_expired: Optional[Callable] = None
        self.on_rotation_needed: Optional[Callable] = None
        self.on_no_accounts: Optional[Callable] = None

        # Load persisted state on startup
        self.load_runtime_state()

    @property
    def is_training(self) -> bool:
        return self.current_session is not None and self.current_session.is_active

    def is_auto_platform(self, platform_key: str) -> bool:
        """Check if a platform can be auto-confirmed after push."""
        return platform_key in self.AUTO_PLATFORMS

    def get_next_account(self) -> Optional[tuple[PlatformConfig, AccountConfig]]:
        """Find the next available account across all platforms.
        
        Priority (AUTO-first for agent friendliness):
        1. Same platform (stack accounts) — regardless of AUTO/MANUAL
        2. AUTO platforms with available accounts (Kaggle, SSH)
        3. MANUAL platforms with available accounts (Colab, notebooks)

        Within each tier, sort by session_limit_hours descending (longer first).
        """
        # First try same platform (stack accounts)
        if self.current_session:
            current_platform = self.current_session.platform
            for account in current_platform.accounts:
                if account.status == PlatformStatus.AVAILABLE:
                    return (current_platform, account)

        # Then AUTO platforms first, then MANUAL
        auto_platforms = [p for p in self.platforms
                         if p.enabled and p.key in self.AUTO_PLATFORMS]
        manual_platforms = [p for p in self.platforms
                           if p.enabled and p.key not in self.AUTO_PLATFORMS]

        for tier in [auto_platforms, manual_platforms]:
            sorted_tier = sorted(tier, key=lambda p: p.session_limit_hours, reverse=True)
            for platform in sorted_tier:
                for account in platform.accounts:
                    if account.status == PlatformStatus.AVAILABLE:
                        return (platform, account)
        return None

    def start_session(self, on_rotate: Optional[Callable] = None) -> Optional[Session]:
        """Start a new session on the next available account.

        The session starts in PENDING state — it must be confirmed
        (via confirm_session) before the countdown timer begins.
        """
        self._on_rotate = on_rotate
        result = self.get_next_account()
        if not result:
            logger.warning("No available accounts to start session")
            return None

        platform, account = result
        session = Session(platform, account)
        with self._lock:
            self.current_session = session
            self._running = True

        logger.info(
            f"Session created (PENDING): {platform.name}/{account.name} "
            f"(limit: {platform.session_limit_hours}h, GPU: {platform.gpu_type})"
        )
        return session

    def confirm_session(self) -> bool:
        """Confirm the current session is running on the platform.

        Starts the countdown timer and sets up auto-rotation.
        Returns True if confirmed, False if no active session.
        Fires on_session_confirmed event callback.
        """
        if not self.current_session or not self.current_session.is_pending:
            return False

        self.current_session.confirm()

        # Fire event callback
        if self.on_session_confirmed:
            try:
                self.on_session_confirmed(self.current_session)
            except Exception as e:
                logger.debug(f"on_session_confirmed callback error: {e}")

        # Set up auto-rotation timer (only after confirmation)
        if self.auto_rotate:
            rotate_at = self.current_session.remaining_seconds - self.rotate_buffer_seconds
            if rotate_at > 0:
                self._rotation_timer = threading.Timer(rotate_at, self._auto_rotate)
                self._rotation_timer.daemon = True
                self._rotation_timer.start()

        return True

    def _auto_rotate(self):
        """Called when session is about to expire — rotate to next account."""
        if not self.current_session or not self.current_session.is_active:
            return

        logger.info(
            f"Auto-rotating from {self.current_session.platform.name}/"
            f"{self.current_session.account.name}"
        )

        # Fire pre-rotation event (agent can reject or prepare)
        if self.on_rotation_needed:
            try:
                self.on_rotation_needed(self.current_session)
            except Exception as e:
                logger.debug(f"on_rotation_needed callback error: {e}")

        # Check if there's a next account available
        result = self._peek_next_account()
        if result:
            # End current and start next
            old_session = self.current_session
            old_name = f"{old_session.platform.name}/{old_session.account.name}"
            old_session.end()

            # Fire expired event
            if self.on_session_expired:
                try:
                    self.on_session_expired(old_session)
                except Exception as e:
                    logger.debug(f"on_session_expired callback error: {e}")

            new_session = self.start_session(self._on_rotate)
            if new_session:
                # Auto-confirm the new session for auto-platforms
                if self.is_auto_platform(new_session.platform.key):
                    self._auto_confirm_with_polling(new_session)

                new_name = f"{new_session.platform.name}/{new_session.account.name}"
                logger.info(f"Rotated: {old_name} -> {new_name}")
                if self._on_rotate:
                    self._on_rotate(old_session, new_session)
            else:
                logger.warning("No account available for rotation")
                self._running = False
                if self._on_rotate:
                    self._on_rotate(old_session, None)
                if self.on_no_accounts:
                    try:
                        self.on_no_accounts()
                    except Exception as e:
                        logger.debug(f"on_no_accounts callback error: {e}")
        else:
            logger.warning("No next account available — training will end")
            if self.on_no_accounts:
                try:
                    self.on_no_accounts()
                except Exception as e:
                    logger.debug(f"on_no_accounts callback error: {e}")

    def _peek_next_account(self) -> Optional[tuple[PlatformConfig, AccountConfig]]:
        """Peek at the next available account without starting a session.

        Uses same AUTO-first priority as get_next_account().
        """
        if self.current_session:
            current_platform = self.current_session.platform
            # Same platform first
            for account in current_platform.accounts:
                if account.status == PlatformStatus.AVAILABLE:
                    return (current_platform, account)
            # Cooldown accounts that will be ready soon
            for account in current_platform.accounts:
                if (account.status == PlatformStatus.COOLDOWN and
                        account.cooldown_until and
                        account.cooldown_until <= time.time() + self.rotate_buffer_seconds):
                    return (current_platform, account)

        # Other platforms: AUTO first, then MANUAL
        auto_platforms = [p for p in self.platforms
                         if p.enabled and p.key in self.AUTO_PLATFORMS
                         and (not self.current_session or p.key != self.current_session.platform.key)]
        manual_platforms = [p for p in self.platforms
                           if p.enabled and p.key not in self.AUTO_PLATFORMS
                           and (not self.current_session or p.key != self.current_session.platform.key)]

        for tier in [auto_platforms, manual_platforms]:
            sorted_tier = sorted(tier, key=lambda p: p.session_limit_hours, reverse=True)
            for platform in sorted_tier:
                for account in platform.accounts:
                    if account.status == PlatformStatus.AVAILABLE:
                        return (platform, account)
        return None

    def stop(self):
        """Stop the current session and all timers."""
        self._running = False
        if self._rotation_timer:
            self._rotation_timer.cancel()
            self._rotation_timer = None
        if self.current_session:
            self.current_session.end()
            # Fire expired event
            if self.on_session_expired:
                try:
                    self.on_session_expired(self.current_session)
                except Exception as e:
                    logger.debug(f"on_session_expired callback error: {e}")
            self.current_session = None
        # Persist state on stop
        self.save_runtime_state()

    def check_current_session_status(self) -> Optional[str]:
        """Check the real status of the current session from the platform.

        Returns the status string from the handler, or None if no session.
        Also updates session state if platform reports it stopped.
        """
        if not self.current_session or not self.current_session.is_active:
            return None

        from handlers import get_handler
        handler = get_handler(self.current_session.platform.key)
        if not handler:
            return None

        try:
            result = handler.check_status(self.current_session.account, self.entry_script)
            if result.get("ok"):
                status = result.get("status", "unknown")
                self.current_session.update_platform_status(status)
                return status
        except Exception as e:
            logger.debug(f"Status check failed: {e}")
            return "error"

        return None

    def get_total_available_hours(self) -> float:
        """Calculate total available training hours across all accounts."""
        total = 0.0
        for platform in self.platforms:
            if not platform.enabled:
                continue
            for account in platform.accounts:
                if account.status in (PlatformStatus.AVAILABLE, PlatformStatus.COOLDOWN):
                    remaining = platform.session_limit_hours
                    if platform.weekly_limit_hours:
                        remaining = max(0, platform.weekly_limit_hours - account.weekly_hours_used)
                    total += remaining
        return total

    def get_status_summary(self) -> list[dict]:
        """Get a summary of all platforms and accounts."""
        summary = []
        for platform in self.platforms:
            accounts_info = []
            for account in platform.accounts:
                accounts_info.append({
                    "name": account.name,
                    "status": account.status.value,
                    "sessions_used": account.sessions_used,
                    "total_hours": round(account.total_hours_used, 1),
                    "weekly_hours": round(account.weekly_hours_used, 1),
                })
            ptype = "AUTO" if platform.key in self.AUTO_PLATFORMS else "MANUAL"
            summary.append({
                "key": platform.key,
                "name": platform.name,
                "enabled": platform.enabled,
                "status": platform.status.value,
                "gpu_type": platform.gpu_type,
                "session_limit": platform.session_limit_hours,
                "type": ptype,
                "accounts": accounts_info,
                "max_continuous_hours": platform.max_continuous_hours,
            })
        return summary

    def get_status_json(self) -> dict:
        """Get full status as a machine-readable dict for --status --json."""
        result = {
            "training": self.is_training,
            "total_available_hours": round(self.get_total_available_hours(), 1),
            "platforms": self.get_status_summary(),
        }
        if self.current_session and self.current_session.is_active:
            s = self.current_session
            result["current_session"] = {
                "platform": s.platform.key,
                "platform_name": s.platform.name,
                "account": s.account.name,
                "gpu_type": s.platform.gpu_type,
                "phase": s.phase.value,
                "type": "AUTO" if s.platform.key in self.AUTO_PLATFORMS else "MANUAL",
                "elapsed_seconds": round(s.elapsed_seconds, 1),
                "remaining_seconds": round(s.remaining_seconds, 1),
                "session_limit_hours": s.platform.session_limit_hours,
            }
        else:
            next_acct = self.get_next_account()
            if next_acct:
                plat, acc = next_acct
                result["next_account"] = {
                    "platform": plat.key,
                    "platform_name": plat.name,
                    "account": acc.name,
                    "gpu_type": plat.gpu_type,
                    "type": "AUTO" if plat.key in self.AUTO_PLATFORMS else "MANUAL",
                }
        return result

    def reset_weekly(self):
        """Reset weekly counters — call at start of new week."""
        for platform in self.platforms:
            for account in platform.accounts:
                if account.status == PlatformStatus.WEEKLY_LIMIT:
                    account.status = PlatformStatus.AVAILABLE
                account.weekly_hours_used = 0.0

    def auto_reset_weekly_if_needed(self):
        """Check if a new week has started and auto-reset weekly counters.

        Uses ISO week number: if the current week differs from the last
        recorded week, all weekly counters are reset and WEEKLY_LIMIT
        accounts are moved back to AVAILABLE.

        This is called from the TUI tick on every iteration so it's
        essentially free — no I/O, just a date comparison.
        """
        import datetime
        current_week = datetime.date.today().isocalendar()[1]
        current_year = datetime.date.today().isocalendar()[0]
        week_key = f"{current_year}-W{current_week:02d}"

        if not hasattr(self, '_last_reset_week'):
            self._last_reset_week = week_key
            return

        if week_key != self._last_reset_week:
            logger.info(f"New week detected ({week_key}), auto-resetting weekly counters")
            self.reset_weekly()
            self._last_reset_week = week_key

    # ── Training Completion ─────────────────────────────────────

    def mark_training_complete(self):
        """Signal that training has finished early (e.g. all epochs done).

        Ends the current session and triggers rotation to the next account.
        This prevents wasting GPU time sitting idle until the session timer
        expires.

        Returns True if a rotation was triggered, False if no session active.
        """
        if not self.current_session or not self.current_session.is_active:
            return False

        logger.info(
            f"Training complete — ending session early: "
            f"{self.current_session.platform.name}/{self.current_session.account.name} "
            f"({_format_seconds(self.current_session.elapsed_seconds)} used of "
            f"{_format_seconds(self.current_session.limit_seconds)} limit)"
        )

        # Save runtime state before ending
        self.save_runtime_state()

        # Fire rotation-needed event so agent can prepare
        if self.on_rotation_needed:
            try:
                self.on_rotation_needed(self.current_session)
            except Exception as e:
                logger.debug(f"on_rotation_needed callback error: {e}")

        # End current session
        old_session = self.current_session
        old_session.end()

        # Fire expired event
        if self.on_session_expired:
            try:
                self.on_session_expired(old_session)
            except Exception as e:
                logger.debug(f"on_session_expired callback error: {e}")

        # Try to rotate to next account
        new_session = self.start_session(self._on_rotate)
        if new_session:
            if self.is_auto_platform(new_session.platform.key):
                self._auto_confirm_with_polling(new_session)

            if self._on_rotate:
                self._on_rotate(old_session, new_session)
            return True
        else:
            self._running = False
            if self._on_rotate:
                self._on_rotate(old_session, None)
            if self.on_no_accounts:
                try:
                    self.on_no_accounts()
                except Exception as e:
                    logger.debug(f"on_no_accounts callback error: {e}")
            return False

    # ── Kaggle Delayed Confirm ───────────────────────────────────

    def _auto_confirm_with_polling(self, session: Session):
        """Auto-confirm a session, with Kaggle-specific delayed confirm.

        For Kaggle: after push, kernels enter a queue and may not be
        running immediately. We poll check_status() every 30 seconds
        for up to 5 minutes. Only after the kernel is "running" do we
        call confirm() and start the countdown timer.

        For other AUTO platforms (SSH): confirm immediately since the
        nohup command starts the process right away.
        """
        if session.platform.key == "kaggle":
            # Delayed confirm: poll until running or timeout
            threading.Thread(
                target=self._poll_kaggle_confirm, args=(session,),
                daemon=True
            ).start()
        else:
            # Immediate confirm for SSH platforms
            session.confirm()
            if self.auto_rotate:
                rotate_at = session.remaining_seconds - self.rotate_buffer_seconds
                if rotate_at > 0:
                    self._rotation_timer = threading.Timer(rotate_at, self._auto_rotate)
                    self._rotation_timer.daemon = True
                    self._rotation_timer.start()
            # Fire confirmed event
            if self.on_session_confirmed:
                try:
                    self.on_session_confirmed(session)
                except Exception as e:
                    logger.debug(f"on_session_confirmed callback error: {e}")

    def _poll_kaggle_confirm(self, session: Session):
        """Poll Kaggle check_status until running, then confirm.

        Polls every 30 seconds for up to 5 minutes (10 attempts).
        This prevents starting the countdown timer while the kernel
        is still queued, which would waste GPU time counting down
        before training actually starts.
        """
        max_attempts = 10
        poll_interval = 30  # seconds

        for attempt in range(max_attempts):
            time.sleep(poll_interval)
            try:
                from handlers import get_handler
                handler = get_handler(session.platform.key)
                if handler:
                    result = handler.check_status(session.account, self.entry_script)
                    if result.get("ok") and result.get("status") == "running":
                        logger.info(f"Kaggle kernel running — confirming session after {attempt + 1} polls")
                        session.confirm()
                        if self.auto_rotate:
                            rotate_at = session.remaining_seconds - self.rotate_buffer_seconds
                            if rotate_at > 0:
                                self._rotation_timer = threading.Timer(rotate_at, self._auto_rotate)
                                self._rotation_timer.daemon = True
                                self._rotation_timer.start()
                        # Fire confirmed event
                        if self.on_session_confirmed:
                            try:
                                self.on_session_confirmed(session)
                            except Exception as e:
                                logger.debug(f"on_session_confirmed callback error: {e}")
                        return
                    logger.debug(f"Kaggle not running yet (attempt {attempt + 1}/{max_attempts}): {result.get('status', 'unknown')}")
            except Exception as e:
                logger.debug(f"Kaggle status poll error: {e}")

        # Timeout — confirm anyway so the timer starts (better than stuck in PENDING forever)
        logger.warning(f"Kaggle confirm timeout ({max_attempts * poll_interval}s) — confirming anyway")
        session.confirm()
        if self.auto_rotate:
            rotate_at = session.remaining_seconds - self.rotate_buffer_seconds
            if rotate_at > 0:
                self._rotation_timer = threading.Timer(rotate_at, self._auto_rotate)
                self._rotation_timer.daemon = True
                self._rotation_timer.start()
        if self.on_session_confirmed:
            try:
                self.on_session_confirmed(session)
            except Exception as e:
                logger.debug(f"on_session_confirmed callback error: {e}")

    # ── Runtime State Persistence ────────────────────────────────

    def save_runtime_state(self):
        """Persist runtime state to disk for crash recovery.

        Saves per-account: weekly_hours_used, total_hours_used, sessions_used,
        cooldown_until, status, and the current session info.
        This allows the app to recover state after a crash or restart.
        """
        state = {
            "version": 1,
            "last_reset_week": getattr(self, '_last_reset_week', None),
            "saved_at": time.time(),
            "platforms": {},
        }
        for platform in self.platforms:
            accounts = []
            for acc in platform.accounts:
                accounts.append({
                    "name": acc.name,
                    "status": acc.status.value,
                    "sessions_used": acc.sessions_used,
                    "total_hours_used": acc.total_hours_used,
                    "weekly_hours_used": acc.weekly_hours_used,
                    "cooldown_until": acc.cooldown_until,
                    "current_session_start": acc.current_session_start,
                })
            state["platforms"][platform.key] = accounts

        # Save current session info if active
        if self.current_session and self.current_session.is_active:
            s = self.current_session
            state["current_session"] = {
                "platform_key": s.platform.key,
                "account_name": s.account.name,
                "phase": s.phase.value,
                "confirmed_at": s._confirmed_at,
            }

        try:
            Path(self.state_path).write_text(json.dumps(state, indent=2))
            logger.debug(f"Runtime state saved to {self.state_path}")
        except Exception as e:
            logger.warning(f"Failed to save runtime state: {e}")

    def load_runtime_state(self):
        """Load runtime state from disk to recover after crash/restart.

        Restores per-account: weekly_hours_used, total_hours_used, sessions_used,
        cooldown_until, and status. Does NOT resume sessions (those are gone
        after a crash — user must /start again).
        """
        path = Path(self.state_path)
        if not path.exists():
            return

        try:
            state = json.loads(path.read_text())
            if state.get("version") != 1:
                return

            # Restore weekly reset tracker
            if state.get("last_reset_week"):
                self._last_reset_week = state["last_reset_week"]

            # Restore per-account state
            for platform in self.platforms:
                saved_accounts = state.get("platforms", {}).get(platform.key, [])
                for saved in saved_accounts:
                    for acc in platform.accounts:
                        if acc.name == saved["name"]:
                            acc.sessions_used = saved.get("sessions_used", 0)
                            acc.total_hours_used = saved.get("total_hours_used", 0.0)
                            acc.weekly_hours_used = saved.get("weekly_hours_used", 0.0)
                            acc.cooldown_until = saved.get("cooldown_until")
                            acc.current_session_start = saved.get("current_session_start")
                            # Restore status (but don't override IN_USE since session is gone)
                            saved_status = saved.get("status", "available")
                            if saved_status != "in_use":
                                acc.status = PlatformStatus(saved_status)
                            else:
                                # Session is gone after restart — put back to available
                                acc.status = PlatformStatus.AVAILABLE
                            # Check if cooldown has expired
                            if acc.status == PlatformStatus.COOLDOWN and acc.cooldown_until:
                                if acc.cooldown_until <= time.time():
                                    acc.status = PlatformStatus.AVAILABLE
                                    acc.cooldown_until = None
                            break

            logger.info(f"Runtime state loaded from {self.state_path}")
        except Exception as e:
            logger.warning(f"Failed to load runtime state: {e}")
