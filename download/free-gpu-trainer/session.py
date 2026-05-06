"""Session manager: handles rotation, cooldowns, and scheduling across accounts.

Key design decisions:
  - Sessions start in PENDING state — timer doesn't count down until confirmed
  - Auto-platforms (Kaggle API, SSH) are auto-confirmed after push_code succeeds
  - Manual platforms (Colab, notebooks) require /confirm from user
  - check_status() is polled periodically to detect real platform state
  - If check_status() reports stopped while session is active, session ends
"""

import time
import threading
import logging
from typing import Optional, Callable
from enum import Enum

from platforms import (
    PlatformConfig, AccountConfig, PlatformStatus,
)

logger = logging.getLogger("free-gpu-trainer")


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
    """Manages sessions across all platforms with auto-rotation."""

    # Platforms with real API push — can be auto-confirmed
    AUTO_PLATFORMS = {"kaggle", "oracle_cloud", "gcp"}
    # HuggingFace is semi-auto — push works but it's NOT for training
    # Manual platforms need /confirm from user

    def __init__(self, platforms: list[PlatformConfig],
                 auto_rotate: bool = True,
                 rotate_buffer_minutes: int = 10,
                 checkpoint_before_rotate: bool = True,
                 entry_script: str = "train.py"):
        self.platforms = platforms
        self.auto_rotate = auto_rotate
        self.rotate_buffer_seconds = rotate_buffer_minutes * 60
        self.checkpoint_before_rotate = checkpoint_before_rotate
        self.entry_script = entry_script
        self.current_session: Optional[Session] = None
        self.session_history: list[dict] = []
        self._rotation_timer: Optional[threading.Timer] = None
        self._on_rotate: Optional[Callable] = None
        self._lock = threading.Lock()
        self._running = False

    @property
    def is_training(self) -> bool:
        return self.current_session is not None and self.current_session.is_active

    def is_auto_platform(self, platform_key: str) -> bool:
        """Check if a platform can be auto-confirmed after push."""
        return platform_key in self.AUTO_PLATFORMS

    def get_next_account(self) -> Optional[tuple[PlatformConfig, AccountConfig]]:
        """Find the next available account across all platforms.
        
        Priority:
        1. Same platform (stack accounts)
        2. Other platforms with available accounts
        """
        # First try same platform (stack accounts)
        if self.current_session:
            current_platform = self.current_session.platform
            for account in current_platform.accounts:
                if account.status == PlatformStatus.AVAILABLE:
                    return (current_platform, account)

        # Then try all platforms sorted by session limit (longer first)
        sorted_platforms = sorted(
            self.platforms,
            key=lambda p: p.session_limit_hours,
            reverse=True
        )
        for platform in sorted_platforms:
            if not platform.enabled:
                continue
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
        """
        if not self.current_session or not self.current_session.is_pending:
            return False

        self.current_session.confirm()

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

        # Check if there's a next account available
        result = self._peek_next_account()
        if result:
            # End current and start next
            old_session = self.current_session
            old_name = f"{old_session.platform.name}/{old_session.account.name}"
            old_session.end()

            new_session = self.start_session(self._on_rotate)
            if new_session:
                # Auto-confirm the new session for auto-platforms
                if self.is_auto_platform(new_session.platform.key):
                    new_session.confirm()
                    # Set up rotation timer for new session
                    if self.auto_rotate:
                        rotate_at = new_session.remaining_seconds - self.rotate_buffer_seconds
                        if rotate_at > 0:
                            self._rotation_timer = threading.Timer(rotate_at, self._auto_rotate)
                            self._rotation_timer.daemon = True
                            self._rotation_timer.start()

                new_name = f"{new_session.platform.name}/{new_session.account.name}"
                logger.info(f"Rotated: {old_name} -> {new_name}")
                if self._on_rotate:
                    self._on_rotate(old_session, new_session)
            else:
                logger.warning("No account available for rotation")
                self._running = False
                if self._on_rotate:
                    self._on_rotate(old_session, None)
        else:
            logger.warning("No next account available — training will end")

    def _peek_next_account(self) -> Optional[tuple[PlatformConfig, AccountConfig]]:
        """Peek at the next available account without starting a session."""
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

        # Other platforms
        sorted_platforms = sorted(
            self.platforms,
            key=lambda p: p.session_limit_hours,
            reverse=True
        )
        for platform in sorted_platforms:
            if not platform.enabled:
                continue
            if self.current_session and platform.key == self.current_session.platform.key:
                continue
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
            self.current_session = None

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
            summary.append({
                "key": platform.key,
                "name": platform.name,
                "enabled": platform.enabled,
                "status": platform.status.value,
                "gpu_type": platform.gpu_type,
                "session_limit": platform.session_limit_hours,
                "accounts": accounts_info,
                "max_continuous_hours": platform.max_continuous_hours,
            })
        return summary

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
