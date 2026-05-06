"""Session manager: handles rotation, cooldowns, and scheduling across accounts."""

import time
import threading
import logging
from typing import Optional, Callable

from platforms import (
    PlatformConfig, AccountConfig, PlatformStatus,
)

logger = logging.getLogger("free-gpu-trainer")


class Session:
    """Represents an active training session on a specific account."""

    def __init__(self, platform: PlatformConfig, account: AccountConfig,
                 on_expire: Optional[Callable] = None):
        self.platform = platform
        self.account = account
        self.started_at = time.time()
        self.limit_seconds = platform.session_limit_hours * 3600
        self._on_expire = on_expire
        self._timer: Optional[threading.Timer] = None
        self._active = True

        # Mark account as in use
        account.status = PlatformStatus.IN_USE
        account.current_session_start = self.started_at
        account.sessions_used += 1

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        return max(0, self.limit_seconds - self.elapsed_seconds)

    @property
    def progress(self) -> float:
        return min(1.0, self.elapsed_seconds / self.limit_seconds)

    @property
    def is_active(self) -> bool:
        return self._active and self.remaining_seconds > 0

    def end(self):
        """Manually end this session."""
        self._active = False
        if self._timer:
            self._timer.cancel()
            self._timer = None
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

    def __init__(self, platforms: list[PlatformConfig],
                 auto_rotate: bool = True,
                 rotate_buffer_minutes: int = 10,
                 checkpoint_before_rotate: bool = True):
        self.platforms = platforms
        self.auto_rotate = auto_rotate
        self.rotate_buffer_seconds = rotate_buffer_minutes * 60
        self.checkpoint_before_rotate = checkpoint_before_rotate
        self.current_session: Optional[Session] = None
        self.session_history: list[dict] = []
        self._rotation_timer: Optional[threading.Timer] = None
        self._on_rotate: Optional[Callable] = None
        self._lock = threading.Lock()
        self._running = False

    @property
    def is_training(self) -> bool:
        return self.current_session is not None and self.current_session.is_active

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
        """Start a new session on the next available account."""
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

        # Set up auto-rotation timer
        if self.auto_rotate:
            rotate_at = session.remaining_seconds - self.rotate_buffer_seconds
            if rotate_at > 0:
                self._rotation_timer = threading.Timer(rotate_at, self._auto_rotate)
                self._rotation_timer.daemon = True
                self._rotation_timer.start()

        logger.info(
            f"Session started: {platform.name}/{account.name} "
            f"(limit: {platform.session_limit_hours}h, "
            f"GPU: {platform.gpu_type})"
        )
        return session

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
