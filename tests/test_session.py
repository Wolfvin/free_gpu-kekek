"""Tests for session.py — session management and rotation."""

import os
import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from platforms import PlatformConfig, AccountConfig, PlatformStatus
from session import Session, SessionManager, SessionPhase


def _make_platform(key="kaggle", limit=9, accounts=None, cooldown=5):
    """Helper to create a PlatformConfig for testing."""
    if accounts is None:
        accounts = [AccountConfig(name=f"acct-{key}-1")]
    return PlatformConfig(
        key=key,
        name=f"Test {key}",
        url=f"https://{key}.example.com",
        gpu_type="T4",
        session_limit_hours=limit,
        cooldown_minutes=cooldown,
        accounts=accounts,
    )


class TestSession:
    """Test Session class."""

    def test_initial_state_is_pending(self):
        p = _make_platform()
        s = Session(p, p.accounts[0])
        assert s.phase == SessionPhase.PENDING
        assert s.is_pending
        assert not s.is_confirmed
        assert s.is_active

    def test_confirm_transitions_to_confirmed(self):
        p = _make_platform()
        s = Session(p, p.accounts[0])
        s.confirm()
        assert s.phase == SessionPhase.CONFIRMED
        assert s.is_confirmed
        assert not s.is_pending
        assert s.elapsed_seconds >= 0

    def test_remaining_seconds_full_when_pending(self):
        p = _make_platform(limit=9)
        s = Session(p, p.accounts[0])
        # Pending session should have full limit remaining
        assert s.remaining_seconds == 9 * 3600

    def test_end_transitions_to_ended(self):
        p = _make_platform()
        s = Session(p, p.accounts[0])
        s.confirm()
        s.end()
        assert s.phase == SessionPhase.ENDED
        assert not s.is_active

    def test_update_platform_status_stops_session(self):
        p = _make_platform()
        s = Session(p, p.accounts[0])
        s.confirm()
        s.update_platform_status("stopped")
        assert s.phase == SessionPhase.ENDED

    def test_update_platform_status_ignores_when_pending(self):
        p = _make_platform()
        s = Session(p, p.accounts[0])
        # Should not end a pending session
        s.update_platform_status("stopped")
        assert s.is_pending


class TestSessionManager:
    """Test SessionManager class."""

    def test_auto_platforms_prioritized(self):
        """AUTO platforms should be selected before MANUAL platforms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "state.json")
            auto_p = _make_platform(key="kaggle", limit=9)
            manual_p = _make_platform(key="google_colab", limit=12)
            sm = SessionManager(
                platforms=[manual_p, auto_p],  # manual first in list
                state_path=state_path,
            )
            result = sm.get_next_account()
            assert result is not None
            platform, account = result
            # Kaggle (AUTO) should be picked over Colab (MANUAL)
            assert platform.key == "kaggle"

    def test_no_available_accounts(self):
        """When all accounts are exhausted, should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "state.json")
            p = _make_platform(key="kaggle", limit=9)
            p.accounts[0].status = PlatformStatus.EXHAUSTED
            sm = SessionManager(platforms=[p], state_path=state_path)
            result = sm.get_next_account()
            assert result is None

    def test_start_session(self):
        """Starting a session should set it as current and pending."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "state.json")
            p = _make_platform(key="kaggle", limit=9)
            sm = SessionManager(platforms=[p], state_path=state_path)
            session = sm.start_session()
            assert session is not None
            assert sm.current_session is session
            assert session.is_pending

    def test_confirm_session(self):
        """Confirming should transition to CONFIRMED."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "state.json")
            p = _make_platform(key="kaggle", limit=9)
            sm = SessionManager(platforms=[p], state_path=state_path)
            sm.start_session()
            result = sm.confirm_session()
            assert result is True
            assert sm.current_session.is_confirmed

    def test_stop_session(self):
        """Stopping should end the session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "state.json")
            p = _make_platform(key="kaggle", limit=9)
            sm = SessionManager(platforms=[p], state_path=state_path)
            sm.start_session()
            sm.confirm_session()
            sm.stop()
            assert sm.current_session is None

    def test_state_persistence(self):
        """Runtime state should persist across restarts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "state.json")
            p = _make_platform(key="kaggle", limit=9)
            p.accounts[0].weekly_hours_used = 5.0
            p.accounts[0].sessions_used = 3

            # Create manager and save state
            sm1 = SessionManager(platforms=[p], state_path=state_path)
            sm1.save_runtime_state()

            # Create new manager and load state
            p2 = _make_platform(key="kaggle", limit=9)
            sm2 = SessionManager(platforms=[p2], state_path=state_path)
            assert p2.accounts[0].weekly_hours_used == 5.0
            assert p2.accounts[0].sessions_used == 3

    def test_mark_training_complete(self):
        """Marking training complete should end session and try to rotate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "state.json")
            p1 = _make_platform(key="kaggle", limit=9)
            p2 = _make_platform(key="oracle_cloud", limit=24)
            sm = SessionManager(platforms=[p1, p2], state_path=state_path)
            sm.start_session()
            sm.confirm_session()
            result = sm.mark_training_complete()
            # Should have rotated to next account
            assert result is True

    def test_event_callbacks(self):
        """Event callbacks should fire at the right times."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "state.json")
            p = _make_platform(key="kaggle", limit=9)
            sm = SessionManager(platforms=[p], state_path=state_path)

            confirmed_events = []
            sm.on_session_confirmed = lambda s: confirmed_events.append("confirmed")

            sm.start_session()
            sm.confirm_session()
            assert len(confirmed_events) == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
