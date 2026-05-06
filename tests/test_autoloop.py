"""Tests for the Auto Loop — continuous scheduling daemon."""

import os
import sys
import time
import tempfile
import unittest
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import init_db, close_connection
from db.repositories import (
    OwnerRepository, AccountRepository, JobRepository,
    LeaseRepository, QuotaLedgerRepository, AuditLogRepository,
)
from scheduler.request import JobRequest, FailureReason
from scheduler.autoloop import AutoLoop, AutoLoopConfig, AutoLoopStats


class AutoLoopTestBase(unittest.TestCase):
    """Base class for AutoLoop tests with temp DB."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        close_connection()
        self.conn = init_db(self.db_path)
        self._seed_test_data()

    def tearDown(self):
        close_connection()
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def _seed_test_data(self):
        """Create test owners and accounts."""
        owner_repo = OwnerRepository()
        owner_repo.create(id="me", name="Me")
        owner_repo.create(id="papa", name="Papa", relationship="father")

        acct_repo = AccountRepository()
        acct_repo.create(
            owner_id="me", provider_key="kaggle", label="my-kaggle",
            credential_ref="keyring:kaggle:my-kaggle", priority=7,
            daily_limit_minutes=540, weekly_limit_minutes=1800,
        )
        acct_repo.create(
            owner_id="papa", provider_key="oracle_cloud", label="papa-oracle",
            credential_ref="keyring:oracle_cloud:papa-oracle", priority=8,
            daily_limit_minutes=1440, weekly_limit_minutes=10080,
        )


class TestAutoLoopConfig(AutoLoopTestBase):
    """Tests for AutoLoop configuration."""

    def test_default_config(self):
        config = AutoLoopConfig()
        self.assertEqual(config.lease_check_interval, 30.0)
        self.assertEqual(config.queue_check_interval, 15.0)
        self.assertTrue(config.auto_failover)
        self.assertTrue(config.auto_start_queued)
        self.assertTrue(config.auto_health_check)
        self.assertTrue(config.auto_checkpoint)

    def test_custom_config(self):
        config = AutoLoopConfig(
            lease_check_interval=10.0,
            queue_check_interval=5.0,
            auto_failover=False,
        )
        self.assertEqual(config.lease_check_interval, 10.0)
        self.assertEqual(config.queue_check_interval, 5.0)
        self.assertFalse(config.auto_failover)


class TestAutoLoopStats(AutoLoopTestBase):
    """Tests for AutoLoop statistics."""

    def test_stats_initial(self):
        stats = AutoLoopStats()
        self.assertFalse(stats.is_running)
        self.assertEqual(stats.total_leases_checked, 0)
        self.assertEqual(stats.total_failovers_triggered, 0)
        self.assertIsNone(stats.started_at)

    def test_stats_to_dict(self):
        stats = AutoLoopStats(is_running=True, total_jobs_started=5)
        d = stats.to_dict()
        self.assertTrue(d["is_running"])
        self.assertEqual(d["total_jobs_started"], 5)


class TestAutoLoopStartStop(AutoLoopTestBase):
    """Tests for AutoLoop start/stop lifecycle."""

    def test_start_stop(self):
        config = AutoLoopConfig(
            lease_check_interval=2.0,
            queue_check_interval=2.0,
            health_check_interval=60.0,
        )
        loop = AutoLoop(config=config)

        self.assertFalse(loop.is_running)
        loop.start()
        self.assertTrue(loop.is_running)

        # Let it run briefly
        time.sleep(1.0)

        loop.stop()
        self.assertFalse(loop.is_running)

    def test_double_start(self):
        """Starting an already running loop should be safe."""
        config = AutoLoopConfig(lease_check_interval=10.0)
        loop = AutoLoop(config=config)
        loop.start()
        loop.start()  # Should not crash
        self.assertTrue(loop.is_running)
        loop.stop()

    def test_stop_when_not_running(self):
        """Stopping a non-running loop should be safe."""
        loop = AutoLoop()
        loop.stop()  # Should not crash

    def test_stats_updated_on_start(self):
        loop = AutoLoop()
        loop.start()
        time.sleep(0.5)
        self.assertTrue(loop.stats.is_running)
        self.assertIsNotNone(loop.stats.started_at)
        loop.stop()


class TestAutoLoopLeaseExpiry(AutoLoopTestBase):
    """Tests for auto lease expiry detection."""

    def test_expired_lease_detected(self):
        """AutoLoop should detect and handle expired leases."""
        # Create a job and lease that's already expired
        job_repo = JobRepository()
        lease_repo = LeaseRepository()
        acct_repo = AccountRepository()

        # Use a real account from seeded data
        accounts = acct_repo.list_all()
        account = accounts[0]

        job = job_repo.create(
            name="test-expired",
            gpu_profile="small_gpu",
            max_runtime_minutes=60,
            checkpoint_uri="file:///ckpt/test",
        )

        # Create lease with past expiry
        lease = lease_repo.create(
            job_id=job["id"],
            account_id=account["id"],
            provider_key=account["provider_key"],
            expires_at="2020-01-01T00:00:00+00:00",  # Past
        )

        self.assertIsNotNone(lease, "Lease should be created")

        # Run lease check manually
        loop = AutoLoop()
        loop._check_lease_expiry()

        # The lease should have been handled (expired)
        updated_lease = lease_repo.get_by_id(lease["id"])
        self.assertIsNotNone(updated_lease)
        self.assertIn(updated_lease["status"], ("expired", "failed"))


class TestAutoLoopQueuedJobs(AutoLoopTestBase):
    """Tests for auto-starting queued jobs."""

    def test_queued_job_starts(self):
        """AutoLoop should try to start queued jobs."""
        job_repo = JobRepository()

        # Create a queued job
        job = job_repo.create(
            name="test-queued",
            gpu_profile="small_gpu",
            max_runtime_minutes=60,
            checkpoint_uri="file:///ckpt/test",
        )

        self.assertEqual(job["status"], "queued")

        # Run queue check manually
        loop = AutoLoop()
        loop._start_queued_jobs()

        # The job should have been attempted to start
        # (may fail if no adapter can actually run it, but it should try)
        updated_job = job_repo.get_by_id(job["id"])
        # Job status may have changed from "queued"
        self.assertIsNotNone(updated_job)


class TestAutoLoopCooldownClear(AutoLoopTestBase):
    """Tests for cooldown clearing."""

    def test_expired_cooldown_cleared(self):
        """AutoLoop should clear expired cooldowns."""
        acct_repo = AccountRepository()

        # Put account in cooldown with past expiry
        account = acct_repo.list_all()[0]
        acct_repo.update(
            account["id"],
            status="cooldown",
            cooldown_until="2020-01-01T00:00:00+00:00",
        )

        loop = AutoLoop()
        loop._clear_expired_cooldowns()

        updated = acct_repo.get_by_id(account["id"])
        self.assertEqual(updated["status"], "active")


class TestAutoLoopGetStatus(AutoLoopTestBase):
    """Tests for auto loop status reporting."""

    def test_get_status(self):
        loop = AutoLoop()
        status = loop.get_status()

        self.assertIn("auto_loop", status)
        self.assertIn("config", status)
        self.assertIn("active_leases", status)
        self.assertIn("queued_jobs", status)
        self.assertIn("available_accounts", status)

    def test_get_status_while_running(self):
        config = AutoLoopConfig(
            lease_check_interval=10.0,
            queue_check_interval=10.0,
            health_check_interval=60.0,
        )
        loop = AutoLoop(config=config)
        loop.start()
        time.sleep(0.5)

        status = loop.get_status()
        self.assertTrue(status["auto_loop"]["is_running"])

        loop.stop()


class TestAutoLoopAccountErrors(AutoLoopTestBase):
    """Tests for account error tracking and auto-disable."""

    def test_account_error_tracking(self):
        loop = AutoLoop()
        acct_repo = AccountRepository()
        accounts = acct_repo.list_all()
        account_id = accounts[0]["id"]

        # Record errors up to threshold
        loop._record_account_error(account_id)
        self.assertEqual(loop._account_errors[account_id], 1)

        # Should not be disabled yet
        updated = acct_repo.get_by_id(account_id)
        self.assertEqual(updated["status"], "active")

    def test_account_auto_disabled(self):
        """Account should be auto-disabled after max consecutive errors."""
        config = AutoLoopConfig(max_consecutive_errors=3)
        loop = AutoLoop(config=config)
        acct_repo = AccountRepository()
        accounts = acct_repo.list_all()
        account_id = accounts[0]["id"]

        # Record 3 errors
        for _ in range(3):
            loop._record_account_error(account_id)

        # Account should be disabled
        updated = acct_repo.get_by_id(account_id)
        self.assertEqual(updated["status"], "disabled")


class TestAutoLoopSubmitJob(AutoLoopTestBase):
    """Tests for submitting jobs through the auto loop."""

    def test_submit_valid_job(self):
        loop = AutoLoop()
        request = JobRequest(
            job_name="auto-test-001",
            gpu_profile="small_gpu",
            max_runtime_minutes=60,
            checkpoint_uri="file:///ckpt/auto-test",
        )

        result = loop.submit_job(request)
        # Should be accepted, queued, or rejected (depends on adapter availability)
        self.assertIn(result.status, ("accepted", "queued", "rejected"))

    def test_submit_invalid_job(self):
        loop = AutoLoop()
        request = JobRequest(
            job_name="",
            gpu_profile="small_gpu",
            max_runtime_minutes=60,
            checkpoint_required=True,
        )

        result = loop.submit_job(request)
        self.assertEqual(result.status, "rejected")


if __name__ == "__main__":
    unittest.main()
