"""Tests for the scheduler — account selection and scoring."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import init_db, close_connection
from db.repositories import (
    OwnerRepository, AccountRepository, JobRepository,
    LeaseRepository, QuotaLedgerRepository, AuditLogRepository,
)
from scheduler.request import JobRequest, FailureReason, GPU_PROFILES
from scheduler.scoring import calculate_score
from scheduler.selector import AccountSelector


class TestSchedulerBase(unittest.TestCase):
    """Base class for scheduler tests with temp DB."""

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
        owner_repo.create(id="mama", name="Mama", relationship="mother")

        acct_repo = AccountRepository()
        # Kaggle account (AUTO, Class B)
        acct_repo.create(
            owner_id="me", provider_key="kaggle", label="my-kaggle",
            credential_ref="keyring:kaggle:my-kaggle", priority=7,
            daily_limit_minutes=540, weekly_limit_minutes=1800,
        )
        # Oracle Cloud account (AUTO, Class A)
        acct_repo.create(
            owner_id="papa", provider_key="oracle_cloud", label="papa-oracle",
            credential_ref="keyring:oracle_cloud:papa-oracle", priority=8,
            daily_limit_minutes=1440, weekly_limit_minutes=10080,
        )
        # GCP account (AUTO, Class A)
        acct_repo.create(
            owner_id="mama", provider_key="gcp", label="mama-gcp",
            credential_ref="keyring:gcp:mama-gcp", priority=6,
            daily_limit_minutes=480, weekly_limit_minutes=3360,
        )
        # Google Colab (MANUAL, Class B)
        acct_repo.create(
            owner_id="me", provider_key="google_colab", label="my-colab",
            credential_ref="keyring:google_colab:my-colab", priority=5,
            daily_limit_minutes=720, weekly_limit_minutes=5040,
        )


class TestJobRequestValidation(TestSchedulerBase):
    """Tests for JobRequest validation."""

    def test_valid_request(self):
        req = JobRequest(
            job_name="train-lora-001",
            gpu_profile="small_gpu",
            max_runtime_minutes=180,
            checkpoint_uri="file:///ckpt/job1",
        )
        errors = req.validate()
        self.assertEqual(len(errors), 0)

    def test_missing_name(self):
        req = JobRequest(job_name="", gpu_profile="small_gpu",
                         max_runtime_minutes=60, checkpoint_uri="file:///ckpt")
        errors = req.validate()
        self.assertTrue(any("job_name" in e for e in errors))

    def test_invalid_profile(self):
        req = JobRequest(job_name="test", gpu_profile="mega_gpu",
                         max_runtime_minutes=60, checkpoint_uri="file:///ckpt")
        errors = req.validate()
        self.assertTrue(any("gpu_profile" in e for e in errors))

    def test_missing_checkpoint_uri(self):
        req = JobRequest(job_name="test", gpu_profile="small_gpu",
                         max_runtime_minutes=60, checkpoint_required=True)
        errors = req.validate()
        self.assertTrue(any("checkpoint_uri" in e for e in errors))

    def test_excessive_runtime(self):
        req = JobRequest(job_name="test", gpu_profile="small_gpu",
                         max_runtime_minutes=2000, checkpoint_uri="file:///ckpt")
        errors = req.validate()
        self.assertTrue(any("1440" in e for e in errors))


class TestAccountScoring(TestSchedulerBase):
    """Tests for the scoring algorithm."""

    def test_higher_priority_scores_more(self):
        high_acct = {"id": "a1", "priority": 9, "last_error_at": None}
        low_acct = {"id": "a2", "priority": 3, "last_error_at": None}
        job = {"priority": "normal"}

        high_score = calculate_score(high_acct, job, 120, 600, "ok",
                                     {"supports_auto": True, "provider_class": "A"})
        low_score = calculate_score(low_acct, job, 120, 600, "ok",
                                    {"supports_auto": True, "provider_class": "A"})
        self.assertGreater(high_score, low_score)

    def test_auto_provider_bonus(self):
        auto_acct = {"id": "a1", "priority": 5, "last_error_at": None}
        manual_acct = {"id": "a2", "priority": 5, "last_error_at": None}
        job = {"priority": "normal"}

        auto_score = calculate_score(auto_acct, job, 120, 600, "ok",
                                     {"supports_auto": True, "provider_class": "A"})
        manual_score = calculate_score(manual_acct, job, 120, 600, "ok",
                                       {"supports_auto": False, "provider_class": "B"})
        self.assertGreater(auto_score, manual_score)

    def test_error_penalty(self):
        clean_acct = {"id": "a1", "priority": 5, "last_error_at": None}
        error_acct = {"id": "a2", "priority": 5, "last_error_at": "2026-01-01T00:00:00"}
        job = {"priority": "normal"}

        clean_score = calculate_score(clean_acct, job, 120, 600, "ok",
                                      {"supports_auto": True, "provider_class": "A"})
        error_score = calculate_score(error_acct, job, 120, 600, "ok",
                                      {"supports_auto": True, "provider_class": "A"})
        self.assertGreater(clean_score, error_score)

    def test_down_health_penalty(self):
        ok_acct = {"id": "a1", "priority": 5, "last_error_at": None}
        down_acct = {"id": "a2", "priority": 5, "last_error_at": None}
        job = {"priority": "normal"}

        ok_score = calculate_score(ok_acct, job, 120, 600, "ok",
                                   {"supports_auto": True, "provider_class": "A"})
        down_score = calculate_score(down_acct, job, 120, 600, "down",
                                     {"supports_auto": True, "provider_class": "A"})
        self.assertGreater(ok_score, down_score)


class TestAccountSelector(TestSchedulerBase):
    """Tests for the AccountSelector."""

    def test_select_account(self):
        selector = AccountSelector()
        request = JobRequest(
            job_name="test-job",
            gpu_profile="small_gpu",
            max_runtime_minutes=60,
            checkpoint_uri="file:///ckpt/test",
        )
        account, failure = selector.select(request)
        self.assertIsNotNone(account)
        self.assertIsNone(failure)

    def test_select_respects_provider_filter(self):
        selector = AccountSelector()
        request = JobRequest(
            job_name="test-job",
            gpu_profile="small_gpu",
            max_runtime_minutes=60,
            checkpoint_uri="file:///ckpt/test",
            allow_providers=["kaggle"],
        )
        account, failure = selector.select(request)
        if account:
            self.assertEqual(account["provider_key"], "kaggle")

    def test_select_respects_deny_filter(self):
        selector = AccountSelector()
        request = JobRequest(
            job_name="test-job",
            gpu_profile="small_gpu",
            max_runtime_minutes=60,
            checkpoint_uri="file:///ckpt/test",
            deny_providers=["google_colab"],
        )
        account, failure = selector.select(request)
        if account:
            self.assertNotEqual(account["provider_key"], "google_colab")

    def test_no_accounts_when_all_disabled(self):
        acct_repo = AccountRepository()
        for acct in acct_repo.list_all():
            acct_repo.disable(acct["id"])

        selector = AccountSelector()
        request = JobRequest(
            job_name="test-job",
            gpu_profile="small_gpu",
            max_runtime_minutes=60,
            checkpoint_uri="file:///ckpt/test",
        )
        account, failure = selector.select(request)
        self.assertIsNone(account)
        self.assertIsNotNone(failure)

    def test_select_skips_busy_account(self):
        # Make all accounts busy by creating leases
        acct_repo = AccountRepository()
        job_repo = JobRepository()
        lease_repo = LeaseRepository()

        for acct in acct_repo.list_all():
            job = job_repo.create(name="blocker", gpu_profile="small_gpu",
                                  max_runtime_minutes=60, checkpoint_uri="file:///ckpt")
            lease_repo.create(job_id=job["id"], account_id=acct["id"],
                              provider_key=acct["provider_key"])

        selector = AccountSelector()
        request = JobRequest(
            job_name="test-job",
            gpu_profile="small_gpu",
            max_runtime_minutes=60,
            checkpoint_uri="file:///ckpt/test",
        )
        account, failure = selector.select(request)
        self.assertIsNone(account)
        self.assertEqual(failure, FailureReason.ALL_ACCOUNTS_BUSY)


class TestGPUProfiles(unittest.TestCase):
    """Tests for GPU profile definitions."""

    def test_all_profiles_defined(self):
        expected = {"cpu_only", "small_gpu", "medium_gpu", "high_vram_gpu", "long_running"}
        self.assertEqual(set(GPU_PROFILES.keys()), expected)

    def test_profile_has_description(self):
        for key, val in GPU_PROFILES.items():
            self.assertIn("description", val)
            self.assertIn("gpu_required", val)


if __name__ == "__main__":
    unittest.main()
