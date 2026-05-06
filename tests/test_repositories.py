"""Tests for the database repositories.

Tests cover:
  - Account CRUD
  - Job lifecycle
  - Lease lifecycle
  - Quota ledger tracking
  - Audit logging
  - Owner management
"""

import os
import sys
import json
import tempfile
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import init_db, close_connection, get_connection
from db.repositories import (
    OwnerRepository, AccountRepository, ProviderRepository,
    JobRepository, LeaseRepository, QuotaLedgerRepository,
    AuditLogRepository, HealthRepository, SecretMetadataRepository,
)


class TestDatabase(unittest.TestCase):
    """Base class for database tests — uses a temp DB for each test."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        close_connection()  # Reset any existing connection
        self.conn = init_db(self.db_path)

    def tearDown(self):
        close_connection()
        os.close(self.db_fd)
        os.unlink(self.db_path)


class TestOwnerRepository(TestDatabase):
    """Tests for OwnerRepository."""

    def test_create_owner(self):
        repo = OwnerRepository()
        owner = repo.create(id="me", name="Me", relationship="self")
        self.assertIsNotNone(owner)
        self.assertEqual(owner["id"], "me")
        self.assertEqual(owner["name"], "Me")

    def test_list_owners(self):
        repo = OwnerRepository()
        repo.create(id="me", name="Me")
        repo.create(id="papa", name="Papa", relationship="father")
        owners = repo.list_all()
        self.assertEqual(len(owners), 2)

    def test_update_owner(self):
        repo = OwnerRepository()
        repo.create(id="me", name="Me")
        updated = repo.update("me", name="Myself")
        self.assertEqual(updated["name"], "Myself")

    def test_delete_owner(self):
        repo = OwnerRepository()
        repo.create(id="me", name="Me")
        self.assertTrue(repo.delete("me"))
        self.assertIsNone(repo.get_by_id("me"))


class TestProviderRepository(TestDatabase):
    """Tests for ProviderRepository."""

    def test_list_providers(self):
        repo = ProviderRepository()
        providers = repo.list_all()
        self.assertEqual(len(providers), 12)  # 12 providers seeded

    def test_get_provider(self):
        repo = ProviderRepository()
        kaggle = repo.get_by_key("kaggle")
        self.assertIsNotNone(kaggle)
        self.assertEqual(kaggle["display_name"], "Kaggle Notebooks")

    def test_enabled_only(self):
        repo = ProviderRepository()
        enabled = repo.list_all(enabled_only=True)
        self.assertEqual(len(enabled), 12)  # All enabled by default

    def test_list_by_class(self):
        repo = ProviderRepository()
        class_a = repo.list_by_class("A")
        class_b = repo.list_by_class("B")
        class_c = repo.list_by_class("C")
        self.assertTrue(len(class_a) > 0)
        self.assertTrue(len(class_b) > 0)
        self.assertTrue(len(class_c) > 0)
        self.assertEqual(len(class_a) + len(class_b) + len(class_c), 12)


class TestAccountRepository(TestDatabase):
    """Tests for AccountRepository."""

    def _create_test_owner(self):
        owner_repo = OwnerRepository()
        owner_repo.create(id="me", name="Me")
        return "me"

    def test_create_account(self):
        owner_id = self._create_test_owner()
        repo = AccountRepository()
        account = repo.create(
            owner_id=owner_id,
            provider_key="kaggle",
            label="my-kaggle-1",
            credential_ref="keyring:kaggle:my-kaggle-1",
        )
        self.assertIsNotNone(account)
        self.assertEqual(account["provider_key"], "kaggle")
        self.assertEqual(account["status"], "active")

    def test_list_accounts(self):
        owner_id = self._create_test_owner()
        repo = AccountRepository()
        repo.create(owner_id=owner_id, provider_key="kaggle", label="k1", credential_ref="keyring:k:k1")
        repo.create(owner_id=owner_id, provider_key="gcp", label="g1", credential_ref="keyring:g:g1")
        accounts = repo.list_all()
        self.assertEqual(len(accounts), 2)

    def test_set_cooldown(self):
        owner_id = self._create_test_owner()
        repo = AccountRepository()
        account = repo.create(owner_id=owner_id, provider_key="kaggle", label="k1", credential_ref="keyring:k:k1")
        repo.set_cooldown(account["id"], 30)
        updated = repo.get_by_id(account["id"])
        self.assertEqual(updated["status"], "cooldown")
        self.assertIsNotNone(updated["cooldown_until"])

    def test_disable_account(self):
        owner_id = self._create_test_owner()
        repo = AccountRepository()
        account = repo.create(owner_id=owner_id, provider_key="kaggle", label="k1", credential_ref="keyring:k:k1")
        repo.disable(account["id"])
        updated = repo.get_by_id(account["id"])
        self.assertEqual(updated["status"], "disabled")


class TestJobRepository(TestDatabase):
    """Tests for JobRepository."""

    def test_create_job(self):
        repo = JobRepository()
        job = repo.create(
            name="train-lora-001",
            gpu_profile="small_gpu",
            max_runtime_minutes=180,
            checkpoint_uri="file:///workspace/checkpoints/job1",
        )
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "queued")

    def test_job_lifecycle(self):
        repo = JobRepository()
        job = repo.create(
            name="test-job",
            gpu_profile="medium_gpu",
            max_runtime_minutes=60,
            checkpoint_uri="file:///ckpt/job",
        )
        job_id = job["id"]

        # Start
        repo.mark_started(job_id)
        job = repo.get_by_id(job_id)
        self.assertEqual(job["status"], "running")

        # Complete
        repo.mark_completed(job_id)
        job = repo.get_by_id(job_id)
        self.assertEqual(job["status"], "completed")

    def test_job_failure(self):
        repo = JobRepository()
        job = repo.create(
            name="fail-job",
            gpu_profile="small_gpu",
            max_runtime_minutes=60,
            checkpoint_uri="file:///ckpt/fail",
        )
        repo.mark_failed(job["id"], "GPU out of memory")
        job = repo.get_by_id(job["id"])
        self.assertEqual(job["status"], "failed")
        self.assertIn("GPU out of memory", job["failure_reason"])


class TestLeaseRepository(TestDatabase):
    """Tests for LeaseRepository."""

    def _create_test_account(self):
        owner_repo = OwnerRepository()
        owner_repo.create(id="me", name="Me")
        acct_repo = AccountRepository()
        account = acct_repo.create(
            owner_id="me", provider_key="kaggle", label="test-kaggle",
            credential_ref="keyring:kaggle:test",
        )
        return account["id"]

    def test_create_lease(self):
        acct_id = self._create_test_account()
        job_repo = JobRepository()
        job = job_repo.create(name="test", gpu_profile="small_gpu",
                              max_runtime_minutes=60, checkpoint_uri="file:///ckpt")
        lease_repo = LeaseRepository()
        lease = lease_repo.create(
            job_id=job["id"], account_id=acct_id, provider_key="kaggle",
        )
        self.assertIsNotNone(lease)
        self.assertEqual(lease["status"], "pending")

    def test_is_account_busy(self):
        acct_id = self._create_test_account()
        job_repo = JobRepository()
        job = job_repo.create(name="test", gpu_profile="small_gpu",
                              max_runtime_minutes=60, checkpoint_uri="file:///ckpt")
        lease_repo = LeaseRepository()
        self.assertFalse(lease_repo.is_account_busy(acct_id))

        lease_repo.create(job_id=job["id"], account_id=acct_id, provider_key="kaggle")
        self.assertTrue(lease_repo.is_account_busy(acct_id))

    def test_lease_lifecycle(self):
        acct_id = self._create_test_account()
        job_repo = JobRepository()
        job = job_repo.create(name="test", gpu_profile="small_gpu",
                              max_runtime_minutes=60, checkpoint_uri="file:///ckpt")
        lease_repo = LeaseRepository()
        lease = lease_repo.create(job_id=job["id"], account_id=acct_id, provider_key="kaggle")

        # Mark running
        lease_repo.mark_running(lease["id"])
        lease = lease_repo.get_by_id(lease["id"])
        self.assertEqual(lease["status"], "running")

        # Mark completed
        lease_repo.mark_completed(lease["id"], runtime_minutes=55)
        lease = lease_repo.get_by_id(lease["id"])
        self.assertEqual(lease["status"], "completed")
        self.assertEqual(lease["runtime_minutes"], 55)


class TestQuotaLedgerRepository(TestDatabase):
    """Tests for QuotaLedgerRepository."""

    def _create_test_account(self):
        """Create a test account to satisfy foreign key constraints."""
        owner_repo = OwnerRepository()
        owner_repo.create(id="quota-owner", name="Quota Owner")
        acct_repo = AccountRepository()
        account = acct_repo.create(
            owner_id="quota-owner",
            provider_key="kaggle",
            label="quota-test-acct",
            credential_ref="keyring:k:qt",
        )
        return account["id"]

    def test_record_usage(self):
        acct_id = self._create_test_account()
        repo = QuotaLedgerRepository()
        entry = repo.record_usage(
            account_id=acct_id,
            provider_key="kaggle",
            used_minutes=60,
        )
        self.assertIsNotNone(entry)
        self.assertEqual(entry["used_minutes"], 60)

    def test_daily_usage(self):
        acct_id = self._create_test_account()
        repo = QuotaLedgerRepository()
        repo.record_usage(acct_id, "kaggle", 60)
        repo.record_usage(acct_id, "kaggle", 30)
        daily = repo.get_daily_usage(acct_id)
        self.assertEqual(daily, 90)

    def test_remaining_quota(self):
        acct_id = self._create_test_account()
        repo = QuotaLedgerRepository()
        repo.record_usage(acct_id, "kaggle", 60)
        remaining = repo.remaining_daily(acct_id, 120)
        self.assertEqual(remaining, 60)


class TestAuditLogRepository(TestDatabase):
    """Tests for AuditLogRepository."""

    def test_log_event(self):
        repo = AuditLogRepository()
        repo.log(
            action="add_account",
            entity_type="account",
            entity_id="acct_123",
            actor="user",
            message="Account added",
        )
        logs = repo.list_recent(limit=10)
        self.assertTrue(len(logs) > 0)
        self.assertEqual(logs[0]["action"], "add_account")

    def test_list_by_entity(self):
        repo = AuditLogRepository()
        repo.log(action="create", entity_type="job", entity_id="job_1")
        repo.log(action="create", entity_type="job", entity_id="job_2")
        logs = repo.list_by_entity("job", "job_1")
        self.assertEqual(len(logs), 1)


class TestHealthRepository(TestDatabase):
    """Tests for HealthRepository."""

    def test_record_health(self):
        # Need an account first
        owner_repo = OwnerRepository()
        owner_repo.create(id="me", name="Me")
        acct_repo = AccountRepository()
        account = acct_repo.create(
            owner_id="me", provider_key="kaggle", label="test",
            credential_ref="keyring:k:k",
        )

        repo = HealthRepository()
        repo.record(account_id=account["id"], provider_key="kaggle",
                     status="ok", message="All good")

        latest = repo.get_latest(account["id"])
        self.assertIsNotNone(latest)
        self.assertEqual(latest["status"], "ok")


if __name__ == "__main__":
    unittest.main()
