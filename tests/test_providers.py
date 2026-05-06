"""Tests for the provider registry and adapter interface."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from providers.base import ProviderAdapter, ProviderResult
from providers.registry import get_adapter, list_adapters, list_by_class, list_auto_adapters


class TestProviderRegistry(unittest.TestCase):
    """Tests for the provider adapter registry."""

    def test_all_12_providers_registered(self):
        adapters = list_adapters()
        self.assertEqual(len(adapters), 12)

    def test_expected_provider_keys(self):
        expected = {
            "google_colab", "kaggle", "huggingface", "paperspace",
            "sagemaker", "lightning_ai", "codesphere", "oracle_cloud",
            "gcp", "intel_devcloud", "deepnote", "nvidia_vgpu",
        }
        actual = set(list_adapters().keys())
        self.assertEqual(actual, expected)

    def test_get_adapter(self):
        adapter = get_adapter("kaggle")
        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.provider_key, "kaggle")

    def test_get_nonexistent_adapter(self):
        adapter = get_adapter("nonexistent_provider")
        self.assertIsNone(adapter)

    def test_class_a_providers(self):
        class_a = list_by_class("A")
        keys = {a.provider_key for a in class_a}
        self.assertTrue("oracle_cloud" in keys)
        self.assertTrue("gcp" in keys)
        self.assertTrue("paperspace" in keys)
        self.assertTrue("lightning_ai" in keys)
        self.assertTrue("codesphere" in keys)

    def test_class_b_providers(self):
        class_b = list_by_class("B")
        keys = {a.provider_key for a in class_b}
        self.assertTrue("google_colab" in keys)
        self.assertTrue("kaggle" in keys)
        self.assertTrue("sagemaker" in keys)
        self.assertTrue("deepnote" in keys)

    def test_class_c_providers(self):
        class_c = list_by_class("C")
        keys = {a.provider_key for a in class_c}
        self.assertTrue("huggingface" in keys)
        self.assertTrue("intel_devcloud" in keys)
        self.assertTrue("nvidia_vgpu" in keys)

    def test_auto_adapters(self):
        auto = list_auto_adapters()
        keys = {a.provider_key for a in auto}
        # At minimum, oracle_cloud, gcp, and kaggle should be auto
        self.assertIn("oracle_cloud", keys)
        self.assertIn("gcp", keys)
        self.assertIn("kaggle", keys)


class TestProviderResult(unittest.TestCase):
    """Tests for ProviderResult dataclass."""

    def test_default_values(self):
        result = ProviderResult()
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.message, "")
        self.assertFalse(result.manual)
        self.assertEqual(result.data, {})

    def test_to_dict(self):
        result = ProviderResult(ok=True, status="running", message="OK")
        d = result.to_dict()
        self.assertTrue(d["ok"])
        self.assertEqual(d["status"], "running")

    def test_manual_required_result(self):
        result = ProviderResult(
            ok=True,
            status="manual_required",
            manual=True,
            message="This provider requires manual runtime start in MVP.",
        )
        self.assertTrue(result.manual)
        self.assertEqual(result.status, "manual_required")


class TestAdapterInterface(unittest.TestCase):
    """Tests that all adapters implement the required interface."""

    def test_all_adapters_have_required_attributes(self):
        required_attrs = [
            "provider_key", "display_name", "provider_class",
            "supports_auto", "supported_profiles",
        ]
        for key, adapter in list_adapters().items():
            for attr in required_attrs:
                self.assertTrue(
                    hasattr(adapter, attr),
                    f"Adapter {key} missing attribute {attr}"
                )

    def test_all_adapters_implement_required_methods(self):
        required_methods = [
            "validate_credentials", "health_check", "estimate_capacity",
            "start_job", "stop_job", "fetch_logs", "sync_checkpoint",
        ]
        for key, adapter in list_adapters().items():
            for method in required_methods:
                self.assertTrue(
                    hasattr(adapter, method) and callable(getattr(adapter, method)),
                    f"Adapter {key} missing method {method}"
                )

    def test_supports_profile(self):
        adapter = get_adapter("oracle_cloud")
        if adapter:
            self.assertTrue(adapter.supports_profile("long_running"))
            self.assertTrue(adapter.supports_profile("small_gpu"))

    def test_get_label(self):
        adapter = get_adapter("kaggle")
        if adapter:
            label = adapter.get_label()
            self.assertIn("kaggle", label.lower())
            # Kaggle is auto, should show AUTO
            self.assertIn("AUTO", label)


class TestCheckpointRequired(unittest.TestCase):
    """Tests that checkpoint_required is enforced."""

    def test_checkpoint_required_in_request(self):
        from scheduler.request import JobRequest
        req = JobRequest(
            job_name="test",
            gpu_profile="small_gpu",
            max_runtime_minutes=60,
            checkpoint_required=True,
            checkpoint_uri="",  # Missing!
        )
        errors = req.validate()
        self.assertTrue(any("checkpoint_uri" in e for e in errors))

    def test_checkpoint_not_required(self):
        from scheduler.request import JobRequest
        req = JobRequest(
            job_name="test",
            gpu_profile="small_gpu",
            max_runtime_minutes=60,
            checkpoint_required=False,
            checkpoint_uri="",
        )
        errors = req.validate()
        # Should NOT complain about missing checkpoint_uri
        self.assertFalse(any("checkpoint_uri" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
