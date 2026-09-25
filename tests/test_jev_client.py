"""
Tests for core.jev_client — all HTTP calls are mocked, no network access.
Run with: python -m unittest tests.test_jev_client -v
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import jev_client  # noqa: E402


class _FakeKeyMixin:
    """
    Forces JEV_AI_API_KEY to a known fake value for the duration of each
    test, regardless of what other test modules imported earlier in the
    same run have already loaded into the real process environment (e.g.
    via config.py's load_dotenv() picking up the real .env file).
    """

    def setUp(self):
        self._env_patcher = patch.dict(os.environ, {"JEV_AI_API_KEY": "test-key"})
        self._env_patcher.start()
        self.addCleanup(self._env_patcher.stop)


def _mock_response(status_code=200, json_data=None, headers=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.headers = headers or {}
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("no json body")
    return resp


class AskTests(_FakeKeyMixin, unittest.TestCase):
    @patch("core.jev_client.requests.post")
    def test_success(self, mock_post):
        mock_post.return_value = _mock_response(
            200,
            json_data={
                "model": "jev-1.13.0",
                "answers": {"urgent": {"noul": 0.82}},
                "usage": {"input_tokens": 42, "output_tokens": 0},
            },
            headers={"X-Jev-Run-Id": "run_123"},
        )

        result = jev_client.ask(
            state="My payment failed. Please help.",
            questions={"urgent": {"type": "noul", "instructions": "Urgent?"}},
        )

        self.assertEqual(result.model, "jev-1.13.0")
        self.assertEqual(result.answers["urgent"]["noul"], 0.82)
        self.assertEqual(result.usage.input_tokens, 42)
        self.assertEqual(result.run_id, "run_123")

        sent_body = mock_post.call_args.kwargs["json"]
        self.assertEqual(sent_body["model"], "jev-latest")
        sent_headers = mock_post.call_args.kwargs["headers"]
        self.assertEqual(sent_headers["Authorization"], "Bearer test-key")

    @patch("core.jev_client.requests.post")
    def test_401_raises_auth_error(self, mock_post):
        mock_post.return_value = _mock_response(401, json_data={"error": "invalid key"})
        with self.assertRaises(jev_client.JevAuthError):
            jev_client.ask("state", {"q": {"type": "noul", "instructions": "?"}})

    @patch("core.jev_client.requests.post")
    def test_402_raises_billing_error(self, mock_post):
        mock_post.return_value = _mock_response(402, json_data={"error": "no balance"})
        with self.assertRaises(jev_client.JevBillingError):
            jev_client.ask("state", {"q": {"type": "noul", "instructions": "?"}})

    @patch("core.jev_client.requests.post")
    def test_422_raises_validation_error(self, mock_post):
        mock_post.return_value = _mock_response(422, json_data={"error": "bad question type"})
        with self.assertRaises(jev_client.JevValidationError):
            jev_client.ask("state", {"q": {"type": "noul", "instructions": "?"}})

    @patch("core.jev_client.requests.post")
    def test_429_carries_retry_after(self, mock_post):
        mock_post.return_value = _mock_response(
            429, json_data={"error": "rate limited"}, headers={"Retry-After": "12"}
        )
        with self.assertRaises(jev_client.JevRateLimitError) as ctx:
            jev_client.ask("state", {"q": {"type": "noul", "instructions": "?"}})
        self.assertEqual(ctx.exception.retry_after, 12.0)

    @patch("core.jev_client.requests.post")
    def test_502_raises_service_error_with_status(self, mock_post):
        mock_post.return_value = _mock_response(502, text="bad gateway")
        with self.assertRaises(jev_client.JevServiceError) as ctx:
            jev_client.ask("state", {"q": {"type": "noul", "instructions": "?"}})
        self.assertEqual(ctx.exception.status_code, 502)

    @patch("core.jev_client.requests.post")
    def test_504_raises_service_error(self, mock_post):
        mock_post.return_value = _mock_response(504, text="gateway timeout")
        with self.assertRaises(jev_client.JevServiceError):
            jev_client.ask("state", {"q": {"type": "noul", "instructions": "?"}})

    def test_missing_api_key_raises_auth_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(jev_client.JevAuthError):
                jev_client.ask("state", {"q": {"type": "noul", "instructions": "?"}})


class ListModelsTests(_FakeKeyMixin, unittest.TestCase):
    @patch("core.jev_client.requests.get")
    def test_filters_to_connected_models(self, mock_get):
        mock_get.return_value = _mock_response(
            200,
            json_data={
                "models": [
                    {"name": "jev-latest", "description": "default", "connected": True},
                    {"name": "laya-english", "description": "laya", "connected": False},
                    {"name": "jev-1.13.0", "description": "pinned"},  # no flag -> assumed connected
                ]
            },
        )
        models = jev_client.list_models()
        names = {m.name for m in models}
        self.assertEqual(names, {"jev-latest", "jev-1.13.0"})

    @patch("core.jev_client.requests.get")
    def test_401_raises_auth_error(self, mock_get):
        mock_get.return_value = _mock_response(401, json_data={"error": "invalid key"})
        with self.assertRaises(jev_client.JevAuthError):
            jev_client.list_models()


if __name__ == "__main__":
    unittest.main()
