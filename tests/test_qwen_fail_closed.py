"""Regression tests for fail-closed Qwen validation behavior."""
from unittest.mock import Mock, patch

import pytest

from invisible_core.critics.qwen_critic import QwenCritic


def _api_critic():
    critic = QwenCritic.__new__(QwenCritic)
    critic.lm_studio_url = "http://localhost:1234/v1/chat/completions"
    critic.use_api = True
    critic.model = None
    critic.processor = None
    critic.loaded = True
    return critic


def _api_response(text: str):
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": text}}]}
    return response


def test_critique_fails_closed_when_validator_cannot_load():
    critic = QwenCritic.__new__(QwenCritic)
    critic.loaded = False
    critic.use_api = False
    critic.load = lambda: None
    passed, reason, score = critic.critique("unused.png")
    assert passed is False
    assert reason.startswith("Validation error:")
    assert score == 0.0


def test_pairwise_fails_closed_when_validator_cannot_load():
    critic = QwenCritic.__new__(QwenCritic)
    critic.loaded = False
    critic.use_api = False
    critic.load = lambda: None
    passed, reason, score = critic.critique_pairwise("unused-a.png", "unused-b.png")
    assert passed is False
    assert reason.startswith("Validation error:")
    assert score == 0.0


def test_critique_fails_closed_on_api_exception(tmp_path):
    image_path = tmp_path / "probe.png"
    image_path.write_bytes(b"placeholder bytes")
    critic = _api_critic()
    with patch("invisible_core.critics.qwen_critic.requests.post", side_effect=TimeoutError()):
        passed, reason, score = critic.critique(str(image_path), target_name="Target")
    assert passed is False
    assert reason.startswith("Validation error:")
    assert score == 0.0


def test_critique_fails_closed_on_malformed_api_response(tmp_path):
    image_path = tmp_path / "probe.png"
    image_path.write_bytes(b"placeholder bytes")
    critic = _api_critic()
    with patch(
        "invisible_core.critics.qwen_critic.requests.post",
        return_value=_api_response("I cannot determine whether this is protected."),
    ):
        passed, reason, score = critic.critique(str(image_path), target_name="Target")
    assert passed is False
    assert reason.startswith("Validation error:")
    assert score == 0.0


def test_pairwise_fails_closed_on_uncertain_answer(tmp_path):
    reference = tmp_path / "reference.png"
    probe = tmp_path / "probe.png"
    reference.write_bytes(b"reference bytes")
    probe.write_bytes(b"probe bytes")
    critic = _api_critic()
    with patch(
        "invisible_core.critics.qwen_critic.requests.post",
        return_value=_api_response("Match: Uncertain\nConfidence: 95%\nManipulation: Unknown"),
    ):
        passed, reason, score = critic.critique_pairwise(str(reference), str(probe))
    assert passed is False
    assert reason.startswith("Validation error:")
    assert score == pytest.approx(0.95)


def test_pairwise_passes_only_confident_no_match(tmp_path):
    reference = tmp_path / "reference.png"
    probe = tmp_path / "probe.png"
    reference.write_bytes(b"reference bytes")
    probe.write_bytes(b"probe bytes")
    critic = _api_critic()
    with patch(
        "invisible_core.critics.qwen_critic.requests.post",
        return_value=_api_response("Match: No\nConfidence: 91%\nManipulation: Detected"),
    ):
        passed, reason, score = critic.critique_pairwise(str(reference), str(probe))
    assert passed is True
    assert "No match" in reason
    assert score == pytest.approx(0.91)


def test_pairwise_low_confidence_no_match_is_inconclusive(tmp_path):
    reference = tmp_path / "reference.png"
    probe = tmp_path / "probe.png"
    reference.write_bytes(b"reference bytes")
    probe.write_bytes(b"probe bytes")
    critic = _api_critic()
    with patch(
        "invisible_core.critics.qwen_critic.requests.post",
        return_value=_api_response("Match: No\nConfidence: 20%\nManipulation: Unknown"),
    ):
        passed, reason, score = critic.critique_pairwise(str(reference), str(probe))
    assert passed is False
    assert reason.startswith("Validation error:")
    assert score == pytest.approx(0.20)
