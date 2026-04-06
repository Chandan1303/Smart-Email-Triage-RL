import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from models import Email
from environment import EmailEnv

LABELS     = ["work", "personal", "spam"]
PRIORITIES = ["high", "medium", "low"]
REQUIRED_FIELDS = {"sender", "subject", "body", "label", "priority"}


@pytest.fixture
def env():
    e = EmailEnv()
    e.reset()
    return e


def run_full_episode(env: EmailEnv):
    """Helper: run one complete 3-step episode, return total reward."""
    env.reset()
    total = 0
    done  = False
    while not done:
        valid  = env.valid_actions
        action = valid[0]
        _, reward, done, _ = env.step(action)
        total += reward
    return total


# Feature: email-triage-rl, Property 1: Email record structural completeness
def test_all_emails_have_required_fields():
    """Validates: Requirements 1.1"""
    env = EmailEnv()
    for email in env.emails:
        for field in REQUIRED_FIELDS:
            val = getattr(email, field)
            assert isinstance(val, str) and val, f"Field '{field}' is empty or missing"
        assert email.label in LABELS
        assert email.priority in PRIORITIES


# Feature: email-triage-rl, Property 2: Action space rejection of invalid actions
@given(st.text())
@settings(max_examples=100)
def test_invalid_action_raises_and_state_unchanged(action_str):
    """Validates: Requirements 2.2, 2.3"""
    if action_str in EmailEnv.ACTION_SPACE:
        return
    env = EmailEnv()
    env.reset()
    before = env.current_email
    with pytest.raises(ValueError):
        env.step(action_str)
    assert env.current_email == before


# Feature: email-triage-rl, Property 10: render() output contains all email fields
def test_render_contains_all_fields(env):
    """Validates: Requirements 4.4, 4.5"""
    output = env.render()
    e = env.current_email
    assert e.sender   in output
    assert e.subject  in output
    assert e.body     in output
    assert e.label    in output
    assert e.priority in output


# Feature: email-triage-rl, Property 11: reset() returns a valid dataset email
@given(st.integers(min_value=0, max_value=99))
@settings(max_examples=100)
def test_reset_returns_dataset_email(_seed):
    """Validates: Requirements 4.1"""
    env   = EmailEnv()
    email = env.reset()
    assert email.__dict__ in [e.__dict__ for e in env.emails]


# Feature: email-triage-rl, Property 12: step() return shape invariant (multi-step)
def test_step_return_shape_multistep():
    """Validates: Requirements 4.2, 4.3 — done=True only after phase 3."""
    env = EmailEnv()
    env.reset()
    results = []
    done = False
    while not done:
        valid = env.valid_actions
        state, reward, done, info = env.step(valid[0])
        results.append((state, reward, done, info))
        assert isinstance(state, Email)
        assert isinstance(reward, int)
        assert "breakdown" in info
    assert results[-1][2] is True   # last step done=True
    assert len(results) <= 3


# Unit: dataset loads correctly
def test_dataset_loads():
    env = EmailEnv()
    assert len(env.emails) >= 27  # dataset may grow as new emails are added


# Unit: missing file raises FileNotFoundError
def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        EmailEnv(data_path="nonexistent.json")


# Unit: malformed JSON raises ValueError
def test_malformed_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    with pytest.raises(ValueError):
        EmailEnv(data_path=str(bad))


# Unit: step before reset raises RuntimeError
def test_step_before_reset_raises():
    env = EmailEnv()
    with pytest.raises(RuntimeError):
        env.step("classify_work")


# Unit: render before reset raises RuntimeError
def test_render_before_reset_raises():
    env = EmailEnv()
    with pytest.raises(RuntimeError):
        env.render()


# Unit: action space contains all 9 actions (including 'done')
def test_action_space_completeness():
    expected = {
        "classify_work", "classify_personal", "classify_spam",
        "set_high_priority", "set_medium_priority", "set_low_priority",
        "archive_email", "flag_urgent", "done",
    }
    assert set(EmailEnv.ACTION_SPACE) == expected


# Unit: dataset covers all label/priority combinations
def test_dataset_coverage():
    env = EmailEnv()
    assert {e.label    for e in env.emails} == {"work", "personal", "spam"}
    assert {e.priority for e in env.emails} == {"high", "medium", "low"}


# Unit: phase advances correctly through 3 steps
def test_phase_progression():
    env = EmailEnv()
    env.reset()
    assert env.current_phase == 0
    env.step("classify_work")
    assert env.current_phase == 1
    env.step("set_high_priority")
    assert env.current_phase == 2
    _, _, done, _ = env.step("done")
    assert done is True


# Unit: wrong-phase action raises ValueError
def test_wrong_phase_action_raises():
    env = EmailEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step("set_high_priority")   # phase 0 expects classify_*
