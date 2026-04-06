import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from models import Email
from summarizer import summarize
from reply_generator import generate_reply, get_templates, set_templates
from environment import EmailEnv

LABELS     = ["work", "personal", "spam"]
PRIORITIES = ["high", "medium", "low"]
DIFFICULTIES = ["easy", "medium", "hard"]

email_st = st.builds(
    Email,
    sender=st.text(min_size=1, max_size=50),
    subject=st.text(min_size=1, max_size=80),
    body=st.text(min_size=1, max_size=200),
    label=st.sampled_from(LABELS),
    priority=st.sampled_from(PRIORITIES),
    difficulty=st.sampled_from(DIFFICULTIES),
)


# ── Summarizer ────────────────────────────────────────────────────────────────

# Feature: email-triage-rl, Property 13: summarize returns ≤15 words
@given(email_st)
@settings(max_examples=100)
def test_summarize_word_count(email):
    """Validates: Requirements 7.1, 7.5"""
    result = summarize(email)
    assert isinstance(result, str)
    assert 1 <= len(result.split()) <= 15


# Feature: email-triage-rl, Property 14: summarize returns non-empty string
@given(email_st)
@settings(max_examples=100)
def test_summarize_non_empty(email):
    """Validates: Requirements 7.4"""
    result = summarize(email)
    assert result and result.strip()


def test_summarize_empty_body_returns_subject():
    """Validates: Requirements 7.3"""
    email = Email(sender="a@b.com", subject="Hello World", body="",
                  label="work", priority="low", difficulty="easy")
    result = summarize(email)
    assert "Hello" in result or result == "Hello World"


# ── Reply Generator ───────────────────────────────────────────────────────────

# Feature: email-triage-rl, Property 15: generate_reply returns non-empty string
@given(email_st)
@settings(max_examples=100)
def test_generate_reply_non_empty(email):
    """Validates: Requirements 8.3"""
    result = generate_reply(email)
    assert isinstance(result, str) and result.strip()


# Feature: email-triage-rl, Property 16: work reply references subject
@given(
    st.builds(
        Email,
        sender=st.text(min_size=1, max_size=50),
        subject=st.text(min_size=3, max_size=80, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs"))),
        body=st.text(min_size=1, max_size=200),
        label=st.just("work"),
        priority=st.sampled_from(PRIORITIES),
        difficulty=st.sampled_from(DIFFICULTIES),
    )
)
@settings(max_examples=100)
def test_work_reply_references_subject(email):
    """Validates: Requirements 8.4"""
    reply = generate_reply(email)
    assert email.subject in reply


# Feature: email-triage-rl, Property 17: spam reply declines
@given(
    st.builds(
        Email,
        sender=st.text(min_size=1, max_size=50),
        subject=st.text(min_size=1, max_size=80),
        body=st.text(min_size=1, max_size=200),
        label=st.just("spam"),
        priority=st.sampled_from(PRIORITIES),
        difficulty=st.sampled_from(DIFFICULTIES),
    )
)
@settings(max_examples=100)
def test_spam_reply_declines(email):
    """Validates: Requirements 8.5"""
    reply = generate_reply(email)
    # Spam reply should not reference the subject (it's a generic decline)
    assert reply  # non-empty
    assert "unsolicited" in reply.lower() or "no action" in reply.lower()


def test_reply_templates_round_trip():
    """Validates: Requirements 8.6 — serialize/deserialize templates without data loss"""
    original = get_templates()
    set_templates(original)
    assert get_templates() == original


# ── Difficulty Levels ─────────────────────────────────────────────────────────

def test_difficulty_easy_filters_correctly():
    """Validates: Requirements 9.2, 9.5"""
    env = EmailEnv(difficulty="easy")
    assert all(e.difficulty == "easy" for e in env.emails)


def test_difficulty_medium_filters_correctly():
    """Validates: Requirements 9.3, 9.5"""
    env = EmailEnv(difficulty="medium")
    assert all(e.difficulty == "medium" for e in env.emails)


def test_difficulty_hard_filters_correctly():
    """Validates: Requirements 9.4, 9.5"""
    env = EmailEnv(difficulty="hard")
    assert all(e.difficulty == "hard" for e in env.emails)


def test_all_difficulties_present_in_dataset():
    """Validates: Requirements 9.6"""
    env = EmailEnv()
    difficulties = {e.difficulty for e in env.emails}
    assert difficulties == {"easy", "medium", "hard"}


def test_invalid_difficulty_raises():
    """Validates: Requirements 9.7"""
    with pytest.raises(ValueError, match="No emails found for difficulty"):
        EmailEnv(difficulty="impossible")


def test_set_difficulty_at_runtime():
    """Validates: Requirements 9.5 — difficulty can be changed after init"""
    env = EmailEnv()
    env.set_difficulty("hard")
    assert all(e.difficulty == "hard" for e in env.emails)
    env.set_difficulty("easy")
    assert all(e.difficulty == "easy" for e in env.emails)


# Feature: email-triage-rl, Property 18: reset() after difficulty change returns correct difficulty
@given(st.sampled_from(["easy", "medium", "hard"]))
@settings(max_examples=30)
def test_reset_respects_difficulty(difficulty):
    """Validates: Requirements 9.2, 9.3, 9.4"""
    env = EmailEnv(difficulty=difficulty)
    email = env.reset()
    assert email.difficulty == difficulty
