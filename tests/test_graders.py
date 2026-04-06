import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings
from hypothesis import strategies as st
from models import Email
from graders import calculate_reward

LABELS = ["work", "personal", "spam"]
PRIORITIES = ["high", "medium", "low"]

email_st = st.builds(
    Email,
    sender=st.text(min_size=1, max_size=50),
    subject=st.text(min_size=1, max_size=100),
    body=st.text(min_size=1, max_size=200),
    label=st.sampled_from(LABELS),
    priority=st.sampled_from(PRIORITIES),
)

high_email_st = st.builds(
    Email,
    sender=st.text(min_size=1, max_size=50),
    subject=st.text(min_size=1, max_size=100),
    body=st.text(min_size=1, max_size=200),
    label=st.sampled_from(LABELS),
    priority=st.just("high"),
)


# Feature: email-triage-rl, Property 3: Classification reward correctness
@given(email_st, st.sampled_from(LABELS))
@settings(max_examples=100)
def test_classification_reward(email, chosen_label):
    """Validates: Requirements 3.1, 3.2"""
    rb = calculate_reward({"label": chosen_label}, email)
    if chosen_label == email.label:
        assert rb.classification == 10
    else:
        assert rb.classification == -5


# Feature: email-triage-rl, Property 4: Priority reward correctness
@given(email_st, st.sampled_from(PRIORITIES))
@settings(max_examples=100)
def test_priority_reward(email, chosen_priority):
    """Validates: Requirements 3.3, 3.4"""
    rb = calculate_reward({"priority": chosen_priority}, email)
    if chosen_priority == email.priority:
        assert rb.priority == 5
    else:
        assert rb.priority == -3


# Feature: email-triage-rl, Property 5: Urgency flag reward correctness
@given(high_email_st, st.booleans())
@settings(max_examples=100)
def test_urgency_reward(email, flagged):
    """Validates: Requirements 3.5, 3.6"""
    rb = calculate_reward({"flag": flagged}, email)
    if flagged:
        assert rb.urgency == 8
    else:
        assert rb.urgency == -10


# Feature: email-triage-rl, Property 6: Archive reward correctness
@given(email_st)
@settings(max_examples=100)
def test_archive_reward(email):
    """Validates: Requirements 3.7, 3.8"""
    rb = calculate_reward({"archive": True}, email)
    if email.label == "spam":
        assert rb.archive == 6
    else:
        assert rb.archive == -15


# Feature: email-triage-rl, Property 7: Reward total equals sum of sub-scores
@given(email_st, st.sampled_from(LABELS), st.sampled_from(PRIORITIES), st.booleans(), st.booleans())
@settings(max_examples=100)
def test_reward_total_is_sum(email, label, priority, flag, archive):
    """Validates: Requirements 6.2"""
    action = {"label": label, "priority": priority, "flag": flag, "archive": archive}
    rb = calculate_reward(action, email)
    assert rb.total() == rb.classification + rb.priority + rb.urgency + rb.archive
