import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from models import RewardBreakdown

# Strategy for generating arbitrary RewardBreakdown instances
reward_breakdown_st = st.builds(
    RewardBreakdown,
    classification=st.integers(min_value=-100, max_value=100),
    priority=st.integers(min_value=-100, max_value=100),
    urgency=st.integers(min_value=-100, max_value=100),
    archive=st.integers(min_value=-100, max_value=100),
)


# Feature: email-triage-rl, Property 8: RewardBreakdown dict round-trip
@given(reward_breakdown_st)
@settings(max_examples=100)
def test_reward_breakdown_dict_roundtrip(rb):
    """Validates: Requirements 3.9"""
    assert RewardBreakdown.from_dict(rb.to_dict()) == rb


# Feature: email-triage-rl, Property 9: RewardBreakdown pretty-print round-trip
@given(reward_breakdown_st)
@settings(max_examples=100)
def test_reward_breakdown_pretty_roundtrip(rb):
    """Validates: Requirements 6.3, 6.4"""
    assert RewardBreakdown.parse_pretty(rb.pretty()) == rb


def test_total_is_sum():
    rb = RewardBreakdown(classification=10, priority=5, urgency=-10, archive=6)
    assert rb.total() == 11


def test_parse_pretty_malformed_raises():
    with pytest.raises(ValueError):
        RewardBreakdown.parse_pretty("not a valid string")
