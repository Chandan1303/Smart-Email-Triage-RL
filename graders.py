from models import Email, RewardBreakdown


def calculate_reward(action: dict, email: Email) -> RewardBreakdown:
    """
    Compute a RewardBreakdown for the given action against the ground-truth email.

    action keys (all optional):
        label    : "work" | "personal" | "spam"
        priority : "high" | "medium" | "low"
        flag     : bool  — whether the agent flagged the email as urgent
        archive  : bool  — whether the agent chose to archive the email
    """
    breakdown = RewardBreakdown()

    # Classification sub-score
    if "label" in action:
        breakdown.classification = 10 if action["label"] == email.label else -5

    # Priority sub-score
    if "priority" in action:
        breakdown.priority = 5 if action["priority"] == email.priority else -3

    # Urgency sub-score — only evaluated when the agent explicitly acts (phase 2).
    # "flag" key must be present in the action dict (True or False).
    # If the action dict has no "flag" key at all, urgency is not scored this step.
    if "flag" in action or "archive" in action or "done" in action:
        if email.priority == "high":
            breakdown.urgency = 8 if action.get("flag") is True else -10

    # Archive sub-score
    if action.get("archive") is True:
        breakdown.archive = 6 if email.label == "spam" else -15

    return breakdown
