"""
reply_generator.py — Suggested reply generator for the Email Triage RL environment.

Generates a short, contextually appropriate reply based on the email's label
and subject without requiring any external API.
"""
from models import Email

# Templates keyed by label
_WORK_TEMPLATE = "Thank you for your email regarding '{subject}'. I will look into this and get back to you shortly."
_PERSONAL_TEMPLATE = "Hi! Thanks for reaching out about '{subject}'. I'll respond as soon as I can."
_SPAM_DECLINE = "This message appears to be unsolicited. No action will be taken."

_TEMPLATES: dict[str, str] = {
    "work": _WORK_TEMPLATE,
    "personal": _PERSONAL_TEMPLATE,
    "spam": _SPAM_DECLINE,
}


def generate_reply(email: Email) -> str:
    """
    Return a suggested reply string for the given email.
    Work emails reference the subject; spam emails decline politely.
    """
    template = _TEMPLATES.get(email.label, _PERSONAL_TEMPLATE)
    return template.format(subject=email.subject)


def get_templates() -> dict[str, str]:
    """Return a copy of the reply templates (for serialization / testing)."""
    return dict(_TEMPLATES)


def set_templates(templates: dict[str, str]) -> None:
    """Replace the reply templates (enables round-trip testing)."""
    global _TEMPLATES
    _TEMPLATES = dict(templates)
