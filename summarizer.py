"""
summarizer.py — Email summarization for the Email Triage RL environment.

Produces a concise ≤15-word summary from an email's subject and body
without requiring any external API.
"""
from models import Email


def summarize(email: Email) -> str:
    """
    Return a summary of at most 15 words derived from the email subject and body.
    Falls back to the subject if the body is empty or whitespace-only.
    Returns a single dash if both subject and body are blank.
    """
    body_stripped = email.body.strip() if email.body else ""
    subject_stripped = email.subject.strip() if email.subject else ""

    if not body_stripped:
        return subject_stripped if subject_stripped else "-"

    subject_words = subject_stripped.split()
    body_words = body_stripped.split()

    if not subject_words and not body_words:
        return "-"

    combined = subject_words[:]
    for word in body_words:
        if word.lower() not in {w.lower() for w in combined}:
            combined.append(word)
        if len(combined) >= 15:
            break

    result = " ".join(combined[:15])
    return result if result.strip() else "-"
