"""
inference.py — OpenEnv inference script for Email Triage RL.

Uses OpenAI client to drive the EmailEnv through 3-step episodes.
Emits structured [START], [STEP], [END] logs for automated evaluation.

Required environment variables:
    API_BASE_URL  — LLM API endpoint
    MODEL_NAME    — model identifier
    HF_TOKEN      — Hugging Face / API key
"""
import os
import json
import time
from openai import OpenAI
from environment import EmailEnv

# ── config from environment ───────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-key",
)

# ── helpers ───────────────────────────────────────────────────────────────────
PHASE_VALID = {
    0: ["classify_work", "classify_personal", "classify_spam"],
    1: ["set_high_priority", "set_medium_priority", "set_low_priority"],
    2: ["flag_urgent", "archive_email", "done"],
}

PHASE_NAMES = {0: "classify", 1: "set_priority", 2: "act"}

def normalize_reward(raw: int) -> float:
    """Map raw reward (-28 to +23) to 0.0–1.0 range."""
    MIN_RAW, MAX_RAW = -28, 23
    return round(max(0.0, min(1.0, (raw - MIN_RAW) / (MAX_RAW - MIN_RAW))), 4)

def ask_llm(email_state: dict, phase: int, valid_actions: list) -> str:
    """Ask the LLM to pick an action for the current phase."""
    prompt = (
        f"You are an email triage agent. Given this email:\n"
        f"  From: {email_state['sender']}\n"
        f"  Subject: {email_state['subject']}\n"
        f"  Body: {email_state['body']}\n"
        f"  (label hint: {email_state['label']}, priority hint: {email_state['priority']})\n\n"
        f"Phase {phase} ({PHASE_NAMES[phase]}). Choose exactly one action from: {valid_actions}\n"
        f"Reply with ONLY the action name, nothing else."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
        )
        action = response.choices[0].message.content.strip().lower()
        # Validate — fall back to first valid action if LLM hallucinates
        if action not in valid_actions:
            action = valid_actions[0]
        return action
    except Exception:
        return valid_actions[0]

def email_to_dict(email) -> dict:
    return {
        "sender":     email.sender,
        "subject":    email.subject,
        "body":       email.body,
        "label":      email.label,
        "priority":   email.priority,
        "difficulty": email.difficulty,
    }

# ── main inference loop ───────────────────────────────────────────────────────
def run_inference(num_episodes: int = 5):
    env = EmailEnv()

    print(json.dumps({
        "event":      "[START]",
        "model":      MODEL_NAME,
        "api_base":   API_BASE_URL,
        "episodes":   num_episodes,
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))

    total_reward = 0.0

    for ep in range(num_episodes):
        email = env.reset()
        state = email_to_dict(email)
        ep_raw_reward = 0
        steps_log = []

        done = False
        step_num = 0
        while not done:
            phase        = env.current_phase
            valid        = PHASE_VALID[phase]
            action       = ask_llm(state, phase, valid)
            _, raw, done, info = env.step(action)
            ep_raw_reward += raw

            step_log = {
                "event":       "[STEP]",
                "episode":     ep,
                "step":        step_num,
                "phase":       PHASE_NAMES[phase],
                "action":      action,
                "raw_reward":  raw,
                "norm_reward": normalize_reward(raw),
                "done":        done,
            }
            print(json.dumps(step_log))
            steps_log.append(step_log)
            step_num += 1

        norm_ep = normalize_reward(ep_raw_reward)
        total_reward += norm_ep

        final_bd = info.get("final_breakdown")
        print(json.dumps({
            "event":            "[END]",
            "episode":          ep,
            "state":            state,
            "raw_episode_reward":  ep_raw_reward,
            "norm_episode_reward": norm_ep,
            "classification":   final_bd.classification if final_bd else 0,
            "priority":         final_bd.priority       if final_bd else 0,
            "urgency":          final_bd.urgency         if final_bd else 0,
            "archive":          final_bd.archive         if final_bd else 0,
            "steps":            step_num,
        }))

    avg_reward = round(total_reward / num_episodes, 4)
    print(json.dumps({
        "event":       "[SUMMARY]",
        "episodes":    num_episodes,
        "avg_norm_reward": avg_reward,
        "model":       MODEL_NAME,
    }))

    return avg_reward


if __name__ == "__main__":
    run_inference(num_episodes=5)
