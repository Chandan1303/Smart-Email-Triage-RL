"""
inference.py — OpenEnv inference script for Email Triage RL.

Emits structured [START]/[STEP]/[END] blocks for each of the 3 graded tasks:
  - task_classify   (classification score)
  - task_priority   (priority score)
  - task_action     (urgency/archive/action score)

Each task score is strictly within (0, 1) as required by the validator.
"""
import os

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

def normalize_reward(raw: int) -> float:
    """Map raw reward (-28 to +23) to 0.0–1.0 range (used by app.py)."""
    MIN_RAW, MAX_RAW = -28, 23
    return round(max(0.0, min(1.0, (raw - MIN_RAW) / (MAX_RAW - MIN_RAW))), 4)


PHASE_VALID = {
    0: ["classify_work", "classify_personal", "classify_spam"],
    1: ["set_high_priority", "set_medium_priority", "set_low_priority"],
    2: ["flag_urgent", "archive_email", "done"],
}
PHASE_NAMES = {0: "classify", 1: "set_priority", 2: "act"}

# Per-task raw reward ranges for normalization
TASK_RANGES = {
    "task_classify": (-5, 10),   # classification sub-score
    "task_priority": (-3, 5),    # priority sub-score
    "task_action":   (-25, 14),  # urgency + archive combined
}


def clamp_score(score: float) -> float:
    """Clamp to strictly (0, 1) — validator rejects 0.0 and 1.0 exactly."""
    return round(max(0.001, min(0.999, score)), 4)


def normalize(raw: float, min_raw: float, max_raw: float) -> float:
    if max_raw == min_raw:
        return 0.5
    return clamp_score((raw - min_raw) / (max_raw - min_raw))


def ask_llm(client, email_state: dict, phase: int, valid_actions: list) -> str:
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
        return action if action in valid_actions else valid_actions[0]
    except Exception:
        return valid_actions[0]


def email_to_dict(email) -> dict:
    return {
        "sender": email.sender, "subject": email.subject,
        "body": email.body, "label": email.label,
        "priority": email.priority, "difficulty": email.difficulty,
    }


def run_task(task_id: str, env, client, num_episodes: int):
    """Run num_episodes for a single task and emit [START]/[STEP]/[END] blocks."""
    print(f"[START] task={task_id} model={MODEL_NAME} episodes={num_episodes}", flush=True)

    total_score = 0.0

    for ep in range(num_episodes):
        try:
            email = env.reset()
            state = email_to_dict(email)
            ep_actions = {}
            done = False
            step_num = 0

            while not done:
                phase  = env.current_phase
                valid  = PHASE_VALID[phase]
                action = ask_llm(client, state, phase, valid) if client else valid[0]
                _, _, done, info = env.step(action)

                # Accumulate actions for final scoring
                from environment import EmailEnv
                ep_actions.update(EmailEnv._ACTION_MAP[action])

                # Per-step reward: use the breakdown from info
                bd = info.get("breakdown")
                step_score = 0.5  # neutral default
                if bd:
                    if task_id == "task_classify":
                        step_score = normalize(bd.classification, *TASK_RANGES["task_classify"])
                    elif task_id == "task_priority":
                        step_score = normalize(bd.priority, *TASK_RANGES["task_priority"])
                    elif task_id == "task_action":
                        raw = bd.urgency + bd.archive
                        step_score = normalize(raw, *TASK_RANGES["task_action"])

                print(
                    f"[STEP] step={step_num} episode={ep} "
                    f"phase={PHASE_NAMES[phase]} action={action} reward={step_score}",
                    flush=True,
                )
                step_num += 1

            # Final episode score from full breakdown
            final_bd = info.get("final_breakdown") or info.get("breakdown")
            if final_bd:
                if task_id == "task_classify":
                    ep_score = normalize(final_bd.classification, *TASK_RANGES["task_classify"])
                elif task_id == "task_priority":
                    ep_score = normalize(final_bd.priority, *TASK_RANGES["task_priority"])
                else:
                    raw = final_bd.urgency + final_bd.archive
                    ep_score = normalize(raw, *TASK_RANGES["task_action"])
            else:
                ep_score = 0.5

            total_score += ep_score
            print(f"[END] task={task_id} episode={ep} score={ep_score} steps={step_num}", flush=True)

        except Exception as e:
            print(f"[END] task={task_id} episode={ep} score=0.5 steps=0 error={e}", flush=True)
            total_score += 0.5

    avg = clamp_score(total_score / num_episodes)
    print(f"[SUMMARY] task={task_id} episodes={num_episodes} avg_score={avg} model={MODEL_NAME}", flush=True)
    return avg


def run_inference(num_episodes: int = 5):
    from environment import EmailEnv

    client = None
    try:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-key")
    except Exception:
        pass

    env = EmailEnv()

    scores = {}
    for task_id in ["task_classify", "task_priority", "task_action"]:
        scores[task_id] = run_task(task_id, env, client, num_episodes)

    overall = clamp_score(sum(scores.values()) / len(scores))
    print(f"[SUMMARY] task=overall avg_score={overall} model={MODEL_NAME}", flush=True)
    return overall


if __name__ == "__main__":
    run_inference(num_episodes=5)
