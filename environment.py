"""
EmailEnv — multi-step RL environment for email triage.

Each episode = one email.
The agent takes up to 3 steps per email:
  Step 1: classify  (classify_work / classify_personal / classify_spam)
  Step 2: prioritize (set_high_priority / set_medium_priority / set_low_priority)
  Step 3: act       (flag_urgent / archive_email / done)

done=True is returned after step 3, or if the agent calls 'done' early.
"""
import json
import random
from models import Email, RewardBreakdown
from graders import calculate_reward


class EmailEnv:
    # Full flat action space (used by single-step / legacy mode)
    ACTION_SPACE = [
        "classify_work",
        "classify_personal",
        "classify_spam",
        "set_high_priority",
        "set_medium_priority",
        "set_low_priority",
        "archive_email",
        "flag_urgent",
        "done",
    ]

    # Multi-step phase → valid actions
    PHASE_ACTIONS = {
        0: ["classify_work", "classify_personal", "classify_spam"],
        1: ["set_high_priority", "set_medium_priority", "set_low_priority"],
        2: ["flag_urgent", "archive_email", "done"],
    }
    PHASE_NAMES = {0: "Classify", 1: "Set Priority", 2: "Act (Flag/Archive/Done)"}

    _ACTION_MAP = {
        "classify_work":       {"label": "work"},
        "classify_personal":   {"label": "personal"},
        "classify_spam":       {"label": "spam"},
        "set_high_priority":   {"priority": "high"},
        "set_medium_priority": {"priority": "medium"},
        "set_low_priority":    {"priority": "low"},
        "archive_email":       {"archive": True, "flag": False},
        "flag_urgent":         {"flag": True},
        "done":                {"done": True, "flag": False},
    }

    def __init__(self, data_path: str = "data.json", difficulty: str | None = None) -> None:
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at path: {data_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in dataset '{data_path}': {e}")

        all_emails: list[Email] = [Email(**r) for r in raw]

        if difficulty is not None:
            filtered = [e for e in all_emails if e.difficulty == difficulty]
            if not filtered:
                raise ValueError(
                    f"No emails found for difficulty '{difficulty}'. "
                    f"Valid values: easy, medium, hard."
                )
            self.emails = filtered
        else:
            self.emails = all_emails

        self.difficulty: str | None = difficulty
        self.current_email: Email | None = None
        self._phase: int = 0
        self._episode_actions: dict = {}
        self._episode_reward: int = 0

    # ── public interface ──────────────────────────────────────────────────────
    def reset(self) -> Email:
        self.current_email   = random.choice(self.emails)
        self._phase          = 0
        self._episode_actions = {}
        self._episode_reward  = 0
        return self.current_email

    def step(self, action: str) -> tuple:
        """
        Returns (email, step_reward, done, info).
        info keys: phase, breakdown (on final step), episode_total
        """
        if self.current_email is None:
            raise RuntimeError("Call reset() before step().")
        if action not in self.ACTION_SPACE:
            raise ValueError(
                f"Invalid action '{action}'. Valid: {self.ACTION_SPACE}"
            )

        valid_now = self.PHASE_ACTIONS[self._phase]
        if action not in valid_now:
            raise ValueError(
                f"Action '{action}' not valid in phase {self._phase} "
                f"({self.PHASE_NAMES[self._phase]}). Valid: {valid_now}"
            )

        # Merge action into episode dict
        self._episode_actions.update(self._ACTION_MAP[action])

        # Compute partial reward for this step
        partial_bd = calculate_reward(self._ACTION_MAP[action], self.current_email)
        step_reward = partial_bd.total()
        self._episode_reward += step_reward

        self._phase += 1
        done = self._phase >= 3 or action == "done"

        info: dict = {
            "phase":         self._phase,
            "phase_name":    self.PHASE_NAMES.get(self._phase, "complete"),
            "step_reward":   step_reward,
            "episode_total": self._episode_reward,
            "breakdown":     partial_bd,
        }

        if done:
            # Final full breakdown across all steps
            full_bd = calculate_reward(self._episode_actions, self.current_email)
            info["final_breakdown"] = full_bd
            info["episode_total"]   = full_bd.total()

        return (self.current_email, step_reward, done, info)

    def render(self) -> str:
        if self.current_email is None:
            raise RuntimeError("Call reset() before render().")
        e = self.current_email
        phase_name = self.PHASE_NAMES.get(self._phase, "complete")
        return (
            f"From    : {e.sender}\n"
            f"Subject : {e.subject}\n"
            f"Body    : {e.body}\n"
            f"Label   : {e.label}\n"
            f"Priority: {e.priority}\n"
            f"Phase   : {phase_name}"
        )

    @property
    def current_phase(self) -> int:
        return self._phase

    @property
    def valid_actions(self) -> list:
        return self.PHASE_ACTIONS.get(self._phase, [])

    def state(self) -> dict:
        """Return current environment state as a typed dict (OpenEnv compliance)."""
        if self.current_email is None:
            return {}
        e = self.current_email
        return {
            "sender":     e.sender,
            "subject":    e.subject,
            "body":       e.body,
            "label":      e.label,
            "priority":   e.priority,
            "difficulty": e.difficulty,
            "phase":      self._phase,
            "phase_name": self.PHASE_NAMES.get(self._phase, "complete"),
            "valid_actions": self.valid_actions,
        }

    def set_difficulty(self, difficulty: str | None) -> None:
        """Filter the email pool to the given difficulty level and reset."""
        import json
        try:
            with open("data.json", "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            raise

        all_emails = [Email(**r) for r in raw]
        if difficulty is not None:
            filtered = [e for e in all_emails if e.difficulty == difficulty]
            if not filtered:
                raise ValueError(
                    f"No emails found for difficulty '{difficulty}'. "
                    f"Valid values: easy, medium, hard."
                )
            self.emails = filtered
        else:
            self.emails = all_emails
        self.difficulty = difficulty
        self.current_email = None
        self._phase = 0
        self._episode_actions = {}
        self._episode_reward = 0
