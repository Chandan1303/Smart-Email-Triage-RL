"""
Q-Learning agent for the multi-step Email Triage RL environment.

State  : (label, priority, phase)  — 9 email states × 3 phases = 27 states
Actions: phase-restricted discrete actions
"""
import random
from collections import defaultdict
from environment import EmailEnv


def email_to_state(email, phase: int = 0) -> tuple:
    return (email.label, email.priority, phase)


class QLearningAgent:
    def __init__(
        self,
        actions: list,
        alpha: float = 0.3,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.997,
    ):
        self.actions       = actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q: dict = defaultdict(lambda: {a: 0.0 for a in actions})

    def choose_action(self, state: tuple, valid_actions: list) -> str:
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        q_valid = {a: self.q[state][a] for a in valid_actions}
        return max(q_valid, key=q_valid.get)

    def update(self, state, action, reward, next_state, next_valid):
        best_next = max(self.q[next_state][a] for a in next_valid) if next_valid else 0.0
        td_target = reward + self.gamma * best_next
        self.q[state][action] += self.alpha * (td_target - self.q[state][action])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env: EmailEnv, episodes: int = 1000) -> list:
        """Run multi-step episodes. Returns list of total episode rewards."""
        episode_rewards = []
        for _ in range(episodes):
            email = env.reset()
            total = 0
            done  = False
            while not done:
                phase        = env.current_phase
                valid        = env.valid_actions
                state        = email_to_state(email, phase)
                action       = self.choose_action(state, valid)
                _, reward, done, info = env.step(action)
                next_phase   = env.current_phase
                next_valid   = env.valid_actions if not done else []
                next_state   = email_to_state(email, next_phase)
                self.update(state, action, reward, next_state, next_valid)
                total += reward
            episode_rewards.append(total)
        return episode_rewards

    def best_action(self, email, phase: int = 0) -> str:
        valid = EmailEnv.PHASE_ACTIONS.get(phase, [])
        state = email_to_state(email, phase)
        q_valid = {a: self.q[state][a] for a in valid}
        return max(q_valid, key=q_valid.get) if q_valid else "done"

    def full_episode_actions(self, email) -> list:
        """Return the agent's best action for each phase."""
        return [self.best_action(email, phase) for phase in range(3)]

    def q_table_summary(self) -> str:
        lines = []
        for state, actions in sorted(self.q.items()):
            valid = EmailEnv.PHASE_ACTIONS.get(state[2], list(actions.keys()))
            best  = max(valid, key=lambda a: actions[a])
            lines.append(
                f"  {state[0]:10s} / {state[1]:8s} / phase {state[2]}  "
                f"→  {best:25s}  Q={actions[best]:+.2f}"
            )
        return "\n".join(lines) if lines else "Not trained yet."
