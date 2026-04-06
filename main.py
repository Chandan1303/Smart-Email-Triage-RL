"""
main.py — test the Email Triage RL environment and Q-Learning agent from CLI.
Run: python main.py
"""
from environment import EmailEnv
from agent import QLearningAgent

EPISODES = 1000


def run_manual_episode(env: EmailEnv):
    """Walk through one episode manually to show multi-step flow."""
    print("\n" + "="*60)
    print("MANUAL EPISODE — multi-step triage")
    print("="*60)
    email = env.reset()
    print(env.render())
    print()

    # Hardcode a sensible action sequence for demo
    actions = []
    if email.label == "work":      actions.append("classify_work")
    elif email.label == "personal":actions.append("classify_personal")
    else:                          actions.append("classify_spam")

    if email.priority == "high":   actions.append("set_high_priority")
    elif email.priority == "medium":actions.append("set_medium_priority")
    else:                          actions.append("set_low_priority")

    if email.priority == "high":   actions.append("flag_urgent")
    elif email.label == "spam":    actions.append("archive_email")
    else:                          actions.append("done")

    total = 0
    for action in actions:
        _, reward, done, info = env.step(action)
        print(f"  Action: {action:25s}  step_reward={reward:+d}")
        total += reward
        if done:
            bd = info["final_breakdown"]
            print(f"\n  Final breakdown:")
            print(f"    Classification : {bd.classification:+d}")
            print(f"    Priority       : {bd.priority:+d}")
            print(f"    Urgency        : {bd.urgency:+d}")
            print(f"    Archive        : {bd.archive:+d}")
            print(f"    Total          : {bd.total():+d}")
            break


def run_training(env: EmailEnv):
    print("\n" + "="*60)
    print(f"TRAINING Q-LEARNING AGENT — {EPISODES} episodes")
    print("="*60)
    agent = QLearningAgent(actions=EmailEnv.ACTION_SPACE)
    rewards = agent.train(env, episodes=EPISODES)

    total   = sum(rewards)
    avg     = total / len(rewards)
    wins    = sum(1 for r in rewards if r > 0)
    win_pct = 100 * wins / len(rewards)

    print(f"  Total reward : {total:+d}")
    print(f"  Avg reward   : {avg:+.2f}")
    print(f"  Win rate     : {win_pct:.1f}%")
    print(f"  Final ε      : {agent.epsilon:.4f}")
    print()
    print("Q-Table (best action per state):")
    print(agent.q_table_summary())

    # Show agent decisions on a few emails
    print("\nAgent decisions on sample emails:")
    for email in env.emails[:5]:
        best = agent.full_episode_actions(email)
        print(f"  [{email.label:10s}/{email.priority:6s}]  {best}")


if __name__ == "__main__":
    env = EmailEnv()
    run_manual_episode(env)
    run_training(env)
    print("\nDone. Run 'python app.py' to launch the Gradio UI.")
