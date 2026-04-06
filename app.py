"""
app.py — Gradio UI for Smart Email Triage RL Environment
Tabs: Manual Play (multi-step) | Train Agent
"""
import gradio as gr
from environment import EmailEnv
from agent import QLearningAgent, email_to_state
from summarizer import summarize
from reply_generator import generate_reply

# ── shared objects ────────────────────────────────────────────────────────────
env   = EmailEnv()
agent = QLearningAgent(actions=EmailEnv.ACTION_SPACE)
play  = {"email": env.reset(), "score": 0, "steps": 0, "ep_reward": 0}

ACTION_LABELS = {
    "classify_work":       "📁 Classify — Work",
    "classify_personal":   "👤 Classify — Personal",
    "classify_spam":       "🚫 Classify — Spam",
    "set_high_priority":   "🔴 Priority — High",
    "set_medium_priority": "🟡 Priority — Medium",
    "set_low_priority":    "🟢 Priority — Low",
    "archive_email":       "📦 Archive Email",
    "flag_urgent":         "🚨 Flag as Urgent",
    "done":                "✔️  Done (no action)",
}
A2L = ACTION_LABELS
L2A = {v: k for k, v in ACTION_LABELS.items()}

PHASE_CHOICES = {
    0: [A2L[a] for a in EmailEnv.PHASE_ACTIONS[0]],
    1: [A2L[a] for a in EmailEnv.PHASE_ACTIONS[1]],
    2: [A2L[a] for a in EmailEnv.PHASE_ACTIONS[2]],
}
PHASE_HINT = {
    0: "Step 1 of 3 — Classify this email",
    1: "Step 2 of 3 — Set the priority",
    2: "Step 3 of 3 — Flag, archive, or mark done",
}

DIFFICULTY_LABELS = ["🟢 Easy", "🟡 Medium", "🔴 Hard"]
DIFFICULTY_MAP    = {"🟢 Easy": "easy", "🟡 Medium": "medium", "🔴 Hard": "hard"}


# ── helpers ───────────────────────────────────────────────────────────────────
def priority_icon(p): return {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(p, "⚪")
def label_icon(l):    return {"work": "💼", "personal": "👤", "spam": "🚫"}.get(l, "📧")

def email_html(email):
    pi  = priority_icon(email.priority)
    li  = label_icon(email.label)
    sm  = summarize(email)
    rep = generate_reply(email)
    diff_badge = {"easy": "&#129001; Easy", "medium": "&#128993; Medium", "hard": "&#128308; Hard"}.get(email.difficulty, email.difficulty)
    return (
        '<div class="email-card">'
        f'<div class="email-row"><span class="email-key">&#9993; From</span><span class="email-val">{email.sender}</span></div>'
        f'<div class="email-row"><span class="email-key">&#128204; Subject</span><span class="email-val">{email.subject}</span></div>'
        f'<div class="email-row"><span class="email-key">&#128221; Body</span><span class="email-val">{email.body}</span></div>'
        f'<div class="email-row"><span class="email-key">&#128161; Summary</span><span class="email-val summary-text">{sm}</span></div>'
        f'<div class="email-row"><span class="email-key">&#128172; Reply</span><span class="email-val reply-text">{rep}</span></div>'
        f'<div class="email-row"><span class="email-key">{li} Label</span><span class="email-val"><span class="tag-label">{email.label.upper()}</span></span></div>'
        f'<div class="email-row"><span class="email-key">{pi} Priority</span><span class="email-val"><span class="tag-priority">{email.priority.upper()}</span></span></div>'
        f'<div class="email-row"><span class="email-key">&#127919; Difficulty</span><span class="email-val">{diff_badge}</span></div>'
        '</div>'
    )

def scoreboard_md():
    avg = play["score"] / play["steps"] if play["steps"] else 0
    acc = max(0, min(100, 50 + avg * 2))
    return (
        f'<span class="stat"><span class="stat-icon">&#127942;</span>Score <span class="stat-val">{play["score"]:+d}</span></span>'
        f'<span class="stat"><span class="stat-icon">&#128140;</span>Emails <span class="stat-val">{play["steps"]}</span></span>'
        f'<span class="stat"><span class="stat-icon">&#128200;</span>Avg Reward <span class="stat-val">{avg:+.1f}</span></span>'
        f'<span class="stat"><span class="stat-icon">&#127919;</span>Accuracy <span class="stat-val">{acc:.0f}%</span></span>'
    )

def reward_html(reward):
    cls  = "pos" if reward > 0 else ("neg" if reward < 0 else "neu")
    sign = "+" if reward > 0 else ""
    return f'<div class="reward-badge {cls}">{sign}{reward} pts</div>'


# ── play callbacks ────────────────────────────────────────────────────────────
def play_reset():
    play["email"]     = env.reset()
    play["ep_reward"] = 0
    phase = env.current_phase
    return (
        email_html(play["email"]),
        gr.update(choices=PHASE_CHOICES[phase], value=None, label=PHASE_HINT[phase]),
        "", "",
        f'<div class="scoreboard">{scoreboard_md()}</div>',
    )

def play_step(action_label: str):
    if not action_label:
        phase = env.current_phase
        return (
            email_html(play["email"]),
            gr.update(choices=PHASE_CHOICES[phase], value=None, label=PHASE_HINT[phase]),
            "",
            '<div class="reward-badge neu">⚠️ Pick an action first</div>',
            f'<div class="scoreboard">{scoreboard_md()}</div>',
        )

    action = L2A[action_label]
    _, reward, done, info = env.step(action)
    play["ep_reward"] += reward

    bd = info.get("final_breakdown") or info["breakdown"]
    bd_html = f"""<div class="breakdown">
  <div class="bd-row"><span>Classification</span><span class="{'pos' if bd.classification>=0 else 'neg'}">{bd.classification:+d}</span></div>
  <div class="bd-row"><span>Priority</span><span class="{'pos' if bd.priority>=0 else 'neg'}">{bd.priority:+d}</span></div>
  <div class="bd-row"><span>Urgency</span><span class="{'pos' if bd.urgency>=0 else 'neg'}">{bd.urgency:+d}</span></div>
  <div class="bd-row"><span>Archive</span><span class="{'pos' if bd.archive>=0 else 'neg'}">{bd.archive:+d}</span></div>
  <div class="bd-row total"><span>Total</span><span class="{'pos' if bd.total()>=0 else 'neg'}">{bd.total():+d}</span></div>
</div>"""

    if done:
        play["score"]    += play["ep_reward"]
        play["steps"]    += 1
        play["email"]     = env.reset()
        play["ep_reward"] = 0

    phase = env.current_phase
    return (
        email_html(play["email"]),
        gr.update(choices=PHASE_CHOICES.get(phase, PHASE_CHOICES[0]),
                  value=None, label=PHASE_HINT.get(phase, PHASE_HINT[0])),
        bd_html,
        reward_html(reward),
        f'<div class="scoreboard">{scoreboard_md()}</div>',
    )

def change_difficulty(diff_label: str):
    difficulty = DIFFICULTY_MAP.get(diff_label)
    env.set_difficulty(difficulty)
    play["email"]     = env.reset()
    play["score"]     = 0
    play["steps"]     = 0
    play["ep_reward"] = 0
    phase = env.current_phase
    return (
        email_html(play["email"]),
        gr.update(choices=PHASE_CHOICES[phase], value=None, label=PHASE_HINT[phase]),
        "", "",
        f'<div class="scoreboard">{scoreboard_md()}</div>',
    )


# ── train callbacks ───────────────────────────────────────────────────────────
def do_train(episodes: int):
    rewards = agent.train(env, episodes=int(episodes))
    total   = sum(rewards)
    avg     = total / len(rewards)
    wins    = sum(1 for r in rewards if r > 0)
    return (
        f"✅  Training complete — {int(episodes)} episodes\n\n"
        f"  Total reward : {total:+d}\n"
        f"  Avg reward   : {avg:+.2f}\n"
        f"  Win rate     : {100*wins/len(rewards):.1f}%\n"
        f"  Final ε      : {agent.epsilon:.4f}\n\n"
        f"── Q-Table (best action per state) ──\n"
        f"{agent.q_table_summary()}"
    )

def do_suggest():
    email   = play["email"]
    actions = agent.full_episode_actions(email)
    labels  = [A2L.get(a, a) for a in actions]
    names   = ["Classify", "Priority", "Act"]
    lines   = [f"  {names[i]:10s} → {labels[i]}" for i in range(3)]
    s       = email_to_state(email, 0)
    q_str   = "\n".join(
        f"  {'→' if a == actions[0] else ' '} {A2L.get(a,a):30s}  Q={v:+.2f}"
        for a, v in sorted(agent.q[s].items(), key=lambda x: -x[1])
        if a in EmailEnv.PHASE_ACTIONS[0]
    )
    return "🤖  Agent's full plan:\n" + "\n".join(lines) + f"\n\n── Q-values (classify phase) ──\n{q_str}"


# ── CSS — Google Material-inspired ───────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container, .main {
    background: #f8f9fa !important;
    color: #202124 !important;
    font-family: 'Google Sans', 'Roboto', 'Segoe UI', Arial, sans-serif !important;
}

.gradio-container {
    max-width: 1040px !important;
    margin: 0 auto !important;
    padding: 0 16px !important;
}

/* ── App Header ── */
.app-header {
    text-align: center;
    padding: 32px 0 16px;
}
.app-header h1 {
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0 0 6px;
    color: #1a73e8;
    letter-spacing: -0.5px;
    -webkit-text-fill-color: #1a73e8;
}
.app-header p {
    color: #5f6368;
    margin: 0;
    font-size: 1rem;
    font-weight: 400;
}

/* ── Scoreboard ── */
.scoreboard {
    background: #1a73e8;
    border-radius: 12px;
    padding: 14px 28px;
    text-align: center;
    font-size: 0.97rem;
    font-weight: 500;
    color: #ffffff;
    margin-bottom: 16px;
    letter-spacing: 0.2px;
    box-shadow: 0 2px 8px rgba(26,115,232,0.30);
}
.scoreboard .stat {
    display: inline-block;
    margin: 0 18px;
}
.scoreboard .stat-icon {
    font-size: 1.1rem;
    margin-right: 5px;
}
.scoreboard .stat-val {
    font-weight: 700;
    font-size: 1.05rem;
}

/* ── Email Card ── */
.email-card {
    background: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 12px;
    padding: 20px 24px;
    font-size: 0.92rem;
    line-height: 1.7;
    box-shadow: 0 1px 4px rgba(60,64,67,0.10);
    margin-bottom: 4px;
}
.email-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 8px 0;
    border-bottom: 1px solid #f1f3f4;
}
.email-row:last-child { border-bottom: none; }
.email-key {
    color: #5f6368;
    min-width: 100px;
    font-weight: 500;
    font-size: 0.83rem;
    padding-top: 1px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 5px;
}
.email-val {
    color: #202124;
    font-weight: 400;
    flex: 1;
    word-break: break-word;
}
.summary-text {
    color: #1a73e8;
    font-style: italic;
    font-weight: 400;
}
.reply-text {
    color: #188038;
    font-style: italic;
    font-weight: 400;
}
.tag-label {
    display: inline-block;
    background: #e8f0fe;
    color: #1a73e8;
    padding: 2px 12px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.4px;
}
.tag-priority {
    display: inline-block;
    background: #fef7e0;
    color: #b06000;
    padding: 2px 12px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.4px;
}

/* ── Reward Badge ── */
.reward-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 10px 24px;
    border-radius: 100px;
    font-size: 1.05rem;
    font-weight: 700;
    margin: 8px 0;
    letter-spacing: 0.2px;
}
.reward-badge.pos {
    background: #e6f4ea;
    color: #188038;
    border: 1.5px solid #a8d5b5;
}
.reward-badge.neg {
    background: #fce8e6;
    color: #c5221f;
    border: 1.5px solid #f5b8b5;
}
.reward-badge.neu {
    background: #f1f3f4;
    color: #5f6368;
    border: 1.5px solid #dadce0;
}

/* ── Breakdown Panel ── */
.breakdown {
    background: #f8f9fa;
    border: 1px solid #dadce0;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.88rem;
    margin-top: 6px;
}
.bd-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #f1f3f4;
    color: #3c4043;
}
.bd-row:last-child { border-bottom: none; }
.bd-row.total {
    font-weight: 700;
    margin-top: 6px;
    padding-top: 8px;
    font-size: 0.93rem;
    color: #202124;
}
.pos { color: #188038 !important; font-weight: 700; }
.neg { color: #c5221f !important; font-weight: 700; }

/* ── Reward Guide ── */
.reward-guide {
    background: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 10px;
    padding: 12px 18px;
    color: #5f6368;
    font-size: 0.82rem;
    margin-top: 14px;
    line-height: 2;
}

/* ── Gradio native overrides ── */
.svelte-1gfkn6j, .wrap, .label-wrap {
    font-family: 'Google Sans', 'Roboto', Arial, sans-serif !important;
}

/* Labels */
label span, .block label span {
    color: #5f6368 !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    font-family: 'Google Sans', 'Roboto', Arial, sans-serif !important;
}

/* Dropdown / Select */
select, .gr-dropdown select {
    font-family: 'Google Sans', 'Roboto', Arial, sans-serif !important;
    font-size: 0.92rem !important;
    color: #202124 !important;
    border: 1px solid #dadce0 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    padding: 8px 12px !important;
}

/* Buttons */
button {
    font-family: 'Google Sans', 'Roboto', Arial, sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    letter-spacing: 0.25px !important;
    transition: box-shadow 0.15s, opacity 0.15s !important;
}
button[data-testid="primary-btn"], .gr-button-primary {
    background: #1a73e8 !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 1px 3px rgba(26,115,232,0.35) !important;
}
button[data-testid="secondary-btn"], .gr-button-secondary {
    background: #ffffff !important;
    color: #1a73e8 !important;
    border: 1px solid #dadce0 !important;
}
button:hover { box-shadow: 0 2px 8px rgba(60,64,67,0.20) !important; opacity: 0.95 !important; }

/* Textbox */
textarea, input[type="text"] {
    font-family: 'Roboto Mono', 'Courier New', monospace !important;
    font-size: 0.85rem !important;
    color: #202124 !important;
    background: #f8f9fa !important;
    border: 1px solid #dadce0 !important;
    border-radius: 8px !important;
    line-height: 1.6 !important;
}

/* Radio buttons */
.gr-radio label, .gr-radio span {
    font-family: 'Google Sans', 'Roboto', Arial, sans-serif !important;
    font-size: 0.9rem !important;
    color: #202124 !important;
}

/* Tabs */
.tabs button {
    font-family: 'Google Sans', 'Roboto', Arial, sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #5f6368 !important;
    border-bottom: 3px solid transparent !important;
    padding: 10px 20px !important;
}
.tabs button.selected, .tabs button[aria-selected="true"] {
    color: #1a73e8 !important;
    border-bottom: 3px solid #1a73e8 !important;
}

/* Slider */
input[type=range] { accent-color: #1a73e8 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f1f3f4; }
::-webkit-scrollbar-thumb { background: #bdc1c6; border-radius: 3px; }
"""

# ── layout ────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Email Triage RL") as demo:

    gr.HTML("""<div class="app-header">
      <h1><span style="color:inherit;-webkit-text-fill-color:initial;">&#128236;</span> Smart Email Triage RL</h1>
      <p>Reinforcement learning environment &mdash; classify emails, set priorities, earn rewards.</p>
    </div>""")

    scoreboard = gr.HTML(value=f'<div class="scoreboard">{scoreboard_md()}</div>')

    with gr.Tabs():

        # ── Tab 1: Play ───────────────────────────────────────────────────────
        with gr.Tab("🎮 Play"):
            with gr.Row():
                diff_radio = gr.Radio(
                    choices=DIFFICULTY_LABELS,
                    value="🟢 Easy",
                    label="🎯 Difficulty Level",
                    interactive=True,
                )

            with gr.Row():
                with gr.Column(scale=3):
                    email_out = gr.HTML(value=email_html(play["email"]))
                    skip_btn  = gr.Button("⏭️  New Email", variant="secondary")

                with gr.Column(scale=2):
                    action_dd  = gr.Dropdown(
                        choices=PHASE_CHOICES[0],
                        label=PHASE_HINT[0],
                        value=None,
                    )
                    submit_btn = gr.Button("✅  Submit Action", variant="primary")
                    reward_out = gr.HTML()
                    bd_out     = gr.HTML()

            gr.HTML("""<div class="reward-guide">
              <b>Reward guide:</b> &nbsp;
              Correct classification <b style="color:#188038">+10</b> &nbsp;|&nbsp;
              Wrong <b style="color:#c5221f">-5</b> &nbsp;|&nbsp;
              Correct priority <b style="color:#188038">+5</b> &nbsp;|&nbsp;
              Wrong <b style="color:#c5221f">-3</b> &nbsp;|&nbsp;
              Flag urgent <b style="color:#188038">+8</b> &nbsp;|&nbsp;
              Miss urgent <b style="color:#c5221f">-10</b> &nbsp;|&nbsp;
              Archive spam <b style="color:#188038">+6</b> &nbsp;|&nbsp;
              Archive important <b style="color:#c5221f">-15</b>
            </div>""")

        # ── Tab 2: Train Agent ────────────────────────────────────────────────
        with gr.Tab("🤖 Train Agent"):
            gr.HTML("<p style='color:#64748b;margin:8px 0 14px'>Train a Q-Learning agent using multi-step episodes. More episodes = smarter decisions.</p>")
            with gr.Row():
                ep_slider = gr.Slider(100, 5000, value=1000, step=100, label="Training Episodes")
                train_btn = gr.Button("🚀  Start Training", variant="primary")
            train_out = gr.Textbox(label="Results & Q-Table", lines=20, interactive=False)

            gr.HTML("<hr style='border-color:#e2e8f0;margin:18px 0'>")
            gr.HTML("<p style='color:#64748b;margin-bottom:10px'>See what the trained agent recommends for the current email:</p>")
            suggest_btn = gr.Button("💡  Agent Suggestion", variant="secondary")
            suggest_out = gr.Textbox(label="Agent Plan", lines=14, interactive=False)

    # ── wiring ────────────────────────────────────────────────────────────────
    submit_btn.click(play_step,         [action_dd],   [email_out, action_dd, bd_out, reward_out, scoreboard])
    skip_btn.click(  play_reset,        [],            [email_out, action_dd, bd_out, reward_out, scoreboard])
    diff_radio.change(change_difficulty, [diff_radio], [email_out, action_dd, bd_out, reward_out, scoreboard])
    train_btn.click( do_train,          [ep_slider],   [train_out])
    suggest_btn.click(do_suggest,       [],            [suggest_out])

if __name__ == "__main__":
    demo.launch(css=CSS)
