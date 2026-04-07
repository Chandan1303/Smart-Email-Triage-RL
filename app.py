# -*- coding: utf-8 -*-
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

    card  = '<div style="background:#ffffff;border:1px solid #dadce0;border-radius:12px;padding:20px 24px;font-size:0.92rem;line-height:1.7;box-shadow:0 1px 4px rgba(60,64,67,0.10);color:#202124;">'
    row   = 'display:flex;align-items:flex-start;gap:14px;padding:8px 0;border-bottom:1px solid #f1f3f4;'
    key   = 'style="color:#5f6368;min-width:105px;font-weight:500;font-size:0.83rem;flex-shrink:0;"'
    val   = 'style="color:#202124;font-weight:400;flex:1;word-break:break-word;"'

    rows = (
        f'<div style="{row}"><span {key}>&#9993; From</span><span {val}>{email.sender}</span></div>'
        f'<div style="{row}"><span {key}>&#128204; Subject</span><span {val}>{email.subject}</span></div>'
        f'<div style="{row}"><span {key}>&#128221; Body</span><span {val}>{email.body}</span></div>'
        f'<div style="{row}"><span {key}>&#128161; Summary</span><span style="color:#1a73e8;font-style:italic;flex:1;">{sm}</span></div>'
        f'<div style="{row}"><span {key}>&#128172; Reply</span><span style="color:#188038;font-style:italic;flex:1;">{rep}</span></div>'
        f'<div style="{row}"><span {key}>{li} Label</span><span {val}><span style="background:#e8f0fe;color:#1a73e8;padding:2px 12px;border-radius:100px;font-size:0.75rem;font-weight:600;">{email.label.upper()}</span></span></div>'
        f'<div style="{row}"><span {key}>{pi} Priority</span><span {val}><span style="background:#fef7e0;color:#b06000;padding:2px 12px;border-radius:100px;font-size:0.75rem;font-weight:600;">{email.priority.upper()}</span></span></div>'
        f'<div style="{row}border-bottom:none;"><span {key}>&#127919; Difficulty</span><span {val}>{diff_badge}</span></div>'
    )
    return card + rows + '</div>'

def scoreboard_md():
    avg = play["score"] / play["steps"] if play["steps"] else 0
    acc = max(0, min(100, 50 + avg * 2))
    s = 'display:inline-block;margin:0 16px;color:#ffffff;font-size:0.95rem;'
    v = 'font-weight:700;font-size:1.05rem;color:#ffffff;'
    inner = (
        f'<span style="{s}">&#127942; Score <span style="{v}">{play["score"]:+d}</span></span>'
        f'<span style="{s}">&#128140; Emails <span style="{v}">{play["steps"]}</span></span>'
        f'<span style="{s}">&#128200; Avg Reward <span style="{v}">{avg:+.1f}</span></span>'
        f'<span style="{s}">&#127919; Accuracy <span style="{v}">{acc:.0f}%</span></span>'
    )
    return (
        '<div style="background:#1a73e8;border-radius:12px;padding:14px 28px;'
        'text-align:center;box-shadow:0 2px 8px rgba(26,115,232,0.30);margin-bottom:16px;">'
        + inner + '</div>'
    )

def reward_html(reward):
    if reward > 0:
        bg, color, border = "#e6f4ea", "#188038", "#a8d5b5"
        sign = "+"
    elif reward < 0:
        bg, color, border = "#fce8e6", "#c5221f", "#f5b8b5"
        sign = ""
    else:
        bg, color, border = "#f1f3f4", "#5f6368", "#dadce0"
        sign = "+"
    return (
        f'<div style="display:inline-flex;align-items:center;padding:10px 24px;'
        f'border-radius:100px;font-size:1.05rem;font-weight:700;margin:8px 0;'
        f'background:{bg};color:{color};border:1.5px solid {border};">'
        f'{sign}{reward} pts</div>'
    )


# ── play callbacks ────────────────────────────────────────────────────────────
def play_reset():
    play["email"]     = env.reset()
    play["ep_reward"] = 0
    phase = env.current_phase
    choices = PHASE_CHOICES[phase]
    return (
        email_html(play["email"]),
        gr.update(choices=choices, value=choices[0], label=PHASE_HINT[phase]),
        '<div style="height:4px;"></div>',
        '<div style="height:4px;"></div>',
        scoreboard_md(),
    )

def play_step(action_label: str):
    if not action_label:
        phase = env.current_phase
        choices = PHASE_CHOICES[phase]
        return (
            email_html(play["email"]),
            gr.update(choices=choices, value=choices[0], label=PHASE_HINT[phase]),
            '<div style="height:4px;"></div>',
            '<div style="display:inline-flex;padding:10px 24px;border-radius:100px;'
            'background:#f1f3f4;color:#5f6368;border:1.5px solid #dadce0;font-weight:600;">&#9888; Pick an action first</div>',
            scoreboard_md(),
        )

    action = L2A[action_label]
    _, reward, done, info = env.step(action)
    play["ep_reward"] += reward

    bd = info.get("final_breakdown") or info["breakdown"]

    def score_color(v):
        return "#188038" if v >= 0 else "#c5221f"

    def bd_row(label, value, border=True):
        border_style = "border-bottom:1px solid #e0e0e0;" if border else ""
        return (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:7px 0;{border_style}">'
            f'<span style="color:#202124;font-size:0.9rem;">{label}</span>'
            f'<span style="font-weight:700;font-size:0.9rem;color:{score_color(value)};">{value:+d}</span>'
            f'</div>'
        )

    bd_html = (
        '<div style="background:#ffffff;border:2px solid #dadce0;border-radius:10px;'
        'padding:14px 18px;margin-top:8px;color:#202124;">'
        + bd_row("Classification", bd.classification)
        + bd_row("Priority", bd.priority)
        + bd_row("Urgency", bd.urgency)
        + bd_row("Archive", bd.archive)
        + f'<div style="display:flex;justify-content:space-between;align-items:center;'
          f'padding:8px 0 2px;border-top:2px solid #dadce0;margin-top:4px;">'
          f'<span style="color:#202124;font-weight:700;font-size:0.95rem;">Total</span>'
          f'<span style="font-weight:700;font-size:0.95rem;color:{score_color(bd.total())};">{bd.total():+d}</span>'
          f'</div>'
        + '</div>'
    )

    if done:
        play["score"]    += play["ep_reward"]
        play["steps"]    += 1
        play["email"]     = env.reset()
        play["ep_reward"] = 0

    phase = env.current_phase
    choices = PHASE_CHOICES.get(phase, PHASE_CHOICES[0])
    return (
        email_html(play["email"]),
        gr.update(choices=choices, value=choices[0], label=PHASE_HINT.get(phase, PHASE_HINT[0])),
        bd_html,
        reward_html(reward),
        scoreboard_md(),
    )

def change_difficulty(diff_label: str):
    difficulty = DIFFICULTY_MAP.get(diff_label)
    env.set_difficulty(difficulty)
    play["email"]     = env.reset()
    play["score"]     = 0
    play["steps"]     = 0
    play["ep_reward"] = 0
    phase = env.current_phase
    choices = PHASE_CHOICES[phase]
    return (
        email_html(play["email"]),
        gr.update(choices=choices, value=choices[0], label=PHASE_HINT[phase]),
        '<div style="height:4px;"></div>',
        '<div style="height:4px;"></div>',
        scoreboard_md(),
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


# ── CSS — clean, polished, responsive ────────────────────────────────────────
CSS = """
*, *::before, *::after { box-sizing: border-box; }

/* ── Base: force light mode ── */
html, body,
.gradio-container, .main, .wrap, .gap,
.block, .panel, .form, .container,
[data-testid], .svelte-1gfkn6j {
    background: #f0f4f8 !important;
    color: #1a202c !important;
    font-family: -apple-system, 'Segoe UI', Roboto, Arial, sans-serif !important;
}
.gradio-container {
    max-width: 1080px !important;
    margin: 0 auto !important;
    padding: 0 16px !important;
}

/* ── Play grid: side-by-side on desktop ── */
.play-grid {
    display: grid;
    grid-template-columns: 1fr 360px;
    gap: 16px;
    align-items: start;
    margin-top: 12px;
}
@media (max-width: 780px) {
    .play-grid { grid-template-columns: 1fr; }
    .gradio-container { padding: 0 8px !important; }
}

/* ── Action panel ── */
.action-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 18px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
}

/* ── Labels ── */
label, label span, .block label span {
    color: #4a5568 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    text-transform: uppercase !important;
    background: transparent !important;
}

/* ── Inputs / Dropdowns ── */
input, select, textarea {
    color: #1a202c !important;
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    padding: 10px 14px !important;
    transition: border-color 0.15s !important;
}
input:focus, select:focus, textarea:focus {
    border-color: #1a73e8 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(26,115,232,0.12) !important;
}
textarea {
    font-family: 'Courier New', monospace !important;
    font-size: 0.84rem !important;
    line-height: 1.65 !important;
}

/* ── Buttons ── */
button {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    transition: all 0.15s !important;
    letter-spacing: 0.2px !important;
}
button.primary, [data-testid="primary-btn"] {
    background: #1a73e8 !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(26,115,232,0.30) !important;
}
button.primary:hover, [data-testid="primary-btn"]:hover {
    background: #1557b0 !important;
    box-shadow: 0 4px 14px rgba(26,115,232,0.40) !important;
}
button.secondary, [data-testid="secondary-btn"] {
    background: #ffffff !important;
    color: #1a73e8 !important;
    border: 1.5px solid #1a73e8 !important;
}
button.secondary:hover {
    background: #e8f0fe !important;
}

/* ── Radio buttons ── */
fieldset label span {
    color: #2d3748 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    text-transform: none !important;
}

/* ── Tabs ── */
.tab-nav button, button[role="tab"] {
    color: #718096 !important;
    background: transparent !important;
    border-bottom: 3px solid transparent !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 10px 18px !important;
    text-transform: none !important;
}
.tab-nav button.selected, button[role="tab"][aria-selected="true"] {
    color: #1a73e8 !important;
    border-bottom: 3px solid #1a73e8 !important;
}

/* ── Slider ── */
input[type=range] { accent-color: #1a73e8 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f0f4f8; }
::-webkit-scrollbar-thumb { background: #cbd5e0; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #a0aec0; }
"""
# ── layout ────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Email Triage RL", theme=gr.themes.Default(), css=CSS) as demo:

    # Header
    gr.HTML("""
    <div style="text-align:center;padding:28px 0 12px;">
      <h1 style="font-size:2rem;font-weight:700;margin:0 0 6px;color:#1a73e8;letter-spacing:-0.5px;">
        Smart Email Triage RL
      </h1>
      <p style="color:#718096;margin:0 0 14px;font-size:0.95rem;font-weight:400;">
        Reinforcement learning environment &mdash; classify emails, set priorities, earn rewards
      </p>
      <div style="display:inline-flex;flex-wrap:wrap;justify-content:center;gap:6px 14px;
                  background:#ffffff;border:1.5px solid #e2e8f0;border-radius:12px;
                  padding:10px 22px;font-size:0.84rem;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
        <span style="color:#4a5568;font-weight:500;">1. <b style="color:#1a202c;">Classify</b> the email</span>
        <span style="color:#cbd5e0;">|</span>
        <span style="color:#4a5568;font-weight:500;">2. Set <b style="color:#1a202c;">Priority</b></span>
        <span style="color:#cbd5e0;">|</span>
        <span style="color:#4a5568;font-weight:500;">3. <b style="color:#1a202c;">Flag / Archive / Done</b></span>
        <span style="color:#cbd5e0;">|</span>
        <span style="color:#4a5568;font-weight:500;">Earn <b style="color:#1a73e8;">Rewards</b></span>
      </div>
    </div>
    """)

    # Scoreboard
    scoreboard = gr.HTML(value=scoreboard_md())

    with gr.Tabs():

        # ── Tab 1: Play ───────────────────────────────────────────────────────
        with gr.Tab("Play"):

            # Difficulty selector
            diff_radio = gr.Radio(
                choices=DIFFICULTY_LABELS,
                value=DIFFICULTY_LABELS[0],
                label="Difficulty Level",
                interactive=True,
            )

            # Desktop: 2-col grid via CSS; Mobile: stacks automatically
            with gr.Row():
                # Left column — email card
                with gr.Column(scale=3):
                    email_out = gr.HTML(value=email_html(play["email"]))

                # Right column — action panel
                with gr.Column(scale=2):
                action_dd = gr.Dropdown(
                    choices=PHASE_CHOICES[0],
                    label=PHASE_HINT[0],
                    value=PHASE_CHOICES[0][0],
                    interactive=True,
                )
                with gr.Row():
                    skip_btn   = gr.Button("Next Email",    variant="secondary", scale=1)
                    submit_btn = gr.Button("Submit Action", variant="primary",   scale=2)
                reward_out = gr.HTML(value='<div style="min-height:20px;color:#a0aec0;font-size:0.85rem;text-align:center;padding:8px 0;">Submit an action to see your reward</div>')
                bd_out     = gr.HTML(value='<div style="min-height:20px;color:#a0aec0;font-size:0.85rem;text-align:center;padding:4px 0;">Score breakdown will appear here</div>')

            # Reward guide
            gr.HTML("""
            <div style="background:#ffffff;border:1.5px solid #e2e8f0;border-radius:12px;
                        padding:14px 20px;margin-top:16px;font-size:0.82rem;
                        box-shadow:0 1px 4px rgba(0,0,0,0.05);">
              <div style="color:#4a5568;font-weight:700;font-size:0.8rem;text-transform:uppercase;
                          letter-spacing:0.5px;margin-bottom:8px;">Reward Guide</div>
              <div style="display:flex;flex-wrap:wrap;gap:4px 0;line-height:2;">
                <span style="color:#718096;">Correct label </span><b style="color:#188038;margin-right:12px;">+10</b>
                <span style="color:#718096;">Wrong </span><b style="color:#c5221f;margin-right:12px;">-5</b>
                <span style="color:#718096;">Correct priority </span><b style="color:#188038;margin-right:12px;">+5</b>
                <span style="color:#718096;">Wrong </span><b style="color:#c5221f;margin-right:12px;">-3</b>
                <span style="color:#718096;">Flag urgent </span><b style="color:#188038;margin-right:12px;">+8</b>
                <span style="color:#718096;">Miss urgent </span><b style="color:#c5221f;margin-right:12px;">-10</b>
                <span style="color:#718096;">Archive spam </span><b style="color:#188038;margin-right:12px;">+6</b>
                <span style="color:#718096;">Archive non-spam </span><b style="color:#c5221f;">-15</b>
              </div>
            </div>
            """)

        # ── Tab 2: Train Agent ────────────────────────────────────────────────
        with gr.Tab("Train Agent"):
            gr.HTML("<p style='color:#5f6368;margin:8px 0 14px;font-size:0.9rem;'>Train a Q-Learning agent. More episodes = smarter decisions.</p>")
            with gr.Row():
                ep_slider = gr.Slider(100, 5000, value=1000, step=100, label="Training Episodes")
                train_btn = gr.Button("Start Training", variant="primary")
            train_out = gr.Textbox(label="Results & Q-Table", lines=18, interactive=False)
            gr.HTML("<hr style='border:none;border-top:1px solid #dadce0;margin:18px 0;'>")
            gr.HTML("<p style='color:#5f6368;margin-bottom:10px;font-size:0.9rem;'>See what the trained agent recommends for the current email:</p>")
            suggest_btn = gr.Button("Agent Suggestion", variant="secondary")
            suggest_out = gr.Textbox(label="Agent Plan", lines=12, interactive=False)

    # ── wiring ────────────────────────────────────────────────────────────────
    submit_btn.click(play_step,          [action_dd],   [email_out, action_dd, bd_out, reward_out, scoreboard])
    skip_btn.click(  play_reset,         [],            [email_out, action_dd, bd_out, reward_out, scoreboard])
    diff_radio.change(change_difficulty, [diff_radio],  [email_out, action_dd, bd_out, reward_out, scoreboard])
    train_btn.click( do_train,           [ep_slider],   [train_out])
    suggest_btn.click(do_suggest,        [],            [suggest_out])

if __name__ == "__main__":
    demo.launch()
