# -*- coding: utf-8 -*-
"""
app.py — Gradio UI + OpenEnv REST API for Smart Email Triage RL Environment
"""
import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from environment import EmailEnv
from agent import QLearningAgent, email_to_state
from summarizer import summarize
from reply_generator import generate_reply
from inference import normalize_reward

# ── OpenEnv REST API ──────────────────────────────────────────────────────────
api_env = EmailEnv()
app = FastAPI()

class StepRequest(BaseModel):
    action: str

@app.post("/reset")
def api_reset():
    api_env.reset()
    state = api_env.state()
    state["valid_actions"] = api_env.valid_actions
    return JSONResponse(state)

@app.post("/step")
def api_step(body: StepRequest):
    # Auto-reset if env hasn't been initialised yet
    if api_env.current_email is None:
        api_env.reset()
    # If the action belongs to phase 0 but we're in a later phase, reset first
    phase0_actions = EmailEnv.PHASE_ACTIONS[0]
    if body.action in phase0_actions and api_env.current_phase != 0:
        api_env.reset()
    try:
        _email, raw_reward, done, info = api_env.step(body.action)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    bd = info.get("final_breakdown") or info["breakdown"]
    state = api_env.state()
    state["valid_actions"] = api_env.valid_actions
    return JSONResponse({
        "state":  state,
        "reward": normalize_reward(raw_reward),
        "done":   done,
        "info": {
            "raw_reward":     raw_reward,
            "classification": bd.classification,
            "priority":       bd.priority,
            "urgency":        bd.urgency,
            "archive":        bd.archive,
            "total":          bd.total(),
        },
    })

@app.get("/state")
def api_state():
    return JSONResponse(api_env.state())

# ── Gradio UI ─────────────────────────────────────────────────────────────────
ui_env = EmailEnv()
agent  = QLearningAgent(actions=EmailEnv.ACTION_SPACE)
play   = {"email": ui_env.reset(), "score": 0, "steps": 0, "ep_reward": 0}

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
EMPTY_BD = [["—", "—"]] * 5


def email_data(email):
    sm   = summarize(email)
    rep  = generate_reply(email)
    pi   = {"high": "🔴 HIGH", "medium": "🟡 MEDIUM", "low": "🟢 LOW"}.get(email.priority, email.priority.upper())
    li   = {"work": "💼 WORK", "personal": "👤 PERSONAL", "spam": "🚫 SPAM"}.get(email.label, email.label.upper())
    diff = {"easy": "🟢 Easy", "medium": "🟡 Medium", "hard": "🔴 Hard"}.get(email.difficulty, email.difficulty)
    return [
        ["✉ From",        email.sender],
        ["📌 Subject",    email.subject],
        ["📝 Body",       email.body],
        ["💡 Summary",    sm],
        ["💬 Reply",      rep],
        ["🏷 Label",      li],
        ["⚡ Priority",   pi],
        ["🎯 Difficulty", diff],
    ]

def scoreboard_data():
    avg = play["score"] / play["steps"] if play["steps"] else 0
    acc = max(0, min(100, 50 + avg * 2))
    return [[f"{play['score']:+d}", str(play["steps"]), f"{avg:+.1f}", f"{acc:.0f}%"]]

def breakdown_data(bd):
    def fmt(v): return f"+{v}" if v >= 0 else str(v)
    return [
        ["Classification", fmt(bd.classification)],
        ["Priority",       fmt(bd.priority)],
        ["Urgency",        fmt(bd.urgency)],
        ["Archive",        fmt(bd.archive)],
        ["TOTAL",          fmt(bd.total())],
    ]

def reward_md(reward):
    sign  = "+" if reward >= 0 else ""
    emoji = "✅" if reward > 0 else ("❌" if reward < 0 else "➖")
    return f"{emoji} Step reward: **{sign}{reward} pts**"


def play_reset():
    play["email"]     = ui_env.reset()
    play["ep_reward"] = 0
    phase   = ui_env.current_phase
    choices = PHASE_CHOICES[phase]
    return (
        email_data(play["email"]),
        gr.update(choices=choices, value=choices[0], label=PHASE_HINT[phase]),
        "*Submit an action to see your reward*",
        EMPTY_BD,
        scoreboard_data(),
    )

def play_step(action_label: str):
    if not action_label:
        phase   = ui_env.current_phase
        choices = PHASE_CHOICES[phase]
        return (
            email_data(play["email"]),
            gr.update(choices=choices, value=choices[0], label=PHASE_HINT[phase]),
            "*Pick an action first*",
            EMPTY_BD,
            scoreboard_data(),
        )
    action = L2A[action_label]
    _, reward, done, info = ui_env.step(action)
    play["ep_reward"] += reward
    bd = info.get("final_breakdown") or info["breakdown"]
    if done:
        play["score"]    += play["ep_reward"]
        play["steps"]    += 1
        play["email"]     = ui_env.reset()
        play["ep_reward"] = 0
    phase   = ui_env.current_phase
    choices = PHASE_CHOICES[phase]
    hint    = PHASE_HINT[phase]
    return (
        email_data(play["email"]),
        gr.update(choices=choices, value=choices[0], label=hint),
        reward_md(reward),
        breakdown_data(bd),
        scoreboard_data(),
    )

def change_difficulty(diff_label: str):
    ui_env.set_difficulty(DIFFICULTY_MAP.get(diff_label))
    play["email"] = ui_env.reset()
    play["score"] = play["steps"] = play["ep_reward"] = 0
    phase   = ui_env.current_phase
    choices = PHASE_CHOICES[phase]
    return (
        email_data(play["email"]),
        gr.update(choices=choices, value=choices[0], label=PHASE_HINT[phase]),
        "*Submit an action to see your reward*",
        EMPTY_BD,
        scoreboard_data(),
    )

def do_train(episodes: int):
    rewards = agent.train(ui_env, episodes=int(episodes))
    total = sum(rewards)
    avg   = total / len(rewards)
    wins  = sum(1 for r in rewards if r > 0)
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


CSS = """
*, *::before, *::after { box-sizing: border-box; }
html, body, .gradio-container, .main, .wrap, .gap, .block, .panel, .form, .container {
    background: #f0f4f8 !important; color: #1a202c !important;
    font-family: -apple-system, 'Segoe UI', Roboto, Arial, sans-serif !important;
}
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; padding: 0 20px !important; }
table, thead, tbody, tfoot, tr, th, td,
.table-wrap, [class*="svelte"] table,
[class*="svelte"] th, [class*="svelte"] td, [class*="svelte"] tr {
    background: #ffffff !important; color: #1a202c !important; border-color: #e2e8f0 !important;
}
th { background: #eef2f7 !important; font-weight: 700 !important; color: #374151 !important;
     font-size: 0.82rem !important; text-transform: uppercase !important;
     letter-spacing: 0.4px !important; padding: 10px 14px !important; }
td { padding: 9px 14px !important; font-size: 0.88rem !important; vertical-align: top !important;
     word-break: break-word !important; white-space: normal !important; }
tr:nth-child(even) td { background: #f8fafc !important; }
td:nth-child(2), th:nth-child(2) { width: 100% !important; min-width: 0 !important; }
input, select, textarea {
    color: #1a202c !important; background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important; border-radius: 10px !important;
    font-size: 0.92rem !important; padding: 10px 14px !important;
}
input:focus, select:focus, textarea:focus {
    border-color: #1a73e8 !important; outline: none !important;
    box-shadow: 0 0 0 3px rgba(26,115,232,0.12) !important;
}
textarea { font-family: 'Courier New', monospace !important; font-size: 0.84rem !important; }
label, label span, .block label span {
    color: #4a5568 !important; font-size: 0.83rem !important; font-weight: 600 !important;
    letter-spacing: 0.3px !important; text-transform: uppercase !important; background: transparent !important;
}
fieldset label span { text-transform: none !important; font-size: 0.9rem !important;
                      font-weight: 500 !important; color: #2d3748 !important; }
button { font-size: 0.92rem !important; font-weight: 600 !important; border-radius: 10px !important; padding: 10px 20px !important; }
button.primary, [data-testid="primary-btn"] {
    background: #1a73e8 !important; color: #fff !important; border: none !important;
    box-shadow: 0 2px 8px rgba(26,115,232,0.30) !important;
}
button.primary:hover { background: #1557b0 !important; }
button.secondary, [data-testid="secondary-btn"] {
    background: #fff !important; color: #1a73e8 !important; border: 1.5px solid #1a73e8 !important;
}
button.secondary:hover { background: #e8f0fe !important; }
.tab-nav button, button[role="tab"] {
    color: #718096 !important; background: transparent !important;
    border-bottom: 3px solid transparent !important; font-weight: 600 !important;
    font-size: 0.92rem !important; padding: 10px 18px !important; text-transform: none !important;
}
.tab-nav button.selected, button[role="tab"][aria-selected="true"] {
    color: #1a73e8 !important; border-bottom: 3px solid #1a73e8 !important;
}
input[type=range] { accent-color: #1a73e8 !important; }
"""

with gr.Blocks(title="Email Triage RL", theme=gr.themes.Soft(), css=CSS) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:24px 0 12px;">
      <h1 style="font-size:1.9rem;font-weight:700;margin:0 0 6px;color:#1a73e8;">📧 Smart Email Triage RL</h1>
      <p style="color:#718096;margin:0 0 12px;font-size:0.92rem;">
        Classify emails · set priorities · earn rewards
      </p>
      <div style="display:inline-flex;flex-wrap:wrap;justify-content:center;gap:6px 16px;
                  background:#fff;border:1.5px solid #e2e8f0;border-radius:12px;
                  padding:8px 22px;font-size:0.84rem;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
        <span style="color:#4a5568;">1. <b style="color:#1a202c;">Classify</b></span>
        <span style="color:#cbd5e0;">|</span>
        <span style="color:#4a5568;">2. <b style="color:#1a202c;">Priority</b></span>
        <span style="color:#cbd5e0;">|</span>
        <span style="color:#4a5568;">3. <b style="color:#1a202c;">Flag / Archive / Done</b></span>
        <span style="color:#cbd5e0;">|</span>
        <span style="color:#4a5568;">Earn <b style="color:#1a73e8;">Rewards 🏆</b></span>
      </div>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("▶ Play"):
            scoreboard = gr.Dataframe(
                value=scoreboard_data(),
                headers=["🏆 Score", "💌 Emails", "📈 Avg Reward", "🎯 Accuracy"],
                interactive=False, wrap=True,
            )
            diff_radio = gr.Radio(
                choices=DIFFICULTY_LABELS, value=DIFFICULTY_LABELS[0],
                label="Difficulty Level", interactive=True,
            )
            with gr.Row():
                with gr.Column(scale=3):
                    email_out = gr.Dataframe(
                        value=email_data(play["email"]),
                        headers=["Field", "Value"],
                        interactive=False, wrap=True,
                    )
                with gr.Column(scale=2):
                    action_dd = gr.Dropdown(
                        choices=PHASE_CHOICES[0], label=PHASE_HINT[0],
                        value=PHASE_CHOICES[0][0], interactive=True,
                    )
                    with gr.Row():
                        skip_btn   = gr.Button("⏭ Next Email",    variant="secondary", scale=1)
                        submit_btn = gr.Button("✅ Submit Action", variant="primary",   scale=2)
                    reward_out = gr.Markdown(value="*Submit an action to see your reward*")
                    bd_out = gr.Dataframe(
                        value=EMPTY_BD,
                        headers=["Category", "Score"],
                        interactive=False, wrap=True,
                    )
            gr.HTML("""
            <div style="background:#fff;border:1.5px solid #e2e8f0;border-radius:12px;
                        padding:16px 22px;margin-top:18px;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
              <div style="font-weight:700;font-size:0.8rem;color:#4a5568;text-transform:uppercase;
                          letter-spacing:0.5px;margin-bottom:12px;">🏆 Reward Guide</div>
              <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;">
                <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:8px 12px;display:flex;justify-content:space-between;">
                  <span style="font-size:0.8rem;color:#374151;">✅ Correct label</span><b style="color:#15803d;">+10</b></div>
                <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:8px 12px;display:flex;justify-content:space-between;">
                  <span style="font-size:0.8rem;color:#374151;">❌ Wrong label</span><b style="color:#dc2626;">-5</b></div>
                <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:8px 12px;display:flex;justify-content:space-between;">
                  <span style="font-size:0.8rem;color:#374151;">✅ Correct priority</span><b style="color:#15803d;">+5</b></div>
                <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:8px 12px;display:flex;justify-content:space-between;">
                  <span style="font-size:0.8rem;color:#374151;">❌ Wrong priority</span><b style="color:#dc2626;">-3</b></div>
                <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:8px 12px;display:flex;justify-content:space-between;">
                  <span style="font-size:0.8rem;color:#374151;">✅ Flag urgent</span><b style="color:#15803d;">+8</b></div>
                <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:8px 12px;display:flex;justify-content:space-between;">
                  <span style="font-size:0.8rem;color:#374151;">❌ Miss urgent</span><b style="color:#dc2626;">-10</b></div>
                <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:8px 12px;display:flex;justify-content:space-between;">
                  <span style="font-size:0.8rem;color:#374151;">✅ Archive spam</span><b style="color:#15803d;">+6</b></div>
                <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:8px 12px;display:flex;justify-content:space-between;">
                  <span style="font-size:0.8rem;color:#374151;">❌ Archive non-spam</span><b style="color:#dc2626;">-15</b></div>
              </div>
            </div>
            """)

        with gr.Tab("🤖 Train Agent"):
            gr.HTML("<p style='color:#5f6368;margin:10px 0 16px;font-size:0.9rem;'>Train a Q-Learning agent. More episodes = smarter decisions.</p>")
            with gr.Row():
                ep_slider = gr.Slider(100, 5000, value=1000, step=100, label="Training Episodes")
                train_btn = gr.Button("🚀 Start Training", variant="primary")
            train_out = gr.Textbox(label="Results & Q-Table", lines=18, interactive=False)
            gr.HTML("<hr style='border:none;border-top:1px solid #e2e8f0;margin:20px 0;'>")
            gr.HTML("<p style='color:#5f6368;margin-bottom:12px;font-size:0.9rem;'>See what the trained agent recommends for the current email:</p>")
            suggest_btn = gr.Button("💡 Agent Suggestion", variant="secondary")
            suggest_out = gr.Textbox(label="Agent Plan", lines=12, interactive=False)

    submit_btn.click(play_step,           [action_dd],  [email_out, action_dd, reward_out, bd_out, scoreboard])
    skip_btn.click(  play_reset,          [],           [email_out, action_dd, reward_out, bd_out, scoreboard])
    diff_radio.change(change_difficulty,  [diff_radio], [email_out, action_dd, reward_out, bd_out, scoreboard])
    train_btn.click( do_train,            [ep_slider],  [train_out])
    suggest_btn.click(do_suggest,         [],           [suggest_out])

# Mount Gradio onto FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
