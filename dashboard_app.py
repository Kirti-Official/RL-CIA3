import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from traffic_env import TrafficSignalEnv

ROOT = Path(__file__).resolve().parent
APPROACHES = {
    "Q-Learning": ROOT / "04_RL_Implementations/Approach_1_q_learning",
    "SARSA": ROOT / "04_RL_Implementations/Approach_2_sarsa",
    "Expected SARSA": ROOT / "04_RL_Implementations/Approach_3_expected_sarsa",
    "Double Q-Learning": ROOT / "04_RL_Implementations/Approach_4_double_q_learning",
}
METRICS_FILE = ROOT / "06_Performance_Comparison/comparison_metrics.json"
PHASE_NAMES = {0: "NS-Green", 1: "EW-Green"}
ACTION_NAMES = {0: "Keep phase", 1: "Switch phase"}
BIN_NAMES = ["Low", "Medium", "High"]

st.set_page_config(page_title="Smart Traffic Signal RL Dashboard", page_icon="🚦", layout="wide")

# ----------------------------- Styling -----------------------------
st.markdown(
    """
    <style>
    .hero {
        padding: 1.1rem 1.3rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid #374151;
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; }
    .hero p { margin: .35rem 0 0; color: #cbd5e1; }
    .state-box {
        border: 1px solid #334155; border-radius: 12px; padding: 12px;
        background: #111827; min-height: 88px;
    }
    .small-muted { color: #94a3b8; font-size: .82rem; }
    .decision {
        border: 1px solid #334155; border-radius: 14px; padding: 16px;
        background: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>🚦 Smart Traffic Signal Optimization</h1>'
    '<p>Interactive RL control center for the CIA-3 micro project • Q-Learning • SARSA • Expected SARSA • Double Q-Learning</p></div>',
    unsafe_allow_html=True,
)


def load_metrics():
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def load_rewards(name):
    with open(APPROACHES[name] / "episode_rewards.json", "r", encoding="utf-8") as f:
        return np.asarray(json.load(f), dtype=float)


def load_q(name):
    return np.load(APPROACHES[name] / "Q_table.npy")


def queue_bin(q):
    return 0 if q <= 5 else 1 if q <= 15 else 2


def state_details(env):
    ns = env.q_north + env.q_south
    ew = env.q_east + env.q_west
    return ns, ew, BIN_NAMES[queue_bin(ns)], BIN_NAMES[queue_bin(ew)]


def reset_sim(algo):
    st.session_state.sim = TrafficSignalEnv(
        episode_len=200, arrival_rate=0.35, saturation_flow=2, switch_penalty=3.0, seed=42
    )
    st.session_state.state = st.session_state.sim.reset()
    st.session_state.last_reward = 0.0
    st.session_state.last_action = None
    st.session_state.last_switched = False
    st.session_state.history = []
    st.session_state.sim_algo = algo


def intersection_figure(env, action=None):
    """Draw a simple live intersection with queue markers and traffic lights."""
    fig = go.Figure()
    road = "#3f4650"
    lane = "#d1d5db"

    # Roads
    fig.add_shape(type="rect", x0=-1.55, x1=1.55, y0=-7, y1=7, fillcolor=road, line_width=0)
    fig.add_shape(type="rect", x0=-7, x1=7, y0=-1.55, y1=1.55, fillcolor=road, line_width=0)
    # Lane separators
    fig.add_shape(type="line", x0=0, x1=0, y0=-7, y1=7, line=dict(color=lane, width=1, dash="dash"))
    fig.add_shape(type="line", x0=-7, x1=7, y0=0, y1=0, line=dict(color=lane, width=1, dash="dash"))

    # Stop lines
    fig.add_shape(type="line", x0=-1.55, x1=1.55, y0=2.0, y1=2.0, line=dict(color="white", width=3))
    fig.add_shape(type="line", x0=-1.55, x1=1.55, y0=-2.0, y1=-2.0, line=dict(color="white", width=3))
    fig.add_shape(type="line", x0=2.0, x1=2.0, y0=-1.55, y1=1.55, line=dict(color="white", width=3))
    fig.add_shape(type="line", x0=-2.0, x1=-2.0, y0=-1.55, y1=1.55, line=dict(color="white", width=3))

    # Center label
    phase = env.phase
    phase_text = "NS GREEN" if phase == 0 else "EW GREEN"
    phase_color = "#22c55e" if phase == 0 else "#22c55e"
    fig.add_annotation(x=0, y=0, text=f"🚦<b>{phase_text}</b>", showarrow=False,
                       font=dict(size=18, color=phase_color))

    # Queue vehicle dots. They are a visualization of queue magnitude, not individual simulated vehicle IDs.
    queues = [
        ("N", env.q_north, 0, 2.7, 0, 1),
        ("S", env.q_south, 0, -2.7, 0, -1),
        ("W", env.q_west, -2.7, 0, -1, 0),
        ("E", env.q_east, 2.7, 0, 1, 0),
    ]
    for label, count, cx, cy, dx, dy in queues:
        shown = min(int(count), 12)
        xs, ys = [], []
        for i in range(shown):
            offset = (i // 3) * 0.55
            across = (i % 3 - 1) * 0.42
            if label == "N":
                xs.append(across); ys.append(cy + offset)
            elif label == "S":
                xs.append(across); ys.append(cy - offset)
            elif label == "W":
                xs.append(cx - offset); ys.append(across)
            else:
                xs.append(cx + offset); ys.append(across)
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=f"{label} queue",
                                     marker=dict(size=12, symbol="square", color="#60a5fa"),
                                     hovertemplate=f"{label} queue: {count} vehicles<extra></extra>"))
        fig.add_annotation(x=cx, y=cy + (1.1 if label in ["N", "W"] else -1.1),
                           text=f"<b>{label}: {count}</b>", showarrow=False, font=dict(size=14))

    # Signal heads: each direction's signal color is derived from the current phase.
    ns_green = phase == 0
    ew_green = phase == 1
    for x, y, green, label in [
        (-2.7, 2.7, ns_green, "N"), (2.7, -2.7, ns_green, "S"),
        (-2.7, -2.7, ew_green, "W"), (2.7, 2.7, ew_green, "E"),
    ]:
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=["●"],
                                 textposition="middle center", marker=dict(size=26, color="#22c55e" if green else "#ef4444"),
                                 name=f"{label} signal", hoverinfo="skip", showlegend=False))

    fig.update_xaxes(visible=False, range=[-7, 7])
    fig.update_yaxes(visible=False, range=[-7, 7], scaleanchor="x", scaleratio=1)
    fig.update_layout(height=500, margin=dict(l=5, r=5, t=5, b=5), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def learning_figure(selected):
    fig = go.Figure()
    for name in selected:
        rewards = load_rewards(name)
        window = 20
        if len(rewards) >= window:
            y = np.convolve(rewards, np.ones(window) / window, mode="valid")
            x = np.arange(window, len(rewards) + 1)
        else:
            y = rewards
            x = np.arange(1, len(rewards) + 1)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    fig.update_layout(title="Training reward — 20-episode rolling mean", xaxis_title="Episode",
                      yaxis_title="Total reward", hovermode="x unified", height=430,
                      legend=dict(orientation="h", y=1.08))
    return fig


def queue_history_figure(history):
    if not history:
        return None
    h = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["step"], y=h["ns"], mode="lines+markers", name="NS queue"))
    fig.add_trace(go.Scatter(x=h["step"], y=h["ew"], mode="lines+markers", name="EW queue"))
    fig.add_trace(go.Scatter(x=h["step"], y=h["total"], mode="lines", name="Total queue", line=dict(width=4)))
    fig.update_layout(title="Live queue trajectory", xaxis_title="Simulation step",
                      yaxis_title="Vehicles", height=350, hovermode="x unified")
    return fig


def policy_table(name):
    q = load_q(name)
    rows = []
    idx = 0
    for ns_bin in range(3):
        for ew_bin in range(3):
            for phase in range(2):
                rows.append({
                    "NS queue": BIN_NAMES[ns_bin],
                    "EW queue": BIN_NAMES[ew_bin],
                    "Current phase": PHASE_NAMES[phase],
                    "Best action": ACTION_NAMES[int(np.argmax(q[idx]))],
                    "Keep Q": round(float(q[idx, 0]), 3),
                    "Switch Q": round(float(q[idx, 1]), 3),
                })
                idx += 1
    return pd.DataFrame(rows)


metrics = load_metrics()
# Normalize algorithm labels from the saved metrics so they match the dashboard labels.
metrics["name"] = metrics["name"].replace({"Q-learning": "Q-Learning", "Double Q-learning": "Double Q-Learning"})

# ----------------------------- Sidebar -----------------------------
st.sidebar.header("🎛️ Dashboard Controls")
selected = st.sidebar.multiselect("Algorithms for comparison", list(APPROACHES), default=list(APPROACHES))
if not selected:
    st.warning("Select at least one algorithm in the sidebar.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("**Environment**")
st.sidebar.write("18 discrete states")
st.sidebar.write("2 actions: Keep / Switch")
st.sidebar.write("Poisson arrivals: λ = 0.35")
st.sidebar.write("Saturation flow: 2 vehicles")
st.sidebar.write("Switch penalty: 3.0")
st.sidebar.write("Episode length: 200 steps")

# ----------------------------- KPI row -----------------------------
best_reward = metrics.loc[metrics["Average Reward (last 50 ep)"].idxmax()]
best_conv = metrics.loc[metrics["Convergence Speed (episodes)"].idxmin()]
best_stability = metrics.loc[metrics["Stability (std dev, last 50 ep)"].idxmin()]
best_success = metrics.loc[metrics["Success Rate (last 50 ep)"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
c1.metric("🏆 Best average reward", best_reward["name"], f'{best_reward["Average Reward (last 50 ep)"]:.2f}')
c2.metric("⚡ Fastest convergence", best_conv["name"], f'{int(best_conv["Convergence Speed (episodes)"])} episodes')
c3.metric("📉 Lowest reward variance", best_stability["name"], f'{best_stability["Stability (std dev, last 50 ep)"]:.2f}')
c4.metric("✅ Highest success rate", best_success["name"], f'{best_success["Success Rate (last 50 ep)"]:.0f}%')

st.divider()

# ----------------------------- Live simulation -----------------------------
st.header("🚦 Live RL Simulation")
sim_algo = st.selectbox("Choose trained policy", list(APPROACHES), index=3)
if "sim" not in st.session_state or st.session_state.get("sim_algo") != sim_algo:
    reset_sim(sim_algo)

env = st.session_state.sim
q = load_q(sim_algo)
ns_q, ew_q, ns_bin, ew_bin = state_details(env)
state = st.session_state.state
current_phase = PHASE_NAMES[env.phase]
recommended_action = int(np.argmax(q[state]))

b1, b2, b3, b4 = st.columns(4)
b1.metric("North + South queue", f"{ns_q} vehicles", ns_bin)
b2.metric("East + West queue", f"{ew_q} vehicles", ew_bin)
b3.metric("Current phase", current_phase)
b4.metric("Simulation step", f"{env.steps_taken} / {env.episode_len}")

left, right = st.columns([1.35, 1])
with left:
    st.plotly_chart(intersection_figure(env, recommended_action), use_container_width=True, config={"displayModeBar": False})
    st.caption("Blue squares visualize queued vehicles. Green/red signal heads show the active phase. Queue markers are capped visually for readability; the numeric queue counts are exact.")

with right:
    st.markdown('<div class="decision">', unsafe_allow_html=True)
    st.subheader("🤖 Agent decision")
    st.write(f"**Algorithm:** {sim_algo}")
    st.write(f"**State:** NS={ns_bin}, EW={ew_bin}, phase={current_phase}")
    st.write(f"**Recommended action:** {'🟢 ' if recommended_action == 0 else '🟠 '}{ACTION_NAMES[recommended_action]}")
    st.write(f"**Last reward:** {st.session_state.last_reward:.2f}")
    st.write(f"**Current total queue:** {ns_q + ew_q} vehicles")
    if st.session_state.last_action is not None:
        st.write(f"**Previous action:** {ACTION_NAMES[st.session_state.last_action]}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🔄 State → Action → Reward → Next State")
    st.info(
        "The trained policy observes the discretized queue state and current phase, chooses Keep or Switch, "
        "then the environment generates stochastic arrivals, updates queues, and returns the reward."
    )

    a, b, c = st.columns(3)
    with a:
        if st.button("▶ Next step", use_container_width=True, type="primary"):
            action = int(np.argmax(q[st.session_state.state]))
            old_phase = env.phase
            next_state, reward, done, info = env.step(action)
            st.session_state.last_action = action
            st.session_state.last_reward = reward
            st.session_state.last_switched = old_phase != env.phase
            st.session_state.state = next_state
            st.session_state.history.append({
                "step": env.steps_taken,
                "ns": env.q_north + env.q_south,
                "ew": env.q_east + env.q_west,
                "total": info["total_queue"],
            })
            if done:
                st.toast("Episode reached 200 steps. Reset to start another run.")
            st.rerun()
    with b:
        if st.button("⏩ Run 20 steps", use_container_width=True):
            for _ in range(20):
                if env.steps_taken >= env.episode_len:
                    break
                action = int(np.argmax(q[st.session_state.state]))
                old_phase = env.phase
                next_state, reward, done, info = env.step(action)
                st.session_state.last_action = action
                st.session_state.last_reward = reward
                st.session_state.last_switched = old_phase != env.phase
                st.session_state.state = next_state
                st.session_state.history.append({"step": env.steps_taken, "ns": env.q_north + env.q_south,
                                                  "ew": env.q_east + env.q_west, "total": info["total_queue"]})
                if done:
                    break
            st.rerun()
    with c:
        if st.button("↻ Reset", use_container_width=True):
            reset_sim(sim_algo)
            st.rerun()

if st.session_state.history:
    fig = queue_history_figure(st.session_state.history)
    if fig:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.caption("Simulation uses the project's custom environment: Poisson arrivals, 2-vehicle saturation flow, 3-unit switching penalty, and 200-step episodes.")

st.divider()

# ----------------------------- Training analysis -----------------------------
st.header("📈 Training Analysis")
st.plotly_chart(learning_figure(selected), use_container_width=True, config={"displayModeBar": False})

st.info("Training curves use the project's saved episode-reward histories. The project also contains the original average-queue visualization in 05_Results_and_Graphs.")

st.divider()

# ----------------------------- Comparison -----------------------------
st.header("📊 Algorithm Comparison")
show = metrics[metrics["name"].isin(selected)].copy()
metric_cols = [
    "Average Reward (last 50 ep)",
    "Cumulative Reward (all ep)",
    "Convergence Speed (episodes)",
    "Stability (std dev, last 50 ep)",
    "Success Rate (last 50 ep)",
]

# Show all selected algorithms and all core metrics in one readable table.
comparison_cols = ["name"] + metric_cols
st.dataframe(show[comparison_cols], use_container_width=True, hide_index=True)

metric_choice = st.selectbox("Visualize metric", metric_cols)
bar = go.Figure(go.Bar(
    x=show["name"], y=show[metric_choice],
    text=show[metric_choice], textposition="auto"
))
bar.update_layout(
    title=f"{metric_choice} — all selected algorithms",
    xaxis_title="Algorithm", yaxis_title=metric_choice, height=420,
    margin=dict(l=20, r=20, t=70, b=40)
)
st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

# A compact interpretation box based only on stored project metrics.
if len(show) > 1:
    if metric_choice in ["Average Reward (last 50 ep)", "Cumulative Reward (all ep)"]:
        winner = show.loc[show[metric_choice].idxmax(), "name"]
        st.success(f"For {metric_choice.lower()}, the highest stored value is from **{winner}**.")
    elif metric_choice in ["Convergence Speed (episodes)", "Stability (std dev, last 50 ep)"]:
        winner = show.loc[show[metric_choice].idxmin(), "name"]
        st.success(f"For {metric_choice.lower()}, the lowest stored value is from **{winner}**.")
    else:
        winner = show.loc[show[metric_choice].idxmax(), "name"]
        st.success(f"For {metric_choice.lower()}, the highest stored value is from **{winner}**.")

st.divider()

# ----------------------------- Policy viewer -----------------------------
st.header("🧠 Final Policy Viewer")
policy_algo = st.selectbox("Policy to inspect", list(APPROACHES), key="policy_algo")
policy_df = policy_table(policy_algo)
st.dataframe(policy_df, use_container_width=True, hide_index=True)

# Policy action matrices by phase: compact and easy to explain in viva.
st.markdown("### Policy summary")
summary = policy_df.pivot_table(index=["NS queue", "EW queue"], columns="Current phase", values="Best action", aggfunc="first").reset_index()
st.dataframe(summary, use_container_width=True, hide_index=True)

# A visual policy map makes the learned decision boundary easier to explain.
def policy_matrix(df, phase_name):
    m = df[df["Current phase"] == phase_name].pivot(index="NS queue", columns="EW queue", values="Best action")
    order = ["Low", "Medium", "High"]
    return m.reindex(index=order, columns=order)

mc1, mc2 = st.columns(2)
with mc1:
    st.markdown("#### NS-Green policy")
    st.dataframe(policy_matrix(policy_df, "NS-Green"), use_container_width=True)
with mc2:
    st.markdown("#### EW-Green policy")
    st.dataframe(policy_matrix(policy_df, "EW-Green"), use_container_width=True)

st.caption("Policy quality should be interpreted together with independent evaluation. The headline comparison metrics shown above are the project's reported training-phase metrics.")

st.divider()
st.caption("CIA-3 • Smart Traffic Signal Optimization using Reinforcement Learning")
