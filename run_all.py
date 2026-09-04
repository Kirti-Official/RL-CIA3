"""
run_all.py — trains all four RL approaches on the Smart Traffic Signal
environment, produces learning curves, final policies, and the performance
comparison (Deliverables 1, 2 and 3 of Component 2).
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from traffic_env import TrafficSignalEnv
from agents.q_learning import train_q_learning
from agents.sarsa import train_sarsa
from agents.expected_sarsa import train_expected_sarsa
from agents.double_q_learning import train_double_q_learning
from metrics import compute_metrics, print_comparison_table

OUT_RESULTS = "05_Results_and_Graphs"
OUT_COMPARISON = "06_Performance_Comparison"
os.makedirs(OUT_RESULTS, exist_ok=True)
os.makedirs(OUT_COMPARISON, exist_ok=True)

N_EPISODES = 500
ACTION_NAMES = {0: "Keep phase", 1: "Switch phase"}
QUEUE_BIN_NAMES = ["Low", "Medium", "High"]
PHASE_NAMES = {0: "NS-Green", 1: "EW-Green"}


def make_env(seed):
    return TrafficSignalEnv(episode_len=200, arrival_rate=0.35, saturation_flow=2,
                             switch_penalty=3.0, seed=seed)


def rolling_mean(x, w=20):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_learning_curve(result, path):
    rewards = result["episode_rewards"]
    smoothed = rolling_mean(rewards, 20)
    plt.figure(figsize=(7, 4.5))
    plt.plot(rewards, alpha=0.3, color="steelblue", label="Episode reward")
    plt.plot(range(19, 19 + len(smoothed)), smoothed, color="darkblue", linewidth=2, label="20-episode rolling mean")
    plt.title(f"Learning Curve — {result['name']}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (higher = less queueing)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_combined_learning_curves(results, path):
    plt.figure(figsize=(8, 5))
    for r in results:
        smoothed = rolling_mean(r["episode_rewards"], 20)
        plt.plot(range(19, 19 + len(smoothed)), smoothed, linewidth=2, label=r["name"])
    plt.title("Learning Curves — All Four Approaches (20-ep rolling mean)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (higher = less queueing)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_avg_queue_curve(results, path):
    plt.figure(figsize=(8, 5))
    for r in results:
        smoothed = rolling_mean(r["episode_avg_queue"], 20)
        plt.plot(range(19, 19 + len(smoothed)), smoothed, linewidth=2, label=r["name"])
    plt.title("Average Queue Length vs Episode (20-ep rolling mean)")
    plt.xlabel("Episode")
    plt.ylabel("Avg. vehicles queued per step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_metric_bars(all_metrics, metric_key, ylabel, path):
    names = [m["name"] for m in all_metrics]
    values = [m[metric_key] for m in all_metrics]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    plt.figure(figsize=(6.5, 4.5))
    bars = plt.bar(names, values, color=colors)
    plt.ylabel(ylabel)
    plt.title(metric_key)
    plt.xticks(rotation=15)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def policy_summary(Q):
    """Human-readable final policy: best action for each of the 18 discretized states."""
    actions = np.argmax(Q, axis=1)
    summary = []
    idx = 0
    for ns_bin in range(3):
        for ew_bin in range(3):
            for phase in range(2):
                summary.append({
                    "NS_queue": QUEUE_BIN_NAMES[ns_bin],
                    "EW_queue": QUEUE_BIN_NAMES[ew_bin],
                    "current_phase": PHASE_NAMES[phase],
                    "best_action": ACTION_NAMES[int(actions[idx])]
                })
                idx += 1
    return summary


def main():
    algorithms = [
        ("q_learning", train_q_learning),
        ("sarsa", train_sarsa),
        ("expected_sarsa", train_expected_sarsa),
        ("double_q_learning", train_double_q_learning),
    ]

    results = []
    for i, (name, train_fn) in enumerate(algorithms, start=1):
        env = make_env(seed=1)  # fixed seed -> identical arrival sequence across agents (fair comparison)
        print(f"Training {name} ...")
        result = train_fn(env, n_episodes=N_EPISODES, seed=7)
        results.append(result)

        approach_dir = f"04_RL_Implementations/Approach_{i}_{name}"
        os.makedirs(approach_dir, exist_ok=True)
        plot_learning_curve(result, os.path.join(approach_dir, "learning_curve.png"))
        np.save(os.path.join(approach_dir, "Q_table.npy"), result["Q"])
        with open(os.path.join(approach_dir, "final_policy.json"), "w") as f:
            json.dump(policy_summary(result["Q"]), f, indent=2)
        with open(os.path.join(approach_dir, "episode_rewards.json"), "w") as f:
            json.dump(result["episode_rewards"].tolist(), f)

    plot_combined_learning_curves(results, os.path.join(OUT_RESULTS, "combined_learning_curves.png"))
    plot_avg_queue_curve(results, os.path.join(OUT_RESULTS, "avg_queue_length_curves.png"))

    all_metrics = [compute_metrics(r) for r in results]
    print("\n=== Performance Comparison ===")
    print_comparison_table(all_metrics)

    with open(os.path.join(OUT_COMPARISON, "comparison_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    plot_metric_bars(all_metrics, "Average Reward (last 50 ep)", "Reward",
                      os.path.join(OUT_COMPARISON, "bar_average_reward.png"))
    plot_metric_bars(all_metrics, "Cumulative Reward (all ep)", "Reward",
                      os.path.join(OUT_COMPARISON, "bar_cumulative_reward.png"))
    plot_metric_bars(all_metrics, "Convergence Speed (episodes)", "Episodes",
                      os.path.join(OUT_COMPARISON, "bar_convergence_speed.png"))
    plot_metric_bars(all_metrics, "Stability (std dev, last 50 ep)", "Std Dev",
                      os.path.join(OUT_COMPARISON, "bar_stability.png"))
    plot_metric_bars(all_metrics, "Success Rate (last 50 ep)", "% low-congestion episodes",
                      os.path.join(OUT_COMPARISON, "bar_success_rate.png"))

    with open(os.path.join(OUT_COMPARISON, "comparison_table.md"), "w") as f:
        cols = list(all_metrics[0].keys())
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for m in all_metrics:
            f.write("| " + " | ".join(str(m[c]) for c in cols) + " |\n")

    print("\nAll outputs saved.")


if __name__ == "__main__":
    main()
