"""
metrics.py — Deliverable 2: Performance Comparison (Transportation / Traffic Signal domain)

Implements 5 clearly-defined metrics (assignment requires a minimum of 4):

1. Average Reward        : mean episodic reward over the LAST 50 episodes (post-convergence
                            performance; reward = -total queue length each step, so closer
                            to 0 is better).
2. Cumulative Reward      : sum of reward across ALL training episodes (total learning-phase
                            performance — rewards both fast learning and a good final policy).
3. Convergence Speed      : first episode index at which the rolling-mean reward (window=20)
                            stays within tolerance of its final rolling-mean value for at
                            least 90% of the remaining episodes. Lower = faster convergence.
4. Stability / Variance   : standard deviation of episodic reward over the last 50 episodes.
                            Lower = more stable/consistent signal-control policy.
5. Success Rate           : fraction of the last 50 episodes where the average queue length
                            stayed below a "low congestion" threshold (<=6 vehicles/step on
                            average) — i.e. the intersection was kept flowing smoothly.
"""
import numpy as np

LOW_CONGESTION_THRESHOLD = 6.0


def _rolling_mean(x, window=20):
    if len(x) < window:
        return x.copy()
    out = np.convolve(x, np.ones(window) / window, mode="valid")
    return out


def compute_metrics(result, tail=50, conv_window=20, conv_tol=0.10):
    rewards = result["episode_rewards"]
    avg_queue = result["episode_avg_queue"]
    n = len(rewards)
    tail = min(tail, n)

    avg_reward = float(np.mean(rewards[-tail:]))
    cumulative_reward = float(np.sum(rewards))
    stability_std = float(np.std(rewards[-tail:]))
    success_rate = float(np.mean(avg_queue[-tail:] <= LOW_CONGESTION_THRESHOLD))

    # Convergence speed: first episode after which the rolling-mean reward stays
    # within `conv_tol` (as a fraction of the overall reward spread) of its final
    # level for at least 90% of the remaining training episodes. Using a 90%
    # threshold (rather than 100%) tolerates the brief noise spikes that a
    # stochastic market environment naturally produces episode to episode,
    # while still requiring the policy to have genuinely settled.
    roll = _rolling_mean(rewards, conv_window)
    final_level = np.mean(roll[-max(1, len(roll) // 10):])  # avg of last 10% as "converged" level
    spread = np.std(roll) if np.std(roll) > 1e-6 else 1.0
    band = conv_tol * spread * 5  # widen band relative to the series' own noise scale

    convergence_episode = n  # default: never converged within tolerance
    for i in range(len(roll)):
        remainder = roll[i:]
        within = np.abs(remainder - final_level) <= band
        if within.mean() >= 0.9:
            convergence_episode = i + conv_window  # offset for rolling window
            break

    return {
        "name": result["name"],
        "Average Reward (last 50 ep)": round(avg_reward, 3),
        "Cumulative Reward (all ep)": round(cumulative_reward, 2),
        "Convergence Speed (episodes)": convergence_episode,
        "Stability (std dev, last 50 ep)": round(stability_std, 3),
        "Success Rate (last 50 ep)": round(success_rate * 100, 1),  # as %
    }


def print_comparison_table(all_metrics):
    cols = list(all_metrics[0].keys())
    widths = [max(len(str(m[c])) for m in all_metrics + [dict(zip(cols, cols))]) for c in cols]
    header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        row = " | ".join(str(m[c]).ljust(w) for c, w in zip(cols, widths))
        print(row)
