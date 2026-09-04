"""
Approach 3: Expected SARSA

Update rule:
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * E_pi[Q(s',.)] - Q(s,a) ]
where E_pi[Q(s',.)] is the EXPECTED value of the next state under the current
epsilon-greedy policy (a probability-weighted average over all actions),
instead of the single sampled action SARSA uses, or the single max Q-learning uses.

This removes the variance introduced by sampling a', which typically gives
smoother, faster and more stable convergence than plain SARSA -- an
important, directly comparable contrast for the performance-comparison
deliverable (stability / variance metric).
"""
import numpy as np


def _epsilon_greedy(Q, state, epsilon, rng, n_actions):
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    return int(np.argmax(Q[state]))


def _expected_value(Q, state, epsilon, n_actions):
    """E_pi[Q(s,.)] under epsilon-greedy policy pi."""
    q_values = Q[state]
    best_action = np.argmax(q_values)
    probs = np.full(n_actions, epsilon / n_actions)
    probs[best_action] += (1.0 - epsilon)
    return float(np.dot(probs, q_values))


def train_expected_sarsa(env, n_episodes=500, alpha=0.1, gamma=0.95,
                          epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.N_ACTIONS))

    epsilon = epsilon_start
    episode_rewards = []
    episode_avg_queue = []

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = _epsilon_greedy(Q, state, epsilon, rng, env.N_ACTIONS)
            next_state, reward, done, info = env.step(action)

            expected_next = _expected_value(Q, next_state, epsilon, env.N_ACTIONS)
            td_target = reward + gamma * expected_next
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_avg_queue.append(env.avg_queue_length())
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return {
        "Q": Q,
        "episode_rewards": np.array(episode_rewards),
        "episode_avg_queue": np.array(episode_avg_queue),
        "name": "Expected SARSA",
    }


def greedy_policy(Q):
    return np.argmax(Q, axis=1)
