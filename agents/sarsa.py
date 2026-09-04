"""
Approach 2: SARSA (on-policy TD control)

Update rule:
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s',a') - Q(s,a) ]
where a' is the action ACTUALLY taken next under the current (epsilon-greedy)
policy -- not the greedy max.

On-policy learning matters for trading because it accounts for the exploration
risk actually being taken: SARSA learns a policy that is aware it will sometimes
explore, producing more "cautious"/robust behaviour than Q-learning's purely
greedy target, which is a useful contrast to compare against.
"""
import numpy as np


def _epsilon_greedy(Q, state, epsilon, rng, n_actions):
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    return int(np.argmax(Q[state]))


def train_sarsa(env, n_episodes=500, alpha=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.N_ACTIONS))

    epsilon = epsilon_start
    episode_rewards = []
    episode_avg_queue = []

    for ep in range(n_episodes):
        state = env.reset()
        action = _epsilon_greedy(Q, state, epsilon, rng, env.N_ACTIONS)
        done = False
        total_reward = 0.0

        while not done:
            next_state, reward, done, info = env.step(action)
            next_action = _epsilon_greedy(Q, next_state, epsilon, rng, env.N_ACTIONS)

            td_target = reward + gamma * Q[next_state, next_action]
            Q[state, action] += alpha * (td_target - Q[state, action])

            state, action = next_state, next_action
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_avg_queue.append(env.avg_queue_length())
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return {
        "Q": Q,
        "episode_rewards": np.array(episode_rewards),
        "episode_avg_queue": np.array(episode_avg_queue),
        "name": "SARSA",
    }


def greedy_policy(Q):
    return np.argmax(Q, axis=1)
