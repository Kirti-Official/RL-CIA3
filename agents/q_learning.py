"""
Approach 1: Q-learning (off-policy TD control)

Update rule:
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Q-learning learns the value of the GREEDY policy regardless of the behaviour
policy used to explore (off-policy) — it directly targets the optimal action
value function, which makes it a natural, well-justified baseline for the
trading problem: we want the agent to eventually act greedily w.r.t. the best
known trading policy while still exploring during training.
"""
import numpy as np


def train_q_learning(env, n_episodes=500, alpha=0.1, gamma=0.95,
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
            if rng.random() < epsilon:
                action = rng.integers(0, env.N_ACTIONS)
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, done, info = env.step(action)

            best_next = np.max(Q[next_state])
            td_target = reward + gamma * best_next
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
        "name": "Q-learning",
    }


def greedy_policy(Q):
    return np.argmax(Q, axis=1)
