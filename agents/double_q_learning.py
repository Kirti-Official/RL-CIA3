"""
Approach 4: Double Q-learning

Maintains TWO independent Q-tables (Qa, Qb). At each step, one is chosen at
random to be updated, using the OTHER table to evaluate the action selected
by the first. This decouples action-selection from action-evaluation and
corrects the maximisation bias that plain Q-learning suffers from (Q-learning
systematically OVERESTIMATES action values because it both selects and
evaluates using the same max operator).

    With prob 0.5:
        a* = argmax_a Qa(s',a)
        Qa(s,a) <- Qa(s,a) + alpha * [ r + gamma * Qb(s',a*) - Qa(s,a) ]
    else swap Qa <-> Qb

This is a directly relevant, real issue in trading: overestimated Q-values
can make an agent overconfident about a trade's expected profitability, so
comparing plain Q-learning against Double Q-learning is a meaningful,
well-motivated fourth approach.
"""
import numpy as np


def _epsilon_greedy_combined(Qa, Qb, state, epsilon, rng, n_actions):
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    combined = Qa[state] + Qb[state]
    return int(np.argmax(combined))


def train_double_q_learning(env, n_episodes=500, alpha=0.1, gamma=0.95,
                             epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995, seed=0):
    rng = np.random.default_rng(seed)
    Qa = np.zeros((env.n_states, env.N_ACTIONS))
    Qb = np.zeros((env.n_states, env.N_ACTIONS))

    epsilon = epsilon_start
    episode_rewards = []
    episode_avg_queue = []

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = _epsilon_greedy_combined(Qa, Qb, state, epsilon, rng, env.N_ACTIONS)
            next_state, reward, done, info = env.step(action)

            if rng.random() < 0.5:
                a_star = np.argmax(Qa[next_state])
                td_target = reward + gamma * Qb[next_state, a_star]
                Qa[state, action] += alpha * (td_target - Qa[state, action])
            else:
                b_star = np.argmax(Qb[next_state])
                td_target = reward + gamma * Qa[next_state, b_star]
                Qb[state, action] += alpha * (td_target - Qb[state, action])

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_avg_queue.append(env.avg_queue_length())
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    Q_combined = (Qa + Qb) / 2.0
    return {
        "Q": Q_combined,
        "Qa": Qa,
        "Qb": Qb,
        "episode_rewards": np.array(episode_rewards),
        "episode_avg_queue": np.array(episode_avg_queue),
        "name": "Double Q-learning",
    }


def greedy_policy(Q):
    return np.argmax(Q, axis=1)
