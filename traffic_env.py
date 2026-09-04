"""
traffic_env.py
----------------
Custom Smart Traffic Signal Control environment for the CIA-3 RL micro project
(Transportation domain).

Problem: A single 4-way signalized intersection (North, South, East, West
approaches) must decide, every control interval, whether to KEEP the current
green phase or SWITCH to the other phase, to minimize vehicle queueing /
waiting time. Vehicles arrive stochastically on each approach; a rule-based
fixed-time signal can't adapt to real-time, uneven traffic load, which is the
motivation for using RL instead.

This is a simplified but standard "queue-based" traffic-signal RL model
(the same style used in most adaptive-signal-control RL literature):
arrivals -> queues -> discharge at a saturation flow rate during green ->
repeat.
"""

import numpy as np


# ----------------------------------------------------------------------
# Discretization helpers
# ----------------------------------------------------------------------
def _bin_queue(q, low_thresh=5, high_thresh=15):
    """0 = Low, 1 = Medium, 2 = High queue length."""
    if q <= low_thresh:
        return 0
    elif q <= high_thresh:
        return 1
    return 2


class TrafficSignalEnv:
    """
    Two signal phases:
        Phase 0 = North-South green (East-West red)
        Phase 1 = East-West green (North-South red)

    Actions:
        0 = Keep current phase
        1 = Switch phase (incurs a yellow-time step where NOTHING is discharged,
            modelling the lost time / clearance interval of a real signal)

    State: (NS_queue_bin, EW_queue_bin, current_phase) -> encoded as an int index.
           3 (NS bins) * 3 (EW bins) * 2 (phase) = 18 discrete states.

    Reward: negative total queue length each step (agent is penalised for
    vehicles waiting), with an additional penalty for switching too soon
    (models real-world yellow/all-red lost time discouraging excessive switching).
    """

    N_ACTIONS = 2
    KEEP, SWITCH = 0, 1
    PHASE_NS, PHASE_EW = 0, 1

    def __init__(self, episode_len=200, arrival_rate=0.35, saturation_flow=2,
                 switch_penalty=3.0, seed=None):
        self.episode_len = episode_len
        self.arrival_rate = arrival_rate      # mean vehicles/step arriving per approach (Poisson)
        self.saturation_flow = saturation_flow  # vehicles discharged per step per green approach-pair
        self.switch_penalty = switch_penalty    # extra reward penalty for switching (lost-time cost)
        self.rng = np.random.default_rng(seed)

        self.n_states = 3 * 3 * 2

    def _encode_state(self):
        ns_bin = _bin_queue(self.q_north + self.q_south)
        ew_bin = _bin_queue(self.q_east + self.q_west)
        return (ns_bin * 3 + ew_bin) * 2 + self.phase

    def reset(self):
        # start with small random queues, like an intersection at an arbitrary point in the day
        self.q_north = int(self.rng.integers(0, 6))
        self.q_south = int(self.rng.integers(0, 6))
        self.q_east = int(self.rng.integers(0, 6))
        self.q_west = int(self.rng.integers(0, 6))
        self.phase = int(self.rng.integers(0, 2))
        self.steps_taken = 0
        self.total_wait = 0
        return self._encode_state()

    def step(self, action):
        # --- stochastic arrivals on all four approaches this step ---
        self.q_north += self.rng.poisson(self.arrival_rate)
        self.q_south += self.rng.poisson(self.arrival_rate)
        self.q_east += self.rng.poisson(self.arrival_rate)
        self.q_west += self.rng.poisson(self.arrival_rate)

        switched = False
        if action == self.SWITCH:
            self.phase = 1 - self.phase
            switched = True
            # yellow/clearance interval: no discharge this step (lost time)
        else:
            # discharge vehicles on the green approaches at the saturation flow rate
            if self.phase == self.PHASE_NS:
                served_n = min(self.q_north, self.saturation_flow)
                served_s = min(self.q_south, self.saturation_flow)
                self.q_north -= served_n
                self.q_south -= served_s
            else:
                served_e = min(self.q_east, self.saturation_flow)
                served_w = min(self.q_west, self.saturation_flow)
                self.q_east -= served_e
                self.q_west -= served_w

        total_queue = self.q_north + self.q_south + self.q_east + self.q_west
        self.total_wait += total_queue

        reward = -float(total_queue)
        if switched:
            reward -= self.switch_penalty

        self.steps_taken += 1
        done = self.steps_taken >= self.episode_len
        next_state = self._encode_state()
        info = {"total_queue": total_queue, "phase": self.phase}
        return next_state, reward, done, info

    def avg_queue_length(self):
        return self.total_wait / self.steps_taken
