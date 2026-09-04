# CIA-3 Component 2 — Micro Project
## Transportation Domain: Smart Traffic Signal Optimization

This project implements and compares four reinforcement learning approaches on a
custom single-intersection traffic signal control environment (continuing the
Component 1 case study).

## The problem
A single 4-way intersection (North/South/East/West approaches) must decide every
control interval whether to KEEP the current green phase (NS-green or EW-green)
or SWITCH to the other phase, to minimize vehicle queueing as traffic arrives
stochastically on each approach. A fixed-time signal can't adapt to uneven,
time-varying demand — this is the motivation for using RL.

## Folder structure
- `traffic_env.py` — the shared custom intersection environment (state, action, reward, dynamics)
- `metrics.py` — the 5 performance metrics used for comparison
- `run_all.py` — trains all four agents and generates every deliverable below
- `agents/` — one file per RL approach, each with a docstring explaining its update rule and why it's relevant here
- `04_RL_Implementations/` — Deliverable 1: per-approach learning curve, final Q-table, final policy, and raw episode rewards
- `05_Results_and_Graphs/` — Deliverable 3: combined learning curves and average-queue-length curves across all four approaches
- `06_Performance_Comparison/` — Deliverable 2: metrics JSON, markdown comparison table, and one bar chart per metric

## Environment details
- **State** (18 discrete states): queue length on NS approaches (Low/Med/High),
  queue length on EW approaches (Low/Med/High), current green phase.
- **Actions** (2): Keep current phase, or Switch phase (switching costs one
  step of zero discharge — modelling the real yellow/all-red clearance interval
  lost time — plus a reward penalty, discouraging thrashing between phases).
- **Reward**: negative total vehicles queued each step (agent is rewarded for
  keeping the intersection flowing).
- **Arrivals**: Poisson-distributed vehicle arrivals per approach per step —
  stochastic, as in Component 1's "Deterministic/Stochastic" characterization.

## The four RL approaches (chosen to be logically connected — all TD-control variants)
1. **Q-learning** — off-policy baseline
2. **SARSA** — on-policy contrast to Q-learning
3. **Expected SARSA** — variance-reduced on-policy update
4. **Double Q-learning** — corrects Q-learning's maximisation-bias overestimation

## How to reproduce
```
pip install numpy matplotlib
python3 run_all.py
```
All four agents train on the identical arrival sequence (shared random seed),
so the comparison is fair — differences come only from the algorithm.

## Headline result
All four agents learned to cut average queueing from ~3000 (untrained,
essentially random switching) down to ~1000 within 500 episodes.
**Double Q-learning converged fastest (173 episodes) and achieved the best
average reward and cumulative reward**, plausibly because correcting Q-value
overestimation prevents the agent from being overconfident about premature
phase switches. **SARSA had the lowest variance (std dev 96.9)**, consistent
with on-policy learning producing more consistent, exploration-aware behaviour.
Full metrics are in `06_Performance_Comparison/comparison_table.md`.
