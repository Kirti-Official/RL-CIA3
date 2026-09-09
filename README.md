# Smart Traffic Signal Optimization using Reinforcement Learning

## CIA-3 Component 2 — Micro Project

### Transportation Domain: Adaptive Traffic Signal Control

A comparative study of four Temporal-Difference Reinforcement Learning approaches for adaptive traffic signal optimization at a simulated single intersection.

---

## 📌 Project Overview

Traffic signal control is a real-time sequential decision-making problem in which the controller must continuously decide how to allocate green time between competing traffic flows.

This project formulates traffic signal optimization as a Reinforcement Learning (RL) problem and implements four closely related Temporal-Difference (TD) control algorithms:

1. Q-Learning
2. SARSA
3. Expected SARSA
4. Double Q-Learning

All four algorithms are trained and evaluated using the same custom traffic-signal environment and experimental configuration.

The objective is to learn signal-control policies that reduce vehicle queueing while avoiding unnecessary switching between traffic phases.

---

## 🚦 Problem Statement

The project models a single four-way signalized intersection with:

- North approach
- South approach
- East approach
- West approach

At every control interval, the RL agent must choose between two actions:

- **Keep** — continue the current green phase
- **Switch** — change to the opposing green phase

Vehicle arrivals are stochastic, and traffic queues evolve continuously based on vehicle arrivals and the signal-control decisions.

The objective is to minimize cumulative queueing over time.

Unlike a fixed-time signal plan, the RL controller can learn different Keep/Switch decisions depending on the observed traffic state.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Formulate adaptive traffic signal control as an RL problem.
- Develop a custom traffic intersection environment.
- Implement four logically related TD-control algorithms.
- Train all four algorithms under comparable conditions.
- Generate learning curves and policy outputs.
- Evaluate the algorithms using multiple performance metrics.
- Compare their learning behaviour and performance.
- Provide an interactive dashboard for visualization.
- Propose a Multi-Agent Reinforcement Learning extension for coordinated intersections.

---

# 🧠 Reinforcement Learning Formulation

## Environment

The environment represents a single four-way signalized intersection.

Two signal phases are considered:

- **NS-Green:** North and South approaches receive green.
- **EW-Green:** East and West approaches receive green.

Vehicles arrive stochastically on all four approaches.

The environment models vehicle queues, signal phases, vehicle discharge, switching cost, and stochastic arrivals.

---

## State Space

The state is represented using three components:

```text
(NS Queue Level, EW Queue Level, Current Phase)
