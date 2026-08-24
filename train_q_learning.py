"""Train and evaluate a tabular Q-learning policy."""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from game_env import MineEnv, PASS


QTable = defaultdict[tuple[tuple[int, ...], int], float]
MAX_EPISODE_STEPS = 500


def select_action(env: MineEnv, q_table: QTable, epsilon: float, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(env.action_mask())
    if len(legal) == 0 or rng.random() < epsilon:
        return int(rng.choice(legal))
    state = tuple(int(value) for value in env._observation())
    values = np.asarray([q_table.get((state, int(action)), 0.0) for action in legal])
    best_actions = legal[np.flatnonzero(values == values.max())]
    if PASS in best_actions:
        return PASS
    return int(best_actions[0])


def _advance_p1_turn(env: MineEnv, action: int) -> tuple[float, bool, bool, dict]:
    if env.players[0].escaped or env.players[0].dead:
        return 0.0, True, False, {"outcome": "ended"}
    while env.planning_player != 0:
        _, _, terminated, truncated, info = env.step_one_player()
        if terminated or truncated:
            return 0.0, terminated, truncated, info
        if env.players[0].escaped or env.players[0].dead:
            return 0.0, True, False, {"outcome": "ended"}
    _, reward, terminated, truncated, info = env.step_one_player(action)
    total_reward = reward
    while not terminated and not truncated and env.planning_player != 0:
        _, reward, terminated, truncated, info = env.step_one_player()
        total_reward += reward
    return total_reward, terminated, truncated, info


def train(episodes: int, opponents: int, policy: str, seed: int) -> QTable:
    env = MineEnv(opponents=opponents, opponent_policy=policy)
    q_table: QTable = defaultdict(float)
    rng = np.random.default_rng(seed)
    alpha, gamma = 0.15, 0.97
    for episode in range(episodes):
        env.reset(seed=int(rng.integers(2**31)))
        done = False
        steps = 0
        epsilon = max(0.05, 1.0 - episode / max(1, episodes * 0.8))
        while not done and steps < MAX_EPISODE_STEPS:
            if not env.action_mask().any():
                done = True
                break
            state = tuple(int(value) for value in env._observation())
            action = select_action(env, q_table, epsilon, rng)
            reward, terminated, truncated, _info = _advance_p1_turn(env, action)
            done = terminated or truncated
            steps += 1
            next_state = tuple(int(value) for value in env._observation())
            next_values = [q_table[(next_state, int(a))] for a in np.flatnonzero(env.action_mask())]
            target = reward + (gamma * max(next_values) if next_values and not done else 0.0)
            q_table[(state, action)] += alpha * (target - q_table[(state, action)])
    return q_table


def evaluate(q_table: QTable, episodes: int, opponents: int, policy: str, seed: int) -> dict[str, float]:
    env = MineEnv(opponents=opponents, opponent_policy=policy)
    rng = np.random.default_rng(seed)
    rewards, escapes, dragons = [], 0, 0
    for _ in range(episodes):
        env.reset(seed=int(rng.integers(2**31)))
        done = False
        steps = 0
        total = 0.0
        info = {"outcome": "ended"}
        while not done and steps < MAX_EPISODE_STEPS:
            if not env.action_mask().any():
                done = True
                break
            action = select_action(env, q_table, 0.0, rng)
            reward, terminated, truncated, info = _advance_p1_turn(env, action)
            total += reward
            done = terminated or truncated
            steps += 1
        rewards.append(total)
        escapes += info.get("outcome") == "escaped"
        dragons += info.get("outcome") == "dragon"
    return {"mean_reward": float(np.mean(rewards)), "escape_rate": escapes / episodes, "dragon_rate": dragons / episodes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5_000)
    parser.add_argument("--opponents", type=int, default=2)
    parser.add_argument("--policy", choices=["random", "cautious", "greedy"], default="random")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    q_table = train(args.episodes, args.opponents, args.policy, args.seed)
    result = evaluate(q_table, args.eval_episodes, args.opponents, args.policy, args.seed + 1)
    print({"states": len(q_table), **result})


if __name__ == "__main__":
    main()
