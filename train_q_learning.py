"""Train and evaluate a heuristic-driven policy with Q-table tie-breaking."""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from game_env import (
    DRAGON,
    DYNAMITE,
    GEM,
    LORRY,
    MAX_COLUMNS,
    PASS,
    PIT_CAGE,
    MineEnv,
)
from rule_policy import (
    _can_displace,
    _dragon_under_opponent,
    _estimate_dragon_risk,
    _is_safe_from_displacement,
    _is_signaled_danger,
    _opponent_by_index,
    _should_announce_blast,
)

QTable = defaultdict[tuple[tuple[int, ...], int, bool], float]
MAX_EPISODE_STEPS = 500

# Path convention shared with mine_gui.py so the trained table can be reused.
TRAINING_CACHE_VERSION = 4
TRAINING_CACHE_DIR = Path(__file__).with_name(".training_cache")


def cache_path(opponents: int, seed: int, episodes: int) -> Path:
    """Return the Q-table cache path the GUI loads (same convention as mine_gui).

    The episode count is intentionally left out of the filename: a table trained
    with any episode count can be loaded by the GUI for the same opponents/seed.
    """
    TRAINING_CACHE_DIR.mkdir(exist_ok=True)
    return TRAINING_CACHE_DIR / (
        f"q_table_opponents_{opponents}_seed_{seed}_v{TRAINING_CACHE_VERSION}.pkl"
    )


def _can_blast(env: MineEnv) -> bool:
    """Return True if the learner may announce a blast this turn."""
    player = env.players[0]
    if player.blast_pending or player.dynamite < 2:
        return False
    legal = np.flatnonzero(env.action_mask())
    return bool(np.any(legal < PIT_CAGE))


def _blast_advises(env: MineEnv) -> bool:
    """Heuristic: should the learner announce a blast right now?

    Tabular Q-learning cannot converge on blast because every blast state is
    essentially unique (random board), so the announce action is visited at
    most a handful of times and its Q-value never accumulates. This heuristic
    mirrors the rule-based opponent policy: blast is a catch-up move used when
    trailing or thin on gems, never into known danger or when protecting a
    strong lead.
    """
    if not _can_blast(env):
        return False
    ctx = env.player_action_context(0)
    return _should_announce_blast(ctx)


def _column_owner_fled(ctx, col: int) -> bool:
    """Return True if the owner of *col* fled voluntarily (danger signal)."""
    if col == ctx.player_index:
        return False
    owner = _opponent_by_index(ctx, col)
    if owner is None:
        return False
    owner_fled = owner.position in (PIT_CAGE, LORRY) or owner.escaped
    if not owner_fled:
        return False
    was_displaced = any(event[0] == col for event in ctx.displacement_events)
    return not was_displaced


def _card_score(ctx, action: int) -> float:
    """Heuristic score for moving onto a card. Higher = better."""
    row, col = divmod(action, MAX_COLUMNS)
    if row >= len(ctx.public_rows) or col >= len(ctx.public_rows[row]):
        return -1000.0
    card_entry = ctx.public_rows[row][col]
    if card_entry is None:
        return -1000.0

    # Dragon risk: own column fronts are visible; others estimated from backside.
    if col == ctx.player_index:
        if card_entry[0] == DRAGON:
            return -1000.0  # Known dragon — avoid at all costs
        dragon_risk = 0.0
    else:
        dragon_risk = _estimate_dragon_risk(ctx, card_entry[1])

    score = (1.0 - dragon_risk) * 100.0

    # Own column: you know exactly what's there.
    if col == ctx.player_index:
        score += 15.0
        # Known dynamite is valuable for blasting; known gems secure loot.
        if card_entry[0] == DYNAMITE:
            score += 60.0
        elif card_entry[0] == GEM:
            score += 40.0

    # Deeper rows are closer to the bottom of the mine (more cards to come).
    score += row * 5.0

    # Displacement: steal a card and knock an opponent back.
    if _can_displace(ctx, row, col):
        score += 15.0

    # Column owner fled = possible dragon under that card.
    if _column_owner_fled(ctx, col):
        score -= 200.0

    # Can another player reach this row to displace us?
    if not _is_safe_from_displacement(ctx, row):
        score -= 150.0

    return score


def _cage_score(ctx, player) -> float:
    """Heuristic score for retreating to the pit cage."""
    score = 0.0
    # Escaping with loot is good; the more gems, the more valuable.
    score += player.gems * 15.0
    score += len(ctx.private_cards) * 1.0
    # If the player is trailing, escaping secures what they have.
    max_opp = max((o.score for o in ctx.opponents), default=ctx.own_score)
    if ctx.own_score < max_opp:
        score += 10.0
    return score


def _lorry_score(ctx, player) -> float:
    """Heuristic score for moving to the lorry (usually bad)."""
    # The lorry is a trap: once there, you can only go cage<->lorry.
    # Only mildly attractive if you have a lot of loot and nowhere safe to go.
    return player.gems * 5.0 - 50.0


def select_action(
    env: MineEnv, q_table: QTable, epsilon: float, rng: np.random.Generator
) -> tuple[int, bool]:
    """Choose a planning action and whether to announce a blast.

    Returns ``(action, announce_blast)``. Decisions are driven by heuristics
    (dragon risk, column safety, loot value, blast opportunity); the Q-table
    is used only as a minor tie-breaker since tabular Q-learning cannot
    converge on this huge, near-unique state space.
    """
    legal = np.flatnonzero(env.action_mask())
    if len(legal) == 0:
        return int(PASS), False

    if rng.random() < epsilon:
        action = int(rng.choice(legal))
        announce_blast = bool(
            action < PIT_CAGE and _can_blast(env) and rng.random() < 0.3
        )
        return action, announce_blast

    state = tuple(int(value) for value in env._observation())
    ctx = env.player_action_context(0)
    player = env.players[0]

    # --- 1. Blast advisory (takes precedence when advisable) ---
    if _blast_advises(env):
        card_actions = [a for a in legal if a < PIT_CAGE]
        if card_actions:
            best_card = max(
                card_actions,
                key=lambda a: _card_score(ctx, a)
                + q_table.get((state, int(a), False), 0.0) * 0.05,
            )
            return int(best_card), True

    # --- 2. Score all non-blast actions heuristically ---
    scored: list[tuple[float, int]] = []
    for a in legal:
        if a < PIT_CAGE:
            base = _card_score(ctx, a)
        elif a == PIT_CAGE:
            base = _cage_score(ctx, player)
        elif a == LORRY:
            base = _lorry_score(ctx, player)
        else:  # PASS
            base = 0.0
        q = q_table.get((state, int(a), False), 0.0)
        scored.append((base + q * 0.05, int(a)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], False


def _resolve_learner_blast(env: MineEnv) -> float:
    """Automatically pick blast cards for the learner during training.

    The Q-table learns the planning decision (whether and where to blast); the
    blast card draw is resolved with the shared heuristic so training does not
    stall waiting for a human resolver.

    Returns a training-only shaping reward crediting the cards kept from the
    blast (same per-card values as ordinary collection), so the blast action's
    Q-value is directly reinforced instead of only indirectly via a much later
    escape bonus.
    """
    from rule_policy import deterministic_blast_choice

    player = env.players[0]
    shaping = 0.0
    while player.blast_pending and player.blast_cards:
        before_gems = player.gems
        before_dynamite = player.dynamite
        env.resolve_blast(
            0, deterministic_blast_choice(tuple(player.blast_cards))
        )
        shaping += 0.25 * (player.gems - before_gems)
        shaping += 0.1 * (player.dynamite - before_dynamite)
    return shaping


def _advance_p1_turn(
    env: MineEnv, action: int, announce_blast: bool
) -> tuple[float, bool, bool, dict]:
    if env.players[0].escaped or env.players[0].dead:
        return 0.0, True, False, {"outcome": "ended"}
    while env.planning_player != 0:
        _, _, terminated, truncated, info = env.step_one_player()
        if terminated or truncated:
            return 0.0, terminated, truncated, info
        if env.players[0].escaped or env.players[0].dead:
            return 0.0, True, False, {"outcome": "ended"}
    _, reward, terminated, truncated, info = env.step_one_player(
        action, announce_blast=announce_blast
    )
    total_reward = reward

    if announce_blast and not terminated and not truncated:
        # A blast is a commitment: once announced, the learner must PASS when
        # their turn comes around again to keep it alive (any other move
        # aborts it). Drive the round until execution prepares the blast cards,
        # resolve them, and attribute the whole announce->execute->resolve
        # sequence to the announce action so Q-learning can credit it.
        macro_steps = 0
        while (
            not terminated
            and not truncated
            and env.players[0].blast_pending
            and not env.players[0].blast_cards
            and macro_steps < 200
        ):
            macro_steps += 1
            if env.planning_player == 0:
                # Learner's turn again: pass keeps the blast alive.
                _, r, terminated, truncated, info = env.step_one_player(PASS)
                total_reward += r
                if terminated or truncated:
                    break
                if env.players[0].escaped or env.players[0].dead:
                    return total_reward, True, False, {"outcome": "ended"}
            else:
                _, r, terminated, truncated, info = env.step_one_player()
                total_reward += r
                if terminated or truncated:
                    break
                if env.players[0].escaped or env.players[0].dead:
                    return total_reward, True, False, {"outcome": "ended"}

        # Safety net: if the blast never reached execution, abort it so training
        # cannot stall forever (opponents may refuse to pass).
        if env.players[0].blast_pending and not env.players[0].blast_cards:
            env._cancel_blast(0)

    # Resolve the learner's blast once execution has prepared the blast cards.
    if env.players[0].blast_pending and env.players[0].blast_cards:
        total_reward += _resolve_learner_blast(env)

    # Let any remaining opponents play until the learner's next turn.
    while not terminated and not truncated and env.planning_player != 0:
        _, reward, terminated, truncated, info = env.step_one_player()
        total_reward += reward
        if env.players[0].escaped or env.players[0].dead:
            return total_reward, True, False, {"outcome": "ended"}

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
            action, announce_blast = select_action(env, q_table, epsilon, rng)
            reward, terminated, truncated, _info = _advance_p1_turn(
                env, action, announce_blast
            )
            done = terminated or truncated
            steps += 1
            next_state = tuple(int(value) for value in env._observation())
            next_values = [
                q_table.get((next_state, int(a), ab), 0.0)
                for a in np.flatnonzero(env.action_mask())
                for ab in (False, True)
                if not ab or (int(a) < PIT_CAGE and _can_blast(env))
            ]
            target = reward + (gamma * max(next_values) if next_values and not done else 0.0)
            q_table[(state, int(action), bool(announce_blast))] += alpha * (
                target - q_table[(state, int(action), bool(announce_blast))]
            )
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
            action, announce_blast = select_action(env, q_table, 0.0, rng)
            reward, terminated, truncated, info = _advance_p1_turn(
                env, action, announce_blast
            )
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
    parser.add_argument("--no-save", action="store_true",
                        help="do not write the trained Q-table to the cache")
    args = parser.parse_args()
    q_table = train(args.episodes, args.opponents, args.policy, args.seed)
    result = evaluate(q_table, args.eval_episodes, args.opponents, args.policy, args.seed + 1)
    print({"states": len(q_table), **result})

    if not args.no_save:
        path = cache_path(args.opponents, args.seed, args.episodes)
        with path.open("wb") as f:
            pickle.dump(q_table, f)
        print(f"Saved Q-table to {path}")


if __name__ == "__main__":
    main()
