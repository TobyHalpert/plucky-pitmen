"""What is a displacement worth? Forked from the identical position.

A displacement is only legal against a rival standing on a card in **that rival's
own column**, and only the owner of a column can see its fronts (README, setup).
So every legal push takes a card the pusher cannot see: 4 gems, 4 dynamite and 2
dragons in a set of 10. On top of that the victim is not robbed -- a displaced
figure goes to the nearest pit cage, where it leaves the haul with every card it
has collected.

This script measures the exchange instead of arguing about it. It forks every
decision where a deterministic seat wants to push: one copy plays the push, the
other makes the same policy pick again with the push targets removed, and both
play out. The difference in that seat's final score is what the push was worth
there. The results are split afterwards by the front under the victim -- the
policy never sees it, so that split is analysis, not information.

    python probe_push_value.py --games 60 --continuations 3
"""

from __future__ import annotations

import argparse
import copy
import dataclasses

import numpy as np

from agents import NeuralAgent, PusherAgent
from game_env import DRAGON, DYNAMITE, GEM, MAX_COLUMNS, PIT_CAGE, MineEnv
from learning import PolicyNet
from selfplay import _choice_for, play_game

TRACKED = 0


def occupied_cells(ctx) -> set[tuple[int, int]]:
    return {
        (int(opponent.position), int(opponent.column))
        for opponent in ctx.opponents
        if not (opponent.escaped or opponent.dead) and opponent.column is not None
    }


def is_push(ctx, action: int) -> bool:
    """True when ``action`` lands on a card a rival is standing on."""
    row, column = divmod(int(action), MAX_COLUMNS)
    return int(action) < PIT_CAGE and (row, column) in occupied_cells(ctx)


def _front(env, row: int, column: int) -> int | None:
    if not 0 <= row < len(env.rows) or not 0 <= column < len(env.rows[row]):
        return None
    cell = env.rows[row][column]
    return None if cell is None else cell[0]


def _own_front(env) -> int | None:
    player = env.players[TRACKED]
    if player.column is None or not 0 <= player.position < int(PIT_CAGE):
        return None
    return _front(env, int(player.position), int(player.column))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/selfplay4.npz")
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--continuations", type=int, default=3)
    parser.add_argument(
        "--tell",
        default="any",
        choices=("any", "held"),
        help="which rivals the tracked seat is willing to push",
    )
    args = parser.parse_args()

    net = PolicyNet.load(args.checkpoint)
    rows: list[tuple[int | None, int | None, list[float], int]] = []

    for seed in range(args.games):
        env = MineEnv(opponents=args.players - 1)
        env.reset(seed=seed)
        agents = [
            NeuralAgent(net, seed=seed * 31 + seat, temperature=0.7)
            for seat in range(args.players)
        ]
        agents[TRACKED] = PusherAgent(tell=args.tell, before_planning=True)
        steps = 0
        forked = False
        while not env.game_finished and steps < 4000 and not forked:
            steps += 1
            if env.tie_breaker_contenders:
                break
            if any(player.blast_pending and player.blast_cards for player in env.players):
                break
            seat = env.planning_player
            ctx = env.player_action_context(seat)
            if not ctx.legal_actions:
                break
            action = int(_choice_for(agents[seat], ctx).action)
            if seat == TRACKED and is_push(ctx, action):
                row, column = divmod(action, MAX_COLUMNS)
                taken = _front(env, row, column)
                own = _own_front(env)
                victim = env._card_occupant(row, column)
                victim_cards = (
                    0 if victim is None else len(env.players[victim].collected_cards)
                )
                alternative_ctx = dataclasses.replace(
                    ctx,
                    legal_actions=tuple(
                        candidate
                        for candidate in ctx.legal_actions
                        if not is_push(ctx, candidate)
                    ),
                )
                deltas: list[float] = []
                for continuation in range(args.continuations):
                    # Re-roll the rivals' sampling identically for both arms, so a
                    # pair differs in the one decision and nothing else.
                    fork = copy.deepcopy(agents)
                    for seat in range(args.players):
                        if seat != TRACKED:
                            fork[seat] = NeuralAgent(
                                net,
                                seed=seed * 97 + seat * 13 + continuation,
                                temperature=0.7,
                            )
                    try:
                        env_a = copy.deepcopy(env)
                        agents_a = copy.deepcopy(fork)
                        env_a.step_one_player(action, force_action=True)
                        result_a = play_game(agents_a, seed=seed, env=env_a)

                        env_b = copy.deepcopy(env)
                        agents_b = copy.deepcopy(fork)
                        alternative = int(
                            _choice_for(agents[TRACKED], alternative_ctx).action
                        )
                        env_b.step_one_player(alternative, force_action=True)
                        result_b = play_game(agents_b, seed=seed, env=env_b)
                    except (ValueError, IndexError):
                        continue
                    deltas.append(
                        float(result_a.scores[TRACKED] - result_b.scores[TRACKED])
                    )
                if deltas:
                    rows.append((taken, own, deltas, victim_cards))
                forked = True
            env.step_one_player(action, force_action=True)

    def show(label: str, subset) -> None:
        if not subset:
            return
        values = [value for row in subset for value in row[2]]
        spread = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        print(
            f"{label:34s} positions {len(subset):3d}  value of the push "
            f"{float(np.mean(values)):+.2f} (se {spread:.2f})"
        )

    print(f"checkpoint {args.checkpoint}, {args.games} games, tell={args.tell}")
    show("all pushes", rows)
    for front, label in ((GEM, "gem"), (DYNAMITE, "dynamite"), (DRAGON, "dragon")):
        show(f"  ...it took a {label}", [row for row in rows if row[0] == front])
    show("  ...it gave up its own card", [row for row in rows if row[1] is not None])
    show("  ...it was standing on nothing", [row for row in rows if row[1] is None])
    if rows:
        hands = [row[3] for row in rows]
        print(
            f"victim's collected cards at the push: mean {np.mean(hands):.1f}, "
            f"empty-handed {100 * sum(1 for hand in hands if hand == 0) / len(hands):.0f}%"
        )


if __name__ == "__main__":
    main()
