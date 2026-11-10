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
there -- and the difference in the *victim's* score is what being pushed was
worth to it, which is the number the proposed "push it to the next card" rule
moves. The results are split afterwards by the front under the victim -- the
policy never sees it, so that split is analysis, not information.

    python probe_push_value.py --games 60 --continuations 3
    python probe_push_value.py --games 60 --displacement next-card
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
    parser.add_argument(
        "--displacement",
        default="cage",
        choices=("cage", "next-card"),
        help="cage: the printed rule. next-card: DISPLACEMENT_STEPS_TO_THE_NEXT_CARD",
    )
    parser.add_argument(
        "--compare-rules",
        action="store_true",
        help=(
            "from the same position and the same push, play it out under both "
            "displacement rules: the only difference left is where the victim goes"
        ),
    )
    args = parser.parse_args()

    import game_env

    want_next = args.displacement == "next-card"
    game_env.DISPLACEMENT_STEPS_TO_THE_NEXT_CARD = want_next
    net = PolicyNet.load(args.checkpoint)
    #: (front taken, front given up, pusher deltas, victim deltas, victim's hand)
    rows: list[tuple[int | None, int | None, list[float], list[float], int]] = []
    rule_rows: list[tuple[float, float, int]] = []

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
                victim_deltas: list[float] = []
                rule_deltas: list[float] = []
                rule_victim: list[float] = []
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
                    if args.compare_rules:
                        # Same position, same push, the other displacement rule:
                        # the only thing left to differ is where the victim lands.
                        try:
                            env_c = copy.deepcopy(env)
                            agents_c = copy.deepcopy(fork)
                            game_env.DISPLACEMENT_STEPS_TO_THE_NEXT_CARD = not want_next
                            env_c.step_one_player(action, force_action=True)
                            result_c = play_game(agents_c, seed=seed, env=env_c)
                        except (ValueError, IndexError):
                            pass
                        else:
                            sign = 1.0 if want_next else -1.0
                            rule_deltas.append(
                                sign
                                * float(
                                    result_a.scores[TRACKED] - result_c.scores[TRACKED]
                                )
                            )
                            if victim is not None:
                                rule_victim.append(
                                    sign
                                    * float(
                                        result_a.scores[victim] - result_c.scores[victim]
                                    )
                                )
                        finally:
                            game_env.DISPLACEMENT_STEPS_TO_THE_NEXT_CARD = want_next
                    deltas.append(
                        float(result_a.scores[TRACKED] - result_b.scores[TRACKED])
                    )
                    if victim is not None:
                        victim_deltas.append(
                            float(result_a.scores[victim] - result_b.scores[victim])
                        )
                if deltas:
                    rows.append((taken, own, deltas, victim_deltas, victim_cards))
                rule_rows.extend(
                    (move, hurt, victim_cards)
                    for move, hurt in zip(rule_deltas, rule_victim)
                )
                forked = True
            env.step_one_player(action, force_action=True)

    def _mean(values: list[float]) -> str:
        if not values:
            return "     --      "
        spread = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        return f"{float(np.mean(values)):+.2f} (se {spread:.2f})"

    def show(label: str, subset) -> None:
        if not subset:
            return
        pusher = [value for row in subset for value in row[2]]
        victim = [value for row in subset for value in row[3]]
        print(
            f"{label:34s} positions {len(subset):3d}  pusher {_mean(pusher)}"
            f"  victim {_mean(victim)}"
        )

    print(
        f"checkpoint {args.checkpoint}, {args.games} games, tell={args.tell}, "
        f"displacement={args.displacement}"
    )
    show("all pushes", rows)
    for front, label in ((GEM, "gem"), (DYNAMITE, "dynamite"), (DRAGON, "dragon")):
        show(f"  ...it took a {label}", [row for row in rows if row[0] == front])
    show("  ...it gave up its own card", [row for row in rows if row[1] is not None])
    show("  ...it was standing on nothing", [row for row in rows if row[1] is None])
    if args.compare_rules and rule_rows:
        move = [row[0] for row in rule_rows]
        hurt = [row[1] for row in rule_rows]
        print(
            "next-card MINUS printed rule, same position and same push: "
            f"pusher {_mean(move)}  victim {_mean(hurt)}"
        )
        for hand, label in ((0, "victim held nothing"), (1, "victim held cards")):
            subset = [
                row
                for row in rule_rows
                if (row[2] == 0) == (hand == 0)
            ]
            if subset:
                print(
                    f"  ...{label:22s} n={len(subset):4d}  "
                    f"pusher {_mean([row[0] for row in subset])}  "
                    f"victim {_mean([row[1] for row in subset])}"
                )
    if rows:
        hands = [row[4] for row in rows]
        print(
            f"victim's collected cards at the push: mean {np.mean(hands):.1f}, "
            f"empty-handed {100 * sum(1 for hand in hands if hand == 0) / len(hands):.0f}%"
        )


if __name__ == "__main__":
    main()
