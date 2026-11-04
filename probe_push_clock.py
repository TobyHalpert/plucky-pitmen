"""Who holds the clock in a planning phase where somebody has been pushed?

The complaint this measures: "A pushes D. B passes, C passes. Then D passes
without needing to signal a dragon." A displaced figure is carried somewhere
safe, keeps its decision, and spends that decision on the pass that closes the
round -- so the seat that was pushed controls when the phase ends, and the seats
still standing on cards have to cash out early.

Every decision of every round is logged, and the last decision before the
execution phase is the one that closed the round. Three rules are compared:

``printed``      a displaced figure goes to the pit cage (README)
``next-card``    it steps to the next card towards the lorries instead
                 (``game_env.DISPLACEMENT_STEPS_TO_THE_NEXT_CARD``)
``no-count``     printed, but a pass by a seat that was pushed this phase does
                 not tick the ``n - 1`` counter -- a diagnostic arm only, not a
                 proposal

    python probe_push_clock.py --games 150
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

import game_env
from agents import NeuralAgent, PusherAgent
from game_env import LORRY, PASS, PIT_CAGE
from learning import PolicyNet
from selfplay import play_game

DECISIONS: list[dict] = []
ORIGINAL_STEP = game_env.MineEnv.step_one_player
ORIGINAL_EXECUTE = game_env.MineEnv._execute_round


def _log(self, action: int) -> None:
    seat = int(self.planning_player)
    pushed = {event[0] for event in self.displacement_events}
    DECISIONS.append(
        {
            "haul": self.haul,
            "round": self.round,
            "seat": seat,
            "action": int(action),
            "pushed": seat in pushed,
            "closed": False,
        }
    )


def install(arm: str) -> None:
    """Instrument one rule arm. ``no-count`` also stops the pushed pass ticking."""

    def step(self, action, **kwargs):
        if arm == "no-count":
            seat = int(self.planning_player)
            was_pushed = any(event[0] == seat for event in self.displacement_events)
            if was_pushed and int(action) == PASS:
                before = self.consecutive_passes
                outcome = ORIGINAL_STEP(self, action, **kwargs)
                self.consecutive_passes = before
                _log(self, action)
                return outcome
        _log(self, action)
        return ORIGINAL_STEP(self, action, **kwargs)

    def execute(self):
        if DECISIONS:
            DECISIONS[-1]["closed"] = True
        return ORIGINAL_EXECUTE(self)

    game_env.MineEnv.step_one_player = step
    game_env.MineEnv._execute_round = execute


def run(net, games: int) -> Counter:
    DECISIONS.clear()
    stats: Counter = Counter()
    for seed in range(games):
        seat = seed % 4
        agents = [
            NeuralAgent(net, seed=seed * 31 + index, temperature=0.7)
            for index in range(4)
        ]
        agents[seat] = PusherAgent(tell="any", before_planning=False)
        start = len(DECISIONS)
        play_game(agents, seed=seed)
        log = DECISIONS[start:]
        rounds: dict[tuple[int, int], list[int]] = {}
        for index, entry in enumerate(log):
            rounds.setdefault((entry["haul"], entry["round"]), []).append(index)
        for indices in rounds.values():
            closing = log[indices[-1]]
            pushed = {
                entry["seat"]
                for entry in log
                if entry["pushed"]
                and (entry["haul"], entry["round"]) == (closing["haul"], closing["round"])
            }
            if not pushed:
                continue
            stats["rounds"] += 1
            if closing["seat"] in pushed and closing["action"] == PASS:
                stats["closed_on_pushed_pass"] += 1
            card_moves = [
                index
                for index in indices
                if log[index]["action"] < PIT_CAGE and log[index]["seat"] not in pushed
            ]
            push_at = card_moves[-1] if card_moves else indices[0]
            rest = [log[index] for index in indices if index > push_at]
            stats["after"] += len(rest)
            stats["cage_after"] += sum(
                1 for entry in rest if entry["action"] in (PIT_CAGE, LORRY)
            )
            next_victim = next((entry for entry in rest if entry["pushed"]), None)
            if next_victim is not None:
                stats["victim_next_seen"] += 1
                if next_victim["action"] == PASS:
                    stats["victim_next_pass"] += 1
            if (
                len(rest) >= 3
                and [entry["action"] for entry in rest[:3]] == [PASS, PASS, PASS]
                and rest[2]["seat"] in pushed
                and rest[2]["closed"]
            ):
                stats["exact_pattern"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/selfplay4.npz")
    parser.add_argument("--games", type=int, default=150)
    args = parser.parse_args()
    net = PolicyNet.load(args.checkpoint)

    print(f"checkpoint {args.checkpoint}, {args.games} games, one pushing seat against three nets")
    print(
        f"{'rule':22s} {'pushed rounds':>13s} {'closed on pushed pass':>22s} "
        f"{'push->pass->pass->pushed pass':>31s} {'victim next act: pass':>22s} "
        f"{'planning left after push':>25s}"
    )
    for arm, rule in (
        ("printed", False),
        ("next-card", True),
        ("no-count", False),
    ):
        game_env.DISPLACEMENT_STEPS_TO_THE_NEXT_CARD = rule
        install(arm)
        stats = run(net, args.games)
        rounds = max(stats["rounds"], 1)
        seen = max(stats["victim_next_seen"], 1)
        print(
            f"{arm:22s} {stats['rounds']:13d} "
            f"{100 * stats['closed_on_pushed_pass'] / rounds:21.0f}% "
            f"{100 * stats['exact_pattern'] / rounds:30.0f}% "
            f"{100 * stats['victim_next_pass'] / seen:21.0f}% "
            f"{stats['after'] / rounds:24.2f}"
        )
    game_env.DISPLACEMENT_STEPS_TO_THE_NEXT_CARD = False


if __name__ == "__main__":
    main()
