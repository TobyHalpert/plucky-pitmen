"""A trained seat against three opponents that never pass up a displacement.

The scripted seats act on a legal occupied dig even when banking, planning or a
known-gem opportunity would otherwise come first. They prefer the rival with the
most total points. The fallback behaviour is the deterministic counter policy.

Every arm puts one neural seat in a rotating position and the other three seats
on the always-pusher policy. Counts use successful engine displacement events,
not merely attempted actions.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from agents import Agent, CounterAgent, NeuralAgent
from game_env import MAX_COLUMNS, MineEnv, PIT_CAGE
from learning import PolicyNet
from selfplay import play_game


class AlwaysDisplaceAgent(Agent):
    name = "always_pusher"

    def __init__(self) -> None:
        self.fallback = CounterAgent()
        self.pushes = 0

    def act(self, ctx):
        candidates = []
        for action in ctx.legal_actions:
            action = int(action)
            if action >= PIT_CAGE:
                continue
            row, column = divmod(action, MAX_COLUMNS)
            if row >= len(ctx.public_rows) or column >= len(ctx.public_rows[row]):
                continue
            occupant = next(
                (
                    opponent
                    for opponent in ctx.opponents
                    if not (opponent.escaped or opponent.dead)
                    and opponent.column is not None
                    and int(opponent.position) == row
                    and int(opponent.column) == column
                ),
                None,
            )
            if occupant is None:
                continue
            # A legal occupied dig is a displacement under the current rule.
            # The extra guard documents that contract and keeps the script
            # honest if the legality test is ever relaxed.
            if int(occupant.player_index) != int(occupant.column):
                continue
            # Highest total score first, then the shallowest row, then action id.
            candidates.append((-int(occupant.score), int(row), action))
        if candidates:
            self.pushes += 1
            return int(min(candidates)[2])
        return int(self.fallback.act(ctx))


def audit(
    net,
    *,
    games: int,
    seed: int,
    label: str,
    temperature: float = 0.7,
):
    original_displace = MineEnv._displace
    original_step = MineEnv.step_one_player
    tracked = [0]
    counts = Counter()
    rows = []

    def displace_spy(self, row, column, player_index):
        before = len(self.displacement_events)
        original_displace(self, row, column, player_index)
        made = len(self.displacement_events) - before
        if not made:
            return
        counts["displacements_all"] += made
        if player_index == tracked[0]:
            counts["displacements_by_net"] += made
        else:
            counts["displacements_by_pushers"] += made
        if self.displacement_events[-1][0] == tracked[0]:
            counts["displacements_on_net"] += made

    def step_spy(self, action=None, *args, **kwargs):
        push_choice = None
        if (
            kwargs.get("force_action")
            and action is not None
            and self.planning_player == tracked[0]
            and self._legal(int(action), tracked[0])
        ):
            options = [
                int(candidate)
                for candidate in range(PIT_CAGE)
                if self._legal(candidate, tracked[0])
                and self._can_displace(*divmod(candidate, MAX_COLUMNS), tracked[0])
            ]
            if options:
                push_choice = int(action) in options
        result = original_step(self, action, *args, **kwargs)
        if push_choice is not None:
            counts["net_leader_agnostic_push_opportunities"] += 1
            counts["net_push_opportunities_taken"] += int(push_choice)
        return result

    MineEnv._displace = displace_spy
    MineEnv.step_one_player = step_spy
    try:
        for index in range(games):
            game_seed = seed + index
            seat = index % 4
            tracked[0] = seat
            agents = [AlwaysDisplaceAgent() for _ in range(4)]
            agents[seat] = NeuralAgent(
                net, seed=game_seed * 31 + seat, temperature=temperature
            )
            result = play_game(agents, seed=game_seed)
            own = int(result.scores[seat])
            rival = [int(score) for i, score in enumerate(result.scores) if i != seat]
            rows.append({
                "won": own > max(rival),
                "tied_top": own == max(rival),
                "score": own,
                "margin": own - float(np.mean(rival)),
                "gems": sum(haul.gems[seat] for haul in result.hauls),
                "pushes_by_net": 0,
                "pushes_on_net": 0,
                "aborted": bool(result.aborted),
            })
    finally:
        MineEnv._displace = original_displace
        MineEnv.step_one_player = original_step

    wins = np.asarray([row["won"] for row in rows], dtype=float)
    ties = np.asarray([row["tied_top"] for row in rows], dtype=float)
    scores = np.asarray([row["score"] for row in rows], dtype=float)
    margins = np.asarray([row["margin"] for row in rows], dtype=float)
    gems = np.asarray([row["gems"] for row in rows], dtype=float)
    return {
        "label": label,
        "games": games,
        "seed": seed,
        "outright_wins": int(wins.sum()),
        "outright_win_rate": float(wins.mean()),
        "tied_top_rate": float(ties.mean()),
        "mean_score": float(scores.mean()),
        "mean_margin": float(margins.mean()),
        "mean_gems": float(gems.mean()),
        **dict(counts),
        "pushes_by_net_per_game": counts["displacements_by_net"] / games,
        "pushes_by_pushers_per_game": counts["displacements_by_pushers"] / games,
        "pushes_on_net_per_game": counts["displacements_on_net"] / games,
        "net_push_opportunity_conversion": (
            counts["net_push_opportunities_taken"]
            / max(1, counts["net_leader_agnostic_push_opportunities"])
        ),
        "unfinished": sum(row["aborted"] for row in rows),
        "_wins": wins,
        "_margins": margins,
    }


def public(summary):
    return {key: value for key, value in summary.items() if not key.startswith("_")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=3300000)
    parser.add_argument("--baseline", default="checkpoints/selfplay4.npz")
    parser.add_argument("--candidate", default="checkpoints/selfplay4_last_haul.npz")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")

    base = audit(
        PolicyNet.load(args.baseline), games=args.games, seed=args.seed,
        label="baseline",
    )
    candidate = audit(
        PolicyNet.load(args.candidate), games=args.games, seed=args.seed,
        label="last_haul",
    )
    win_change = candidate["_wins"] - base["_wins"]
    margin_change = candidate["_margins"] - base["_margins"]
    report = {
        "arms": [public(base), public(candidate)],
        "paired": {
            "outright_win_change_percentage_points": float(win_change.mean() * 100),
            "win_give_or_take_percentage_points": float(
                win_change.std(ddof=1) / np.sqrt(len(win_change)) * 100
            ) if len(win_change) > 1 else None,
            "margin_change_points": float(margin_change.mean()),
            "margin_give_or_take": float(
                margin_change.std(ddof=1) / np.sqrt(len(margin_change))
            ) if len(margin_change) > 1 else None,
        },
        "definition": (
            "4 players, 3 hauls. One neural seat against three always-pushers. "
            "The neural seat rotates through all four positions. Pushers target "
            "the current highest total score, then the shallowest legal card."
        ),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n")


if __name__ == "__main__":
    main()
