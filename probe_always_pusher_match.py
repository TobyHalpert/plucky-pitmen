"""A trained seat against three opponents that never pass up a displacement.

The scripted seats act on a legal occupied dig even when banking, planning or a
known-gem opportunity would otherwise come first. They prefer the rival with the
most total points. The fallback behaviour is the deterministic counter policy.

Every arm puts one neural seat against three always-pushers. By default the net
rotates through all seats; ``--fixed-p1-first`` instead keeps it in Player 1's
seat and makes it the first player to act. The probe records own-column Gem
choices and whether a successful push displaced that choice before execution.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from agents import AlwaysDisplaceAgent, NeuralAgent
from features import split_action
from game_env import GEM, MAX_COLUMNS, MineEnv, PASS, PIT_CAGE
from learning import PolicyNet
from selfplay import play_game


def audit(
    net,
    *,
    games: int,
    seed: int,
    label: str,
    temperature: float = 0.7,
    fixed_p1_first: bool = False,
):
    original_displace = MineEnv._displace
    original_step = MineEnv.step_one_player
    original_execute = MineEnv._execute_round
    tracked = [0]
    states: dict[int, dict] = {}
    counts = Counter()
    rows = []

    def displace_spy(self, row, column, player_index):
        state = states.get(id(self))
        occupant = self._card_occupant(row, column)
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

        if state is None or occupant != state["seat"]:
            return
        event = state["current_gem"]
        if event is not None and (
            event["haul"], event["round"], event["row"], event["column"]
        ) == (self.haul, self.round, row, column):
            event["displaced"] = True
            event["pusher"] = int(player_index)
        opening = state["opening_event"]
        if opening is not None and (
            opening["haul"], opening["round"], opening["row"], opening["column"]
        ) == (self.haul, self.round, row, column):
            opening["displaced"] = True
            opening["pusher"] = int(player_index)
            state["opening"]["displaced"] = True

    def step_spy(self, action=None, *args, **kwargs):
        push_choice = None
        state = states.get(id(self))
        seat = tracked[0]
        if (
            kwargs.get("force_action")
            and action is not None
            and self.planning_player == seat
            and self._legal(int(action), seat)
        ):
            primitive, _announces_blast = split_action(int(action))
            options = [
                int(candidate)
                for candidate in range(PIT_CAGE)
                if self._legal(candidate, seat)
                and self._can_displace(*divmod(candidate, MAX_COLUMNS), seat)
            ]
            if options:
                push_choice = primitive in options

            if state is not None:
                opening_now = (
                    state["opening"] is None
                    and self.haul == 1
                    and self.round == 0
                )
                if opening_now:
                    state["opening"] = {
                        "action": primitive,
                        "own_gem": False,
                        "displaced": False,
                        "stolen": False,
                    }

                # Passing holds the current destination; any other action
                # replaces it. A Gem event remains in history so a push can be
                # linked to the exact choice even after P1 gets a forced turn.
                if primitive != int(PASS):
                    state["current_gem"] = None
                    if 0 <= primitive < PIT_CAGE:
                        row, column = divmod(primitive, MAX_COLUMNS)
                        entry = self.rows[row][column]
                        if (
                            column == seat
                            and entry is not None
                            and entry[0] == GEM
                        ):
                            event = {
                                "haul": int(self.haul),
                                "round": int(self.round),
                                "row": int(row),
                                "column": int(column),
                                "displaced": False,
                                "stolen": False,
                            }
                            state["gem_choices"].append(event)
                            state["current_gem"] = event
                            if opening_now:
                                state["opening"].update(
                                    own_gem=True,
                                    row=int(row),
                                    column=int(column),
                                )
                                state["opening_event"] = event

        result = original_step(self, action, *args, **kwargs)
        if push_choice is not None:
            counts["net_leader_agnostic_push_opportunities"] += 1
            counts["net_push_opportunities_taken"] += int(push_choice)
        return result

    def execute_spy(self):
        state = states.get(id(self))
        if state is not None:
            seat = state["seat"]
            seen_cells = set()
            for event in state["gem_choices"]:
                if (
                    not event["displaced"]
                    or event["haul"] != self.haul
                    or event["round"] != self.round
                ):
                    continue
                cell = (event["haul"], event["round"], event["row"], event["column"])
                if cell in seen_cells:
                    continue
                seen_cells.add(cell)
                card = self.rows[event["row"]][event["column"]]
                occupant = (
                    self._card_occupant(event["row"], event["column"])
                    if card is not None and card[0] == GEM
                    else None
                )
                if (
                    occupant is not None
                    and occupant != seat
                    and not self.players[occupant].displaced
                ):
                    event["stolen"] = True
                    state["stolen_cells"].add(cell)
                    if event is state["opening_event"]:
                        state["opening"]["stolen"] = True
        return original_execute(self)

    MineEnv._displace = displace_spy
    MineEnv.step_one_player = step_spy
    MineEnv._execute_round = execute_spy
    try:
        for index in range(games):
            game_seed = seed + index
            seat = 0 if fixed_p1_first else index % 4
            tracked[0] = seat
            agents = [AlwaysDisplaceAgent() for _ in range(4)]
            agents[seat] = NeuralAgent(
                net, seed=game_seed * 31 + seat, temperature=temperature
            )
            env = MineEnv(opponents=3)
            env.reset(seed=game_seed)
            if fixed_p1_first:
                env.starting_player = seat
                env.planning_player = seat
            states[id(env)] = {
                "seat": seat,
                "opening": None,
                "opening_event": None,
                "gem_choices": [],
                "current_gem": None,
                "stolen_cells": set(),
            }
            result = play_game(agents, seed=game_seed, env=env)
            state = states.pop(id(env))
            own = int(result.scores[seat])
            rival = [
                int(score)
                for player, score in enumerate(result.scores)
                if player != seat
            ]
            opening = state["opening"] or {}
            gem_choices = state["gem_choices"]
            rows.append({
                "won": own > max(rival),
                "tied_top": own == max(rival),
                "score": own,
                "margin": own - float(np.mean(rival)),
                "gems": sum(haul.gems[seat] for haul in result.hauls),
                "opening_own_gem": bool(opening.get("own_gem", False)),
                "opening_gem_displaced": bool(opening.get("displaced", False)),
                "opening_gem_stolen": bool(opening.get("stolen", False)),
                "own_gem_choices": len(gem_choices),
                "own_gem_choices_displaced": sum(
                    bool(event["displaced"]) for event in gem_choices
                ),
                "own_gem_cards_stolen": len(state["stolen_cells"]),
                "aborted": bool(result.aborted),
            })
    finally:
        MineEnv._displace = original_displace
        MineEnv.step_one_player = original_step
        MineEnv._execute_round = original_execute

    wins = np.asarray([row["won"] for row in rows], dtype=float)
    ties = np.asarray([row["tied_top"] for row in rows], dtype=float)
    scores = np.asarray([row["score"] for row in rows], dtype=float)
    margins = np.asarray([row["margin"] for row in rows], dtype=float)
    gems = np.asarray([row["gems"] for row in rows], dtype=float)
    opening_choices = sum(row["opening_own_gem"] for row in rows)
    opening_displaced = sum(row["opening_gem_displaced"] for row in rows)
    opening_stolen = sum(row["opening_gem_stolen"] for row in rows)
    all_gem_choices = sum(row["own_gem_choices"] for row in rows)
    all_gem_displaced = sum(row["own_gem_choices_displaced"] for row in rows)
    all_gem_stolen = sum(row["own_gem_cards_stolen"] for row in rows)
    return {
        "label": label,
        "games": games,
        "seed": seed,
        "seat_setup": "P1 fixed and first to act" if fixed_p1_first else "neural seat rotates",
        "outright_wins": int(wins.sum()),
        "outright_win_rate": float(wins.mean()),
        "tied_top_rate": float(ties.mean()),
        "mean_score": float(scores.mean()),
        "mean_margin": float(margins.mean()),
        "mean_gems": float(gems.mean()),
        "opening_own_column_gem_choices": int(opening_choices),
        "opening_own_column_gem_choice_rate": float(opening_choices / games),
        "opening_gem_displaced_before_resolution": int(opening_displaced),
        "opening_gem_displacement_given_choice": float(
            opening_displaced / max(1, opening_choices)
        ),
        "opening_gem_stolen_by_pusher_at_resolution": int(opening_stolen),
        "opening_gem_theft_given_choice": float(
            opening_stolen / max(1, opening_choices)
        ),
        "all_own_column_gem_choices": int(all_gem_choices),
        "all_own_column_gem_displaced": int(all_gem_displaced),
        "all_own_column_gem_displacement_rate": float(
            all_gem_displaced / max(1, all_gem_choices)
        ),
        "displaced_gem_cards_stolen": int(all_gem_stolen),
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
    parser.add_argument("--candidate", default="checkpoints/selfplay4_pushaware.npz")
    parser.add_argument("--fixed-p1-first", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")

    base = audit(
        PolicyNet.load(args.baseline), games=args.games, seed=args.seed,
        label="baseline", fixed_p1_first=args.fixed_p1_first,
    )
    candidate = audit(
        PolicyNet.load(args.candidate), games=args.games, seed=args.seed,
        label="candidate", fixed_p1_first=args.fixed_p1_first,
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
            "Pushers take legal occupied-card displacements, target the highest-"
            "score victim then the shallowest row, and otherwise use CounterAgent. "
            "The audit counts a Gem choice as displaced only when the engine logs "
            "a successful push of that seat off the selected own-column cell. "
            "A separate theft count records whether a pusher occupies that Gem "
            "when execution resolves."
        ),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n")


if __name__ == "__main__":
    main()
