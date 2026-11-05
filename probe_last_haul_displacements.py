"""Measure displacement taken by the last-haul candidate in its held continuations.

This repeats the unforced held-out evaluation from ``train_last_haul.py`` while
instrumenting actual ``MineEnv._displace`` events.  Nothing is rewarded for
pushing; the audit only reports behaviour.

"Leader" means one unique maximum total score immediately before the action.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from agents import NeuralAgent
from game_env import MineEnv, MAX_COLUMNS, PIT_CAGE
from learning import PolicyNet
from selfplay import play_game
from train_last_haul import collect_last, sample_last_hidden


def _push_options(env: MineEnv, player_index: int):
    """Return the displacement choices that hit the unique point leader."""
    scores = [player.score for player in env.players]
    top = max(scores)
    leaders = [index for index, score in enumerate(scores) if score == top]
    if scores[player_index] >= top or len(leaders) != 1:
        return [], top, leaders
    leader = leaders[0]
    options = []
    for action in range(PIT_CAGE):
        if not env._legal(action, player_index):
            continue
        row, column = divmod(action, MAX_COLUMNS)
        if env._can_displace(row, column, player_index):
            victim = env._card_occupant(row, column)
            if victim == leader:
                options.append((int(action), row, column))
    return options, top, leaders


def audit(positions, baseline, candidate, *, repeats: int, base_seed: int):
    original_displace = MineEnv._displace
    original_step = MineEnv.step_one_player
    records = defaultdict(list)

    for policy_name, net in (("baseline", baseline), ("last_haul", candidate)):
        for position_index, (source, seat, _) in enumerate(positions):
            for repeat in range(repeats):
                seed = base_seed + position_index * 101 + repeat
                env = sample_last_hidden(source, seat, np.random.default_rng(seed))
                events = []
                opportunities = []

                def displace_spy(self, row, column, player_index):
                    if self._can_displace(row, column, player_index):
                        victim = self._card_occupant(row, column)
                        scores = [player.score for player in self.players]
                        top = max(scores)
                        leaders = [i for i, score in enumerate(scores) if score == top]
                        events.append({
                            "actor": player_index,
                            "victim": victim,
                            "tracked": player_index == seat,
                            "actor_trailing": scores[player_index] < top,
                            "victim_unique_leader": (
                                victim in leaders and len(leaders) == 1
                            ),
                            "scores": scores,
                            "round": self.round,
                        })
                    return original_displace(self, row, column, player_index)

                def step_spy(self, action=None, *args, **kwargs):
                    pending_opportunity = None
                    if kwargs.get("force_action") and action is not None:
                        player_index = self.planning_player
                        if self._legal(int(action), player_index):
                            options, _top, _leaders = _push_options(self, player_index)
                            if options:
                                option_ids = [option[0] for option in options]
                                pending_opportunity = {
                                    "actor": player_index,
                                    "tracked": player_index == seat,
                                    "options": option_ids,
                                    "taken": int(action) in option_ids,
                                    "round": self.round,
                                }
                    # ``play_game`` retries an invalid blast announcement.  Record
                    # the pre-action opportunity only if this step actually applies,
                    # so the retry does not count it twice.
                    result = original_step(self, action, *args, **kwargs)
                    if pending_opportunity is not None:
                        opportunities.append(pending_opportunity)
                    return result

                MineEnv._displace = displace_spy
                MineEnv.step_one_player = step_spy
                try:
                    agents = [
                        NeuralAgent(baseline, seed=seed * 31 + other, temperature=0.7)
                        for other in range(4)
                    ]
                    agents[seat] = NeuralAgent(
                        net, seed=seed * 31 + seat, temperature=0.7
                    )
                    result = play_game(agents, seed=seed, env=env)
                finally:
                    MineEnv._displace = original_displace
                    MineEnv.step_one_player = original_step

                own = result.scores[seat]
                other_scores = [
                    score for other, score in enumerate(result.scores) if other != seat
                ]
                records[policy_name].append({
                    "position": position_index,
                    "repeat": repeat,
                    "seat": seat,
                    "won": float(own > max(other_scores)),
                    "score": own,
                    "events": events,
                    "opportunities": opportunities,
                })

    summary = []
    for policy_name, continuations in records.items():
        all_events = [event for row in continuations for event in row["events"]]
        all_opps = [
            opportunity for row in continuations for opportunity in row["opportunities"]
        ]
        summary.append({
            "policy": policy_name,
            "continuations": len(continuations),
            "tracked_trailing_to_leader_pushes": sum(
                event["tracked"] and event["actor_trailing"]
                and event["victim_unique_leader"] for event in all_events
            ),
            "tracked_displacements_any_target": sum(event["tracked"] for event in all_events),
            "tracked_displacements_while_trailing": sum(
                event["tracked"] and event["actor_trailing"] for event in all_events
            ),
            "tracked_leader_push_opportunities": sum(
                opportunity["tracked"] for opportunity in all_opps
            ),
            "tracked_leader_pushes_taken": sum(
                opportunity["tracked"] and opportunity["taken"] for opportunity in all_opps
            ),
            "all_seats_trailing_to_leader_pushes": sum(
                event["actor_trailing"] and event["victim_unique_leader"]
                for event in all_events
            ),
            "all_seats_leader_push_opportunities": len(all_opps),
            "all_seats_leader_pushes_taken": sum(bool(opportunity["taken"]) for opportunity in all_opps),
            "continuations_with_tracked_leader_push": sum(
                any(event["tracked"] and event["actor_trailing"]
                    and event["victim_unique_leader"] for event in row["events"])
                for row in continuations
            ),
            "tracked_leader_push_positions": sorted({
                row["position"] for row in continuations if any(
                    event["tracked"] and event["actor_trailing"]
                    and event["victim_unique_leader"] for event in row["events"]
                )
            }),
        })

    deltas = []
    for bucket in ("all", "tracked_leader_push_branches"):
        baseline_wins = []
        candidate_wins = []
        for base_row, candidate_row in zip(records["baseline"], records["last_haul"]):
            if bucket != "all" and not any(
                event["tracked"] and event["actor_trailing"]
                and event["victim_unique_leader"] for event in candidate_row["events"]
            ):
                continue
            baseline_wins.append(base_row["won"])
            candidate_wins.append(candidate_row["won"])
        deltas.append({
            "branch_bucket": bucket,
            "branches": len(candidate_wins),
            "baseline_win_rate": float(np.mean(baseline_wins)) if baseline_wins else None,
            "last_haul_win_rate": float(np.mean(candidate_wins)) if candidate_wins else None,
        })
    return summary, deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="checkpoints/selfplay4.npz")
    parser.add_argument("--candidate", default="checkpoints/selfplay4_last_haul.npz")
    parser.add_argument("--positions", type=int, default=48)
    parser.add_argument("--position-seed", type=int, default=2400000)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--evaluation-seed", type=int, default=2800000)
    parser.add_argument("--out")
    args = parser.parse_args()
    if min(args.positions, args.repeats) < 1:
        parser.error("positions and repeats must be positive")

    baseline = PolicyNet.load(args.baseline)
    candidate = PolicyNet.load(args.candidate)
    positions = collect_last(baseline, args.positions, args.position_seed)
    summary, deltas = audit(
        positions, baseline, candidate, repeats=args.repeats, base_seed=args.evaluation_seed
    )
    report = {"summary": summary, "paired_branches": deltas, "config": vars(args)}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(rendered + "\n")


if __name__ == "__main__":
    main()
