"""What do the two most-behind players do when both are under one network?

The previous held-out audit changed one seat at a time.  This script instead
plays ordinary complete games with all four seats using the same checkpoint,
locks on the two lowest-scoring active players at the start of the final haul,
and measures how those two trailers act.  It is behavioural evidence, not a
causal proof: the two trailers are evaluated in the same game and can affect
each other.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agents import NeuralAgent
from game_env import MineEnv, PASS, PIT_CAGE, MAX_COLUMNS
from learning import PolicyNet
from selfplay import play_game


def _top_indices(env: MineEnv):
    scores = [player.score for player in env.players]
    top = max(scores)
    leaders = {index for index, score in enumerate(scores) if score == top}
    trailers = [index for index, score in enumerate(scores) if score < top]
    return scores, top, leaders, sorted(trailers, key=lambda i: (scores[i], i))[:2]


def _legal_leader_pushes(env: MineEnv, player_index: int, leaders):
    options = []
    for action in range(PIT_CAGE):
        if not env._legal(action, player_index):
            continue
        row, column = divmod(action, MAX_COLUMNS)
        if env._can_displace(row, column, player_index):
            victim = env._card_occupant(row, column)
            if victim in leaders:
                options.append(action)
    return options


def audit(net, *, games: int, seed: int, label: str):
    original_begin = MineEnv.begin_blast
    original_cancel = MineEnv._cancel_blast
    original_displace = MineEnv._displace
    original_step = MineEnv.step_one_player
    totals = Counter()
    traces = []

    for game_index in range(games):
        game_seed = seed + game_index
        state = {"final_haul": False, "pair": set(), "leaders": set(), "scores": []}

        def begin_spy(self, player_index):
            if state["final_haul"] and player_index in state["pair"]:
                totals["announcements"] += 1
                if self.round == 0:
                    totals["round0_announcements"] += 1
                    state.setdefault("round0_announcers", set()).add(player_index)
            return original_begin(self, player_index)

        def cancel_spy(self, player_index):
            if state["final_haul"] and player_index in state["pair"]:
                totals["cancellations"] += 1
            return original_cancel(self, player_index)

        def displace_spy(self, row, column, player_index):
            if self._can_displace(row, column, player_index):
                victim = self._card_occupant(row, column)
                scores, top, leaders, _pair = _top_indices(self)
                if (
                    state["final_haul"]
                    and player_index in state["pair"]
                    and scores[player_index] < top
                ):
                    totals["trailing_displacements"] += 1
                    if victim in leaders:
                        totals["trailing_to_leader_displacements"] += 1
                    if victim in state["pair"]:
                        totals["trailer_displaces_trailer"] += 1
            return original_displace(self, row, column, player_index)

        def step_spy(self, action=None, *args, **kwargs):
            # Detect the boundary before this first final-haul decision.
            if self.haul == 3 and not state["final_haul"]:
                scores, _top, leaders, pair = _top_indices(self)
                state.update(
                    final_haul=True,
                    pair=set(pair),
                    leaders=set(leaders),
                    scores=scores[:],
                    round0_announcers=set(),
                )
                totals["games_with_two_trailers"] += 1
                totals["final_haul_initial_pair_sum"] += sum(scores[i] for i in pair)
                totals["final_haul_initial_leader_sum"] += sum(scores[i] for i in leaders)

            pending = None
            push = None
            if (
                kwargs.get("force_action")
                and action is not None
                and state["final_haul"]
                and self._legal(int(action), self.planning_player)
            ):
                player_index = self.planning_player
                if player_index in state["pair"]:
                    scores, top, leaders, _pair = _top_indices(self)
                    if self.players[player_index].blast_pending:
                        pending = True
                    if scores[player_index] < top:
                        options = _legal_leader_pushes(self, player_index, leaders)
                        if options:
                            push = options
            result = original_step(self, action, *args, **kwargs)

            # Only commit the pre-action observation if this step actually applied.
            if pending is not None:
                totals["pending_decisions"] += 1
                totals["pending_kept"] += int(int(action) == PASS)
                totals["pending_cancelled"] += int(int(action) != PASS)
            if push is not None:
                totals["leader_push_opportunities"] += 1
                totals["leader_pushes_taken"] += int(int(action) in push)
            return result

        MineEnv.begin_blast = begin_spy
        MineEnv._cancel_blast = cancel_spy
        MineEnv._displace = displace_spy
        MineEnv.step_one_player = step_spy
        try:
            agents = [
                NeuralAgent(net, seed=game_seed * 31 + index, temperature=0.7)
                for index in range(4)
            ]
            result = play_game(agents, seed=game_seed)
        finally:
            MineEnv.begin_blast = original_begin
            MineEnv._cancel_blast = original_cancel
            MineEnv._displace = original_displace
            MineEnv.step_one_player = original_step

        pair = state["pair"]
        scores = result.scores
        totals["games"] += 1
        if not result.aborted and pair:
            top = max(scores)
            totals["completed_games_with_two_trailers"] += 1
            totals["trailer_outright_wins"] += sum(
                1 for seat in pair if scores[seat] == top
                and scores.count(scores[seat]) == 1
            )
            totals["trailer_top_ties"] += sum(
                1 for seat in pair if scores[seat] == top and scores.count(scores[seat]) > 1
            )
            totals["trailer_final_score_sum"] += sum(scores[seat] for seat in pair)
            totals["games_where_pair_reaches_top"] += int(
                any(scores[seat] == top for seat in pair)
            )
            totals["fired_blast_hauls"] += sum(
                len(set(haul.blasts) & pair) for haul in result.hauls if result.hauls
            )
            totals["games_with_pair_blast_fire"] += int(
                any(set(haul.blasts) & pair for haul in result.hauls)
            )
            round0_announcers = state.get("round0_announcers", set())
            totals["games_both_round0_announced"] += int(len(round0_announcers) == 2)
            totals["games_no_round0_announcement"] += int(len(round0_announcers) == 0)
            traces.append({"seed": game_seed, "initial_scores": state["scores"],
                           "final_scores": scores[:], "pair": sorted(pair),
                           "round0_announcers": sorted(round0_announcers)})
    counts = dict(totals)
    branches = counts.get("completed_games_with_two_trailers", 0)
    pending = counts.get("pending_decisions", 0)
    opportunities = counts.get("leader_push_opportunities", 0)
    summary = {
        "label": label,
        "games": counts.get("games", 0),
        "games_with_two_trailers_at_final_haul": counts.get("games_with_two_trailers", 0),
        "analysed_games": branches,
        "mean_final_score_per_trailer": counts.get("trailer_final_score_sum", 0) / max(1, 2 * branches),
        "outright_win_rate_per_trailer": counts.get("trailer_outright_wins", 0) / max(1, 2 * branches),
        "tied_top_rate_per_trailer": counts.get("trailer_top_ties", 0) / max(1, 2 * branches),
        "games_where_at_least_one_trailer_top": counts.get("games_where_pair_reaches_top", 0),
        "announcements": counts.get("announcements", 0),
        "round0_announcements": counts.get("round0_announcements", 0),
        "cancellations": counts.get("cancellations", 0),
        "pending_decisions": pending,
        "pending_kept_rate": counts.get("pending_kept", 0) / max(1, pending),
        "pending_cancelled": counts.get("pending_cancelled", 0),
        "fired_blast_hauls": counts.get("fired_blast_hauls", 0),
        "games_with_trailer_blast_fire": counts.get("games_with_pair_blast_fire", 0),
        "games_both_announce_round0": counts.get("games_both_round0_announced", 0),
        "games_neither_announces_round0": counts.get("games_no_round0_announcement", 0),
        "trailing_displacements": counts.get("trailing_displacements", 0),
        "trailing_to_leader_displacements": counts.get("trailing_to_leader_displacements", 0),
        "trailer_displaces_trailer": counts.get("trailer_displaces_trailer", 0),
        "leader_push_opportunities": opportunities,
        "leader_pushes_taken": counts.get("leader_pushes_taken", 0),
        "leader_push_taken_rate": counts.get("leader_pushes_taken", 0) / max(1, opportunities),
        "unfinished": counts.get("unfinished", 0),
    }
    return summary, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2900000)
    parser.add_argument("--out")
    parser.add_argument("--trace-out")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")

    report = {}
    traces = {}
    for label, path in (
        ("baseline", "checkpoints/selfplay4.npz"),
        ("last_haul", "checkpoints/selfplay4_last_haul.npz"),
    ):
        net = PolicyNet.load(path)
        summary, game_traces = audit(net, games=args.games, seed=args.seed, label=label)
        report[label] = summary
        traces[label] = game_traces
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n")
    if args.trace_out:
        Path(args.trace_out).write_text(json.dumps(traces, indent=2) + "\n")


if __name__ == "__main__":
    main()
