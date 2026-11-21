"""Compare baseline and last-haul behaviour for the two trailers at the final haul.

For each game seed, this controller plays normally through the first two hauls
with ``selfplay4``.  At the exact final-haul boundary it forks two branches with
identical cards, hands, scores, planning order and agent RNG states:

* ``baseline`` -- all four seats stay on ``selfplay4``.
* ``candidate_pair`` -- only the two lowest-scoring seats use the last-haul net.

The audit describes those two seats' final-haul decisions.  No model is trained
or tuned on these seeds.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from agents import NeuralAgent
from game_env import DRAGON, MineEnv, PASS, PIT_CAGE
from gui_controller import GameController
from learning import PolicyNet
from probe_two_trailing_players import _legal_leader_pushes, _top_indices


def _advance_to_final_haul(controller: GameController, limit: int = 5000):
    for _ in range(limit):
        if controller.env.haul == 3 or controller.env.game_finished:
            break
        controller.step_one_player_auto()
    if controller.env.haul != 3 or controller.env.round != 0:
        return None
    _scores, _top, _leaders, pair = _top_indices(controller.env)
    return pair


def _run_branch(branch: GameController, pair):
    totals = Counter()
    round0_announcers = set()
    fired_before_scores = set()

    original_begin = MineEnv.begin_blast
    original_cancel = MineEnv._cancel_blast
    original_displace = MineEnv._displace
    original_resolve = MineEnv.resolve_blast
    original_step = MineEnv.step_one_player

    def begin_spy(self, player_index):
        if player_index in pair:
            totals["announcements"] += 1
            if self.round == 0:
                totals["round0_announcements"] += 1
                round0_announcers.add(player_index)
        return original_begin(self, player_index)

    def cancel_spy(self, player_index):
        if player_index in pair and self.players[player_index].blast_pending:
            totals["cancellations"] += 1
        return original_cancel(self, player_index)

    def resolve_spy(self, player_index, choice_index):
        player = self.players[player_index]
        about_to_succeed = (
            player_index in pair
            and player.blast_pending
            and len(player.blast_cards) == 2
        )
        chosen = original_resolve(self, player_index, choice_index)
        if about_to_succeed and chosen[0] != DRAGON:
            totals["successful_blast_resolutions"] += 1
            fired_before_scores.add(player_index)
        return chosen

    def displace_spy(self, row, column, player_index):
        if self._can_displace(row, column, player_index):
            victim = self._card_occupant(row, column)
            scores, top, leaders, _pair = _top_indices(self)
            if (
                player_index in pair
                and scores[player_index] < top
            ):
                totals["trailing_displacements"] += 1
                if victim in leaders:
                    totals["trailing_to_leader_displacements"] += 1
                if victim in pair:
                    totals["trailer_displaces_trailer"] += 1
        return original_displace(self, row, column, player_index)

    def step_spy(self, action=None, *args, **kwargs):
        pending = None
        push = None
        if (
            kwargs.get("force_action")
            and action is not None
            and self._legal(int(action), self.planning_player)
        ):
            player_index = self.planning_player
            if player_index in pair:
                scores, top, leaders, _pair = _top_indices(self)
                if self.players[player_index].blast_pending:
                    pending = True
                if scores[player_index] < top:
                    options = _legal_leader_pushes(self, player_index, leaders)
                    if options:
                        push = options
        result = original_step(self, action, *args, **kwargs)
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
    MineEnv.resolve_blast = resolve_spy
    MineEnv.step_one_player = step_spy
    try:
        for _ in range(5000):
            if branch.env.game_finished:
                break
            branch.step_one_player_auto()
    finally:
        MineEnv.begin_blast = original_begin
        MineEnv._cancel_blast = original_cancel
        MineEnv._displace = original_displace
        MineEnv.resolve_blast = original_resolve
        MineEnv.step_one_player = original_step

    scores = [player.score for player in branch.env.players]
    top = max(scores)
    totals["branches"] += 1
    totals["finished"] += int(branch.env.game_finished)
    totals["trailer_final_score_sum"] += sum(scores[i] for i in pair)
    totals["trailer_outright_wins"] += sum(
        1 for i in pair if scores[i] == top and scores.count(scores[i]) == 1
    )
    totals["trailer_top_ties"] += sum(
        1 for i in pair if scores[i] == top and scores.count(scores[i]) > 1
    )
    totals["branch_any_trailer_top"] += int(any(scores[i] == top for i in pair))
    return totals, scores, round0_announcers


def audit(*, games: int, seed: int, baseline_path: str, candidate_path: str):
    baseline = PolicyNet.load(baseline_path)
    candidate = PolicyNet.load(candidate_path)
    source = GameController(3, f"neural:{Path(baseline_path).stem}", seed, all_seats=True)
    source.reset(3, f"neural:{Path(baseline_path).stem}", seed, all_seats=True, advance=False)
    totals = {"baseline": Counter(), "candidate_pair": Counter()}
    details = []

    for index in range(games):
        game_seed = seed + index
        source.reset(3, f"neural:{Path(baseline_path).stem}", game_seed, all_seats=True, advance=False)
        pair = _advance_to_final_haul(source)
        if pair is None or len(pair) < 2:
            continue
        pair = set(int(x) for x in pair)

        rows = {}
        for arm in ("baseline", "candidate_pair"):
            branch = copy.copy(source)
            branch.env = copy.deepcopy(source.env)
            branch.agents = copy.deepcopy(source.agents)
            branch.last_choice = None
            branch.last_choice_seat = None
            for seat, agent in enumerate(branch.agents):
                agent.net = candidate if (arm == "candidate_pair" and seat in pair) else baseline
            counts, final_scores, round0 = _run_branch(branch, pair)
            rows[arm] = counts
            rows[f"{arm}_final"] = final_scores
            rows[f"{arm}_round0"] = sorted(round0)
        totals["baseline"].update(rows["baseline"])
        totals["candidate_pair"].update(rows["candidate_pair"])
        details.append({
            "seed": game_seed,
            "initial_scores": [player.score for player in source.env.players],
            "pair": sorted(pair),
            "baseline_final": rows["baseline_final"],
            "candidate_final": rows["candidate_pair_final"],
            "baseline_round0_announcers": rows["baseline_round0"],
            "candidate_round0_announcers": rows["candidate_pair_round0"],
        })

    output = {}
    for arm, counts in totals.items():
        branches = counts.get("branches", 0)
        pending = counts.get("pending_decisions", 0)
        opportunities = counts.get("leader_push_opportunities", 0)
        output[arm] = {
            "eligible_boundary_games": branches,
            "mean_final_score_per_trailer": counts.get("trailer_final_score_sum", 0) / max(1, 2 * branches),
            "outright_win_rate_per_trailer": counts.get("trailer_outright_wins", 0) / max(1, 2 * branches),
            "tied_top_rate_per_trailer": counts.get("trailer_top_ties", 0) / max(1, 2 * branches),
            "games_with_trailer_at_top": counts.get("branch_any_trailer_top", 0),
            "announcements": counts.get("announcements", 0),
            "round0_announcements": counts.get("round0_announcements", 0),
            "cancellations": counts.get("cancellations", 0),
            "successful_blast_resolutions": counts.get("successful_blast_resolutions", 0),
            "pending_decisions": pending,
            "pending_kept": counts.get("pending_kept", 0),
            "pending_kept_rate": counts.get("pending_kept", 0) / max(1, pending),
            "pending_cancelled": counts.get("pending_cancelled", 0),
            "trailing_displacements": counts.get("trailing_displacements", 0),
            "trailing_to_leader_displacements": counts.get("trailing_to_leader_displacements", 0),
            "trailer_displaces_trailer": counts.get("trailer_displaces_trailer", 0),
            "leader_push_opportunities": opportunities,
            "leader_pushes_taken": counts.get("leader_pushes_taken", 0),
            "leader_push_taken_rate": counts.get("leader_pushes_taken", 0) / max(1, opportunities),
            "finished": counts.get("finished", 0),
        }
    return output, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3100000)
    parser.add_argument("--baseline", default="checkpoints/selfplay4.npz")
    parser.add_argument("--candidate", default="checkpoints/selfplay4_last_haul.npz")
    parser.add_argument("--out")
    parser.add_argument("--details-out")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")
    summary, details = audit(
        games=args.games, seed=args.seed,
        baseline_path=args.baseline, candidate_path=args.candidate,
    )
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n")
    if args.details_out:
        Path(args.details_out).write_text(json.dumps(details, indent=2) + "\n")


if __name__ == "__main__":
    main()
