"""Compare Row-0 checkpoints with paired matches and direct habit audits."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from agents import NeuralAgent
from game_env import MineEnv
from learning import PolicyNet
from selfplay import play_game
from train_row0_blast import audit_row0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="checkpoints/selfplay4_last_haul.npz")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--audit-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3600000)
    parser.add_argument("--out", default="reports/row0_candidate_evaluation.json")
    args = parser.parse_args()

    fixed = PolicyNet.load(args.baseline)
    original_begin, original_cancel = MineEnv.begin_blast, MineEnv._cancel_blast
    rows = []
    baseline_wins = None
    for path in (args.baseline, *args.candidates):
        if not Path(path).exists():
            continue
        net = PolicyNet.load(path)
        counts = Counter()
        wins, scores, margins = [], [], []
        tracked = [0]

        def begin(env, player_index):
            result = original_begin(env, player_index)
            counts["announced"] += int(player_index == tracked[0])
            return result

        def cancel(env, player_index):
            if player_index == tracked[0] and env.players[player_index].blast_pending:
                counts["cancelled"] += int(env.players[player_index].position == 0)
                counts["cancelled_any"] += 1
            return original_cancel(env, player_index)

        MineEnv.begin_blast, MineEnv._cancel_blast = begin, cancel
        try:
            for index in range(args.games):
                seed = args.seed + index
                seat = index % 4
                tracked[0] = seat
                agents = [
                    NeuralAgent(net if j == seat else fixed, seed=seed * 31 + j, temperature=0.7)
                    for j in range(4)
                ]
                result = play_game(agents, seed=seed)
                own = result.scores[seat]
                other = [score for j, score in enumerate(result.scores) if j != seat]
                win = float(own > max(other))
                wins.append(win)
                tied = own == max(other)
                counts["tied_top"] += int(tied)
                counts["unfinished"] += int(result.aborted)
                scores.append(own)
                margins.append(own - np.mean(other))
        finally:
            MineEnv.begin_blast, MineEnv._cancel_blast = original_begin, original_cancel

        win_array = np.asarray(wins)
        if baseline_wins is None:
            baseline_wins = win_array
        paired = win_array - baseline_wins
        audit = audit_row0(net, games=args.audit_games, seed=args.seed + 500000)
        rows.append({
            "checkpoint": path,
            "games": args.games,
            "outright_win_rate": float(np.mean(win_array)),
            "tied_top_rate": counts["tied_top"] / args.games,
            "score": float(np.mean(scores)),
            "margin": float(np.mean(margins)),
            "announced": counts["announced"],
            "cancelled_row0": counts["cancelled"],
            "cancelled_any": counts["cancelled_any"],
            "paired_win_change": float(np.mean(paired)),
            "paired_give_or_take": float(paired.std(ddof=1) / np.sqrt(args.games)) if args.games > 1 else 0.0,
            "fresh_all_seats_row0_audit": audit,
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
