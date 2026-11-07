"""Fresh-game comparison against three unchanged nets, with blast behaviour."""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from agents import NeuralAgent
from game_env import MineEnv
from learning import PolicyNet
from selfplay import play_game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='checkpoints/selfplay4.npz')
    parser.add_argument('--candidate', default='checkpoints/selfplay4_tactical.npz')
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1900000)
    parser.add_argument('--out', default='reports/tactical_evaluation.json')
    args = parser.parse_args()
    if args.games < 1:
        parser.error('games must be positive')
    fixed = PolicyNet.load(args.baseline)
    original_begin = MineEnv.begin_blast
    original_cancel = MineEnv._cancel_blast
    rows = []
    for path in (args.baseline, args.candidate):
        net = PolicyNet.load(path)
        counts = Counter()
        scores, margins, wins = [], [], []
        tracked = [0]
        def begin(env, player_index):
            result = original_begin(env, player_index)
            if player_index == tracked[0]:
                counts['announced'] += 1
            return result
        def cancel(env, player_index):
            if player_index == tracked[0] and env.players[player_index].blast_pending:
                counts['cancelled'] += 1
            return original_cancel(env, player_index)
        MineEnv.begin_blast, MineEnv._cancel_blast = begin, cancel
        try:
            for i in range(args.games):
                seed = args.seed + i
                seat = i % 4
                tracked[0] = seat
                agents = [NeuralAgent(fixed, seed=seed * 31 + j, temperature=0.7)
                          for j in range(4)]
                agents[seat] = NeuralAgent(net, seed=seed * 31 + seat, temperature=0.7)
                result = play_game(agents, seed=seed)
                own = result.scores[seat]
                other = [s for j, s in enumerate(result.scores) if j != seat]
                wins.append(float(own > max(other)))
                counts['tied_top'] += int(own == max(other))
                counts['unfinished'] += int(result.aborted)
                counts['blast_hauls'] += sum(seat in h.blasts for h in result.hauls)
                counts['decisions'] += result.turns
                scores.append(own)
                margins.append(own - np.mean(other))
        finally:
            MineEnv.begin_blast, MineEnv._cancel_blast = original_begin, original_cancel
        if not rows:
            baseline_wins = np.array(wins)
        delta = np.array(wins) - baseline_wins
        rows.append(dict(checkpoint=path, games=args.games, seed=args.seed,
                         outright_win_rate=float(np.mean(wins)),
                         tied_top_rate=counts['tied_top'] / args.games,
                         score=float(np.mean(scores)), margin=float(np.mean(margins)),
                         announced=counts['announced'], cancelled=counts['cancelled'],
                         hauls_with_own_blast_fired=counts['blast_hauls'],
                         unfinished=counts['unfinished'],
                         decisions_per_game=counts['decisions'] / args.games,
                         paired_win_change=float(delta.mean()),
                         paired_give_or_take=float(delta.std(ddof=1) / np.sqrt(args.games))
                         if args.games > 1 else None))
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
