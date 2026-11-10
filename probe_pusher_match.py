"""Deterministic pusher against three nets under the current displacement rule.

Matching seeds and rotated seats for four controls. Counts actual successful
pushes, not intended actions. All agents see only their normal action context.
"""
from __future__ import annotations

import argparse
import json
import numpy as np

import game_env
from agents import CounterAgent, NeuralAgent, PusherAgent
from learning import PolicyNet
from selfplay import play_game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/selfplay4.npz')
    parser.add_argument('--games', type=int, default=400)
    parser.add_argument('--seed', type=int, default=424242)
    parser.add_argument('--score-aware', action='store_true')
    args = parser.parse_args()
    if args.games < 1:
        parser.error('--games must be positive')
    net = PolicyNet.load(args.checkpoint)
    original = game_env.MineEnv._displace
    print(json.dumps(dict(checkpoint=args.checkpoint, games=args.games,
                         players=4, hauls=3, seed=args.seed,
                         next_card=game_env.DISPLACEMENT_STEPS_TO_THE_NEXT_CARD,
                         net_temperature=0.7)))
    controls = [
        ('no pushing', CounterAgent),
        ('old any-tell pusher', lambda: PusherAgent(tell='any', before_planning=False)),
    ]
    if args.score_aware:
        controls += [
            ('prefer top scorer', lambda: PusherAgent(tell='any', before_planning=False,
                                                     target='top-score')),
            ('only attack a leader', lambda: PusherAgent(tell='any', before_planning=False,
                                                       target='leader-only')),
        ]
    else:
        controls += [
            ('pusher held (default)', PusherAgent),
            ('pusher any', lambda: PusherAgent(tell='any')),
        ]
    for label, factory in controls:
        pushes = [0]
        tracked = [0]
        def displace(env, row, column, player_index):
            before = len(env.displacement_events)
            original(env, row, column, player_index)
            if player_index == tracked[0]:
                pushes[0] += len(env.displacement_events) - before
        game_env.MineEnv._displace = displace
        wins = ties = unfinished = 0
        scores, rivals, margins, gems = [], [], [], []
        try:
            for index in range(args.games):
                seed = args.seed + index
                seat = index % 4
                tracked[0] = seat
                agents = [NeuralAgent(net, seed=seed * 31 + i, temperature=0.7)
                          for i in range(4)]
                agents[seat] = factory()
                result = play_game(agents, seed=seed)
                own = result.scores[seat]
                other = [s for i, s in enumerate(result.scores) if i != seat]
                wins += int(own > max(other))
                ties += int(own == max(other))
                unfinished += int(result.aborted)
                scores.append(own)
                rivals.append(float(np.mean(other)))
                margins.append(own - float(np.mean(other)))
                gems.append(sum(h.gems[seat] for h in result.hauls))
        finally:
            game_env.MineEnv._displace = original
        if label == 'no pushing':
            baseline = np.array(margins)
        paired = np.array(margins) - baseline
        print(json.dumps(dict(agent=label, outright_wins=wins, tied_top=ties,
                              win_rate=wins / args.games,
                              mean_score=float(np.mean(scores)),
                              mean_rival_score=float(np.mean(rivals)),
                              margin=float(np.mean(margins)),
                              actual_pushes=pushes[0],
                              banked_gems_per_game=float(np.mean(gems)),
                              unfinished=unfinished,
                              paired_margin_change=float(np.mean(paired)),
                              paired_give_or_take=(float(np.std(paired, ddof=1) /
                                                        np.sqrt(args.games))
                                                  if args.games > 1 else None))), flush=True)


if __name__ == '__main__':
    main()
