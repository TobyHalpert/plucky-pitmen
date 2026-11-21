"""Compare old and retrained nets in four-seat, three-haul mirror games.

Counts actual applied card moves and successful displacement events, not inferred
pushes or pass patterns. Both checkpoints play the current rule at temperature
0.7 with matching game and agent seeds. These are descriptive mirror profiles,
not head-to-head win rates.
"""
from __future__ import annotations

import argparse
from collections import Counter

from agents import NeuralAgent
from game_env import MAX_COLUMNS, PIT_CAGE, MineEnv
from learning import PolicyNet
from selfplay import play_game


def profile(path: str, games: int, seed: int) -> dict:
    net = PolicyNet.load(path)
    counts = Counter()
    original = MineEnv._apply_action

    def apply(env, player_index, action):
        before = len(env.displacement_events)
        original(env, player_index, action)
        if action < PIT_CAGE:
            counts['card_moves'] += 1
            counts['own_column_moves'] += int(action % MAX_COLUMNS == player_index)
        counts['pushes'] += len(env.displacement_events) - before

    MineEnv._apply_action = apply
    try:
        for index in range(games):
            agents = [NeuralAgent(net, seed=(seed + index) * 31 + seat,
                                  temperature=0.7) for seat in range(4)]
            result = play_game(agents, seed=seed + index)
            counts['aborted'] += int(result.aborted)
            counts['turns'] += result.turns
            counts['scores'] += sum(result.scores)
            counts['hauls'] += len(result.hauls)
            for haul in result.hauls:
                counts['gems'] += sum(haul.gems)
    finally:
        MineEnv._apply_action = original
    return dict(checkpoint=path, games=games,
                mean_seat_score=counts['scores'] / (games * 4),
                pushes_per_game=counts['pushes'] / games,
                card_moves_per_game=counts['card_moves'] / games,
                own_column_move_share=counts['own_column_moves'] / max(1, counts['card_moves']),
                gems_per_haul=counts['gems'] / max(1, counts['hauls']),
                decisions_per_game=counts['turns'] / games,
                unfinished_games=counts['aborted'])


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--seed', type=int, default=424242)
    parser.add_argument('--checkpoints', default='checkpoints/selfplay4_pitcage.npz,checkpoints/selfplay4.npz')
    args = parser.parse_args()
    print(json.dumps([profile(p, args.games, args.seed)
                      for p in args.checkpoints.split(',')], indent=2))


if __name__ == '__main__':
    main()
