"""Fork P2's seed-7 haul-2 push, blast, and cancellation decisions.

The pre-fork history is exactly the all-neural viewer replay. Each arm retains
that actual hidden board and deck; only future policy sampling is rerolled.
These are full-information *analysis* continuations, not information exposed to
P2 and not a belief-state estimate over unknown cards. Tied tops are reported
separately from outright final-score wins (no tie-break wins are counted).
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter

import numpy as np
import game_env
from analyze_game import transcript, scripted_replay
from features import action_mask
from game_env import PASS, PIT_CAGE


def evaluate(source, action, blast, samples, seed):
    scores = []
    aborted = 0
    next_p2 = Counter()
    for index in range(samples):
        branch = copy.copy(source)
        branch.env = copy.deepcopy(source.env)
        branch.agents = copy.deepcopy(source.agents)
        for seat, agent in enumerate(branch.agents):
            if hasattr(agent, 'rng'):
                agent.rng = np.random.default_rng(seed + index * 101 + seat)
        outcome = branch.env.step_one_player(action, announce_blast=blast, force_action=True)
        if outcome[-1].get('illegal_action'):
            raise ValueError('illegal branch action')
        steps = 0
        seen_next = False
        while not branch.env.game_finished and steps < 1200:
            before_turns = branch.env.planning_turns
            before_seat = branch.env.planning_player
            before_haul = branch.env.haul
            before_round = branch.env.round
            branch.step_one_player_auto()
            if (not seen_next and before_seat == 1
                    and before_haul == source.env.haul
                    and before_round == source.env.round
                    and branch.env.planning_turns != before_turns):
                choice = branch.last_choice
                if choice is not None:
                    next_p2[str(int(choice.action)) + ('+blast' if choice.announce_blast else '')] += 1
                    seen_next = True
            steps += 1
        aborted += int(not branch.env.game_finished)
        scores.append([p.score for p in branch.env.players])
    scores = np.asarray(scores)
    own = scores[:, 1]
    rivals = scores[:, [0, 2, 3]]
    wins = own > rivals.max(axis=1)
    tied = own == rivals.max(axis=1)
    return dict(outright_win_rate=float(wins.mean()), tied_top_rate=float(tied.mean()),
                mean_scores=scores.mean(axis=0).tolist(), unfinished=aborted,
                next_p2_actions=dict(next_p2)), wins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=400)
    parser.add_argument('--seed', type=int, default=920000)
    parser.add_argument('--stage', default='all', choices=('all', 'lorry', 'placement'))
    args = parser.parse_args()
    controller, events = transcript(7, 'neural:selfplay4', 3, False, all_seats=True)
    initial = next(i for i, e in enumerate(events)
                   if e['haul'] == 2 and e['round'] == 0 and e['seat'] == 1)
    cancel = next(i for i, e in enumerate(events)
                  if i > initial and e['haul'] == 2 and e['round'] == 0
                  and e['seat'] == 1 and e['ctx'].own_blast_pending)
    lorry = next(i for i, e in enumerate(events)
                 if i > cancel and e['haul'] == 2 and e['round'] == 0
                 and e['seat'] == 1)
    for stage, index, arms in (
        ('initial choice', initial, [('own gem, no blast', 1, False),
                                     ('own gem + blast (recorded)', 1, True),
                                     ('push P1, no blast', 0, False),
                                     ('push P1 + blast', 0, True),
                                     ('own row 2 + blast', 11, True)]),
        ('next P2 turn after recorded blast', cancel,
         [('cage, cancel blast (recorded)', int(PIT_CAGE), False),
          ('pass, carry out blast', int(PASS), False)]),
        ('after cancellation, already in cage', lorry,
         [('lorry (recorded)', int(game_env.LORRY), False),
          ('pass, leave empty (engine-legal, policy-filtered)', int(PASS), False)]),
    ):
        if args.stage == 'lorry' and index != lorry:
            continue
        if args.stage == 'placement' and index != initial:
            continue
        if args.stage == 'placement':
            arms = [('own row 0 + blast', 1, True),
                    ('own row 2 + blast', 11, True)]
        source, _ = scripted_replay(controller, events[:index], 7, play_out=False)
        assert source.env.planning_player == 1
        ctx = source.env.player_action_context(1)
        print(json.dumps(dict(stage=stage, samples=args.samples, continuation_seed=args.seed,
                             record_index=index, scores=[p.score for p in source.env.players],
                             own_blast_pending=ctx.own_blast_pending,
                             policy_pass_allowed=bool(action_mask(ctx)[PASS]))), flush=True)
        baseline = None
        for name, action, blast in arms:
            result, wins = evaluate(source, action, blast, args.samples, args.seed)
            if baseline is None:
                baseline = wins
            delta = wins.astype(float) - baseline.astype(float)
            result.update(arm=name, paired_win_change=float(delta.mean()),
                          paired_give_or_take=float(delta.std(ddof=1) / np.sqrt(args.samples)))
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
