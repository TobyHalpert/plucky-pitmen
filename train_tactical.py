"""Branch-based tactical policy distillation, with disjoint held-out game seeds.

No seed-7 positions are used. Hidden board/hand fronts and the deck are sampled
subject to the acting seat's current observations and safe collected cards.
This is an observation-consistent determinization, NOT a full-history Bayesian
posterior: opponents' earlier actions are not used as likelihood evidence.
A frozen policy plays continuations after every candidate first action.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np

from agents import NeuralAgent, make_scripted_agent
from features import (action_mask, state_action_matrix, split_action,
                      N_ACTIONS)
from game_env import MineEnv, DRAGON, DYNAMITE, PIT_CAGE, PASS, MAX_COLUMNS
from learning import PolicyNet
from selfplay import play_game


def sample_hidden(source, seat, rng):
    """Preserve visible cards/backs, safe hands and card inventory; vary secrets."""
    env = copy.deepcopy(source)
    slots = []
    for row, cards in enumerate(env.rows):
        for col, card in enumerate(cards):
            if card is not None and col != seat:
                slots.append((cards, col, card, False))
    for index, player in enumerate(env.players):
        if index != seat:
            for pos, card in enumerate(player.collected_cards):
                slots.append((player.collected_cards, pos, card, True))
    for pos, card in enumerate(env.cards_remaining):
        slots.append((env.cards_remaining, pos, card, False))
    for back in {card[1] for _, _, card, _ in slots}:
        group = [slot for slot in slots if slot[2][1] == back]
        fronts = [slot[2][0] for slot in group]
        # Conditioning all collected-card slots to be non-dragons.
        safe = [slot for slot in group if slot[3]]
        rest = [slot for slot in group if not slot[3]]
        for container, index, _, _ in safe:
            choices = [i for i, front in enumerate(fronts) if front != DRAGON]
            selected = int(rng.choice(choices))
            container[index] = (fronts.pop(selected), back)
        rng.shuffle(fronts)
        for (container, index, _, _), front in zip(rest, fronts):
            container[index] = (front, back)
    # Even the deck's backside order is hidden.
    rng.shuffle(env.cards_remaining)
    env.rng = np.random.default_rng(int(rng.integers(2**31)))
    return env


def collect(net, count, seed):
    """At most two tactical planning positions from each generated game."""
    positions = []
    original = MineEnv.step_one_player
    game_index = 0
    while len(positions) < count:
        taken = [0]
        game_seed = seed + game_index
        def step(env, action=None, **kw):
            ctx = env.player_action_context(env.planning_player)
            tactical = (sum(f == DYNAMITE for f, _ in ctx.private_cards) >= 2
                        or any(o.blast_pending for o in ctx.opponents))
            follow_up = (taken[0] == 0 or ctx.own_blast_pending)
            if (tactical and follow_up and taken[0] < 2 and len(positions) < count
                    and env.phase == 0 and not any(p.blast_cards for p in env.players)
                    and action_mask(ctx).sum() > 1):
                positions.append((copy.deepcopy(env), env.planning_player, game_seed))
                taken[0] += 1
            return original(env, action, **kw)
        MineEnv.step_one_player = step
        try:
            # Mix neural responders with actual blasters and deterministic pushers.
            agents = [NeuralAgent(net, seed=game_seed * 31 + i, temperature=0.9)
                      for i in range(4)]
            if game_index % 3 == 1:
                agents[game_index % 4] = make_scripted_agent('rule', blast='always')
            if game_index % 3 == 2:
                agents[game_index % 4] = make_scripted_agent('pusher-leader')
            play_game(agents, seed=game_seed)
        finally:
            MineEnv.step_one_player = original
        game_index += 1
        if game_index > count * 100:
            raise RuntimeError('could not collect enough tactical positions')
    return positions


def candidates(net, ctx):
    mask = action_mask(ctx)
    probs = net.masked_probs(state_action_matrix(ctx), mask, temperature=0.7)
    actions = set(int(a) for a in np.argsort(-probs)[:3] if mask[a])
    for primitive in (PASS, PIT_CAGE):
        if mask[primitive]:
            actions.add(int(primitive))
    blasts = [int(a) for a in np.flatnonzero(mask) if split_action(int(a))[1]]
    if blasts:
        for a in (min(blasts), max(blasts)):
            actions.add(a)
            base, _ = split_action(a)
            if mask[base]:
                actions.add(base)
    # Include one legal attack on the highest-scoring reachable rival.
    targets = []
    for opponent in ctx.opponents:
        if opponent.column == opponent.player_index and 0 <= opponent.position < PIT_CAGE:
            a = opponent.position * MAX_COLUMNS + opponent.column
            if mask[a]:
                targets.append((-opponent.score, a))
    if targets:
        actions.add(min(targets)[1])
    return sorted(actions)


def label_positions(net, positions, samples, seed):
    rows = []
    for index, (source, seat, game_seed) in enumerate(positions):
        ctx = source.player_action_context(seat)
        actions = candidates(net, ctx)
        values = np.zeros(len(actions))
        aborted = 0
        for sample in range(samples):
            rollout_seed = seed + index * 1009 + sample
            hidden = sample_hidden(source, seat, np.random.default_rng(rollout_seed))
            for j, action in enumerate(actions):
                env = copy.deepcopy(hidden)
                primitive, blast = split_action(action)
                env.step_one_player(primitive, announce_blast=blast, force_action=True)
                agents = [NeuralAgent(net, seed=rollout_seed * 31 + i, temperature=0.7)
                          for i in range(4)]
                result = play_game(agents, seed=rollout_seed, env=env)
                if result.aborted:
                    aborted += 1
                    raise RuntimeError('aborted tactical continuation; refusing biased labels')
                # Final-score win share: ties split, no score or escape reward.
                scores = np.asarray(result.scores)
                if scores[seat] == scores.max():
                    values[j] += 1.0 / int((scores == scores.max()).sum())
        values /= samples
        rows.append((ctx, actions, values))
        print(json.dumps(dict(position=index + 1, game_seed=game_seed,
                              actions=actions, win_shares=values.tolist(),
                              continuations=len(actions) * samples)), flush=True)
    return rows


def assess(net, rows):
    expected, best, coverage = [], [], []
    for ctx, actions, values in rows:
        probs = net.masked_probs(state_action_matrix(ctx), action_mask(ctx), temperature=0.7)
        mass = probs[actions].sum()
        expected.append(float(probs[actions] @ values / mass))
        best.append(float(values.max()))
        coverage.append(float(mass))
    return dict(conditional_estimated_win_share=float(np.mean(expected)),
                mean_candidate_probability_mass=float(np.mean(coverage)),
                teacher_best=float(np.mean(best)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', default='checkpoints/selfplay4.npz')
    parser.add_argument('--out', default='checkpoints/selfplay4_tactical.npz')
    parser.add_argument('--positions', type=int, default=64)
    parser.add_argument('--held-out', type=int, default=32)
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=1300000)
    args = parser.parse_args()
    if min(args.positions, args.held_out, args.samples, args.epochs) < 1:
        parser.error('budgets must be positive')
    frozen = PolicyNet.load(args.resume)
    train = collect(frozen, args.positions, args.seed)
    held = collect(frozen, args.held_out, args.seed + 100000)
    train_rows = label_positions(frozen, train, args.samples, args.seed + 200000)
    held_rows = label_positions(frozen, held, args.samples, args.seed + 300000)
    learner = PolicyNet.load(args.resume, lr=0.0005)
    targets = []
    for ctx, actions, values in train_rows:
        prior = frozen.masked_probs(state_action_matrix(ctx), action_mask(ctx))
        teacher = np.exp((values - values.max()) / 0.15)
        teacher /= teacher.sum()
        target = 0.5 * prior
        target[actions] += 0.5 * teacher
        targets.append((ctx, target))
    losses = learner.fit_tactical(targets, epochs=args.epochs, seed=args.seed)
    def source_summary(positions):
        return dict(game_seeds=sorted({s for _, _, s in positions}),
                    own_blast_pending_positions=sum(env.players[seat].blast_pending
                                                    for env, seat, _ in positions),
                    haul_counts=dict(Counter(env.haul for env, _, _ in positions)))
    def label_summary(positions, rows):
        return [dict(position=i + 1, game_seed=positions[i][2], actions=actions,
                     win_shares=values.tolist(), continuations=len(actions) * args.samples)
                for i, (_, actions, values) in enumerate(rows)]
    report = dict(config=vars(args), training_loss=losses,
                  train_source_summary=source_summary(train), held_source_summary=source_summary(held),
                  train_labels=label_summary(train, train_rows), held_labels=label_summary(held, held_rows),
                  train_before=assess(frozen, train_rows), train_after=assess(learner, train_rows),
                  held_before=assess(frozen, held_rows), held_after=assess(learner, held_rows),
                  training_continuations=sum(len(a) * args.samples for _, a, _ in train_rows),
                  held_continuations=sum(len(a) * args.samples for _, a, _ in held_rows),
                  limitations=['observation-consistent, not history-conditioned hidden sampling',
                               'frozen all-neural continuation teacher',
                               'held evaluation conditional on measured candidate set',
                               'finite-sample noisy labels; not proof of global strength'])
    learner.name = 'selfplay4_tactical'
    learner.metadata['tactical_training'] = report['config']
    learner.save(args.out)
    Path(args.out).with_suffix('.history.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
