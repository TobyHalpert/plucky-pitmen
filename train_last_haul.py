"""Last-haul tactical curriculum: varied scores, dynamite and late planning.

Synthetic starting scores/retained dynamite are explicit scenario conditions.
Late boards are produced by three complete safe execution rounds, not by merely
changing the round counter. Seed 7 is excluded. Train/test game seeds are disjoint.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from agents import NeuralAgent, ActionChoice
from features import action_mask, split_action, state_action_matrix
from game_env import MineEnv, DRAGON, DYNAMITE, GEM, PASS, PIT_CAGE, MAX_COLUMNS
from learning import PolicyNet
from selfplay import play_game, _choice_for
from train_tactical import assess, candidates, sample_hidden


def scenario(seed: int, late_round: int):
    rng = np.random.default_rng(seed)
    env = MineEnv(opponents=3)
    env.reset(seed=seed)
    env.haul = 3
    for player in env.players:
        count = int(rng.choice([0, 1, 2, 4, 6]))
        # Retained dynamite is additional to the fresh haul deck, as in engine.
        player.collected_cards = [(DYNAMITE, int(rng.integers(3))) for _ in range(count)]
    env.tactical_retained_counts = [p.dynamite for p in env.players]
    for _ in range(late_round):
        start_round = env.round
        while env.round == start_round and not env.game_finished:
            seat = env.planning_player
            if env.planned_players[seat]:
                action = PASS
            else:
                choices = []
                for action in env.player_action_context(seat).legal_actions:
                    if action >= PIT_CAGE:
                        continue
                    row, col = divmod(action, MAX_COLUMNS)
                    card = env.rows[row][col]
                    if (card is not None and card[0] != DRAGON
                            and env._card_occupant(row, col) is None):
                        choices.append((col != seat, card[0] != GEM, -row, action))
                if not choices:
                    return None
                action = min(choices)[-1]
            env.step_one_player(action, force_action=True)
        if env.game_finished:
            return None
    # Rotate a unique leader; cover small and large point deficits.
    leader = seed % 4
    top = int(rng.choice([3, 6, 9, 12]))
    for seat, player in enumerate(env.players):
        player.score = top if seat == leader else max(0, top - int(rng.choice([1, 3, 5, 8])))
    # Choose a trailer in three quarters of cases, a leader in one quarter.
    tracked = leader if rng.random() < 0.25 else (leader + int(rng.integers(1, 4))) % 4
    env.starting_player = tracked
    env.planning_player = tracked
    env.planned_players = [False] * 4
    env.consecutive_passes = 0
    env.planning_turns = 0
    return env, tracked


def sample_last_hidden(source, seat, rng):
    """Retained start-of-haul dynamite is known safe/dynamite, not unknown loot."""
    base = copy.deepcopy(source)
    prefixes = {}
    for other, count in enumerate(source.tactical_retained_counts):
        if other != seat:
            prefixes[other] = base.players[other].collected_cards[:count]
            base.players[other].collected_cards = base.players[other].collected_cards[count:]
    sampled = sample_hidden(base, seat, rng)
    for other, cards in prefixes.items():
        sampled.players[other].collected_cards = cards + sampled.players[other].collected_cards
    return sampled


def collect_last(net, count, seed):
    positions = []
    index = 0
    while len(positions) < count:
        game_seed = seed + index
        # Alternate fresh last-haul and round-3 boards.
        source = scenario(game_seed, 0 if index % 2 == 0 else 3)
        index += 1
        if source is None:
            continue
        env, seat = source
        ctx = env.player_action_context(seat)
        if action_mask(ctx).sum() <= 1:
            continue
        positions.append((copy.deepcopy(env), seat, game_seed))
        if len(positions) >= count:
            break
        # Also teach the keep/cancel decision after a legal blast placement.
        blasts = [int(a) for a in np.flatnonzero(action_mask(ctx)) if split_action(int(a))[1]]
        if blasts:
            action, blast = split_action(blasts[game_seed % len(blasts)])
            env.step_one_player(action, announce_blast=blast, force_action=True)
            round_before = env.round
            agents = [NeuralAgent(net, seed=game_seed * 31 + i, temperature=0.7) for i in range(4)]
            for _ in range(12):
                if env.game_finished or env.round != round_before:
                    break
                if env.planning_player == seat:
                    follow = env.player_action_context(seat)
                    if action_mask(follow).sum() > 1:
                        positions.append((copy.deepcopy(env), seat, game_seed))
                    break
                player = env.planning_player
                choice = _choice_for(agents[player], env.player_action_context(player))
                env.step_one_player(choice.action, announce_blast=choice.announce_blast, force_action=True)
    return positions


class BlastPlanTeacher(NeuralAgent):
    """Explore completing a blast instead of inheriting the net's cancellations.

    Label generation only. Evaluation and the saved net do NOT get this rule.
    Root cancellation actions remain separate candidates, so blasting is not
    rewarded directly: a completed plan must actually win its continuation.
    """
    def decide(self, ctx):
        if ctx.own_blast_pending and action_mask(ctx)[PASS]:
            return ActionChoice(PASS)
        return super().decide(ctx)


def label_last(net, positions, samples, seed):
    rows = []
    for index, (source, seat, game_seed) in enumerate(positions):
        ctx = source.player_action_context(seat)
        actions = candidates(net, ctx)
        values = np.zeros(len(actions))
        for sample in range(samples):
            rollout_seed = seed + index * 1009 + sample
            hidden = sample_last_hidden(source, seat, np.random.default_rng(rollout_seed))
            for j, action in enumerate(actions):
                env = copy.deepcopy(hidden)
                primitive, blast = split_action(action)
                env.step_one_player(primitive, announce_blast=blast, force_action=True)
                agents = [NeuralAgent(net, seed=rollout_seed * 31 + i, temperature=0.7)
                          for i in range(4)]
                agents[seat] = BlastPlanTeacher(net, seed=rollout_seed * 31 + seat, temperature=0.7)
                result = play_game(agents, seed=rollout_seed, env=env)
                if result.aborted:
                    raise RuntimeError('aborted label continuation')
                scores = np.asarray(result.scores)
                if scores[seat] == scores.max():
                    values[j] += 1.0 / int((scores == scores.max()).sum())
        values /= samples
        rows.append((ctx, actions, values))
        print(json.dumps(dict(position=index + 1, game_seed=game_seed, haul=source.haul,
                              round=source.round, own_pending=ctx.own_blast_pending,
                              actions=actions, win_shares=values.tolist(),
                              continuations=len(actions) * samples)), flush=True)
    return rows


def behaviour(net, positions):
    rows = []
    for env, seat, seed in positions:
        ctx = env.player_action_context(seat)
        probs = net.masked_probs(state_action_matrix(ctx), action_mask(ctx), temperature=0.7)
        blast_mass = sum(float(p) for a, p in enumerate(probs) if split_action(a)[1])
        rows.append(dict(seed=seed, round=env.round, own_pending=ctx.own_blast_pending,
                         trailing=ctx.own_score < max(p.score for p in env.players),
                         blast_probability=blast_mass, pass_probability=float(probs[PASS])))
    summaries = {}
    for label, predicate in (
        ('trailing_initial', lambda r: r['trailing'] and not r['own_pending']),
        ('leading_initial', lambda r: not r['trailing'] and not r['own_pending']),
        ('trailing_pending', lambda r: r['trailing'] and r['own_pending']),
        ('round3_initial', lambda r: r['round'] == 3 and not r['own_pending']),
    ):
        subset = [r for r in rows if predicate(r)]
        summaries[label] = dict(count=len(subset),
                               blast_probability=float(np.mean([r['blast_probability'] for r in subset])) if subset else None,
                               pass_probability=float(np.mean([r['pass_probability'] for r in subset])) if subset else None)
    return summaries


def evaluate_from_positions(frozen, candidate, positions, repeats=8):
    output = []
    for name, net in [('baseline', frozen), ('last_haul', candidate)]:
        wins, ties, scores = [], [], []
        announced = cancelled = fired = unfinished = 0
        begin, cancel = MineEnv.begin_blast, MineEnv._cancel_blast
        tracked = [0]
        def spy_begin(env, seat):
            nonlocal announced
            result = begin(env, seat)
            announced += int(seat == tracked[0])
            return result
        def spy_cancel(env, seat):
            nonlocal cancelled
            cancelled += int(seat == tracked[0] and env.players[seat].blast_pending)
            return cancel(env, seat)
        MineEnv.begin_blast, MineEnv._cancel_blast = spy_begin, spy_cancel
        try:
            for i, (source, seat, _) in enumerate(positions):
                tracked[0] = seat
                for repeat in range(repeats):
                    seed = 2800000 + i * 101 + repeat
                    # A held-out, newly resampled world shared between policies.
                    env = sample_last_hidden(source, seat, np.random.default_rng(seed))
                    agents = [NeuralAgent(frozen, seed=seed * 31 + j, temperature=0.7) for j in range(4)]
                    agents[seat] = NeuralAgent(net, seed=seed * 31 + seat, temperature=0.7)
                    result = play_game(agents, seed=seed, env=env)
                    own = result.scores[seat]
                    others = [s for j, s in enumerate(result.scores) if j != seat]
                    wins.append(float(own > max(others)))
                    ties.append(float(own == max(others)))
                    scores.append(own)
                    fired += sum(seat in h.blasts for h in result.hauls)
                    unfinished += int(result.aborted)
        finally:
            MineEnv.begin_blast, MineEnv._cancel_blast = begin, cancel
        if name == 'baseline':
            baseline = np.asarray(wins)
        delta = np.asarray(wins) - baseline
        # Cluster uncertainty by source game, not repeated continuations.
        clusters = {}
        for i, (_, _, source_seed) in enumerate(positions):
            clusters.setdefault(source_seed, []).extend(delta[i * repeats:(i + 1) * repeats])
        cluster_means = [np.mean(v) for v in clusters.values()]
        groups = {}
        for label, predicate in [
            ('trailing', lambda env, seat: env.players[seat].score < max(p.score for p in env.players)),
            ('round3', lambda env, seat: env.round == 3),
        ]:
            indices = [i * repeats + j for i, (env, seat, _) in enumerate(positions)
                       if predicate(env, seat) for j in range(repeats)]
            groups[label] = dict(continuations=len(indices),
                                 outright_win_rate=float(np.mean(np.asarray(wins)[indices])) if indices else None)
        output.append(dict(policy=name, groups=groups, continuations=len(wins), outright_win_rate=float(np.mean(wins)),
                           tied_top_rate=float(np.mean(ties)), score=float(np.mean(scores)),
                           announcements=announced, cancellations=cancelled, fired_blast_hauls=fired,
                           unfinished=unfinished, paired_win_change=float(delta.mean()),
                           source_cluster_give_or_take=float(np.std(cluster_means, ddof=1) / np.sqrt(len(cluster_means)))))
    return output


def diagnose_seed7(candidate_path):
    """Switch weights only at haul 3, preserving the real viewer RNG history."""
    from gui_controller import GameController
    source = GameController(3, 'neural:selfplay4', 7)
    source.reset(3, 'neural:selfplay4', 7, all_seats=True, advance=False)
    for _ in range(1200):
        if source.env.haul == 3:
            break
        source.step_one_player_auto()
    if source.env.haul != 3 or source.env.round != 0:
        raise RuntimeError('could not reach seed-7 haul-3 boundary')
    results = []
    for path in ['checkpoints/selfplay4.npz', candidate_path]:
        branch = copy.copy(source)
        branch.env = copy.deepcopy(source.env)
        branch.agents = copy.deepcopy(source.agents)
        net = PolicyNet.load(path)
        for agent in branch.agents:
            agent.net = net
        fired, trace = set(), []
        for _ in range(1200):
            if branch.env.game_finished:
                break
            pending = branch.pending_blast_seat()
            if pending is not None:
                fired.add((branch.env.round, pending + 1))
            seat = branch.env.planning_player
            round_index = branch.env.round
            turns = branch.env.planning_turns
            was_pending = branch.env.players[seat].blast_pending
            branch.step_one_player_auto()
            if (branch.env.planning_turns != turns and branch.last_choice is not None
                    and branch.last_choice_seat == seat and seat in (1, 2)):
                choice = branch.last_choice
                trace.append(dict(player=seat + 1, round=round_index, action=choice.action,
                                  announce=choice.announce_blast, was_pending=was_pending))
        results.append(dict(policy=path, starting_scores=[p.score for p in source.env.players],
                            finished=branch.env.game_finished,
                            final_scores=[p.score for p in branch.env.players],
                            fired=[list(x) for x in sorted(fired)], p2_p3_trace=trace))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', default='checkpoints/selfplay4.npz')
    parser.add_argument('--out', default='checkpoints/selfplay4_last_haul.npz')
    parser.add_argument('--positions', type=int, default=96)
    parser.add_argument('--held-out', type=int, default=48)
    parser.add_argument('--samples', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2300000)
    parser.add_argument('--diagnose-seed7', action='store_true',
                        help='compare existing candidate only; do not train')
    args = parser.parse_args()
    if args.diagnose_seed7:
        print(json.dumps(diagnose_seed7(args.out), indent=2))
        return
    if min(args.positions, args.held_out, args.samples, args.epochs) < 1:
        parser.error('budgets must be positive')
    frozen = PolicyNet.load(args.resume)
    train = collect_last(frozen, args.positions, args.seed)
    held = collect_last(frozen, args.held_out, args.seed + 100000)
    train_rows = label_last(frozen, train, args.samples, args.seed + 200000)
    held_rows = label_last(frozen, held, args.samples, args.seed + 300000)
    learner = PolicyNet.load(args.resume, lr=0.0003)
    targets = []
    for ctx, actions, values in train_rows:
        prior = frozen.masked_probs(state_action_matrix(ctx), action_mask(ctx))
        teacher = np.exp((values - values.max()) / 0.1)
        teacher /= teacher.sum()
        target = prior.copy() if np.ptp(values) < 1e-9 else 0.6 * prior
        if np.ptp(values) >= 1e-9:
            target[actions] += 0.4 * teacher
        targets.append((ctx, target))
    losses = learner.fit_tactical(targets, epochs=args.epochs, seed=args.seed)
    report = dict(config=vars(args), loss=losses,
                  train_before=assess(frozen, train_rows), train_after=assess(learner, train_rows),
                  held_before=assess(frozen, held_rows), held_after=assess(learner, held_rows),
                  held_behaviour_before=behaviour(frozen, held), held_behaviour_after=behaviour(learner, held),
                  evaluation=evaluate_from_positions(frozen, learner, held),
                  training_continuations=sum(len(a) * args.samples for _, a, _ in train_rows),
                  held_continuations=sum(len(a) * args.samples for _, a, _ in held_rows),
                  training_labels=[dict(position=i + 1, game_seed=train[i][2], actions=a,
                                        win_shares=v.tolist(), continuations=len(a) * args.samples)
                                   for i, (_, a, v) in enumerate(train_rows)],
                  held_labels=[dict(position=i + 1, game_seed=held[i][2], actions=a,
                                    win_shares=v.tolist(), continuations=len(a) * args.samples)
                               for i, (_, a, v) in enumerate(held_rows)],
                  source_positions=[dict(split=split, seed=seed, seat=seat, round=env.round,
                                         starting_dynamite=env.tactical_retained_counts,
                                         scores=[p.score for p in env.players],
                                         dynamite=[p.dynamite for p in env.players],
                                         own_pending=env.players[seat].blast_pending)
                                    for split, positions in [('train', train), ('held', held)]
                                    for env, seat, seed in positions])
    learner.name = 'selfplay4_last_haul'
    learner.metadata['last_haul_training'] = vars(args)
    learner.save(args.out)
    Path(args.out).with_suffix('.history.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
