"""Correct the row-0 blast-then-exit habit with outcome-labelled branches.

The habit has two linked decisions:

1. move to row 0 while announcing a blast;
2. on the next planning turn from row 0, leave via the pit cage or lorry, which
   cancels the announced blast.

Positions are harvested from real games.  Both the initial announcement state
and the later cancellation state are labelled.  A continuation teacher is used
only to explore completing a announced blast; it is not part of the saved or
evaluated policy.  A cancellation is therefore corrected only when its actual
continuation value is poor, not by a hard rule.

Nothing in this file sees seed 7.  The default checkpoint is never overwritten.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np

from agents import NeuralAgent
from features import (
    N_ACTIONS,
    action_mask,
    split_action,
    state_action_matrix,
)
from game_env import DRAGON, LORRY, MAX_COLUMNS, MAX_ROWS, MineEnv, PASS, PIT_CAGE
from learning import PolicyNet
from selfplay import play_game
from train_last_haul import BlastPlanTeacher
from train_tactical import assess, candidates, sample_hidden


def sample_row0_hidden(source: MineEnv, seat: int, rng) -> MineEnv:
    """Resample hidden fronts while preserving a pending blast's own cards.

    The generic tactical sampler deliberately conditions collected hands to be
    safe.  A pending blast is different: its drawn cards have not been revealed
    to opponents and may be dragons, so opponent blast-card fronts must remain
    part of the hidden deck.  Acting-seat private cards, including its own
    pending blast cards, stay exactly as observed.
    """
    from train_tactical import _public_cell_propensities, assign_rest_fronts

    env = copy.deepcopy(source)
    prop = _public_cell_propensities(env, seat)
    slots = []

    for row_idx, row in enumerate(env.rows):
        for column, card in enumerate(row):
            if card is not None and column != seat:
                slots.append((row, column, card, False, prop.get((row_idx, column), 1.0)))

    for player_index, player in enumerate(env.players):
        if player_index == seat:
            continue
        for index, card in enumerate(player.collected_cards):
            slots.append((player.collected_cards, index, card, True, 1.0))
        for index, card in enumerate(player.blast_cards):
            slots.append((player.blast_cards, index, card, False, 1.0))

    for index, card in enumerate(env.cards_remaining):
        slots.append((env.cards_remaining, index, card, False, 1.0))

    for backside in {card[1] for _container, _index, card, _safe, _w in slots}:
        group = [slot for slot in slots if slot[2][1] == backside]
        fronts = [slot[2][0] for slot in group]
        safe = [slot for slot in group if slot[3]]
        rest = [slot for slot in group if not slot[3]]
        for container, index, _, _, _ in safe:
            choices = [i for i, front in enumerate(fronts) if front != DRAGON]
            selected = int(rng.choice(choices))
            container[index] = (fronts.pop(selected), backside)
        assign_rest_fronts(rest, fronts, rng)

    rng.shuffle(env.cards_remaining)
    env.rng = np.random.default_rng(int(rng.integers(2**31)))
    return env


def has_useful_alternative(env: MineEnv, seat: int, kind: str) -> bool:
    """Require a real alternative so labels correct the sequence, not announcement broadly.

    An initial Row-0 announcement is only interesting when Row 1 or Row 2 was
    also available.  A cancellation is only interesting when staying to carry
    out the announced blast remains legal.
    """
    ctx = env.player_action_context(seat)
    mask = action_mask(ctx)
    if kind == "follow_up":
        return bool(mask[int(PASS)])
    return any(
        bool(mask[action])
        for action in range(MAX_COLUMNS, min(3 * MAX_COLUMNS, PIT_CAGE))
    )


def collect_row0(net: PolicyNet, count: int, seed: int, *, limit: int | None = None,
                 require_alternative: bool = True,
                 include_kinds: tuple[str, ...] = ("initial", "follow_up")):
    """Collect actual row-0 blasts and row-0 cancellations from ordinary games."""
    positions = []
    overflow = []
    kind_counts = {k: 0 for k in include_kinds}
    original = MineEnv.step_one_player
    max_games = limit or max(500, count * 40)
    game_index = 0

    while len(positions) < count and game_index < max_games:
        game_seed = seed + game_index
        taken = [0]

        def step(env, action=None, **kwargs):
            if (
                kwargs.get("force_action")
                and action is not None
                and taken[0] < 2
                and len(positions) < count
            ):
                seat = env.planning_player
                player = env.players[seat]
                primitive = int(action)
                kind = None
                if (
                    0 <= primitive < PIT_CAGE
                    and primitive // MAX_COLUMNS == 0
                    and player.position > 0
                    and player.dynamite >= 2
                    and kwargs.get("announce_blast")
                ):
                    kind = "initial"
                elif (
                    primitive in (int(PIT_CAGE), int(LORRY))
                    and player.blast_pending
                    and player.position == 0
                ):
                    kind = "follow_up"
                if (
                    kind is not None
                    and kind in include_kinds
                    and (
                        not require_alternative
                        or has_useful_alternative(env, seat, kind)
                    )
                ):
                    missing_others = sum(
                        1 for k, c in kind_counts.items() if k != kind and c == 0
                    )
                    item = (copy.deepcopy(env), seat, kind, game_seed, primitive)
                    if count >= len(include_kinds) and len(positions) + missing_others >= count:
                        if len(overflow) < count:
                            overflow.append(item)
                    else:
                        positions.append(item)
                        kind_counts[kind] += 1
                        taken[0] += 1
            return original(env, action, **kwargs)

        MineEnv.step_one_player = step
        try:
            agents = [
                NeuralAgent(net, seed=game_seed * 31 + seat, temperature=0.7)
                for seat in range(4)
            ]
            play_game(agents, seed=game_seed)
        finally:
            MineEnv.step_one_player = original
        game_index += 1

    if len(positions) < count and overflow:
        positions.extend(overflow[: count - len(positions)])
    if len(positions) < count:
        raise RuntimeError(
            f"only collected {len(positions)} row-0 positions in {game_index} games"
        )
    return positions


def row0_candidates(net: PolicyNet, ctx, kind: str):
    """Candidate first actions focused on row 0 and blast follow-ups."""
    mask = action_mask(ctx)
    actions = set(candidates(net, ctx))
    probs = net.masked_probs(state_action_matrix(ctx), mask, temperature=0.7)
    legal = [int(a) for a in np.flatnonzero(mask)]

    if kind == "initial":
        def primitive_row(action):
            primitive = split_action(action)[0]
            return primitive // MAX_COLUMNS if 0 <= primitive < PIT_CAGE else None

        row0 = [a for a in legal if primitive_row(a) == 0]
        nearby = [a for a in legal if primitive_row(a) in (1, 2)]
        actions.update(row0)
        actions.update(nearby)
        # Compare the same useful blast/no-blast options at both the bad Row 0
        # and the safer nearby rows.  Top-3 distillation alone can miss them.
        for group_name, group in (("row0", row0), ("nearby", nearby)):
            del group_name
            for wanted in (True, False):
                choices = [a for a in group if bool(split_action(a)[1]) == wanted]
                if choices:
                    actions.add(max(choices, key=lambda a: probs[a]))
        non_row0 = [a for a in legal if primitive_row(a) != 0]
        actions.update(sorted(non_row0, key=lambda a: -probs[a])[:3])
    else:
        actions.update(a for a in (int(PASS), int(PIT_CAGE), int(LORRY)) if mask[a])
        moves = [a for a in legal if 0 <= a < PIT_CAGE]
        actions.update(sorted(moves, key=lambda a: -probs[a])[:3])

    return sorted(actions)


def label_row0(net: PolicyNet, positions, samples: int, seed: int, *,
               teacher_mode: str = "blast"):
    rows = []
    metadata = []
    for index, (source, seat, kind, game_seed, observed_action) in enumerate(positions):
        ctx = source.player_action_context(seat)
        actions = row0_candidates(net, ctx, kind)
        observed = None
        # Preserve the precise bad action in the candidate set even if top-3
        # distillation would omit it.  ``initial`` observes the primitive card;
        # follow-up observes a special leave action.
        if kind == "initial" and actions:
            candidates_for_row0 = [a for a in actions if split_action(a)[1]
                                   and split_action(a)[0] == observed_action]
            observed = max(candidates_for_row0, default=None)
        elif kind == "follow_up" and observed_action in actions:
            observed = observed_action
        if observed is not None:
            actions = sorted(set(actions) | {int(observed)})

        values = np.zeros(len(actions))
        for sample in range(samples):
            rollout_seed = seed + index * 1009 + sample
            hidden = sample_row0_hidden(source, seat, np.random.default_rng(rollout_seed))
            for action_index, action in enumerate(actions):
                primitive, blast = split_action(action)
                env = copy.deepcopy(hidden)
                env.step_one_player(primitive, announce_blast=blast, force_action=True)
                agents = [
                    NeuralAgent(net, seed=rollout_seed * 31 + other, temperature=0.7)
                    for other in range(4)
                ]
                if teacher_mode == "blast":
                    agents[seat] = BlastPlanTeacher(
                        net, seed=rollout_seed * 31 + seat, temperature=0.7
                    )
                result = play_game(agents, seed=rollout_seed, env=env)
                if result.aborted:
                    raise RuntimeError("aborted row-0 label continuation")
                scores = np.asarray(result.scores)
                if scores[seat] == scores.max():
                    values[action_index] += 1.0 / int((scores == scores.max()).sum())
        values /= samples

        rows.append((ctx, actions, values))
        metadata.append({"kind": kind, "seat": seat, "source": source})
        print(json.dumps({
            "position": index + 1,
            "game_seed": game_seed,
            "kind": kind,
            "haul": source.haul,
            "round": source.round,
            "actions": actions,
            "win_shares": values.tolist(),
            "continuations": len(actions) * samples,
        }), flush=True)
    return rows, metadata


def bad_action_indices(kind: str, actions):
    """Return candidate indices that form the actual complaint."""
    if kind == "initial":
        return [
            index for index, action in enumerate(actions)
            if split_action(action)[1]
            and 0 <= split_action(action)[0] < PIT_CAGE
            and split_action(action)[0] // MAX_COLUMNS == 0
        ]
    return [
        index for index, action in enumerate(actions)
        if action in (int(PIT_CAGE), int(LORRY))
    ]


def nearby_blast_indices(actions):
    """Indices for a row-2 blast announcement, or row 1 if row 2 is unavailable."""
    for preferred_row in (2, 1):
        matches = [
            index for index, action in enumerate(actions)
            if split_action(action)[1]
            and 0 <= split_action(action)[0] < PIT_CAGE
            and split_action(action)[0] // MAX_COLUMNS == preferred_row
        ]
        if matches:
            return matches
    return []


def build_targets(frozen: PolicyNet, rows, metadata, *, mode: str,
                  prior_weight: float, temperature: float, bad_reduction: float,
                  evidence_scale: float, transfer_margin: float = 0.08):
    targets = []
    for (ctx, actions, values), meta in zip(rows, metadata):
        prior = frozen.masked_probs(state_action_matrix(ctx), action_mask(ctx))
        if mode == "distill":
            if np.ptp(values) < 1e-9:
                target = prior.copy()
            else:
                teacher = np.exp((values - values.max()) / temperature)
                teacher /= teacher.sum()
                target = prior_weight * prior
                target[actions] += (1 - prior_weight) * teacher
                total = target.sum()
                if total > 0:
                    target /= total
            targets.append((ctx, target))
            continue

        if mode == "sequence":
            target = prior.copy()
            actions = list(actions)
            values = np.asarray(values, dtype=float)
            bad = bad_action_indices(meta["kind"], actions)
            alt = (nearby_blast_indices(actions) if meta["kind"] == "initial"
                   else [index for index, action in enumerate(actions) if action == int(PASS)])
            if bad and alt and len(values) == len(actions):
                if meta["kind"] == "initial":
                    reference_bad = float(values[bad].mean())
                else:
                    reference_bad = float(values[bad].max())
                best_alt = float(values[alt].max())
                if best_alt >= reference_bad - transfer_margin:
                    denominator = max(evidence_scale, transfer_margin)
                    strength = min(1.0, max(0.0, (best_alt - reference_bad + transfer_margin) / denominator))
                    bad_mass = float(sum(prior[actions[index]] for index in bad))
                    reduction = bad_mass * bad_reduction * strength
                    if reduction > 0:
                        keep = 1.0 - bad_reduction * strength
                        for index in bad:
                            target[actions[index]] *= keep
                        alt_values = values[alt]
                        if np.ptp(alt_values) < 1e-9:
                            weights = np.array([prior[actions[index]] for index in alt])
                        else:
                            weights = np.exp((alt_values - alt_values.max()) / temperature)
                        total = float(weights.sum())
                        if total > 0:
                            weights /= total
                            for index, weight in zip(alt, weights):
                                target[actions[index]] += reduction * weight
                    total = float(target.sum())
                    if total > 0:
                        target /= total
            targets.append((ctx, target))
            continue

        target = prior.copy()
        actions = list(actions)
        values = np.asarray(values, dtype=float)
        bad = bad_action_indices(meta["kind"], actions)
        good = [index for index in range(len(actions)) if index not in bad]
        if bad and good and len(values) == len(actions):
            average_bad = float(values[bad].mean())
            best_good = float(values[good].max())
            strength = min(1.0, max(0.0, (best_good - average_bad) / evidence_scale))
            if strength > 0:
                bad_mass = float(sum(prior[actions[index]] for index in bad))
                reduction = bad_mass * bad_reduction * strength
                if reduction > 0:
                    keep = 1.0 - bad_reduction * strength
                    for index in bad:
                        target[actions[index]] *= keep
                    good_values = values[good]
                    if np.ptp(good_values) < 1e-9:
                        weights = np.array([prior[actions[index]] for index in good])
                    else:
                        weights = np.exp((good_values - good_values.max()) / temperature)
                    total = float(weights.sum())
                    if total > 0:
                        weights /= total
                        for index, weight in zip(good, weights):
                            target[actions[index]] += reduction * weight
                    total = float(target.sum())
                    if total > 0:
                        target /= total
        targets.append((ctx, target))
    return targets


def targeted_assess(net: PolicyNet, rows, metadata):
    """Measure held-out outcome share and the probability of the labelled bad choices."""
    expected = []
    bad_mass = []
    coverage = []
    for (ctx, actions, values), meta in zip(rows, metadata):
        probs = net.masked_probs(state_action_matrix(ctx), action_mask(ctx), temperature=0.7)
        actions = list(actions)
        mass = float(probs[actions].sum())
        if mass > 0:
            expected.append(float(probs[actions] @ np.asarray(values) / mass))
            bad = bad_action_indices(meta["kind"], actions)
            bad_mass.append(float(probs[[actions[index] for index in bad]].sum()))
            coverage.append(mass)
    return {
        "conditional_estimated_win_share": float(np.mean(expected)) if expected else 0.0,
        "mean_bad_probability": float(np.mean(bad_mass)) if bad_mass else 0.0,
        "mean_candidate_probability_mass": float(np.mean(coverage)) if coverage else 0.0,
    }


def audit_row0(net: PolicyNet, *, games: int, seed: int):
    """Count actual row-announcement and cancellation choices in ordinary play."""
    class LoggingAgent(NeuralAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.log = []

        def decide(self, ctx):
            choice = super().decide(ctx)
            self.log.append((
                int(ctx.haul), int(ctx.round), int(ctx.own_position),
                int(ctx.own_score), int(choice.action), bool(choice.announce_blast),
                bool(ctx.own_blast_pending),
            ))
            return choice

    counts = Counter()
    for game_index in range(games):
        game_seed = seed + game_index
        agents = [
            LoggingAgent(net, seed=game_seed * 31 + seat, temperature=0.7)
            for seat in range(4)
        ]
        result = play_game(agents, seed=game_seed)
        for agent in agents:
            for index, (haul, _round, before_position, _score, action, blast, _pending) in enumerate(agent.log):
                is_row0_move = 0 <= action < PIT_CAGE and action // MAX_COLUMNS == 0
                if is_row0_move and blast and before_position > 0:
                    counts["row0_blast_moves"] += 1
                    next_entry = (
                        agent.log[index + 1]
                        if index + 1 < len(agent.log) and agent.log[index + 1][0] == haul
                        else None
                    )
                    if next_entry is not None and next_entry[4] in (int(PIT_CAGE), int(LORRY)):
                        counts["exit_after_row0_blast"] += 1
                    later_exits = [
                        entry for entry in agent.log[index + 1:]
                        if entry[0] == haul and entry[4] in (int(PIT_CAGE), int(LORRY))
                    ]
                    if later_exits:
                        counts["eventual_exit_after_row0_blast"] += 1
                    counts["exit_while_blast_pending"] += sum(
                        bool(entry[6]) for entry in later_exits
                    )

                if 0 <= action < PIT_CAGE and blast:
                    destination_row = action // MAX_COLUMNS
                    source_row = before_position // MAX_COLUMNS if before_position else None
                    moved = source_row is not None and source_row != destination_row
                    counts[f"row{destination_row}_announce"] += 1
                    if moved:
                        counts[f"row{destination_row}_move_announce"] += 1
                    next_entry = (
                        agent.log[index + 1]
                        if index + 1 < len(agent.log) and agent.log[index + 1][0] == haul
                        else None
                    )
                    if next_entry is not None and next_entry[4] in (int(PIT_CAGE), int(LORRY)):
                        counts[f"row{destination_row}_immediate_cancel"] += 1
                        if moved:
                            counts[f"row{destination_row}_move_announce_immediate_cancel"] += 1
        top = max(result.scores)
        counts["seat_outright_wins"] += sum(
            1 for score in result.scores if score == top and result.scores.count(score) == 1
        )
        counts["games"] += 1
        counts["finished"] += int(not result.aborted)
    counts["row0_blast_moves_per_game"] = counts["row0_blast_moves"] / games
    counts["exit_after_row0_blast_per_game"] = counts["exit_after_row0_blast"] / games
    for row in range(MAX_ROWS):
        announced = counts[f"row{row}_move_announce"]
        cancelled = counts[f"row{row}_move_announce_immediate_cancel"]
        counts[f"row{row}_move_announces_per_game"] = announced / games
        counts[f"row{row}_immediate_cancel_rate"] = cancelled / announced if announced else 0.0
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", default="checkpoints/selfplay4_last_haul.npz")
    parser.add_argument("--out", default="checkpoints/selfplay4_row0_blast.npz")
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--held-out", type=int, default=32)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--prior-weight", type=float, default=0.35)
    parser.add_argument("--teacher-temperature", type=float, default=0.1)
    parser.add_argument("--kinds", nargs="+", choices=("initial", "follow_up"),
                        default=["initial", "follow_up"],
                        help="which half of the complaint to train; use follow_up to preserve placement")
    parser.add_argument("--teacher-mode", choices=("blast", "policy"), default="blast",
                        help="label continuations with a blast-completion teacher or the policy itself")
    parser.add_argument("--target-mode", choices=("sequence", "targeted", "distill"),
                        default="sequence",
                        help="shift Row-0 announcements to nearby blasts and cancellations to pass, "
                             "shift all labelled bad choices to any candidate, or distil all candidates")
    parser.add_argument("--bad-reduction", type=float, default=0.65)
    parser.add_argument("--evidence-scale", type=float, default=0.08)
    parser.add_argument("--transfer-margin", type=float, default=0.08,
                        help="how much worse the nearby replacement may be before no probability is moved")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-improvement", type=float, default=0.002)
    parser.add_argument("--save-every", type=int, default=0,
                        help="also save out.epochN.npz every N epochs for fresh-game selection")
    parser.add_argument("--held-tolerance", type=float, default=0.015,
                        help="absolute held-out win-share drop accepted in exchange for a better Row-0 habit")
    parser.add_argument("--no-require-alternative", dest="require_alternative",
                        action="store_false", default=True,
                        help="train every Row-0 announcement, including ones with no useful nearby move")
    parser.add_argument("--seed", type=int, default=3600000)
    parser.add_argument("--audit-games", type=int, default=200)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--dry-collect", action="store_true",
                        help="collect positions and report classes; do not label or train")
    args = parser.parse_args()
    if args.audit_only:
        report = {}
        for name, path in (
            ("baseline", "checkpoints/selfplay4.npz"),
            ("last_haul", args.resume),
            ("row0", args.out),
        ):
            if Path(path).exists():
                report[name] = audit_row0(
                    PolicyNet.load(path),
                    games=args.audit_games,
                    seed=args.seed + 500000,
                )
        print(json.dumps(report, indent=2))
        return

    if min(args.positions, args.held_out, args.samples, args.epochs, args.patience) < 1:
        parser.error("training budgets must be positive")
    if not 0 <= args.prior_weight <= 1 or args.teacher_temperature <= 0:
        parser.error("distillation settings must be in range")
    if not 0 <= args.bad_reduction <= 1 or args.evidence_scale <= 0 or args.transfer_margin < 0:
        parser.error("targeted-distillation settings must be in range")
    if args.min_improvement < 0 or args.held_tolerance < 0 or args.save_every < 0:
        parser.error("selection tolerances and --save-every must not be negative")

    frozen = PolicyNet.load(args.resume)
    include_kinds = tuple(dict.fromkeys(args.kinds))
    train_positions = collect_row0(
        frozen, args.positions, args.seed, require_alternative=args.require_alternative,
        include_kinds=include_kinds,
    )
    held_positions = collect_row0(
        frozen, args.held_out, args.seed + 100000,
        require_alternative=args.require_alternative,
        include_kinds=include_kinds,
    )
    if args.dry_collect:
        summary = {
            "train_kinds": dict(Counter(row[2] for row in train_positions)),
            "held_kinds": dict(Counter(row[2] for row in held_positions)),
        }
        print(json.dumps(summary, indent=2))
        return

    train_rows, train_meta = label_row0(
        frozen, train_positions, args.samples, args.seed + 200000,
        teacher_mode=args.teacher_mode,
    )
    held_rows, held_meta = label_row0(
        frozen, held_positions, args.samples, args.seed + 300000,
        teacher_mode=args.teacher_mode,
    )

    learner = PolicyNet.load(args.resume, lr=args.lr)
    targets = build_targets(
        frozen, train_rows, train_meta, mode=args.target_mode,
        prior_weight=args.prior_weight, temperature=args.teacher_temperature,
        bad_reduction=args.bad_reduction, evidence_scale=args.evidence_scale,
        transfer_margin=args.transfer_margin,
    )

    losses = []
    held_history = []
    initial_held = assess(frozen, held_rows)
    initial_targeted = targeted_assess(frozen, held_rows, held_meta)
    allowed_held_drop = args.held_tolerance
    best = {
        "epoch": 0,
        "expected": float(initial_held["conditional_estimated_win_share"]),
        "bad_probability": float(initial_targeted["mean_bad_probability"]),
        "net": copy.deepcopy(frozen),
        "held": initial_held,
        "targeted": initial_targeted,
    }
    for epoch in range(1, args.epochs + 1):
        losses.extend(learner.fit_tactical(targets, epochs=1, seed=args.seed + epoch))
        held = assess(learner, held_rows)
        targeted = targeted_assess(learner, held_rows, held_meta)
        expected = float(held["conditional_estimated_win_share"])
        bad_probability = float(targeted["mean_bad_probability"])
        eligible = expected >= best["expected"] - allowed_held_drop
        improves_bad = bad_probability <= best["bad_probability"] - args.min_improvement
        held_history.append({
            "epoch": epoch,
            "eligible": bool(eligible),
            "improves_bad_probability": bool(improves_bad),
            "held": held,
            "targeted": targeted,
        })
        if args.save_every and epoch % args.save_every == 0:
            path = Path(args.out).with_name(f"{Path(args.out).stem}.epoch{epoch}.npz")
            snapshot = copy.deepcopy(learner)
            snapshot.name = f"{Path(args.out).stem}.epoch{epoch}"
            snapshot.save(path)
            held_history[-1]["checkpoint"] = str(path)
        if eligible and improves_bad and bad_probability < best["bad_probability"] - args.min_improvement:
            best = {
                "epoch": epoch,
                "expected": expected,
                "bad_probability": bad_probability,
                "net": copy.deepcopy(learner),
                "held": held,
                "targeted": targeted,
            }
        elif epoch >= args.patience and best["epoch"] == 0 and bad_probability >= best["bad_probability"]:
            # No useful held-out candidate after the patience window; stop early,
            # but still check the already accepted best in later epochs.
            if all(
                not (entry["eligible"] and entry["improves_bad_probability"])
                for entry in held_history
            ):
                break
    learner = best["net"]

    before_audit = audit_row0(
        frozen, games=args.audit_games, seed=args.seed + 500000
    )
    after_audit = audit_row0(
        learner, games=args.audit_games, seed=args.seed + 500000
    )
    report = {
        "config": vars(args),
        "loss": losses,
        "best_epoch": best["epoch"],
        "train_before": assess(frozen, train_rows),
        "train_after": assess(learner, train_rows),
        "held_before": initial_held,
        "held_after": best["held"],
        "held_targeted_before": initial_targeted,
        "held_targeted_after": best["targeted"],
        "held_history": held_history,
        "fresh_row0_audit": {"before": before_audit, "after": after_audit},
        "training_continuations": sum(len(a) * args.samples for _, a, _ in train_rows),
        "held_continuations": sum(len(a) * args.samples for _, a, _ in held_rows),
        "held_kinds": dict(Counter(row[2] for row in held_positions)),
    }
    learner.name = "selfplay4_row0_blast"
    learner.metadata["row0_blast_training"] = vars(args)
    learner.save(args.out)
    Path(args.out).with_suffix(".history.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
