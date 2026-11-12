"""Distil the blast *trigger* from committed-blast continuations.

The trigger is the one decision the network owns (the keeper is inherited
hand-code, BALANCE §5).  Self-play does not teach it, for a mechanical reason:
86% of the blasts it announces are cancelled by its own next move, so the return
that arrives for a fired blast is attached to a decision that is almost never
followed through.  Labelling the decision directly fixes the attribution:

1. **Harvest** real positions where a blast is legal -- two dynamite, no blast
   pending, and a card to move to.
2. **Fork two branches from the same hidden world**: the seat's preferred card
   move with the blast announced, and the identical move without it.  The
   continuation for *both* branches uses ``BlastPlanTeacher``, which passes while
   a blast is pending: that is what prices a **committed** blast.  Without it the
   branch would be a bluff, and the net would learn the announcement it already
   over-uses.
3. **Label** each branch with final-score win share over ``--samples`` resampled
   hidden worlds, and distill the pairwise preference into the policy with
   ``PolicyNet.fit_tactical``, mixed with the frozen policy so nothing else
   forgets.

Held-out positions come from a disjoint seed range and never enter the fit.  The
saved checkpoint is a candidate; the shipped default is untouched.  Blasting is
not rewarded directly -- a committed blast has to win its continuation.

Usage
-----
    python train_blast_trigger.py --positions 64 --held-out 32 --samples 16 \
        --epochs 20 --blind --out checkpoints/selfplay4_blast_trigger.npz
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from agents import ActionChoice, NeuralAgent
from features import (
    N_ACTIONS,
    action_mask,
    digs_face_up_dragon,
    join_action,
    split_action,
    state_action_matrix,
)
from game_env import PIT_CAGE, MineEnv
from learning import PolicyNet
from selfplay import play_game
from train_last_haul import BlastPlanTeacher
from train_tactical import assess, sample_hidden


# ----------------------------------------------------------------------
# Harvest
# ----------------------------------------------------------------------
class HarvestEnv(MineEnv):
    """MineEnv that records positions where the seat could announce a blast.

    ``player_action_context`` is the one call every decision goes through, so it
    is the cheapest hook that yields the live environment next to the context.
    Keys on the planning clock so a decision is recorded once.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.found: list[tuple[MineEnv, int]] = []
        self._seen: set[tuple[int, int, int, int]] = set()

    def _snapshot(self) -> MineEnv:
        """Copy the live state without copying the collected copies.

        A naive ``deepcopy(self)`` also copies ``self.found``, i.e. every
        environment collected so far and each of their own lists, which grows
        the copy multiplicatively and stalls the harvest.
        """
        found = self.found
        self.found = []
        try:
            snapshot = copy.deepcopy(self)
        finally:
            self.found = found
        snapshot.found = []
        return snapshot

    def player_action_context(self, player_index: int):
        ctx = super().player_action_context(player_index)
        key = (self.haul, self.round, self.planning_turns, player_index)
        if key in self._seen:
            return ctx
        mask = action_mask(ctx)
        dynamite = sum(1 for front, _ in ctx.private_cards if front >= 0)
        if (
            ctx.own_blast_pending
            or not any(split_action(a)[1] for a in np.flatnonzero(mask))
        ):
            return ctx
        self._seen.add(key)
        self.found.append((self._snapshot(), player_index))
        return ctx


def collect(net: PolicyNet, count: int, seed: int, *, blind: bool, limit: int = 600):
    positions: list[tuple[MineEnv, int, int]] = []
    game = 0
    while len(positions) < count and game < limit:
        game_seed = seed + game
        env = HarvestEnv(opponents=3, blast_choice_sees_fronts=not blind)
        env.reset(seed=game_seed)
        agents = [
            NeuralAgent(net, seed=game_seed * 1000 + seat, temperature=0.7)
            for seat in range(4)
        ]
        play_game(agents, seed=game_seed, env=env, blast_choice_sees_fronts=not blind)
        for source, seat in env.found:
            if len(positions) >= count:
                break
            positions.append((source, seat, game_seed))
        game += 1
    return positions


# ----------------------------------------------------------------------
# Candidates and labels
# ----------------------------------------------------------------------
def trigger_candidates(net: PolicyNet, ctx) -> list[int]:
    """The seat's preferred card move, with and without the announcement."""
    mask = action_mask(ctx)
    probs = net.masked_probs(state_action_matrix(ctx), mask, temperature=0.7)
    card_actions = [
        int(a) for a in np.flatnonzero(mask) if split_action(int(a))[0] < PIT_CAGE
    ]
    if not card_actions:
        return []
    plain = [a for a in card_actions if not split_action(a)[1]]
    announce = [a for a in card_actions if split_action(a)[1]]
    if not plain or not announce:
        return []
    safe_plain = [
        a
        for a in plain
        if not digs_face_up_dragon(ctx, a) and join_action(a, True) in announce
    ]
    pool = safe_plain if safe_plain else [a for a in plain if join_action(a, True) in announce]
    if not pool:
        return []
    best_plain = max(pool, key=lambda a: probs[a])
    twin = join_action(best_plain, True)
    return [int(best_plain), int(twin)]


class HoldRoundTeacher(NeuralAgent):
    """Keep dynamite unspent for the remainder of the root planning phase."""

    def __init__(self, net: PolicyNet, *, hold_haul: int, hold_round: int, **kwargs: Any) -> None:
        super().__init__(net, **kwargs)
        self.hold_haul = hold_haul
        self.hold_round = hold_round

    def decide(self, ctx):
        if ctx.haul == self.hold_haul and ctx.round == self.hold_round:
            mask = action_mask(ctx)
            plain_mask = np.array(
                [bool(mask[a]) and not split_action(a)[1] for a in range(len(mask))],
                dtype=bool,
            )
            if plain_mask.any():
                probs = self.net.masked_probs(
                    state_action_matrix(ctx), plain_mask, temperature=self.temperature
                )
                chosen = int(self.rng.choice(len(probs), p=probs))
                prim, bl = split_action(chosen)
                return ActionChoice(prim, announce_blast=bl)
        return super().decide(ctx)


def label(net: PolicyNet, positions, samples: int, seed: int, *, blind: bool):
    rows, metadata = [], []
    for index, (source, seat, game_seed) in enumerate(positions):
        ctx = source.player_action_context(seat)
        actions = trigger_candidates(net, ctx)
        if len(actions) < 2:
            continue
        values = np.zeros(len(actions))
        for sample in range(samples):
            rollout_seed = seed + index * 1009 + sample
            hidden = sample_hidden(source, seat, np.random.default_rng(rollout_seed))
            for position, action in enumerate(actions):
                primitive, blast = split_action(action)
                env = copy.deepcopy(hidden)
                env.step_one_player(primitive, announce_blast=blast, force_action=True)
                agents = [
                    NeuralAgent(net, seed=rollout_seed * 31 + other, temperature=0.7)
                    for other in range(4)
                ]
                if blast:
                    # Pass while the blast is pending: price a committed blast.
                    agents[seat] = BlastPlanTeacher(
                        net, seed=rollout_seed * 31 + seat, temperature=0.7
                    )
                else:
                    # Do not immediately re-announce on another card in the same round.
                    agents[seat] = HoldRoundTeacher(
                        net,
                        hold_haul=source.haul,
                        hold_round=source.round,
                        seed=rollout_seed * 31 + seat,
                        temperature=0.7,
                    )
                result = play_game(
                    agents, seed=rollout_seed, env=env,
                    blast_choice_sees_fronts=not blind,
                )
                if result.aborted:
                    raise RuntimeError("aborted blast-trigger continuation")
                scores = np.asarray(result.scores)
                if scores[seat] == scores.max():
                    values[position] += 1.0 / int((scores == scores.max()).sum())
        values /= samples
        rows.append((ctx, actions, values))
        metadata.append({"seat": seat, "game_seed": game_seed})
        print(
            json.dumps(
                {
                    "position": index + 1,
                    "game_seed": game_seed,
                    "haul": source.haul,
                    "round": source.round,
                    "actions": actions,
                    "win_shares": [round(float(v), 4) for v in values],
                    "continuations": len(actions) * samples,
                }
            ),
            flush=True,
        )
    return rows, metadata


def build_examples(frozen: PolicyNet, rows, *, temperature: float, prior_weight: float):
    examples = []
    for ctx, actions, values in rows:
        mask = action_mask(ctx)
        frozen_probs = frozen.masked_probs(
            state_action_matrix(ctx), mask, temperature=0.7
        )
        shifted = np.exp((values - values.max()) / max(temperature, 1e-6))
        soft = np.zeros(N_ACTIONS, dtype=np.float64)
        for action, weight in zip(actions, shifted / shifted.sum()):
            soft[action] = weight
        # Mix with the frozen policy on the same candidates so the fit only moves
        # the pairwise preference it was given evidence about.
        mix = np.zeros(N_ACTIONS, dtype=np.float64)
        pool = float(frozen_probs[actions].sum())
        if pool <= 0:
            continue
        for action in actions:
            mix[action] = (1 - prior_weight) * soft[action] + prior_weight * (
                frozen_probs[action] / pool
            )
        total = mix[actions].sum()
        if total <= 0:
            continue
        mix /= total
        examples.append((ctx, mix))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", default="checkpoints/selfplay4.npz")
    parser.add_argument("--out", default="checkpoints/selfplay4_blast_trigger.npz")
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--held-out", type=int, default=32)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--prior-weight", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--blind", action="store_true",
                        help="label and evaluate under the printed face-down rule")
    parser.add_argument("--seed", type=int, default=5_100_000)
    parser.add_argument("--held-seed", type=int, default=5_200_000)
    parser.add_argument("--name", default="selfplay4_blast_trigger")
    parser.add_argument("--dry-collect", action="store_true")
    args = parser.parse_args()

    frozen = PolicyNet.load(args.resume)
    blind = bool(args.blind)

    print("harvesting training positions ...", flush=True)
    train_positions = collect(
        frozen, args.positions, args.seed, blind=blind
    )
    print(f"  {len(train_positions)} positions", flush=True)
    print("harvesting held-out positions ...", flush=True)
    held_positions = collect(
        frozen, args.held_out, args.held_seed, blind=blind
    )
    print(f"  {len(held_positions)} positions", flush=True)
    if args.dry_collect:
        return

    print("labelling training positions ...", flush=True)
    train_rows, _ = label(frozen, train_positions, args.samples, args.seed + 1, blind=blind)
    print("labelling held-out positions ...", flush=True)
    held_rows, _ = label(frozen, held_positions, args.samples, args.held_seed + 1, blind=blind)

    before_train = assess(frozen, train_rows)
    before_held = assess(frozen, held_rows)
    print(json.dumps({"before_train": before_train, "before_held": before_held}, indent=2))

    net = PolicyNet.load(args.resume, lr=args.lr)
    net.name = args.name
    examples = build_examples(
        frozen,
        train_rows,
        temperature=args.temperature,
        prior_weight=args.prior_weight,
    )
    losses = net.fit_tactical(examples, epochs=args.epochs, seed=args.seed)
    after_train = assess(net, train_rows)
    after_held = assess(net, held_rows)
    net.trained_generations = frozen.trained_generations
    net.save(args.out)

    summary = {
        "checkpoint": str(args.out),
        "resume": args.resume,
        "blind": blind,
        "positions": len(train_rows),
        "held_out": len(held_rows),
        "samples": args.samples,
        "continuations": int(sum(len(r[1]) for r in train_rows) * args.samples),
        "held_continuations": int(sum(len(r[1]) for r in held_rows) * args.samples),
        "before_train": before_train,
        "after_train": after_train,
        "before_held": before_held,
        "after_held": after_held,
        "final_loss": float(losses[-1]) if losses else None,
    }
    history = args.out.replace(".npz", ".history.json")
    Path(history).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
