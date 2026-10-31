"""Is displacement worth anything - for the seat that pushes, or the one that parks?

The viewer's seed 7 has two ejections in it, both from sampler tails: P4 pushes
P3 off its own front-row gem in haul 1, P3 pushes P4 off its own front-row gem in
haul 3. Both look like the sharpest move in the game. `analyze_game.py --forks`
prices them at +1.07 and +1.57 *from those positions*.

This probe asks the question the single game cannot: is the mechanic worth
anything over a league of games, for either side of it?

* ``pusher``   - the same network, but whenever a displacement is on offer it
  takes the one the hand-tuned rule policy would take. That is the aggressive
  side, at 8x the network's own rate.
* ``cautious`` - the same network, but it will not end a planning phase on a card
  in its own column while a rival is still deep enough to displace it and its
  hands are empty. (A push sends the victim to a pit cage, i.e. safely out of the
  mine with whatever it is holding; empty-handed that is the whole haul.)
* ``row0``     - the narrow version of the same rule: only the front row.

All arms run on the same seeds and the same per-seat agent seeds, so the numbers
are paired. ``--places`` also runs the aggressive seat in the other seats.

    python probe_displacement.py --games 300
    python probe_displacement.py --games 1000 --seat 2
"""

from __future__ import annotations

import argparse
import collections

import numpy as np

import features
from agents import ActionChoice, NeuralAgent
from game_env import GEM, LORRY, MAX_COLUMNS, PIT_CAGE, MineEnv
from learning import PolicyNet
from rule_policy import deterministic_policy
from selfplay import play_game


def movement_depth(position: int, cage_index: int | None) -> int:
    """How deep a figure still reaches, as ``MineEnv._movement_depth`` has it."""
    if position == PIT_CAGE:
        return cage_index if cage_index is not None else 0
    if position == LORRY:
        return 0
    return position


def displacements_on_offer(ctx) -> list[int]:
    """Legal moves that would push a rival standing on a card in its own column."""
    offered = []
    for action in ctx.legal_actions:
        action = int(action)
        if not 0 <= action < PIT_CAGE:
            continue
        row, column = divmod(action, MAX_COLUMNS)
        for opponent in ctx.opponents:
            if opponent.escaped or opponent.dead:
                continue
            if opponent.position != row or opponent.column != column:
                continue
            if column == opponent.player_index:
                offered.append(action)
    return offered


def can_be_pushed(ctx, row: int) -> bool:
    """Whether any active rival is still deep enough to move onto ``row``."""
    for opponent in ctx.opponents:
        if opponent.escaped or opponent.dead:
            continue
        if movement_depth(opponent.position, opponent.cage_index) > row:
            return True
    return False


class Pusher(NeuralAgent):
    """The network, but it takes the hand rule's displacements for it."""

    def decide(self, ctx):
        offered = displacements_on_offer(ctx)
        if offered:
            choice = int(deterministic_policy(ctx))
            if choice in offered and features.action_mask(ctx)[choice]:
                return ActionChoice(choice)
        return super().decide(ctx)


def cautious_mask(seat: int, only_front_row: bool):
    """A mask that hides own-column cards a rival can still push the seat off.

    The rule only bites when the seat is empty-handed: a push then costs it the
    whole haul, whereas a seat holding a gem is *banked* by the push.
    """
    original = features.action_mask

    def patched(ctx):
        mask = original(ctx)
        if ctx.player_index != seat:
            return mask
        if any(front == GEM for front, _ in ctx.private_cards):
            return mask
        trial = mask.copy()
        for action in ctx.legal_actions:
            action = int(action)
            if not 0 <= action < PIT_CAGE:
                continue
            row, column = divmod(action, MAX_COLUMNS)
            if column != ctx.player_index:
                continue
            if only_front_row and row != 0:
                continue
            if can_be_pushed(ctx, row):
                trial[action] = False
        return trial if trial.any() else mask

    return patched


def run_arm(net: PolicyNet, arm: str, seat: int, games: int) -> dict:
    """Play ``games`` seeded games with one seat in the given arm."""
    original = features.action_mask
    if arm in ("cautious", "row0"):
        features.action_mask = cautious_mask(seat, only_front_row=arm == "row0")
    try:
        scores, wins = [], []
        pushes_made = pushes_taken = 0
        for seed in range(games):
            agents = []
            for index in range(4):
                make_pusher = arm == "pusher" and index == seat
                cls = Pusher if make_pusher else NeuralAgent
                agents.append(cls(net, seed=seed * 10 + index, temperature=0.7))
            env = MineEnv(opponents=3, opponent_policy="simple")
            env.reset(seed=seed)
            result = play_game(agents, seed=seed, env=env)
            if not result.scores:
                continue
            scores.append(result.scores[seat])
            wins.append(1.0 if result.winner == seat else 0.0)
            pushes_made += sum(1 for _victim, by in env.displacement_events if by == seat)
            pushes_taken += sum(1 for victim, _by in env.displacement_events if victim == seat)
    finally:
        features.action_mask = original
    return {
        "scores": np.array(scores),
        "wins": np.array(wins),
        "made": pushes_made,
        "taken": pushes_taken,
    }


def report(arm: str, result: dict, base: dict) -> None:
    games = len(result["scores"])
    gap = result["scores"] - base["scores"][:games]
    win_gap = result["wins"] - base["wins"][:games]
    score_se = gap.std(ddof=1) / np.sqrt(games) if games > 1 else 0.0
    win_se = win_gap.std(ddof=1) / np.sqrt(games) * 100 if games > 1 else 0.0
    print(
        f"  {arm:9s} win {result['wins'].mean():5.1%}  score {result['scores'].mean():+.2f}"
        f"  pushed {result['made'] / games:4.2f}/game  pushed off "
        f"{result['taken'] / games:4.2f}/game"
        f"   paired {win_gap.mean() * 100:+.1f}pp (se {win_se:.1f}),"
        f" {gap.mean():+.2f} pts (se {score_se:.2f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/selfplay4.npz")
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--seat", type=int, default=0)
    args = parser.parse_args()

    net = PolicyNet.load(args.checkpoint)
    print(f"four seats, {args.games} seed-paired games, the arm in seat {args.seat + 1}")
    arms = collections.OrderedDict(
        (name, run_arm(net, name, args.seat, args.games))
        for name in ("baseline", "pusher", "cautious", "row0")
    )
    base = arms["baseline"]
    print(
        f"  {'baseline':9s} win {base['wins'].mean():5.1%}  "
        f"score {base['scores'].mean():+.2f}  pushed {base['made'] / len(base['scores']):4.2f}/game"
        f"  pushed off {base['taken'] / len(base['scores']):4.2f}/game"
    )
    for name, result in arms.items():
        if name != "baseline":
            report(name, result, base)


if __name__ == "__main__":
    main()
