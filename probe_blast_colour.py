"""Does the net reason about the colour of a dragon it has already seen?

The rules make one piece of knowledge unusually valuable.  Each of the three
backside colours is a fixed set of ten cards with **two** dragons, so a dragon a
seat can see in its own column is a dragon that is *not* in the draw pile.  A
blast draws three cards face down, keeps two, and wakes the dragon if either is
one -- so the keeper should prefer cards whose backside colour has already spent
its dragons.  Seeing one dragon of a colour halves that colour's risk (1 in 9
rather than 2 in 10); seeing both makes it zero.

Two separate questions, and they have different answers:

1. **The keeper** (which of the three drawn cards to keep).  ``NeuralAgent`` does
   not implement ``blast_choice`` at all: it inherits ``Agent.blast_choice`` ->
   ``agents.preferred_blast_choice``, the hand-written rule that keeps the card
   with the lowest ``features.backside_stats`` risk.  So every agent, trained or
   not, already plays the colour rule.  This probe checks the rule works and
   prices it against picking blind.
2. **The announcement** (whether to blast at all).  This *is* the network's
   decision, and the state carries the per-colour risk, so it could condition on
   it.  This probe measures whether it does: the announcement rate against how
   much the seat knows about the colours.

Usage
-----
    python probe_blast_colour.py --checkpoint checkpoints/selfplay4.npz --games 600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import dataclasses

from agents import NeuralAgent, owns_dynamite_pair
from features import (
    action_mask,
    backside_stats,
    split_action,
    state_action_matrix,
)
from game_env import DRAGON, GEM, PIT_CAGE
from learning import PolicyNet
from selfplay import play_game


def _card_actions(ctx) -> list[int]:
    return [int(a) for a in ctx.legal_actions if int(a) < PIT_CAGE]


class RecordingNet(NeuralAgent):
    """Records blastable decisions and every keeper choice the seat makes."""

    def __init__(self, net, *, log: dict[str, Any], **kwargs) -> None:
        super().__init__(net, **kwargs)
        self.log = log

    def decide(self, ctx):
        choice = super().decide(ctx)
        if owns_dynamite_pair(ctx) and _card_actions(ctx):
            known_dragons, _safe, unknown, risk = backside_stats(ctx)
            self.log["contexts"].append(ctx)
            self.log["blastable"].append(
                {
                    "announced": bool(choice.announce_blast),
                    "best_colour_dragons": float(known_dragons.max()),
                    "min_risk": float(risk.min()),
                    "n_zero_risk_colours": int(
                        sum(1 for index in range(3) if risk[index] == 0 and unknown[index] > 0)
                    ),
                }
            )
        return choice

    def blast_choice(self, cards, ctx):
        _known_dragons, _safe, _unknown, risk = backside_stats(ctx)
        offered = [float(risk[int(back)]) for _front, back in cards]
        choice = super().blast_choice(cards, ctx)
        self.log["keepers"].append(
            {
                "offered": offered,
                "chosen": int(choice),
                "chosen_risk": offered[int(choice)] if choice < len(offered) else None,
                "min_offered": min(offered),
                "zero_offered": any(value == 0.0 for value in offered),
                "picked_zero": bool(
                    offered[int(choice)] == 0.0 if choice < len(offered) else False
                ),
            }
        )
        return choice


def run(
    net: PolicyNet,
    *,
    games: int,
    seed: int,
    players: int = 4,
    temperature: float = 0.7,
    blind: bool = False,
) -> dict[str, Any]:
    log: dict[str, Any] = {
        "blastable": [],
        "keepers": [],
        "kept_fronts": [],
        "contexts": [],
    }
    for game in range(games):
        game_seed = seed + game
        agents = [
            RecordingNet(
                net,
                log=log,
                seed=game_seed * 1000 + seat,
                temperature=temperature,
            )
            for seat in range(players)
        ]
        play_game(
            agents,
            seed=game_seed,
            blast_choice_sees_fronts=not blind,
        )
    return log


def announce_probability(net: PolicyNet, ctx, temperature: float = 0.7) -> float:
    """Probability the policy puts on *any* blast-announcing action here."""
    mask = action_mask(ctx)
    if not mask.any():
        return 0.0
    probs = net.masked_probs(
        state_action_matrix(ctx), mask, temperature=temperature
    )
    return float(
        sum(
            probability
            for action, probability in enumerate(probs)
            if split_action(action)[1]
        )
    )


def without_the_seen_dragon(ctx):
    """The same context with one visible own-column dragon turned into a gem.

    Everything a seat can see stays put except the *fact* it knows: the card is
    still there, still that backside colour, still in its own column.  What
    changes is that this colour no longer has one of its two dragons accounted
    for, so the same colour in the draw pile is 2-in-10 risky instead of 1-in-9.
    Returns ``None`` when the seat can see no dragon at all.
    """
    own = ctx.player_index
    for row_index, row in enumerate(ctx.public_rows):
        if own >= len(row) or row[own] is None:
            continue
        front, back = row[own]
        if front == DRAGON:
            rows = [list(entries) for entries in ctx.public_rows]
            rows[row_index][own] = (GEM, back)
            return dataclasses.replace(ctx, public_rows=tuple(tuple(r) for r in rows))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/selfplay4.npz")
    parser.add_argument("--games", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1_310_000)
    parser.add_argument(
        "--blind",
        action="store_true",
        help="keep blast cards face down (the printed rule) instead of peeking",
    )
    parser.add_argument("--out", default="reports/blast_colour.md")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    net = PolicyNet.load(args.checkpoint)
    log = run(net, games=args.games, seed=args.seed, blind=args.blind)

    blastable = log["blastable"]
    keepers = log["keepers"]

    # 1. Does the announcement rate move with what the seat knows about colours?
    buckets: dict[str, list[float]] = {"0 dragons seen": [], "1": [], "2": []}
    for entry in blastable:
        key = str(int(entry["best_colour_dragons"]))
        buckets.setdefault(key, []).append(float(entry["announced"]))

    # 2. Is the keeper taking the safest colour, and what does it cost not to?
    chosen_risk = np.array([k["chosen_risk"] for k in keepers if k["chosen_risk"] is not None])
    min_risk = np.array([k["min_offered"] for k in keepers if k["chosen_risk"] is not None])
    zero_offers = [k for k in keepers if k["zero_offered"]]
    zero_taken = [k for k in zero_offers if k["picked_zero"]]

    # 3. The paired test: same position, the seen dragon becomes a gem, so the
    # colour carrying it is two-in-ten risky again instead of one-in-nine.
    pairs: list[dict[str, float]] = []
    for ctx in log["contexts"]:
        doctored = without_the_seen_dragon(ctx)
        if doctored is None:
            continue
        pairs.append(
            {
                "known": announce_probability(net, ctx),
                "unknown": announce_probability(net, doctored),
            }
        )
    deltas = [p["known"] - p["unknown"] for p in pairs]

    report = {
        "checkpoint": args.checkpoint,
        "games": args.games,
        "blastable_decisions": len(blastable),
        "announce_rate_overall": float(np.mean([e["announced"] for e in blastable]))
        if blastable
        else 0.0,
        "announce_by_known_dragons": {
            key: {
                "decisions": len(values),
                "announce_rate": float(np.mean(values)) if values else 0.0,
            }
            for key, values in buckets.items()
        },
        "keeper_calls": len(keepers),
        "keeper_took_the_safest": float(np.mean(chosen_risk <= min_risk + 1e-9))
        if len(keepers)
        else 0.0,
        "mean_chosen_risk": float(chosen_risk.mean()) if len(chosen_risk) else 0.0,
        "mean_min_offered": float(min_risk.mean()) if len(min_risk) else 0.0,
        "zero_risk_offered": len(zero_offers),
        "zero_risk_taken": len(zero_taken),
        "kept_fronts": log["kept_fronts"],
        "counterfactual_pairs": len(pairs),
        "mean_announce_prob_with_dragon_known": float(
            np.mean([p["known"] for p in pairs])
        )
        if pairs
        else 0.0,
        "mean_announce_prob_without": float(np.mean([p["unknown"] for p in pairs]))
        if pairs
        else 0.0,
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
        "share_rising": float(np.mean([d > 0 for d in deltas])) if deltas else 0.0,
        "blind": args.blind,
    }

    lines = [
        "# Does the net use the colour of a dragon it has seen?\n\n",
        f"`{args.checkpoint}`, {args.games} four-seat neural games, seed {args.seed}. "
        f"{report['blastable_decisions']} decisions where a blast was legal.\n\n",
        "## 1. The announcement (the network's own decision)\n\n",
        "| dragons already seen of the best colour | decisions | announced |\n",
        "| :-- | --: | --: |\n",
    ]
    for key in ("0", "1", "2"):
        bucket = report["announce_by_known_dragons"][key]
        lines.append(
            f"| {key} | {bucket['decisions']} | {100 * bucket['announce_rate']:.1f}% |\n"
        )
    lines.append(
        f"\nOverall announcement rate: {100 * report['announce_rate_overall']:.1f}%.\n\n"
    )
    lines.append(
        "## 2. The keeper (hand-written, not learned)\n\n"
        f"`NeuralAgent` inherits `Agent.blast_choice` -> `preferred_blast_choice`, "
        f"which keeps the lowest `backside_stats` risk. Over {report['keeper_calls']} "
        f"keeper calls it took the safest offered card "
        f"{100 * report['keeper_took_the_safest']:.1f}% of the time "
        f"(mean risk of the card it kept {report['mean_chosen_risk']:.3f}, mean best "
        f"offered {report['mean_min_offered']:.3f}).\n\n"
        f"A zero-risk colour (both dragons of that backside already seen) was offered "
        f"in {report['zero_risk_offered']} calls and taken in "
        f"{report['zero_risk_taken']} of them.\n\n"
        + (
            "Blast cards were kept **face down** (the printed rule), so the risk "
            "array is the only information the keeper has.\n"
            if args.blind
            else "Blast cards were **peeked at** (the engine default), so a known gem "
            "outranks the colour risk and the colour rule only binds on face-down "
            "cards. Re-run with `--blind` for the printed rule.\n"
        )
    )
    lines.append(
        "## 3. The paired test: does the announcement move when the colour is safer?\n\n"
        f"At {report['counterfactual_pairs']} unchanged positions the one dragon the "
        f"seat can see in its own column is turned into a gem: same card, same "
        f"backside, same board, but that colour is now 2-in-10 risky in the deck "
        f"instead of 1-in-9. If the net reasons about colour, the announcement "
        f"probability should FALL in the doctored world.\n\n"
        f"| world | mean P(announce) |\n| :-- | --: |\n"
        f"| dragon known (safer colour) | {report['mean_announce_prob_with_dragon_known']:.4f} |\n"
        f"| same card is a gem (riskier colour) | {report['mean_announce_prob_without']:.4f} |\n\n"
        f"Mean shift **{report['mean_delta']:+.4f}**, rising in "
        f"{100 * report['share_rising']:.0f}% of the paired positions.\n"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    print("".join(lines))


if __name__ == "__main__":
    main()
