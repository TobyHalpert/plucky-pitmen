"""Can a rival's departure tell you where a dragon is? Price the inference.

Each backside colour is ten cards with two dragons, so a dragon sitting in
anybody's column is a dragon that is *not* in the draw pile.  You can see your
own column's fronts; you cannot see a rival's.  One public tell is available: a
seat that *walks off* a card in its own column had that card turned towards it,
so the step is evidence the card was a dragon -- it would otherwise stay on a gem
it can see.  `features.suspected_dragon_counts` counts those departures per
colour and `features.backside_stats(trust_suspected=...)` prices them.

This probe answers three questions in order, because the first two are what would
justify the third:

1. **Is the tell accurate?**  For every departure, was the abandoned card really
   a dragon, compared with the base rate for an own-column card?  The env holds
   the ground truth; no policy can see it.
2. **How much does it move the risk?**  Per colour, with the inference off and on.
3. **Does it change what a seat does, and is the table better for it?**  A paired
   counterfactual on the blast announcement, and a whole-table comparison of four
   nets with the inference on and off over the same seeds.

Usage
-----
    python probe_vacated_dragons.py --checkpoint checkpoints/selfplay4.npz --games 400
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

import features
from agents import NeuralAgent
from analyze_blasts import bootstrap_ci
from features import backside_stats, suspected_dragon_counts
from game_env import DRAGON, MineEnv
from learning import PolicyNet
from probe_blast_colour import announce_probability
from selfplay import play_game


def _nets(net: PolicyNet, game_seed: int, seats: int = 4, cls=NeuralAgent, **kwargs):
    return [
        cls(net, seed=game_seed * 1000 + seat, temperature=0.7, **kwargs)
        for seat in range(seats)
    ]


def measure_tell(net: PolicyNet, *, games: int, seed: int) -> dict[str, Any]:
    """Score every recorded departure against the truth the env keeps."""
    departures = dragons = 0
    cards = card_dragons = 0
    per_colour = [0, 0, 0]
    per_colour_dragons = [0, 0, 0]

    for game in range(games):
        game_seed = seed + game
        env = MineEnv(opponents=3)
        env.reset(seed=game_seed)
        play_game(_nets(net, game_seed), seed=game_seed, env=env)
        for _player, _row, _column, back, front in env.vacated_events:
            departures += 1
            per_colour[int(back)] += 1
            if int(front) == DRAGON:
                dragons += 1
                per_colour_dragons[int(back)] += 1
        for row in env.rows:
            for card in row:
                if card is None:
                    continue
                cards += 1
                card_dragons += int(int(card[0]) == DRAGON)

    return {
        "games": games,
        "departures": departures,
        "departures_per_game": departures / games,
        "departures_that_were_dragons": dragons,
        "precision": dragons / departures if departures else 0.0,
        "mine_cards": cards,
        "mine_dragons": card_dragons,
        "base_rate": card_dragons / cards if cards else 0.0,
        "per_colour": per_colour,
        "per_colour_dragons": per_colour_dragons,
    }


def decision_effect(net: PolicyNet, *, games: int, seed: int) -> dict[str, Any]:
    """Paired counterfactual: the same position with the evidence added/removed."""
    contexts: list[Any] = []

    class RecordingNet(NeuralAgent):
        def decide(self, ctx):
            if suspected_dragon_counts(ctx).sum() > 0:
                contexts.append(ctx)
            return super().decide(ctx)

    for game in range(games):
        game_seed = seed + game
        play_game(_nets(net, game_seed, cls=RecordingNet), seed=game_seed)

    # Both distributions have to be scored with the inference ON, or the
    # counterfactual compares two identical encoders and finds nothing by
    # construction.
    previous = features.TRUST_SUSPECTED_DRAGONS
    features.TRUST_SUSPECTED_DRAGONS = 1.0
    try:
        without_evidence = [
            announce_probability(net, dataclasses.replace(ctx, vacated_backsides=()))
            for ctx in contexts
        ]
        with_evidence = [announce_probability(net, ctx) for ctx in contexts]
    finally:
        features.TRUST_SUSPECTED_DRAGONS = previous
    deltas = [a - b for a, b in zip(with_evidence, without_evidence)]
    low, high = bootstrap_ci(deltas) if deltas else (0.0, 0.0)
    risk_off = [
        float(backside_stats(dataclasses.replace(ctx, vacated_backsides=()))[3].min())
        for ctx in contexts
    ]
    risk_on = [float(backside_stats(ctx, trust_suspected=1.0)[3].min()) for ctx in contexts]
    return {
        "positions_with_evidence": len(contexts),
        "mean_prob_without": float(np.mean(without_evidence)) if contexts else 0.0,
        "mean_prob_with": float(np.mean(with_evidence)) if contexts else 0.0,
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
        "delta_ci": (low, high),
        "share_rising": float(np.mean([d > 0 for d in deltas])) if deltas else 0.0,
        "mean_best_risk_off": float(np.mean(risk_off)) if risk_off else 0.0,
        "mean_best_risk_on": float(np.mean(risk_on)) if risk_on else 0.0,
    }


def table_value(net: PolicyNet, *, games: int, seed: int, trust: float) -> dict[str, Any]:
    previous = features.TRUST_SUSPECTED_DRAGONS
    features.TRUST_SUSPECTED_DRAGONS = trust
    try:
        scores: list[float] = []
        margins: list[float] = []
        dragons = hauls = aborted = blades = 0
        for game in range(games):
            game_seed = seed + game
            result = play_game(_nets(net, game_seed), seed=game_seed)
            aborted += int(result.aborted)
            blades += sum(len(haul.blasts) for haul in result.hauls)
            for haul in result.hauls:
                hauls += 1
                dragons += int("dragon" in haul.outcome)
            for seat, score in enumerate(result.scores):
                rivals = [s for index, s in enumerate(result.scores) if index != seat]
                scores.append(score)
                margins.append(score - float(np.mean(rivals)))
    finally:
        features.TRUST_SUSPECTED_DRAGONS = previous
    return {
        "trust": trust,
        "games": games,
        "mean_score": float(np.mean(scores)),
        "mean_margin": float(np.mean(margins)),
        "dragon_rate": dragons / hauls if hauls else 0.0,
        "blasts_fired": blades,
        "aborted": aborted,
        "margins": margins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/selfplay4.npz")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2_120_000)
    parser.add_argument("--out", default="reports/vacated_dragons.md")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    net = PolicyNet.load(args.checkpoint)
    probe_games = max(80, args.games)
    tell = measure_tell(net, games=probe_games, seed=args.seed)
    effect = decision_effect(net, games=probe_games, seed=args.seed + 1)
    off = table_value(net, games=args.games, seed=args.seed, trust=0.0)
    on = table_value(net, games=args.games, seed=args.seed, trust=1.0)
    paired = [a - b for a, b in zip(on["margins"], off["margins"])]
    low, high = bootstrap_ci(paired)

    report = {"tell": tell, "effect": effect, "off": off, "on": on}
    lines = [
        "# Does a rival's departure reveal a dragon?\n\n",
        f"`{args.checkpoint}`, four-seat neural games, seed {args.seed}. A *departure* "
        f"is a seat stepping from a card in its own column onto another card; the env "
        f"records the front it left, and no policy can see that.\n\n",
        "## 1. How accurate is the tell?\n\n",
        "| | value |\n| :-- | --: |\n",
        f"| departures recorded | {tell['departures']} "
        f"({tell['departures_per_game']:.2f}/game) |\n",
        f"| ...that were dragons | {tell['departures_that_were_dragons']} |\n",
        f"| **precision** | **{100 * tell['precision']:.1f}%** |\n",
        f"| base rate, any card left in the mine | {100 * tell['base_rate']:.1f}% |\n",
        "\n## 2. How much does it move the risk?\n\n",
        f"At the {effect['positions_with_evidence']} positions where the evidence "
        f"already exists, the lowest per-colour risk is "
        f"{effect['mean_best_risk_off']:.3f} with the inference off and "
        f"{effect['mean_best_risk_on']:.3f} with it on.\n\n",
        "## 3. Does it change a decision, and is the table better for it?\n\n",
        f"Paired counterfactual on the blast announcement: P(announce) "
        f"{effect['mean_prob_without']:.4f} without the evidence, "
        f"{effect['mean_prob_with']:.4f} with it -- shift "
        f"{effect['mean_delta']:+.4f} [{effect['delta_ci'][0]:+.4f}, "
        f"{effect['delta_ci'][1]:+.4f}], rising in "
        f"{100 * effect['share_rising']:.0f}% of positions.\n\n",
        f"Whole table, {args.games} paired games each:\n\n",
        "| inference | mean score / seat | mean margin | hauls ending on a dragon | blasts fired |\n",
        "| :-- | --: | --: | --: | --: |\n",
        f"| off (default) | {off['mean_score']:+.2f} | {off['mean_margin']:+.2f} | "
        f"{100 * off['dragon_rate']:.1f}% | {off['blasts_fired']} |\n",
        f"| on (trust = 1) | {on['mean_score']:+.2f} | {on['mean_margin']:+.2f} | "
        f"{100 * on['dragon_rate']:.1f}% | {on['blasts_fired']} |\n",
        f"\nPaired per-seat margin, on minus off: **{np.mean(paired):+.2f}** "
        f"[{low:+.2f}, {high:+.2f}] over {len(paired)} seat-games.\n",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines))
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "tell": tell,
                    "effect": effect,
                    "off": {k: v for k, v in off.items() if k != "margins"},
                    "on": {k: v for k, v in on.items() if k != "margins"},
                },
                indent=2,
            )
        )
    print("".join(lines))


if __name__ == "__main__":
    main()
