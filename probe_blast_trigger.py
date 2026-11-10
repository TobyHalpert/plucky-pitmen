"""Does a checkpoint actually fire blasts, under each blast rule, and does it pay?

`analyze_blasts.py` reports blasts per game for a seat, but it plays the engine
default, which **peeks** at the three drawn cards.  The printed rule keeps them
face down, and the difference is not cosmetic: with the fronts visible the keeper
can always dodge a dragon, so a committed blast is cheap, and BALANCE §5 prices it
at +0.49 from a real position against +0.06 face down.

This probe reports, per checkpoint and per rule, what the seat does with the
mechanic -- announcements, cancellations, blasts that actually stood, and how
often a fired blast woke the dragon -- and optionally a paired head-to-head of one
candidate seat against three copies of a baseline checkpoint, all under the same
rule.

Usage
-----
    python probe_blast_trigger.py --checkpoints checkpoints/selfplay4.npz,checkpoints/selfplay4_blindblast.npz
    python probe_blast_trigger.py --candidate checkpoints/selfplay4_blindblast.npz \
        --baseline checkpoints/selfplay4.npz --games 600 --blind
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from agents import NeuralAgent
from analyze_blasts import bootstrap_ci
from game_env import MineEnv
from learning import PolicyNet
from selfplay import play_game


def behaviour(
    path: str, *, games: int, seed: int, blind: bool, temperature: float = 0.7
) -> dict[str, Any]:
    net = PolicyNet.load(path)
    original_begin, original_cancel = MineEnv.begin_blast, MineEnv._cancel_blast
    counts = {"announced": 0, "cancelled": 0, "fired": 0, "dragons": 0, "blast_hauls": 0}

    def begin(env, player_index):
        counts["announced"] += 1
        return original_begin(env, player_index)

    def cancel(env, player_index):
        if env.players[player_index].blast_pending:
            counts["cancelled"] += 1
        return original_cancel(env, player_index)

    MineEnv.begin_blast, MineEnv._cancel_blast = begin, cancel
    try:
        for game in range(games):
            game_seed = seed + game
            agents = [
                NeuralAgent(net, seed=game_seed * 1000 + seat, temperature=temperature)
                for seat in range(4)
            ]
            result = play_game(
                agents, seed=game_seed, blast_choice_sees_fronts=not blind
            )
            for haul in result.hauls:
                counts["blast_hauls"] += len(haul.blasts)
                counts["fired"] += len(haul.blasts)
                counts["dragons"] += int(haul.outcome == "blast_dragon")
    finally:
        MineEnv.begin_blast, MineEnv._cancel_blast = original_begin, original_cancel

    games_x_seats = games * 4
    return {
        "checkpoint": path,
        "blind": blind,
        "games": games,
        "announced_per_game_seat": counts["announced"] / games_x_seats,
        "cancelled_per_game_seat": counts["cancelled"] / games_x_seats,
        "fired_per_game_seat": counts["fired"] / games_x_seats,
        "fire_rate_of_announcements": (
            1 - counts["cancelled"] / counts["announced"] if counts["announced"] else 0.0
        ),
        "fired_that_woke_a_dragon": counts["dragons"],
        "fired": counts["fired"],
        "dragon_rate_of_fired": counts["dragons"] / counts["fired"] if counts["fired"] else 0.0,
    }


def head_to_head(
    candidate: str, baseline: str, *, games: int, seed: int, blind: bool
) -> dict[str, Any]:
    net_base = PolicyNet.load(baseline)
    net_cand = PolicyNet.load(candidate)
    wins_base: list[float] = []
    wins_cand: list[float] = []
    for game in range(games):
        game_seed = seed + game
        seat = game % 4
        for arm, wins in (("base", wins_base), ("cand", wins_cand)):
            chosen = net_cand if arm == "cand" else net_base
            agents = [
                NeuralAgent(net_base, seed=game_seed * 1000 + index, temperature=0.7)
                for index in range(4)
            ]
            agents[seat] = NeuralAgent(
                chosen, seed=game_seed * 1000 + seat, temperature=0.7
            )
            result = play_game(
                agents, seed=game_seed, blast_choice_sees_fronts=not blind
            )
            wins.append(float(result.scores[seat] == max(result.scores)))
    paired = [a - b for a, b in zip(wins_cand, wins_base)]
    low, high = bootstrap_ci(paired)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "games": games,
        "blind": blind,
        "candidate_top_rate": float(np.mean(wins_cand)),
        "baseline_top_rate": float(np.mean(wins_base)),
        "paired_change": float(np.mean(paired)),
        "paired_ci": (low, high),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--baseline", default="checkpoints/selfplay4.npz")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=4_240_000)
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--out", default="reports/blast_trigger.md")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    rows = []
    paths = (
        [p.strip() for p in args.checkpoints.split(",") if p.strip()]
        if args.checkpoints
        else []
    )
    if args.candidate:
        paths = paths or [args.baseline, args.candidate]
    for path in paths:
        for blind in ((False, True) if args.blind else (False,)):
            rows.append(behaviour(path, games=args.games, seed=args.seed, blind=blind))
    report: dict[str, Any] = {"behaviour": rows}
    if args.candidate:
        report["head_to_head"] = head_to_head(
            args.candidate,
            args.baseline,
            games=args.games,
            seed=args.seed,
            blind=args.blind,
        )

    lines = [
        "# Does the seat actually fire, and under which rule?\n\n",
        f"{args.games} four-seat games per row, seed {args.seed}. Announcements and "
        f"cancellations are per game *per seat*.\n\n",
        "| checkpoint | blast rule | announced | cancelled | **fired** | fired that woke a dragon |\n",
        "| :-- | :-- | --: | --: | --: | --: |\n",
    ]
    for row in rows:
        lines.append(
            f"| `{Path(row['checkpoint']).stem}` | "
            f"{'printed (face down)' if row['blind'] else 'engine default (peek)'} | "
            f"{row['announced_per_game_seat']:.2f} | "
            f"{row['cancelled_per_game_seat']:.2f} | "
            f"**{row['fired_per_game_seat']:.3f}** | "
            f"{row['fired_that_woke_a_dragon']}/{row['fired']} "
            f"({100 * row['dragon_rate_of_fired']:.0f}%) |\n"
        )
    if "head_to_head" in report:
        h2h = report["head_to_head"]
        low, high = h2h["paired_ci"]
        lines.append(
            f"\nPaired head-to-head, one candidate seat against three "
            f"`{Path(h2h['baseline']).stem}` seats under the "
            f"{'printed rule' if h2h['blind'] else 'peek engine'}: candidate tops "
            f"{100 * h2h['candidate_top_rate']:.1f}% against "
            f"{100 * h2h['baseline_top_rate']:.1f}% for the baseline seat, paired "
            f"change {100 * h2h['paired_change']:+.2f} pp "
            f"[{100 * low:+.2f}, {100 * high:+.2f}] over {h2h['games']} games.\n"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    print("".join(lines))


if __name__ == "__main__":
    main()
