"""Does a neural seat *actually* flinch when a rival's blast is pending?

``analyze_blasts.py`` answers the counterfactual version of this question: the
same context scored twice, once with a credible opponent blast pending.  That
is the right test of whether the threat is in the policy at all, but it is
reported as a total-variation distance and a probability shift, and a shift can
come from any action.

This probe measures what happens in played games instead.  One tracked seat
runs the checkpoint against three scripted opponents that blast every chance
they get (``rule`` with ``blast=always``), and every decision it takes while an
opponent's blast is pending is counted by the action it chose:

* **retreat**  - ``PIT_CAGE`` or ``LORRY``: the answer that banks the haul now.
* **stay**     - a card move, or a pass: the haul continues under the threat.
* **blocked**  - a threat decision with no cage legal at all.

A seat that flinches shows ``retreat`` climbing while its margin against a
blasting field stops being negative.  A seat that ignores threats keeps the
same answer distribution as the calm case, which is what BALANCE.md §6 found.

Usage
-----
    python probe_threat_reaction.py \
        --checkpoints checkpoints/selfplay4.npz,checkpoints/selfplay4_threatleague.npz \
        --games 300 --seed 515000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from agents import NeuralAgent, make_scripted_agent
from analyze_blasts import bootstrap_ci, reaction_test
from game_env import LORRY, PASS, PIT_CAGE
from learning import PolicyNet
from selfplay import play_series


def _threatened(ctx) -> bool:
    return any(
        opponent.blast_pending and not opponent.escaped and not opponent.dead
        for opponent in ctx.opponents
    )


def play_arm(
    path: str, *, games: int, seed: int, players: int = 4
) -> dict[str, Any]:
    net = PolicyNet.load(path)
    counts: dict[str, int] = {
        "threat_decisions": 0,
        "cage": 0,
        "lorry": 0,
        "pass": 0,
        "stay": 0,
        "blocked": 0,
        "announced": 0,
        "control_decisions": 0,
        "control_cage": 0,
    }

    class TrackingAgent(NeuralAgent):
        def decide(self, ctx):
            choice = super().decide(ctx)
            if _threatened(ctx):
                counts["threat_decisions"] += 1
                if choice.action == PIT_CAGE:
                    counts["cage"] += 1
                elif choice.action == LORRY:
                    counts["lorry"] += 1
                elif choice.action == PASS:
                    counts["pass"] += 1
                elif PIT_CAGE in ctx.legal_actions:
                    counts["stay"] += 1
                else:
                    counts["blocked"] += 1
            elif PIT_CAGE in ctx.legal_actions:
                # Control: the escape was on offer *without* a pending blast.
                # If it is taken here too, the threat did not cause it.
                counts["control_decisions"] += 1
                counts["control_cage"] += int(choice.action == PIT_CAGE)
            return choice

    factories = [
        lambda s: TrackingAgent(net, seed=s, temperature=0.7, name="tracked")
    ] + [
        (lambda s: make_scripted_agent("rule", seed=s, blast="always"))
        for _ in range(players - 1)
    ]
    series = play_series(factories, games, seed=seed, rotate_seats=True)

    wins = tops = aborted = 0
    margins: list[float] = []
    scores: list[float] = []
    for game in series:
        seat = game.policy_of_seat.index(0)
        own = game.result.scores[seat]
        rivals = [s for index, s in enumerate(game.result.scores) if index != seat]
        wins += int(game.result.winner == seat)
        tops += int(max(game.result.scores) == own)
        aborted += int(game.result.aborted)
        scores.append(own)
        margins.append(own - float(np.mean(rivals)))

    threat_decisions = counts["threat_decisions"]
    retreat = counts["cage"] + counts["lorry"]
    return {
        "checkpoint": path,
        "games": games,
        "aborted": aborted,
        "win_rate": wins / games,
        "top_rate": tops / games,
        "mean_score": float(np.mean(scores)),
        "margin": float(np.mean(margins)),
        "margin_ci": bootstrap_ci(margins),
        "threat_decisions_per_game": threat_decisions / games,
        "retreat_rate": retreat / threat_decisions if threat_decisions else 0.0,
        "cage_rate": counts["cage"] / threat_decisions if threat_decisions else 0.0,
        "lorry_rate": counts["lorry"] / threat_decisions if threat_decisions else 0.0,
        "pass_rate": counts["pass"] / threat_decisions if threat_decisions else 0.0,
        "stay_rate": counts["stay"] / threat_decisions if threat_decisions else 0.0,
        "blocked_rate": counts["blocked"] / threat_decisions if threat_decisions else 0.0,
        "control_decisions_per_game": counts["control_decisions"] / games,
        "control_cage_rate": (
            counts["control_cage"] / counts["control_decisions"]
            if counts["control_decisions"]
            else 0.0
        ),
        "counts": counts,
    }


def render(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Does the net flinch? (behaviour, not counterfactual)\n\n",
        "One tracked seat against three `rule` opponents with `blast=always`.\n\n",
        "| checkpoint | games | win % | margin | margin 95% CI | threat decisions/game "
        "| retreated | cage | lorry | pass | stayed | no cage legal "
        "| cage rate with no threat |\n",
        "| :-- | --: | --: | --: | :-- | --: | --: | --: | --: | --: | --: | --: | --: |\n",
    ]
    for row in rows:
        low, high = row["margin_ci"]
        lines.append(
            f"| `{Path(row['checkpoint']).stem}` | {row['games']} | "
            f"{100 * row['win_rate']:.1f}% | {row['margin']:+.2f} | "
            f"[{low:+.2f}, {high:+.2f}] | {row['threat_decisions_per_game']:.2f} | "
            f"{100 * row['retreat_rate']:.0f}% | {100 * row['cage_rate']:.0f}% | "
            f"{100 * row['lorry_rate']:.0f}% | {100 * row['pass_rate']:.0f}% | "
            f"{100 * row['stay_rate']:.0f}% | {100 * row['blocked_rate']:.0f}% | "
            f"{100 * row['control_cage_rate']:.0f}% |\n"
        )
    lines.append(
        "\n`retreated` = cage or lorry on a decision taken while a rival's blast "
        "was pending; `stayed` = a card move or pass; `no cage legal` = a "
        "threatened decision with no cage available at all.  The last column is "
        "the control: how often the same seat takes a cage when one was legal and "
        "*no* rival blast was pending, so the threat's own effect is the gap "
        "between the two.\n"
    )
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--seed", type=int, default=515_000)
    parser.add_argument("--players", type=int, default=4, choices=(3, 4, 5))
    parser.add_argument(
        "--reaction-games",
        type=int,
        default=0,
        help="also run the paired counterfactual TV test with this many games",
    )
    parser.add_argument("--out", default="reports/threat_reaction.md")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    paths = [p.strip() for p in args.checkpoints.split(",") if p.strip()]
    rows = [play_arm(p, games=args.games, seed=args.seed, players=args.players) for p in paths]
    for row in rows:
        print(
            f"{row['checkpoint']}: win {row['win_rate']:.3f} margin {row['margin']:+.2f} "
            f"threat decisions/game {row['threat_decisions_per_game']:.2f} "
            f"retreat {100 * row['retreat_rate']:.0f}%",
            flush=True,
        )

    report: dict[str, Any] = {"behaviour": rows}
    if args.reaction_games > 0:
        reaction = {}
        for path in paths:
            net = PolicyNet.load(path)
            reaction[path] = reaction_test(
                net,
                players=args.players,
                games=args.reaction_games,
                seed=args.seed,
            )
            print(f"{path}: paired TV {reaction[path]['mean_tv']:.4f}", flush=True)
        report["reaction"] = reaction

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = render(rows)
    if args.reaction_games > 0:
        text += "\n## Paired counterfactual (same context, threat toggled)\n\n"
        text += "| checkpoint | paired decisions | mean TV [95% CI] | shift to retreat | shift to cage |\n"
        text += "| :-- | --: | :-- | --: | --: |\n"
        for path, stats in report["reaction"].items():
            low, high = stats["reaction_ci"]
            text += (
                f"| `{Path(path).stem}` | {stats['pairs']} | "
                f"{stats['mean_tv']:.4f} [{low:.4f}, {high:.4f}] | "
                f"{stats['mean_retreat_shift']:+.4f} | {stats['mean_cage_shift']:+.4f} |\n"
            )
    out.write_text(text)
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
