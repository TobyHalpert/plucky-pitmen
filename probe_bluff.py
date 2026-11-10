"""Is the table's flinch worth bluffing? Price an announced blast nobody fires.

The viewer shows a pattern that reads as two mistakes at once:

* a net collects dynamite, announces a blast, and then cancels it — the
  announcement never becomes an explosion, so the turn looks wasted;
* every other seat answers the announcement by walking to a pit cage, giving up
  the card it had planned to take.

Those two can be rational together: the announcement is a *public* commitment
that resolves only if the figure is still standing on the card when the phase
closes, so a seat holding two dynamite can buy the table's flinch and then abort
it.  Whether that trade pays is an empirical question, and this probe answers it
by putting a seat that does exactly that against three unchanged nets, with the
identical seat playing the identical movement policy but *never* announcing as
the control.  Same seeds, same net seeds, seats rotated, so the difference is
the bluff and nothing else.

Usage
-----
    python probe_bluff.py --checkpoint checkpoints/selfplay4.npz --games 400
    python probe_bluff.py --games 400 --no-bluff        # the control arm alone
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from agents import ActionChoice, BlufferAgent, NeuralAgent
from analyze_blasts import bootstrap_ci
from game_env import PIT_CAGE, MineEnv
from learning import PolicyNet
from selfplay import play_game


def _threatened_by(ctx, bluffer_index: int) -> bool:
    return any(
        opponent.player_index == bluffer_index
        and opponent.blast_pending
        and not opponent.escaped
        and not opponent.dead
        for opponent in ctx.opponents
    )


class MuteNet(NeuralAgent):
    """The unchanged net with its blast announcements switched off.

    Isolates the *net's own* announce-and-cancel cycle: same weights, same
    sampling, same movement distribution, no announcements at all.
    """

    def decide(self, ctx):
        choice = super().decide(ctx)
        return ActionChoice(int(choice.action), False)


def play_arm(
    net: PolicyNet,
    *,
    games: int,
    seed: int,
    label: str,
    players: int = 4,
    temperature: float = 0.7,
) -> dict[str, Any]:
    original_begin = MineEnv.begin_blast
    original_cancel = MineEnv._cancel_blast

    counts = {
        "announced": 0,
        "cancelled": 0,
        "blast_hauls": 0,
        "dragons_woken_by_bluffer": 0,
        "rival_threat_decisions": 0,
        "rival_cage_answers": 0,
        "rival_escapes_on_threat_haul": 0,
    }
    rows: list[dict[str, float]] = []
    bluffer_seat = [0]

    def begin(env, player_index):
        if player_index == bluffer_seat[0]:
            counts["announced"] += 1
        return original_begin(env, player_index)

    def cancel(env, player_index):
        if player_index == bluffer_seat[0] and env.players[player_index].blast_pending:
            counts["cancelled"] += 1
        return original_cancel(env, player_index)

    class CountingNet(NeuralAgent):
        def decide(self, ctx):
            choice = super().decide(ctx)
            if _threatened_by(ctx, bluffer_seat[0]):
                counts["rival_threat_decisions"] += 1
                counts["rival_cage_answers"] += int(choice.action == PIT_CAGE)
            return choice

    MineEnv.begin_blast, MineEnv._cancel_blast = begin, cancel
    try:
        for game in range(games):
            game_seed = seed + game
            bluffer_seat[0] = game % players
            agents = [
                CountingNet(net, seed=game_seed * 1000 + seat, temperature=temperature)
                for seat in range(players)
            ]
            if label == "net":
                tracked = CountingNet(
                    net, seed=game_seed * 1000 + bluffer_seat[0], temperature=temperature
                )
            elif label == "net, no announcements":
                tracked = MuteNet(
                    net, seed=game_seed * 1000 + bluffer_seat[0], temperature=temperature
                )
            elif label == "quiet (control)":
                tracked = BlufferAgent(bluff=False)
            else:
                tracked = BlufferAgent(bluff=label.replace("bluff, ", ""))
            agents[bluffer_seat[0]] = tracked
            result = play_game(agents, seed=game_seed)
            seat = bluffer_seat[0]
            own = result.scores[seat]
            rivals = [s for index, s in enumerate(result.scores) if index != seat]
            counts["blast_hauls"] += sum(
                1 for haul in result.hauls if seat in haul.blasts
            )
            for haul in result.hauls:
                if haul.dragon_player == seat:
                    counts["dragons_woken_by_bluffer"] += 1
                if seat in haul.blasts:
                    counts["rival_escapes_on_threat_haul"] += len(haul.escaped)
            rows.append(
                {
                    "own": own,
                    "margin": own - float(np.mean(rivals)),
                    "win": float(result.winner == seat),
                    "top": float(own == max(result.scores)),
                }
            )
    finally:
        MineEnv.begin_blast, MineEnv._cancel_blast = original_begin, original_cancel

    margins = [row["margin"] for row in rows]
    low, high = bootstrap_ci(margins)
    threat_decisions = counts["rival_threat_decisions"]
    return {
        "arm": label,
        "games": games,
        "score": float(np.mean([row["own"] for row in rows])),
        "margin": float(np.mean(margins)),
        "margin_ci": (low, high),
        "win_rate": float(np.mean([row["win"] for row in rows])),
        "top_rate": float(np.mean([row["top"] for row in rows])),
        "announced_per_game": counts["announced"] / games,
        "cancelled_per_game": counts["cancelled"] / games,
        "fire_rate": (
            1 - counts["cancelled"] / counts["announced"]
            if counts["announced"]
            else 0.0
        ),
        "blast_hauls": counts["blast_hauls"],
        "dragons_woken_per_game": counts["dragons_woken_by_bluffer"] / games,
        "rival_threat_decisions_per_game": threat_decisions / games,
        "rival_cage_answer_rate": (
            counts["rival_cage_answers"] / threat_decisions if threat_decisions else 0.0
        ),
        "margins": margins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/selfplay4.npz")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=811_000)
    parser.add_argument("--players", type=int, default=4, choices=(3, 4, 5))
    parser.add_argument("--no-bluff", action="store_true", help="run the net arm only")
    parser.add_argument("--out", default="reports/bluff.md")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    net = PolicyNet.load(args.checkpoint)
    labels = ["net"] if args.no_bluff else [
        "net",
        "net, no announcements",
        "quiet (control)",
        "bluff, free",
        "bluff, forced",
    ]
    results = [
        play_arm(
            net,
            games=args.games,
            seed=args.seed,
            label=label,
            players=args.players,
        )
        for label in labels
    ]
    print(json.dumps([{k: v for k, v in r.items() if k != "margins"} for r in results], indent=2))

    lines = [
        "# What is a bluff worth? (announce a blast, then abort it)\n\n",
        f"One seat against three unchanged `{Path(args.checkpoint).stem}` nets, "
        f"{args.games} games, seats rotated, seed {args.seed}.  Both arms run the "
        "*same* movement policy; the only difference is whether it announces.\n\n",
        "| seat | score | margin [95% CI] | win % | announcements/game | share that fired "
        "| dragons it woke/game | rivals' threatened decisions/game | answered with a cage |\n",
        "| :-- | --: | :-- | --: | --: | --: | --: | --: | --: |\n",
    ]
    for row in results:
        low, high = row["margin_ci"]
        lines.append(
            f"| {row['arm']} | "
            f"{row['score']:+.2f} | {row['margin']:+.2f} [{low:+.2f}, {high:+.2f}] | "
            f"{100 * row['win_rate']:.1f}% | {row['announced_per_game']:.2f} | "
            f"{100 * row['fire_rate']:.0f}% | {row['dragons_woken_per_game']:.2f} | "
            f"{row['rival_threat_decisions_per_game']:.2f} | "
            f"{100 * row['rival_cage_answer_rate']:.0f}% |\n"
        )
    by_arm = {row["arm"]: row for row in results}
    for treatment, reference in (
        ("bluff, free", "quiet (control)"),
        ("bluff, forced", "quiet (control)"),
        ("bluff, free", "net"),
        ("net, no announcements", "net"),
    ):
        if treatment not in by_arm or reference not in by_arm:
            continue
        paired = [
            a - b
            for a, b in zip(by_arm[treatment]["margins"], by_arm[reference]["margins"])
        ]
        low, high = bootstrap_ci(paired)
        lines.append(
            f"\n- `{treatment}` minus `{reference}`: **{np.mean(paired):+.2f} points** "
            f"[{low:+.2f}, {high:+.2f}], paired over {len(paired)} games.\n"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines))
    if args.json:
        Path(args.json).write_text(
            json.dumps([{k: v for k, v in r.items() if k != "margins"} for r in results], indent=2)
        )
    print("".join(lines))


if __name__ == "__main__":
    main()
