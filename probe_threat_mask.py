"""Why does the threat-reaction TV test say "no reaction" while the seat flinches?

BALANCE.md §6 reports a total-variation shift of 0.0065 when a credible opponent
blast is toggled on in an unchanged context, and reads it as "the learner
ignores the threat".  ``probe_threat_reaction.py`` shows the same checkpoint
answering 71% of its threatened decisions with a pit cage against a 16% control.
Both cannot be the whole story.

The reason they disagree is where the reaction lives.  ``action_mask`` treats a
pending blast as certain danger (``round_dragon_hazard`` returns 1.0), and that
is what makes an *empty-handed* cage move legal at all (``cage_banks_nothing``,
rule 4).  The threat therefore changes the **legal set**, not only the value of
the actions in it -- and a masked-softmax policy can only shift probability
among the actions both contexts allow.  This probe splits the paired decisions:

* **mask moved** - the toggle opened (usually the cage) or closed actions.  The
  net has somewhere new to put probability; this is where a reaction can appear.
* **mask identical** - the same actions were legal before and after.  A policy
  that read the threat as a *feature* would still shift here, and only here is
  the §6 test really asking the network rather than the mask.

Usage
-----
    python probe_threat_mask.py --checkpoint checkpoints/selfplay4.npz --games 200
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from agents import NeuralAgent
from analyze_blasts import (
    bootstrap_ci,
    credible_threat_index,
    primitive_distribution,
    scripted,
)
from features import N_BASE_ACTIONS, action_mask, split_action
from game_env import GEM, PIT_CAGE
from learning import PolicyNet
from selfplay import play_series


def _cage_mass(primitives: np.ndarray) -> float:
    return float(primitives[PIT_CAGE])


def mask_split(
    net: PolicyNet, *, players: int, games: int, seed: int, temperature: float = 0.7
) -> dict[str, Any]:
    opponents = ["rule"] * (players - 1)
    contexts = []

    class RecordingAgent(NeuralAgent):
        def decide(self, ctx):  # type: ignore[override]
            contexts.append(ctx)
            return super().decide(ctx)

    factories = [lambda s: RecordingAgent(net, seed=s, temperature=temperature)] + [
        scripted(name) for name in opponents
    ]
    play_series(factories, games, seed=seed, rotate_seats=True)

    groups: dict[str, dict[str, list[float]]] = {
        "mask_moved": {"tv": [], "cage": [], "added": []},
        "mask_identical": {"tv": [], "cage": []},
    }
    by_hand: dict[str, dict[str, int]] = {
        "loot": {"pairs": 0, "mask_moved": 0, "cage_picks": 0, "stay_picks": 0},
        "empty": {"pairs": 0, "mask_moved": 0, "cage_picks": 0, "stay_picks": 0},
    }
    for ctx in contexts:
        index = credible_threat_index(ctx)
        if index is None:
            continue
        opponents_list = list(ctx.opponents)
        slot = next(
            position
            for position, opponent in enumerate(opponents_list)
            if opponent.player_index == index
        )
        threatened = dataclasses.replace(
            ctx,
            opponents=tuple(
                dataclasses.replace(opponent, blast_pending=True)
                if position == slot
                else opponent
                for position, opponent in enumerate(opponents_list)
            ),
        )
        mask_base = action_mask(ctx)
        mask_threat = action_mask(threatened)
        base, _ = primitive_distribution(net, ctx, temperature)
        threat, _ = primitive_distribution(net, threatened, temperature)
        if base.sum() == 0:
            continue
        tv = float(0.5 * np.abs(base - threat).sum())
        moved = not np.array_equal(mask_base, mask_threat)
        key = "mask_moved" if moved else "mask_identical"
        groups[key]["tv"].append(tv)
        groups[key]["cage"].append(_cage_mass(threat) - _cage_mass(base))
        if moved:
            # Probability the threat put on actions that were not legal before.
            opened = np.zeros(N_BASE_ACTIONS, dtype=bool)
            for action in np.flatnonzero(mask_threat):
                primitive, _ = split_action(int(action))
                if primitive < N_BASE_ACTIONS and not mask_base[action]:
                    opened[primitive] = True
            groups[key]["added"].append(float(threat[opened].sum()))

    for group, data in groups.items():
        data["pairs"] = len(data["tv"])
        data["mean_tv"] = float(np.mean(data["tv"])) if data["tv"] else 0.0
        data["tv_ci"] = bootstrap_ci(data["tv"]) if data["tv"] else (0.0, 0.0)
        data["mean_cage_shift"] = float(np.mean(data["cage"])) if data["cage"] else 0.0
        if data.get("added"):
            data["mean_mass_on_newly_legal"] = float(np.mean(data["added"]))
        del data["tv"], data["cage"]
        data.pop("added", None)

    return {
        "groups": groups,
        "pairs": sum(group["pairs"] for group in groups.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/selfplay4.npz")
    parser.add_argument("--players", type=int, default=4, choices=(3, 4, 5))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=424_242)
    parser.add_argument("--out", default="reports/threat_mask.md")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    net = PolicyNet.load(args.checkpoint)
    report = mask_split(
        net, players=args.players, games=args.games, seed=args.seed
    )
    report["checkpoint"] = args.checkpoint
    report["games"] = args.games

    total = report["pairs"]
    lines = [
        "# Where the threat reaction lives: the mask or the policy?\n\n",
        f"`{args.checkpoint}`, {args.games} games against `rule` opponents, "
        f"{total} paired threat decisions.\n\n",
        "| pair type | pairs | share | mean TV [95% CI] | cage-probability shift | mass on newly legal actions |\n",
        "| :-- | --: | --: | :-- | --: | --: |\n",
    ]
    for name, label in (
        ("mask_moved", "mask moved"),
        ("mask_identical", "mask identical"),
    ):
        group = report["groups"][name]
        low, high = group["tv_ci"]
        extra = (
            f"{100 * group['mean_mass_on_newly_legal']:.0f}%"
            if "mean_mass_on_newly_legal" in group
            else "n/a"
        )
        lines.append(
            f"| {label} | {group['pairs']} | "
            f"{100 * group['pairs'] / total if total else 0:.0f}% | "
            f"{group['mean_tv']:.4f} [{low:.4f}, {high:.4f}] | "
            f"{group['mean_cage_shift']:+.4f} | {extra} |\n"
        )
    lines.append(
        "\nA pending blast is treated as certain danger by `round_dragon_hazard`, "
        "so the toggle often opens the pit cage for a seat with an empty hand.  "
        "Where the legal set is unchanged, a shift can only come from the network "
        "reading the feature; that is the number §6 quotes.  Where it moved, the "
        "reaction is already in the mask, and the question becomes what the "
        "network does with the new option.\n"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    print("".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
