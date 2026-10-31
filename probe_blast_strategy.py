"""Is "blast when a dragon is known to be waiting for you" actually good?

The proposal: a player holding two dynamite receives two extra cards from the
blast; if the front row of its own column holds a known dragon, sitting still is
losing anyway, so it should blast. And when choosing the keepers, taking cards
from a backside other than that known dragon's should cut the risk of waking it.

This measures the strategy instead of arguing about it: the rule bot's movement
in every seat, with seat 1 varying only in its blast trigger and, in the second
table, in how it picks the two keepers. Committed blasts pass for the rest of
the planning phase, because any other move would abort the blast.

    python probe_blast_strategy.py --games 300
"""
from __future__ import annotations

import argparse
import statistics

import numpy as np

from agents import (
    ActionChoice,
    _card_actions,
    make_scripted_agent,
    owns_dynamite_pair,
    preferred_blast_choice,
    should_blast,
    visible_dragon_ahead,
)
from game_env import LORRY, MAX_COLUMNS, PASS, PIT_CAGE
from rule_policy import deterministic_policy
from selfplay import play_game


class CommittedBlaster:
    """Rule movement, a configurable blast trigger, and a committed blast."""

    def __init__(self, blast: str, choice: str = "risk", seed: int = 0):
        self.blast = blast
        self.choice = choice
        self.rng = np.random.default_rng(seed)
        self.rule = make_scripted_agent("rule", seed=seed)
        self.gem_gaps: list[int] = []

    def decide(self, ctx):
        if ctx.own_blast_pending and PASS in ctx.legal_actions:
            return ActionChoice(PASS)
        action = int(deterministic_policy(ctx))
        announce = False
        if action < PIT_CAGE and should_blast(self.blast, ctx):
            announce = True
            if ctx.opponents:
                self.gem_gaps.append(
                    ctx.own_score - max(opponent.score for opponent in ctx.opponents)
                )
        return ActionChoice(action, announce)

    def blast_choice(self, cards, ctx):
        if self.choice == "random":
            return int(self.rng.integers(len(cards)))
        if self.choice in ("avoid_known", "prefer_known"):
            known = visible_dragon_ahead(ctx)
            if known is None:
                return preferred_blast_choice(cards, ctx)
            backside = known[1]
            wanted = [
                index for index, (_front, back) in enumerate(cards)
                if (back != backside) == (self.choice == "avoid_known")
            ]
            if wanted:
                return wanted[0] if self.choice == "avoid_known" else wanted[-1]
            return 0
        return preferred_blast_choice(cards, ctx)

    def wants_blast(self, ctx):
        return False  # decide() already sets the flag

    def tie_break(self, ctx):
        return self.rule.tie_break(ctx)


def play_arm(blast, choice, games, players, seed0=0):
    margins, blasts, woke = [], 0, 0
    gaps: list[int] = []
    per_game_scores = []
    for g in range(games):
        seat0 = CommittedBlaster(blast, choice, seed=seed0 + g)
        agents = [seat0] + [
            make_scripted_agent("rule", seed=7000 + 13 * g + i) for i in range(1, players)
        ]
        result = play_game(agents, seed=seed0 + g, blast_choice_sees_fronts=False)
        others = [s for i, s in enumerate(result.scores) if i != 0]
        margins.append(result.scores[0] - sum(others) / len(others))
        per_game_scores.append(result.scores[0])
        blasts += sum(len(h.blasts) for h in result.hauls)
        gaps.extend(seat0.gem_gaps)
        woke += sum(1 for h in result.hauls if h.outcome == "blast_dragon")
    mean = statistics.mean(margins)
    se = statistics.pstdev(margins) / len(margins) ** 0.5
    return {
        "margin": mean,
        "se": se,
        "blasts": blasts / games,
        "gem_gap": statistics.mean(gaps) if gaps else 0.0,
        "behind_share": (
            sum(1 for gap in gaps if gap < 0) / len(gaps) if gaps else 0.0
        ),
        "woke": woke,
        "fired": blasts,
        "scores": per_game_scores,
        "margins": margins,
    }


class ExitSpy:
    """Rule movement, but counts the ways out of a "known dragon ahead" state."""

    def __init__(self, seed: int):
        self.rule = make_scripted_agent("rule", seed=seed)
        self.states = 0
        self.pair = 0
        self.exits = {"leave on the lorry": 0, "dig another column": 0,
                      "retreat to a pit cage": 0, "pass": 0}

    def decide(self, ctx):
        if visible_dragon_ahead(ctx) is not None:
            self.states += 1
            self.pair += int(owns_dynamite_pair(ctx))
            self.exits["leave on the lorry"] += int(LORRY in ctx.legal_actions)
            self.exits["dig another column"] += int(
                any(a % MAX_COLUMNS != ctx.player_index for a in _card_actions(ctx))
            )
            self.exits["retreat to a pit cage"] += int(PIT_CAGE in ctx.legal_actions)
            self.exits["pass"] += int(PASS in ctx.legal_actions)
        return self.rule.decide(ctx)

    def blast_choice(self, cards, ctx):
        return self.rule.blast_choice(cards, ctx)

    def tie_break(self, ctx):
        return self.rule.tie_break(ctx)


def report_exits(games: int, players: int, seed0: int = 0) -> None:
    """Is "staying put loses anyway" true? Count the alternatives, not the excuse."""
    states = pair = 0
    exits = {"leave on the lorry": 0, "dig another column": 0,
             "retreat to a pit cage": 0, "pass": 0}
    for g in range(games):
        spies = [ExitSpy(seed0 + 4 * g + i) for i in range(players)]
        play_game(spies, seed=seed0 + g)
        for spy in spies:
            states += spy.states
            pair += spy.pair
            for name, count in spy.exits.items():
                exits[name] += count
    print("\nAt a decision where the seat can see a dragon still ahead in its own "
          "column")
    print(f"({states} decisions, {pair} of them with a dynamite pair):")
    for name, count in exits.items():
        print(f"- {name}: {100 * count / max(1, states):.1f}% of them")


def paired_difference(a, b):
    diffs = [x - y for x, y in zip(a["margins"], b["margins"])]
    mean = statistics.mean(diffs)
    se = statistics.pstdev(diffs) / len(diffs) ** 0.5
    return mean, se


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-exits", action="store_true")
    args = parser.parse_args()

    print(f"{args.players} players, {args.games} games per arm, seat 1 varies, "
          f"the others are `rule` bots.\n")
    print("Trigger (keepers chosen by backside risk):")
    print("| trigger | blasts/game | gems behind at the blast | woke the dragon | "
          "seat margin [se] | vs never (paired) |")
    print("|:--|--:|--:|--:|:--|:--|")
    arms = {}
    for label, mode in (
        ("never (control)", "never"),
        ("always", "always"),
        ("trap: known dragon ahead", "trap"),
        ("behind on gems", "behind"),
    ):
        arms[mode] = play_arm(mode, "risk", args.games, args.players, args.seed)
        a = arms[mode]
        delta, delta_se = paired_difference(a, arms["never"])
        woke = f"{a['woke']}/{a['fired']}" if a["fired"] else "—"
        gap = (
            f"{a['gem_gap']:+.1f} ({100 * a['behind_share']:.0f}% behind)"
            if a["blasts"]
            else "—"
        )
        print(
            f"| {label} | {a['blasts']:.2f} | {gap} | {woke} | "
            f"{a['margin']:+.2f} [{a['se']:.2f}] | {delta:+.2f} [{delta_se:.2f}] |"
        )

    print("\nKeepers, with the trap trigger:")
    print("| choice | blasts/game | woke the dragon | seat margin [se] |")
    print("|:--|--:|--:|:--|")
    for label, choice in (
        ("backside risk (default)", "risk"),
        ("different backside than the known dragon", "avoid_known"),
        ("same backside as the known dragon", "prefer_known"),
        ("at random", "random"),
    ):
        a = play_arm("trap", choice, args.games, args.players, args.seed)
        woke = f"{a['woke']}/{a['fired']} ({100 * a['woke'] / max(1, a['fired']):.0f}%)"
        print(f"| {label} | {a['blasts']:.2f} | {woke} | {a['margin']:+.2f} [{a['se']:.2f}] |")

    if not args.skip_exits:
        report_exits(args.games, args.players, args.seed)


if __name__ == "__main__":
    main()
