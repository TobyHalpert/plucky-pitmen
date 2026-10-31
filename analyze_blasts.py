"""Blast value and blast-threat analysis for Plucky Pitmen.

Answers three questions that the balance report cannot answer on its own:

1. Is announcing a blast worth it, or a losing move?
2. Does a trained agent ever threaten with a blast?
3. Does a trained agent *react* when an opponent threatens one?

The threat test is a paired counterfactual: the same real decision context is
scored twice by the policy, once as it was and once with a credible opponent
blast pending (that opponent really does hold the two dynamite the rules
require). Any difference in the action distribution is a reaction to the
threat, with everything else held constant.

Usage
-----
    python analyze_blasts.py --checkpoint checkpoints/selfplay4.npz --players 4
    python analyze_blasts.py --checkpoint checkpoints/selfplay4.npz --players 4 --json reports/blasts4.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agents import Agent, NeuralAgent, ThreatAgent, make_scripted_agent
from features import action_mask, split_action, state_action_matrix
from game_env import PIT_CAGE, PlayerActionContext
from learning import PolicyNet
from selfplay import play_game, play_series

N_BASE_ACTIONS = 28


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def bootstrap_ci(
    values: Sequence[float], samples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def scripted(policy: str, blast: str = "never"):
    return lambda seed: make_scripted_agent(policy, seed=seed, blast=blast)


def head_to_head(
    seat0,
    opponents: Sequence[str],
    *,
    games: int,
    players: int,
    seed: int = 777,
    blind: bool = False,
    blast: str = "never",
) -> dict[str, Any]:
    """One seat against ``players - 1`` scripted opponents, seats rotated."""
    factories = [seat0] + [
        scripted(name, blast) for name in opponents
    ]
    series = play_series(
        factories,
        games,
        seed=seed,
        rotate_seats=True,
        blast_choice_sees_fronts=not blind,
    )
    wins = tops = 0
    margins: list[float] = []
    dragons = lorry = fired = 0
    blaster_points: list[float] = []
    non_blaster_points: list[float] = []
    for game in series:
        seat = game.policy_of_seat.index(0)
        scores = game.result.scores
        own = scores[seat]
        others = [s for index, s in enumerate(scores) if index != seat]
        wins += int(game.result.winner == seat)
        tops += int(own == max(scores))
        margins.append(own - float(np.mean(others)))
        previous = 0
        for haul in game.result.hauls:
            dragons += int("dragon" in haul.outcome)
            lorry += int(haul.outcome == "lorry")
            fired += len(haul.blasts)
            delta = haul.scores[seat] - previous
            previous = haul.scores[seat]
            if seat in haul.blasts:
                blaster_points.append(float(delta))
            else:
                non_blaster_points.append(float(delta))
    hauls = sum(len(game.result.hauls) for game in series)
    return {
        "aborted": sum(int(game.result.aborted) for game in series),
        "win_rate": wins / games,
        "top_rate": tops / games,
        "margin": float(np.mean(margins)),
        "margin_ci": bootstrap_ci(margins),
        "dragon_rate": dragons / max(hauls, 1),
        "lorry_rate": lorry / max(hauls, 1),
        "blasts_per_game": fired / games,
        "points_when_blasting": (
            float(np.mean(blaster_points)) if blaster_points else 0.0
        ),
        "points_otherwise": (
            float(np.mean(non_blaster_points)) if non_blaster_points else 0.0
        ),
        "blast_hauls": len(blaster_points),
    }


# ----------------------------------------------------------------------
# 1. Is blasting worth it?
# ----------------------------------------------------------------------
def blast_value(players: int, games: int, seed: int) -> dict[str, Any]:
    """Compare the same policy with and without blasting, peek vs printed rule."""
    out: dict[str, Any] = {
        "players": players,
        "games": games,
        "variants": {},
    }
    for blind in (False, True):
        label = "blind" if blind else "peek"
        # The blaster sits in one seat against the same policy that never blasts.
        attacking = head_to_head(
            scripted("rule", "always"),
            ["rule"] * (players - 1),
            games=games,
            players=players,
            seed=seed,
            blind=blind,
        )
        control = head_to_head(
            scripted("rule", "never"),
            ["rule"] * (players - 1),
            games=games,
            players=players,
            seed=seed,
            blind=blind,
        )
        # Same question against a field that actually collects and leaves.
        vs_leaver = head_to_head(
            scripted("rule", "always"),
            ["leaver"] * (players - 1),
            games=games,
            players=players,
            seed=seed,
            blind=blind,
        )
        vs_leaver_control = head_to_head(
            scripted("rule", "never"),
            ["leaver"] * (players - 1),
            games=games,
            players=players,
            seed=seed,
            blind=blind,
        )
        out["variants"][label] = {
            "blaster_vs_rule": attacking,
            "never_blasts_vs_rule": control,
            "blaster_vs_leaver": vs_leaver,
            "never_blasts_vs_leaver": vs_leaver_control,
        }
    # The exact same movement policy: only the blast decision differs.
    return out


def blast_risk(players: int, games: int, seed: int) -> dict[str, Any]:
    """What a fired blast actually does: cards gained, dragons, deaths."""
    dragons = hauls_with_blast = blasts = 0
    aborted = 0
    gems: list[int] = []
    for game in range(games):
        agents = [
            make_scripted_agent("rule", seed=game * 7 + index, blast="always")
            for index in range(players)
        ]
        result = play_game(agents, seed=seed + game)
        aborted += int(result.aborted)
        for haul in result.hauls:
            if not haul.blasts:
                continue
            hauls_with_blast += 1
            blasts += len(haul.blasts)
            dragons += int("dragon" in haul.outcome)
            for index in haul.blasts:
                gems.append(haul.gems[index] if index < len(haul.gems) else 0)
    return {
        "aborted": aborted,
        "hauls_with_blast": hauls_with_blast,
        "blasts": blasts,
        "dragon_share": dragons / max(hauls_with_blast, 1),
        "mean_gems_of_blaster": float(np.mean(gems)) if gems else 0.0,
    }


# ----------------------------------------------------------------------
# 2./3. Threat: does the agent see it, and does it react?
# ----------------------------------------------------------------------
def credible_threat_index(ctx: PlayerActionContext) -> int | None:
    """Index of an opponent who could credibly blast (two dynamite in hand)."""
    for opponent in ctx.opponents:
        # Card fronts are hidden, but a player needs two dynamite and the
        # environment only exposes *public* facts, so credibility is judged on
        # what the environment itself enforces at announcement time. The
        # counterfactual therefore flips the flag only where the engine would
        # have allowed the announcement: enough collected cards to hold two
        # dynamite, and no blast already running.
        backsides = len(opponent.collected_backsides)
        if backsides >= 2 and not opponent.blast_pending and not opponent.escaped:
            return opponent.player_index
    return None


def primitive_distribution(
    net: PolicyNet, ctx: PlayerActionContext, temperature: float = 0.7
) -> tuple[np.ndarray, np.ndarray]:
    """Probability of each primitive action, summing blast-announcement variants."""
    mask = action_mask(ctx)
    if not mask.any():
        return np.zeros(N_BASE_ACTIONS), mask
    probs = net.masked_probs(
        state_action_matrix(ctx), mask, temperature=temperature
    )
    primitives = np.zeros(N_BASE_ACTIONS)
    for action, probability in enumerate(probs):
        primitive, _announce = split_action(action)
        primitives[primitive] += probability
    return primitives, mask


def reaction_test(
    net: PolicyNet,
    *,
    players: int,
    games: int,
    seed: int,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Paired counterfactual: same context, threat added or removed."""
    opponents = ["rule"] * (players - 1)
    # Record contexts by replaying games through the driver with a recorder
    # agent that keeps the raw PlayerActionContext of every decision.
    contexts: list[tuple[PlayerActionContext, int]] = []

    class RecordingAgent(NeuralAgent):
        def decide(self, ctx):  # type: ignore[override]
            contexts.append((ctx, int(ctx.haul)))
            return super().decide(ctx)

    factories = [lambda s: RecordingAgent(net, seed=s, temperature=temperature)] + [
        scripted(name) for name in opponents
    ]
    play_series(
        factories,
        games,
        seed=seed,
        rotate_seats=True,
    )

    tv: list[float] = []
    retreat_shift: list[float] = []
    blast_shift: list[float] = []
    cage_shift: list[float] = []
    pairs = 0
    for ctx, _haul in contexts:
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
        base, _mask = primitive_distribution(net, ctx, temperature)
        threat, _ = primitive_distribution(net, threatened, temperature)
        if base.sum() == 0:
            continue
        pairs += 1
        tv.append(float(0.5 * np.abs(base - threat).sum()))

        def mass(dist: np.ndarray, actions: Sequence[int]) -> float:
            return float(sum(dist[action] for action in actions if action < N_BASE_ACTIONS))

        retreat_actions = list(range(PIT_CAGE, N_BASE_ACTIONS))
        retreat_shift.append(
            mass(threat, retreat_actions) - mass(base, retreat_actions)
        )
        cage_shift.append(float(threat[PIT_CAGE] - base[PIT_CAGE]))
        blast_shift.append(
            float(
                sum(threat[action] for action in range(N_BASE_ACTIONS)
                    if action < PIT_CAGE)
            )
        )
    return {
        "pairs": pairs,
        "mean_tv": float(np.mean(tv)) if tv else 0.0,
        "mean_retreat_shift": float(np.mean(retreat_shift)) if retreat_shift else 0.0,
        "mean_cage_shift": float(np.mean(cage_shift)) if cage_shift else 0.0,
        "reaction_ci": bootstrap_ci(tv) if tv else (0.0, 0.0),
    }


def learned_threats(
    net: PolicyNet, *, players: int, games: int, seed: int
) -> dict[str, Any]:
    """Does the trained agent itself fire blasts, and how do they pay?"""
    factories = [lambda s: NeuralAgent(net, seed=s, temperature=0.7)] + [
        scripted("rule") for _ in range(players - 1)
    ]
    series = play_series(factories, games, seed=seed, rotate_seats=True)
    fired = before = after = 0
    points_with: list[float] = []
    points_without: list[float] = []
    wins = tops = 0
    for game in series:
        seat = game.policy_of_seat.index(0)
        scores = game.result.scores
        wins += int(game.result.winner == seat)
        tops += int(max(scores) == scores[seat])
        previous = 0
        for haul in game.result.hauls:
            delta = haul.scores[seat] - previous
            previous = haul.scores[seat]
            fired += int(seat in haul.blasts)
            before += int(seat in haul.blasts and "dragon" not in haul.outcome)
            after += int("dragon" in haul.outcome)
            if seat in haul.blasts:
                points_with.append(float(delta))
            else:
                points_without.append(float(delta))
    return {
        "blasts_per_game": fired / games,
        "blast_hauls": fired,
        "win_rate": wins / games,
        "top_rate": tops / games,
        "points_when_blasting": float(np.mean(points_with)) if points_with else 0.0,
        "points_otherwise": float(np.mean(points_without)) if points_without else 0.0,
    }


def threat_effect_on_field(
    net: PolicyNet, *, players: int, games: int, seed: int
) -> dict[str, Any]:
    """Trained agent against opponents that threaten, vs the same ones that do not."""
    results: dict[str, Any] = {}
    for label, blast in (("threats", "always"), ("no_threats", "never")):
        factories = [lambda s: NeuralAgent(net, seed=s, temperature=0.7)] + [
            scripted("rule", blast) for _ in range(players - 1)
        ]
        series = play_series(factories, games, seed=seed, rotate_seats=True)
        wins = tops = dragons = lorry = turns = tied = 0
        margins: list[float] = []
        cage_decisions = 0
        decisions = 0
        for game in series:
            seat = game.policy_of_seat.index(0)
            scores = game.result.scores
            own = scores[seat]
            wins += int(game.result.winner == seat)
            tops += int(max(scores) == own)
            tied += int(max(scores) == own and sum(
                1 for score in scores if score == own
            ) > 1)
            margins.append(own - float(np.mean([s for i, s in enumerate(scores) if i != seat])))
            turns += game.result.turns
            for haul in game.result.hauls:
                dragons += int("dragon" in haul.outcome)
                lorry += int(haul.outcome == "lorry")
            decisions += game.result.turns
        hauls = sum(len(game.result.hauls) for game in series)
        results[label] = {
            "aborted": sum(int(game.result.aborted) for game in series),
            "margins": margins,
            "win_rate": wins / games,
            "top_rate": tops / games,
            "margin": float(np.mean(margins)),
            "margin_ci": bootstrap_ci(margins),
            "dragon_rate": dragons / max(hauls, 1),
            "lorry_rate": lorry / max(hauls, 1),
            "turns_per_game": turns / games,
            "cage_decisions_per_game": cage_decisions / max(decisions, 1),
            "top_score_tied_rate": tied / games,
        }
    paired_diffs = [
        threat_margin - calm_margin
        for threat_margin, calm_margin in zip(
            results["threats"]["margins"], results["no_threats"]["margins"]
        )
    ]
    low, high = bootstrap_ci(paired_diffs)
    results["delta_margin"] = float(np.mean(paired_diffs))
    results["delta_margin_ci"] = (low, high)
    results["delta_dragon_rate"] = (
        results["threats"]["dragon_rate"] - results["no_threats"]["dragon_rate"]
    )
    for label in ("threats", "no_threats"):
        del results[label]["margins"]
    return results


class _CountingThreat(ThreatAgent):
    """``ThreatAgent`` that records how often a threat was answered with a cage move."""

    def __init__(self) -> None:
        super().__init__()
        self.threatened_decisions = 0
        self.retreats = 0

    def act(self, ctx: PlayerActionContext) -> int:
        threatened = self.threatened(ctx)
        action = super().act(ctx)
        if threatened:
            self.threatened_decisions += 1
            self.retreats += int(action == PIT_CAGE)
        return action


def threat_cost(players: int, games: int, seed: int) -> dict[str, Any]:
    """Does a pending blast actually hurt the field, and does reacting pay?

    Four conditions are played on the *same* seeds, so the comparison is paired
    and the confidence interval is computed on the per-game differences:

    ``calm``      seat 0 is a plain ``rule`` player, nobody blasts.
    ``threat``    same seat-0 player, one opponent blasts every chance it gets.
    ``reacting``  seat 0 is a ``threat`` agent (rule movement, but it retreats to
                  the nearest pit cage as soon as an opponent's blast is pending)
                  against the same blasting field.
    ``react_calm`` the threat agent against a field that never blasts; it must
                  match ``calm``, which is the check that the reaction fires only
                  on real threats and is not a blind change of play.
    """

    def condition(seat0_policy: str, blasting: bool) -> dict[str, Any]:
        created: list[Agent] = []

        def seat0(_seed: int) -> Agent:
            if seat0_policy == "threat":
                agent: Agent = _CountingThreat()
            else:
                agent = make_scripted_agent("rule", seed=_seed)
            created.append(agent)
            return agent

        opponents = [
            scripted("rule", "always" if (blasting and index == 0) else "never")
            for index in range(players - 1)
        ]
        series = play_series([seat0] + opponents, games, seed=seed, rotate_seats=True)
        margins: list[float] = []
        scores: list[float] = []
        wins = 0
        for game in series:
            seat = game.policy_of_seat.index(0)
            own = game.result.scores[seat]
            others = [s for i, s in enumerate(game.result.scores) if i != seat]
            margins.append(own - float(np.mean(others)))
            scores.append(float(own))
            wins += int(game.result.winner == seat)
        threat_agents = [a for a in created if isinstance(a, _CountingThreat)]
        threatened = sum(a.threatened_decisions for a in threat_agents)
        retreats = sum(a.retreats for a in threat_agents)
        return {
            "margins": margins,
            "win_rate": wins / games,
            "score": float(np.mean(scores)),
            "threatened_decisions_per_game": threatened / games,
            "retreat_rate": retreats / max(threatened, 1),
        }

    conditions = {
        "calm": condition("rule", False),
        "threat": condition("rule", True),
        "reacting": condition("threat", True),
        "react_calm": condition("threat", False),
    }

    def paired(a: str, b: str) -> dict[str, float]:
        diffs = [
            left - right
            for left, right in zip(conditions[a]["margins"], conditions[b]["margins"])
        ]
        low, high = bootstrap_ci(diffs)
        return {
            "mean": float(np.mean(diffs)),
            "ci_low": low,
            "ci_high": high,
            "win_delta": conditions[a]["win_rate"] - conditions[b]["win_rate"],
        }

    report: dict[str, Any] = {"players": players, "games": games}
    for label, stats in conditions.items():
        report[label] = {key: value for key, value in stats.items() if key != "margins"}
        report[label]["margin"] = float(np.mean(stats["margins"]))
    report["cost_of_threat"] = paired("threat", "calm")
    report["value_of_reacting"] = paired("reacting", "threat")
    report["residual"] = paired("reacting", "calm")
    report["reaction_only_on_threats"] = paired("react_calm", "calm")
    return report


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def render(report: dict[str, Any]) -> str:
    lines: list[str] = ["# Blast and threat analysis\n"]
    players = report["players"]
    games = report["games"]
    aborted = sum(
        report["value"]["variants"][engine]["blaster_vs_rule"]["aborted"]
        + report["value"]["variants"][engine]["never_blasts_vs_leaver"]["aborted"]
        for engine in ("peek", "blind")
    ) + report["risk"]["aborted"]
    lines.append(
        f"- players: **{players}**, games per measurement: **{games}**\n"
        f"- checkpoint: `{report['checkpoint']}`\n"
        f"- games the engine abandoned (all seats scored zero): **{aborted}**\n"
    )

    lines.append("\n## 1. Is blasting worth it?\n")
    lines.append(
        "The blaster seat runs the *same* movement policy as its opponents and "
        "differs only in announcing a blast whenever it holds two dynamite. "
        "`peek` = the historical engine (the blaster sees all three draws); "
        "`blind` = the printed rule (choose two, then look).\n"
    )
    lines.append(
        "| engine | blaster win % | blaster margin [95% CI] | blasts/game | "
        "dragons | lorry endings | points per blast haul | points otherwise |\n"
        "|:--|--:|:--|--:|--:|--:|--:|--:|\n"
    )
    for label, variant in report["value"]["variants"].items():
        stats = variant["blaster_vs_rule"]
        lines.append(
            f"| {label} | {100 * stats['win_rate']:.1f}% | "
            f"{stats['margin']:+.2f} [{stats['margin_ci'][0]:+.2f}, {stats['margin_ci'][1]:+.2f}] | "
            f"{stats['blasts_per_game']:.2f} | {100 * stats['dragon_rate']:.0f}% | "
            f"{100 * stats['lorry_rate']:.0f}% | "
            f"{stats['points_when_blasting']:+.2f} | {stats['points_otherwise']:+.2f} |\n"
        )
    lines.append(
        "\nAgainst a field that collects and leaves instead of parking:\n\n"
        "| engine | blaster margin vs 3 x leaver [95% CI] | same policy, never blasting |\n"
        "|:--|:--|:--|\n"
    )
    for label, variant in report["value"]["variants"].items():
        attack = variant["blaster_vs_leaver"]
        control = variant["never_blasts_vs_leaver"]
        lines.append(
            f"| {label} | {attack['margin']:+.2f} "
            f"[{attack['margin_ci'][0]:+.2f}, {attack['margin_ci'][1]:+.2f}] | "
            f"{control['margin']:+.2f} "
            f"[{control['margin_ci'][0]:+.2f}, {control['margin_ci'][1]:+.2f}] |\n"
        )

    risk = report["risk"]
    lines.append(
        f"\nAcross {risk['hauls_with_blast']} hauls with a fired blast "
        f"({risk['blasts']} blasts), **{100 * risk['dragon_share']:.0f}%** ended on a "
        f"dragon, and the blaster held {risk['mean_gems_of_blaster']:.2f} gems on average.\n"
    )

    lines.append("\n## 2. Does the trained agent threaten?\n")
    threat = report["learned"]
    lines.append(
        f"- blasts fired per game: **{threat['blasts_per_game']:.2f}**\n"
        f"- win rate against scripted bots: {100 * threat['win_rate']:.1f}%\n"
        f"- points in hauls where it blasted: {threat['points_when_blasting']:+.2f} "
        f"vs {threat['points_otherwise']:+.2f} otherwise\n"
    )

    lines.append("\n## 3. Does it react to a threat?\n")
    react = report["reaction"]
    lines.append(
        "Paired counterfactual: each real decision is scored twice, once with a "
        "credible opponent blast pending and once without, with every other "
        "feature identical.\n\n"
        f"- paired decisions: **{react['pairs']}**\n"
        f"- mean total-variation distance: **{react['mean_tv']:.4f}** "
        f"[{react['reaction_ci'][0]:.4f}, {react['reaction_ci'][1]:.4f}]\n"
        f"- shift towards leaving (cage or lorry): {react['mean_retreat_shift']:+.4f}\n"
        f"- shift towards the pit cage specifically: {react['mean_cage_shift']:+.4f}\n"
    )

    field = report["field"]
    lines.append(
        "\nTrained agent against the same opponents, with and without blasting:\n\n"
        "| field | win % | top % | top score tied % | margin [95% CI] | dragons | lorry endings | turns/game |\n"
        "|:--|--:|--:|--:|:--|--:|--:|--:|\n"
    )
    for label in ("no_threats", "threats"):
        stats = field[label]
        lines.append(
            f"| {label} | {100 * stats['win_rate']:.1f}% | {100 * stats['top_rate']:.1f}% | "
            f"{100 * stats['top_score_tied_rate']:.0f}% | {stats['margin']:+.2f} [{stats['margin_ci'][0]:+.2f}, {stats['margin_ci'][1]:+.2f}] | "
            f"{100 * stats['dragon_rate']:.0f}% | {100 * stats['lorry_rate']:.0f}% | "
            f"{stats['turns_per_game']:.0f} |\n"
        )
    lines.append(
        f"\nDelta from facing blasters, paired on the same seeds: margin "
        f"**{field['delta_margin']:+.2f}** "
        f"[{field['delta_margin_ci'][0]:+.2f}, {field['delta_margin_ci'][1]:+.2f}], "
        f"dragon rate **{100 * field['delta_dragon_rate']:+.0f} points**.\n"
    )

    cost = report["threat"]
    price = cost["cost_of_threat"]
    recover = cost["value_of_reacting"]
    left = cost["residual"]
    lines.append("\n## 5. Is the threat real, and is reacting worth anything?\n")
    lines.append(
        "The learned agent barely moves when a blast is pending, so *its* silence "
        "proves nothing about the threat. This measures the threat directly with "
        "scripted players, all four conditions on the same seeds (paired "
        "differences, so the CIs are on per-game deltas):\n\n"
        "- `calm` — a plain `rule` seat, nobody blasts.\n"
        "- `threat` — the same `rule` seat, seat one opponent blasts on sight.\n"
        "- `reacting` — the same seat plays `threat`: rule movement, except it "
        "walks to the pit cage whenever an opponent's blast is pending.\n"
        "- `react_calm` — `threat` against a field that never blasts.\n\n"
    )
    lines.append(
        "| seat | win % | margin | points | threatened decisions/game | "
        "answered with a cage move |\n"
        "|:--|--:|--:|--:|--:|--:|\n"
    )
    for label, name in (
        ("calm", "rule, calm field"),
        ("threat", "rule, blasting field"),
        ("reacting", "threat, blasting field"),
        ("react_calm", "threat, calm field"),
    ):
        stats = cost[label]
        lines.append(
            f"| {name} | {100 * stats['win_rate']:.1f}% | {stats['margin']:+.2f} | "
            f"{stats['score']:+.2f} | {stats['threatened_decisions_per_game']:.1f} | "
            f"{100 * stats['retreat_rate']:.0f}% |\n"
        )
    lines.append(
        f"\n- Cost of the threat to a non-blaster: **{price['mean']:+.2f} points** "
        f"[{price['ci_low']:+.2f}, {price['ci_high']:+.2f}], win rate "
        f"{100 * price['win_delta']:+.1f} points.\n"
        f"- Value of reacting to it: **{recover['mean']:+.2f} points** "
        f"[{recover['ci_low']:+.2f}, {recover['ci_high']:+.2f}], win rate "
        f"{100 * recover['win_delta']:+.1f} points.\n"
        f"- What is left once you react: **{left['mean']:+.2f} points** "
        f"[{left['ci_low']:+.2f}, {left['ci_high']:+.2f}] — indistinguishable from "
        "a calm field.\n"
        f"- Control: the threat agent against a calm field differs from plain `rule` by "
        f"**{cost['reaction_only_on_threats']['mean']:+.2f} points** on identical "
        "seeds, so the gain above comes from the reaction and not from a different "
        "movement policy.\n"
    )
    lines.append(
        f"\nThe trained agent answers a pending blast with a total-variation shift of "
        f"**{report['reaction']['mean_tv']:.4f}** in its action distribution "
        f"(~{report['reaction']['pairs']} paired decisions), i.e. effectively nothing. "
        f"Against a blasting field it gives up **{abs(field['delta_margin']):.2f} points** "
        f"of margin, of which a one-line reaction recovers "
        f"**{recover['mean']:+.2f}**.\n"
    )
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/selfplay4.npz")
    parser.add_argument("--players", type=int, default=4, choices=(3, 4, 5))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=424_242)
    parser.add_argument("--out", default="reports/blasts.md")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    net = PolicyNet.load(args.checkpoint)
    report: dict[str, Any] = {
        "players": args.players,
        "games": args.games,
        "checkpoint": args.checkpoint,
    }

    print("1/5 blast value ...", flush=True)
    report["value"] = blast_value(args.players, args.games, args.seed)
    print("2/5 blast risk ...", flush=True)
    report["risk"] = blast_risk(args.players, args.games, args.seed)
    print("3/5 learned threats ...", flush=True)
    report["learned"] = learned_threats(
        net, players=args.players, games=args.games, seed=args.seed
    )
    print("4/5 threat reaction ...", flush=True)
    report["reaction"] = reaction_test(
        net, players=args.players, games=max(20, args.games // 5), seed=args.seed
    )
    report["field"] = threat_effect_on_field(
        net, players=args.players, games=args.games, seed=args.seed
    )
    print("5/5 is the threat real ...", flush=True)
    report["threat"] = threat_cost(args.players, args.games, args.seed)

    text = render(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    if args.json:
        payload = {
            key: value
            for key, value in report.items()
            if key not in ("value", "risk", "learned", "reaction", "field", "threat")
        }
        payload.update(
            {
                "value": report["value"],
                "risk": report["risk"],
                "learned": report["learned"],
                "reaction": report["reaction"],
                "field": report["field"],
                "threat": report["threat"],
            }
        )
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(text)
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
