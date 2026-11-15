"""Measure and mitigate own-column row-0 Gem displacement sequences.

The focal seat is a trained neural agent.  The control uses today's action
mask; the treatment additionally masks a row-0 home-column Gem whenever an
active rival has movement depth greater than zero and another policy action
remains. The engine's legal actions never change. Full games are paired by seed
and rotating focal seat against neural, rule, and always-pusher fields.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import features
from agents import AlwaysDisplaceAgent, NeuralAgent, RuleAgent
from game_env import GEM, MAX_COLUMNS, MineEnv, PIT_CAGE
from learning import PolicyNet
from selfplay import play_game


ACTIVE_TRACES: dict[int, dict[str, Any]] = {}


def _movement_depth(opponent: Any) -> int:
    """Maximum row an opponent can still reach, including from a pit cage."""
    position = int(opponent.position)
    if position == PIT_CAGE:
        return int(opponent.cage_index) if opponent.cage_index is not None else 0
    if position < 0 or position >= PIT_CAGE:
        return 0
    return position


def _row0_gem(ctx: Any) -> bool:
    """Whether the requesting seat can see a Gem in its row-0 home cell."""
    return bool(
        ctx.public_rows
        and len(ctx.public_rows[0]) > ctx.player_index
        and ctx.public_rows[0][ctx.player_index] is not None
        and ctx.public_rows[0][ctx.player_index][0] == GEM
    )


def _rival_state(ctx: Any) -> dict[str, Any]:
    active = [op for op in ctx.opponents if not op.escaped and not op.dead]
    deeper = [op for op in active if _movement_depth(op) > 0]
    deeper_cards = [
        op for op in active
        if 0 < int(op.position) < PIT_CAGE and op.column is not None
    ]
    unsettled_deeper = [
        op
        for op in active
        if (not op.planned or getattr(op, "displaced", False))
        and _movement_depth(op) > 0
    ]
    return {
        "active_rivals": len(active),
        "deeper_rivals": len(deeper),
        "deeper_card_rivals": len(deeper_cards),
        "unsettled_deeper_rivals": len(unsettled_deeper),
        "any_deeper_rival": bool(deeper),
        "all_active_rivals_deeper": bool(active) and len(deeper) == len(active),
        "all_active_rivals_on_deeper_cards": (
            bool(active) and len(deeper_cards) == len(active)
        ),
        "unsettled_deeper_rival": bool(unsettled_deeper),
    }


class Row0AuditAgent(NeuralAgent):
    """Record row-0 Gem offers and choices without exposing truth to the policy."""

    def __init__(self, net: Any, *, seed: int, trace: dict[str, Any], guard: bool):
        super().__init__(
            net,
            seed=seed,
            temperature=0.7,
            avoid_exposed_row0_gem=guard,
        )
        self.trace = trace

    def decide(self, ctx):
        action = int(ctx.player_index)  # row 0, home column
        gem_offered = _row0_gem(ctx) and action in ctx.legal_actions
        exposed = gem_offered and action in features.exposed_own_gem_actions(ctx)
        rival_state = _rival_state(ctx)
        guard_keeps_action = bool(
            features.action_mask(ctx, avoid_exposed_row0_gem=True)[action]
        ) if gem_offered else False
        if gem_offered:
            self.trace["offers"].append(
                {
                    "haul": int(ctx.haul),
                    "round": int(ctx.round),
                    "exposed": bool(exposed),
                    **rival_state,
                    "guard_keeps_action": guard_keeps_action,
                }
            )

        choice = super().decide(ctx)
        if gem_offered and int(choice.action) == action:
            self.trace["choices"].append(
                {
                    "game_seed": self.trace["game_seed"],
                    "seat": int(ctx.player_index),
                    "haul": int(ctx.haul),
                    "round": int(ctx.round),
                    "exposed": bool(exposed),
                    **rival_state,
                    "guard_keeps_action": guard_keeps_action,
                    "displaced_before_reveal": False,
                    "pusher": None,
                    "reveal_processed": False,
                    "gem_resolved": False,
                    "reveal_occupant": None,
                    "pusher_took_gem": False,
                    "nonowner_took_gem": False,
                    "owner_took_gem": False,
                    "resolution_status": "pending",
                }
            )
        return choice


def _install_spies():
    """Instrument the engine only during this probe; always restore afterwards."""
    original_displace = MineEnv._displace
    original_execute = MineEnv._execute_round

    def displace_spy(self, row, column, player_index):
        state = ACTIVE_TRACES.get(id(self))
        occupant = self._card_occupant(row, column)
        before = len(self.displacement_events)
        original_displace(self, row, column, player_index)
        if state is None or len(self.displacement_events) == before:
            return
        if row != 0 or column != state["seat"] or occupant != state["seat"]:
            return
        event = next(
            (
                item
                for item in reversed(state["choices"])
                if item["haul"] == self.haul
                and item["round"] == self.round
                and not item["displaced_before_reveal"]
                and not item["reveal_processed"]
            ),
            None,
        )
        if event is not None:
            event["displaced_before_reveal"] = True
            event["pusher"] = int(player_index)

    def execute_spy(self):
        state = ACTIVE_TRACES.get(id(self))
        pending = []
        if state is not None:
            for event in state["choices"]:
                if (
                    event["haul"] != self.haul
                    or event["round"] != self.round
                    or event["reveal_processed"]
                ):
                    continue
                card = self.rows[0][event["seat"]] if self.rows else None
                if card is None:
                    event["reveal_processed"] = True
                    event["resolution_status"] = "cell_already_empty"
                    continue
                if card[0] != GEM:
                    event["reveal_processed"] = True
                    event["resolution_status"] = "cell_no_longer_gem"
                    continue
                occupant = self._card_occupant(0, event["seat"])
                before_gems = (
                    self.players[occupant].gems if occupant is not None else 0
                )
                pending.append((event, occupant, before_gems))

        result = original_execute(self)
        info = result[1] if isinstance(result, tuple) and len(result) > 1 else {}
        if state is not None:
            for event, occupant, before_gems in pending:
                cleared = bool(self.rows and self.rows[0][event["seat"]] is None)
                gem_gain = (
                    self.players[occupant].gems > before_gems
                    if occupant is not None
                    else False
                )
                event["reveal_processed"] = True
                event["gem_resolved"] = bool(cleared and gem_gain)
                event["reveal_occupant"] = occupant
                event["owner_took_gem"] = bool(
                    event["gem_resolved"] and occupant == event["seat"]
                )
                event["nonowner_took_gem"] = bool(
                    event["gem_resolved"]
                    and occupant is not None
                    and occupant != event["seat"]
                )
                event["pusher_took_gem"] = bool(
                    event["gem_resolved"]
                    and event["displaced_before_reveal"]
                    and occupant == event["pusher"]
                )
                if event["gem_resolved"]:
                    event["resolution_status"] = "gem_collected"
                elif info.get("outcome") == "dragon":
                    event["resolution_status"] = "preempted_by_dragon"
                elif occupant is None:
                    event["resolution_status"] = "not_occupied_at_reveal"
                else:
                    event["resolution_status"] = "not_collected"
        return result

    MineEnv._displace = displace_spy
    MineEnv._execute_round = execute_spy
    return original_displace, original_execute


def _restore_spies(originals) -> None:
    MineEnv._displace, MineEnv._execute_round = originals
    ACTIVE_TRACES.clear()


def _make_opponent(name: str, net: PolicyNet, seed: int):
    if name == "neural":
        return NeuralAgent(net, seed=seed, temperature=0.7)
    if name == "rule":
        return RuleAgent(blast="safe")
    if name == "always-pusher":
        return AlwaysDisplaceAgent()
    raise ValueError(f"unknown opponent field: {name}")


def _play_one(
    net: PolicyNet,
    *,
    game_seed: int,
    focal_seat: int,
    opponent: str,
    guard: bool,
) -> dict[str, Any]:
    env = MineEnv(opponents=3)
    env.reset(seed=game_seed)
    trace: dict[str, Any] = {
        "game_seed": int(game_seed),
        "seat": int(focal_seat),
        "offers": [],
        "choices": [],
    }
    agents = []
    for seat in range(4):
        agent_seed = game_seed * 31 + seat
        if seat == focal_seat:
            agents.append(
                Row0AuditAgent(net, seed=agent_seed, trace=trace, guard=guard)
            )
        else:
            agents.append(_make_opponent(opponent, net, agent_seed))

    ACTIVE_TRACES[id(env)] = trace
    try:
        result = play_game(agents, seed=game_seed, env=env)
    finally:
        ACTIVE_TRACES.pop(id(env), None)

    own = int(result.scores[focal_seat])
    rivals = [score for seat, score in enumerate(result.scores) if seat != focal_seat]
    return {
        "game_seed": int(game_seed),
        "focal_seat": int(focal_seat),
        "win": bool(result.winner == focal_seat),
        "top_score": bool(own >= max(rivals)),
        "score": own,
        "rank": 1 + sum(score > own for score in rivals),
        "margin": own - float(np.mean(rivals)),
        "unfinished": bool(result.aborted),
        "offers": trace["offers"],
        "choices": trace["choices"],
    }


def evaluate_arm(
    net: PolicyNet,
    *,
    games: int,
    seed: int,
    opponent: str,
    guard: bool,
) -> list[dict[str, Any]]:
    rows = []
    for index in range(games):
        rows.append(
            _play_one(
                net,
                game_seed=seed + index,
                focal_seat=index % 4,
                opponent=opponent,
                guard=guard,
            )
        )
    return rows


def _rate(count: int, denominator: int) -> float:
    return float(count / denominator) if denominator else 0.0


def _format_rate(count: int, denominator: int) -> str:
    return "n/a" if denominator <= 0 else f"{count / denominator:.1%}"


def _format_wilson_interval(interval: list[float], denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{interval[0]:.1%}–{interval[1]:.1%}"


def _wilson_interval(count: int, denominator: int) -> list[float]:
    """Two-sided 95% Wilson score interval for a binomial proportion."""
    if denominator <= 0:
        return [0.0, 0.0]
    z = 1.96
    p = count / denominator
    z2 = z * z
    scale = 1.0 + z2 / denominator
    center = (p + z2 / (2.0 * denominator)) / scale
    half = z * math.sqrt(
        p * (1.0 - p) / denominator + z2 / (4.0 * denominator**2)
    ) / scale
    return [max(0.0, center - half), min(1.0, center + half)]


def _condition_stats(
    rows: list[dict[str, Any]], condition: str
) -> dict[str, Any]:
    offers = [
        offer
        for row in rows
        for offer in row["offers"]
        if bool(offer[condition])
    ]
    choices = [
        event
        for row in rows
        for event in row["choices"]
        if bool(event[condition])
    ]
    displaced = [event for event in choices if event["displaced_before_reveal"]]
    resolved = [event for event in choices if event["gem_resolved"]]
    displaced_resolved = [
        event
        for event in choices
        if event["displaced_before_reveal"] and event["gem_resolved"]
    ]
    pusher_took = [event for event in displaced_resolved if event["pusher_took_gem"]]
    guard_exception = [event for event in choices if event["guard_keeps_action"]]
    return {
        "offers": len(offers),
        "choices": len(choices),
        "displaced_before_reveal": len(displaced),
        "gem_resolved": len(resolved),
        "displaced_and_gem_resolved": len(displaced_resolved),
        "pusher_took_gem": len(pusher_took),
        "guard_kept_choice_due_to_no_alternative": len(guard_exception),
        "guard_would_block_choice": len(choices) - len(guard_exception),
        "choice_rate_given_offer": _rate(len(choices), len(offers)),
        "displacement_rate_given_choice": _rate(len(displaced), len(choices)),
        "displacement_rate_wilson_95_ci": _wilson_interval(
            len(displaced), len(choices)
        ),
        "gem_resolution_rate_given_choice": _rate(len(resolved), len(choices)),
        "displaced_and_resolved_rate_given_choice": _rate(
            len(displaced_resolved), len(choices)
        ),
        "displaced_and_resolved_rate_wilson_95_ci": _wilson_interval(
            len(displaced_resolved), len(choices)
        ),
        "displacement_rate_given_resolved_gem": _rate(
            len(displaced_resolved), len(resolved)
        ),
        "pusher_take_rate_given_displaced_resolved_gem": _rate(
            len(pusher_took), len(displaced_resolved)
        ),
        "pusher_take_rate_wilson_95_ci": _wilson_interval(
            len(pusher_took), len(displaced_resolved)
        ),
        "displaced_examples": [
            event for event in displaced_resolved if event["pusher_took_gem"]
        ][:10],
        "resolution_status_counts": {
            status: sum(event["resolution_status"] == status for event in choices)
            for status in sorted(
                {event["resolution_status"] for event in choices}
            )
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = {
        "games": len(rows),
        "unfinished_games": sum(bool(row["unfinished"]) for row in rows),
        "win_rate": float(np.mean([row["win"] for row in rows])) if rows else 0.0,
        "top_score_rate": float(np.mean([row["top_score"] for row in rows])) if rows else 0.0,
        "mean_score": float(np.mean([row["score"] for row in rows])) if rows else 0.0,
        "mean_rank": float(np.mean([row["rank"] for row in rows])) if rows else 0.0,
        "mean_margin": float(np.mean([row["margin"] for row in rows])) if rows else 0.0,
    }
    condition_names = (
        "any_deeper_rival",
        "all_active_rivals_deeper",
        "all_active_rivals_on_deeper_cards",
        "unsettled_deeper_rival",
    )
    condition_stats = {
        name: _condition_stats(rows, name) for name in condition_names
    }
    all_offers = [offer for row in rows for offer in row["offers"]]
    all_choices = [event for row in rows for event in row["choices"]]
    totals = {
        "row0_gem_offers": len(all_offers),
        "row0_gem_choices": len(all_choices),
        "displaced_before_reveal": sum(
            event["displaced_before_reveal"] for event in all_choices
        ),
        "gem_resolved": sum(event["gem_resolved"] for event in all_choices),
        "displaced_and_gem_resolved": sum(
            event["displaced_before_reveal"] and event["gem_resolved"]
            for event in all_choices
        ),
        "pusher_took_gem": sum(event["pusher_took_gem"] for event in all_choices),
    }
    outcomes["row0_gem_offers"] = totals["row0_gem_offers"]
    outcomes["row0_gem_choices"] = totals["row0_gem_choices"]
    outcomes["unconditional_displacement_rate_given_choice"] = _rate(
        totals["displaced_before_reveal"], totals["row0_gem_choices"]
    )
    outcomes["condition_counts"] = condition_stats
    outcomes["totals"] = totals
    return outcomes


def paired_change(
    control: list[dict[str, Any]], treatment: list[dict[str, Any]]
) -> dict[str, Any]:
    output = {}
    for metric in ("win", "top_score", "score", "rank", "margin"):
        diff = np.asarray(
            [new[metric] - old[metric] for old, new in zip(control, treatment)],
            dtype=np.float64,
        )
        mean = float(diff.mean()) if len(diff) else 0.0
        se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
        output[metric] = {
            "treatment_minus_control": mean,
            "paired_se": se,
            "normal_95_ci": [mean - 1.96 * se, mean + 1.96 * se],
        }
    return output


def _per_game_condition_counts(
    row: dict[str, Any], condition: str
) -> dict[str, int]:
    offers = [offer for offer in row["offers"] if offer[condition]]
    choices = [event for event in row["choices"] if event[condition]]
    displaced = [event for event in choices if event["displaced_before_reveal"]]
    resolved = [event for event in choices if event["gem_resolved"]]
    exact = [
        event
        for event in choices
        if event["displaced_before_reveal"] and event["gem_resolved"]
    ]
    return {
        "offers": len(offers),
        "choices": len(choices),
        "displaced_before_reveal": len(displaced),
        "gem_resolved": len(resolved),
        "displaced_and_gem_resolved": len(exact),
        "pusher_took_gem": sum(event["pusher_took_gem"] for event in exact),
    }


def paired_targeted_change(
    control: list[dict[str, Any]], treatment: list[dict[str, Any]]
) -> dict[str, Any]:
    """Paired per-game count changes for the primary all-rivals-deeper event."""
    condition = "all_active_rivals_on_deeper_cards"
    before = [_per_game_condition_counts(row, condition) for row in control]
    after = [_per_game_condition_counts(row, condition) for row in treatment]
    metrics = (
        "offers",
        "choices",
        "displaced_before_reveal",
        "gem_resolved",
        "displaced_and_gem_resolved",
        "pusher_took_gem",
    )
    changes = {}
    for metric in metrics:
        old = np.asarray([row[metric] for row in before], dtype=np.float64)
        new = np.asarray([row[metric] for row in after], dtype=np.float64)
        diff = new - old
        mean = float(diff.mean()) if len(diff) else 0.0
        se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
        changes[metric] = {
            "default_mean_per_game": float(old.mean()) if len(old) else 0.0,
            "guard_mean_per_game": float(new.mean()) if len(new) else 0.0,
            "guard_minus_default_per_game": mean,
            "paired_se": se,
            "normal_95_ci": [mean - 1.96 * se, mean + 1.96 * se],
        }
    return {"condition": condition, "metrics": changes}


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Exposed row-0 own-column Gems\n\n",
        f"Checkpoint: `{report['checkpoint']}`. Each game has one focal neural "
        "seat against three fixed opponents; focal seat rotates. Default and "
        "guard arms share game seeds and neural RNG seeds. The guard masks the "
        "Gem whenever an active rival has movement depth greater than zero and "
        "another policy action remains. Engine legality is unchanged.\n\n",
        "## Event definition and denominators\n\n",
        "An **offer** is a neural decision at which its legal row-0 home-column "
        "cell is a visible Gem. A **choice** is the neural policy selecting that "
        "action. The primary ‘all active rivals on deeper cards’ condition means "
        "every rival still in the mine is on a card row greater than 0 at the "
        "moment of choice (it is false if no rival is active). ‘Any deeper rival’ "
        "instead means at least one active rival has movement depth greater than "
        "0, including a rival in a pit cage at that depth. An **actual displacement** "
        "is a successful engine push of the focal seat off row 0 before that "
        "round’s reveal. **Gem resolved** means the row-0 Gem was actually "
        "collected during that execution; a dragon that ends the reveal first is "
        "not counted as resolved. **Pusher took Gem** means the same seat that "
        "displaced the focal seat collected that Gem in that reveal. Offers "
        "and choices are decision instances (the same Gem can be offered again "
        "if planning revisits it); the choice and offer denominators are shown "
        "separately.\n\n",
        "The candidate is a policy-only mask over the existing neural checkpoint, "
        "not a retrain. Performance differences are paired by seed and focal seat; "
        "outcome intervals are paired normal 95% intervals. Event-rate intervals "
        "are Wilson 95% binomial intervals.\n\n",
    ]
    condition_order = (
        ("all_active_rivals_on_deeper_cards", "All active rivals on deeper cards"),
        ("any_deeper_rival", "At least one active rival has deeper movement depth (guard condition)"),
        ("unsettled_deeper_rival", "At least one unsettled deeper rival (existing exposure signal)"),
    )
    for opponent, title in report["opponent_fields"].items():
        lines.extend([f"## {title}\n\n"])
        for arm in ("default", "guard"):
            stat = report["comparisons"][opponent][arm]
            lines.extend([
                f"### {arm.title()} policy\n\n",
                "| Condition at choice | Gem offers | Gem choices / offers | "
                "Displaced before reveal / choices | Gem resolved / choices | "
                "Displaced + resolved / choices | Same pusher took / displaced + resolved |\n",
                "| :-- | --: | --: | --: | --: | --: | --: |\n",
            ])
            for key, label in condition_order:
                event = stat["condition_counts"][key]
                interval = event["pusher_take_rate_wilson_95_ci"]
                displacement_interval = _format_wilson_interval(
                    event["displacement_rate_wilson_95_ci"], event["choices"]
                )
                exact_interval = _format_wilson_interval(
                    event["displaced_and_resolved_rate_wilson_95_ci"],
                    event["choices"],
                )
                pusher_interval = _format_wilson_interval(
                    interval, event["displaced_and_gem_resolved"]
                )
                lines.append(
                    f"| {label} | {event['offers']} | "
                    f"{event['choices']}/{event['offers']} "
                    f"({_format_rate(event['choices'], event['offers'])}) | "
                    f"{event['displaced_before_reveal']}/{event['choices']} "
                    f"({_format_rate(event['displaced_before_reveal'], event['choices'])}; "
                    f"Wilson 95% CI {displacement_interval}) | "
                    f"{event['gem_resolved']}/{event['choices']} "
                    f"({_format_rate(event['gem_resolved'], event['choices'])}) | "
                    f"{event['displaced_and_gem_resolved']}/{event['choices']} "
                    f"({_format_rate(event['displaced_and_gem_resolved'], event['choices'])}; "
                    f"Wilson 95% CI {exact_interval}) | "
                    f"{event['pusher_took_gem']}/{event['displaced_and_gem_resolved']} "
                    f"({_format_rate(event['pusher_took_gem'], event['displaced_and_gem_resolved'])}; "
                    f"Wilson 95% CI {pusher_interval}) |\n"
                )
            guard_condition = stat["condition_counts"]["any_deeper_rival"]
            guard_exc = guard_condition["guard_kept_choice_due_to_no_alternative"]
            guard_blocked = guard_condition["guard_would_block_choice"]
            lines.extend([
                f"\nAll row-0 Gem offers: **{stat['row0_gem_offers']}**; policy chose "
                f"the row-0 Gem **{stat['row0_gem_choices']}** times; successful "
                f"displacements before reveal: **{stat['totals']['displaced_before_reveal']}**; "
                f"Gems actually collected at reveal: **{stat['totals']['gem_resolved']}**; "
                f"same pusher collected: **{stat['totals']['pusher_took_gem']}**. "
                f"Among this arm's choices with any deeper rival, the guard "
                f"would block {guard_blocked} and retain {guard_exc} because no "
                f"alternative policy action remained. Unfinished games: "
                f"{stat['unfinished_games']}/{stat['games']}.\n\n"
            ])
            examples = stat["condition_counts"]["all_active_rivals_on_deeper_cards"][
                "displaced_examples"
            ]
            if examples:
                lines.append("Examples where the pusher got the Gem:\n\n")
                for event in examples[:5]:
                    lines.append(
                        f"- seed {event['game_seed']}, focal P{event['seat'] + 1}, "
                        f"haul {event['haul']} round {event['round']}, pusher "
                        f"P{event['pusher'] + 1}, reveal occupant "
                        f"P{event['reveal_occupant'] + 1}.\n"
                    )
                lines.append("\n")

        target_change = report["comparisons"][opponent]["paired_targeted_change"]
        lines.extend([
            "### Paired target-event counts\n\n",
            "Per-game counts for choices made while every active rival was on a "
            "deeper card; the same seeds and focal seats are paired.\n\n",
            "| Measure | Default per game | Guard per game | Guard − default per game (paired 95% CI) |\n",
            "| :-- | --: | --: | --: |\n",
        ])
        for metric, label in (
            ("offers", "Gem offers"),
            ("choices", "Gem choices"),
            ("displaced_before_reveal", "Displaced before reveal"),
            ("gem_resolved", "Gem resolved"),
            ("displaced_and_gem_resolved", "Exact sequence: displaced + Gem resolved"),
            ("pusher_took_gem", "Same pusher took Gem"),
        ):
            values = target_change["metrics"][metric]
            low, high = values["normal_95_ci"]
            lines.append(
                f"| {label} | {values['default_mean_per_game']:.3f} | "
                f"{values['guard_mean_per_game']:.3f} | "
                f"{values['guard_minus_default_per_game']:+.3f} "
                f"[{low:+.3f}, {high:+.3f}] |\n"
            )
        lines.append("\n")

        default = report["comparisons"][opponent]["default"]
        guard = report["comparisons"][opponent]["guard"]
        change = report["comparisons"][opponent]["paired_change"]
        lines.extend([
            "### Full-game outcomes\n\n",
            "| Outcome | Default | Guard | Guard − default (paired 95% CI) |\n",
            "| :-- | --: | --: | --: |\n",
        ])
        for metric, label, scale, fmt, unit in (
            ("win_rate", "Win rate", 100.0, ".1f", "%"),
            ("top_score_rate", "Top-score rate", 100.0, ".1f", "%"),
            ("mean_score", "Mean score", 1.0, ".3f", ""),
            ("mean_rank", "Mean rank (lower is better)", 1.0, ".3f", ""),
            ("mean_margin", "Mean score margin", 1.0, ".3f", ""),
        ):
            before = default[metric] * scale
            after = guard[metric] * scale
            key = (
                "win"
                if metric == "win_rate"
                else "top_score"
                if metric == "top_score_rate"
                else metric.removeprefix("mean_")
            )
            interval = change[key]["normal_95_ci"]
            low, high = (value * scale for value in interval)
            unit_delta = "pp" if unit == "%" else ""
            lines.append(
                f"| {label} | {before:{fmt}}{unit} | {after:{fmt}}{unit} | "
                f"{change[key]['treatment_minus_control'] * scale:+{fmt}} "
                f"[{low:{fmt}}, {high:{fmt}}] {unit_delta} |\n"
            )
        lines.append("\n")
    lines.append(
        "**Interpretation:** the mask prevents the neural policy from entering "
        "a row-0 home Gem while any active deeper rival remains, unless that is "
        "the only remaining policy action. This is an experimental policy wrapper; "
        "it changes neither game legality nor the trained weights.\n"
    )
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/selfplay4_pushaware.npz")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=8200000)
    parser.add_argument(
        "--opponents",
        choices=("all", "neural", "rule", "always-pusher"),
        default="all",
    )
    parser.add_argument("--json", default="ROW0_GEM_EXPOSURE_AUDIT.json")
    parser.add_argument("--report", default="ROW0_GEM_EXPOSURE_AUDIT.md")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")

    net = PolicyNet.load(args.checkpoint)
    originals = _install_spies()
    try:
        fields = (
            ("neural", "Three forced-policy neural controls"),
            ("rule", "Three rule agents"),
            ("always-pusher", "Three always-pushers"),
        )
        report: dict[str, Any] = {
            "checkpoint": args.checkpoint,
            "games_per_arm": args.games,
            "seed": args.seed,
            "opponent_fields": {},
            "comparisons": {},
        }
        for field_index, (opponent, title) in enumerate(fields):
            if args.opponents not in ("all", opponent):
                continue
            print(f"Auditing {title} ({args.games} paired games)...", flush=True)
            field_seed = args.seed + field_index * 100_000
            default_rows = evaluate_arm(
                net,
                games=args.games,
                seed=field_seed,
                opponent=opponent,
                guard=False,
            )
            guard_rows = evaluate_arm(
                net,
                games=args.games,
                seed=field_seed,
                opponent=opponent,
                guard=True,
            )
            report["opponent_fields"][opponent] = title
            report["comparisons"][opponent] = {
                "seed": field_seed,
                "default": summarize(default_rows),
                "guard": summarize(guard_rows),
                "paired_change": paired_change(default_rows, guard_rows),
                "paired_targeted_change": paired_targeted_change(
                    default_rows, guard_rows
                ),
            }
    finally:
        _restore_spies(originals)

    Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    rendered = render_report(report)
    Path(args.report).write_text(rendered)
    print(rendered)
    print(f"Wrote {args.json} and {args.report}")


if __name__ == "__main__":
    main()
