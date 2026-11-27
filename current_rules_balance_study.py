#!/usr/bin/env python3
"""Run a multi-seed, multi-opponent current-rules balance study.

This is intentionally separate from the historical one-off reports produced by
``evaluate_balance.py``. It holds the rules fixed, rotates the target seat, and
aggregates independent evaluation-seed blocks so the same policy panel can be
checked at 3, 4, and 5 players.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from evaluate_balance import bootstrap_ci, parse_agents
from selfplay import SeriesGame, play_series

MODEL_SPECS = {
    "selfplay4": "checkpoints/selfplay4.npz",
    "selfplay4_tactical": "checkpoints/selfplay4_tactical.npz",
    "selfplay4_balance_s7002": "checkpoints/selfplay4_balance_s7002.npz",
    "selfplay4_balance_s7003": "checkpoints/selfplay4_balance_s7003.npz",
}
OPPONENTS = ("rule", "greedy", "cautious", "reader", "rusher", "leaver")
SCRIPTED_CONTROLS = ("rule", "rusher", "leaver")
DEFAULT_SEEDS = (1729, 8191, 65537)
DEFAULT_PLAYERS = (3, 4, 5)


def _metrics(game: SeriesGame) -> dict[str, float] | None:
    """Extract one target-seat observation, or None for an aborted game."""
    if game.result.aborted:
        return None
    scores = game.scores_by_policy()
    own = float(scores[0])
    rivals = [float(value) for value in scores[1:]]
    winners = game.winners_policy()
    seat = game.policy_of_seat.index(0)
    return {
        "score": own,
        "margin": own - float(np.mean(rivals)),
        "win": float(0 in winners),
        "top_score": float(own == max(scores)),
        "rank": float(1 + sum(value > own for value in scores)),
        "seat": float(seat),
    }


def _summarise_records(
    records: Sequence[dict[str, float] | None], *, bootstrap_seed: int = 0
) -> dict[str, Any]:
    complete = [record for record in records if record is not None]
    n = len(complete)
    result: dict[str, Any] = {"games": n}
    for index, (key, source) in enumerate(
        (
            ("mean_score", "score"),
            ("mean_margin", "margin"),
            ("win_rate", "win"),
            ("top_score_rate", "top_score"),
            ("mean_rank", "rank"),
        )
    ):
        values = [record[source] for record in complete]
        result[key] = float(np.mean(values)) if values else 0.0
        if values:
            result[f"{key}_ci"] = bootstrap_ci(
                values, seed=bootstrap_seed + index
            )
        else:
            result[f"{key}_ci"] = (0.0, 0.0)
    result["seat_mean_scores"] = {
        str(seat): float(
            np.mean([record["score"] for record in complete if record["seat"] == seat])
        )
        for seat in sorted({int(record["seat"]) for record in complete})
    }
    return result


def _seed_summary(
    records_by_seed: dict[int, list[dict[str, float] | None]], *, seed_offset: int
) -> dict[str, Any]:
    return {
        str(seed): _summarise_records(records, bootstrap_seed=seed_offset + index * 10)
        for index, (seed, records) in enumerate(records_by_seed.items())
    }


def _ci_text(mean: float, ci: Sequence[float], *, percent: bool = False) -> str:
    if percent:
        return f"{100 * mean:.1f}% [{100 * ci[0]:.1f}%, {100 * ci[1]:.1f}%]"
    return f"{mean:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"


def _plain_ci_text(mean: float, ci: Sequence[float]) -> str:
    return f"{mean:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def _range_text(values: Sequence[float], *, signed: bool = True) -> str:
    if not values:
        return "n/a"
    low, high = min(values), max(values)
    if signed:
        return f"{low:+.2f} to {high:+.2f}"
    return f"{low:.2f} to {high:.2f}"


def _paired_margin_delta(
    left: Sequence[dict[str, float] | None],
    right: Sequence[dict[str, float] | None],
) -> list[float]:
    return [
        new["margin"] - baseline["margin"]
        for baseline, new in zip(left, right)
        if baseline is not None and new is not None
    ]


def _mirror_summary(
    records: Sequence[dict[str, Any] | None], *, bootstrap_seed: int
) -> dict[str, Any]:
    complete = [record for record in records if record is not None]
    spreads = [float(record["spread"]) for record in complete]
    tied = [float(record["tied_top"]) for record in complete]
    all_scores = [float(score) for record in complete for score in record["scores"]]
    seat_count = len(complete[0]["scores"]) if complete else 0
    return {
        "games": len(complete),
        "mean_score_per_seat": float(np.mean(all_scores)) if all_scores else 0.0,
        "mean_score_range": float(np.mean(spreads)) if spreads else 0.0,
        "mean_score_range_ci": bootstrap_ci(spreads, seed=bootstrap_seed),
        "top_tied_share": float(np.mean(tied)) if tied else 0.0,
        "top_tied_share_ci": bootstrap_ci(tied, seed=bootstrap_seed + 1),
        "seat_mean_scores": [
            float(np.mean([record["scores"][seat] for record in complete]))
            for seat in range(seat_count)
        ],
    }


def _make_report(data: dict[str, Any]) -> str:
    meta = data["metadata"]
    model_panel_aborts = sum(
        data["benchmarks"][str(players)][model][opponent]["aborted_games"]
        for players in meta["players"]
        for model in meta["model_names"]
        for opponent in OPPONENTS
    )
    if model_panel_aborts == 0:
        portfolio_note = (
            "All six opponent styles received the same number of attempted games, and no neural benchmark game aborted, "
            "so this panel is equally weighted by completed games. It describes the selected bot set, not a universal expected win rate.\n\n"
        )
    else:
        portfolio_note = (
            "All six opponent styles received the same number of attempted games. Aborted simulations are excluded from estimates, "
            "so completed-game weights can differ; inspect cell abort counts, since outcome-dependent aborts could bias a cell. "
            "This panel describes the selected bot set, not a universal expected win rate.\n\n"
        )
    lines = [
        "# Current-rules balance study\n\n",
        f"Generated **{meta['report_date']}**. This report is separate from the historical `BALANCE.md` measurements.\n\n",
        "## Scope and method\n\n",
        "This is a bot/self-play stress test, **not evidence that human games are balanced**. All games use the same current rules: "
        f"`{meta['pass_rule']}` consecutive-pass rule, solo-escape haul bonus **on**, and blind sequential blast-card choice **on**. "
        "The target's seat is rotated; each benchmark cell pits one target against identical copies of one named opponent policy. "
        "All reported intervals are 95% bootstrap intervals over completed games; engine-aborted games are excluded, not scored as losses, and their counts are reported per cell. The tables show the range of seed-block mean margins, and the JSON retains per-seed summaries. These intervals do not capture variation over human players or a population of independently trained models.\n\n",
        f"- Evaluation seeds: `{', '.join(str(seed) for seed in meta['evaluation_seeds'])}`\n",
        f"- Games per seed and cell: **{meta['games_per_seed']}** (seat-rotated; {meta['games_per_seed'] * len(meta['evaluation_seeds'])} attempted per cell)\n",
        f"- Neural action temperature: **{meta['temperature']:.2f}**\n",
        f"- Complete/aborted games: **{meta['completed_games']:,} / {meta['aborted_games']:,}**\n",
        f"- Total simulated games (including mirror/control games): **{meta['attempted_games']:,}**\n",
    ]
    section_aborts = meta.get("aborted_by_section", {})
    if section_aborts:
        lines.append(
            "- Aborted by block: "
            + ", ".join(f"{name} {count:,}" for name, count in sorted(section_aborts.items()))
            + "\n"
        )
    reasons = meta.get("abort_reasons", {})
    if reasons:
        lines.append(
            "- Abort reasons: "
            + ", ".join(f"`{name}` {count:,}" for name, count in sorted(reasons.items()))
            + "\n"
        )
    lines.extend([
        "\n### Checkpoints\n\n",
        "| policy | checkpoint | SHA-256 (prefix) | role in this study |\n|:--|:--|:--|:--|\n",
    ])
    roles = {
        "selfplay4": "shipped reference",
        "selfplay4_tactical": "downstream tactical policy (not an independent training replicate)",
        "selfplay4_balance_s7002": "60-generation continuation from `selfplay4`, training seed 7002",
        "selfplay4_balance_s7003": "60-generation continuation from `selfplay4`, training seed 7003",
    }
    for name, item in meta["models"].items():
        lines.append(
            f"| `{name}` | `{item['path']}` | `{item['sha256'][:12]}` | {roles[name]} |\n"
        )

    lines.extend(
        [
            "\n## Quick read\n\n",
            "For each player count, the model-panel margin range is the min-to-max of the four checkpoint means against the six selected bot styles; it is not a confidence interval. Paired deltas compare each new continuation with the shipped checkpoint on the same game seeds and opponent cells.\n\n",
            "| players | model-panel margin range | seed 7002 Δ margin [95% CI] | seed 7003 Δ margin [95% CI] | mirror mean score-range range |\n",
            "|--:|:--|:--|:--|:--|\n",
        ]
    )
    for players in meta["players"]:
        pkey = str(players)
        panel_margins = [
            data["benchmarks"][pkey][model]["_portfolio"]["pooled"]["mean_margin"]
            for model in meta["model_names"]
        ]
        seed7002 = data["paired_vs_selfplay4"][pkey]["selfplay4_balance_s7002"]
        seed7003 = data["paired_vs_selfplay4"][pkey]["selfplay4_balance_s7003"]
        mirror_ranges = [
            data["mirror"][pkey][model]["mean_score_range"]
            for model in meta["model_names"]
        ]
        lines.append(
            f"| {players} | {_range_text(panel_margins)} | "
            f"{_ci_text(seed7002['mean_delta'], seed7002['delta_ci'])} | "
            f"{_ci_text(seed7003['mean_delta'], seed7003['delta_ci'])} | "
            f"{_range_text(mirror_ranges, signed=False)} |\n"
        )

    for players in meta["players"]:
        pkey = str(players)
        lines.extend(
            [
                f"\n## {players}-player games\n\n",
                f"Each row has **{meta['games_per_seed'] * len(meta['evaluation_seeds'])} attempted games** ({meta['games_per_seed']} per evaluation seed) unless aborted; margins are target score minus the mean score of the other seats. The seed range is the minimum-to-maximum of the {len(meta['evaluation_seeds'])} seed-block mean margins, not a confidence interval.\n\n",
                "| target policy | opponent style | completed N | win rate [95% CI] | top-score rate | mean margin [95% CI] | mean score [95% CI] | seed margin range | aborted |\n",
                "|:--|:--|--:|:--|--:|:--|:--|:--|--:|\n",
            ]
        )
        block = data["benchmarks"][pkey]
        for model in meta["model_names"]:
            for opponent in OPPONENTS:
                cell = block[model][opponent]
                pooled = cell["pooled"]
                seed_margins = [row["mean_margin"] for row in cell["per_seed"].values()]
                lines.append(
                    f"| `{model}` | `{opponent}` | {pooled['games']} | "
                    f"{_ci_text(pooled['win_rate'], pooled['win_rate_ci'], percent=True)} | "
                    f"{100 * pooled['top_score_rate']:.1f}% | "
                    f"{_ci_text(pooled['mean_margin'], pooled['mean_margin_ci'])} | "
                    f"{_ci_text(pooled['mean_score'], pooled['mean_score_ci'])} | "
                    f"{_range_text(seed_margins)} | {cell['aborted_games']} |\n"
                )

        lines.extend(
            [
                "\n### Equal-weight opponent portfolio\n\n",
                portfolio_note,
                "| target policy | completed N | win rate [95% CI] | top-score rate | mean margin [95% CI] | mean score [95% CI] | seed margin range |\n",
                "|:--|--:|:--|--:|:--|:--|:--|\n",
            ]
        )
        for model in meta["model_names"]:
            portfolio = block[model]["_portfolio"]
            seed_margins = [row["mean_margin"] for row in portfolio["per_seed"].values()]
            pooled = portfolio["pooled"]
            lines.append(
                f"| `{model}` | {pooled['games']} | "
                f"{_ci_text(pooled['win_rate'], pooled['win_rate_ci'], percent=True)} | "
                f"{100 * pooled['top_score_rate']:.1f}% | "
                f"{_ci_text(pooled['mean_margin'], pooled['mean_margin_ci'])} | "
                f"{_ci_text(pooled['mean_score'], pooled['mean_score_ci'])} | "
                f"{_range_text(seed_margins)} |\n"
            )

        lines.extend(
            [
                "\n### Paired margin differences vs shipped `selfplay4`\n\n",
                f"Positive values mean the named checkpoint had a higher score margin than `selfplay4` on the same evaluation seed and opponent-style cells. The intervals bootstrap paired game-level differences across all {len(OPPONENTS)} styles and {len(meta['evaluation_seeds'])} seed blocks. These compare particular checkpoints; they are not estimates of training-algorithm effects.\n\n",
                "| checkpoint | paired games | margin difference [95% CI] |\n",
                "|:--|--:|:--|\n",
            ]
        )
        for model in meta["model_names"]:
            if model == "selfplay4":
                continue
            comparison = data["paired_vs_selfplay4"][pkey][model]
            lines.append(
                f"| `{model}` | {comparison['games']} | "
                f"{_ci_text(comparison['mean_delta'], comparison['delta_ci'])} |\n"
            )

        lines.extend(
            [
                "\n### Scripted reference policies vs `rule`\n\n",
                f"These controls make the tested opponent styles easier to interpret; each had {meta['games_per_seed'] * len(meta['evaluation_seeds'])} attempted games, with engine-aborted games excluded from estimates.\n\n",
                "| target policy | completed N | win rate [95% CI] | mean margin [95% CI] | seed margin range | aborted |\n",
                "|:--|--:|:--|:--|:--|--:|\n",
            ]
        )
        for policy in SCRIPTED_CONTROLS:
            control = data["scripted_controls"][pkey][policy]
            pooled = control["pooled"]
            seed_margins = [row["mean_margin"] for row in control["per_seed"].values()]
            lines.append(
                f"| `{policy}` | {pooled['games']} | "
                f"{_ci_text(pooled['win_rate'], pooled['win_rate_ci'], percent=True)} | "
                f"{_ci_text(pooled['mean_margin'], pooled['mean_margin_ci'])} | "
                f"{_range_text(seed_margins)} | {control['aborted_games']} |\n"
            )

        control_totals = {
            policy: data["scripted_controls"][pkey][policy]["aborted_games"]
            for policy in SCRIPTED_CONTROLS
        }
        control_aborted = sum(control_totals.values())
        highest_abort_policy = max(control_totals, key=control_totals.get)
        control_reasons = meta.get("abort_reasons", {})
        reason_text = ", ".join(
            f"`{reason}` {count}"
            for reason, count in sorted(control_reasons.items())
        ) or "no aborts"
        section_aborts = meta.get("aborted_by_section", {})
        other_aborts = section_aborts.get("neural benchmark", 0) + section_aborts.get("mirror", 0)
        main_note = (
            "No neural benchmark or mirror games aborted"
            if other_aborts == 0
            else f"{other_aborts} neural benchmark/mirror games also aborted"
        )
        lines.append(
            f"\nAt this player count, **{control_aborted}** scripted-control games aborted. "
            f"Across the full study, the abort reasons were {reason_text}. {main_note}. "
            f"The `{highest_abort_policy}` control has the highest abort count here; "
            "high-abort control estimates are limited diagnostics, not reliable policy estimates.\n"
        )

        lines.extend(
            [
                "\n### Identical-policy mirror matches (luck/seat-spread proxy)\n\n",
                "Every seat uses the same checkpoint. The mean within-game score range (highest minus lowest score) captures stochastic outcome and seat effects together; it is not a pure decomposition of luck.\n\n",
                "| checkpoint | completed N | mean score range [95% CI] | games with tied top score [95% CI] | mean score by seat |\n",
                "|:--|--:|:--|:--|:--|\n",
            ]
        )
        for model in meta["model_names"]:
            mirror = data["mirror"][pkey][model]
            seat_values = ", ".join(f"{value:+.2f}" for value in mirror["seat_mean_scores"])
            lines.append(
                f"| `{model}` | {mirror['games']} | "
                f"{_plain_ci_text(mirror['mean_score_range'], mirror['mean_score_range_ci'])} | "
                f"{_ci_text(mirror['top_tied_share'], mirror['top_tied_share_ci'], percent=True)} | "
                f"{seat_values} |\n"
            )

    lines.extend(
        [
            "\n## Interpretation and limits\n\n",
            "- The primary evidence is the **mean-margin pattern across opponent styles and player counts**, with the seed-block range alongside each pooled interval. A single high win rate against one script is not treated as a balance verdict.\n",
            "- Mirror ranges show that identical policies can finish at different scores even in the same game; compare these descriptive spreads with policy margins before over-reading small differences.\n",
            "- The two new continuation checkpoints share the same starting network and are only two training seeds. The tactical checkpoint is a related policy, not an independent replicate. More independent training seeds and held-out human play would be needed for stronger claims.\n",
            "- These bots are limited, hand-designed or self-play policies. This report **does not establish balance among people**, and the selected opponent mixture is not a model of a human population.\n",
            "- The zero-gem solo-escape bonus remains enabled throughout; no counterfactual no-bonus rules are included. Blast-conditioned escape analysis is out of scope.\n",
            "\n## Reproduction\n\n",
            "From the repository root, with dependencies installed:\n\n",
            "```sh\npython current_rules_balance_study.py\n```\n\n",
            "The command writes `reports/current_rules_balance_study.md` and the machine-readable `reports/current_rules_balance_study.json`. Options `--games-per-seed`, `--seeds`, `--players`, `--output`, and `--json` can change the run.\n",
        ]
    )
    return "".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    players = tuple(int(value) for value in args.players.split(",") if value.strip())
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise SystemExit("provide at least two unique evaluation seeds")
    if not players or any(value not in (3, 4, 5) for value in players):
        raise SystemExit("--players must be a comma-separated subset of 3,4,5")
    if args.games_per_seed <= 0:
        raise SystemExit("--games-per-seed must be positive")

    model_factories = dict(
        parse_agents(
            list(MODEL_SPECS.values()),
            temperature=args.temperature,
            epsilon=0.0,
        )
    )
    opponent_factories = dict(parse_agents(list(OPPONENTS)))
    control_factories = dict(parse_agents(list(SCRIPTED_CONTROLS)))

    # Retain only compact per-game outcomes; do not keep full game logs in RAM.
    benchmark_records: dict[
        tuple[int, str, str, int], list[dict[str, float] | None]
    ] = {}
    control_records: dict[
        tuple[int, str, int], list[dict[str, float] | None]
    ] = {}
    mirror_records: dict[
        tuple[int, str, int], list[dict[str, Any] | None]
    ] = {}
    attempted = aborted = 0
    abort_reasons: Counter[str] = Counter()
    aborted_by_section: Counter[str] = Counter(
        {"neural benchmark": 0, "scripted control": 0, "mirror": 0}
    )

    def evaluate_target(
        target_name: str,
        target_factory: Any,
        opponent_name: str,
        opponent_factory: Any,
        player_count: int,
        seed: int,
        section: str,
    ) -> list[dict[str, float] | None]:
        nonlocal attempted, aborted
        series = play_series(
            [target_factory] + [opponent_factory] * (player_count - 1),
            args.games_per_seed,
            seed=seed,
            rotate_seats=True,
            passes="n-1",
            solo_escape_bonus=True,
            blast_choice_sees_fronts=False,
        )
        attempted += len(series)
        result = [_metrics(game) for game in series]
        n_aborted = sum(record is None for record in result)
        aborted += n_aborted
        aborted_by_section[section] += n_aborted
        abort_reasons.update(
            game.result.abort_reason or "unknown"
            for game in series
            if game.result.aborted
        )
        print(
            f"benchmark p={player_count} target={target_name} vs={opponent_name} "
            f"seed={seed}: {len(series) - n_aborted}/{len(series)} complete",
            flush=True,
        )
        return result

    for player_count in players:
        for model_name, model_factory in model_factories.items():
            for opponent_name, opponent_factory in opponent_factories.items():
                for seed in seeds:
                    benchmark_records[(player_count, model_name, opponent_name, seed)] = (
                        evaluate_target(
                            model_name,
                            model_factory,
                            opponent_name,
                            opponent_factory,
                            player_count,
                            seed,
                            "neural benchmark",
                        )
                    )
        for policy_name, policy_factory in control_factories.items():
            for seed in seeds:
                control_records[(player_count, policy_name, seed)] = evaluate_target(
                    policy_name,
                    policy_factory,
                    "rule",
                    opponent_factories["rule"],
                    player_count,
                    seed,
                    "scripted control",
                )
        for model_name, model_factory in model_factories.items():
            for seed in seeds:
                series = play_series(
                    [model_factory] * player_count,
                    args.games_per_seed,
                    seed=seed,
                    rotate_seats=False,
                    passes="n-1",
                    solo_escape_bonus=True,
                    blast_choice_sees_fronts=False,
                )
                attempted += len(series)
                records: list[dict[str, Any] | None] = []
                for game in series:
                    if game.result.aborted:
                        records.append(None)
                        aborted += 1
                        aborted_by_section["mirror"] += 1
                        abort_reasons.update([game.result.abort_reason or "unknown"])
                    else:
                        scores = [float(score) for score in game.result.scores]
                        top = max(scores)
                        records.append(
                            {
                                "scores": scores,
                                "spread": max(scores) - min(scores),
                                "tied_top": float(sum(score == top for score in scores) > 1),
                            }
                        )
                mirror_records[(player_count, model_name, seed)] = records
                print(
                    f"mirror p={player_count} model={model_name} seed={seed}: "
                    f"{len(series) - sum(record is None for record in records)}/{len(series)} complete",
                    flush=True,
                )

    benchmarks: dict[str, Any] = {}
    scripted_controls: dict[str, Any] = {}
    mirror: dict[str, Any] = {}
    paired: dict[str, Any] = {}

    for player_count in players:
        pkey = str(player_count)
        benchmarks[pkey] = {}
        for model_index, model_name in enumerate(model_factories):
            benchmarks[pkey][model_name] = {}
            all_model_by_seed: dict[int, list[dict[str, float] | None]] = {
                seed: [] for seed in seeds
            }
            for opponent_index, opponent_name in enumerate(opponent_factories):
                by_seed = {
                    seed: benchmark_records[(player_count, model_name, opponent_name, seed)]
                    for seed in seeds
                }
                pooled_records = [record for seed in seeds for record in by_seed[seed]]
                per_seed = _seed_summary(
                    by_seed,
                    seed_offset=player_count * 100_000 + model_index * 10_000 + opponent_index * 100,
                )
                benchmarks[pkey][model_name][opponent_name] = {
                    "attempted_games": len(pooled_records),
                    "aborted_games": sum(record is None for record in pooled_records),
                    "pooled": _summarise_records(
                        pooled_records,
                        bootstrap_seed=player_count * 100_000 + model_index * 10_000 + opponent_index,
                    ),
                    "per_seed": per_seed,
                }
                for seed in seeds:
                    all_model_by_seed[seed].extend(by_seed[seed])

            portfolio_records = [
                record
                for opponent_name in opponent_factories
                for seed in seeds
                for record in benchmark_records[(player_count, model_name, opponent_name, seed)]
            ]
            portfolio_seed_summaries = _seed_summary(
                all_model_by_seed,
                seed_offset=player_count * 1_000_000 + model_index * 10_000,
            )
            benchmarks[pkey][model_name]["_portfolio"] = {
                "attempted_games": len(portfolio_records),
                "aborted_games": sum(record is None for record in portfolio_records),
                "pooled": _summarise_records(
                    portfolio_records,
                    bootstrap_seed=player_count * 1_000_000 + model_index,
                ),
                "per_seed": portfolio_seed_summaries,
            }

        # Paired comparisons use the same opponent style, evaluation seed, and
        # game index for both checkpoints. Aborted games are excluded pairwise.
        paired[pkey] = {}
        for model_index, model_name in enumerate(model_factories):
            if model_name == "selfplay4":
                continue
            deltas: list[float] = []
            for opponent_name in opponent_factories:
                for seed in seeds:
                    baseline = benchmark_records[(player_count, "selfplay4", opponent_name, seed)]
                    candidate = benchmark_records[(player_count, model_name, opponent_name, seed)]
                    deltas.extend(_paired_margin_delta(baseline, candidate))
            ci = bootstrap_ci(deltas, seed=player_count * 1_000_000 + model_index)
            paired[pkey][model_name] = {
                "games": len(deltas),
                "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
                "delta_ci": ci,
            }

        scripted_controls[pkey] = {}
        for policy_index, policy_name in enumerate(control_factories):
            by_seed = {
                seed: control_records[(player_count, policy_name, seed)] for seed in seeds
            }
            all_records = [record for seed in seeds for record in by_seed[seed]]
            scripted_controls[pkey][policy_name] = {
                "attempted_games": len(all_records),
                "aborted_games": sum(record is None for record in all_records),
                "pooled": _summarise_records(
                    all_records,
                    bootstrap_seed=player_count * 1_000_000 + policy_index,
                ),
                "per_seed": _seed_summary(
                    by_seed,
                    seed_offset=player_count * 2_000_000 + policy_index * 10_000,
                ),
            }

        mirror[pkey] = {}
        for model_index, model_name in enumerate(model_factories):
            all_mirror_records = [
                record
                for seed in seeds
                for record in mirror_records[(player_count, model_name, seed)]
            ]
            mirror[pkey][model_name] = _mirror_summary(
                all_mirror_records,
                bootstrap_seed=player_count * 10_000 + model_index,
            )

    model_meta = {}
    for name, path_value in MODEL_SPECS.items():
        path = Path(path_value)
        model_meta[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    completed = attempted - aborted
    data: dict[str, Any] = {
        "metadata": {
            "report_date": "2026-11-10",
            "pass_rule": "n-1",
            "solo_escape_bonus": True,
            "blind_blast_choice": True,
            "temperature": args.temperature,
            "evaluation_seeds": list(seeds),
            "games_per_seed": args.games_per_seed,
            "players": list(players),
            "model_names": list(model_factories),
            "models": model_meta,
            "completed_games": completed,
            "aborted_games": aborted,
            "attempted_games": attempted,
            "abort_reasons": dict(abort_reasons),
            "aborted_by_section": dict(aborted_by_section),
            "elapsed_seconds": time.time() - started,
        },
        "benchmarks": benchmarks,
        "paired_vs_selfplay4": paired,
        "scripted_controls": scripted_controls,
        "mirror": mirror,
    }
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-seed", type=int, default=60)
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS)
    )
    parser.add_argument(
        "--players", default=",".join(str(value) for value in DEFAULT_PLAYERS)
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default="reports/current_rules_balance_study.md")
    parser.add_argument("--json", default="reports/current_rules_balance_study.json")
    args = parser.parse_args()

    data = run(args)
    output = Path(args.output)
    json_output = Path(args.json)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_make_report(data))
    json_output.write_text(json.dumps(data, indent=2))
    print(
        f"\nStudy complete: {data['metadata']['completed_games']:,}/"
        f"{data['metadata']['attempted_games']:,} games complete, "
        f"{data['metadata']['aborted_games']:,} aborted.\n"
        f"Report: {output}\nData:   {json_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
