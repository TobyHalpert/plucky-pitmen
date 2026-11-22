"""Measure starting-player advantage in matched identical-policy games.

For each seed and player count, every possible starting seat is forced once
while keeping the shuffled deal and per-seat agent RNG seeds fixed. This makes
starter vs nonstarter outcomes a paired role comparison, rather than a
correlation with naturally sampled seeds.

Example
-------
python starting_player_study.py --seeds 180 --seed 47000000
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agents import NeuralAgent, make_scripted_agent
from game_env import MineEnv
from learning import PolicyNet
from selfplay import GameResult, play_game

DEFAULT_POLICIES = (
    "rule",
    "greedy",
    "cautious",
    "reader",
    "rusher",
    "leaver",
    "selfplay4",
    "selfplay4_winfirst_s8111",
)
CHECKPOINTS = {
    "selfplay4": Path("checkpoints/selfplay4.npz"),
    "selfplay4_winfirst_s8111": Path("checkpoints/selfplay4_winfirst_s8111.npz"),
}
METRICS = (
    "starter_win_share",
    "nonstarter_mean_win_share",
    "win_share_delta",
    "starter_win_rate",
    "nonstarter_mean_win_rate",
    "score_margin",
    "starter_top_score_rate",
)


def _bootstrap_ci(
    values: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    estimates = array[indices].mean(axis=1)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def _game_record(result: GameResult, starter: int, players: int) -> dict[str, Any]:
    winners = result.winners
    share_per_winner = 1.0 / len(winners) if winners else 0.0
    nonstarter_scores = [
        score for seat, score in enumerate(result.scores) if seat != starter
    ]
    starter_share = share_per_winner if starter in winners else 0.0
    nonstarter_mean_share = sum(
        share_per_winner for seat in winners if seat != starter
    ) / (players - 1)
    return {
        "seed": result.seed,
        "starter": starter,
        "aborted": result.aborted,
        "starter_win_share": starter_share,
        "nonstarter_mean_win_share": nonstarter_mean_share,
        "win_share_delta": starter_share - nonstarter_mean_share,
        "starter_win_rate": float(starter in winners),
        "nonstarter_mean_win_rate": sum(
            seat in winners for seat in range(players) if seat != starter
        ) / (players - 1),
        "score_margin": float(
            result.scores[starter] - np.mean(nonstarter_scores)
        ),
        "starter_top_score_rate": float(
            result.scores[starter] == max(result.scores)
        ),
    }


def _seed_level_records(
    records: Sequence[dict[str, Any]], players: int
) -> tuple[list[int], dict[str, list[float]]]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_seed[int(record["seed"])].append(record)
    metrics: dict[str, list[float]] = {key: [] for key in METRICS}
    complete_seeds: list[int] = []
    for seed in sorted(by_seed):
        block = by_seed[seed]
        if len(block) != players or any(record["aborted"] for record in block):
            continue
        if {int(record["starter"]) for record in block} != set(range(players)):
            continue
        complete_seeds.append(seed)
        for key in METRICS:
            metrics[key].append(float(np.mean([record[key] for record in block])))
    return complete_seeds, metrics


def _summary(
    records: Sequence[dict[str, Any]],
    *,
    players: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    complete_seeds, by_metric = _seed_level_records(records, players)
    result: dict[str, Any] = {
        "attempted_games": len(records),
        "aborted_games": sum(bool(record["aborted"]) for record in records),
        "complete_seed_blocks": len(complete_seeds),
    }
    for offset, metric in enumerate(METRICS):
        values = by_metric[metric]
        result[metric] = float(np.mean(values)) if values else 0.0
        result[f"{metric}_ci"] = _bootstrap_ci(
            values, samples=samples, seed=seed + offset
        )
    return result


def _make_agents(
    policy: str,
    nets: dict[str, PolicyNet],
    *,
    game_seed: int,
    players: int,
) -> list[Any]:
    agents = []
    for seat in range(players):
        agent_seed = game_seed * 1000 + seat
        if policy in nets:
            agents.append(
                NeuralAgent(
                    nets[policy],
                    seed=agent_seed,
                    temperature=0.7,
                    name=policy,
                )
            )
        else:
            agents.append(make_scripted_agent(policy, seed=agent_seed))
    return agents


def evaluate_policy(
    policy: str,
    nets: dict[str, PolicyNet],
    *,
    players: int,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    records = []
    for game_seed in seeds:
        for starter in range(players):
            env = MineEnv(
                opponents=players - 1,
                passes="n-1",
                solo_escape_bonus=True,
                blast_choice_sees_fronts=False,
            )
            env.reset(seed=game_seed)
            # Keep the shuffled deck fixed; alter only the seat with first move
            # and the corresponding turn-order marker.
            env.starting_player = starter
            env.planning_player = starter
            result = play_game(
                _make_agents(
                    policy,
                    nets,
                    game_seed=game_seed,
                    players=players,
                ),
                seed=game_seed,
                env=env,
            )
            records.append(_game_record(result, starter, players))
    return records


def _aggregate_policies(
    summaries: dict[str, dict[str, Any]],
    *,
    common_seeds: Sequence[int],
    metric: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    # Each policy contributes equally within each paired seed block.
    values_by_policy_seed: dict[str, dict[int, float]] = {}
    for policy, summary in summaries.items():
        values_by_policy_seed[policy] = {
            int(s): float(value)
            for s, value in zip(summary["_seed_ids"], summary["_seed_values"][metric])
        }
    per_seed = [
        float(np.mean([values_by_policy_seed[p][s] for p in summaries]))
        for s in common_seeds
    ]
    return {
        "policies": list(summaries),
        "seed_blocks": len(per_seed),
        "mean": float(np.mean(per_seed)) if per_seed else 0.0,
        "ci": _bootstrap_ci(per_seed, samples=samples, seed=seed),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    nets = {
        name: PolicyNet.load(CHECKPOINTS[name])
        for name in args.policies
        if name in CHECKPOINTS
    }
    seeds = [args.seed + index for index in range(args.seeds)]
    results: dict[str, Any] = {}
    for players in args.players:
        results[str(players)] = {}
        summaries_for_player_count: dict[str, dict[str, Any]] = {}
        for policy_index, policy in enumerate(args.policies):
            records = evaluate_policy(
                policy,
                nets,
                players=players,
                seeds=seeds,
            )
            summary = _summary(
                records,
                players=players,
                samples=args.bootstrap_samples,
                seed=args.seed + players * 10_000 + policy_index * 100,
            )
            complete_seed_ids, seed_metrics = _seed_level_records(records, players)
            summary["_seed_ids"] = complete_seed_ids
            summary["_seed_values"] = seed_metrics
            results[str(players)][policy] = summary
            summaries_for_player_count[policy] = summary
            print(
                f"{players}p {policy:<28} "
                f"starter Δ share {summary['win_share_delta']:+.3f} "
                f"[{summary['win_share_delta_ci'][0]:+.3f}, "
                f"{summary['win_share_delta_ci'][1]:+.3f}]  "
                f"N={summary['complete_seed_blocks']}",
                flush=True,
            )

        common_seeds = sorted(
            set.intersection(
                *[
                    set(summary["_seed_ids"])
                    for summary in summaries_for_player_count.values()
                ]
            )
        )
        panel = {}
        for metric in METRICS:
            panel[metric] = _aggregate_policies(
                summaries_for_player_count,
                common_seeds=common_seeds,
                metric=metric,
                samples=args.bootstrap_samples,
                seed=args.seed + players * 100_000 + METRICS.index(metric),
            )
        results[str(players)]["_panel"] = panel
        results[str(players)]["_panel"]["common_seed_blocks"] = len(common_seeds)

    generated = args.report_date or datetime.now(timezone.utc).date().isoformat()
    data = {
        "generated_date": generated,
        "evaluation": {
            "players": args.players,
            "policies": args.policies,
            "seed_base": args.seed,
            "seed_count": args.seeds,
            "starter_positions_per_seed": "every initial seat, once per fixed-deal seed",
            "starter_scope": "initial starter is forced; subsequent haul order follows normal rules",
            "shared_initial_deal": True,
            "per_seat_agent_seed": "game_seed * 1000 + seat",
            "temperature": 0.7,
            "pass_rule": "n-1",
            "solo_escape_bonus": True,
            "blind_blast_choice": True,
            "primary_metric": "starter official win share minus mean nonstarter seat win share; ties split among official winners",
            "panel_weighting": "equal weight per policy within each fixed-deal seed block",
            "confidence_interval": "95% percentile bootstrap over fixed-deal seed blocks",
            "bootstrap_samples": args.bootstrap_samples,
        },
        "results": results,
    }
    md_path = Path(args.report)
    json_path = Path(args.json_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2))
    md_path.write_text(render_report(data))
    print(f"Wrote {md_path} and {json_path}")
    return data


def _pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _pct_ci(value: float, ci: Sequence[float]) -> str:
    return f"{_pct(value)} [{_pct(ci[0])}, {_pct(ci[1])}]"


def render_report(data: dict[str, Any]) -> str:
    cfg = data["evaluation"]
    policies = cfg["policies"]
    lines = [
        "# Starting-player advantage study",
        "",
        f"Generated **{data['generated_date']}**.",
        "",
        "## Method",
        "",
        f"For each policy and player count, {cfg['seed_count']} fixed-deal seeds ({cfg['seed_base']}–{cfg['seed_base'] + cfg['seed_count'] - 1}) were tested from every possible initial starting seat. Each rotation reused the same shuffled deck seed and each seat's agent RNG seed, then forced the initial starter and planning seat; later haul order followed normal game rules. All seats used the same policy (mirror games).",
        "The primary estimate is starter official win share minus the mean win share of nonstarting seats, averaged over every starter rotation within each fixed-deal seed block. Official ties split one unit among listed winners; score margin is diagnostic only.",
        f"The pooled panel weights the {len(policies)} policies equally within each seed block. The 95% percentile interval uses {cfg['bootstrap_samples']:,} bootstrap resamples of seed blocks, not individual games.",
        f"Rules: {cfg['players']} players, `n-1` passes, solo escape bonus on, blind blast choice on; neural action temperature {cfg['temperature']:.2f}.",
        "",
        "## Pooled starter effect",
        "",
        "| Players | Attempted games | Aborted | Policies in panel | Common complete seed blocks | Starter win share | Mean nonstarter seat win share | Paired starter Δ [95% CI] | Score-margin Δ (points) |",
        "|--:|--:|--:|--:|--:|--:|--:|:--|--:|",
    ]
    for players in cfg["players"]:
        player_results = data["results"][str(players)]
        panel = player_results["_panel"]
        attempted = sum(player_results[p]["attempted_games"] for p in policies)
        aborted = sum(player_results[p]["aborted_games"] for p in policies)
        lines.append(
            f"| {players} | {attempted} | {aborted} | {len(policies)} | {panel['common_seed_blocks']} | "
            f"{_pct_ci(panel['starter_win_share']['mean'], panel['starter_win_share']['ci'])} | "
            f"{_pct_ci(panel['nonstarter_mean_win_share']['mean'], panel['nonstarter_mean_win_share']['ci'])} | "
            f"{_pct_ci(panel['win_share_delta']['mean'], panel['win_share_delta']['ci'])} | "
            f"{panel['score_margin']['mean']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## By policy",
            "",
            "| Players | Policy | Complete seed blocks | Aborted games | Starter win share | Nonstarter seat win share | Paired Δ [95% CI] | Starter win rate | Score-margin Δ |",
            "|--:|:--|--:|--:|--:|--:|--:|:--|--:|",
        ]
    )
    for players in cfg["players"]:
        for policy in policies:
            s = data["results"][str(players)][policy]
            lines.append(
                f"| {players} | {policy} | {s['complete_seed_blocks']} | {s['aborted_games']} | "
                f"{_pct_ci(s['starter_win_share'], s['starter_win_share_ci'])} | "
                f"{_pct_ci(s['nonstarter_mean_win_share'], s['nonstarter_mean_win_share_ci'])} | "
                f"{_pct_ci(s['win_share_delta'], s['win_share_delta_ci'])} | "
                f"{s['starter_win_rate']:.1%} | {s['score_margin']:+.2f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "A positive paired Δ means the seat chosen to start receives more official win share than the average nonstarting seat. The pooled estimates are across equally weighted mirror-policy profiles, not a single universal policy.",
        ]
    )
    pooled_intervals = [
        data["results"][str(players)]["_panel"]["win_share_delta"]["ci"]
        for players in cfg["players"]
    ]
    if all(low <= 0 <= high for low, high in pooled_intervals):
        lines.append(
            "All pooled 95% intervals include zero, so this study does not establish a general starter advantage for 3–5 players under the tested policies and rules."
        )
    else:
        lines.append(
            "At least one pooled 95% interval excludes zero; interpret that player-count result as applying to this tested policy panel and ruleset."
        )
    rusher = data["results"].get("5", {}).get("rusher")
    if rusher and (rusher["win_share_delta_ci"][0] > 0 or rusher["win_share_delta_ci"][1] < 0):
        lines.append(
            f"Exploratory subgroup: five-player rusher mirror games show {_pct_ci(rusher['win_share_delta'], rusher['win_share_delta_ci'])} starter Δ. This is one of {len(policies) * len(cfg['players'])} policy/player comparisons and is not adjusted for multiple testing."
        )
    lines.extend(
        [
            "The study uses same-policy mirror games; it does not estimate starting-seat effects in mixed-policy matches or every possible player strategy. The confidence intervals reflect deal-to-deal sampling variability, not uncertainty over which policies were included.",
            "Aborted games are reported above. A seed block with any aborted starter rotation is excluded from that policy's summary; the pooled panel uses only seed blocks complete for every included policy. The JSON records complete seed IDs, seed-block means, and abort counts.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, nargs="+", choices=(3, 4, 5), default=(3, 4, 5))
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    parser.add_argument("--seeds", type=int, default=180)
    parser.add_argument("--seed", type=int, default=47_000_000)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--report-date", default=None)
    parser.add_argument("--report", default="reports/starting_player_advantage.md")
    parser.add_argument("--json", dest="json_path", default="reports/starting_player_advantage.json")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
