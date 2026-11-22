"""Paired, seat-rotated comparison for terminal-win self-play checkpoints.

Examples
--------
python compare_win_first.py \
    --baseline checkpoints/selfplay4_balance_s7003.npz \
    --candidate checkpoints/selfplay4_winfirst_s8111.npz \
    --games 180 --seed 31000000

The candidate and baseline get the same game seeds, opponent style, and seat
rotation. The primary outcome is official win share: a tied official win is
split across its winners, including wins decided by the rules' tie-break.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agents import NeuralAgent, make_scripted_agent
from learning import PolicyNet
from selfplay import SeriesGame, play_series

DEFAULT_OPPONENTS = ("rule", "greedy", "cautious", "reader", "rusher", "leaver")
METRICS = (
    "win_share",
    "win_rate",
    "unique_win_rate",
    "top_score_rate",
    "mean_score",
    "score_margin",
    "escapes_per_game",
    "deaths_per_game",
    "blasts_per_game",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_ci(
    values: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _stratified_mean_ci(
    groups: Sequence[Sequence[float]], *, samples: int, seed: int
) -> tuple[float, tuple[float, float]]:
    """Equal-weight opponent styles, resampling games independently per style."""
    arrays = [np.asarray(group, dtype=float) for group in groups if len(group)]
    if not arrays:
        return 0.0, (0.0, 0.0)
    mean = float(np.mean([array.mean() for array in arrays]))
    rng = np.random.default_rng(seed)
    draws = [
        array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
        for array in arrays
    ]
    estimates = np.stack(draws, axis=1).mean(axis=1)
    ci = (
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )
    return mean, ci


def _record_game(game: SeriesGame) -> dict[str, Any]:
    result = game.result
    seat = game.policy_of_seat.index(0)
    winners = result.winners
    won = seat in winners
    own_score = result.scores[seat]
    rival_scores = [score for index, score in enumerate(result.scores) if index != seat]
    escapes = 0
    escape_gems = 0
    deaths = 0
    blasts = 0
    for haul in result.hauls:
        if seat in haul.escaped:
            escapes += 1
            if seat < len(haul.gems):
                escape_gems += int(haul.gems[seat])
        deaths += int(seat in haul.dead)
        blasts += int(seat in haul.blasts)
    return {
        "seed": int(result.seed),
        "seat": seat,
        "aborted": bool(result.aborted),
        "win_share": (1.0 / len(winners)) if won and winners else 0.0,
        "win_rate": float(won),
        "unique_win_rate": float(won and len(winners) == 1),
        "top_score_rate": float(own_score == max(result.scores)),
        "mean_score": float(own_score),
        "score_margin": float(own_score - np.mean(rival_scores)),
        "escapes_per_game": float(escapes),
        "escape_gems": int(escape_gems),
        "escape_count": int(escapes),
        "deaths_per_game": float(deaths),
        "blasts_per_game": float(blasts),
    }


def _summarise(records: Sequence[dict[str, Any]], *, seed: int, samples: int) -> dict[str, Any]:
    completed = [record for record in records if not record["aborted"]]
    output: dict[str, Any] = {
        "attempted": len(records),
        "completed": len(completed),
        "aborted": len(records) - len(completed),
    }
    for offset, metric in enumerate(METRICS):
        values = [float(record[metric]) for record in completed]
        output[metric] = float(np.mean(values)) if values else 0.0
        if metric in ("win_share", "score_margin"):
            output[f"{metric}_ci"] = _bootstrap_ci(
                values, samples=samples, seed=seed + offset
            )
    total_escapes = sum(int(record["escape_count"]) for record in completed)
    output["gems_per_escape"] = (
        sum(int(record["escape_gems"]) for record in completed) / total_escapes
        if total_escapes
        else 0.0
    )
    return output


def _paired_delta(
    baseline: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    base_by_seed = {record["seed"]: record for record in baseline}
    cand_by_seed = {record["seed"]: record for record in candidate}
    if base_by_seed.keys() != cand_by_seed.keys():
        raise ValueError("baseline and candidate evaluations used different seeds")
    pairs = [
        (base_by_seed[game_seed], cand_by_seed[game_seed])
        for game_seed in sorted(base_by_seed)
        if not base_by_seed[game_seed]["aborted"]
        and not cand_by_seed[game_seed]["aborted"]
    ]
    output: dict[str, Any] = {
        "matched_completed": len(pairs),
        "baseline_only_aborts": sum(r["aborted"] for r in baseline),
        "candidate_only_aborts": sum(r["aborted"] for r in candidate),
    }
    for offset, metric in enumerate(METRICS):
        deltas = [float(cand[metric] - base[metric]) for base, cand in pairs]
        output[metric] = float(np.mean(deltas)) if deltas else 0.0
        if metric in ("win_share", "score_margin"):
            output[f"{metric}_ci"] = _bootstrap_ci(
                deltas, samples=samples, seed=seed + offset
            )
    return output


def _target_factory(net: PolicyNet, name: str, temperature: float):
    def factory(seed: int):
        return NeuralAgent(
            net,
            seed=seed,
            temperature=temperature,
            name=name,
        )

    return factory


def evaluate_pair(
    baseline: PolicyNet,
    candidate: PolicyNet,
    *,
    players: int,
    opponents: Sequence[str],
    games: int,
    seed: int,
    temperature: float,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, dict[str, list[dict[str, Any]]]]]:
    """Run matched baseline/candidate games against each fixed opponent style."""
    cells: dict[str, Any] = {}
    all_records: dict[str, dict[str, list[dict[str, Any]]]] = {
        "baseline": {},
        "candidate": {},
    }
    for opponent_index, opponent in enumerate(opponents):
        game_seed = seed + opponent_index * games
        opponent_factory = lambda s, policy=opponent: make_scripted_agent(  # noqa: E731
            policy, seed=s
        )
        cell_records: dict[str, list[dict[str, Any]]] = {}
        for label, net in (("baseline", baseline), ("candidate", candidate)):
            series = play_series(
                [_target_factory(net, label, temperature)]
                + [opponent_factory] * (players - 1),
                games,
                seed=game_seed,
                rotate_seats=True,
            )
            cell_records[label] = [_record_game(game) for game in series]

        baseline_by_seed = {r["seed"]: r for r in cell_records["baseline"]}
        candidate_by_seed = {r["seed"]: r for r in cell_records["candidate"]}
        if baseline_by_seed.keys() != candidate_by_seed.keys():
            raise ValueError("baseline and candidate evaluations used different seeds")
        matched_seeds = [
            game_seed
            for game_seed in sorted(baseline_by_seed)
            if not baseline_by_seed[game_seed]["aborted"]
            and not candidate_by_seed[game_seed]["aborted"]
        ]
        matched_baseline = [baseline_by_seed[game_seed] for game_seed in matched_seeds]
        matched_candidate = [candidate_by_seed[game_seed] for game_seed in matched_seeds]
        all_records["baseline"][opponent] = matched_baseline
        all_records["candidate"][opponent] = matched_candidate
        paired_delta = _paired_delta(
            cell_records["baseline"],
            cell_records["candidate"],
            seed=seed + opponent_index * 100 + 3,
            samples=bootstrap_samples,
        )
        cells[opponent] = {
            "attempted": games,
            "baseline_aborted": sum(r["aborted"] for r in cell_records["baseline"]),
            "candidate_aborted": sum(r["aborted"] for r in cell_records["candidate"]),
            "baseline": _summarise(
                matched_baseline,
                seed=seed + opponent_index * 100 + 1,
                samples=bootstrap_samples,
            ),
            "candidate": _summarise(
                matched_candidate,
                seed=seed + opponent_index * 100 + 2,
                samples=bootstrap_samples,
            ),
            "paired_delta": paired_delta,
        }
        print(
            f"{players}p vs {opponent:<8} | "
            f"win share {cells[opponent]['baseline']['win_share']:.3f} -> "
            f"{cells[opponent]['candidate']['win_share']:.3f} | "
            f"paired Δ {paired_delta['win_share']:+.3f}",
            flush=True,
        )
    cells["_overall"] = _overall_summary(
        all_records,
        opponents=opponents,
        seed=seed + 777,
        samples=bootstrap_samples,
    )
    return cells, all_records


def _overall_summary(
    records: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    opponents: Sequence[str],
    seed: int,
    samples: int,
) -> dict[str, Any]:
    """Equal-weight opponent styles over the common completed-game sample."""
    output: dict[str, Any] = {}
    for label in ("baseline", "candidate"):
        output[label] = {}
        groups = records[label]
        for metric in METRICS:
            values_by_style = [
                [float(record[metric]) for record in groups[opponent]]
                for opponent in opponents
            ]
            mean, ci = _stratified_mean_ci(
                values_by_style,
                samples=samples,
                seed=seed + (0 if label == "baseline" else 1000) + METRICS.index(metric),
            )
            output[label][metric] = mean
            if metric in ("win_share", "score_margin"):
                output[label][f"{metric}_ci"] = ci
        output[label]["completed"] = sum(
            len(groups[opponent]) for opponent in opponents
        )
        output[label]["aborted"] = 0
        escape_count = sum(
            int(record["escape_count"])
            for opponent in opponents
            for record in groups[opponent]
        )
        escape_gems = sum(
            int(record["escape_gems"])
            for opponent in opponents
            for record in groups[opponent]
        )
        output[label]["gems_per_escape"] = (
            escape_gems / escape_count if escape_count else 0.0
        )

    delta_output: dict[str, Any] = {}
    for metric in METRICS:
        delta_groups = []
        for opponent in opponents:
            base_by_seed = {
                record["seed"]: record for record in records["baseline"][opponent]
            }
            candidate_by_seed = {
                record["seed"]: record for record in records["candidate"][opponent]
            }
            deltas = [
                float(candidate_by_seed[game_seed][metric] - base_by_seed[game_seed][metric])
                for game_seed in sorted(base_by_seed.keys() & candidate_by_seed.keys())
            ]
            delta_groups.append(deltas)
        mean, ci = _stratified_mean_ci(
            delta_groups,
            samples=samples,
            seed=seed + 2000 + METRICS.index(metric),
        )
        delta_output[metric] = mean
        if metric in ("win_share", "score_margin"):
            delta_output[f"{metric}_ci"] = ci
    delta_output["matched_completed"] = sum(len(group) for group in delta_groups)
    output["paired_delta"] = delta_output
    return output


def _fmt_ci(mean: float, ci: Sequence[float], *, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return (
        f"{mean * scale:+.2f}{suffix} "
        f"[{ci[0] * scale:+.2f}{suffix}, {ci[1] * scale:+.2f}{suffix}]"
    )


def _make_report(data: dict[str, Any]) -> str:
    cfg = data["evaluation"]
    baseline_aborts = sum(
        data["results"][str(players)][opponent]["baseline_aborted"]
        for players in cfg["players"]
        for opponent in cfg["opponents"]
    )
    candidate_aborts = sum(
        data["results"][str(players)][opponent]["candidate_aborted"]
        for players in cfg["players"]
        for opponent in cfg["opponents"]
    )
    training = data.get("training", {})
    if training.get("generations_requested") is not None:
        run_games = (
            training["generations_requested"] * training["games_per_generation"]
        )
        training_line = (
            f"- Training run: {training['generations_requested']} generations × "
            f"{training['games_per_generation']} games = {run_games} games; "
            f"seed {training['seed']}, {training['players']} players, mirror "
            f"probability {training['mirror_prob']:.2f}, league probability "
            f"{training['league_prob']:.2f}."
        )
    else:
        training_line = "- Training run details are not embedded in the candidate checkpoint."
    lines = [
        "# Win-first retraining comparison",
        "",
        f"Generated **{data['generated_utc'][:10]}**.",
        "",
        "## What changed",
        "",
        f"- Warm start: `{data['checkpoints']['baseline']['path']}`.",
        f"- Retrained checkpoint: `{data['checkpoints']['candidate']['path']}`.",
        f"- Training objective: **{data['training']['objective']}**, with `gamma={data['training']['gamma']}`.",
        "- The previous trainer rewarded score deltas and intermediate gem/card/escape/death shaping, then added a winner bonus.",
        "- The new mode gives only the final official winners reward, split as `1 / number_of_winners`; all other seats get zero.",
        "  It disables intermediate rewards and forces `gamma=1`, so runner-up placement and points do not receive positive reward.",
        "- Batched REINFORCE returns now reset at each game's episode boundary instead of leaking outcomes between games.",
        training_line,
        "",
        "## Evaluation method",
        "",
        f"- Player counts tested: **{', '.join(map(str, cfg['players']))}** seats.",
        f"- Opponent styles: {', '.join(f'`{name}`' for name in cfg['opponents'])}.",
        f"- **{cfg['games_per_opponent']}** games per opponent and player count for each model; seats rotate evenly.",
        f"- Candidate and baseline use the same held-out seeds, opponent style, and seat schedule; neural action temperature is **{cfg['temperature']:.2f}**.",
        "- Primary metric is official win share: a co-win is shared evenly, after applying the official tie-break. Score/margin are diagnostics only.",
        "- Paired differences use 95% percentile bootstrap intervals over games completed by both models. Overall panels equally weight the opponent styles.",
        f"- Aborted games are excluded: baseline {baseline_aborts}, retrained {candidate_aborts}; table N is the matched completed count.",
        "",
        "## Primary result: official win share",
        "",
        "| Players | Matched games | Baseline | Win-only retrain | Paired Δ [95% CI] | Unique win rate, before → after |",
        "|--:|--:|:--|:--|:--|:--|",
    ]
    for players in cfg["players"]:
        summary = data["results"][str(players)]["_overall"]
        base = summary["baseline"]
        candidate = summary["candidate"]
        delta = summary["paired_delta"]
        n = delta["matched_completed"]
        lines.append(
            f"| {players} | {n} | "
            f"{_fmt_ci(base['win_share'], base['win_share_ci'], percent=True)} | "
            f"{_fmt_ci(candidate['win_share'], candidate['win_share_ci'], percent=True)} | "
            f"{_fmt_ci(delta['win_share'], delta['win_share_ci'], percent=True)} | "
            f"{base['unique_win_rate']:.1%} → {candidate['unique_win_rate']:.1%} |"
        )
    overall_deltas = {
        players: data["results"][str(players)]["_overall"]["paired_delta"]
        for players in cfg["players"]
    }
    all_uncertain = all(
        delta["win_share_ci"][0] <= 0.0 <= delta["win_share_ci"][1]
        for delta in overall_deltas.values()
    )
    trend_text = ", ".join(
        f"{players}p {delta['win_share']:+.1%}"
        for players, delta in overall_deltas.items()
    )
    margin_text = ", ".join(
        f"{players}p {delta['score_margin']:+.2f} points"
        for players, delta in overall_deltas.items()
    )
    if all_uncertain:
        interpretation = (
            f"**Reading:** The paired 95% win-share interval crosses zero at every player count. "
            f"The point estimates are mixed ({trend_text}), so this run does not establish a reliable win-share improvement. "
            f"Mean score-margin changes were {margin_text}; those remain diagnostics, not the objective."
        )
    else:
        interpretation = (
            f"**Reading:** Pooled paired win-share changes are {trend_text}. "
            f"Mean score-margin changes were {margin_text}; these remain diagnostics, not the objective."
        )
    lines.extend(["", interpretation])
    lines.extend(
        [
            "",
            "## Per-opponent win-share deltas",
            "",
            "| Players | Opponent | Matched games | Baseline win share | Win-only win share | Paired Δ [95% CI] | Mean-margin Δ (points) |",
            "|--:|:--|--:|--:|--:|:--|--:|",
        ]
    )
    for players in cfg["players"]:
        cells = data["results"][str(players)]
        for opponent in cfg["opponents"]:
            cell = cells[opponent]
            paired = cell["paired_delta"]
            lines.append(
                f"| {players} | {opponent} | {paired['matched_completed']} | "
                f"{cell['baseline']['win_share']:.1%} | {cell['candidate']['win_share']:.1%} | "
                f"{_fmt_ci(paired['win_share'], paired['win_share_ci'], percent=True)} | "
                f"{paired['score_margin']:+.2f} |"
            )
    lines.extend(
        [
            "",
            "## Behavior and score diagnostics",
            "",
            "These describe play in this bot panel; none supersedes the official win outcome.",
            "",
            "| Players | Mean score, before → after | Mean point margin, before → after | Escapes/game, before → after | Gems per escaped haul, before → after | Deaths/game, before → after | Blasts/game, before → after |",
            "|--:|:--|:--|:--|:--|:--|:--|",
        ]
    )
    for players in cfg["players"]:
        summary = data["results"][str(players)]["_overall"]
        base, candidate = summary["baseline"], summary["candidate"]
        lines.append(
            f"| {players} | {base['mean_score']:.2f} → {candidate['mean_score']:.2f} | "
            f"{base['score_margin']:+.2f} → {candidate['score_margin']:+.2f} | "
            f"{base['escapes_per_game']:.2f} → {candidate['escapes_per_game']:.2f} | "
            f"{base['gems_per_escape']:.2f} → {candidate['gems_per_escape']:.2f} | "
            f"{base['deaths_per_game']:.2f} → {candidate['deaths_per_game']:.2f} | "
            f"{base['blasts_per_game']:.2f} → {candidate['blasts_per_game']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "This is a controlled comparison against scripted bots, not an estimate of human-relative strength. It tests one warm start and one training seed; it cannot establish that the new objective is generally superior across independent training runs.",
            "The checkpoint-level result is still useful as a before/after check of this specific retraining run. See the accompanying JSON for exact seeds, intervals, hashes, and cell summaries.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    baseline = PolicyNet.load(baseline_path)
    candidate = PolicyNet.load(candidate_path)
    results: dict[str, Any] = {}
    for players in args.players:
        player_seed = args.seed + players * 1_000_000
        cells, _records = evaluate_pair(
            baseline,
            candidate,
            players=players,
            opponents=args.opponents,
            games=args.games,
            seed=player_seed,
            temperature=args.temperature,
            bootstrap_samples=args.bootstrap_samples,
        )
        results[str(players)] = cells
    data = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": {
            "baseline": {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
            "candidate": {"path": str(candidate_path), "sha256": _sha256(candidate_path)},
        },
        "training": {
            "objective": candidate.metadata.get("training_objective", "unknown"),
            "gamma": candidate.metadata.get("training_gamma"),
            "seed": candidate.metadata.get("training_seed"),
            "generations_requested": candidate.metadata.get("training_generations_requested"),
            "games_per_generation": candidate.metadata.get("training_games_per_generation"),
            "players": candidate.metadata.get("training_players"),
            "mirror_prob": candidate.metadata.get("training_mirror_prob"),
            "league_prob": candidate.metadata.get("training_league_prob"),
            "resumed_from": candidate.metadata.get("resumed_from"),
        },
        "evaluation": {
            "players": args.players,
            "opponents": args.opponents,
            "games_per_opponent": args.games,
            "seed_base": args.seed,
            "seed_rule": "seed_base + players*1,000,000 + opponent_index*games + game_index",
            "temperature": args.temperature,
            "seat_rotation": True,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed_rule": "deterministic per player/opponent/metric",
            "tied_win_share": "one divided equally among official winners after tie-break",
        },
        "results": results,
    }
    report_path = Path(args.report)
    json_path = Path(args.json_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_make_report(data))
    json_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {report_path} and {json_path}")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="checkpoints/selfplay4_balance_s7003.npz")
    parser.add_argument("--candidate", default="checkpoints/selfplay4_winfirst_s8111.npz")
    parser.add_argument("--players", type=int, nargs="+", choices=(3, 4, 5), default=(3, 4, 5))
    parser.add_argument("--opponents", nargs="+", default=DEFAULT_OPPONENTS)
    parser.add_argument("--games", type=int, default=180)
    parser.add_argument("--seed", type=int, default=31_000_000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--report", default="reports/win_first_retraining.md")
    parser.add_argument("--json", dest="json_path", default="reports/win_first_retraining.json")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
