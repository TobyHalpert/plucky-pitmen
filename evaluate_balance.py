"""Balance report for Plucky Pitmen.

Runs agent-vs-agent matches, then reports win rates, score differentials,
confidence intervals, game dynamics (dragons, blasts) and seat effects.  It is
the measurement half of the self-play pipeline: train agents with
``train_selfplay.py``, then compare them here against each other and against
the hand-tuned rule policy.

Examples
--------
``python evaluate_balance.py --agents rule,greedy,random,cautious``
``python evaluate_balance.py --agents rule,checkpoints/selfplay.npz --games 300``
``python evaluate_balance.py --agents rule --mirror --games 300``

A strategy is only evidence of dominance if it stays ahead across seats,
opponent policies and seeds - the report therefore rotates seats and shows
95% bootstrap intervals instead of a single win rate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from agents import NeuralAgent, make_scripted_agent
from learning import PolicyNet
from selfplay import SeriesGame, play_series


# ----------------------------------------------------------------------
# Agent specs
# ----------------------------------------------------------------------
Factory = Callable[[int], Any]


def parse_agents(
    specs: Sequence[str],
    *,
    temperature: float = 0.7,
    epsilon: float = 0.0,
) -> list[tuple[str, Factory]]:
    agents: list[tuple[str, Factory]] = []
    for spec in specs:
        path = Path(spec)
        if path.exists() or spec.endswith(".npz"):
            net = PolicyNet.load(spec)
            name = path.stem
            agents.append(
                (
                    name,
                    lambda seed, net=net, name=name: NeuralAgent(
                        net,
                        seed=seed,
                        temperature=temperature,
                        epsilon=epsilon,
                        name=name,
                    ),
                )
            )
            continue
        policy, _, blast = spec.partition(":")
        blast = blast or "never"
        name = spec.replace(":", "-")
        agents.append(
            (
                name,
                lambda seed, policy=policy, blast=blast: make_scripted_agent(
                    policy, seed=seed, blast=blast
                ),
            )
        )
    return agents


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(samples, len(array)))
    estimates = statistic(array[draws], axis=1)
    return (
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


@dataclass
class PolicyStats:
    name: str
    games: int
    wins: int
    top_scores: int
    mean_score: float
    score_ci: tuple[float, float]
    mean_margin: float
    margin_ci: tuple[float, float]
    mean_rank: float
    wins_ci: tuple[float, float]
    escape_rate: float = 0.0
    zero_gem_escape_rate: float = 0.0
    mean_gems_on_escape: float = 0.0
    # Split of points by whether a rival banked loot. A policy that only scores
    # when its opponents stay in the mine is farming their mistakes.
    hauls: int = 0
    rival_loot_hauls: int = 0
    points_when_rivals_loot: float = 0.0
    points_when_rivals_stay: float = 0.0
    # Games the engine could not finish. They score every seat zero, so they
    # must be zero or the numbers below are not trustworthy.
    aborted: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "games": self.games,
            "win_rate": self.wins / self.games if self.games else 0.0,
            "win_rate_ci": self.wins_ci,
            "top_score_rate": self.top_scores / self.games if self.games else 0.0,
            "mean_score": self.mean_score,
            "mean_score_ci": self.score_ci,
            "mean_margin": self.mean_margin,
            "mean_margin_ci": self.margin_ci,
            "mean_rank": self.mean_rank,
            "escape_rate": self.escape_rate,
            "zero_gem_escape_rate": self.zero_gem_escape_rate,
            "mean_gems_on_escape": self.mean_gems_on_escape,
            "rival_loot_haul_rate": (
                self.rival_loot_hauls / self.hauls if self.hauls else 0.0
            ),
            "points_when_rivals_loot": self.points_when_rivals_loot,
            "points_when_rivals_stay": self.points_when_rivals_stay,
            "aborted": self.aborted,
        }


def summarise(series: Sequence[SeriesGame], name: str) -> PolicyStats:
    aborted = sum(int(game.result.aborted) for game in series)
    wins = 0
    tops = 0
    scores: list[float] = []
    margins: list[float] = []
    ranks: list[float] = []
    for game in series:
        policy_scores = game.scores_by_policy()
        own = policy_scores[0]
        others = policy_scores[1:]
        scores.append(own)
        margins.append(own - float(np.mean(others)))
        wins += int(game.winner_policy() == 0)
        tops += int(own == max(policy_scores))
        ranks.append(1 + sum(1 for score in policy_scores if score > own))
    win_flags = [1.0 if game.winner_policy() == 0 else 0.0 for game in series]

    escapes = 0
    zero_gem_escapes = 0
    gems_on_escape: list[int] = []
    rival_loot_hauls = 0
    points_loot: list[float] = []
    points_stay: list[float] = []
    for game in series:
        seat = game.policy_of_seat.index(0)
        previous = 0
        for haul in game.result.hauls:
            delta = haul.scores[seat] - previous
            previous = haul.scores[seat]
            rivals_looted = any(
                index != seat
                and index < len(haul.gems)
                and index in haul.escaped
                and haul.gems[index] > 0
                for index in range(len(haul.gems))
            )
            if rivals_looted:
                rival_loot_hauls += 1
                points_loot.append(float(delta))
            else:
                points_stay.append(float(delta))
            if seat in haul.escaped:
                escapes += 1
                gems = haul.gems[seat] if seat < len(haul.gems) else 0
                gems_on_escape.append(gems)
                zero_gem_escapes += int(gems == 0)
    hauls = sum(len(game.result.hauls) for game in series)

    return PolicyStats(
        name=name,
        games=len(series),
        wins=wins,
        top_scores=tops,
        mean_score=float(np.mean(scores)) if scores else 0.0,
        score_ci=bootstrap_ci(scores) if scores else (0.0, 0.0),
        mean_margin=float(np.mean(margins)) if margins else 0.0,
        margin_ci=bootstrap_ci(margins) if margins else (0.0, 0.0),
        mean_rank=float(np.mean(ranks)) if ranks else 0.0,
        wins_ci=bootstrap_ci(win_flags) if win_flags else (0.0, 0.0),
        escape_rate=escapes / hauls if hauls else 0.0,
        zero_gem_escape_rate=(
            zero_gem_escapes / escapes if escapes else 0.0
        ),
        mean_gems_on_escape=(
            float(np.mean(gems_on_escape)) if gems_on_escape else 0.0
        ),
        hauls=hauls,
        rival_loot_hauls=rival_loot_hauls,
        points_when_rivals_loot=(
            float(np.mean(points_loot)) if points_loot else 0.0
        ),
        points_when_rivals_stay=(
            float(np.mean(points_stay)) if points_stay else 0.0
        ),
        aborted=aborted,
    )


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------
def benchmark(
    target: tuple[str, Factory],
    opponent: tuple[str, Factory],
    *,
    games: int,
    players: int,
    seed: int,
    passes: str = "n-1",
    solo_escape_bonus: bool = True,
    blast_choice_sees_fronts: bool = True,
) -> tuple[PolicyStats, list[SeriesGame]]:
    name, factory = target
    _, opponent_factory = opponent
    factories = [factory] + [opponent_factory] * (players - 1)
    series = play_series(
        factories,
        games,
        seed=seed,
        rotate_seats=True,
        passes=passes,
        solo_escape_bonus=solo_escape_bonus,
        blast_choice_sees_fronts=blast_choice_sees_fronts,
    )
    return summarise(series, name), series


def pairwise(
    left: tuple[str, Factory],
    right: tuple[str, Factory],
    *,
    games: int,
    players: int,
    seed: int,
    passes: str = "n-1",
    solo_escape_bonus: bool = True,
    blast_choice_sees_fronts: bool = True,
) -> tuple[PolicyStats, PolicyStats]:
    left_name, left_factory = left
    right_name, right_factory = right
    left_series = play_series(
        [left_factory] + [right_factory] * (players - 1),
        games,
        seed=seed,
        rotate_seats=True,
        passes=passes,
        solo_escape_bonus=solo_escape_bonus,
        blast_choice_sees_fronts=blast_choice_sees_fronts,
    )
    right_series = play_series(
        [right_factory] + [left_factory] * (players - 1),
        games,
        seed=seed + 1_000_000,
        rotate_seats=True,
        passes=passes,
        solo_escape_bonus=solo_escape_bonus,
        blast_choice_sees_fronts=blast_choice_sees_fronts,
    )
    return (
        summarise(left_series, left_name),
        summarise(right_series, right_name),
    )


@dataclass
class GameDynamics:
    games: int
    hauls: int
    dragon_hauls: int
    lorry_hauls: int
    blast_announcements: int
    blast_choices: int
    mean_turns: float
    seat_scores: list[float]


def dynamics(series: Sequence[SeriesGame], players: int) -> GameDynamics:
    hauls = dragons = stalls = blasts = choices = 0
    turns = []
    seat_scores = [[] for _ in range(players)]
    for game in series:
        result = game.result
        turns.append(result.turns)
        blasts += result.blasts_announced
        choices += result.blast_choices
        for haul in result.hauls:
            hauls += 1
            dragons += int("dragon" in haul.outcome)
            stalls += int(haul.outcome == "lorry")
        for seat, score in enumerate(result.scores):
            seat_scores[seat].append(score)
    return GameDynamics(
        games=len(series),
        hauls=hauls,
        dragon_hauls=dragons,
        lorry_hauls=stalls,
        blast_announcements=blasts,
        blast_choices=choices,
        mean_turns=float(np.mean(turns)) if turns else 0.0,
        seat_scores=[float(np.mean(s)) if s else 0.0 for s in seat_scores],
    )


def mirror_games(
    agent: tuple[str, Factory],
    *,
    games: int,
    players: int,
    seed: int,
    passes: str = "n-1",
    solo_escape_bonus: bool = True,
    blast_choice_sees_fronts: bool = True,
) -> dict[str, Any]:
    """Play identical agents against each other to measure pure luck."""
    _, factory = agent
    series = play_series(
        [factory] * players,
        games,
        seed=seed,
        rotate_seats=False,
        passes=passes,
        solo_escape_bonus=solo_escape_bonus,
        blast_choice_sees_fronts=blast_choice_sees_fronts,
    )
    spreads = []
    ties = 0
    scores = []
    for game in series:
        values = game.result.scores
        scores.extend(values)
        spreads.append(max(values) - min(values))
        best = max(values)
        ties += int(sum(1 for value in values if value == best) > 1)
    return {
        "games": games,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "score_std_within_game": float(np.mean(spreads)) if spreads else 0.0,
        "max_spread": float(np.max(spreads)) if spreads else 0.0,
        "tie_share": ties / games if games else 0.0,
        "seat_scores": dynamics(series, players).seat_scores,
    }


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def format_ci(low: float, high: float, digits: int = 2) -> str:
    return f"[{low:+.{digits}f}, {high:+.{digits}f}]"


def build_markdown(
    *,
    players: int,
    games: int,
    benchmark_name: str,
    leaderboard: list[PolicyStats],
    pair_results: list[tuple[PolicyStats, PolicyStats]],
    dynamics_all: GameDynamics,
    mirror: dict[str, Any] | None,
    seed: int,
) -> str:
    lines: list[str] = []
    lines.append("# Plucky Pitmen balance report\n")
    lines.append(
        f"- players per game: **{players}**\n"
        f"- games per benchmark: **{games}**\n"
        f"- benchmark opponents: **{benchmark_name}**\n"
        f"- base seed: `{seed}`\n"
        f"- games the engine abandoned (all seats scored zero): "
        f"**{sum(stat.aborted for stat in leaderboard)}**\n"
    )

    lines.append("\n## Leaderboard\n")
    lines.append(
        "Each policy occupies one seat (rotated every game) against "
        f"{players - 1} `{benchmark_name}` agents. "
        "`margin` is the policy's score minus the mean score of the other seats.\n"
    )
    lines.append(
        "| policy | games | win % | top-score % | mean score [95% CI] | margin [95% CI] | escapes (0 gems) | gems on escape |\n"
        "|:--|--:|--:|--:|:--|:--|:--|--:|\n"
    )
    for stat in sorted(leaderboard, key=lambda item: item.mean_margin, reverse=True):
        lines.append(
            f"| {stat.name} | {stat.games} | {100 * stat.wins / max(stat.games, 1):.1f}% | "
            f"{100 * stat.top_scores / max(stat.games, 1):.1f}% | "
            f"{stat.mean_score:+.2f} {format_ci(*stat.score_ci)} | "
            f"{stat.mean_margin:+.2f} {format_ci(*stat.margin_ci)} | "
            f"{100 * stat.escape_rate:.0f}% ({100 * stat.zero_gem_escape_rate:.0f}%) | "
            f"{stat.mean_gems_on_escape:.2f} |\n"
        )

    if pair_results:
        lines.append("\n## Pairwise head-to-head\n")
        lines.append(
            "`A vs B` means A is the single A seat against "
            f"{players - 1} B agents; `B vs A` is the mirror image. "
            "A policy is a credible dominant strategy only if it wins in both directions "
            "and the confidence interval excludes zero.\n"
        )
        lines.append(
            "| A | B | A top-score % | A margin [95% CI] | B top-score % | B margin [95% CI] |\n"
            "|:--|:--|--:|:--|--:|:--|\n"
        )
        for left, right in pair_results:
            lines.append(
                f"| {left.name} | {right.name} | "
                f"{100 * left.top_scores / max(left.games, 1):.1f}% | "
                f"{left.mean_margin:+.2f} {format_ci(*left.margin_ci)} | "
                f"{100 * right.top_scores / max(right.games, 1):.1f}% | "
                f"{right.mean_margin:+.2f} {format_ci(*right.margin_ci)} |\n"
            )

    lines.append("\n## Where the points come from\n")
    lines.append(
        "Per-haul points split by whether **a rival left the mine carrying "
        "loot**. A policy that scores while its opponents stay is being paid "
        "for their mistakes, not for its own play: head-to-head margins hide "
        "this, because the benchmark opponents may simply never leave.\n"
    )
    lines.append(
        "| policy | hauls with a rival banking loot | points/haul then | points/haul otherwise |\n"
        "|:--|--:|--:|--:|\n"
    )
    for stat in sorted(leaderboard, key=lambda item: item.mean_margin, reverse=True):
        lines.append(
            f"| {stat.name} | "
            f"{100 * stat.rival_loot_hauls / max(stat.hauls, 1):.1f}% | "
            f"{stat.points_when_rivals_loot:+.2f} | "
            f"{stat.points_when_rivals_stay:+.2f} |\n"
        )

    lines.append("\n## Game dynamics\n")
    lines.append(
        f"- hauls ending on a dragon: **{100 * dynamics_all.dragon_hauls / max(dynamics_all.hauls, 1):.1f}%** "
        f"({dynamics_all.dragon_hauls}/{dynamics_all.hauls})\n"
        f"- hauls ending in a lorry stand-off (everyone left inside on a lorry): "
        f"**{100 * dynamics_all.lorry_hauls / max(dynamics_all.hauls, 1):.1f}%** "
        f"({dynamics_all.lorry_hauls}/{dynamics_all.hauls})\n"
        f"- blast announcements per game: **{dynamics_all.blast_announcements / max(dynamics_all.games, 1):.2f}**\n"
        f"- planning decisions per game (all seats): **{dynamics_all.mean_turns:.1f}**\n"
    )

    lines.append("\n## Seat effects\n")
    lines.append(
        "Mean score by seat across every game in the report. "
        "A large spread means position (who owns which column) matters.\n\n"
    )
    lines.append("| seat | mean score |\n|--:|--:|\n")
    for seat, score in enumerate(dynamics_all.seat_scores):
        lines.append(f"| {seat} | {score:+.2f} |\n")

    if mirror:
        lines.append("\n## Luck vs skill\n")
        lines.append(
            "Mirror matches: every seat runs the *same* policy, so any score "
            "spread is luck plus seat position, not skill.\n\n"
            f"- mean score: **{mirror['mean_score']:+.2f}**\n"
            f"- mean within-game spread (max - min): **{mirror['score_std_within_game']:.2f} points**\n"
            f"- games where the top score was tied: **{100 * mirror['tie_share']:.1f}%**\n"
        )
        if mirror["score_std_within_game"] > 0:
            lines.append(
                "\nCompare this spread with the between-policy margins above. "
                "If mirror spread is larger than the margins, most of the score "
                "variance is luck, and small win-rate differences need many games "
                "before they mean anything.\n"
            )

    lines.append(
        "\n---\n\n**How to read this:** with a heavily luck-driven game, "
        "`win %` is dominated by tie-breaks. Use `margin` and `top-score %`, "
        "and require non-overlapping confidence intervals before calling a "
        "strategy better. Re-run with a different `--seed` before drawing "
        "conclusions.\n"
    )
    return "".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agents",
        default="rule,greedy,random,cautious",
        help="comma separated: rule, greedy, random, cautious, rule:rich, path.npz",
    )
    parser.add_argument("--players", type=int, choices=(3, 4, 5), default=3)
    parser.add_argument("--games", type=int, default=200, help="games per benchmark")
    parser.add_argument("--pair-games", type=int, default=100)
    parser.add_argument("--mirror-games", type=int, default=100)
    parser.add_argument("--benchmark", default="rule")
    parser.add_argument("--mirror-agent", default=None)
    parser.add_argument("--skip-pairs", action="store_true")
    parser.add_argument("--skip-mirror", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument(
        "--pass-rule",
        choices=("n-1", "n"),
        default="n-1",
        help="consecutive passes needed to resolve a planning phase",
    )
    parser.add_argument(
        "--solo-escape-bonus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="award the haul bonus to a sole escapee even without gems",
    )
    parser.add_argument(
        "--blind-blast-choice",
        action="store_true",
        help="choose blast cards face down, as the printed rules say",
    )
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument("--out", default="reports/balance.md")
    parser.add_argument("--json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    specs = [spec.strip() for spec in args.agents.split(",") if spec.strip()]
    agents = parse_agents(
        specs, temperature=args.temperature, epsilon=args.epsilon
    )
    if len(agents) < 2:
        raise SystemExit("need at least two agents")
    benchmark_agent = parse_agents(
        [args.benchmark], temperature=args.temperature, epsilon=args.epsilon
    )[0]

    leaderboard: list[PolicyStats] = []
    all_series: list[SeriesGame] = []
    for index, agent in enumerate(agents):
        stats, series = benchmark(
            agent,
            benchmark_agent,
            games=args.games,
            players=args.players,
            seed=args.seed + index * 100_000,
            passes=args.pass_rule,
            solo_escape_bonus=args.solo_escape_bonus,
            blast_choice_sees_fronts=not args.blind_blast_choice,
        )
        leaderboard.append(stats)
        all_series.extend(series)
        print(
            f"benchmark {agent[0]:24s} margin {stats.mean_margin:+.2f} "
            f"{format_ci(*stats.margin_ci)}",
            flush=True,
        )

    pair_results: list[tuple[PolicyStats, PolicyStats]] = []
    if not args.skip_pairs:
        for left_index in range(len(agents)):
            for right_index in range(left_index + 1, len(agents)):
                left, right = pairwise(
                    agents[left_index],
                    agents[right_index],
                    games=args.pair_games,
                    players=args.players,
                    seed=args.seed + 3_000_000 + left_index * 10_000 + right_index,
                        passes=args.pass_rule,
                    solo_escape_bonus=args.solo_escape_bonus,
                    blast_choice_sees_fronts=not args.blind_blast_choice,
                )
                pair_results.append((left, right))
                print(
                    f"pair {left.name} vs {right.name}: "
                    f"{left.mean_margin:+.2f} / {right.mean_margin:+.2f}",
                    flush=True,
                )

    all_dynamics = dynamics(all_series, args.players)

    mirror = None
    if not args.skip_mirror:
        mirror_spec = args.mirror_agent or specs[0]
        mirror_agent = parse_agents(
            [mirror_spec], temperature=args.temperature, epsilon=args.epsilon
        )[0]
        mirror = mirror_games(
            mirror_agent,
            games=args.mirror_games,
            players=args.players,
            seed=args.seed + 7_000_000,
            passes=args.pass_rule,
            solo_escape_bonus=args.solo_escape_bonus,
            blast_choice_sees_fronts=not args.blind_blast_choice,
        )

    markdown = build_markdown(
        players=args.players,
        games=args.games,
        benchmark_name=benchmark_agent[0],
        leaderboard=leaderboard,
        pair_results=pair_results,
        dynamics_all=all_dynamics,
        mirror=mirror,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown)

    payload = {
        "players": args.players,
        "games_per_benchmark": args.games,
        "benchmark": benchmark_agent[0],
        "seed": args.seed,
        "leaderboard": [stat.as_dict() for stat in leaderboard],
        "pairs": [
            {"left": left.as_dict(), "right": right.as_dict()}
            for left, right in pair_results
        ],
        "dynamics": {
            "games": all_dynamics.games,
            "hauls": all_dynamics.hauls,
            "dragon_hauls": all_dynamics.dragon_hauls,
            "lorry_hauls": all_dynamics.lorry_hauls,
            "blast_announcements": all_dynamics.blast_announcements,
            "mean_turns": all_dynamics.mean_turns,
            "seat_scores": all_dynamics.seat_scores,
        },
        "mirror": mirror,
    }
    json_path = Path(args.json) if args.json else out.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"\nreport: {out}\ndata:   {json_path}")


if __name__ == "__main__":
    main()
