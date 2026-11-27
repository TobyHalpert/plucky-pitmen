"""Paired full-game audit of hard versus confidence-weighted dragon signals.

The current policy hard-masks Pass when a player stands on a foreign card that
its owner appears to have signalled as a dragon. This probe first estimates the
signal precision in forced-policy self-play, then compares that hard response
against a soft mixture: with calibrated confidence ``q``, use the learned action
distribution with weight ``1 - q`` and the same distribution conditioned on
not passing with weight ``q``. Thus q=0 is no signal response and q=1 is the
existing hard dodge.

The confidence estimates are calibrated on disjoint games and are not fitted to
the comparison outcomes. The probe does not change default policy behavior or
engine legality.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import features
from agents import ActionChoice, NeuralAgent, RuleAgent
from game_env import DRAGON, MAX_COLUMNS, PASS, MineEnv, PlayerActionContext
from learning import PolicyNet
from selfplay import play_game

SIGNAL_SOURCES = ("walkoff", "owner_cage")
DRAGON_PRIOR = 2.0
SAFE_PRIOR = 8.0


def signalled_dragon_source(ctx: PlayerActionContext) -> str | None:
    """Identify which public cue made the current card count as signalled."""
    if not features.own_card_is_signalled_dragon(ctx):
        return None
    row = int(ctx.own_position)
    column = ctx.own_column
    if column is None:
        return None
    if any(
        int(vacated_row) == row and int(vacated_column) == int(column)
        for vacated_row, vacated_column, _back in getattr(ctx, "vacated_cells", ())
    ):
        return "walkoff"
    return "owner_cage"


def confidence_weighted_probs(
    probabilities: np.ndarray,
    pass_action: int,
    confidence: float,
) -> np.ndarray:
    """Mix the learned distribution with its Pass-free conditional distribution.

    A confidence of ``q`` assigns weight ``q`` to the learned policy conditioned
    on dodging, and ``1-q`` to the unmodified policy. This makes the Pass
    probability exactly ``(1-q) * P(Pass)`` while preserving the neural policy's
    relative preferences among all non-Pass actions.
    """
    q = float(confidence)
    if not 0.0 <= q <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or not 0 <= int(pass_action) < len(probs):
        raise ValueError("pass_action must index a one-dimensional probability vector")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("probabilities must have positive total mass")
    probs = probs / total
    if q == 0.0 or probs[int(pass_action)] == 0.0:
        return probs

    dodge = probs.copy()
    dodge[int(pass_action)] = 0.0
    nonpass_mass = float(dodge.sum())
    if nonpass_mass <= 0.0:
        # Match action_mask's safety behavior: keep Pass if no alternative exists.
        return probs
    dodge /= nonpass_mass
    adjusted = (1.0 - q) * probs + q * dodge
    return adjusted / adjusted.sum()


class SignalResponseAgent(NeuralAgent):
    """Neural agent with a probe-only forced or confidence-weighted response."""

    def __init__(
        self,
        net: PolicyNet,
        *,
        mode: str,
        confidence: dict[str, float] | None = None,
        seed: int = 0,
        temperature: float = 0.7,
        env: MineEnv | None = None,
    ) -> None:
        if mode not in {"forced", "soft"}:
            raise ValueError("mode must be 'forced' or 'soft'")
        super().__init__(net, seed=seed, temperature=temperature)
        self.mode = mode
        self.confidence = confidence or {}
        self.env = env  # ground truth is read only for audit counts, never for action choice
        self.signal_decisions: Counter[str] = Counter()
        self.signal_passes: Counter[str] = Counter()
        self.signal_dodges: Counter[str] = Counter()
        self.signal_observations: list[dict[str, Any]] = []
        self._seen_signal_cells: set[tuple[int, int, int]] = set()

    def _record_signal(self, ctx: PlayerActionContext, source: str | None) -> None:
        if source is None:
            return
        row = int(ctx.own_position)
        column = int(ctx.own_column)
        self.signal_decisions[source] += 1
        if self.env is None or not (0 <= row < len(self.env.rows)):
            return
        card = self.env.rows[row][column]
        if card is None:
            return
        key = (int(ctx.haul), row, column)
        if key in self._seen_signal_cells:
            return
        self._seen_signal_cells.add(key)
        self.signal_observations.append(
            {
                "haul": int(ctx.haul),
                "row": row,
                "column": column,
                "source": source,
                "dragon": int(card[0]) == DRAGON,
            }
        )

    def decide(self, ctx: PlayerActionContext) -> ActionChoice:
        source = signalled_dragon_source(ctx)
        self._record_signal(ctx, source)
        if self.mode == "forced":
            choice = super().decide(ctx)
        else:
            choice = self._soft_decide(ctx, source)
        if source is not None:
            if int(choice.action) == int(PASS):
                self.signal_passes[source] += 1
            else:
                self.signal_dodges[source] += 1
        return choice

    def _soft_decide(
        self, ctx: PlayerActionContext, source: str | None
    ) -> ActionChoice:
        mask = features.action_mask(
            ctx, force_signalled_dragon_dodge=False
        )
        if not mask.any():
            return ActionChoice(int(PASS))

        state = features.state_features(ctx)
        matrix = features.state_action_matrix(ctx, state=state)
        probabilities = self.net.masked_probs(
            matrix, mask, temperature=self.temperature
        )
        if source is not None and mask[int(PASS)]:
            probabilities = confidence_weighted_probs(
                probabilities,
                int(PASS),
                self.confidence.get(source, 0.2),
            )

        legal = np.flatnonzero(mask)
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            action = int(self.rng.choice(legal))
        elif self.deterministic:
            action = int(np.argmax(probabilities))
        else:
            action = int(self.rng.choice(len(probabilities), p=probabilities))
        logp = float(np.log(probabilities[action] + 1e-12))
        value = self.net.value(state)

        if self.recorder is not None:
            from learning import Decision

            self.recorder.append(
                Decision(
                    player=ctx.player_index if self.player is None else self.player,
                    ctx=ctx,
                    state=state,
                    action=action,
                    logp=logp,
                    value=value,
                )
            )
        primitive, announces_blast = features.split_action(action)
        return ActionChoice(primitive, announces_blast)


def _agents_for_game(
    net: PolicyNet,
    env: MineEnv,
    *,
    focal_seat: int,
    mode: str,
    confidence: dict[str, float],
    seed: int,
    opponents: str,
    audit_all_seats: bool = False,
) -> list[Any]:
    agents = []
    for seat in range(env.n_players):
        agent_seed = seed * 31 + seat
        if audit_all_seats or seat == focal_seat:
            agent_mode = "forced" if audit_all_seats else mode
            agent_confidence = {} if audit_all_seats else confidence
            agents.append(
                SignalResponseAgent(
                    net,
                    mode=agent_mode,
                    confidence=agent_confidence,
                    seed=agent_seed,
                    temperature=0.7,
                    env=env,
                )
            )
        elif opponents == "neural":
            agents.append(
                NeuralAgent(
                    net,
                    seed=agent_seed,
                    temperature=0.7,
                    name="forced-neural-control",
                )
            )
        elif opponents == "rule":
            agents.append(RuleAgent(blast="safe"))
        else:  # pragma: no cover - guarded by argparse
            raise ValueError(f"unknown opponent field: {opponents}")
    return agents


def calibrate_confidence(
    net: PolicyNet,
    *,
    games: int,
    seed: int,
) -> dict[str, Any]:
    """Estimate signal precision on forced-policy self-play, using disjoint seeds."""
    observations: dict[str, dict[tuple[int, int, int, int], bool]] = {
        source: {} for source in SIGNAL_SOURCES
    }
    decision_counts: Counter[str] = Counter()
    for game_index in range(games):
        game_seed = seed + game_index
        env = MineEnv(opponents=3)
        env.reset(seed=game_seed)
        agents = _agents_for_game(
            net,
            env,
            focal_seat=0,
            mode="forced",
            confidence={},
            seed=game_seed,
            opponents="neural",
            audit_all_seats=True,
        )
        play_game(agents, seed=game_seed, env=env)
        for agent in agents:
            decision_counts.update(agent.signal_decisions)
            for item in agent.signal_observations:
                key = (
                    game_index,
                    int(item["haul"]),
                    int(item["row"]),
                    int(item["column"]),
                )
                observations[str(item["source"])].setdefault(
                    key, bool(item["dragon"])
                )

    summary: dict[str, Any] = {"games": games, "prior": "Beta(2, 8)"}
    confidence: dict[str, float] = {}
    for source in SIGNAL_SOURCES:
        values = list(observations[source].values())
        dragons = sum(values)
        count = len(values)
        q = (dragons + DRAGON_PRIOR) / (
            count + DRAGON_PRIOR + SAFE_PRIOR
        )
        confidence[source] = float(q)
        summary[source] = {
            "unique_signalled_cards": count,
            "dragon_cards": int(dragons),
            "empirical_precision": dragons / count if count else None,
            "posterior_mean_confidence": float(q),
            "signalled_decisions": int(decision_counts[source]),
        }
    summary["confidence_used"] = confidence
    return summary


def _game_row(result: Any, seat: int, focal: SignalResponseAgent) -> dict[str, Any]:
    own_score = int(result.scores[seat])
    rivals = [int(score) for index, score in enumerate(result.scores) if index != seat]
    return {
        "win": bool(result.winner == seat),
        "top_score": bool(own_score >= max(rivals)),
        "score": own_score,
        "rank": 1 + sum(score > own_score for score in rivals),
        "margin": own_score - float(np.mean(rivals)),
        "dragons": int(result.dragons),
        "unfinished": bool(result.aborted),
        "signal_decisions": int(sum(focal.signal_decisions.values())),
        "signal_passes": int(sum(focal.signal_passes.values())),
        "signal_dodges": int(sum(focal.signal_dodges.values())),
        "signal_decisions_by_source": dict(focal.signal_decisions),
        "signal_passes_by_source": dict(focal.signal_passes),
        "signal_dodges_by_source": dict(focal.signal_dodges),
    }


def evaluate_arm(
    net: PolicyNet,
    *,
    games: int,
    seed: int,
    mode: str,
    confidence: dict[str, float],
    opponents: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    signal_observations = []
    for game_index in range(games):
        game_seed = seed + game_index
        focal_seat = game_index % 4
        env = MineEnv(opponents=3)
        env.reset(seed=game_seed)
        agents = _agents_for_game(
            net,
            env,
            focal_seat=focal_seat,
            mode=mode,
            confidence=confidence,
            seed=game_seed,
            opponents=opponents,
        )
        result = play_game(agents, seed=game_seed, env=env)
        focal = agents[focal_seat]
        assert isinstance(focal, SignalResponseAgent)
        rows.append(_game_row(result, focal_seat, focal))
        signal_observations.extend(focal.signal_observations)
    return rows, signal_observations


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "games": len(rows),
        "win_rate": float(np.mean([row["win"] for row in rows])) if rows else 0.0,
        "top_score_rate": float(np.mean([row["top_score"] for row in rows])) if rows else 0.0,
        "mean_score": float(np.mean([row["score"] for row in rows])) if rows else 0.0,
        "mean_rank": float(np.mean([row["rank"] for row in rows])) if rows else 0.0,
        "mean_margin": float(np.mean([row["margin"] for row in rows])) if rows else 0.0,
        "dragons_per_game": float(np.mean([row["dragons"] for row in rows])) if rows else 0.0,
        "unfinished": int(sum(row["unfinished"] for row in rows)),
        "signal_decisions": int(sum(row["signal_decisions"] for row in rows)),
        "signal_passes": int(sum(row["signal_passes"] for row in rows)),
        "signal_dodges": int(sum(row["signal_dodges"] for row in rows)),
        "signal_pass_rate": float(
            sum(row["signal_passes"] for row in rows)
            / max(1, sum(row["signal_decisions"] for row in rows))
        ),
        "signal_decisions_by_source": dict(
            sum(
                (Counter(row["signal_decisions_by_source"]) for row in rows),
                Counter(),
            )
        ),
        "signal_passes_by_source": dict(
            sum(
                (Counter(row["signal_passes_by_source"]) for row in rows),
                Counter(),
            )
        ),
        "signal_dodges_by_source": dict(
            sum(
                (Counter(row["signal_dodges_by_source"]) for row in rows),
                Counter(),
            )
        ),
    }


def paired_change(
    forced_rows: list[dict[str, Any]], soft_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(forced_rows) != len(soft_rows):
        raise ValueError("paired arms must have the same number of games")
    output = {}
    for metric in ("win", "top_score", "score", "rank", "margin"):
        forced = np.asarray([row[metric] for row in forced_rows], dtype=np.float64)
        soft = np.asarray([row[metric] for row in soft_rows], dtype=np.float64)
        differences = soft - forced
        mean = float(differences.mean()) if len(differences) else 0.0
        se = (
            float(differences.std(ddof=1) / np.sqrt(len(differences)))
            if len(differences) > 1
            else 0.0
        )
        output[metric] = {
            "soft_minus_forced": mean,
            "paired_se": se,
            "normal_95_ci": [mean - 1.96 * se, mean + 1.96 * se],
        }
    return output


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dragon-signal response: forced versus confidence-weighted\n\n",
        f"Checkpoint: `{report['checkpoint']}`. Full games use one focal neural "
        "seat against three fixed opponents; focal seats rotate. Evaluation seeds "
        "are paired between arms. The confidence calibration uses a disjoint set "
        "of forced-policy four-neural games.\n\n",
        "## Confidence calibration\n\n",
        "A Beta(2, 8) prior (20% dragon, the base rate within a ten-card backside) "
        "smooths the raw truth audit. The soft arm uses the resulting posterior "
        "mean for each signal source.\n\n",
        "| Signal source | Unique cards | Dragons | Raw precision | Confidence used | Signal decisions |\n",
        "| :-- | --: | --: | --: | --: | --: |\n",
    ]
    calibration = report["calibration"]
    for source, label in (("walkoff", "Owner walked off own-column card"),
                          ("owner_cage", "Owner in pit cage, low outside hazard")):
        row = calibration[source]
        raw = "n/a" if row["empirical_precision"] is None else f"{row['empirical_precision']:.1%}"
        lines.append(
            f"| {label} | {row['unique_signalled_cards']} | {row['dragon_cards']} | "
            f"{raw} | {row['posterior_mean_confidence']:.1%} | "
            f"{row['signalled_decisions']} |\n"
        )
    lines.extend([
        "\n## Paired full-game results\n\n",
        "Soft response mixes the normal policy distribution with the same "
        "distribution conditioned on not passing. The mixture weight is the "
        "calibrated dragon confidence; therefore confidence 1 exactly matches "
        "the forced dodge, while confidence 0 leaves the policy unchanged.\n\n",
    ])
    for opponent_key, title in (("neural", "Three forced-policy neural controls"),
                                ("rule", "Three rule agents")):
        if opponent_key not in report["comparisons"]:
            continue
        comparison = report["comparisons"][opponent_key]
        lines.extend([
            f"### {title}\n\n",
            "| Metric | Forced dodge | Soft response | Soft − forced (paired 95% CI) |\n",
            "| :-- | --: | --: | --: |\n",
        ])
        metric_rows = (
            ("win_rate", "Win rate", 100.0, ".1f", "%", "pp"),
            ("top_score_rate", "Top-score rate", 100.0, ".1f", "%", "pp"),
            ("mean_score", "Mean score", 1.0, ".3f", "", ""),
            ("mean_rank", "Mean rank (lower is better)", 1.0, ".3f", "", ""),
            ("mean_margin", "Mean margin", 1.0, ".3f", "", ""),
        )
        for key, label, scale, fmt, value_suffix, change_suffix in metric_rows:
            before = comparison["forced"][key] * scale
            after = comparison["soft"][key] * scale
            change = comparison["paired_change"][
                "win" if key == "win_rate" else
                "top_score" if key == "top_score_rate" else key.removeprefix("mean_")
            ]
            delta = change["soft_minus_forced"] * scale
            low, high = (value * scale for value in change["normal_95_ci"])
            delta_fmt = (
                f"{delta:+{fmt}} [{low:{fmt}}, {high:{fmt}}] {change_suffix}"
                if change_suffix
                else f"{delta:+{fmt}} [{low:{fmt}}, {high:{fmt}}]"
            )
            lines.append(
                f"| {label} | {before:{fmt}}{value_suffix} | "
                f"{after:{fmt}}{value_suffix} | {delta_fmt} |\n"
            )
        for mode in ("forced", "soft"):
            stats = comparison[mode]
            lines.append(
                f"\n{mode.title()} signal response: {stats['signal_passes']}/"
                f"{stats['signal_decisions']} signalled decisions passed "
                f"({stats['signal_pass_rate']:.1%}); {stats['unfinished']} unfinished games.\n"
            )
    soft_signal_decisions = sum(
        report["comparisons"][key]["soft"]["signal_decisions"]
        for key in report["comparisons"]
    )
    soft_signal_passes = sum(
        report["comparisons"][key]["soft"]["signal_passes"]
        for key in report["comparisons"]
    )
    walkoff_n = calibration["walkoff"]["unique_signalled_cards"]
    lines.append(
        "\n## Interpretation and limitations\n\n"
        f"Across these opponent fields the soft arm passed on {soft_signal_passes} "
        f"of {soft_signal_decisions} signalled decisions; the forced arm passed on "
        "none. The paired full-game intervals include zero for every reported "
        "outcome, so this run finds no measurable advantage for either response. "
        "That is not proof of equivalence: the treatment changed very few actions.\n\n"
        f"The owner-in-cage cue calibrated to {calibration['owner_cage']['empirical_precision']:.1%} "
        f"dragon precision over {calibration['owner_cage']['unique_signalled_cards']} "
        "unique cards in forced self-play. The walk-off cue had only "
        f"{walkoff_n} unique cards ({calibration['walkoff']['dragon_cards']}/"
        f"{walkoff_n} were dragons), so its posterior confidence is prior-dominated "
        "and too uncertain to "
        "justify a strategic conclusion. The hard rule remains the default; "
        "the probe does not change production behavior or engine legality.\n\n"
        "The reported intervals are paired normal 95% intervals (mean difference "
        "±1.96 standard errors), bounded to this checkpoint, seed range, and "
        "opponent field.\n"
    )
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/selfplay4_pushaware.npz")
    parser.add_argument("--games", type=int, default=400, help="paired games per opponent field")
    parser.add_argument("--calibration-games", type=int, default=600)
    parser.add_argument("--seed", type=int, default=6100000)
    parser.add_argument("--opponents", choices=("both", "neural", "rule"), default="both")
    parser.add_argument("--json", default="SIGNALLED_DRAGON_RESPONSE_AUDIT.json")
    parser.add_argument("--report", default="SIGNALLED_DRAGON_RESPONSE_AUDIT.md")
    args = parser.parse_args()
    if args.games < 1 or args.calibration_games < 1:
        parser.error("--games and --calibration-games must be positive")

    net = PolicyNet.load(args.checkpoint)
    print(
        f"Calibrating signal confidence on {args.calibration_games} forced-policy games...",
        flush=True,
    )
    calibration = calibrate_confidence(
        net, games=args.calibration_games, seed=args.seed
    )
    confidence = calibration["confidence_used"]
    report: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "calibration_seed": args.seed,
        "evaluation_seed": args.seed + 1_000_000,
        "calibration": calibration,
        "comparisons": {},
        "method": (
            "One focal neural seat in forced or soft mode against three controls. "
            "The focal seat rotates; each forced/soft pair uses the same game seed "
            "and per-seat policy RNG seeds. Confidence is a disjoint forced-policy "
            "self-play calibration with Beta(2,8) smoothing. Soft probabilities "
            "are (1-q)*P_net + q*P_net(conditioned on not Pass)."
        ),
    }
    opponent_fields = (
        ("neural", "forced-policy neural controls"),
        ("rule", "rule agents"),
    )
    for index, (opponents, label) in enumerate(opponent_fields):
        if args.opponents not in ("both", opponents):
            continue
        eval_seed = args.seed + 1_000_000 + index * 100_000
        print(
            f"Evaluating {args.games} paired games versus {label}...",
            flush=True,
        )
        forced_rows, _ = evaluate_arm(
            net,
            games=args.games,
            seed=eval_seed,
            mode="forced",
            confidence=confidence,
            opponents=opponents,
        )
        soft_rows, _ = evaluate_arm(
            net,
            games=args.games,
            seed=eval_seed,
            mode="soft",
            confidence=confidence,
            opponents=opponents,
        )
        report["comparisons"][opponents] = {
            "opponents": label,
            "seed": eval_seed,
            "forced": summarize(forced_rows),
            "soft": summarize(soft_rows),
            "paired_change": paired_change(forced_rows, soft_rows),
        }

    rendered_json = json.dumps(report, indent=2)
    Path(args.json).write_text(rendered_json + "\n")
    Path(args.report).write_text(markdown_report(report))
    print(markdown_report(report))
    print(f"\nWrote {args.json} and {args.report}")


if __name__ == "__main__":
    main()
