"""Paired four-player intervention test for blast commitment and frequency.

Each game places one instrumented focal ``selfplay4`` agent among three
unchanged copies. The focal seat rotates evenly and every arm uses the same
initial game seeds, policy RNG seeds, and focal seats. Trajectories can still
diverge after an intervention changes play; this is a paired-seed policy test,
not a same-position counterfactual.

Arms separate three effects:

* ``natural`` preserves the learned announcement and cancellation decisions;
* ``commit_natural`` leaves its trigger untouched but passes on every pending
  blast, isolating the effect of completing blasts the policy would otherwise
  cancel;
* ``extra_*`` randomly forces announcements on eligible card moves and passes
  while a blast is pending. The movement action is not changed. ``force_card_moves``
  announces on every such eligible move.

An announcement is not counted as a completed blast unless the environment
actually draws the three cards. The report records announcements, cancellations,
completed draws, score margins, and dragon outcomes separately.

Example::

    PYTHONPATH=. OMP_NUM_THREADS=1 python probe_blast_frequency.py \\
        --games 800 --seed 8400000 --out /tmp/blast_frequency.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from agents import ActionChoice, NeuralAgent
from game_env import DYNAMITE, PASS, PIT_CAGE, MineEnv
from learning import PolicyNet
from selfplay import GameResult, play_game

ARMS = (
    ("natural", 0.0, False),
    ("commit_natural", 0.0, True),
    ("extra_25pct", 0.25, True),
    ("extra_50pct", 0.50, True),
    ("force_card_moves", 1.0, True),
)


class CountedMineEnv(MineEnv):
    """Count announcements, cancellations, and completed draws by seat."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.blast_counts: dict[str, Any] = {
            "announced": [0] * self.n_players,
            "cancelled": [0] * self.n_players,
            "fired": [0] * self.n_players,
        }

    def begin_blast(self, player_index: int) -> None:
        was_pending = self.players[player_index].blast_pending
        super().begin_blast(player_index)
        if not was_pending and self.players[player_index].blast_pending:
            self.blast_counts["announced"][player_index] += 1

    def _cancel_blast(self, player_index: int) -> None:
        if self.players[player_index].blast_pending:
            self.blast_counts["cancelled"][player_index] += 1
        super()._cancel_blast(player_index)

    def _prepare_blasts(self) -> None:
        pending = [player.blast_pending for player in self.players]
        super()._prepare_blasts()
        for index, (was_pending, player) in enumerate(zip(pending, self.players)):
            # A real blast needs three cards to be drawn. If the deck is short,
            # the engine returns the available cards and does not resolve a blast.
            if was_pending and player.blast_cards:
                self.blast_counts["fired"][index] += 1


class BlastFrequencyAgent(NeuralAgent):
    """Neural policy with a stochastic trigger and optional forced commitment."""

    def __init__(
        self,
        net: PolicyNet,
        *,
        seed: int,
        force_rate: float = 0.0,
        force_seed: int = 0,
        commit_pending: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(net, seed=seed, temperature=0.7, **kwargs)
        self.force_rate = float(force_rate)
        self.force_rng = np.random.default_rng(force_seed)
        self.commit_pending = bool(commit_pending)
        self.eligible = 0
        self.selected = 0
        self.forced = 0
        self.pending_decisions = 0
        self.policy_would_cancel = 0
        self.commit_overrides = 0

    def decide(self, ctx):
        if ctx.own_blast_pending:
            self.pending_decisions += 1
            # Evaluate the policy even in a commit arm. This keeps its random
            # stream aligned with the natural arm at the intervention point.
            natural_choice = super().decide(ctx)
            would_cancel = int(int(natural_choice.action) != PASS)
            self.policy_would_cancel += would_cancel
            if self.commit_pending and PASS in ctx.legal_actions:
                self.commit_overrides += would_cancel
                return ActionChoice(PASS)
            return natural_choice

        has_pair = sum(front == DYNAMITE for front, _ in ctx.private_cards) >= 2
        has_card_move = any(0 <= int(action) < PIT_CAGE for action in ctx.legal_actions)
        if has_pair and has_card_move:
            self.eligible += 1

        choice = super().decide(ctx)
        if has_pair and has_card_move and 0 <= int(choice.action) < PIT_CAGE:
            if (
                not choice.announce_blast
                and self.force_rate > 0
                and self.force_rng.random() < self.force_rate
            ):
                choice = ActionChoice(int(choice.action), announce_blast=True)
                self.forced += 1
            if choice.announce_blast:
                self.selected += 1
        return choice


def _play(
    net: PolicyNet,
    game_seed: int,
    *,
    focal_seat: int,
    force_rate: float,
    commit_pending: bool,
) -> tuple[GameResult, CountedMineEnv, BlastFrequencyAgent]:
    env = CountedMineEnv(opponents=3, blast_choice_sees_fronts=False)
    env.reset(seed=game_seed)
    agents: list[NeuralAgent] = []
    focal: BlastFrequencyAgent | None = None
    for seat in range(4):
        policy_seed = game_seed * 31 + seat
        if seat == focal_seat:
            focal = BlastFrequencyAgent(
                net,
                seed=policy_seed,
                force_seed=game_seed * 97 + seat,
                force_rate=force_rate,
                commit_pending=commit_pending,
            )
            agents.append(focal)
        else:
            agents.append(NeuralAgent(net, seed=policy_seed, temperature=0.7))
    assert focal is not None
    result = play_game(
        agents,
        seed=game_seed,
        env=env,
        blast_choice_sees_fronts=False,
    )
    return result, env, focal


def _game_metrics(
    result: GameResult,
    env: CountedMineEnv,
    focal: BlastFrequencyAgent,
    focal_seat: int,
) -> dict[str, Any]:
    own_score = float(result.scores[focal_seat])
    opponents = [
        float(score)
        for seat, score in enumerate(result.scores)
        if seat != focal_seat
    ]
    best_score = max(result.scores)
    top_count = sum(score == best_score for score in result.scores)
    if focal.selected != env.blast_counts["announced"][focal_seat]:
        raise RuntimeError("focal announcement count disagrees with the engine")
    return {
        "seed": result.seed,
        "focal_seat": focal_seat,
        "score": own_score,
        "margin": own_score - statistics.mean(opponents),
        "score_spread": float(max(result.scores) - min(result.scores)),
        "top_share": float(own_score == best_score) / top_count,
        "dragon_hauls": sum("dragon" in haul.outcome for haul in result.hauls),
        "blast_dragon_hauls": sum(
            haul.outcome == "blast_dragon" for haul in result.hauls
        ),
        "focal_dragon_death_hauls": sum(
            focal_seat in haul.dead and "dragon" in haul.outcome
            for haul in result.hauls
        ),
        "focal_blast_death_hauls": sum(
            focal_seat in haul.dead and haul.outcome == "blast_dragon"
            for haul in result.hauls
        ),
        "other_blast_death_hauls": sum(
            sum(seat != focal_seat for seat in haul.dead)
            for haul in result.hauls
            if haul.outcome == "blast_dragon"
        ),
        "eligible_decisions": focal.eligible,
        "announcements": focal.selected,
        "forced_announcements": focal.forced,
        "completed_blasts": env.blast_counts["fired"][focal_seat],
        "engine_cancellations": env.blast_counts["cancelled"][focal_seat],
        "pending_decisions": focal.pending_decisions,
        "policy_would_cancel": focal.policy_would_cancel,
        "commit_overrides": focal.commit_overrides,
        "aborted": result.aborted,
    }


def run_focal(
    net: PolicyNet,
    *,
    games: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Run all intervention arms with the same seeds and rotating focal seats."""
    if games < 4 or games % 4:
        raise ValueError("--games must be a positive multiple of four")

    raw: dict[str, list[dict[str, Any]]] = {}
    for label, force_rate, commit_pending in ARMS:
        rows: list[dict[str, Any]] = []
        for offset in range(games):
            game_seed = seed + offset
            focal_seat = offset % 4
            result, env, focal = _play(
                net,
                game_seed,
                focal_seat=focal_seat,
                force_rate=force_rate,
                commit_pending=commit_pending,
            )
            rows.append(_game_metrics(result, env, focal, focal_seat))
        raw[label] = rows

    # All games must complete: aborted runs cannot support score comparisons.
    aborted = {
        label: sum(bool(row["aborted"]) for row in rows)
        for label, rows in raw.items()
    }
    if any(aborted.values()):
        raise RuntimeError(f"aborted games in the evaluation: {aborted}")

    report: dict[str, Any] = {}
    for label, _rate, _commit in ARMS:
        rows = raw[label]
        report[label] = {
            "games": len(rows),
            "mean_score": float(np.mean([row["score"] for row in rows])),
            "mean_margin": float(np.mean([row["margin"] for row in rows])),
            "margin_sd": float(np.std([row["margin"] for row in rows], ddof=1)),
            "mean_game_score_spread": float(
                np.mean([row["score_spread"] for row in rows])
            ),
            "focal_top_share_pct": 100.0
            * float(np.mean([row["top_share"] for row in rows])),
            "eligible_decisions": int(sum(row["eligible_decisions"] for row in rows)),
            "announcements": int(sum(row["announcements"] for row in rows)),
            "announcements_per_game": float(
                np.mean([row["announcements"] for row in rows])
            ),
            "forced_announcements": int(
                sum(row["forced_announcements"] for row in rows)
            ),
            "completed_blasts": int(sum(row["completed_blasts"] for row in rows)),
            "completed_blasts_per_game": float(
                np.mean([row["completed_blasts"] for row in rows])
            ),
            "engine_cancellations": int(
                sum(row["engine_cancellations"] for row in rows)
            ),
            "engine_cancellations_per_game": float(
                np.mean([row["engine_cancellations"] for row in rows])
            ),
            "policy_would_cancel": int(
                sum(row["policy_would_cancel"] for row in rows)
            ),
            "commit_overrides": int(sum(row["commit_overrides"] for row in rows)),
            "dragon_hauls_per_game": float(
                np.mean([row["dragon_hauls"] for row in rows])
            ),
            "blast_dragon_hauls_per_game": float(
                np.mean([row["blast_dragon_hauls"] for row in rows])
            ),
            "focal_dragon_death_hauls_per_game": float(
                np.mean([row["focal_dragon_death_hauls"] for row in rows])
            ),
            "focal_blast_death_hauls_per_game": float(
                np.mean([row["focal_blast_death_hauls"] for row in rows])
            ),
            "other_blast_death_hauls_per_game": float(
                np.mean([row["other_blast_death_hauls"] for row in rows])
            ),
        }

    comparisons: dict[str, Any] = {}
    for label, *_ in ARMS:
        if label == "natural":
            continue
        comparisons[label] = {
            "vs_natural": _paired_comparison(raw[label], raw["natural"], seed),
            "vs_commit_natural": (
                None
                if label == "commit_natural"
                else _paired_comparison(raw[label], raw["commit_natural"], seed + 10)
            ),
        }
    return raw, {"arms": report, "paired_comparisons": comparisons, "aborted_games": aborted}


def _bootstrap_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int = 10_000,
) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    means = np.empty(resamples, dtype=float)
    batch_size = 256
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return [float(lower), float(upper)]


def _bootstrap_sd_delta_ci(
    candidate: np.ndarray,
    control: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int = 10_000,
) -> list[float]:
    deltas = np.empty(resamples, dtype=float)
    batch_size = 128
    n = len(candidate)
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = rng.integers(0, n, size=(stop - start, n))
        deltas[start:stop] = (
            candidate[indices].std(axis=1, ddof=1)
            - control[indices].std(axis=1, ddof=1)
        )
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    return [float(lower), float(upper)]


def _paired_comparison(
    candidate_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    bootstrap_seed: int,
) -> dict[str, Any]:
    if len(candidate_rows) != len(control_rows):
        raise ValueError("paired arms have different lengths")
    for candidate, control in zip(candidate_rows, control_rows):
        if (candidate["seed"], candidate["focal_seat"]) != (
            control["seed"],
            control["focal_seat"],
        ):
            raise ValueError("paired arms do not share seed and focal seat")

    rng = np.random.default_rng(bootstrap_seed)
    metrics = (
        "margin",
        "score",
        "score_spread",
        "blast_dragon_hauls",
        "focal_blast_death_hauls",
        "other_blast_death_hauls",
    )
    output: dict[str, Any] = {}
    for metric in metrics:
        candidate = np.asarray([row[metric] for row in candidate_rows], dtype=float)
        control = np.asarray([row[metric] for row in control_rows], dtype=float)
        differences = candidate - control
        output[metric] = {
            "mean_delta": float(differences.mean()),
            "paired_se": float(differences.std(ddof=1) / np.sqrt(len(differences))),
            "bootstrap_95_ci": _bootstrap_ci(differences, rng=rng),
        }
    candidate_margins = np.asarray(
        [row["margin"] for row in candidate_rows], dtype=float
    )
    control_margins = np.asarray([row["margin"] for row in control_rows], dtype=float)
    output["margin_sd"] = {
        "candidate": float(candidate_margins.std(ddof=1)),
        "control": float(control_margins.std(ddof=1)),
        "delta": float(
            candidate_margins.std(ddof=1) - control_margins.std(ddof=1)
        ),
        "bootstrap_95_ci": _bootstrap_sd_delta_ci(
            candidate_margins, control_margins, rng=rng
        ),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/selfplay4.npz")
    parser.add_argument("--games", type=int, default=800)
    parser.add_argument("--seed", type=int, default=8_400_000)
    parser.add_argument("--out", default=None, help="write machine-readable JSON here")
    args = parser.parse_args()

    net = PolicyNet.load(args.checkpoint)
    _raw, report = run_focal(net, games=args.games, seed=args.seed)
    checkpoint_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    output = {
        "experiment": "single focal seat blast frequency and commitment",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": checkpoint_hash,
        "players": 4,
        "rules": {
            "passes": "n-1",
            "solo_escape_bonus": True,
            "blast_choice_sees_fronts": False,
            "temperature": 0.7,
            "opponents": "three unchanged copies of the checkpoint policy",
            "intervention": "focal seat only; rotates evenly across seats",
        },
        "seed": args.seed,
        "games_per_arm": args.games,
        "confidence_intervals": "95% percentile bootstrap over paired whole-game differences; 10,000 resamples",
        **report,
    }
    serialized = json.dumps(output, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
