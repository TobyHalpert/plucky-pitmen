"""Train self-improving Plucky Pitmen agents with league self-play.

Example
-------
``python train_selfplay.py --generations 80 --games-per-generation 64``

Each game puts the learner in one rotating seat and fills the other seats from
a league of opponents: the hand-tuned rule policy, simple baselines and older
snapshots of the learner itself.  A fraction of games is pure mirror self-play.
The learner is updated with REINFORCE + value baseline on the cheat-safe
features from ``features.py``.

Use ``--eval-games 0`` to skip the periodic benchmark, and ``--out`` to pick
where the checkpoint and learning curve are written.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from agents import NeuralAgent, make_scripted_agent
from learning import PolicyNet
from selfplay import play_game, play_series

SCRIPTED_POOL = ("rule", "rule", "rule", "greedy", "random", "cautious")


def _scripted_factory(policy: str, blast: str = "never"):
    return lambda seed: make_scripted_agent(policy, seed=seed, blast=blast)


def _neural_factory(net: PolicyNet, *, temperature: float = 0.7, seed_offset: int = 0):
    def factory(seed: int):
        return NeuralAgent(
            net,
            seed=seed + seed_offset,
            temperature=temperature,
            name=net.name,
        )

    return factory


def evaluate(
    net: PolicyNet,
    *,
    games: int,
    players: int,
    seed: int,
    opponent: str,
    deterministic: bool = False,
    passes: str = "n-1",
    solo_escape_bonus: bool = True,
    blast_choice_sees_fronts: bool = True,
) -> dict[str, float]:
    """Benchmark the learner against a fixed scripted opponent."""
    if games <= 0:
        return {}
    learner_factory = lambda s: NeuralAgent(  # noqa: E731
        net,
        seed=s,
        deterministic=deterministic,
        temperature=0.7,
        name="learner",
    )
    factories = [learner_factory] + [
        _scripted_factory(opponent) for _ in range(players - 1)
    ]
    series = play_series(
        factories,
        games,
        seed=seed,
        rotate_seats=True,
        passes=passes,
        solo_escape_bonus=solo_escape_bonus,
        blast_choice_sees_fronts=blast_choice_sees_fronts,
    )
    wins = 0
    top_scores = 0
    scores = []
    ranks = []
    dragons = 0
    stalls = 0
    for game in series:
        policy_scores = game.scores_by_policy()
        learner_score = policy_scores[0]
        scores.append(learner_score)
        wins += int(game.winner_policy() == 0)
        top_scores += int(learner_score == max(policy_scores))
        ranks.append(1 + sum(1 for score in policy_scores if score > learner_score))
        dragons += game.result.dragons
    stalls += game.result.lorry_standoffs
    return {
        "win_rate": wins / games,
        "top_score_rate": top_scores / games,
        "mean_score": float(np.mean(scores)),
        "mean_rank": float(np.mean(ranks)),
        "dragons_per_game": dragons / games,
        "lorry_standoffs_per_game": stalls / games,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)

    if args.resume:
        net = PolicyNet.load(args.resume, lr=args.lr)
        net.name = args.name
    else:
        net = PolicyNet(
            hidden=args.hidden,
            value_hidden=args.value_hidden,
            seed=args.seed,
            lr=args.lr,
            name=args.name,
        )

    league: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    total_games = 0

    for generation in range(1, args.generations + 1):
        decisions: list[Any] = []
        generation_results = []
        for game in range(args.games_per_generation):
            game_seed = args.seed * 1_000_003 + generation * 10_007 + game
            mirror = bool(rng.random() < args.mirror_prob)
            learner_seats = (
                list(range(args.players))
                if mirror
                else [(generation * args.games_per_generation + game) % args.players]
            )
            recorder: list[Any] = []
            agents: list[Any] = []
            for seat in range(args.players):
                if seat in learner_seats:
                    agents.append(
                        NeuralAgent(
                            net,
                            seed=game_seed + seat,
                            temperature=args.temperature,
                            epsilon=args.epsilon,
                            recorder=recorder,
                        )
                    )
                elif league and rng.random() < args.league_prob:
                    snapshot = league[int(rng.integers(len(league)))]
                    agents.append(
                        NeuralAgent(
                            snapshot["net"],
                            seed=game_seed + seat,
                            temperature=0.7,
                            name=snapshot["name"],
                        )
                    )
                else:
                    policy = SCRIPTED_POOL[int(rng.integers(len(SCRIPTED_POOL)))]
                    blast = (
                        args.league_blast_mode
                        if args.league_blast_mode != "never"
                        and rng.random() < args.league_blast_prob
                        else "never"
                    )
                    agents.append(
                        make_scripted_agent(
                            policy, seed=game_seed + seat, blast=blast
                        )
                    )

            result = play_game(
                agents,
                seed=game_seed,
                decisions=recorder,
                passes=args.pass_rule,
                solo_escape_bonus=args.solo_escape_bonus,
                escape_shaping=args.escape_shaping,
                blast_choice_sees_fronts=not args.blind_blast_choice,
            )
            generation_results.append(result)
            decisions.extend(recorder)
            total_games += 1

        stats = net.fit(
            decisions,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            rng=rng,
        )
        net.trained_generations += 1

        entry: dict[str, Any] = {
            "generation": generation,
            "games": total_games,
            "decisions": len(decisions),
            "mean_score": float(np.mean([np.mean(r.scores) for r in generation_results])),
            "dragons": float(np.mean([r.dragons for r in generation_results])),
            "blasts": float(np.mean([r.blasts_announced for r in generation_results])),
            "aborted": sum(r.aborted for r in generation_results),
            **{f"train_{key}": float(value) for key, value in stats.items()},
        }

        if args.eval_games > 0 and (
            generation % args.eval_every == 0 or generation == args.generations
        ):
            evaluation = evaluate(
                net,
                games=args.eval_games,
                players=args.players,
                seed=args.eval_seed,
                opponent=args.eval_opponent,
                deterministic=args.deterministic_eval,
                passes=args.pass_rule,
                solo_escape_bonus=args.solo_escape_bonus,
                blast_choice_sees_fronts=not args.blind_blast_choice,
            )
            entry.update({f"eval_{key}": value for key, value in evaluation.items()})
            print(
                f"gen {generation:4d} | "
                f"win {evaluation['win_rate']:.3f} | "
                f"top {evaluation['top_score_rate']:.3f} | "
                f"score {evaluation['mean_score']:+.2f} | "
                f"rank {evaluation['mean_rank']:.2f} | "
                f"dragon {evaluation['dragons_per_game']:.2f} | "
                f"lorry {evaluation['lorry_standoffs_per_game']:.2f} | "
                f"entropy {stats['entropy']:.3f} | "
                f"blasts {entry['blasts']:.2f}",
                flush=True,
            )
        elif generation % max(1, args.log_every) == 0:
            print(
                f"gen {generation:4d} | decisions {len(decisions)} | "
                f"score {entry['mean_score']:+.2f} | "
                f"entropy {stats['entropy']:.3f} | "
                f"blasts {entry['blasts']:.2f}",
                flush=True,
            )

        history.append(entry)

        if (
            args.snapshot_every > 0
            and generation % args.snapshot_every == 0
        ):
            snapshot = net.clone(name=f"{args.name}-gen{generation}")
            snapshot.trained_generations = generation
            snapshot.metadata = {"generation": generation}
            directory = out.parent / "league"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{args.name}-gen{generation:04d}.npz"
            snapshot.save(path)
            league.append({"name": snapshot.name, "net": snapshot, "path": str(path)})
            while len(league) > args.league_size:
                league.pop(0)

        if generation % args.save_every == 0:
            net.save(out)

    net.save(out)
    curve_path = out.with_suffix(".history.json")
    curve_path.write_text(json.dumps(history, indent=2))

    elapsed = time.time() - started
    summary = {
        "checkpoint": str(out),
        "history": str(curve_path),
        "generations": net.trained_generations,
        "games": total_games,
        "seconds": round(elapsed, 1),
        "final": history[-1] if history else None,
    }
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--games-per-generation", type=int, default=64)
    parser.add_argument("--players", type=int, choices=(3, 4, 5), default=3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--value-hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--mirror-prob", type=float, default=0.25)
    parser.add_argument("--league-prob", type=float, default=0.5)
    parser.add_argument("--league-size", type=int, default=8)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=120)
    parser.add_argument("--eval-seed", type=int, default=90_000_000)
    parser.add_argument("--eval-opponent", default="rule")
    parser.add_argument("--deterministic-eval", action="store_true", default=False)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--blind-blast-choice",
        action="store_true",
        help="choose blast cards face down, as the printed rules say",
    )
    parser.add_argument("--pass-rule", choices=("n-1", "n"), default="n-1")
    parser.add_argument(
        "--league-blast-mode",
        choices=("never", "rich", "always", "trap", "behind"),
        default="rich",
        help="how often a scripted league opponent blasts, when it does blast",
    )
    parser.add_argument(
        "--league-blast-prob",
        type=float,
        default=0.15,
        help="share of scripted league opponents that blast at all",
    )
    parser.add_argument(
        "--escape-shaping",
        type=float,
        default=0.3,
        help="dense reward for leaving the mine safely (set 0 to remove)",
    )
    parser.add_argument(
        "--solo-escape-bonus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="award the haul bonus to a sole escapee even without gems",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--name", default="selfplay")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--out", default="checkpoints/selfplay.npz")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
