"""Does the learner actually read a backside when it picks a card?

A backside is the only public information about a face-down card, and it is
informative: the two dragons of each set cannot be drawn twice, so a backside
whose dragons are already face up is *diluted* (fewer dragons left in it) and
one still hiding both is *loaded*. ``rule_policy._estimate_dragon_risk`` turns
that into a per-backside dragon probability.

This script asks whether the network's choices move with that number. It walks
the learner's own games, and every time the learner is offered two or more
mine cards it records

* ``risk`` of the card it took, against the mean risk of the cards it could have
  taken (a negative gap means it prefers the diluted backside),
* whether it took the lowest-risk card that was offered, and
* the same numbers split by whether the card was in its own column (fronts
  visible) or a rival's (only the backside is visible, which is where the
  reading has to happen).
"""

from __future__ import annotations

import argparse

from agents import NeuralAgent, make_scripted_agent
from features import backside_stats
from game_env import MAX_COLUMNS, PIT_CAGE, MineEnv
from learning import PolicyNet
from selfplay import _choice_for


def collect(checkpoint: str, games: int, field: str) -> None:
    net = PolicyNet.load(checkpoint)
    gaps: list[float] = []
    foreign_gaps: list[float] = []
    own_gaps: list[float] = []
    foreign_best = 0
    foreign_chance = 0.0
    foreign_n = 0
    foreign_informative = 0
    for seed in range(games):
        env = MineEnv(opponents=3, opponent_policy="simple")
        env.reset(seed=seed)
        agents = [NeuralAgent(net, seed=seed, temperature=0.7)] + [
            make_scripted_agent(field, seed=seed * 7 + i) for i in range(3)
        ]
        steps = 0
        while not env.game_finished and steps < 5000:
            steps += 1
            if env.tie_breaker_contenders:
                break
            if any(p.blast_pending and p.blast_cards for p in env.players):
                break
            seat = env.planning_player
            ctx = env.player_action_context(seat)
            if not ctx.legal_actions:
                break
            choice = _choice_for(agents[seat], ctx)
            action = int(choice.action)
            if seat == 0 and 0 <= action < PIT_CAGE:
                _row, column = divmod(action, MAX_COLUMNS)
                risk = backside_stats(ctx)[3]
                cards = []
                for candidate in ctx.legal_actions:
                    if not 0 <= candidate < PIT_CAGE:
                        continue
                    c_row, c_column = divmod(int(candidate), MAX_COLUMNS)
                    if c_row >= len(ctx.public_rows) or c_column >= len(ctx.public_rows[c_row]):
                        continue
                    entry = ctx.public_rows[c_row][c_column]
                    if entry is None:
                        continue
                    front, backside = entry
                    if front is not None:  # known card: not a reading question
                        continue
                    cards.append((int(candidate), float(risk[backside])))
                if len(cards) >= 2:
                    chosen = [risk for candidate, risk in cards if candidate == action]
                    if chosen:
                        risks = [risk for _candidate, risk in cards]
                        mean_risk = sum(risks) / len(risks)
                        informative = max(risks) - min(risks) > 0.05
                        gap = chosen[0] - mean_risk
                        if informative:  # ties would dilute the gap to zero
                            gaps.append(gap)
                        if column != ctx.player_index:
                            if informative:
                                foreign_gaps.append(gap)
                            foreign_n += 1
                            if informative:
                                foreign_informative += 1
                                foreign_best += chosen[0] == min(risks)
                                foreign_chance += 1.0 / len(risks)
                        elif informative:
                            own_gaps.append(gap)
            env.step_one_player(
                choice.action,
                announce_blast=choice.announce_blast,
                force_action=True,
            )

    def line(label: str, values: list[float], n: int | None = None) -> None:
        if not values:
            print(f"  {label:18s} -")
            return
        mean = sum(values) / len(values)
        print(f"  {label:18s} n={len(values):5d}  mean risk gap {mean:+.3f}")
        del n

    print(f"\n{checkpoint} vs 3x{field}, {games} games")
    line("all card choices", gaps)
    line("rival-column choice", foreign_gaps)
    line("own-column choice", own_gaps)
    print(
        f"  informative choices {foreign_informative} where the backsides differ: "
        f"lowest-risk taken {100 * foreign_best / max(1, foreign_informative):.0f}% "
        f"(chance {100 * foreign_chance / max(1, foreign_informative):.0f}%)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--games", type=int, default=150)
    parser.add_argument("--field", default="rule")
    args = parser.parse_args()
    for checkpoint in args.checkpoints:
        collect(checkpoint, args.games, args.field)


if __name__ == "__main__":
    main()
