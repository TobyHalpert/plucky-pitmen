"""What is a blast worth, and does choosing by backside reduce its risk?

Two questions that the league measurements in ``analyze_blasts.py`` cannot
answer, because they compare whole policies rather than single decisions:

1. ``--value``  From every position where a seat holds two dynamite and can
   move to a card, fork the game: announce the blast in one copy, stay quiet in
   the other, and play both out. The difference in that seat's final score is
   what the blast was worth *there*. Reported under both blast-choice engines:
   ``peek`` (the historical one, where the blaster sees all three drawn fronts)
   and the printed rule (choose blind, then look).
2. ``--choice`` Under the printed rule, does picking the keepers by backside
   knowledge beat picking them at random? Same movement and the same blast
   trigger, only the choice differs.

Run ``python probe_blast_value.py``; it prints both tables and needs no
checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import statistics

import numpy as np

from agents import ActionChoice, make_scripted_agent, preferred_blast_choice
from game_env import PASS, PIT_CAGE, MineEnv
from selfplay import _choice_for, play_game


class Committed:
    """Passes while a blast is pending, so the announcement is not cancelled.

    The rules abort a blast if the player later picks anything but "Pass" in
    the same planning phase, so a forked rollout has to commit to passing for
    the blast to fire at all.
    """

    def __init__(self, inner):
        self.inner = inner

    def decide(self, ctx):
        if ctx.own_blast_pending:
            return ActionChoice(PASS)
        return self.inner.decide(ctx)

    def __getattr__(self, name):
        return getattr(self.inner, name)



def value_of_blasting(games: int, players: int, blind: bool, seed0: int = 0):
    deltas: list[float] = []
    behind_flags: list[bool] = []
    fired = 0
    for g in range(games):
        env = MineEnv(
            opponents=players - 1,
            opponent_policy="simple",
            blast_choice_sees_fronts=not blind,
        )
        env.reset(seed=seed0 + g)
        agents = [make_scripted_agent("rule", seed=100 + 7 * g + i) for i in range(players)]
        steps = 0
        done = False
        while not env.game_finished and not done and steps < 5000:
            steps += 1
            if env.tie_breaker_contenders:
                break
            if any(p.blast_pending and p.blast_cards for p in env.players):
                break
            seat = env.planning_player
            ctx = env.player_action_context(seat)
            if not ctx.legal_actions:
                break
            if seat == 0 and env.players[0].dynamite >= 2 and not env.players[0].blast_pending:
                choice = _choice_for(agents[0], ctx)
                if 0 <= choice.action < PIT_CAGE:
                    own_gems = env.players[0].gems
                    best_other = max(p.gems for p in env.players[1:])
                    try:
                        env_a, agents_a = copy.deepcopy(env), copy.deepcopy(agents)
                        agents_a[0] = Committed(agents_a[0])
                        env_a.step_one_player(
                            choice.action, announce_blast=True, force_action=True
                        )
                        result_a = play_game(agents_a, seed=seed0 + g, env=env_a)
                        env_b, agents_b = copy.deepcopy(env), copy.deepcopy(agents)
                        env_b.step_one_player(choice.action, force_action=True)
                        result_b = play_game(agents_b, seed=seed0 + g, env=env_b)
                    except (ValueError, IndexError):
                        pass
                    else:
                        deltas.append(result_a.scores[0] - result_b.scores[0])
                        behind_flags.append(own_gems < best_other)
                        fired += sum(len(h.blasts) for h in result_a.hauls)
                        done = True
            choice = _choice_for(agents[seat], ctx)
            env.step_one_player(
                choice.action,
                announce_blast=choice.announce_blast,
                force_action=True,
            )
    return deltas, behind_flags, fired


def report_value(label: str, deltas, behind_flags, fired, games: int) -> None:
    print(f"\n{label}")
    if not deltas:
        print("  no positions found")
        return
    mean = statistics.mean(deltas)
    se = statistics.pstdev(deltas) / len(deltas) ** 0.5
    better = sum(d > 0 for d in deltas)
    worse = sum(d < 0 for d in deltas)
    print(
        f"  {len(deltas)} forked games ({fired / max(1, len(deltas)):.2f} blasts fired per fork), "
        f"{games} games scanned"
    )
    print(
        f"  blasting was worth {mean:+.2f} points (se {se:.2f}): "
        f"better in {better}, worse in {worse}, identical in {len(deltas) - better - worse}"
    )
    for flag, name in ((True, "behind on gems"), (False, "level or ahead")):
        subset = [d for d, f in zip(deltas, behind_flags) if f == flag]
        if subset:
            sub_se = statistics.pstdev(subset) / len(subset) ** 0.5
            print(f"    {name}: n={len(subset)} mean {statistics.mean(subset):+.2f} (se {sub_se:.2f})")


class RandomBlastChoice:
    """Same movement and blast trigger, but keeps blast cards at random."""

    def __init__(self, inner, seed: int):
        self.inner = inner
        self.rng = np.random.default_rng(seed)

    def decide(self, ctx):
        return self.inner.decide(ctx)

    def wants_blast(self, ctx):
        return self.inner.wants_blast(ctx)

    def blast_choice(self, cards, ctx):
        return int(self.rng.integers(len(cards)))

    def tie_break(self, ctx):
        return self.inner.tie_break(ctx)


class InformedBlastChoice(RandomBlastChoice):
    """Keeps the cards the backside bookkeeping likes best."""

    def blast_choice(self, cards, ctx):
        return preferred_blast_choice(cards, ctx)


def choice_experiment(players: int, games: int) -> None:
    print("\nIs backside knowledge worth anything when the choice is blind?")
    for label, factory in (
        ("informed choice", lambda g, i: InformedBlastChoice(
            make_scripted_agent("rule", seed=50 + g, blast="always"), seed=g)),
        ("random choice", lambda g, i: RandomBlastChoice(
            make_scripted_agent("rule", seed=50 + g, blast="always"), seed=g)),
        ("never blasts (control)", lambda g, i: make_scripted_agent(
            "rule", seed=50 + g, blast="never")),
    ):
        fired = woke = 0
        margins = []
        for g in range(games):
            agents = [factory(g, 0)] + [
                make_scripted_agent("rule", seed=1000 + 11 * g + i)
                for i in range(1, players)
            ]
            result = play_game(agents, seed=2000 + g, blast_choice_sees_fronts=False)
            fired += sum(len(h.blasts) for h in result.hauls)
            woke += sum(1 for h in result.hauls if h.outcome == "blast_dragon")
            others = [s for i, s in enumerate(result.scores) if i != 0]
            margins.append(result.scores[0] - sum(others) / len(others))
        margin = statistics.mean(margins)
        se = statistics.pstdev(margins) / len(margins) ** 0.5
        share = 100 * woke / max(1, fired)
        print(
            f"  {label:22s}: {fired / games:.2f} blasts/game, "
            f"{woke}/{fired} woke the dragon ({share:.0f}%), "
            f"seat margin {margin:+.2f} (se {se:.2f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=150)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-value", action="store_true")
    parser.add_argument("--skip-choice", action="store_true")
    args = parser.parse_args()

    if not args.skip_value:
        for blind in (False, True):
            deltas, behind, fired = value_of_blasting(
                args.games, args.players, blind, seed0=args.seed
            )
            report_value(
                "peek engine (the blaster sees the three drawn fronts)" if not blind
                else "printed rule (choose two blind, then look)",
                deltas, behind, fired, args.games,
            )
    if not args.skip_choice:
        choice_experiment(args.players, args.games)


if __name__ == "__main__":
    main()
