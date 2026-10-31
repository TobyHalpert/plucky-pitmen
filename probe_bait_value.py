"""What is a move onto a dragon the player can see worth? Forked, per opponent.

The learner announces a move onto a dragon it can see in its own column about
0.4 times a game. Two stories explain that: it is a *blunder* (nothing can save
you from the card under your figure), or it is a *trap* (any rival may displace
you, and a displaced figure is moved to the nearest pit cage, i.e. out of the
mine safely with its loot, while the rival takes the dragon).

Which story is true depends on the opponent, so this script forks every such
decision once per game, in one copy plays the move and in the other makes the
same policy pick again with known-dragon digs removed, then plays both out.
The difference in the seat's final score is what the move was worth *there*.
"""

from __future__ import annotations

import argparse
import copy
import statistics

from agents import ActionChoice, NeuralAgent, make_scripted_agent
from features import (
    action_mask,
    digs_face_up_dragon,
    join_action,
    rival_can_push,
    state_action_matrix,
    state_features,
)
from game_env import PASS, MineEnv
from learning import PolicyNet
from selfplay import _choice_for, play_game

TRACKED = 0


class NoBait(NeuralAgent):
    """The same network, but it may not dig a dragon it can already see."""

    def decide(self, ctx):
        mask = action_mask(ctx)
        for action in ctx.legal_actions:
            if digs_face_up_dragon(ctx, action):
                mask[action] = False
                mask[join_action(int(action), True)] = False
        if not mask.any():
            return ActionChoice(int(PASS))
        state = state_features(ctx)
        matrix = state_action_matrix(ctx, state=state)
        action, _logp, _value = self.net.select_action(
            matrix,
            mask,
            self.rng,
            deterministic=self.deterministic,
            temperature=self.temperature,
            epsilon=self.epsilon,
        )
        return ActionChoice(int(action))


def field_agents(field: str, net: PolicyNet, seed: int, fresh_seat0: bool):
    if field == "selfplay4":
        seats = [NeuralAgent(net, seed=seed * 11 + i, temperature=0.7) for i in range(4)]
    else:
        seats = [None] + [
            make_scripted_agent(field, seed=seed * 7 + i) for i in range(1, 4)
        ]
    seats[0] = NeuralAgent(net, seed=seed if fresh_seat0 else seed * 11, temperature=0.7)
    return seats


def run(field: str, games: int, checkpoint: str) -> None:
    net = PolicyNet.load(checkpoint)
    deltas: list[float] = []
    pushable: list[bool] = []
    taken: list[bool] = []
    eligible = 0
    for seed in range(games):
        env = MineEnv(opponents=3, opponent_policy="simple")
        env.reset(seed=seed)
        agents = field_agents(field, net, seed, fresh_seat0=True)
        steps = 0
        forked = False
        while not env.game_finished and steps < 5000 and not forked:
            steps += 1
            if env.tie_breaker_contenders:
                break
            if any(p.blast_pending and p.blast_cards for p in env.players):
                break
            seat = env.planning_player
            ctx = env.player_action_context(seat)
            if not ctx.legal_actions:
                break
            if seat == TRACKED and not env.players[0].blast_pending:
                choice = _choice_for(agents[0], ctx)
                action = int(choice.action)
                if digs_face_up_dragon(ctx, action):
                    eligible += 1
                    alternative = NoBait(
                        net, seed=9999 + seed, temperature=0.7
                    ).decide(ctx)
                    try:
                        env_a = copy.deepcopy(env)
                        agents_a = copy.deepcopy(agents)
                        env_a.step_one_player(action, force_action=True)
                        result_a = play_game(agents_a, seed=seed, env=env_a)
                        env_b = copy.deepcopy(env)
                        agents_b = copy.deepcopy(agents)
                        env_b.step_one_player(
                            int(alternative.action), force_action=True
                        )
                        result_b = play_game(agents_b, seed=seed, env=env_b)
                    except (ValueError, IndexError):
                        pass
                    else:
                        deltas.append(float(result_a.scores[0] - result_b.scores[0]))
                        pushable.append(rival_can_push(ctx, action))
                        taken.append(
                            any(haul.outcome == "dragon" for haul in result_a.hauls)
                        )
                        forked = True
                        continue
            choice = _choice_for(agents[seat], ctx)
            env.step_one_player(
                choice.action,
                announce_blast=choice.announce_blast,
                force_action=True,
            )

    def report(label: str, values: list[float]) -> None:
        if not values:
            print(f"  {label:14s} -")
            return
        mean = statistics.mean(values)
        se = statistics.pstdev(values) / len(values) ** 0.5
        wins = sum(v > 0 for v in values)
        losses = sum(v < 0 for v in values)
        print(
            f"  {label:14s} n={len(values):4d}  worth {mean:+6.2f} (se {se:.2f})"
            f"  better {100 * wins / len(values):3.0f}%  worse {100 * losses / len(values):3.0f}%"
        )

    print(f"\n{field}: {len(deltas)} forks from {eligible} bait announcements in {games} games")
    report("all", deltas)
    report("rival can push", [d for d, p in zip(deltas, pushable) if p])
    report("no rival can push", [d for d, p in zip(deltas, pushable) if not p])
    if taken:
        print(f"  the forked game woke a dragon {100 * sum(taken) / len(taken):.0f}% of the time")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--checkpoint", default="checkpoints/selfplay4.npz")
    parser.add_argument("--fields", default="rule,selfplay4,reader")
    args = parser.parse_args()
    for field in args.fields.split(","):
        run(field, args.games, args.checkpoint)


if __name__ == "__main__":
    main()
