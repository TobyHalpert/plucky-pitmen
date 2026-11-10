"""How often does a seat park on its lorry, and what does a lorry haul look like?

Standing on a lorry collects nothing and does **not** protect the figure: a woken
dragon kills every player still in the mine, the ones on their lorry included
(only a pit cage removes a figure from the haul). So the lorry has exactly one
use — refusing the card under your own figure — and that is only right when that
card is a dragon the seat can see.

Sampled at temperature 0.7 the network parked constantly, and because each seat
dodges for itself the whole table could park in one planning phase, which ends
the haul with nobody having collected anything. This script reports both halves:
how often the tracked seat parks (per field), and how often a haul ends in that
stand-off (with the checkpoint in every seat, which is what the viewer shows).

    python probe_lorry.py                       # the shipped checkpoint
    python probe_lorry.py --checkpoint other.npz --games 400
"""

from __future__ import annotations

import argparse

from agents import NeuralAgent, make_scripted_agent
from game_env import LORRY, MineEnv
from learning import PolicyNet
from selfplay import _choice_for, play_game

TRACKED = 0


def parked_by_tracked_seat(
    net: PolicyNet, field: str, games: int
) -> tuple[float, float]:
    """Parking moves the tracked seat makes per game, and in a first phase."""
    parked = 0
    first_phase = 0
    for seed in range(games):
        env = MineEnv(opponents=3, opponent_policy="simple")
        env.reset(seed=seed)
        agents = [NeuralAgent(net, seed=seed, temperature=0.7)] + [
            make_scripted_agent(field, seed=seed * 7 + i) for i in range(1, 4)
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
            if seat == TRACKED and int(choice.action) == LORRY:
                parked += 1
                first_phase += env.haul == 1 and env.round == 0
            env.step_one_player(
                choice.action,
                announce_blast=choice.announce_blast,
                force_action=True,
            )
    return parked / games, first_phase / games


def standoffs_in_a_peer_field(
    net: PolicyNet, games: int, greedy: bool = False
) -> tuple[float, float]:
    """Share of games whose first haul / any haul ends in a lorry stand-off."""
    first = 0
    any_haul = 0
    for seed in range(games):
        agents = [
            NeuralAgent(
                net,
                seed=seed + 101 * i,
                temperature=0.7,
                deterministic=greedy,
            )
            for i in range(4)
        ]
        result = play_game(agents, seed=seed)
        if result.hauls and result.hauls[0].outcome == "lorry":
            first += 1
        any_haul += any(haul.outcome == "lorry" for haul in result.hauls)
    return 100 * first / games, 100 * any_haul / games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/selfplay4.npz")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--fields", default="rule,reader")
    args = parser.parse_args()
    net = PolicyNet.load(args.checkpoint)

    print(f"{args.checkpoint}, 4 players, temperature 0.7")
    for field in args.fields.split(","):
        per_game, first_phase = parked_by_tracked_seat(net, field, args.games)
        print(
            f"  seat 0 parks {per_game:.2f}/game against 3 x {field:7s}"
            f"  (first planning phase of a haul: {first_phase:.2f}/game)"
        )

    first, any_haul = standoffs_in_a_peer_field(net, args.games)
    print(f"  the network in all four seats: first haul is a stand-off in {first:.1f}% of games")
    print(f"  ... any haul is a stand-off in {any_haul:.1f}%")


if __name__ == "__main__":
    main()
