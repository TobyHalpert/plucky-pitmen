"""Do seats step onto a dragon they can see and get off again before the reveal?

The proposal: a blast that can be cancelled cheaply makes a known dragon
survivable, so a seat that knows it acts again in the same round should walk onto
the card and cancel later.  The engine's phase order decides whether that exists:

* the card under a figure is revealed in phase 2 of the round, and
  ``_prepare_blasts`` only runs when phase 2 woke nothing -- a pending blast is
  therefore *never* drawn against the card you are standing on (see
  ``test_dragon_step.py``);
* any move other than PASS cancels a pending blast, so a seat cannot both keep
  the blast and leave the card.

So the bluff and the shield are mutually exclusive, and the blast adds nothing to
a deliberate step.  What is left is the plain trap: step on, be gone before the
reveal.  This script measures whether that actually happens, and how it ends,
using the engine's own bookkeeping rather than a forked counterfactual (which
``probe_bait_value.py`` already reports).
"""

from __future__ import annotations

import argparse
from agents import NeuralAgent
from features import digs_face_up_dragon
from game_env import LORRY, MAX_COLUMNS, PIT_CAGE, MineEnv
from learning import PolicyNet
from selfplay import _choice_for

MAX_STEPS = 5000


class StepEnv(MineEnv):
    """MineEnv that snapshots known-dragon digs just before the reveal.

    Positions are reset by ``_set_players_to_starting_positions`` as soon as a
    round resolves, so the question "was the seat still on that card?" can only
    be answered before ``_execute_round`` runs -- not after the step returns.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decisions: dict[tuple[int, int, int], int] = {}
        self.open_digs: list[dict] = []
        self.digs: list[dict] = []

    def close_digs(self) -> None:
        for dig in self.open_digs:
            player = self.players[dig["seat"]]
            on_card = (player.position, player.column) == (
                dig["row"],
                dig["column"],
            )
            pushed = any(
                entry[0] == dig["seat"]
                for entry in self.displacement_events[dig["displacement_mark"]:]
            )
            if player.dead and on_card:
                fate = "eaten"
            elif player.dead:
                fate = "died_elsewhere"
            elif player.escaped:
                fate = "escaped"
            elif player.position in (PIT_CAGE, LORRY):
                fate = "at_exit"
            elif pushed:
                fate = "pushed_off"
            elif on_card:
                fate = "stood_on_it"
            else:
                fate = "walked_off"
            dig["fate"] = fate
            dig["acted_again"] = (
                self.decisions.get(
                    (dig["haul"], dig["round"], dig["seat"]), 0
                )
                - dig["decision"]
            )
        self.digs.extend(self.open_digs)
        self.open_digs = []

    def _execute_round(self, *args, **kwargs):
        if self.open_digs:
            self.close_digs()
        return super()._execute_round(*args, **kwargs)


def run(games: int, checkpoint: str, *, blind: bool) -> dict:
    net = PolicyNet.load(checkpoint)
    digs: list[dict] = []

    for seed in range(games):
        env = StepEnv(opponents=3, blast_choice_sees_fronts=not blind)
        env.reset(seed=seed)
        agents = [
            NeuralAgent(net, seed=seed * 11 + seat, temperature=0.7) for seat in range(4)
        ]
        steps = 0

        while not env.game_finished and steps < MAX_STEPS:
            steps += 1
            if env.tie_breaker_contenders:
                break

            # A blast that has drawn cards blocks the round until its keeper
            # names what to keep; without this the driver stalls and never
            # reaches the reveal it is trying to observe.
            pending = [
                index
                for index, player in enumerate(env.players)
                if player.blast_pending and player.blast_cards
            ]
            if pending:
                blaster = pending[0]
                blast_ctx = env.player_action_context(blaster)
                cards = tuple(blast_ctx.own_blast_cards)
                pick = int(agents[blaster].blast_choice(cards, blast_ctx))
                env.resolve_blast(blaster, min(max(pick, 0), len(cards) - 1))
                continue

            seat = env.planning_player
            ctx = env.player_action_context(seat)
            if not ctx.legal_actions:
                break
            choice = _choice_for(agents[seat], ctx)
            action = int(choice.action)
            key = (env.haul, env.round, seat)
            env.decisions[key] = env.decisions.get(key, 0) + 1

            for dig in env.open_digs:
                if dig["seat"] == seat and ctx.own_blast_pending:
                    dig["announced"] = True

            if digs_face_up_dragon(ctx, action) and action < PIT_CAGE:
                row, column = divmod(action, MAX_COLUMNS)
                dig = {
                    "seat": seat,
                    "haul": env.haul,
                    "round": env.round,
                    "row": row,
                    "column": column,
                    "decision": env.decisions[key],
                    "first_decision": env.decisions[key] == 1,
                    "announced": bool(choice.announce_blast),
                    "displacement_mark": len(env.displacement_events),
                }
                env.open_digs.append(dig)

            env.step_one_player(
                action,
                announce_blast=choice.announce_blast,
                force_action=True,
            )
            if env.open_digs and (env.haul, env.round) != (
                env.open_digs[0]["haul"],
                env.open_digs[0]["round"],
            ):
                env.close_digs()

        if env.open_digs:
            env.close_digs()
        digs.extend(env.digs)

    return summarise(digs, games)


def summarise(digs, games) -> dict:
    total = len(digs)
    out: dict = {"games": games, "digs": total, "digs_per_game": total / max(games, 1)}
    if not total:
        return out

    def share(predicate) -> float:
        return 100.0 * sum(1 for d in digs if predicate(d)) / total

    fates: dict[str, int] = {}
    for dig in digs:
        fates[dig["fate"]] = fates.get(dig["fate"], 0) + 1

    out["acted_again_pct"] = share(lambda d: d["acted_again"] > 0)
    out["announced_pct"] = share(lambda d: d["announced"])
    out["still_on_card_pct"] = share(lambda d: d["fate"] == "stood_on_it")
    out["eaten_pct"] = share(lambda d: d["fate"] == "eaten")
    out["left_card_pct"] = share(
        lambda d: d["fate"] in ("escaped", "at_exit", "pushed_off", "walked_off")
    )
    out["first_decision_pct"] = share(lambda d: d["first_decision"])
    out["fates"] = fates
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--checkpoint", default="checkpoints/selfplay4.npz")
    parser.add_argument("--blind", action="store_true",
                        help="play the printed face-down blast rule")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    import json

    summary = run(args.games, args.checkpoint, blind=args.blind)
    rule = "printed rule (blind blast fronts)" if args.blind else "engine default (fronts peeked)"
    print(f"\n{args.checkpoint} under the {rule}, {args.games} four-seat games")
    print(f"  digs of a face-up dragon   {summary['digs_per_game']:.2f} per game "
          f"({summary['digs']} in {args.games})")
    if summary["digs"]:
        print(f"  seat acted again that round {summary['acted_again_pct']:5.1f}%")
        print(f"  announced a blast somewhere {summary['announced_pct']:5.1f}%")
        print(f"  still on the card at reveal {summary['still_on_card_pct']:5.1f}%")
        print(f"  eaten by that card          {summary['eaten_pct']:5.1f}%")
        print(f"  off the card before reveal  {summary['left_card_pct']:5.1f}%")
        for fate, count in sorted(summary["fates"].items(), key=lambda kv: -kv[1]):
            print(f"    {fate:14s} {count:4d}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
