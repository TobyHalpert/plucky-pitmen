"""How often does a table throw a haul away by all fleeing into pit cages?

The pit cage is the only exit that takes a figure out of the haul before the
cards are turned *and* banks the gems in hand, so it is worth a round when there
is loot to bank or something to escape from. The trap is the obvious
implementation of "something to escape from": treating **"somebody is already in
a cage"** as a threat. Ordinary raiding displaces rivals into cages, so one seat
reads another's cage as danger, the next reads two, and the whole table is in a
cage before a single card is turned. Seed 7 of the viewer, four nets, lost haul 1
that way with nothing collected, and the two remaining hauls then decide a game
of three rounds by luck.

This script measures both halves: how often a haul is thrown away (a haul where
at least three seats escaped and the table collected no gems), and what the
escape rule is worth against scripted fields. It A/Bs three predicates:

    shipped   the rule as first written: no blast, nobody out, no rival on our
              dragon  ->  flee
    no-herd   the same, minus "somebody is out/caged" as a reason
    banks     (in the library now) an escape must bank a gem or answer a
              visible threat

    python probe_escape.py                     # degenerate hauls, 4 nets
    python probe_escape.py --field rule        # strength A/B against 3 x rule
"""

from __future__ import annotations

import argparse
import collections

import features
from agents import NeuralAgent, make_scripted_agent
from game_env import DRAGON, PIT_CAGE
from gui_controller import GameController
from learning import PolicyNet
from selfplay import play_game

SHIPPED = "shipped"
NO_HERD = "no-herd"
ESCAPED = "escaped"
BANKS = "banks"


def _visible_dragon_threat(ctx) -> bool:
    """A dragon the seat can see is about to be turned: its own card, or a rival on its column."""
    row = ctx.own_position
    if 0 <= row < len(ctx.public_rows) and ctx.player_index < len(ctx.public_rows[row]):
        entry = ctx.public_rows[row][ctx.player_index]
        if entry is not None and entry[0] is not None and int(entry[0]) == DRAGON:
            return True
    for opponent in ctx.opponents:
        if opponent.dead or opponent.escaped or opponent.position == PIT_CAGE:
            continue
        if opponent.column != ctx.player_index:
            continue
        if not 0 <= opponent.position < len(ctx.public_rows):
            continue
        entry = ctx.public_rows[opponent.position][ctx.player_index]
        if entry is not None and entry[0] is not None and int(entry[0]) == DRAGON:
            return True
    return False


def shipped(ctx) -> bool:
    """The rule as first written: flee when nothing visible threatens."""
    if ctx.own_blast_pending or any(o.blast_pending for o in ctx.opponents):
        return False
    if ctx.own_position == PIT_CAGE:
        return False
    for opponent in ctx.opponents:
        if opponent.dead:
            continue
        if opponent.escaped or opponent.position == PIT_CAGE:
            return False
        if opponent.column != ctx.player_index:
            continue
        if 0 <= opponent.position < len(ctx.public_rows):
            entry = ctx.public_rows[opponent.position][ctx.player_index]
            if entry is not None and entry[0] == DRAGON:
                return False
    return True


def no_herd(ctx) -> bool:
    """The same, without reading somebody else's cage as a threat."""
    if ctx.own_blast_pending or any(o.blast_pending for o in ctx.opponents):
        return False
    if ctx.own_position == PIT_CAGE:
        return False
    return not _visible_dragon_threat(ctx)


def escaped_only(ctx) -> bool:
    """A rival a displaced raid put in a cage is not a threat; a rival who *left* is.

    Once somebody has left the mine no new rows are dealt, so the cards that
    remain are the only ones coming and they are the risky ones; that is a
    reason to consider leaving too. Somebody merely queued in a cage - which is
    what a raid produces - is not.
    """
    if ctx.own_blast_pending or any(o.blast_pending for o in ctx.opponents):
        return False
    if ctx.own_position == PIT_CAGE:
        return False
    if any(o.escaped and not o.dead for o in ctx.opponents):
        return False
    return not _visible_dragon_threat(ctx)


RULES = {
    SHIPPED: shipped,
    NO_HERD: no_herd,
    ESCAPED: escaped_only,
    BANKS: features.cage_banks_nothing,
}


def with_rule(name: str):
    features.cage_banks_nothing = RULES[name] if name != SHIPPED else shipped


def restore() -> None:
    features.cage_banks_nothing = RULES[BANKS]


def wasted_hauls_in_four_net_games(
    net: PolicyNet, games: int, rule: str
) -> tuple[float, float, float]:
    """Share of hauls thrown away, share of games with one, and points per haul."""
    with_rule(rule)
    wasted = 0
    games_with_waste = 0
    hauls = 0
    gems = 0
    for seed in range(games):
        controller = GameController(3, "neural:selfplay4", seed)
        controller.reset(3, "neural:selfplay4", seed, all_seats=True)
        if net is not None:  # keep the loader honest; the controller caches it
            pass
        steps = 0
        while not controller.env.game_finished and steps < 400:
            steps += 1
            controller.step_one_player_auto()
        history = controller.env.haul_history
        hauls += len(history)
        wasted_here = 0
        for haul in history:
            if len(haul["escaped"]) >= 3 and sum(haul["gems"]) == 0:
                wasted += 1
                wasted_here += 1
            gems += sum(haul["gems"])
        games_with_waste += wasted_here > 0
    restore()
    return (
        100 * wasted / max(1, hauls),
        100 * games_with_waste / games,
        gems / max(1, hauls),
    )


def strength(rule: str, field: str, games: int, net: PolicyNet) -> tuple[float, float]:
    with_rule(rule)
    win = score = 0.0
    for seed in range(games):
        agents = [NeuralAgent(net, seed=seed, temperature=0.7)] + [
            make_scripted_agent(field, seed=seed * 7 + i) for i in range(1, 4)
        ]
        result = play_game(agents, seed=seed)
        win += result.scores[0] == max(result.scores)
        score += result.scores[0]
    restore()
    return 100 * win / games, score / games


def seed_trace(net: PolicyNet, rule: str, seed: int = 7) -> None:
    """The game the viewer plays for one seed, haul by haul."""
    del net  # the controller loads the checkpoint named by the policy
    with_rule(rule)
    controller = GameController(3, "neural:selfplay4", seed)
    controller.reset(3, "neural:selfplay4", seed, all_seats=True)
    steps = 0
    while not controller.env.game_finished and steps < 400:
        steps += 1
        controller.step_one_player_auto()
    restore()
    lines = [
        f"{haul['outcome']} gems {haul['gems']} out {haul['escaped']}"
        for haul in controller.env.haul_history
    ]
    scores = [player.score for player in controller.env.players]
    print(f"  {rule:8s} seed {seed}: " + " | ".join(lines) + f" | scores {scores}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/selfplay4.npz")
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--field", default="")
    parser.add_argument("--strength-games", type=int, default=400)
    args = parser.parse_args()
    net = PolicyNet.load(args.checkpoint)

    if args.field:
        for rule in (SHIPPED, NO_HERD, ESCAPED, BANKS):
            win, score = strength(rule, args.field, args.strength_games, net)
            print(f"3 x {args.field:7s} rule {rule:8s}: win {win:5.1f}%  score {score:+.2f}")
        return

    print(f"four nets, {args.games} games, temperature 0.7")
    table: collections.Counter = collections.Counter()
    for rule in (SHIPPED, NO_HERD, ESCAPED, BANKS):
        wasted, games_with_waste, gems = wasted_hauls_in_four_net_games(
            net, args.games, rule
        )
        table[rule] = wasted
        print(
            f"  rule {rule:8s}: hauls thrown away {wasted:4.1f}%   "
            f"games with one {games_with_waste:5.1f}%   gems collected per haul {gems:4.2f}"
        )
    for rule in (SHIPPED, NO_HERD, ESCAPED, BANKS):
        seed_trace(net, rule)


if __name__ == "__main__":
    main()
