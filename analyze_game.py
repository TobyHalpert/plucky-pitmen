"""One game, move by move: what each seat knew, chose, and what it cost.

The balance numbers in BALANCE.md are aggregate. This tool looks at a single
game — the one the viewer plays for a given seed — and prints every planning
decision with

* the board and every figure's position *as they stood at that moment*,
* what the deciding seat can see that a spectator cannot (its own-column fronts
  and what it is carrying),
* the action it took, the sentence ``features.action_reason`` gives for it, and
  the probability the network put on it next to its top alternatives — which
  separates "the policy believes this" from "the sampler picked it at 3%",
* what the execution phase then did with the board, because a plan only counts
  once it executes.

With ``--forks`` it rebuilds the position in front of each flagged decision
(exit, blast, card in a rival's column while a known gem sat in its own) and
plays that position out twice: once with the move the seat made, once with the
alternative.  Both branches get the same re-rolled continuations, so what comes
out is the value of the move from that position.  It is deliberately *not* the
price the move exacted in the watched game: one continuation of a sampled game
is far too noisy to price a move against, and forcing the recorded script after
a changed move rarely lines up for long.

    python analyze_game.py --seed 7
    python analyze_game.py --seed 7 --forks
"""

from __future__ import annotations

import argparse
import copy

import numpy as np

from features import (
    N_BASE_ACTIONS,
    action_mask,
    action_reason,
    backside_stats,
    state_action_matrix,
)
from game_env import (
    BACKSIDE_A,
    DRAGON,
    DYNAMITE,
    GEM,
    LORRY,
    MAX_COLUMNS,
    PASS,
    PIT_CAGE,
    MineEnv,
)
from gui_controller import NEURAL_PREFIX, NEURAL_TEMPERATURE, GameController
from learning import PolicyNet

FACES = {GEM: "gem", DRAGON: "dragon", DYNAMITE: "dyn"}
MARKS = {BACKSIDE_A: "o", BACKSIDE_A + 1: "v", BACKSIDE_A + 2: "~"}
FLAGGED = {int(PIT_CAGE), int(LORRY)}


def label(action: int) -> str:
    action = int(action)
    if action == PIT_CAGE:
        return "cage"
    if action == LORRY:
        return "lorry"
    if action == PASS:
        return "pass"
    if action >= N_BASE_ACTIONS:
        action %= N_BASE_ACTIONS
        return f"card r{action // MAX_COLUMNS}c{action % MAX_COLUMNS} +bl"
    return f"card r{action // MAX_COLUMNS}c{action % MAX_COLUMNS}"


def board_text(env: MineEnv) -> str:
    lines = []
    for row_index, row in enumerate(env.rows):
        cells = []
        for entry in row:
            if entry is None:
                cells.append("   .  ")
            else:
                front, back = entry
                cells.append(f"{FACES[front]}{MARKS[back]:<2}")
        lines.append(f"    r{row_index}: " + " | ".join(cells))
    return "\n".join(lines)


def positions_text(env: MineEnv) -> str:
    parts = []
    for index, player in enumerate(env.players):
        where = (
            "lorry" if player.position == LORRY
            else "cage" if player.position == PIT_CAGE
            else f"r{player.position}c{player.column}"
        )
        held = [front for front, _ in player.collected_cards]
        mark = " OUT" if player.escaped else " DEAD" if player.dead else ""
        parts.append(
            f"P{index + 1} {where} {held.count(GEM)}g/{held.count(DYNAMITE)}d{mark}"
        )
    return "   ".join(parts)


def known_gem_available(ctx) -> int | None:
    """A legal card in the seat's own column whose front it can see is a gem."""
    for candidate in ctx.legal_actions:
        candidate = int(candidate)
        if not 0 <= candidate < PIT_CAGE:
            continue
        row, column = divmod(candidate, MAX_COLUMNS)
        if column != ctx.player_index or row >= len(ctx.public_rows):
            continue
        entry = ctx.public_rows[row][column]
        if entry is not None and entry[0] == GEM:
            return candidate
    return None


def knowledge_text(ctx) -> str:
    own = []
    for row_index, row in enumerate(ctx.public_rows):
        if ctx.player_index >= len(row):
            continue
        entry = row[ctx.player_index]
        if entry is not None and entry[0] is not None:
            own.append(f"r{row_index}={FACES[entry[0]]}")
    held = [front for front, _ in ctx.private_cards]
    return (
        f"sees own column [{' '.join(own) or 'nothing'}]   "
        f"holding {held.count(GEM)}g/{held.count(DYNAMITE)}d   score {ctx.own_score:+d}"
    )


def probabilities(net: PolicyNet, ctx) -> np.ndarray | None:
    if net is None or not ctx.legal_actions:
        return None
    mask = action_mask(ctx)
    if not mask.any():
        return None
    return net.masked_probs(
        state_action_matrix(ctx), mask, temperature=NEURAL_TEMPERATURE
    )


def transcript(
    seed: int, policy: str, opponents: int, greedy: bool, *, all_seats: bool = True
):
    """Replay a game decision by decision.

    ``all_seats=True`` is the all-neural table used for every measurement in
    BALANCE.md.  ``all_seats=False`` is what the viewer does by default: the
    chosen policy drives P1 and the environment's built-in ``simple`` bot plays
    the rest.  The two are different games, so the header names each seat's
    policy and ``--viewer`` reproduces the viewer's table.
    """
    controller = GameController(opponents, policy, seed)
    controller.reset(
        opponents, policy, seed,
        all_seats=all_seats, deterministic=greedy, advance=False,
    )
    env = controller.env
    net = controller.policy_net()
    events: list[dict] = []
    steps = 0
    while not env.game_finished and steps < 800:
        steps += 1
        seat = env.planning_player
        ctx = env.player_action_context(seat)
        if controller.in_tie_breaker or not ctx.legal_actions:
            controller.step_one_player_auto()
            continue
        owned = seat < len(controller.agents) and controller.agents[seat] is not None
        probs = probabilities(net, ctx) if owned else None
        before = [len(player.collected_cards) for player in env.players]
        state = (board_text(env), positions_text(env))
        haul_before, round_before = env.haul, env.round
        turns_before = env.planning_turns
        controller.step_one_player_auto()
        if owned:
            choice = controller.last_choice
            if choice is None or controller.last_choice_seat != seat:
                continue
            action, blast = int(choice.action), bool(choice.announce_blast)
        else:
            if env.planning_turns != turns_before + 1:
                continue
            action, blast = int(env.last_action), False
        reveals = []
        for index, player in enumerate(env.players):
            for card in player.collected_cards[before[index]:]:
                reveals.append((index, card[0], card[1]))
        events.append(
            {
                # The phase the decision was made in, not the one it advanced to:
                # a step that ends a haul would otherwise be filed under the next.
                "haul": haul_before,
                "round": round_before,
                "seat": seat,
                "ctx": ctx,
                "board": state[0],
                "figures": state[1],
                "action": action,
                "blast": blast,
                "reason": action_reason(ctx, action),
                "p": None if probs is None else float(probs[action]),
                "top": []
                if probs is None
                else sorted(
                    ((int(a), float(probs[a])) for a in np.flatnonzero(probs)),
                    key=lambda item: -item[1],
                )[:4],
                "reveals": reveals,
            }
        )
    return controller, events


def print_game(events: list[dict], controller: GameController, policy: str, seed: int) -> None:
    print(f"policy {policy}, seed {seed}, {len(events)} planning decisions")
    seated = ", ".join(
        f"P{seat + 1} {name}" for seat, name in enumerate(controller.seat_policies())
    )
    print(f"  seated: {seated}")
    current = None
    for record in events:
        if (record["haul"], record["round"]) != current:
            current = (record["haul"], record["round"])
            print(f"\n=== haul {record['haul']}, planning round {record['round']} ===")
            print(record["board"])
            print(f"  {record['figures']}")
        share = "n/a" if record["p"] is None else f"{record['p']:.0%}"
        top = ", ".join(f"{label(a)} {p:.0%}" for a, p in record["top"])
        blast = " + BLAST" if record["blast"] else ""
        print(
            f"  P{record['seat'] + 1} -> {label(record['action'])}{blast}"
            f"   [model {share} | {top}]"
        )
        print(f"      why   {record['reason']}")
        print(f"      knows {knowledge_text(record['ctx'])}")
        free_gem = known_gem_available(record["ctx"])
        takes_a_card = 0 <= record["action"] < PIT_CAGE or record["action"] == PASS
        if free_gem is not None and record["action"] != free_gem and takes_a_card:
            print(
                f"      NOTE  passed a known safe gem at r{free_gem // 5}c{free_gem % 5}"
                if record["action"] != int(PASS)
                else f"      NOTE  passed while a known safe gem sat at r{free_gem // 5}c{free_gem % 5}"
            )
        if record["reveals"]:
            turned = ", ".join(
                f"P{index + 1} took {FACES[front]}{MARKS[back]}"
                for index, front, back in record["reveals"]
            )
            print(f"      THEN  {turned}")
    print("\n=== hauls ===")
    for haul in controller.env.haul_history:
        print(
            f"  haul {haul['haul']}: {haul['outcome']:9s} rounds {haul['rounds']} "
            f"gems {haul['gems']} out {haul['escaped']} dead {haul['dead']}"
        )
    print(f"  final scores {[player.score for player in controller.env.players]}")


def dragon_risk(ctx, action: int) -> float:
    """The chance the seat believes a dragon is behind the card ``action`` digs."""
    row, column = divmod(int(action), MAX_COLUMNS)
    if row >= len(ctx.public_rows) or column >= len(ctx.public_rows[row]):
        return 0.0
    entry = ctx.public_rows[row][column]
    if entry is None:
        return 0.0
    front, back = entry
    if front is not None:
        return 1.0 if int(front) == DRAGON else 0.0
    _known_dragons, _known_safe, _unknown, risk = backside_stats(ctx)
    return float(risk[int(back)])


def decisions_worth_checking(events: list[dict]) -> list[int]:
    """Indices of moves worth a counterfactual.

    An exit or a blast always is. So is a card in a rival's column taken while a
    known gem sat in its own, and so is a 20%-dragon card taken with nothing in
    hand: those are the two ways a haul gets gambled, so they are the moves whose
    value has to be measured rather than assumed.
    """
    indices = []
    for index, record in enumerate(events):
        ctx = record["ctx"]
        action = record["action"]
        if action in FLAGGED or record["blast"]:
            indices.append(index)
            continue
        if not 0 <= action < PIT_CAGE:
            continue
        raided = action % MAX_COLUMNS != ctx.player_index
        if raided and known_gem_available(ctx) is not None:
            indices.append(index)
            continue
        holding = [front for front, _ in ctx.private_cards]
        if GEM not in holding and dragon_risk(ctx, action) >= 0.2:
            indices.append(index)
    return indices


def scripted_replay(
    controller: GameController,
    events: list[dict],
    seed: int,
    play_out: bool = True,
) -> tuple[GameController, int]:
    """Replay the recorded decisions, optionally replacing one of them.

    The replay is exact: every recorded action is forced, so the final scores
    come out identical to the game the viewer played.  That is what makes the
    transcript trustworthy -- and it is also why the replay stops at the first
    decision that no longer lines up with the seats to move.  With ``play_out``
    the rest of the game is then played by the seats' own agents; ``autos``
    counts those steps, so a large number warns that the branch left the
    recorded game behind.
    """
    fresh = GameController(controller.opponents, controller.policy, seed)
    fresh.reset(
        controller.opponents,
        controller.policy,
        seed,
        all_seats=True,
        deterministic=controller.deterministic,
        advance=False,
    )
    env = fresh.env
    autos = 0
    steps = 0
    for record in events:
        if env.game_finished or steps > 800:
            break
        action, blast = record["action"], record["blast"]
        if env.planning_player != record["seat"]:
            break
        try:
            env.step_one_player(action, announce_blast=blast, force_action=True)
        except ValueError:
            break
        steps += 1
    while play_out and not env.game_finished and steps < 800:
        fresh.step_one_player_auto()
        autos += 1
        steps += 1
    return fresh, autos


def _play_out(controller: GameController, steps: int = 800) -> list[int]:
    played = 0
    while not controller.env.game_finished and played < steps:
        controller.step_one_player_auto()
        played += 1
    return [player.score for player in controller.env.players]


def _branch(
    source: GameController, action: int, blast: bool, seed: int
) -> GameController:
    """A copy of the position with one action applied, dice reset to ``seed``."""
    branch = copy.copy(source)
    branch.env = copy.deepcopy(source.env)
    branch.agents = copy.deepcopy(source.agents)
    for agent in branch.agents:
        if hasattr(agent, "rng"):
            agent.rng = np.random.default_rng(seed)
    branch.env.step_one_player(action, announce_blast=blast, force_action=True)
    return branch


def _value(
    source: GameController, action: int, blast: bool, seat: int, samples: int
) -> tuple[float, float]:
    scores = np.array(
        [_play_out(_branch(source, action, blast, seed))[seat] for seed in range(samples)],
        dtype=float,
    )
    error = scores.std(ddof=1) / np.sqrt(len(scores)) if samples > 1 else 0.0
    return float(scores.mean()), float(error)


def fork_prices(
    controller: GameController,
    events: list[dict],
    seed: int,
    limit: int,
    samples: int,
) -> None:
    """What one move was worth: the same position, re-rolled ``samples`` times.

    Each branch is played out from the identical position with the agents' dice
    reset to the same seeds, so the chosen move and the alternative are compared
    on the same re-rolls.  This is the value of the move from that position, not
    the price it happened to exact in the game that was watched: one sample of
    the rest of the game is far too noisy to price a move against.
    """
    print(f"\n=== what the move was worth: {samples} re-rolled continuations ===")
    realized = [player.score for player in controller.env.players]
    checked = decisions_worth_checking(events)
    print(f"  {len(checked)} flagged decisions; the viewer's game ended {realized}")
    shown = 0
    total = 0.0
    for index in checked:
        if shown >= limit:
            break
        record = events[index]
        ctx = record["ctx"]
        seat = record["seat"]
        action = record["action"]
        alternatives: list[tuple[int, str]] = []
        if record["top"] and record["top"][0][0] != action:
            best = int(record["top"][0][0])
            alternatives.append((best, f"its own top {label(best)}"))
        gem = known_gem_available(ctx)
        if gem is not None and gem != action and all(gem != a for a, _ in alternatives):
            alternatives.append((int(gem), f"the safe gem {label(gem)}"))
        if not alternatives:
            continue
        fresh, _ = scripted_replay(controller, events[:index], seed, play_out=False)
        if fresh.env.planning_player != seat or fresh.env.game_finished:
            continue
        base, base_se = _value(fresh, action, record["blast"], seat, samples)
        for alternative, description in alternatives:
            alt, alt_se = _value(fresh, alternative, False, seat, samples)
            gap = base - alt
            total += gap
            print(
                f"  h{record['haul']}r{record['round']} P{seat + 1}: "
                f"{label(action)} {base:+.2f} +-{base_se:.2f}   vs "
                f"{description} {alt:+.2f} +-{alt_se:.2f}   "
                f"[{gap:+.2f} for the move made]"
            )
            shown += 1
    print(f"  total for the deciding seats: {total:+.2f} points per game")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", default=f"{NEURAL_PREFIX}selfplay4")
    parser.add_argument("--opponents", type=int, default=3)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="play the viewer's default table: the policy on P1, the built-in "
        "`simple` bot on the other seats",
    )
    parser.add_argument("--forks", action="store_true")
    parser.add_argument("--forks-max", type=int, default=40)
    parser.add_argument("--fork-samples", type=int, default=60)
    args = parser.parse_args()

    controller, events = transcript(
        args.seed, args.policy, args.opponents, args.greedy,
        all_seats=not args.viewer,
    )
    print_game(events, controller, args.policy, args.seed)
    print(f"\n  decisions worth checking: {len(decisions_worth_checking(events))}")
    if args.forks:
        fork_prices(
            controller, events, args.seed, args.forks_max, args.fork_samples
        )


if __name__ == "__main__":
    main()
