"""Agent interface and scripted baselines for Plucky Pitmen.

Every agent receives exactly one object: a
:class:`game_env.PlayerActionContext`.  The environment itself is never passed
to an agent, so no policy can accidentally read hidden card fronts.

An agent exposes four decisions:

``decide(ctx)``          planning action, optionally announcing a blast
``blast_choice(cards)``  which of the three blast draws to keep
``tie_break(ctx)``       where to draw a tie-break card from

Scripted agents are deliberately *context-based* re-implementations of the
built-in opponent policies, which makes them usable at any seat in a self-play
match (the environment's built-in policies only ever play seats 1..n).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from game_env import (
    DYNAMITE,
    DRAGON,
    FRONT_UNKNOWN,
    GEM,
    LORRY,
    MAX_COLUMNS,
    PASS,
    PIT_CAGE,
    PlayerActionContext,
)
from features import backside_stats, round_dragon_hazard, split_action


@dataclass(frozen=True)
class ActionChoice:
    """A primitive environment action plus an optional blast announcement."""

    action: int
    announce_blast: bool = False


class Agent:
    """Base class with safe defaults for everything but ``act``."""

    name = "agent"

    def act(self, ctx: PlayerActionContext) -> int:
        raise NotImplementedError

    def decide(self, ctx: PlayerActionContext) -> ActionChoice:
        return ActionChoice(int(self.act(ctx)))

    # Blasting ---------------------------------------------------------
    def wants_blast(self, ctx: PlayerActionContext) -> bool:
        return False

    def blast_choice(
        self, cards: Sequence[tuple[int, int]], ctx: PlayerActionContext
    ) -> int:
        return preferred_blast_choice(cards, ctx)

    # Tie-break --------------------------------------------------------
    def tie_break(
        self, ctx: PlayerActionContext
    ) -> tuple[str, int | None, int | None]:
        return safe_tie_break(ctx)


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def owns_dynamite_pair(ctx: PlayerActionContext) -> bool:
    dynamite = sum(1 for front, _ in ctx.private_cards if front == DYNAMITE)
    return dynamite >= 2 and not ctx.own_blast_pending


def preferred_blast_choice(
    cards: Sequence[tuple[int, int]],
    ctx: PlayerActionContext | None = None,
) -> int:
    """Choose which blast draw to keep.

    When the fronts are visible (the historical engine behaviour) keep a gem,
    then dynamite, and never knowingly keep a dragon. When the cards are face
    down, as the printed rules require, keep the card whose backside implies the
    lowest dragon risk.
    """
    known = [
        index
        for index, (front, _back) in enumerate(cards)
        if front != FRONT_UNKNOWN
    ]
    if known:
        safe = [index for index in known if cards[index][0] != DRAGON]
        if not safe:
            return 0
        for index in safe:
            if cards[index][0] == GEM:
                return index
        for index in safe:
            if cards[index][0] == DYNAMITE:
                return index
        return safe[0]

    if ctx is None:
        return 0
    _known_dragons, _known_safe, _unknown, risk = backside_stats(ctx)
    return min(
        range(len(cards)),
        key=lambda index: float(risk[cards[index][1]]),
    )


def behind_on_score(ctx: PlayerActionContext) -> bool:
    """True when some rival is ahead of this player on points."""
    return any(opponent.score > ctx.own_score for opponent in ctx.opponents)


def safe_tie_break(
    ctx: PlayerActionContext,
) -> tuple[str, int | None, int | None]:
    """Draw from a known-safe own-column card, otherwise from the deck."""
    own_column = ctx.player_index
    for row_index, row in enumerate(ctx.public_rows):
        if own_column >= len(row):
            continue
        entry = row[own_column]
        if entry is None or entry[0] in (None, DRAGON):
            continue
        return "mine", row_index, own_column
    return "deck", None, None


def _movement_depth(ctx: PlayerActionContext) -> int:
    if ctx.own_position == PIT_CAGE:
        return int(ctx.own_cage_index or 0)
    if ctx.own_position == LORRY:
        return 0
    return int(ctx.own_position)


def _card_actions(ctx: PlayerActionContext) -> list[int]:
    return [int(a) for a in ctx.legal_actions if int(a) < PIT_CAGE]


def _retreat(ctx: PlayerActionContext) -> int | None:
    if PIT_CAGE in ctx.legal_actions:
        return PIT_CAGE
    if LORRY in ctx.legal_actions:
        return LORRY
    if PASS in ctx.legal_actions:
        return PASS
    return None


# ----------------------------------------------------------------------
# Scripted agents
# ----------------------------------------------------------------------
class RandomAgent(Agent):
    """Uniform over free cards, retreating when no card is reachable."""

    name = "random"

    def __init__(self, seed: int = 0, blast: str = "never") -> None:
        self.rng = np.random.default_rng(seed)
        self.blast = blast

    def act(self, ctx: PlayerActionContext) -> int:
        cards = _card_actions(ctx)
        if cards:
            return int(self.rng.choice(cards))
        # Already planned this round: passing is what lets the round resolve.
        if PASS in ctx.legal_actions:
            return PASS
        retreat = _retreat(ctx)
        if retreat is not None:
            return retreat
        return PASS

    def wants_blast(self, ctx: PlayerActionContext) -> bool:
        return should_blast(self.blast, ctx)


class GreedyAgent(Agent):
    """Dives to the deepest reachable row of cards."""

    name = "greedy"

    def __init__(self, seed: int = 0, blast: str = "never") -> None:
        self.rng = np.random.default_rng(seed)
        self.blast = blast

    def act(self, ctx: PlayerActionContext) -> int:
        cards = _card_actions(ctx)
        if cards:
            depth = _movement_depth(ctx)
            nearest = [
                action for action in cards
                if action // MAX_COLUMNS == depth - 1
            ]
            if nearest:
                return int(self.rng.choice(nearest))
            return int(self.rng.choice(cards))
        if PASS in ctx.legal_actions:
            return PASS
        retreat = _retreat(ctx)
        return retreat if retreat is not None else PASS

    def wants_blast(self, ctx: PlayerActionContext) -> bool:
        return should_blast(self.blast, ctx)


class CautiousAgent(Agent):
    """Greedy early, but retreats to the pit cage once the mine deepens."""

    name = "cautious"

    def __init__(self, seed: int = 0, retreat_chance: float = 0.35) -> None:
        self.rng = np.random.default_rng(seed)
        self.retreat_chance = retreat_chance
        self._greedy = GreedyAgent(seed=seed)

    def act(self, ctx: PlayerActionContext) -> int:
        if ctx.round > 0 and _movement_depth(ctx) > 0:
            if self.rng.random() < self.retreat_chance:
                retreat = _retreat(ctx)
                if retreat is not None:
                    return retreat
        return self._greedy.act(ctx)


class RuleAgent(Agent):
    """Wraps ``rule_policy.deterministic_policy`` (the hand-tuned baseline)."""

    name = "rule"

    def __init__(self, blast: str = "never") -> None:
        self.blast = blast

    def act(self, ctx: PlayerActionContext) -> int:
        from rule_policy import deterministic_policy

        return int(deterministic_policy(ctx))

    def wants_blast(self, ctx: PlayerActionContext) -> bool:
        return should_blast(self.blast, ctx)

    def blast_choice(
        self, cards: Sequence[tuple[int, int]], ctx: PlayerActionContext
    ) -> int:
        return preferred_blast_choice(cards, ctx)


class ReaderAgent(Agent):
    """The rule baseline, but it raids on the backside instead of on confidence.

    ``RuleAgent`` treats a rival sitting quietly on its own column as proof that
    the card is safe and raids it. That inference is what makes a visible dragon
    in one's own column a trap. ``ReaderAgent`` reads only the backside and the
    cards already known off it, so a fresh backside stays too risky to raid.
    """

    name = "reader"

    def __init__(self, blast: str = "never") -> None:
        self.blast = blast

    def act(self, ctx: PlayerActionContext) -> int:
        from rule_policy import deterministic_policy

        return int(deterministic_policy(ctx, trust_sitter=False))

    def wants_blast(self, ctx: PlayerActionContext) -> bool:
        return should_blast(self.blast, ctx)

    def blast_choice(
        self, cards: Sequence[tuple[int, int]], ctx: PlayerActionContext
    ) -> int:
        return preferred_blast_choice(cards, ctx)


class RusherAgent(Agent):
    """Leaves the mine as fast as the rules allow, whatever the loot.

    This is the minimal implementation of the suspected dominant strategy:
    walk to the nearest pit cage, sit there, and escape on the first execution
    phase.  It ignores gems, dynamite and blasts entirely.  If it beats the
    hand-tuned rule policy, "leave on turn one" is dominant; if it does not,
    the extra points for mining are doing their job.
    """

    name = "rusher"

    def act(self, ctx: PlayerActionContext) -> int:
        if ctx.own_cage_index is not None:
            return PASS if PASS in ctx.legal_actions else int(ctx.legal_actions[0])
        cages = [
            action
            for action in ctx.legal_actions
            if PIT_CAGE <= action < LORRY
        ]
        if cages:
            # Any cage ends the turn outside the mine; take the lowest one.
            return int(min(cages))
        if PASS in ctx.legal_actions:
            return PASS
        return int(ctx.legal_actions[0])


class LeaverAgent(Agent):
    """Take a gem, then walk out. The counter to :class:`RusherAgent`.

    A rusher only scores while its opponents *stay* in the mine: the printed
    rules compare gems among everybody who left, so an empty-handed escape is
    worth nothing as soon as somebody else leaves with loot. This agent is the
    cheapest policy that exploits that: collect one gem, then go to the nearest
    pit cage.
    """

    name = "leaver"

    def __init__(self, target_gems: int = 1) -> None:
        self.target_gems = target_gems
        self._planning_key: tuple[int, int] | None = None
        self._moved = False

    def act(self, ctx: PlayerActionContext) -> int:
        key = (int(ctx.haul), int(ctx.round))
        if key != self._planning_key:
            self._planning_key = key
            self._moved = False

        gems = sum(1 for front, _ in ctx.private_cards if front == GEM)
        if gems >= self.target_gems and PIT_CAGE in ctx.legal_actions:
            return PIT_CAGE
        if self._moved:
            # Standing on the chosen card: hold position, or the planning phase
            # walks straight past it and collects nothing.
            if PASS in ctx.legal_actions:
                return PASS
            return PIT_CAGE if PIT_CAGE in ctx.legal_actions else int(ctx.legal_actions[0])

        cards = _card_actions(ctx)
        own = [action for action in cards if action % MAX_COLUMNS == ctx.player_index]
        known_gems = [action for action in cards if card_front(ctx, action) == GEM]
        safe_own = [action for action in own if card_front(ctx, action) != DRAGON]
        for pool in (known_gems, safe_own, own, cards):
            if pool:
                # The deepest reachable row, i.e. collect something now.
                self._moved = True
                return int(max(pool))
        if PIT_CAGE in ctx.legal_actions:
            return PIT_CAGE
        return PASS if PASS in ctx.legal_actions else int(ctx.legal_actions[0])


def card_front(ctx: PlayerActionContext, action: int) -> int | None:
    """Front of a mine card, or ``None`` if that card is hidden to the player."""
    row, col = divmod(int(action), MAX_COLUMNS)
    if row >= len(ctx.public_rows) or col >= len(ctx.public_rows[row]):
        return None
    entry = ctx.public_rows[row][col]
    return None if entry is None else entry[0]


class ThreatAgent(Agent):
    """The rule policy, but it leaves the mine when an opponent announces a blast.

    A pending blast is a public commitment: the blaster keeps two of three drawn
    cards, and if either is a dragon the haul ends and everyone still inside
    dies. This is the cheapest *reaction* to that threat — retreat to the
    nearest pit cage, which leaves the mine safely at the end of the round.
    It exists so the threat can be measured: if reacting pays, a trained agent
    that ignores the threat is leaving value on the table.
    """

    name = "threat"

    def __init__(self, blast: str = "never") -> None:
        self.blast = blast

    @staticmethod
    def threatened(ctx: PlayerActionContext) -> bool:
        return any(
            opponent.blast_pending and not opponent.escaped and not opponent.dead
            for opponent in ctx.opponents
        )

    def act(self, ctx: PlayerActionContext) -> int:
        if self.threatened(ctx) and PIT_CAGE in ctx.legal_actions:
            return PIT_CAGE
        from rule_policy import deterministic_policy

        return int(deterministic_policy(ctx))

    def wants_blast(self, ctx: PlayerActionContext) -> bool:
        return should_blast(self.blast, ctx)

    def blast_choice(
        self, cards: Sequence[tuple[int, int]], ctx: PlayerActionContext
    ) -> int:
        return preferred_blast_choice(cards, ctx)


class CounterAgent(Agent):
    """A deterministic seat built to beat the learned nets on their own tells.

    Three measured facts about the shipped checkpoint, all in BALANCE.md:

    * **It never displaces.** 1 push taken out of 1,953 offered, and the pushes
      it declines are worth -0.30 points. So a seat can stand on a card in its
      own column with a rival deeper in the mine and never be pushed off it, and
      there is nothing to gain by pushing back. This agent never pushes.
    * **It always signals a dragon.** The escape rule gives a seat two reasons
      to take a cage with an empty hand: a charged blast, or a dragon it can see
      under itself. A net that leaves the mine having collected nothing and with
      no blast in the table's hand has therefore told the table exactly which
      card it was standing on, and that card is a dragon. That is the tell this
      agent reads: a rival's empty-handed exit marks its last cell as a dragon,
      which is a cell never to step on.
    * **It hands the haul over.** The nets escape 8.6 times a game and throw
      away 30% of the hauls (three or more seats out with two gems or fewer
      between them). The bonus goes to the escapee with the most gems, so a seat
      that stays, collects, and leaves with a hand beats a table that leaves
      early with one gem each.

    So the policy is deliberately the opposite of the net's: dig until the loot
    is worth banking, bank it when a blast is charged or the mine is genuinely
    dangerous, and never spend a round on the lorry, a push, or an exit that
    banks nothing.
    """

    name = "counter"

    def __init__(
        self,
        bank_gems: int = 2,
        hazard_bar: float = 0.45,
        read_signals: bool = True,
        allow_push: bool = False,
        bank_on_sight: bool = True,
        safe_only: bool = True,
    ) -> None:
        self.bank_gems = int(bank_gems)
        self.hazard_bar = float(hazard_bar)
        #: Leave as soon as ``bank_gems`` are in hand, whatever the hazard. A
        #: banked gem is scored; a gem carried into a woken dragon is not.
        self.bank_on_sight = bool(bank_on_sight)
        #: Only ever stand on a card this seat has seen the front of (its own
        #: column). A card nobody can see is a one-in-three dragon, and a seat
        #: that never takes one is never killed by a card.
        self.safe_only = bool(safe_only)
        #: Read a rival's empty-handed exit as "the card under it was a dragon".
        self.read_signals = bool(read_signals)
        #: Take a displacement when one is on offer. Off by default: a push is
        #: worth -0.30 points to the seat that makes it (BALANCE.md).
        self.allow_push = bool(allow_push)
        #: rival -> (haul, row, column, cards held) at its last sighting.
        self._seen: dict[int, tuple[int, int, int, int]] = {}
        #: cells this agent has read as dragons, as (haul, row, column).
        self.dragon_cells: set[tuple[int, int, int]] = set()
        self._haul = -1

    # -- the tell ------------------------------------------------------
    def _observe(self, ctx: PlayerActionContext) -> None:
        if ctx.haul != self._haul:
            self._haul = int(ctx.haul)
            self._seen.clear()
        blast = ctx.own_blast_pending or any(
            opponent.blast_pending for opponent in ctx.opponents
        )
        for opponent in ctx.opponents:
            previous = self._seen.get(opponent.player_index)
            if (
                previous is not None
                and self.read_signals
                and opponent.escaped
                and not blast
                and previous[3] == opponent.card_count
            ):
                # It left the mine with an empty hand and nothing to run from
                # but the card it stood on.
                self.dragon_cells.add((previous[0], previous[1], previous[2]))
            if (
                not (opponent.escaped or opponent.dead)
                and 0 <= opponent.position < PIT_CAGE
                and opponent.column is not None
            ):
                self._seen[opponent.player_index] = (
                    int(ctx.haul),
                    int(opponent.position),
                    int(opponent.column),
                    int(opponent.card_count),
                )

    def _known_dragon_cells(self, ctx: PlayerActionContext) -> set[tuple[int, int]]:
        return {
            (row, column)
            for haul, row, column in self.dragon_cells
            if haul == ctx.haul
        }

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _gems(ctx: PlayerActionContext) -> int:
        return sum(1 for front, _ in ctx.private_cards if front == GEM)

    @staticmethod
    def _occupied(ctx: PlayerActionContext) -> set[tuple[int, int]]:
        return {
            (opponent.position, opponent.column)
            for opponent in ctx.opponents
            if not (opponent.escaped or opponent.dead) and opponent.column is not None
        }

    @staticmethod
    def _alone(ctx: PlayerActionContext) -> bool:
        return all(
            opponent.escaped or opponent.dead for opponent in ctx.opponents
        )

    def _own_cells(self, ctx: PlayerActionContext, legal: list[int]):
        """Legal free cards in this seat's own column, with what is known of them."""
        out = []
        for action in legal:
            row, column = divmod(action, MAX_COLUMNS)
            if column != ctx.player_index or row >= len(ctx.public_rows):
                continue
            entry = ctx.public_rows[row][column]
            front = None if entry is None else entry[0]
            back = None if entry is None else int(entry[1])
            out.append((row, int(action), front, back))
        return out

    # -- the decision --------------------------------------------------
    def act(self, ctx: PlayerActionContext) -> int:
        self._observe(ctx)
        legal = [int(a) for a in ctx.legal_actions]
        if not legal:
            return int(PASS)
        legal_set = set(legal)
        gems = self._gems(ctx)
        known = self._known_dragon_cells(ctx)
        occupied = self._occupied(ctx)
        blast = ctx.own_blast_pending or any(
            opponent.blast_pending for opponent in ctx.opponents
        )

        # Already out of the mine (in a cage): the departure is the decision.
        if ctx.own_position == PIT_CAGE:
            if PASS in legal_set:
                return int(PASS)
            return int(LORRY) if LORRY in legal_set else legal[0]

        # Bank it when something can take the hand away.
        if PIT_CAGE in legal_set and gems >= 1 and blast:
            return int(PIT_CAGE)
        if (
            PIT_CAGE in legal_set
            and gems >= self.bank_gems
            and self.bank_on_sight
        ):
            return int(PIT_CAGE)
        if (
            PIT_CAGE in legal_set
            and gems >= self.bank_gems
            and round_dragon_hazard(ctx) >= self.hazard_bar
        ):
            return int(PIT_CAGE)
        # Alone in the mine and holding loot: leave before the mine ends, or the
        # hand is never scored.
        if PIT_CAGE in legal_set and gems >= 1 and self._alone(ctx):
            return int(PIT_CAGE)

        own = self._own_cells(ctx, legal)
        on_dragon = (
            ctx.own_column == ctx.player_index
            and ctx.own_position >= 0
            and ctx.own_position < len(ctx.public_rows)
            and ctx.public_rows[ctx.own_position][ctx.player_index] is not None
            and ctx.public_rows[ctx.own_position][ctx.player_index][0] == DRAGON
        )
        if on_dragon:
            # Nothing to be gained by sitting on it: step onto anything in the
            # own column that cannot be a dragon, deepest gem first.
            safe = [
                (row, action, front)
                for row, action, front, _back in own
                if front is not None and front != DRAGON
            ]
            safe.sort(key=lambda item: (item[2] != GEM, -item[0]))
            if safe:
                return safe[0][1]
            if PIT_CAGE in legal_set and gems >= 1:
                return int(PIT_CAGE)
            if LORRY in legal_set:
                return int(LORRY)
            if PASS in legal_set:
                return int(PASS)

        # The figure is where it should be for this round: let the phase close.
        if ctx.own_planned and PASS in legal_set:
            return int(PASS)

        # Dig. Own column first (the fronts are known and a dragon there is
        # free information), then a raid on an unoccupied card the table has
        # not signalled as a dragon.
        known_gems = [
            (row, action) for row, action, front, _b in own if front == GEM
        ]
        if known_gems:
            return max(known_gems)[1]
        free_own = [
            (row, action)
            for row, action, front, _b in own
            if front is None and (row, ctx.player_index) not in known
        ] if not self.safe_only else []
        if free_own:
            return max(free_own)[1]
        own_dynamite = [
            (row, action) for row, action, front, _b in own if front == DYNAMITE
        ]
        if own_dynamite and not ctx.private_cards:
            return max(own_dynamite)[1]

        try:
            from features import backside_stats

            _known, _safe, _unknown, risk = backside_stats(ctx)
        except Exception:  # pragma: no cover - defensive
            risk = None
        raids = []
        for action in legal:
            row, column = divmod(action, MAX_COLUMNS)
            if column == ctx.player_index or row >= len(ctx.public_rows):
                continue
            if (row, column) in known:
                continue
            entry = ctx.public_rows[row][column]
            if (row, column) in occupied:
                # A push. Only when asked for: it is worth -0.30 points, and the
                # branch it pays best (the victim answering with the lorry) is
                # cancelled by the branch where the phase closes early.
                if self.allow_push:
                    raids.append((-1.0, -row, action))
                continue
            if self.safe_only or entry is None:
                continue  # a card nobody has seen the front of is a 1-in-3 dragon
            back = int(entry[1])
            score = -float(risk[back]) if risk is not None else 0.0
            raids.append((score, -row, action))
        if raids:
            return max(raids)[2]

        if PIT_CAGE in legal_set and gems >= 1:
            return int(PIT_CAGE)
        if PASS in legal_set:
            return int(PASS)
        if LORRY in legal_set:
            return int(LORRY)
        return legal[0]


class PusherAgent(CounterAgent):
    """Deterministic counter with optional displacement and score-aware targeting.

    ``target="shallowest"`` reproduces the original policy. ``"top-score"``
    prefers the reachable rival with the most total points; ``"leader-only"``
    rejects anyone below the table's maximum total score (including our score).
    Ties are broken by shallowest row, then action index. All information comes
    from the normal public context. Banking, known-gem and dragon-risk safeguards
    remain unchanged. See BALANCE.md for measured results, not assumed strength.
    """

    name = "pusher"

    def __init__(
        self,
        bank_gems: int = 2,
        hazard_bar: float = 0.45,
        *,
        tell: str = "held",
        push_risk_bar: float = 0.2,
        before_planning: bool = True,
        target: str = "shallowest",
    ) -> None:
        super().__init__(bank_gems=bank_gems, hazard_bar=hazard_bar)
        if tell not in ("held", "any"):
            raise ValueError(f"unknown push tell: {tell}")
        #: ``"held"``: only a rival this seat has already seen on that cell, so it
        #: has spent a decision not moving. ``"any"``: every rival it can reach.
        if target not in ("shallowest", "top-score", "leader-only"):
            raise ValueError(f"unknown push target: {target}")
        self.target = target
        self.tell = tell
        #: Skip the push when the backside says the card is likely a dragon.
        self.push_risk_bar = float(push_risk_bar)
        #: Only push on the decision that decides this seat's round. Pushing after
        #: planning is climbing off the card it chose, which the mask refuses for
        #: every seat: the pusher would be buying an unknown card with a known one.
        self.before_planning = bool(before_planning)
        self.pushes = 0

    # -- what the seat can see -----------------------------------------
    def _held_cells(self, ctx: PlayerActionContext) -> set[tuple[int, int]]:
        """Cells a rival was already standing on at this seat's previous turn."""
        held: set[tuple[int, int]] = set()
        for opponent in ctx.opponents:
            if opponent.escaped or opponent.dead or opponent.column is None:
                continue
            if not 0 <= int(opponent.position) < int(PIT_CAGE):
                continue
            if int(opponent.column) != int(opponent.player_index):
                continue  # only a card in the rival's own column can be pushed
            recorded = self._seen.get(int(opponent.player_index))
            if (
                recorded is not None
                and recorded[0] == int(ctx.haul)
                and recorded[1] == int(opponent.position)
                and recorded[2] == int(opponent.column)
            ):
                held.add((int(opponent.position), int(opponent.column)))
        return held

    def _own_card_is_a_gem(self, ctx: PlayerActionContext) -> bool:
        if ctx.own_column != ctx.player_index:
            return False
        row = int(ctx.own_position)
        if not 0 <= row < len(ctx.public_rows):
            return False
        entry = ctx.public_rows[row][ctx.player_index]
        return entry is not None and entry[0] == GEM

    def _best_push(
        self,
        ctx: PlayerActionContext,
        held: set[tuple[int, int]],
        legal: list[int],
    ) -> int | None:
        known = self._known_dragon_cells(ctx)
        _dragons, _safe, _unknown, risk = backside_stats(ctx)
        candidates = []
        for action in legal:
            row, column = divmod(action, MAX_COLUMNS)
            if column == ctx.player_index or row >= len(ctx.public_rows):
                continue
            occupant = next(
                (
                    opponent
                    for opponent in ctx.opponents
                    if not (opponent.escaped or opponent.dead)
                    and opponent.column is not None
                    and int(opponent.position) == row
                    and int(opponent.column) == column
                    and int(opponent.column) == int(opponent.player_index)
                ),
                None,
            )
            if occupant is None:
                continue  # nothing there, or not a pushable position
            if (row, column) in known:
                continue  # it left empty-handed with no blast to run from
            if self.tell == "held" and (row, column) not in held:
                continue
            entry = ctx.public_rows[row][column]
            if entry is not None:
                # ``risk`` is float32; comparing it to a Python float needs a
                # tolerance, or the base rate (2 dragons in 10 = 0.2) reads as
                # "above the bar" and every push is refused by rounding.
                if float(risk[int(entry[1])]) > self.push_risk_bar + 1e-6:
                    continue  # the backside says this card is likelier than even
            # "Leader" means most total points, not cards held this haul.
            if self.target == "leader-only" and occupant.score < max(
                [ctx.own_score] + [other.score for other in ctx.opponents]
            ):
                continue
            priority = -int(occupant.score) if self.target != "shallowest" else 0
            candidates.append((priority, row, action))
        if not candidates:
            return None
        # Score priority first when enabled; shallowest row breaks ties.
        return min(candidates)[2]

    # -- the decision --------------------------------------------------
    def act(self, ctx: PlayerActionContext) -> int:
        legal = [int(a) for a in ctx.legal_actions]
        held = self._held_cells(ctx)
        self._observe(ctx)
        if not legal:
            return int(PASS)

        gems = self._gems(ctx)
        blast = ctx.own_blast_pending or any(
            opponent.blast_pending for opponent in ctx.opponents
        )
        banking = int(PIT_CAGE) in legal and (
            (gems >= 1 and blast)
            or gems >= self.bank_gems
            or (gems >= 1 and self._alone(ctx))
        )
        decided = ctx.own_planned and self.before_planning
        if not banking and not decided and not self._own_card_is_a_gem(ctx):
            target = self._best_push(ctx, held, legal)
            if target is not None:
                self.pushes += 1
                return int(target)
        return super().act(ctx)


def visible_dragon_ahead(ctx: PlayerActionContext) -> tuple[int, int] | None:
    """A dragon the player can see in its own column, if it can still be forced onto it.

    Fronts in a player's own colour column are visible to that player, so a
    dragon there is known. Rows below the figure can never be dug again (a move
    must go towards the lorries), so only rows above the current position count.
    Returns ``(row, backside)`` — the backside matters: a dragon known to be in
    the mine is one dragon of that set that is *not* in the draw pile.
    """
    column = ctx.player_index
    depth = _movement_depth(ctx)
    for row in range(min(depth, len(ctx.public_rows))):
        cards = ctx.public_rows[row]
        if column >= len(cards):
            continue
        entry = cards[column]
        if entry is not None and entry[0] == DRAGON:
            return row, int(entry[1])
    return None


def should_blast(mode: str, ctx: PlayerActionContext) -> bool:
    """Shared blast heuristic for scripted agents.

    ``never``  - the baseline rule policy never blasts.
    ``rich``   - blast once two gems are safe in hand.
    ``always`` - blast whenever two dynamite are available.
    ``trap``   - blast when a dragon is known to be waiting in the player's own
                 column, i.e. sitting still is a losing move anyway.
    ``behind`` - blast when another player is ahead on gems.
    """
    if mode in ("never", "", None):
        return False
    if not owns_dynamite_pair(ctx):
        return False
    if not _card_actions(ctx):
        return False
    if mode == "trap":
        return visible_dragon_ahead(ctx) is not None
    if mode == "behind":
        return behind_on_score(ctx)
    if mode in ("never", "", None):
        return False
    if not owns_dynamite_pair(ctx):
        return False
    if not _card_actions(ctx):
        return False
    if mode == "always":
        return True
    if mode == "rich":
        gems = sum(1 for front, _ in ctx.private_cards if front == GEM)
        return gems >= 2
    raise ValueError(f"unknown blast mode: {mode}")


# ----------------------------------------------------------------------
# Learned agent
# ----------------------------------------------------------------------
class NeuralAgent(Agent):
    """Samples actions from a :class:`learning.PolicyNet`.

    ``recorder`` collects one :class:`learning.Decision` per move.  Only pass a
    recorder for seats the learner actually controls.
    """

    name = "neural"

    def __init__(
        self,
        net: Any,
        *,
        name: str | None = None,
        deterministic: bool = False,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        seed: int = 0,
        recorder: list | None = None,
        player: int | None = None,
    ) -> None:
        self.net = net
        self.name = name or getattr(net, "name", "neural")
        self.deterministic = deterministic
        self.temperature = temperature
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.recorder = recorder
        self.player = player

    def act(self, ctx: PlayerActionContext) -> int:
        choice = self.decide(ctx)
        return choice.action

    def decide(self, ctx: PlayerActionContext) -> ActionChoice:
        from features import action_mask, state_action_matrix, state_features

        mask = action_mask(ctx)
        if not mask.any():
            return ActionChoice(int(PASS))

        state = state_features(ctx)
        matrix = state_action_matrix(ctx, state=state)
        action, logp, value = self.net.select_action(
            matrix,
            mask,
            self.rng,
            deterministic=self.deterministic,
            temperature=self.temperature,
            epsilon=self.epsilon,
        )
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
        primitive, announces_blast = split_action(action)
        return ActionChoice(primitive, announces_blast)


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
SCRIPTED = (
    "random",
    "greedy",
    "cautious",
    "rule",
    "reader",
    "rusher",
    "leaver",
    "threat",
    "counter",
    "pusher",
    "pusher-leader",
    "pusher-leader-only",
)


def make_scripted_agent(
    policy: str, *, seed: int = 0, blast: str = "never"
) -> Agent:
    if policy == "random":
        return RandomAgent(seed=seed, blast=blast)
    if policy == "greedy":
        return GreedyAgent(seed=seed, blast=blast)
    if policy == "cautious":
        return CautiousAgent(seed=seed)
    if policy == "rule":
        return RuleAgent(blast=blast)
    if policy == "reader":
        return ReaderAgent(blast=blast)
    if policy == "rusher":
        return RusherAgent()
    if policy == "leaver":
        return LeaverAgent()
    if policy == "threat":
        return ThreatAgent(blast=blast)
    if policy == "counter":
        return CounterAgent()
    if policy == "pusher":
        return PusherAgent()
    if policy in ("pusher-leader", "pusher-leader-only"):
        target = "top-score" if policy == "pusher-leader" else "leader-only"
        return PusherAgent(tell="any", before_planning=False, target=target)
    raise ValueError(f"unknown scripted agent: {policy}")


__all__ = [
    "ActionChoice",
    "Agent",
    "CautiousAgent",
    "CounterAgent",
    "GreedyAgent",
    "NeuralAgent",
    "PusherAgent",
    "RandomAgent",
    "ReaderAgent",
    "RuleAgent",
    "LeaverAgent",
    "RusherAgent",
    "ThreatAgent",
    "SCRIPTED",
    "make_scripted_agent",
    "card_front",
    "preferred_blast_choice",
    "safe_tie_break",
    "should_blast",
]
