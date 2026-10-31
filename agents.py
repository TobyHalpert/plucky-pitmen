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
from features import backside_stats, split_action


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


def should_blast(mode: str, ctx: PlayerActionContext) -> bool:
    """Shared blast heuristic for scripted agents.

    ``never``  - the baseline rule policy never blasts.
    ``rich``   - blast once two gems are safe in hand.
    ``always`` - blast whenever two dynamite are available.
    """
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
SCRIPTED = ("random", "greedy", "cautious", "rule", "rusher", "leaver", "threat")


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
    if policy == "rusher":
        return RusherAgent()
    if policy == "leaver":
        return LeaverAgent()
    if policy == "threat":
        return ThreatAgent(blast=blast)
    raise ValueError(f"unknown scripted agent: {policy}")


__all__ = [
    "ActionChoice",
    "Agent",
    "CautiousAgent",
    "GreedyAgent",
    "NeuralAgent",
    "RandomAgent",
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
