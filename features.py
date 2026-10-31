"""Seat-invariant, cheat-safe feature encoding for Plucky Pitmen policies.

Everything here is derived from :class:`game_env.PlayerActionContext` only.
That context already hides opponent card fronts and opponent private blast
draws, so an encoder built on top of it can never leak hidden information.

The encoded policy input is ``[state_features | action_features]``.  The same
state vector is repeated for every candidate action and differs only in the
action block, which makes the encoding usable both by a linear softmax policy
and by a small MLP trained with policy gradients.
"""

from __future__ import annotations

import numpy as np

from game_env import (
    BACKSIDE_A,
    BACKSIDE_B,
    BACKSIDE_C,
    DRAGONS_PER_BACKSIDE,
    DRAGON,
    DYNAMITE,
    GEM,
    HAULS_PER_GAME,
    LORRY,
    MAX_COLUMNS,
    MAX_ROWS,
    PASS,
    PIT_CAGE,
    TOTAL_PER_BACKSIDE,
    PlayerActionContext,
)

# The environment's primitive action space ends at PASS (27 for a 5x5 mine).
N_BASE_ACTIONS = PASS + 1
# A learned policy may additionally announce a blast together with its move.
BLAST_OFFSET = N_BASE_ACTIONS
N_ACTIONS = 2 * N_BASE_ACTIONS

BACKSIDES = (BACKSIDE_A, BACKSIDE_B, BACKSIDE_C)
_FRONT_SLOT = {GEM: 0, DRAGON: 1, DYNAMITE: 2}
MAX_TRACKED_OPPONENTS = 4
N_OPPONENT_FEATURES = 11
N_STATE_FEATURES = 25 + 9 + MAX_TRACKED_OPPONENTS * N_OPPONENT_FEATURES
N_ACTION_FEATURES = 21
N_INPUT_FEATURES = N_STATE_FEATURES + N_ACTION_FEATURES


def split_action(action: int) -> tuple[int, bool]:
    """Return ``(primitive_action, announces_blast)`` for a policy action."""
    action = int(action)
    return action % N_BASE_ACTIONS, action >= N_BASE_ACTIONS


def join_action(primitive_action: int, announces_blast: bool) -> int:
    return int(primitive_action) + (BLAST_OFFSET if announces_blast else 0)


# ----------------------------------------------------------------------
# Backside bookkeeping (public knowledge + the player's own cards)
# ----------------------------------------------------------------------
def backside_stats(
    ctx: PlayerActionContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(known_dragons, known_safe, unknown, dragon_risk)`` per backside.

    A player knows a card is *not* a dragon when it is in their own collection,
    in an opponent's collection (dragon draws end the haul, so collected cards
    are never dragons), was used for a blast, or lies face up in their own
    home column.  Dragons are only ever known from the own home column.
    """
    known_dragons = np.zeros(3, dtype=np.float32)
    known_safe = np.zeros(3, dtype=np.float32)

    own_column = ctx.player_index
    for row in ctx.public_rows:
        if own_column >= len(row):
            continue
        entry = row[own_column]
        if entry is None:
            continue
        front, back = entry
        if front is None:
            continue
        if front == DRAGON:
            known_dragons[back] += 1
        else:
            known_safe[back] += 1

    for front, back in ctx.private_cards:
        if front != DRAGON:
            known_safe[back] += 1

    for opponent in ctx.opponents:
        for back in opponent.collected_backsides:
            known_safe[back] += 1

    for back in ctx.used_blast_backsides:
        known_safe[back] += 1

    unknown = np.maximum(
        0.0,
        TOTAL_PER_BACKSIDE - known_dragons - known_safe,
    )
    remaining_dragons = np.maximum(0.0, DRAGONS_PER_BACKSIDE - known_dragons)
    with np.errstate(divide="ignore", invalid="ignore"):
        risk = np.where(unknown > 0, remaining_dragons / np.maximum(unknown, 1), 0.0)
    return known_dragons, known_safe, unknown, risk.astype(np.float32)


def _own_depth(ctx: PlayerActionContext) -> int:
    if ctx.own_position == PIT_CAGE:
        return int(ctx.own_cage_index or 0)
    if ctx.own_position == LORRY:
        return 0
    return int(ctx.own_position)


def _visible_home_counts(ctx: PlayerActionContext) -> tuple[int, int, int, int]:
    gems = dynamite = dragons = unknown = 0
    own_column = ctx.player_index
    for row in ctx.public_rows:
        if own_column >= len(row):
            continue
        entry = row[own_column]
        if entry is None:
            continue
        front, _back = entry
        if front == DRAGON:
            dragons += 1
        elif front == GEM:
            gems += 1
        elif front == DYNAMITE:
            dynamite += 1
        elif front is None:
            unknown += 1
    return gems, dynamite, dragons, unknown


# ----------------------------------------------------------------------
# State features
# ----------------------------------------------------------------------
def state_features(ctx: PlayerActionContext) -> np.ndarray:
    """Encode the decision point without reference to a particular action."""
    n_players = max(1, ctx.n_players)
    active = sum(1 for o in ctx.opponents if not o.escaped and not o.dead)
    active += 0 if ctx.own_position is None else 1  # the requesting player
    active = min(active, n_players)

    own_gems = sum(1 for front, _ in ctx.private_cards if front == GEM)
    own_dynamite = sum(1 for front, _ in ctx.private_cards if front == DYNAMITE)
    own_cards = len(ctx.private_cards)

    opponent_scores = [o.score for o in ctx.opponents]
    best_other_score = max(opponent_scores) if opponent_scores else ctx.own_score

    own_row = ctx.own_position if 0 <= ctx.own_position < PIT_CAGE else -1
    on_own_column = ctx.own_column is not None and ctx.own_column == ctx.player_index
    cage_index = ctx.own_cage_index if ctx.own_cage_index is not None else -1

    visible_gems, visible_dynamite, visible_dragons, visible_unknown = (
        _visible_home_counts(ctx)
    )
    known_dragons, _known_safe, unknown, risk = backside_stats(ctx)

    values = np.zeros(N_STATE_FEATURES, dtype=np.float32)

    values[0] = ctx.haul / HAULS_PER_GAME
    values[1] = ctx.round / MAX_ROWS
    values[2] = n_players / 5.0
    values[3] = active / n_players
    values[4] = ctx.cards_remaining_in_deck / 30.0
    values[5] = ctx.own_score / 10.0
    values[6] = (ctx.own_score - best_other_score) / 10.0
    values[7] = own_gems / 5.0
    values[8] = own_dynamite / 3.0
    values[9] = own_cards / 8.0
    values[10] = float(own_row >= 0)
    values[11] = max(own_row, 0) / MAX_ROWS
    values[12] = float(ctx.own_position == PIT_CAGE)
    values[13] = float(ctx.own_position == LORRY)
    values[14] = max(cage_index, 0) / MAX_ROWS
    values[15] = float(on_own_column)
    values[16] = float(ctx.own_blast_pending)
    values[17] = float(any(o.blast_pending for o in ctx.opponents))
    values[18] = len(ctx.used_blast_backsides) / 6.0
    values[19] = max(1, active - 1) / n_players
    values[20] = visible_gems / 5.0
    values[21] = visible_dynamite / 5.0
    values[22] = visible_dragons / 2.0
    values[23] = visible_unknown / 5.0
    values[24] = float(PASS in ctx.legal_actions)

    for index in range(3):
        base = 25 + index * 3
        values[base] = risk[index]
        values[base + 1] = known_dragons[index] / DRAGONS_PER_BACKSIDE
        values[base + 2] = unknown[index] / TOTAL_PER_BACKSIDE

    displaced = {event[0] for event in ctx.displacement_events}
    ordered = sorted(
        ctx.opponents,
        key=lambda o: (o.player_index - ctx.player_index) % n_players,
    )
    slot_base = 34
    for slot in range(MAX_TRACKED_OPPONENTS):
        base = slot_base + slot * N_OPPONENT_FEATURES
        if slot >= len(ordered):
            continue
        opponent = ordered[slot]
        row = opponent.position if 0 <= opponent.position < PIT_CAGE else -1
        values[base] = 1.0
        values[base + 1] = opponent.score / 10.0
        values[base + 2] = max(row, 0) / MAX_ROWS
        values[base + 3] = float(opponent.position == PIT_CAGE)
        values[base + 4] = float(opponent.position == LORRY)
        values[base + 5] = float(opponent.escaped)
        values[base + 6] = float(opponent.dead)
        values[base + 7] = opponent.card_count / 8.0
        values[base + 8] = float(opponent.blast_pending)
        values[base + 9] = float(
            opponent.column is not None
            and opponent.column == opponent.player_index
        )
        values[base + 10] = float(opponent.player_index in displaced)

    return values


# ----------------------------------------------------------------------
# Action features
# ----------------------------------------------------------------------
def action_features(
    ctx: PlayerActionContext,
    action: int,
    risk: np.ndarray | None = None,
) -> np.ndarray:
    primitive, announces_blast = split_action(action)
    values = np.zeros(N_ACTION_FEATURES, dtype=np.float32)
    if risk is None:
        _known_dragons, _known_safe, _unknown, risk = backside_stats(ctx)

    n_players = max(1, ctx.n_players)
    depth = _own_depth(ctx)
    own_column = ctx.player_index

    if primitive < PIT_CAGE:
        row, column = divmod(primitive, MAX_COLUMNS)
        entry = None
        if row < len(ctx.public_rows) and column < len(ctx.public_rows[row]):
            entry = ctx.public_rows[row][column]

        values[0] = 1.0
        values[4] = row / MAX_ROWS
        values[5] = np.clip((depth - row) / MAX_ROWS, -1.0, 1.0)
        values[6] = float(column == own_column)
        values[7] = ((column - own_column) % n_players) / n_players

        if entry is not None:
            front, back = entry
            values[8 + int(back)] = 1.0
            if front is None:
                values[15] = 1.0
                values[16] = float(risk[int(back)])
                values[17] = 1.0 - float(risk[int(back)])
            else:
                # 12 = gem, 13 = dragon, 14 = dynamite, 15 = unknown front
                values[12 + _FRONT_SLOT[int(front)]] = 1.0
                values[17] = 1.0

            occupant = _occupant(ctx, row, column)
            if occupant is not None:
                values[18] = 1.0
                values[19] = float(
                    occupant != ctx.player_index and occupant == column
                )
    elif primitive == PIT_CAGE:
        values[1] = 1.0
    elif primitive == LORRY:
        values[2] = 1.0
    else:
        values[3] = 1.0

    values[20] = float(announces_blast)
    return values


def _occupant(ctx: PlayerActionContext, row: int, column: int) -> int | None:
    if ctx.own_position == row and ctx.own_column == column:
        return ctx.player_index
    for opponent in ctx.opponents:
        if opponent.escaped or opponent.dead:
            continue
        if opponent.position == row and opponent.column == column:
            return opponent.player_index
    return None


# ----------------------------------------------------------------------
# Matrices for batched policy evaluation
# ----------------------------------------------------------------------
def state_action_matrix(
    ctx: PlayerActionContext, state: np.ndarray | None = None
) -> np.ndarray:
    """Return an ``(N_ACTIONS, N_INPUT_FEATURES)`` policy input matrix."""
    if state is None:
        state = state_features(ctx)
    _known_dragons, _known_safe, _unknown, risk = backside_stats(ctx)
    matrix = np.empty((N_ACTIONS, N_INPUT_FEATURES), dtype=np.float32)
    matrix[:, :N_STATE_FEATURES] = state
    for action in range(N_ACTIONS):
        matrix[action, N_STATE_FEATURES:] = action_features(ctx, action, risk=risk)
    return matrix


def digs_face_up_dragon(ctx: PlayerActionContext, primitive: int) -> bool:
    """True when ``primitive`` digs a card the player can already see is a dragon.

    Only the player's own column shows its fronts, so only a dig there can be a
    *known* dragon. Rivals see nothing but the backside, which makes the card a
    lever rather than a mistake: ``rival_can_push`` tells the two cases apart.
    """
    primitive = int(primitive)
    if primitive < 0 or primitive >= PIT_CAGE:
        return False
    row, column = divmod(primitive, MAX_COLUMNS)
    if column != ctx.player_index:
        return False
    if row >= len(ctx.public_rows) or column >= len(ctx.public_rows[row]):
        return False
    entry = ctx.public_rows[row][column]
    return entry is not None and entry[0] == DRAGON


def rival_can_push(ctx: PlayerActionContext, primitive: int) -> bool:
    """Whether a rival is deep enough to displace the player off ``primitive``.

    A rival standing on row ``p`` may name any card in rows ``< p``, and may
    displace an opponent who occupies a card in that opponent's own column. So
    sitting on an own-column dragon is only *bait* while somebody is deeper in
    the mine: the push sends the bait into a pit cage, which is a safe exit that
    banks the gems already collected, and hands the dragon to the rival.
    """
    primitive = int(primitive)
    if primitive < 0 or primitive >= PIT_CAGE:
        return False
    row, _column = divmod(primitive, MAX_COLUMNS)
    return any(
        not opponent.escaped
        and not opponent.dead
        and row < opponent.position < MAX_ROWS
        for opponent in ctx.opponents
    )


def has_dragon_free_move(ctx: PlayerActionContext) -> bool:
    """Whether any legal action avoids a dragon the player can already see."""
    return any(
        not digs_face_up_dragon(ctx, int(action)) for action in ctx.legal_actions
    )


def doomed_dragon_digs(ctx: PlayerActionContext) -> set[int]:
    """Face-up-dragon digs that no rival can punish: the digger eats it itself."""
    return {
        int(action)
        for action in ctx.legal_actions
        if digs_face_up_dragon(ctx, action) and not rival_can_push(ctx, action)
    }


def action_mask(ctx: PlayerActionContext) -> np.ndarray:
    """Legal policy actions, including legal blast announcements.

    Planning onto a card the player can see is a dragon is usually *bait*, not a
    blunder: while a rival stands deeper in the mine, that rival can displace
    the bait into a pit cage — a safe exit that banks the gems — and then eats
    the dragon itself. The mask keeps that option, and hides only the variant
    with no rival deep enough to push: there the digger eats the dragon itself,
    which is the one reading of the move that really is a pure loss. See
    BALANCE.md, "The dragon in your own column is bait, not a blunder".
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)
    own_dynamite = sum(1 for front, _ in ctx.private_cards if front == DYNAMITE)
    can_blast = own_dynamite >= 2 and not ctx.own_blast_pending
    for primitive in ctx.legal_actions:
        primitive = int(primitive)
        if not 0 <= primitive < N_BASE_ACTIONS:
            continue
        mask[primitive] = True
        if can_blast and primitive < PIT_CAGE:
            mask[join_action(primitive, True)] = True
    doomed = doomed_dragon_digs(ctx)
    if doomed and any(int(a) not in doomed for a in ctx.legal_actions):
        for primitive in doomed:
            mask[primitive] = False
            mask[join_action(primitive, True)] = False
    return mask


__all__ = [
    "N_ACTIONS",
    "N_ACTION_FEATURES",
    "N_BASE_ACTIONS",
    "N_INPUT_FEATURES",
    "N_STATE_FEATURES",
    "action_features",
    "action_mask",
    "digs_face_up_dragon",
    "doomed_dragon_digs",
    "has_dragon_free_move",
    "rival_can_push",
    "backside_stats",
    "join_action",
    "split_action",
    "state_action_matrix",
    "state_features",
]
