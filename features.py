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
    DRAGON,
    DRAGONS_PER_BACKSIDE,
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
N_OPPONENT_FEATURES = 12
# Layout of the state block, so callers never have to count by hand:
#   0..24  the decision point, 25..33 backside bookkeeping, then one block per
#   tracked opponent, then the table's dragon hazard for this round.
BACKSIDE_BLOCK = 25
OPPONENT_BASE = BACKSIDE_BLOCK + 9
HAZARD_FEATURE = OPPONENT_BASE + MAX_TRACKED_OPPONENTS * N_OPPONENT_FEATURES
N_STATE_FEATURES = HAZARD_FEATURE + 1
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


def cell_dragon_risk(
    ctx: PlayerActionContext, row: int, column: int | None
) -> float | None:
    """How likely one cell is a dragon, from this seat's knowledge.

    ``None`` means there is no card there at all (an empty slot, or a figure
    standing in front of the mine).  A face-up card is certain either way; a
    face-down card is priced by the backside left in the deck.
    """
    if column is None or not 0 <= int(row) < len(ctx.public_rows):
        return None
    if not 0 <= int(column) < len(ctx.public_rows[int(row)]):
        return None
    entry = ctx.public_rows[int(row)][int(column)]
    if entry is None:
        return None
    front, back = entry
    if front is not None:
        return 1.0 if int(front) == DRAGON else 0.0
    _known_dragons, _known_safe, _unknown, risk = backside_stats(ctx)
    return float(risk[int(back)])


def _figure_cell(
    ctx: PlayerActionContext, player_index: int
) -> tuple[int, int | None]:
    """``(row, column)`` of a figure, in the coordinates of the mine."""
    if player_index == ctx.player_index:
        column = ctx.own_column if ctx.own_column is not None else ctx.player_index
        return int(ctx.own_position), int(column)
    for opponent in ctx.opponents:
        if opponent.player_index == player_index:
            column = (
                opponent.column
                if opponent.column is not None
                else opponent.player_index
            )
            return int(opponent.position), int(column)
    return PIT_CAGE, None


def figure_dragon_risk(ctx: PlayerActionContext, player_index: int) -> float:
    """The dragon risk of the card one figure is standing on, as this seat sees it.

    A rival standing in **its own column** is the case that has to be read
    carefully, and getting it wrong is what made the first version of this
    feature useless: that rival can see the card under it, so it will step away
    from a dragon rather than turn one -- to the cage, which is always available
    from a card. Only a figure standing on a card *nobody at the table can see*
    (a raid into somebody else's column) is a dragon risk to the rest of the
    mine. Pricing the two the same way put seed 7's haul-1 round-1 table at a
    61% hazard when no seat at that table was standing on anything it could not
    see, and in 200 re-rolls from that position the haul ended on a dragon 1% of
    the time.
    """
    row, column = _figure_cell(ctx, player_index)
    if not 0 <= row < PIT_CAGE:
        return 0.0
    if player_index != ctx.player_index and column == player_index:
        return 0.0  # the sitter can see this card and will not turn a dragon
    risk = cell_dragon_risk(ctx, row, column)
    return 0.0 if risk is None else risk


def round_dragon_hazard(ctx: PlayerActionContext) -> float:
    """How likely a dragon is woken this round, from this seat's knowledge.

    This is the only thing a pit cage insures against: a dragon ends the haul and
    everybody still in the mine scores for a dead figure, loot and haul bonus
    gone with it.  A figure in front of the mine, in a cage or on a lorry is
    standing on nothing and carries no risk.  A figure standing on a card
    *nobody at the table can see* carries the risk of that card -- and that is
    the only way a dragon is woken in this game, because a seat can see its own
    column and will step away from a dragon rather than turn it (see
    ``figure_dragon_risk``).  A charged blast can wake one whatever the board
    says, so it reads as certain danger.

    It is deliberately a *lower* bound on the truth: a rival is free to walk
    onto a card mid-round, so a table where nobody currently stands on a
    dragon-capable card is more dangerous than zero.  What it does capture is
    the case the seat can read: is anybody -- including itself -- about to turn
    a card that can be a dragon?
    """
    if ctx.own_blast_pending or any(o.blast_pending for o in ctx.opponents):
        return 1.0
    seats = [ctx.player_index]
    seats.extend(
        o.player_index for o in ctx.opponents if not (o.escaped or o.dead)
    )
    survive = 1.0
    for index in seats:
        survive *= 1.0 - figure_dragon_risk(ctx, index)
    return float(1.0 - survive)


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
        # How keen this opponent is on displacing is the one read a seat has on
        # whether standing on a home-column dragon will be taken as bait.
        values[base + 11] = min(opponent.displacements, 3) / 3.0

    # The board only enters the input through the candidate-action block, so a
    # seat cannot see the card under a rival's figure.  That leaves the escape
    # decision blind to the one quantity that makes it right or wrong: whether
    # the mine is about to blow this round (BALANCE.md, "A cage is insurance").
    values[HAZARD_FEATURE] = round_dragon_hazard(ctx)

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
                card_risk = float(risk[int(back)])
            else:
                # 12 = gem, 13 = dragon, 14 = dynamite, 15 = unknown front
                values[12 + _FRONT_SLOT[int(front)]] = 1.0
                # A visible front is not an estimate: it is the answer. Reporting
                # it through the same risk/safety channels as a backside is what
                # keeps a visible dragon from looking as safe as a visible gem.
                card_risk = 1.0 if int(front) == DRAGON else 0.0
            values[16] = card_risk
            values[17] = 1.0 - card_risk

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


def walks_past_a_free_card(ctx: PlayerActionContext) -> set[int]:
    """Face-down digs that pay more than a card they climb past.

    Every planning phase starts with all figures at the front of the mine
    (``_set_players_to_starting_positions``), so ending the round on a deeper
    card buys nothing *from one round to the next*: the only thing that differs
    between two face-down cards is the price on their backside. (It does buy
    something inside the phase — see ``strands_the_figure`` — and that is why the
    rule for cards the seat can see runs the same way.) A card is therefore dominated when a *deeper* free card is at
    least as safe, because the figure walks past that card on the way and could
    have taken it instead. Choosing the dearer or the equally priced one of the
    two cannot be explained to a spectator and is not a trade — the backside is
    the price. Measured on the promoted checkpoint, 4 players, seed-paired:
    +1.5pp win rate against `rule` (800 games), +2.0pp against `reader` (800),
    -1.2pp against `leaver` (400), so the rule costs nothing measurable and the
    point of it is legibility, not strength.

    A card whose front the player can see is not dominated: "a known gem at row
    1" against "an unknown card at row 2" is a real trade, and so is raiding a
    rival who stands on their own card.
    """
    _known_dragons, _known_safe, _unknown, risk = backside_stats(ctx)
    free: list[tuple[int, float, int]] = []
    for action in ctx.legal_actions:
        action = int(action)
        if not 0 <= action < PIT_CAGE:
            continue
        row, column = divmod(action, MAX_COLUMNS)
        if row >= len(ctx.public_rows) or column >= len(ctx.public_rows[row]):
            continue
        entry = ctx.public_rows[row][column]
        if entry is None or entry[0] is not None:
            continue
        if _occupant(ctx, row, column) is not None:
            continue
        free.append((row, float(risk[int(entry[1])]), action))
    hidden: set[int] = set()
    for row, price, action in free:
        if any(
            other_row > row and other_price <= price + 1e-9
            for other_row, other_price, _other in free
        ):
            hidden.add(action)
    return hidden


def own_card_is_known_dragon(ctx: PlayerActionContext) -> bool:
    """True when the figure stands on a card the seat can see is a dragon."""
    row = ctx.own_position
    if not 0 <= row < len(ctx.public_rows):
        return False
    column = ctx.own_column if ctx.own_column is not None else ctx.player_index
    if column != ctx.player_index or column >= len(ctx.public_rows[row]):
        return False
    entry = ctx.public_rows[row][column]
    return entry is not None and entry[0] is not None and int(entry[0]) == DRAGON


def known_safe_card_available(ctx: PlayerActionContext) -> bool:
    """True when the seat can step onto a card it can see is not a dragon.

    Only the seat's own column is face up to it, so this is cheap to read and
    exact: a front that is a gem or dynamite is a card that cannot kill anyone.
    """
    for action in ctx.legal_actions:
        action = int(action)
        if not 0 <= action < PIT_CAGE:
            continue
        row, column = divmod(action, MAX_COLUMNS)
        if column != ctx.player_index or row >= len(ctx.public_rows):
            continue
        entry = ctx.public_rows[row][column]
        if entry is None or entry[0] is None:
            continue
        if int(entry[0]) != DRAGON:
            return True
    return False


def lorry_skips_a_card_worth_taking(ctx: PlayerActionContext) -> bool:
    """True when parking on the lorry is a move with nothing to gain.

    Standing on a lorry collects nothing, and it does not protect the figure
    either: the rules kill **every** player still in the mine when a dragon is
    woken, the ones on their lorry included — only a pit cage removes a figure
    from the haul. So the lorry has exactly one use, which is to skip the card
    under your own figure, and that is only worth a round when that card is a
    dragon you can see. Everywhere else it trades a card (eight cards in ten are
    a gem or a dynamite) for nothing.

    It is also a coordination trap, and the reason this rule exists: each seat
    dodges for itself, so at a table that samples its moves all four can end up
    parked in the same planning phase, and then the haul is over with nobody
    having collected anything (+1 point apiece, if that). With the shipped
    checkpoint in all four seats, 93 of 200 sampled games lost the first haul
    that way (seed 7 among them); under the argmax, which picks the lorry less
    often, none did. See BALANCE.md, "Four nets parked on their lorries".

    A figure standing in a pit cage is decided elsewhere, in ``action_mask``,
    because there the pair lorry/pass has to be read together: the cage leaves
    the mine by itself when the cards are turned, so a pass *is* the departure
    and the lorry is the only move that keeps the haul alive. This rule on its
    own masked the lorry there and left nothing but that pass -- seed 7 of the
    viewer, haul 1, P3 displaced into cage index 0, hazard 0.00, one legal move
    left.
    """
    if ctx.own_position == LORRY:
        return False  # parked already; the move is not legal from here anyway
    return not own_card_is_known_dragon(ctx)


CAGE_ANSWERS_A_HAZARD = 0.15


def cage_banks_nothing(ctx: PlayerActionContext) -> bool:
    """True when leaving through a pit cage would collect nothing and answer nothing.

    The cage is the only exit that takes the figure out of the haul *before* the
    cards are turned and banks the gems in hand, so it is worth a round for
    exactly two reasons: there is loot to bank, **and** there is something to
    escape from. Loot means a gem in hand (dynamite is not scored and survives an
    escape either way). Something to escape from means a blast is charged -- the
    blast ends the haul if it turns a dragon -- or a dragon the seat can see is
    about to be turned: the card under its own figure, a card in its column with
    a rival standing on it, or, at the weakest, a table where somebody is
    standing on a card that can be a dragon (``round_dragon_hazard``).

    The second reason is the one this mask used to miss, and the reason the
    clause has to exist at all: leaving is a **decision that repeats**. The seat
    can leave later in the same planning phase, once the rivals have moved and
    the danger they create is on the board, and it can leave later in the haul
    with more gems in hand. Banking before anybody has committed to a card buys
    nothing that waiting one move does not buy. Do it anyway and the round is
    spent for nothing: seed 7 of the viewer, the same seat in two hauls, one gem
    in hand, nobody standing on a dragon-capable card. See BALANCE.md, "A cage is
    insurance".

    The hazard is a *lower* bound on the danger (a rival is free to walk onto a
    card mid-round), so the threshold sits at the smallest non-trivial reading
    rather than at some share of it.

    The same question is asked of a *pass* taken while already in a cage: a
    figure in a cage leaves the mine when the cards are turned, so passing there
    is the same departure one decision later.

    Everything else is a wasted round, and this is the clause that gets that
    wrong in the obvious implementation: **"somebody is already in a cage"** is
    not a threat. Ordinary raiding displaces rivals into cages all the time, and
    once one seat reads another's cage as a reason to flee, the whole table
    follows: an earlier checkpoint, seed 7, haul 1, ended with every figure in a
    cage and not one gem collected, and the same thing happened in 40% of sampled
    four-net games. See BALANCE.md, "An escape has to bank something".
    """
    if any(front == GEM for front, _ in ctx.private_cards):
        return round_dragon_hazard(ctx) <= CAGE_ANSWERS_A_HAZARD
    if ctx.own_blast_pending or any(o.blast_pending for o in ctx.opponents):
        return False
    if own_card_is_known_dragon(ctx):
        # Refusing the dragon is the point, but spending the haul is not the
        # only way to do it: a card the seat can see is a gem it can take, and
        # one row away is close enough. Seed 7, haul 3, round 0 -- P3 sits on a
        # dragon it can see, holds nothing, plans the cage at 62% while a gem it
        # can see sits at r0c2, and takes that gem at its next decision in the
        # same phase; from that position the gem is worth +2.59 +- 0.12 and the
        # cage +2.23 +- 0.10.
        return known_safe_card_available(ctx)
    for opponent in ctx.opponents:
        if opponent.dead or opponent.escaped or opponent.position == PIT_CAGE:
            continue
        if opponent.column != ctx.player_index:
            continue
        if not 0 <= opponent.position < len(ctx.public_rows):
            continue
        entry = ctx.public_rows[opponent.position][ctx.player_index]
        if entry is not None and entry[0] is not None and int(entry[0]) == DRAGON:
            return False
    return True


_FRONT_NAMES = {GEM: "gem", DRAGON: "dragon", DYNAMITE: "dynamite"}
_BACKSIDE_NAMES = {0: "o", 1: "v", 2: "~"}


def _cage_reason(ctx: PlayerActionContext) -> str:
    """What an escape banks and what it is running from, for the viewer.

    The spectator sees backsides, so the danger a cage answers is often private
    to the seat that takes it: a dragon in its own column is drawn as a plain
    card.  Without this the viewer shows "pit cage" and the move looks arbitrary.
    """
    gems = sum(1 for front, _ in ctx.private_cards if front == GEM)
    why: list[str] = []
    if own_card_is_known_dragon(ctx):
        why.append("the dragon under me")
    if ctx.own_blast_pending:
        why.append("my blast")
    elif any(opponent.blast_pending for opponent in ctx.opponents):
        why.append("a rival's blast")
    for opponent in ctx.opponents:
        if opponent.dead or opponent.escaped:
            continue
        if opponent.column != ctx.player_index:
            continue
        if not 0 <= opponent.position < len(ctx.public_rows):
            continue
        entry = ctx.public_rows[opponent.position][ctx.player_index]
        if entry is not None and entry[0] is not None and int(entry[0]) == DRAGON:
            why.append(f"P{opponent.player_index + 1} is on my dragon")
    if not why:
        why.append(f"{round_dragon_hazard(ctx):.0%} the mine blows this round")
    return f"pit cage, banks {gems}g ({', '.join(why)})"


def action_reason(ctx: PlayerActionContext, action: int) -> str:
    """One short sentence for *why* an action looks the way it does to its player.

    The viewer draws nothing but backsides, so a move that is obviously right to
    the seat that makes it can look arbitrary to the person watching: only the
    owner of a column ever sees the fronts in it. This spells out what the
    deciding seat knew.
    """
    primitive, announces_blast = split_action(int(action))
    suffix = " + blast" if announces_blast else ""
    if primitive == PIT_CAGE:
        return _cage_reason(ctx) + suffix
    if primitive == LORRY:
        refusal = (
            "lorry, refuses the dragon I can see"
            if own_card_is_known_dragon(ctx)
            else "lorry"
        )
        return refusal + suffix
    if primitive == PASS:
        return "pass" + suffix
    row, column = divmod(primitive, MAX_COLUMNS)
    if row >= len(ctx.public_rows) or column >= len(ctx.public_rows[row]):
        return "empty slot" + suffix
    entry = ctx.public_rows[row][column]
    if entry is None:
        return "empty slot" + suffix
    front, back = entry
    _known_dragons, _known_safe, _unknown, risk = backside_stats(ctx)
    chance = float(risk[int(back)])
    occupant = _occupant(ctx, row, column)
    raiding = occupant is not None and occupant != ctx.player_index and occupant == column

    if raiding:
        return f"raid P{int(occupant) + 1}, back {_BACKSIDE_NAMES.get(int(back), '?')} ({chance:.0%} dragon){suffix}"
    if column == ctx.player_index and front is not None:
        return f"own column, {_FRONT_NAMES.get(int(front), 'card')}{suffix}"
    if column == ctx.player_index:
        return f"own column, back {_BACKSIDE_NAMES.get(int(back), '?')} ({chance:.0%} dragon){suffix}"
    return f"back {_BACKSIDE_NAMES.get(int(back), '?')} ({chance:.0%} dragon){suffix}"


def strands_the_figure(ctx: PlayerActionContext) -> set[int]:
    """Own-column gems taken at a row that leaves the seat nowhere to recover.

    A raid pushes a figure into the pit cage with ``cage_index = row it stood
    on``, and from a cage only rows **shallower** than the cage index are legal
    again (``MineEnv._movement_depth``). So the row a card is taken at decides
    what the seat can do if a rival raids that call mid-phase: from row 0 the
    answer is nothing — the only moves left are the lorry and the pass, and a
    pass in a cage is a departure — while from row 3 rows 0-2 are still open.

    When the seat can see two gems in its own column, taking the deeper one
    first cannot be worse: it is the same gem, it is raidable by fewer seats
    (only figures standing at the front of the mine reach a deep card), and a
    raid on it leaves the whole column still reachable. Taking the front gem
    first forfeits that. Seed 7 of the viewer, haul 1, round 1: P1's column holds
    gems at r0, r1 and r3, it takes r0c0 at 55%, and a raid on that call would
    have left it with no card move at all.

    This is the one place the codebase used to argue the other way. The rule
    that hides a face-down dig behind a cheaper deeper one (``walks_past_a_free_card``)
    states that "ending the round on a deeper card buys nothing", which is true
    only because every round resets the figures to the front; mid-phase it buys
    the difference between a live phase and a forced exit. Measured over 200
    four-net games, 209 displacements landed on a figure standing on a card, 114
    of them from row 0 and 136 (65%) left the victim with no card move at all.

    The counter-argument, and the reason this is not free: front cards are the
    contested ones, 507 of 1,588 digs into a rival's column in the same sample
    were row 0 against 166 at row 3, so taking the front gem *claims* it before
    somebody else does. That is a race, not a guarantee, and the deeper gem
    cannot run away in the same way — a rival has to reach the bottom of the
    mine to take it.
    """
    own_gems: dict[int, int] = {}
    for action in ctx.legal_actions:
        primitive = int(action)
        if not 0 <= primitive < PIT_CAGE:
            continue
        row, column = divmod(primitive, MAX_COLUMNS)
        if column != ctx.player_index or row >= len(ctx.public_rows):
            continue
        entry = ctx.public_rows[row][column]
        if entry is None or entry[0] != GEM:
            continue
        own_gems[row] = primitive
    if len(own_gems) < 2:
        return set()
    deepest = max(own_gems)
    return {action for row, action in own_gems.items() if row < deepest}


def digs_past_a_known_gem(ctx: PlayerActionContext) -> set[int]:
    """Face-down digs that a gem the seat can see makes pointless.

    A front of the seat's **own** column is information the rest of the table
    does not have: a card there is a gem or a dynamite, never a dragon. A card
    still face down may be a dragon — two of them per backside — and the seat
    that turns it is the dragon player. So the two are not a trade. Take the
    known gem and compare the branch each way:

    - the face-down card turns out a gem: equal to the gem that was there;
    - it turns out dynamite: it is not scored at all, so worse;
    - it turns out a dragon: the hull is over and its turner scores -2.

    That is weak dominance in every branch, and it holds whether the face-down
    card is in the seat's own column or a rival's. The one thing that could
    outweigh it is a **displacement** — pushing a rival who stands on its own
    column into the cage costs that rival its position — so a dig that displaces
    is left alone, and only an unoccupied face-down cell is hidden. Seed 7 of the
    viewer, haul 1, planning phase 1: P1 has gems at r0 and r1 of its own column
    and takes **r1c1**, an empty cell in P2's column, at 25% dragon risk — P2 was
    standing in P3's column at the time, so there was nobody to displace and the
    move bought nothing but the lottery ticket.

    Note what this rule does **not** need: a threshold. Face-down dragons cost
    25% at that table and 0% at another, and the argument is the same in both,
    because a known gem is 100% a gem.
    """
    own_gem_legal = False
    for action in ctx.legal_actions:
        primitive = int(action)
        if not 0 <= primitive < PIT_CAGE:
            continue
        row, column = divmod(primitive, MAX_COLUMNS)
        if column != ctx.player_index or row >= len(ctx.public_rows):
            continue
        entry = ctx.public_rows[row][column]
        if entry is not None and entry[0] == GEM:
            own_gem_legal = True
            break
    if not own_gem_legal:
        return set()

    hidden: set[int] = set()
    for action in ctx.legal_actions:
        primitive = int(action)
        if not 0 <= primitive < PIT_CAGE:
            continue
        row, column = divmod(primitive, MAX_COLUMNS)
        if row >= len(ctx.public_rows) or column >= len(ctx.public_rows[row]):
            continue
        entry = ctx.public_rows[row][column]
        if entry is None or entry[0] is not None:
            continue
        if _occupant(ctx, row, column) is not None:
            continue
        hidden.add(primitive)
    return hidden


def action_mask(ctx: PlayerActionContext) -> np.ndarray:
    """Legal policy actions, including legal blast announcements.

    Seven narrow rules ride on top of the legal moves, all of them policies rather
    than rules of the game, which is why they live here and not in the
    environment (see BALANCE.md):

    1. A dig onto a dragon the seat can see, with no rival deep enough to punish
       it, is pure loss — hiding it costs nothing measured. The same dig *with* a
       rival deep enough is bait and stays: the rival can displace the figure
       into a pit cage, a safe exit that banks the loot, and then eats the dragon
       itself. ``doomed_dragon_digs``.
    2. A face-down card that pays more than a deeper free card is dominated — the
       backside is the price and the position a round ends on buys nothing.
       ``walks_past_a_free_card``.
    3. The lorry collects nothing and does not protect the figure, so it is only
       a way to refuse a dragon the seat can see under itself.
       ``lorry_skips_a_card_worth_taking``.
    4. The pit cage is the only exit that banks loot before the cards are turned,
       so an escape has to bank something or answer a visible threat.
       ``cage_banks_nothing``.
    5. Of two gems the seat can see in its own column, the deeper one comes
       first: the row a card is taken at is the cage index a raid would leave the
       figure with, and a shallow row leaves it nothing to step back onto.
       ``strands_the_figure``.
    6. A face-down dig is not a trade against a gem the seat can see in its own
       column: the same one card a round, and only one of the two can be a
       dragon. Raids are left alone, because pushing a rival is worth something
       the card is not. ``digs_past_a_known_gem``.
    7. A figure in a pit cage is already halfway out: whoever is still in a cage
       when the cards are turned leaves the mine.  So from a cage the lorry and
       the pass are one decision, not two — the escape is taken when it banks
       something, and refused (staying in the haul on the lorry) when it does
       not.  Before this the lorry was masked in the cage and the only move left
       was a pass, so the seat "chose" an empty departure at 100%.

    Rules 3 and 4 are about wasted rounds, and 4 in particular exists because
    "somebody is already in a cage" is *not* a threat: a raid puts a rival there,
    every seat then reads the herd as danger, and a haul ends with the whole
    table in cages and nothing collected (seed 7 of the viewer, four nets).
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
    passed = {
        action for action in walks_past_a_free_card(ctx) if mask[int(action)]
    }
    if passed and any(int(a) not in passed and mask[int(a)] for a in ctx.legal_actions):
        for primitive in passed:
            mask[primitive] = False
            mask[join_action(primitive, True)] = False
    if int(ctx.own_position) == PIT_CAGE:
        # From a cage the escape is already half taken: a figure still in a cage
        # when the cards are turned leaves the mine.  So the lorry/pass pair has
        # to be read together.  A departure that banks nothing is refused and the
        # seat stays in the haul on the lorry (seed 7, haul 1: P3 displaced into
        # cage index 0, hazard 0.00, lorry +0.93 +- 0.11 against pass +0.55 +- 0.10);
        # one that is worth taking keeps the pass and drops the lorry.
        trial = mask.copy()
        if cage_banks_nothing(ctx):
            trial[int(PASS)] = False
            trial[int(LORRY)] = True
        else:
            trial[int(LORRY)] = False
        if trial.any():
            mask = trial
    elif mask[int(LORRY)] and lorry_skips_a_card_worth_taking(ctx):
        trial = mask.copy()
        trial[int(LORRY)] = False
        if trial.any():
            mask = trial
    ranked = {action for action in strands_the_figure(ctx) if mask[int(action)]}
    if ranked and any(int(a) not in ranked and mask[int(a)] for a in ctx.legal_actions):
        for primitive in ranked:
            mask[primitive] = False
            mask[join_action(primitive, True)] = False
    lottery = {action for action in digs_past_a_known_gem(ctx) if mask[int(action)]}
    if lottery:
        trial = mask.copy()
        for primitive in lottery:
            trial[primitive] = False
            trial[join_action(primitive, True)] = False
        keeps_a_gem = any(
            0 <= int(a) < PIT_CAGE
            and int(a) % MAX_COLUMNS == ctx.player_index
            and ctx.public_rows[int(a) // MAX_COLUMNS][ctx.player_index] is not None
            and ctx.public_rows[int(a) // MAX_COLUMNS][ctx.player_index][0] == GEM
            for a in ctx.legal_actions
            if trial[int(a)]
        )
        if keeps_a_gem:
            mask = trial
    if mask[int(PIT_CAGE)] and cage_banks_nothing(ctx):
        trial = mask.copy()
        trial[int(PIT_CAGE)] = False
        trial[join_action(int(PIT_CAGE), True)] = False
        if trial.any():
            mask = trial
    return mask


__all__ = [
    "CAGE_ANSWERS_A_HAZARD",
    "N_ACTIONS",
    "N_ACTION_FEATURES",
    "N_BASE_ACTIONS",
    "N_INPUT_FEATURES",
    "N_STATE_FEATURES",
    "OPPONENT_BASE",
    "action_features",
    "action_mask",
    "action_reason",
    "backside_stats",
    "cage_banks_nothing",
    "cell_dragon_risk",
    "digs_face_up_dragon",
    "digs_past_a_known_gem",
    "doomed_dragon_digs",
    "figure_dragon_risk",
    "has_dragon_free_move",
    "join_action",
    "known_safe_card_available",
    "lorry_skips_a_card_worth_taking",
    "own_card_is_known_dragon",
    "rival_can_push",
    "round_dragon_hazard",
    "split_action",
    "state_action_matrix",
    "state_features",
    "strands_the_figure",
    "walks_past_a_free_card",
]
