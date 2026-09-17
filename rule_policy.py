"""Rule-based heuristic policy for Plucky Pitmen.
"""

from __future__ import annotations

from game_env import (
    DYNAMITE,
    DRAGON,
    GEM,
    LORRY,
    MAX_COLUMNS,
    PASS,
    PIT_CAGE,
    DRAGONS_PER_BACKSIDE,
    TOTAL_PER_BACKSIDE,
    PlayerActionContext,
    OpponentPublicView,
)


# ----------------------------------------------------------------------
# Small helpers over the context
# ----------------------------------------------------------------------


def _opponent_by_index(ctx: PlayerActionContext, index: int) -> OpponentPublicView | None:
    """Return the public view of the opponent with the given index, or None."""
    for opp in ctx.opponents:
        if opp.player_index == index:
            return opp
    return None


def _active_opponents(ctx: PlayerActionContext) -> list[OpponentPublicView]:
    return [opp for opp in ctx.opponents if not opp.escaped and not opp.dead]


# ----------------------------------------------------------------------
# Information-based inference
# ----------------------------------------------------------------------


def _estimate_dragon_risk(ctx: PlayerActionContext, backside: int) -> float:
    """Estimate the probability that an unknown-front card with the given
    backside is a DRAGON.

    Dragons are never collected — pulling one ends the haul. Known non-dragons
    therefore come from four sources (all publicly derivable, except own-column
    fronts which are private to the requesting player):

    - Own collected cards with this backside (definitely non-dragons).
    - Opponents' collected cards with this backside (backsides are public).
    - Used blast backsides matching this backside (definitely dynamite).
    - Non-dragon fronts visible in the player's own home column with this backside.

    Known dragons can only come from dragons visible in the player's own
    home column, since dragons never enter any player's hand.
    """
    my_column = ctx.player_index

    known_dragons = 0
    known_non_dragons = 0

    # Own collected cards: never dragons.
    for _, back in ctx.private_cards:
        if back == backside:
            known_non_dragons += 1

    # Opponents' collected cards: backsides are public, fronts are hidden.
    # The fronts are guaranteed to be non-dragons regardless.
    for opp in ctx.opponents:
        for back in opp.collected_backsides:
            if back == backside:
                known_non_dragons += 1

    # Used blast backsides: always dynamite.
    for back in ctx.used_blast_backsides:
        if back == backside:
            known_non_dragons += 1

    # Own home column: fronts are fully visible.
    for row in ctx.public_rows:
        if my_column >= len(row):
            continue
        card_entry = row[my_column]
        if card_entry is None or card_entry[1] != backside:
            continue
        front = card_entry[0]
        if front == DRAGON:
            known_dragons += 1
        else:
            known_non_dragons += 1

    remaining_dragons = max(0, DRAGONS_PER_BACKSIDE - known_dragons)
    unknown_cards = TOTAL_PER_BACKSIDE - known_dragons - known_non_dragons

    if unknown_cards <= 0:
        return 0.0
    return remaining_dragons / unknown_cards


def _dragon_under_opponent(ctx: PlayerActionContext) -> bool:
    """Return True if any opponent stands on a DRAGON in the player's home column.

    Fronts in the player's own home column are visible, so this is fully
    derivable from the context without any cheating.
    """
    my_column = ctx.player_index
    for opp in ctx.opponents:
        if opp.escaped or opp.dead:
            continue
        if opp.position is None or opp.position >= PIT_CAGE:
            continue
        if opp.column != my_column:
            continue
        if opp.position >= len(ctx.public_rows):
            continue
        card_entry = ctx.public_rows[opp.position][my_column]
        if card_entry is not None and card_entry[0] == DRAGON:
            return True
    return False


def _is_signaled_danger(ctx: PlayerActionContext) -> bool:
    """Return True if the owner of our current column has fled to signal a
    danger or an escape.
    """
    my_col = ctx.own_column
    if my_col is None or my_col == ctx.player_index:
        return False

    owner = _opponent_by_index(ctx, my_col)
    if owner is None:
        return False

    owner_fled = owner.position in (PIT_CAGE, LORRY) or owner.escaped
    if not owner_fled:
        return False

    # Verify the owner was not physically displaced by another player this round.
    was_displaced = any(event[0] == my_col for event in ctx.displacement_events)
    if was_displaced:
        return False

    # Owner left voluntarily: either they see a dragon or they are securing
    # a sole-survivor win. Either way, staying on the card is bad for us.
    return True


def _is_safe_from_displacement(ctx: PlayerActionContext, row: int) -> bool:
    """Return True if no active opponent can reach the given row to displace us."""
    for opp in _active_opponents(ctx):
        opp_pos = opp.position
        if opp_pos == PIT_CAGE:
            opp_depth = opp.cage_index if opp.cage_index is not None else 0
        elif opp_pos == LORRY:
            opp_depth = 0
        else:
            opp_depth = opp_pos

        # Opponents can only move strictly shallower than their movement depth.
        if row < opp_depth:
            return False
    return True


def _card_occupant(ctx: PlayerActionContext, row: int, column: int) -> int | None:
    """Return the player index standing on (row, column), or None.

    Considers both the requesting player and the visible opponents.
    """
    if (
        ctx.own_position == row
        and ctx.own_column == column
    ):
        return ctx.player_index
    for opp in ctx.opponents:
        if opp.escaped or opp.dead:
            continue
        if opp.position == row and opp.column == column:
            return opp.player_index
    return None


def _can_displace(ctx: PlayerActionContext, row: int, column: int) -> bool:
    """Return True if we can displace whoever stands at (row, column).

    Displacement is only legal against an opponent standing on their own
    home column (occupant_index == column).
    """
    occupant = _card_occupant(ctx, row, column)
    return (
        occupant is not None
        and occupant != ctx.player_index
        and occupant == column
    )


# ----------------------------------------------------------------------
# Displacement raid detection
# ----------------------------------------------------------------------


def _find_displacement_raid(
    ctx: PlayerActionContext, legal_actions: list[int]
) -> int | None:
    """Find a raid on an opponent sitting on a known/inferred gem.

    A player who sits stably on their own home column has effectively
    revealed the card is safe (they would not sit on a dragon). We can
    steal that card by displacement and knock them back to the pit cage.
    """
    my_column = ctx.player_index
    best_raid: int | None = None
    best_raid_score = -1.0

    for action in legal_actions:
        if action >= PIT_CAGE:
            continue
        row, col = divmod(action, MAX_COLUMNS)
        if row >= len(ctx.public_rows) or col >= len(ctx.public_rows[row]):
            continue

        if not _can_displace(ctx, row, col):
            continue

        card_entry = ctx.public_rows[row][col]
        if card_entry is None:
            continue

        occupant = _card_occupant(ctx, row, col)
        if occupant is None or occupant == ctx.player_index:
            continue

        # Determine if the card is a KNOWN or INFERRED gem
        is_gem = False
        confidence = 0.0

        if col == my_column:
            # Direct visibility: own home column front is fully known.
            if card_entry[0] == GEM:
                is_gem = True
                confidence = 1.0
        else:
            # Inference: occupant sits stably on their own column, so it
            # is very unlikely a dragon.
            dragon_risk = _estimate_dragon_risk(ctx, card_entry[1])
            adjusted_risk = dragon_risk * 0.1
            if adjusted_risk < 0.15:
                is_gem = True
                confidence = 1.0 - adjusted_risk

        if not is_gem:
            continue

        score = 100.0 * confidence
        score += 30.0  # Sabotage bonus: opponent loses their position
        score += 15.0  # Displacement mechanic bonus
        score += row * 5.0  # Prefer deeper rows

        if score > best_raid_score:
            best_raid_score = score
            best_raid = action

    return best_raid


# ----------------------------------------------------------------------
# Main policy
# ----------------------------------------------------------------------


def deterministic_policy(ctx: PlayerActionContext) -> int:
    """Choose a legal action from a cheat-safe context."""
    original_legal_actions = list(ctx.legal_actions)

    if not original_legal_actions:
        return int(PASS)

    current_pos = ctx.own_position
    my_column = ctx.player_index

    # A pass is only legal after the player has already made a plan this round.
    has_planned = PASS in original_legal_actions

    # --- 1. DANGER & ESCAPE ASSESSMENT ---
    opponent_blasting = any(opp.blast_pending for opp in ctx.opponents)

    # PRIVATE INFO: only the player itself can count its own gems/dynamite.
    own_gems = sum(1 for card, _ in ctx.private_cards if card == GEM)
    own_card_count = len(ctx.private_cards)

    rich_in_gems = own_gems >= 3
    rich_in_loot = own_card_count >= 5
    has_moderate_loot = own_gems >= 2 or own_card_count >= 3

    dragon_under_opponent = _dragon_under_opponent(ctx)
    signaled_threat = _is_signaled_danger(ctx)

    emergency_exit = opponent_blasting and has_moderate_loot
    must_escape = rich_in_gems or rich_in_loot or emergency_exit

    # --- 0. STABILITY GUARD ---
    if has_planned:
        if current_pos == LORRY:
            return int(PASS)
        if current_pos == PIT_CAGE and (must_escape or dragon_under_opponent):
            return int(PASS)

    # Escape / retreat triggers: allow cage only when needed.
    if must_escape or dragon_under_opponent or signaled_threat:
        legal_actions = list(original_legal_actions)
    else:
        legal_actions = [a for a in original_legal_actions if a != PIT_CAGE]

    # Tactical adjustment: if signaled, we MUST NOT PASS.
    if signaled_threat:
        legal_actions = [a for a in legal_actions if a != PASS]

    if not legal_actions:
        legal_actions = list(original_legal_actions)

    # --- 2. DRAGON THREAT IMMEDIATE SIGNAL FLEE ---
    if dragon_under_opponent and current_pos not in (PIT_CAGE, LORRY):
        if PIT_CAGE in legal_actions:
            return int(PIT_CAGE)
        if LORRY in legal_actions:
            return int(LORRY)

    # --- 2.5. DISPLACEMENT RAID (STEAL KNOWN/INFERRED GEMS) ---
    if not must_escape and not signaled_threat:
        raid_action = _find_displacement_raid(ctx, legal_actions)
        if raid_action is not None:
            return int(raid_action)

    # --- 3. DISPLACED PLAYER RE-ENTRY (from PIT_CAGE) ---
    if (
        current_pos == PIT_CAGE
        and not dragon_under_opponent
        and not signaled_threat
        and not must_escape
    ):
        max_reentry_row = (
            ctx.own_cage_index
            if ctx.own_cage_index is not None
            else len(ctx.public_rows) - 1
        )

        reentry_candidates: dict[int, list[int]] = {}
        for action in legal_actions:
            if action >= PIT_CAGE:
                continue
            row, col = divmod(action, MAX_COLUMNS)
            if row > max_reentry_row or row >= len(ctx.public_rows):
                continue
            reentry_candidates.setdefault(row, []).append(action)

        for row in sorted(reentry_candidates.keys(), reverse=True):
            candidates = reentry_candidates[row]

            best_action: int | None = None
            best_score = -1.0
            for action in candidates:
                _, col = divmod(action, MAX_COLUMNS)
                card_entry = ctx.public_rows[row][col]
                if card_entry is None:
                    continue

                if col == my_column:
                    if card_entry[0] == DRAGON:
                        continue
                    safety = 1.0
                else:
                    risk = _estimate_dragon_risk(ctx, card_entry[1])
                    safety = 1.0 - risk

                score = safety * 100.0
                if col == my_column:
                    score += 10.0
                    if not _is_safe_from_displacement(ctx, row):
                        score -= 150.0
                else:
                    owner = _opponent_by_index(ctx, col)
                    if owner is not None:
                        if owner.position == PIT_CAGE or owner.escaped:
                            score -= 40.0
                        elif owner.position == LORRY:
                            score -= 10.0

                if score > best_score:
                    best_score = score
                    best_action = action

            if best_action is not None:
                return int(best_action)

        if LORRY in legal_actions:
            return int(LORRY)

    # --- 4. CONFIRMATION / STALLING ---
    if has_planned and PASS in legal_actions:
        if emergency_exit and current_pos < PIT_CAGE:
            if PIT_CAGE in legal_actions:
                return int(PIT_CAGE)
        if (
            current_pos == PIT_CAGE
            and not must_escape
            and not dragon_under_opponent
        ):
            pass
        else:
            return int(PASS)

    # --- 5. UNCONDITIONAL ESCAPE (To PIT_CAGE, not LORRY) ---
    if must_escape:
        if PIT_CAGE in legal_actions:
            return int(PIT_CAGE)
        if PASS in legal_actions:
            return int(PASS)

    # --- 6. EVALUATE MINE GRID MOVES ---
    mine_actions = [a for a in legal_actions if a < PIT_CAGE]
    if mine_actions:
        scored_moves: list[tuple[float, int]] = []

        for action in mine_actions:
            row, col = divmod(action, MAX_COLUMNS)
            if row >= len(ctx.public_rows) or col >= len(ctx.public_rows[row]):
                continue
            card_entry = ctx.public_rows[row][col]
            if card_entry is None:
                continue

            if col == my_column:
                dragon_risk = 1.0 if card_entry[0] == DRAGON else 0.0
            else:
                dragon_risk = _estimate_dragon_risk(ctx, card_entry[1])

            score = (1.0 - dragon_risk) * 100.0

            if col == my_column:
                if not _is_safe_from_displacement(ctx, row):
                    score -= 150.0
            else:
                owner = _opponent_by_index(ctx, col)
                if owner is not None:
                    if owner.position == PIT_CAGE or owner.escaped:
                        score -= 40.0
                    elif owner.position == LORRY:
                        score -= 10.0

            if _can_displace(ctx, row, col):
                score += 15.0

            score += row * 5.0

            if ctx.own_column is not None and col == ctx.own_column:
                score -= 5.0

            scored_moves.append((score, action))

        if scored_moves:
            scored_moves.sort(key=lambda x: x[0], reverse=True)
            best_score, best_action = scored_moves[0]

            if best_score < 60.0:
                if has_moderate_loot and PIT_CAGE in original_legal_actions:
                    return int(PIT_CAGE)
                if LORRY in legal_actions:
                    return int(LORRY)
                return int(best_action)

            return int(best_action)

    # --- 7. FALLBACK SEQUENCE ---
    if signaled_threat:
        if LORRY in legal_actions:
            return int(LORRY)
        if PIT_CAGE in original_legal_actions:
            return int(PIT_CAGE)

    if (
        current_pos == PIT_CAGE
        and not must_escape
        and not dragon_under_opponent
    ):
        if LORRY in legal_actions:
            return int(LORRY)

    if PASS in legal_actions:
        return int(PASS)
    if PIT_CAGE in original_legal_actions:
        return int(PIT_CAGE)
    if LORRY in legal_actions:
        return int(LORRY)

    return int(legal_actions[0])


# ----------------------------------------------------------------------
# Blast resolution
# ----------------------------------------------------------------------


def deterministic_blast_choice(blast_cards: tuple[tuple[int, int], ...]) -> int:
    """Choose which card index to keep during `env.resolve_blast()`.

    Takes the player's own blast draws (which are the ONLY thing they are
    allowed to inspect) rather than the environment. Preference order:
    GEM > DYNAMITE > any non-DRAGON > 0.
    """
    if not blast_cards:
        return 0

    safe_indices = [
        idx for idx, (card_type, _) in enumerate(blast_cards)
        if card_type != DRAGON
    ]

    if not safe_indices:
        return 0

    for idx in safe_indices:
        if blast_cards[idx][0] == GEM:
            return idx

    for idx in safe_indices:
        if blast_cards[idx][0] == DYNAMITE:
            return idx

    return safe_indices[0]