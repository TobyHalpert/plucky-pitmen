"""Rule-based heuristic policy for Plucky Pitmen with home-column visibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game_env import MineEnv

from game_env import (
    DYNAMITE,
    DRAGON,
    GEM,
    LORRY,
    MAX_COLUMNS,
    PASS,
    PIT_CAGE,
)


def _estimate_dragon_risk(env: MineEnv, player_index: int, backside: int) -> float:
    """Estimate probability that a card with the given backside is a DRAGON.

    Uses strictly player-accessible information (own hand + known used blast cards).
    Each backside pool starts with 2 Dragons, 4 Gems, 4 Dynamites (10 cards total).
    """
    player = env.players[player_index]
    total_dragons_per_back = 2
    total_cards_per_back = 10

    known_dragons = sum(
        1 for card, back in player.collected_cards
        if card == DRAGON and back == backside
    )
    known_cards = sum(
        1 for _, back in player.collected_cards
        if back == backside
    )

    known_cards += sum(
        1 for back in env.used_blast_backsides
        if back == backside
    )

    remaining_dragons = max(0, total_dragons_per_back - known_dragons)
    remaining_cards = max(1, total_cards_per_back - known_cards)

    return remaining_dragons / remaining_cards


def _dragon_under_opponent(env: MineEnv, player_index: int) -> bool:
    """Return True if any opponent stands on a DRAGON in the player's home column."""
    my_column = player_index
    for idx, player in enumerate(env.players):
        if idx == player_index or player.escaped or player.dead:
            continue
        if player.position is None or player.position >= PIT_CAGE:
            continue
        if player.column != my_column:
            continue
        if player.position >= len(env.rows):
            continue
        card_entry = env.rows[player.position][my_column]
        if card_entry is not None and card_entry[0] == DRAGON:
            return True
    return False


def _is_signaled_danger(env: MineEnv, player_index: int) -> bool:
    """Return True if the owner of our current column has fled to signal a danger or escape."""
    player = env.players[player_index]
    my_col = player.column
    
    if my_col is None or my_col == player_index:
        return False
        
    owner = env.players[my_col]
    
    # Check if the owner has fled the mine floor (or escaped)
    owner_fled = owner.position in (PIT_CAGE, LORRY) or owner.escaped
    if owner_fled:
        # Verify the owner wasn't physically displaced by someone else this round
        was_displaced = any(event[0] == my_col for event in env.displacement_events)
        if not was_displaced:
            # They left voluntarily: either they see a dragon or they are securing a sole survival win.
            # In either case, the card under us is unsafe/unproductive to stay on.
            return True
                
    return False


def _is_safe_from_displacement(env: MineEnv, player_index: int, row: int) -> bool:
    """Return True if no active opponent can reach the given row to displace us."""
    for idx, opponent in enumerate(env.players):
        if idx == player_index or opponent.escaped or opponent.dead:
            continue
        
        # Determine opponent's movement depth limits
        opp_pos = opponent.position
        if opp_pos == PIT_CAGE:
            opp_depth = opponent.cage_index if opponent.cage_index is not None else 0
        elif opp_pos == LORRY:
            opp_depth = 0
        else:
            opp_depth = opp_pos
            
        # Opponents can only move strictly shallower than their movement depth.
        # If the opponent can reach 'row', then we are not safe from displacement.
        if row < opp_depth:
            return False
    return True


def _find_displacement_raid(env: MineEnv, player_index: int, legal_actions: list[int]) -> int | None:
    """Find an opportunity to displace an opponent sitting on a known/inferred gem.
    
    A player sitting on their own column stably (not fleeing) has revealed that card
    is safe. If it's a GEM, we can steal it AND set them back to the PIT_CAGE.
    """
    my_column = player_index
    best_raid: int | None = None
    best_raid_score = -1.0
    
    for action in legal_actions:
        if action >= PIT_CAGE:
            continue
        row, col = divmod(action, MAX_COLUMNS)
        if row >= len(env.rows) or col >= len(env.rows[row]):
            continue
        
        # Displacement is only legal if an opponent sits on their own home column
        if not env._can_displace(row, col, player_index):
            continue
        
        card_entry = env.rows[row][col]
        if card_entry is None:
            continue
        
        occupant = env._card_occupant(row, col)
        if occupant is None or occupant == player_index:
            continue
        
        # Determine if the card is a KNOWN or INFERRED gem
        is_gem = False
        confidence = 0.0
        
        if col == my_column:
            # Direct visibility: our own home column front is fully known
            if card_entry[0] == GEM:
                is_gem = True
                confidence = 1.0
        else:
            # Inference: the occupant sits on their own column stably.
            # Estimate risk that it's a dragon (should be near 0 if occupant is rational)
            dragon_risk = _estimate_dragon_risk(env, player_index, card_entry[1])
            # Trust rationality: if a rational opponent sits stably, it's very unlikely a dragon
            adjusted_risk = dragon_risk * 0.1  # Heavy discount due to rational inference
            
            if adjusted_risk < 0.15:
                is_gem = True  # Likely gem/dynamite, both non-fatal
                confidence = 1.0 - adjusted_risk
        
        if not is_gem:
            continue
        
        # Score the raid: displacement bonus + gem value + confidence
        score = 100.0 * confidence
        score += 30.0  # Sabotage bonus: opponent loses their position
        score += 15.0  # Displacement mechanic bonus
        score += row * 5.0  # Prefer deeper rows
        
        if score > best_raid_score:
            best_raid_score = score
            best_raid = action
    
    return best_raid


def deterministic_policy(env: MineEnv) -> int:
    """Choose a legal action for the current planning player."""
    player_index = env.planning_player
    mask = env.action_mask(player_index)
    original_legal_actions = [int(action) for action, allowed in enumerate(mask) if allowed]

    if not original_legal_actions:
        return int(PASS)

    current_player = env.players[player_index]
    current_pos = current_player.position
    has_planned = env.planned_players[player_index]
    my_column = player_index

    # --- 1. DANGER & ESCAPE ASSESSMENT ---
    opponent_blasting = any(
        p.blast_pending for idx, p in enumerate(env.players) if idx != player_index
    )
    
    # Check if we want to exit the mine voluntarily to lock in points
    rich_in_gems = current_player.gems >= 3
    rich_in_loot = len(current_player.collected_cards) >= 5
    has_moderate_loot = (
        current_player.gems >= 2
        or len(current_player.collected_cards) >= 3
    )

    # Danger warnings
    dragon_under_opponent = _dragon_under_opponent(env, player_index)
    signaled_threat = _is_signaled_danger(env, player_index)

    # Physical threat: Opponent is blasting while we hold good items
    emergency_exit = opponent_blasting and has_moderate_loot
    must_escape = rich_in_gems or rich_in_loot or emergency_exit

    # --- 0. STABILITY GUARD ---
    if has_planned and PASS in original_legal_actions:
        if current_pos == LORRY:
            return int(PASS)
        if current_pos == PIT_CAGE and (must_escape or dragon_under_opponent):
            return int(PASS)

    # Escape / Retreat triggers: allow cage if we must escape or are signaling
    if must_escape or dragon_under_opponent or signaled_threat:
        legal_actions = list(original_legal_actions)
    else:
        # Avoid voluntarily retreating to the cage if we have no reason to
        legal_actions = [a for a in original_legal_actions if a != PIT_CAGE]

    # Tactical adjustment: If signaled, we MUST NOT PASS (must step off the dragon)
    if signaled_threat:
        legal_actions = [a for a in legal_actions if a != PASS]

    if not legal_actions:
        legal_actions = list(original_legal_actions)

    # --- 2. DRAGON THREAT IMMEDIATE SIGNAL FLEE ---
    # Signaler action: If an opponent is on a dragon in our column, we flee to signal them.
    if dragon_under_opponent and current_pos not in (PIT_CAGE, LORRY):
        if PIT_CAGE in legal_actions:
            return int(PIT_CAGE)
        if LORRY in legal_actions:
            return int(LORRY)

    # --- 2.5. DISPLACEMENT RAID (STEAL KNOWN/INFERRED GEMS) ---
    if not must_escape and not signaled_threat:
        raid_action = _find_displacement_raid(env, player_index, legal_actions)
        if raid_action is not None:
            return int(raid_action)

    # --- 3. DISPLACED PLAYER RE-ENTRY (from PIT_CAGE) ---
    if current_pos == PIT_CAGE and not dragon_under_opponent and not signaled_threat and not must_escape:
        max_reentry_row = (
            current_player.cage_index
            if current_player.cage_index is not None
            else len(env.rows) - 1
        )

        reentry_candidates: dict[int, list[int]] = {}
        for action in legal_actions:
            if action >= PIT_CAGE:
                continue
            row, col = divmod(action, MAX_COLUMNS)
            if row > max_reentry_row or row >= len(env.rows):
                continue
            reentry_candidates.setdefault(row, []).append(action)

        for row in sorted(reentry_candidates.keys(), reverse=True):
            candidates = reentry_candidates[row]

            best_action: int | None = None
            best_score = -1.0
            for action in candidates:
                _, col = divmod(action, MAX_COLUMNS)
                card_entry = env.rows[row][col]
                if card_entry is None:
                    continue

                if col == my_column:
                    if card_entry[0] == DRAGON:
                        continue
                    safety = 1.0
                else:
                    risk = _estimate_dragon_risk(env, player_index, card_entry[1])
                    safety = 1.0 - risk

                score = safety * 100.0
                if col == my_column:
                    score += 10.0
                    # TACTICAL RULE: Do not choose own column unless safe from displacement
                    if not _is_safe_from_displacement(env, player_index, row):
                        score -= 150.0
                else:
                    # Assess target column owner's location
                    owner = env.players[col]
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
        if current_pos == PIT_CAGE and not must_escape and not dragon_under_opponent:
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
            if row >= len(env.rows) or col >= len(env.rows[row]):
                continue
            card_entry = env.rows[row][col]
            if card_entry is None:
                continue

            if col == my_column:
                dragon_risk = 1.0 if card_entry[0] == DRAGON else 0.0
            else:
                dragon_risk = _estimate_dragon_risk(env, player_index, card_entry[1])

            score = (1.0 - dragon_risk) * 100.0

            # Apply displacement safety rules for own home column
            if col == my_column:
                if not _is_safe_from_displacement(env, player_index, row):
                    score -= 150.0  # Massive penalty to block unsafe home column selection
            else:
                # Evaluate foreign column destination target safety
                owner = env.players[col]
                if owner.position == PIT_CAGE or owner.escaped:
                    score -= 40.0
                elif owner.position == LORRY:
                    score -= 10.0

            if env._can_displace(row, col, player_index):
                score += 15.0

            score += row * 5.0

            if current_player.column is not None and col == current_player.column:
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

    if current_pos == PIT_CAGE and not must_escape and not dragon_under_opponent:
        if LORRY in legal_actions:
            return int(LORRY)
            
    if PASS in legal_actions:
        return int(PASS)
    if PIT_CAGE in original_legal_actions:
        return int(PIT_CAGE)
    if LORRY in legal_actions:
        return int(LORRY)

    return int(legal_actions[0])


def deterministic_blast_choice(env: MineEnv, player_index: int) -> int:
    """Choose which card index (0, 1, or 2) to keep during `env.resolve_blast()`.

    Preference order: GEM > DYNAMITE > any non-DRAGON > 0.
    """
    player = env.players[player_index]
    if not player.blast_cards:
        return 0

    safe_indices = [
        idx for idx, (card_type, _) in enumerate(player.blast_cards)
        if card_type != DRAGON
    ]

    if not safe_indices:
        return 0

    for idx in safe_indices:
        if player.blast_cards[idx][0] == GEM:
            return idx

    for idx in safe_indices:
        if player.blast_cards[idx][0] == DYNAMITE:
            return idx

    return safe_indices[0]