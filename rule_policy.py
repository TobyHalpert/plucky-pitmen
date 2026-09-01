# rule_policy.py
import numpy as np
from game_env import MineEnv, PASS, PIT_CAGE, LORRY, DRAGON

def deterministic_policy(env: MineEnv) -> int:
    """Choose a legal move without inspecting hidden card fronts.

    This policy is intentionally limited to public information: legal actions,
    visible positions of other players, and the current player's own status.
    """
    mask = env.action_mask()
    legal_actions = [action for action, allowed in enumerate(mask) if allowed]

    if not legal_actions:
        return PASS

    player_index = env.planning_player
    current_player = env.players[player_index]
    current_pos = current_player.position if current_player.position is not None else -1

    other_player_rows = []
    for idx, player in enumerate(env.players):
        if idx != player_index and not player.escaped and not player.dead:
            if player.position is not None and player.position < len(env.rows):
                other_player_rows.append(player.position)

    other_player_rows.sort(reverse=True)

    for target_row in other_player_rows:
        for action in legal_actions:
            if action < PIT_CAGE and (action // 5) == target_row:
                return action

    if current_pos >= 0 and current_pos < len(env.rows):
        for action in legal_actions:
            if action < PIT_CAGE and (action // 5) == current_pos:
                return action

    deeper_actions = [a for a in legal_actions if a < PIT_CAGE and (a // 5) > current_pos]
    if deeper_actions:
        return min(deeper_actions, key=lambda a: a // 5)

    shallower_actions = [a for a in legal_actions if a < PIT_CAGE]
    if shallower_actions:
        return max(shallower_actions)

    for fallback_action in (PASS, LORRY, PIT_CAGE):
        if fallback_action in legal_actions:
            return fallback_action

    return legal_actions[0]
