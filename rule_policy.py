# rule_policy.py
import numpy as np
from game_env import MineEnv, PASS, PIT_CAGE, LORRY, DRAGON

def deterministic_policy(env: MineEnv) -> int:
    """Wählt Züge basierend auf zwei kooperativen Regeln aus den legalen Optionen."""
    mask = env.action_mask()
    legal_actions = [action for action, allowed in enumerate(mask) if allowed]
    
    if not legal_actions:
        return PASS

    player_index = env.planning_player
    current_player = env.players[player_index]
    current_pos = current_player.position if current_player.position is not None else -1

    # =========================================================================
    # REGEL 2: DRACHEN-WARNUNG (KOOPERATIVE NOTFLUCHT)
    # Wenn ein ANDERER Spieler auf MEINEM bekannten Drachen sitzt -> Sofort flüchten!
    # =========================================================================
    if PIT_CAGE in legal_actions:
        for row_idx, row in enumerate(env.rows):
            if player_index < len(row) and row[player_index] is not None:
                card, _backside = row[player_index]
                if card == DRAGON:
                    # Prüfe, ob wirklich jemand Fremdes auf dieser exakten Position sitzt
                    if env._card_occupied(row_idx, player_index, ignore_player=player_index):
                        return PIT_CAGE

    # =========================================================================
    # REGEL 1: REIHEN-LOGIK (NUR AUS LEGAL_ACTIONS WÄHLEN)
    # =========================================================================
    
    # 1. Schritt: Sammle die Reihen aller anderen aktiven Spieler im Schacht
    other_player_rows = []
    for idx, p in enumerate(env.players):
        if idx != player_index and not p.escaped and not p.dead:
            if p.position is not None and p.position < len(env.rows):
                other_player_rows.append(p.position)
                
    # Sortiere absteigend (unterste besetzte Reihe zuerst)
    other_player_rows.sort(reverse=True)

    # 2. Schritt: Versuche, auf eine Reihe eines anderen Spielers aufzuschließen
    for target_row in other_player_rows:
        for action in legal_actions:
            if action < PIT_CAGE and (action // 5) == target_row:
                return action

    # 3. Schritt: Wenn dort nichts frei ist, plündere die eigene aktuelle Reihe
    if current_pos >= 0 and current_pos < len(env.rows):
        for action in legal_actions:
            if action < PIT_CAGE and (action // 5) == current_pos:
                return action

    # 4. Schritt: Wenn der aktuelle Bereich voll ist, rücke tiefer in den Schacht vor!
    # Wir filtern ALLE legalen Schachtkarten, die tiefer liegen als unsere aktuelle Position.
    deeper_actions = [a for a in legal_actions if a < PIT_CAGE and (a // 5) > current_pos]
    if deeper_actions:
        # Wir sortieren so, dass wir die am nächsten gelegene Reihe bevorzugen
        return min(deeper_actions, key=lambda a: a // 5)

    # 5. Schritt: Wenn nach unten alles blockiert ist, nimm eine beliebige höhere Schachtkarte
    # Hauptsache wir bleiben im Schacht, solange keine Drachengefahr droht!
    shallower_actions = [a for a in legal_actions if a < PIT_CAGE]
    if shallower_actions:
        return max(shallower_actions)  # Nimm die tiefst-mögliche der höheren Karten

    # =========================================================================
    # ABSOLUTER FALLBACK (NUR WENN DER SCHACHT REIN MATHEMATISCH BLOCKIERT IST)
    # =========================================================================
    # Wenn wir auf der Startlinie festsitzen und keine Schachtkarte legal angeboten wird,
    # weichen wir kooperativ aus (zuerst PASS, um Platz zu machen, dann LORRY/PIT_CAGE).
    for fallback_action in (PASS, LORRY, PIT_CAGE):
        if fallback_action in legal_actions:
            return fallback_action
            
    return legal_actions[0]
