"""Gymnasium environment for Plucky Pitmen."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


GEM = 0
DRAGON = 1
DYNAMITE = 2
MAX_ROWS = 5
MAX_COLUMNS = 5
PIT_CAGE = MAX_ROWS * MAX_COLUMNS
LORRY = PIT_CAGE + 1
PASS = LORRY + 1

BACKSIDE_A = 0
BACKSIDE_B = 1
BACKSIDE_C = 2
BACKSIDE_UNKNOWN = 3

HAULS_PER_GAME = 3
PIT_CAGE_REWARD = 0.0
LORRY_REWARD = -0.05


@dataclass(frozen=True)
class PlayerActionContext:
    """Public, player-scoped view for agent behaviour.
    """

    player_index: int
    legal_actions: tuple[int, ...]
    public_rows: tuple[tuple[tuple[int | None, int] | None, ...], ...]
    private_cards: tuple[tuple[int, int], ...]
    blast_pending_by_player: tuple[bool, ...]  # Which players are blasting (public)
    own_blast_cards: tuple[tuple[int, int], ...] = ()  # Only for requesting player
    used_blast_backsides: tuple[int, ...] = ()  # Dynamite backs used during this haul


@dataclass
class Player:
    position: int = 3
    column: int | None = None
    cage_index: int | None = None
    collected_cards: list[tuple[int, int]] = field(default_factory=list)
    escaped: bool = False
    dead: bool = False
    score: int = 0
    blast_pending: bool = False
    blast_cards: list[tuple[int, int]] = field(default_factory=list)

    @property
    def collected_backsides(self) -> list[int]:
        """Return the back sides from the cards this player collected."""
        return [backside for _, backside in self.collected_cards]

    @property
    def gems(self) -> int:
        """Return the number of gems in the collected cards."""
        return sum(card == GEM for card, _ in self.collected_cards)

    @property
    def dynamite(self) -> int:
        """Return the number of dynamite cards collected."""
        return sum(card == DYNAMITE for card, _ in self.collected_cards)


class MineEnv(gym.Env[np.ndarray, np.int64]):

    metadata = {"render_modes": []}

    def __init__(self, opponents: int = 3, opponent_policy: str = "random") -> None:
        if opponents not in (2, 3, 4):
            raise ValueError("opponents must be 2, 3, or 4")
        if opponent_policy not in {"simple", "random", "cautious", "greedy"}:
            raise ValueError("unknown opponent policy")
        self.n_players = opponents + 1
        self.opponent_policy = opponent_policy
        self.action_space = spaces.Discrete(PASS + 1)
        max_rows = MAX_ROWS
        max_visible_cards = max_rows * MAX_COLUMNS
        # High bound must cover PIT_CAGE (25) / LORRY (26) in `position`
        # and unbounded-ish counters such as planning_turns.
        self.observation_space = spaces.Box(
            0, 127, shape=(15 + max_visible_cards * 2,), dtype=np.int8
        )
        self.rng = np.random.default_rng()
        self.players: list[Player] = []
        self.rows: list[list[tuple[int, int] | None]] = []
        self.cards_remaining: list[tuple[int, int]] = []
        self.round = 0
        self.phase = 0
        self.last_action = PASS
        self.last_reward = 0.0
        self.consecutive_passes = 0
        self.planning_turns = 0
        self.starting_player = 0
        self.planning_player = 0
        self.planned_players: list[bool] = [False] * self.n_players
        self.haul = 1
        self.tie_breaker_contenders: list[int] = []
        self.tie_breaker_turn = 0
        self.used_blast_backsides: list[int] = []
        self.displacement_events: list[tuple[int, int]] = []
        self.game_finished = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.players = [Player() for _ in range(self.n_players)]
        self.cards_remaining = self._new_deck()
        self.rng.shuffle(self.cards_remaining)
        self.rows = self._deal_next_rows(3)
        self.round = 0
        self.phase = 0
        self.last_action = PASS
        self.last_reward = 0.0
        self.consecutive_passes = 0
        self.planning_turns = 0
        starter_rng = np.random.default_rng(seed)
        self.starting_player = int(starter_rng.integers(self.n_players))
        self.haul = 1
        self.tie_breaker_contenders = []
        self.tie_breaker_turn = 0
        self.used_blast_backsides = []
        self.displacement_events = []
        self.planning_player = self.starting_player
        self.planned_players = [False] * self.n_players
        self.game_finished = False
        return self._observation(), {"round": self.round, "haul": self.haul}

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")
        if self._game_over():
            return self._observation(), 0.0, True, False, {"game_over": True}
        if not self._legal(action, player_index=0):
            return self._observation(), -0.15, False, False, {"illegal_action": True}

        self.last_action = action
        self.planning_turns += 1
        self.planned_players[0] = True
        planning_reward = self._planning_reward(action)
        if self.players[0].blast_pending and action != PASS:
            self._cancel_blast(0)
        if action == PASS:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0
            self._apply_action(0, action)

        if action != PASS:
            for player_index in range(1, self.n_players):
                player = self.players[player_index]
                if player.escaped or player.dead:
                    continue
                self.planning_turns += 1
                opponent_action = self._opponent_action(player_index)
                self.planned_players[player_index] = True
                if opponent_action == PASS:
                    continue
                self._apply_action(player_index, opponent_action)
        else:
            for _player_index in range(1, self.n_players):
                if self.consecutive_passes >= self._pass_threshold():
                    break
                self.consecutive_passes += 1

        should_execute = self.consecutive_passes >= self._pass_threshold()
        if not should_execute:
            reward = self.last_reward if action == PASS else planning_reward
            self.last_reward = reward
            return self._observation(), reward, False, False, {
                "round": self.round,
                "action": action,
                "outcome": "planning",
                "consecutive_passes": self.consecutive_passes,
            }

        self.consecutive_passes = 0
        self.planning_turns = 0
        reward, info = self._execute_round()
        self.last_reward = reward
        reward, terminated = self._finish_round(reward, info)
        return self._observation(), reward, terminated, False, info

    def step_one_player(self, action: int | None = None, *, announce_blast: bool = False):
        """Advance exactly one player's planning decision."""
        if self._game_over():
            return self._observation(), 0.0, True, False, {"game_over": True}

        active_players = [
            index for index, player in enumerate(self.players)
            if not player.escaped and not player.dead
        ]
        if not active_players:
            return self._observation(), 0.0, True, False, {"game_over": True}
        if self.planning_player not in active_players:
            self.planning_player = active_players[0]

        player_index = self.planning_player
        player = self.players[player_index]
        if player_index == 0:
            if action is None:
                action = PASS
            action = int(action)
            if not self.action_space.contains(action) or not self._legal(action, player_index):
                return self._observation(), 0.0, False, False, {"illegal_action": True}
        else:
            action = self._opponent_action(player_index)

        if announce_blast:
            self.begin_blast(player_index)

        self.last_action = action
        self.planning_turns += 1
        self.planned_players[player_index] = True
        planning_reward = self._planning_reward(action, player_index)
        if player.blast_pending and action != PASS and not announce_blast:
            self._cancel_blast(player_index)
        if action == PASS:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0
            self._apply_action(player_index, action)

        next_players = [
            index for offset in range(1, self.n_players + 1)
            if (index := (player_index + offset) % self.n_players) in active_players
        ]
        self.planning_player = next_players[0] if next_players else active_players[0]
        if self.consecutive_passes < len(active_players):
            reward = self.last_reward if action == PASS else planning_reward
            self.last_reward = reward
            return self._observation(), reward, False, False, {
                "round": self.round,
                "action": action,
                "player": player_index,
                "outcome": "planning",
            }

        self.consecutive_passes = 0
        self.planning_turns = 0
        self.planning_player = 0
        self.planned_players = [False] * self.n_players
        reward, info = self._execute_round()
        self.last_reward = reward
        reward, terminated = self._finish_round(reward, info)

        if not terminated and not self._haul_just_ended(info):
            active_next_round = [
                i for i, p in enumerate(self.players) if not p.escaped and not p.dead
            ]
            if active_next_round:
                self.planning_player = min(
                    active_next_round,
                    key=lambda x: (x - self.starting_player) % self.n_players,
                )
            else:
                self.planning_player = self.starting_player

        info["player"] = player_index
        return self._observation(), reward, terminated, False, info

    def _haul_just_ended(self, info: dict[str, Any]) -> bool:
        return "haul" in info

    def _finish_round(self, reward: float, info: dict[str, Any]) -> tuple[float, bool]:
        """Shared post-resolution bookkeeping for both stepping APIs."""
        mine_empty = not self._mine_has_cards()
        haul_over = self._haul_over()

        if not self._can_deal_row() or mine_empty:
            info["mine_empty"] = True

        if not haul_over and not any(player.escaped for player in self.players):
            self.rows.append(self._deal_next_row_in_turn_order(self.starting_player))

        self._advance_starting_player()
        self.planning_player = self.starting_player

        if not haul_over:
            self.round += 1
            self._set_players_to_starting_positions()
            return reward, False

        # --- Haul is over -------------------------------------------------
        self._return_used_blast_cards()
        self._prepare_haul_scoring(info)
        self._score_haul(info)
        info["haul"] = self.haul
        info["game"] = f"{self.haul}/{HAULS_PER_GAME}"
        info["scores"] = [player.score for player in self.players]

        if self.haul < HAULS_PER_GAME:
            self._start_next_haul()
            info["next_haul"] = self.haul
            return reward, False

        # --- Final haul: decide the game ---------------------------------
        if self._begin_tie_breaker():
            info["tie_break"] = True
            info["tie_break_contenders"] = list(self.tie_breaker_contenders)
            return reward, False

        self.game_finished = True
        info["game_over"] = True
        info["winner"] = self._leader()
        return self._game_reward(), True

    def _apply_action(self, player_index: int, action: int) -> None:
        player = self.players[player_index]
        if action < PIT_CAGE:
            row, column = divmod(action, MAX_COLUMNS)
            self._displace(row, column, player_index)
            player.position = row
            player.column = column
            player.cage_index = None
            return

        previous_position = player.position
        player.position = action
        player.column = None
        if action == PIT_CAGE:
            if player.cage_index is None:
                player.cage_index = (
                    0 if previous_position == LORRY
                    else min(previous_position, len(self.rows) - 1)
                )
            else:
                player.cage_index = max(0, player.cage_index - 1)
        else:
            player.cage_index = None

    def _advance_starting_player(self) -> None:
        self.starting_player = (self.starting_player + 1) % self.n_players

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _planning_reward(self, action: int, player_index: int = 0) -> float:
        if player_index != 0:
            return 0.0
        if action == PIT_CAGE:
            return PIT_CAGE_REWARD
        if action == LORRY:
            return LORRY_REWARD
        if action == PASS:
            return self.last_reward
        return 0.0

    def _leader(self) -> int:
        highest_score = max(player.score for player in self.players)
        return next(
            index for index, player in enumerate(self.players)
            if player.score == highest_score
        )

    def _game_reward(self) -> float:
        highest_score = max(player.score for player in self.players)
        return 1.0 if self.players[0].score == highest_score else 0.0

    # ------------------------------------------------------------------
    # Haul lifecycle
    # ------------------------------------------------------------------

    def _set_players_to_starting_positions(self) -> None:
        starting_position = len(self.rows)
        for player in self.players:
            if not player.escaped and not player.dead:
                player.position = starting_position
                player.column = None

    def _prepare_haul_scoring(self, info: dict[str, Any]) -> None:
        # On a dragon every survivor is already flagged escaped and every
        # victim is already flagged dead, so there is nothing left to assign.
        if info.get("outcome") == "dragon":
            return
        active = [player for player in self.players if not player.escaped and not player.dead]
        if info.get("mine_empty") or len(active) <= 1:
            for player in active:
                player.escaped = True

    def _score_haul(self, info: dict[str, Any]) -> None:
        escaped = [player for player in self.players if player.escaped]
        if escaped:
            highest_gems = max(player.gems for player in escaped)
            gem_winners = [player for player in escaped if player.gems == highest_gems]
            if len(escaped) == 1 or len(gem_winners) == 1:
                gem_winners[0].score += 3 if self.haul < HAULS_PER_GAME else 4
            else:
                for player in gem_winners:
                    player.score += 1

        dragon_player = info.get("dragon_player")
        no_one_escaped = not escaped
        for player_index, player in enumerate(self.players):
            if dragon_player == player_index:
                player.score -= 2
            elif player.dead:
                player.score += (
                    2 if no_one_escaped and info.get("outcome") == "dragon" else -1
                )
            player.score = max(-1, player.score)

    def _return_used_blast_cards(self) -> None:
        self.cards_remaining.extend(
            (DYNAMITE, backside) for backside in self.used_blast_backsides
        )
        self.rng.shuffle(self.cards_remaining)

    def _start_next_haul(self) -> None:
        retained_dynamite_cards = [
            [card for card in player.collected_cards if card[0] == DYNAMITE]
            if not player.dead else []
            for player in self.players
        ]
        scores = [player.score for player in self.players]
        self.haul += 1
        self.players = [
            Player(collected_cards=dynamite_cards, score=score)
            for dynamite_cards, score in zip(retained_dynamite_cards, scores)
        ]
        self.cards_remaining = self._new_deck()
        self.rng.shuffle(self.cards_remaining)
        self.rows = self._deal_next_rows(3)
        self.round = 0
        self.phase = 0
        self.last_action = PASS
        self.last_reward = 0.0
        self.consecutive_passes = 0
        self.planning_turns = 0
        self.planning_player = self.starting_player
        self.planned_players = [False] * self.n_players
        self.tie_breaker_contenders = []
        self.tie_breaker_turn = 0
        self.used_blast_backsides = []
        self.displacement_events = []

    # ------------------------------------------------------------------
    # Deck & mine
    # ------------------------------------------------------------------

    def _new_deck(self) -> list[tuple[int, int]]:
        return [
            (card, backside)
            for backside in (BACKSIDE_A, BACKSIDE_B, BACKSIDE_C)
            for card in ([DRAGON] * 2 + [GEM] * 4 + [DYNAMITE] * 4)
        ]

    def _deal_next_rows(self, depth: int) -> list[list[tuple[int, int]]]:
        cards = self.cards_remaining[:depth * self.n_players]
        self.cards_remaining = self.cards_remaining[len(cards):]
        return [
            cards[row * self.n_players : (row + 1) * self.n_players]
            for row in range(depth)
        ]

    def _deal_next_row_in_turn_order(self, starting_player: int) -> list[tuple[int, int]]:
        cards = self._deal_next_rows(1)[0]
        row: list[tuple[int, int] | None] = [None] * self.n_players
        for offset, card in enumerate(cards):
            player_index = (starting_player + offset) % self.n_players
            row[player_index] = card
        return row

    def _can_deal_row(self) -> bool:
        return len(self.cards_remaining) >= self.n_players

    def _mine_has_cards(self) -> bool:
        return any(card is not None for row in self.rows for card in row)

    def _pass_threshold(self) -> int:
        return max(
            1, sum(not player.escaped and not player.dead for player in self.players) - 1
        )

    # ------------------------------------------------------------------
    # Legality
    # ------------------------------------------------------------------

    def action_mask(self, player_index: int = 0) -> np.ndarray:
        return np.array(
            [self._legal(action, player_index) for action in range(self.action_space.n)],
            dtype=np.int8,
        )

    def _legal(self, action: int, player_index: int = 0) -> bool:
        player = self.players[player_index]
        if player.dead:
            return False
        if player.escaped:
            return False
        if action < PIT_CAGE:
            if player.position == LORRY:
                return False
            row, column = divmod(action, MAX_COLUMNS)
            movement_depth = self._movement_depth(player)
            return (
                row < len(self.rows)
                and row < movement_depth
                and column < len(self.rows[row])
                and self.rows[row][column] is not None
                and (
                    not self._card_occupied(row, column, ignore_player=player_index)
                    or self._can_displace(row, column, player_index)
                )
            )
        if action == PIT_CAGE:
            return player.position != PIT_CAGE and player.position >= 0
        if action == LORRY:
            return player.position != LORRY and player.position >= 0
        if action == PASS:
            return self.planned_players[player_index]
        return False

    def _movement_depth(self, player: Player) -> int:
        if player.position == PIT_CAGE:
            return player.cage_index if player.cage_index is not None else 0
        if player.position == LORRY:
            return 0
        return player.position

    def _card_occupied(self, row: int, column: int, ignore_player: int | None = None) -> bool:
        return any(
            player_index != ignore_player
            and not player.escaped
            and not player.dead
            and player.position == row
            and player.column == column
            for player_index, player in enumerate(self.players)
        )

    def _card_occupant(self, row: int, column: int) -> int | None:
        for player_index, player in enumerate(self.players):
            if (
                not player.escaped
                and not player.dead
                and player.position == row
                and player.column == column
            ):
                return player_index
        return None

    def _can_displace(self, row: int, column: int, player_index: int) -> bool:
        occupant = self._card_occupant(row, column)
        return occupant is not None and occupant != player_index and occupant == column

    def _displace(self, row: int, column: int, player_index: int) -> None:
        if not self._can_displace(row, column, player_index):
            return
        occupant = self._card_occupant(row, column)
        if occupant is not None:
            cage_index = min(self.players[occupant].position, len(self.rows) - 1)
            self.players[occupant].position = PIT_CAGE
            self.players[occupant].column = None
            self.players[occupant].cage_index = cage_index
            self.displacement_events.append((occupant, player_index))

    # ------------------------------------------------------------------
    # Opponents
    # ------------------------------------------------------------------

    def _opponent_action(self, player_index: int) -> int:
        player = self.players[player_index]

        if self.opponent_policy == "simple":
            from rule_policy import deterministic_policy
            return deterministic_policy(self)

        if player.position in (PIT_CAGE, LORRY):
            if not self.planned_players[player_index]:
                if player.position == LORRY:
                    return PIT_CAGE
                return LORRY
            return PASS
        movement_depth = self._movement_depth(player)
        available_cards = [
            row * MAX_COLUMNS + column
            for row, cards in enumerate(self.rows)
            if row < movement_depth
            for column in range(len(cards))
            if cards[column] is not None
            if (
                not self._card_occupied(row, column, ignore_player=player_index)
                or self._can_displace(row, column, player_index)
            )
        ]
        if self.opponent_policy == "cautious" and self.round > 0:
            if movement_depth > 0 and self.rng.random() < 0.35:
                return PIT_CAGE
        if self.opponent_policy == "greedy":
            nearest_cards = [
                action for action in available_cards
                if action // MAX_COLUMNS == movement_depth - 1
            ]
            if nearest_cards:
                return int(self.rng.choice(nearest_cards))
        if available_cards:
            return int(self.rng.choice(available_cards))
        return PIT_CAGE if player.position > 0 else LORRY

    # ------------------------------------------------------------------
    # Round resolution
    # ------------------------------------------------------------------

    def _execute_round(self) -> tuple[float, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {"round": self.round, "action": self.last_action}
        learner_was_escaped = self.players[0].escaped
        player_order = [
            (self.starting_player + offset) % self.n_players
            for offset in range(self.n_players)
        ]

        # Phase 1: everyone standing in the pit cage leaves the mine and is now
        # immune to any dragon revealed later in this round.
        for player_index in player_order:
            player = self.players[player_index]
            if player.escaped or player.dead:
                continue
            if player.position == PIT_CAGE:
                player.escaped = True

        # Phase 2: reveal cards. A dragon ends the haul immediately.
        for player_index in player_order:
            player = self.players[player_index]
            if player.escaped or player.dead:
                continue
            if (
                player.position == LORRY
                or player.position >= len(self.rows)
                or player.column is None
            ):
                continue
            card_entry = self.rows[player.position][player.column]
            if card_entry is None:
                continue
            card, backside = card_entry
            self.rows[player.position][player.column] = None
            player.collected_cards.append((card, backside))

            if card == DRAGON:
                for remaining_player in self.players:
                    if not remaining_player.escaped:
                        remaining_player.dead = True
                info["outcome"] = "dragon"
                info["dragon_player"] = player_index
                if player_index == 0:
                    reward -= 2.0
                elif self.players[0].dead:
                    # The learner was still in the mine and died as well.
                    reward -= 1.0
                break
            if card == GEM:
                if player_index == 0:
                    reward += 0.25
            else:
                if player_index == 0:
                    reward += 0.1

        if self.players[0].escaped and not learner_was_escaped:
            reward += 0.5 + self.players[0].gems
            if not info.get("outcome"):
                info["outcome"] = "escaped"
        elif not info.get("outcome"):
            info["outcome"] = "continued"

        if info["outcome"] != "dragon":
            self._prepare_blasts()
        return reward, info

    def _haul_over(self) -> bool:
        """Return True when the current haul must stop."""
        # A death ends the haul at once: nobody has to sit and wait.
        if any(player.dead for player in self.players):
            return True
        active = [p for p in self.players if not p.escaped and not p.dead]
        if not active:
            return True
        if not self._mine_has_cards():
            return True
        if not self._can_deal_row():
            return True
        return False

    def _game_over(self) -> bool:
        """Return True once the final haul is scored and any tie-break is settled."""
        return self.game_finished

    # ------------------------------------------------------------------
    # Blasting
    # ------------------------------------------------------------------

    def begin_blast(self, player_index: int) -> None:
        """Announce a blast during planning for a player."""
        if not 0 <= player_index < self.n_players:
            raise IndexError("player_index out of range")
        player = self.players[player_index]
        if player.dynamite < 2:
            raise ValueError("player needs at least two dynamite cards to blast")
        if player.blast_pending:
            return
        player.blast_cards = []
        player.blast_pending = True

    def resolve_blast(self, player_index: int, choice_index: int) -> tuple[int, int]:
        """Choose one of the blast cards kept after execution."""
        if not 0 <= player_index < self.n_players:
            raise IndexError("player_index out of range")
        player = self.players[player_index]
        if not player.blast_pending:
            raise ValueError("no blast in progress")
        if not 0 <= choice_index < len(player.blast_cards):
            raise ValueError("blast choice out of range")

        chosen_card = player.blast_cards.pop(choice_index)
        if chosen_card[0] == DRAGON:
            for remaining_player in self.players:
                if not remaining_player.escaped:
                    remaining_player.dead = True
            player.blast_cards = []
            player.blast_pending = False
            info: dict[str, Any] = {"outcome": "dragon", "dragon_player": player_index}
            self._return_used_blast_cards()
            self._prepare_haul_scoring(info)
            self._score_haul(info)
            if self.haul < HAULS_PER_GAME:
                self._start_next_haul()
            elif not self._begin_tie_breaker():
                self.game_finished = True
            return chosen_card

        player.collected_cards.append(chosen_card)
        if len(player.blast_cards) == 1:
            self.cards_remaining.append(player.blast_cards.pop())
            player.blast_pending = False
        return chosen_card

    def _cancel_blast(self, player_index: int) -> None:
        player = self.players[player_index]
        player.blast_pending = False
        player.blast_cards = []

    def _prepare_blasts(self) -> None:
        for player in self.players:
            if not player.blast_pending:
                continue
            dynamite_removed = 0
            remaining_cards = []
            for card in player.collected_cards:
                if card[0] == DYNAMITE and dynamite_removed < 2:
                    dynamite_removed += 1
                    self.used_blast_backsides.append(card[1])
                else:
                    remaining_cards.append(card)
            player.collected_cards = remaining_cards

            draw_count = min(3, len(self.cards_remaining))
            player.blast_cards = self.cards_remaining[:draw_count]
            self.cards_remaining = self.cards_remaining[draw_count:]
            if draw_count < 3:
                player.collected_cards.extend(player.blast_cards)
                player.blast_cards = []
                player.blast_pending = False

    # ------------------------------------------------------------------
    # Tie-breaker
    # ------------------------------------------------------------------

    def _begin_tie_breaker(self) -> bool:
        highest_score = max(player.score for player in self.players)
        contenders = [
            index for index, player in enumerate(self.players)
            if player.score == highest_score
        ]
        if len(contenders) < 2:
            self.tie_breaker_contenders = []
            return False
        self.tie_breaker_contenders = contenders
        first_contender = next(
            (self.starting_player + offset) % self.n_players
            for offset in range(self.n_players)
            if (self.starting_player + offset) % self.n_players in contenders
        )
        self.tie_breaker_turn = contenders.index(first_contender)
        return True

    def resolve_tie_breaker(
        self, source: str, row: int | None = None, column: int | None = None,
    ) -> dict[str, Any]:
        """Draw a tie-break card for the current contender."""
        if len(self.tie_breaker_contenders) < 2:
            raise ValueError("no tie-breaker is active")
        player_index = self.tie_breaker_contenders[self.tie_breaker_turn]
        if source == "deck":
            if not self.cards_remaining:
                raise ValueError("the draw pile is empty")
            card = self.cards_remaining.pop(0)
        elif source == "mine":
            if row is None or column is None:
                raise ValueError("a mine row and column are required")
            if not 0 <= row < len(self.rows) or not 0 <= column < len(self.rows[row]):
                raise ValueError("mine card out of range")
            card = self.rows[row][column]
            if card is None:
                raise ValueError("no card at selected mine position")
            self.rows[row][column] = None
        else:
            raise ValueError("tie-break source must be 'mine' or 'deck'")

        result = {
            "player": player_index,
            "card": card,
            "source": source,
            "outcome": "safe",
        }
        if card[0] == DRAGON:
            result["outcome"] = "dragon"
            self.tie_breaker_contenders.pop(self.tie_breaker_turn)
            if len(self.tie_breaker_contenders) == 1:
                result["winner"] = self.tie_breaker_contenders[0]
                result["game_over"] = True
                self.tie_breaker_contenders = []
                self.game_finished = True
                return result
            self.tie_breaker_turn %= len(self.tie_breaker_contenders)
        else:
            self.players[player_index].collected_cards.append(card)
            self.tie_breaker_turn = (
                (self.tie_breaker_turn + 1) % len(self.tie_breaker_contenders)
            )
        return result

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def player_view(self, player_index: int) -> dict[str, Any]:
        """Return the information a specific player is allowed to inspect."""
        if not 0 <= player_index < self.n_players:
            raise IndexError("player_index out of range")

        player = self.players[player_index]
        public_rows = []
        for row in self.rows[:MAX_ROWS]:
            public_row = []
            for col, card_entry in enumerate(row[:MAX_COLUMNS]):
                if card_entry is None:
                    public_row.append(None)
                elif col == player_index:
                    public_row.append((card_entry[0], card_entry[1]))
                else:
                    public_row.append((None, card_entry[1]))
            public_rows.append(public_row)

        return {
            "rows": public_rows,
            "private_cards": list(player.collected_cards),
            "player_index": player_index,
            "blast_pending_by_player": [p.blast_pending for p in self.players],
            "own_blast_cards": list(player.blast_cards) if player.blast_pending else [],
            "used_blast_backsides": list(self.used_blast_backsides),
        }

    def player_action_context(self, player_index: int) -> PlayerActionContext:
        """Return the public action context for a given player."""
        if not 0 <= player_index < self.n_players:
            raise IndexError("player_index out of range")

        public_rows = []
        for row in self.rows[:MAX_ROWS]:
            public_row = []
            for col, card_entry in enumerate(row[:MAX_COLUMNS]):
                if card_entry is None:
                    public_row.append(None)
                elif col == player_index:
                    public_row.append((card_entry[0], card_entry[1]))
                else:
                    public_row.append((None, card_entry[1]))
            public_rows.append(tuple(public_row))

        player = self.players[player_index]
        legal_actions = tuple(
            int(action)
            for action, allowed in enumerate(self.action_mask(player_index))
            if allowed
        )
        own_blast_cards = (
            tuple(tuple(card) for card in player.blast_cards)
            if player.blast_pending else ()
        )

        return PlayerActionContext(
            player_index=player_index,
            legal_actions=legal_actions,
            public_rows=tuple(public_rows),
            private_cards=tuple(tuple(card) for card in player.collected_cards),
            blast_pending_by_player=tuple(p.blast_pending for p in self.players),
            own_blast_cards=own_blast_cards,
            used_blast_backsides=tuple(self.used_blast_backsides),
        )

    def _observation(self) -> np.ndarray:
        player = self.players[0] if self.players else Player()
        values = [
            self.round, player.position, player.gems, player.dynamite,
            int(player.escaped), int(player.dead), self.n_players,
            sum(not p.escaped and not p.dead for p in self.players),
            int(self.last_action == PIT_CAGE), int(self.last_action == LORRY),
            len(self.rows),
            player.column if player.column is not None else MAX_COLUMNS,
            self.consecutive_passes, self.planning_turns,
            player.cage_index if player.cage_index is not None else MAX_ROWS,
        ]
        cells = MAX_ROWS * MAX_COLUMNS
        backside_values = [BACKSIDE_UNKNOWN] * cells
        own_front_values = [BACKSIDE_UNKNOWN] * cells
        for row_index, row in enumerate(self.rows[:MAX_ROWS]):
            for col_index, card_entry in enumerate(row[:MAX_COLUMNS]):
                if card_entry is not None:
                    backside_values[row_index * MAX_COLUMNS + col_index] = card_entry[1]
                    # Home column of the learner is column 0 (col == player_index).
                    if col_index == 0:
                        own_front_values[row_index * MAX_COLUMNS + col_index] = (
                            card_entry[0] + 4
                        )
        values.extend(backside_values)
        values.extend(own_front_values)
        return np.asarray(values, dtype=np.int8)