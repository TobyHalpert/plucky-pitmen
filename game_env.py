"""Gymnasium environment for a simplified Plucky Pitmen strategy model."""

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
PARTIES_PER_MATCH = 3
SAFE_CARD_REWARD = 0.2
DRAGON_CARD_REWARD = -0.3
PIT_CAGE_REWARD = 0.0
LORRY_REWARD = -0.05


@dataclass
class Player:
    position: int = 3
    column: int | None = None
    cage_index: int | None = None
    collected_cards: list[tuple[int, int]] = field(default_factory=list)
    escaped: bool = False
    dead: bool = False
    score: int = 0

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
    """Single-agent Gymnasium environment for learning mine-exit decisions.

    This is an intentionally compact abstraction of the supplied rules:
    rows are represented by depths 0..2, and each round the agent chooses a
    destination while opponents use a fixed policy. Card identities remain
    hidden; only the agent's own collected cards are observed. The visible
    backside pattern of every card in the mine is also included in the
    observation; card fronts remain hidden until resolved.
    """

    metadata = {"render_modes": []}

    def __init__(self, opponents: int = 2, opponent_policy: str = "random") -> None:
        if opponents not in (2, 3, 4):
            raise ValueError("opponents must be 2, 3, or 4")
        if opponent_policy not in {"simple", "random", "cautious", "greedy"}:
            raise ValueError("unknown opponent policy")
        self.n_players = opponents + 1
        self.opponent_policy = opponent_policy
        self.action_space = spaces.Discrete(PASS + 1)
        max_rows = 5
        max_visible_cards = max_rows * 5
        self.observation_space = spaces.Box(
            0, 6, shape=(15 + max_visible_cards * 2,), dtype=np.int8
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
        self.partie = 1

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
        self.partie = 1
        self.planning_player = self.starting_player
        self.planned_players = [False] * self.n_players
        return self._observation(), {"round": self.round, "partie": self.partie}

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")
        if not self._legal(action):
            return self._observation(), -0.15, False, False, {"illegal_action": True}

        self.last_action = action
        self.planning_turns += 1
        self.planned_players[0] = True
        planning_reward = self._planning_reward(action)
        if action == PASS:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0
            if action < PIT_CAGE:
                row, column = divmod(action, MAX_COLUMNS)
                self._displace(row, column, 0)
                self.players[0].position = row
                self.players[0].column = column
                self.players[0].cage_index = None
            else:
                previous_position = self.players[0].position
                self.players[0].position = action
                self.players[0].column = None
                if action == PIT_CAGE:
                    if self.players[0].cage_index is None:
                        self.players[0].cage_index = 0 if previous_position == LORRY else min(previous_position, len(self.rows) - 1)
                    else:
                        self.players[0].cage_index = max(0, self.players[0].cage_index - 1)
                else:
                    self.players[0].cage_index = None

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
                if opponent_action < PIT_CAGE:
                    row, column = divmod(opponent_action, MAX_COLUMNS)
                    self._displace(row, column, player_index)
                    player.position, player.column = row, column
                    player.cage_index = None
                else:
                    previous_position = player.position
                    player.position = opponent_action
                    player.column = None
                    if opponent_action == PIT_CAGE:
                        if player.cage_index is None:
                            player.cage_index = 0 if previous_position == LORRY else min(previous_position, len(self.rows) - 1)
                        else:
                            player.cage_index = max(0, player.cage_index - 1)
                    else:
                        player.cage_index = None
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
        self._advance_starting_player()
        partie_over = self._game_over() or not self._can_deal_row()
        if not self._can_deal_row():
            info["mine_empty"] = True
        terminated = False
        if partie_over:
            self._prepare_partie_scoring(info)
            self._score_partie(info)
            info["partie"] = self.partie
            info["match"] = f"{self.partie}/{PARTIES_PER_MATCH}"
            if self.partie == PARTIES_PER_MATCH:
                terminated = True
                info["match_over"] = True
                reward = self._match_reward()
            else:
                self._start_next_partie()
                info["next_partie"] = self.partie
        else:
            self.round += 1
            if not any(player.escaped for player in self.players):
                self.rows.append(self._deal_next_rows(1)[0])
            self._set_players_to_starting_positions()
        return self._observation(), reward, terminated, False, info

    def step_one_player(self, action: int | None = None):
        """Advance exactly one player's planning decision.

        This is used by the viewer for inspecting individual player turns;
        ``step`` retains the learner-plus-opponents Gymnasium behavior.
        """
        active_players = [
            index for index, player in enumerate(self.players)
            if not player.escaped and not player.dead
        ]
        if not active_players:
            return self._observation(), 0.0, True, False, {"match_over": True}
        if self.planning_player not in active_players:
            self.planning_player = active_players[0]

        player_index = self.planning_player
        if player_index == 0:
            if action is None:
                action = PASS
            action = int(action)
            if not self.action_space.contains(action) or not self._legal(action):
                return self._observation(), 0.0, False, False, {"illegal_action": True}
        else:
            action = self._opponent_action(player_index)

        self.last_action = action
        self.planning_turns += 1
        self.planned_players[player_index] = True
        planning_reward = self._planning_reward(action, player_index)
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
        self._advance_starting_player()
        self.planning_player = self.starting_player
        partie_over = self._game_over() or not self._can_deal_row()
        if not self._can_deal_row():
            info["mine_empty"] = True
        terminated = False
        if partie_over:
            self._prepare_partie_scoring(info)
            self._score_partie(info)
            info["partie"] = self.partie
            info["match"] = f"{self.partie}/{PARTIES_PER_MATCH}"
            if self.partie == PARTIES_PER_MATCH:
                terminated = True
                info["match_over"] = True
                reward = self._match_reward()
            else:
                self._start_next_partie()
                info["next_partie"] = self.partie
        else:
            self.round += 1
            if not any(player.escaped for player in self.players):
                self.rows.append(self._deal_next_rows(1)[0])
            self._set_players_to_starting_positions()

            active_next_round = [i for i, p in enumerate(self.players) if not p.escaped and not p.dead]
            if active_next_round:
                self.planning_player = min(active_next_round, key=lambda x: (x - self.starting_player) % self.n_players)
            else:
                self.planning_player = self.starting_player

        info["player"] = player_index
        return self._observation(), reward, terminated, False, info

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
                player.cage_index = 0 if previous_position == LORRY else min(previous_position, len(self.rows) - 1)
            else:
                player.cage_index = max(0, player.cage_index - 1)
        else:
            player.cage_index = None

    def _advance_starting_player(self) -> None:
        self.starting_player = (self.starting_player + 1) % self.n_players

    def _planning_reward(self, action: int, player_index: int = 0) -> float:
        if player_index != 0:
            return 0.0
        if action == PIT_CAGE:
            return PIT_CAGE_REWARD
        if action == LORRY:
            return LORRY_REWARD
        if action == PASS:
            return self.last_reward
        if action >= PIT_CAGE:
            return 0.0
        row, column = divmod(action, MAX_COLUMNS)
        card_entry = self.rows[row][column]
        if card_entry is None:
            return 0.0
        return DRAGON_CARD_REWARD if card_entry[0] == DRAGON else SAFE_CARD_REWARD

    def _match_reward(self) -> float:
        highest_score = max(player.score for player in self.players)
        return 1.0 if self.players[0].score == highest_score else 0.0

    def _set_players_to_starting_positions(self) -> None:
        starting_position = len(self.rows)
        for player in self.players:
            if not player.escaped and not player.dead:
                player.position = starting_position
                player.column = None

    def _prepare_partie_scoring(self, info: dict[str, Any]) -> None:
        if info.get("outcome") == "dragon":
            return
        active = [player for player in self.players if not player.escaped and not player.dead]
        if info.get("mine_empty") or len(active) <= 1:
            for player in active:
                player.escaped = True

    def _score_partie(self, info: dict[str, Any]) -> None:
        escaped = [player for player in self.players if player.escaped]
        if escaped:
            highest_gems = max(player.gems for player in escaped)
            gem_winners = [player for player in escaped if player.gems == highest_gems]
            if len(escaped) == 1 or len(gem_winners) == 1:
                gem_winners[0].score += 3 if self.partie < PARTIES_PER_MATCH else 4
            else:
                for player in gem_winners:
                    player.score += 1

        dragon_player = info.get("dragon_player")
        no_one_escaped = not escaped
        for player_index, player in enumerate(self.players):
            if dragon_player == player_index:
                player.score -= 2
            elif player.dead:
                player.score += 2 if no_one_escaped and info.get("outcome") == "dragon" else -1
            player.score = max(-1, player.score)

    def _start_next_partie(self) -> None:
        retained_dynamite_cards = [
            [card for card in player.collected_cards if card[0] == DYNAMITE]
            if player.escaped else []
            for player in self.players
        ]
        scores = [player.score for player in self.players]
        self.partie += 1
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

    def _can_deal_row(self) -> bool:
        return len(self.cards_remaining) >= self.n_players

    def _pass_threshold(self) -> int:
        return max(1, sum(not player.escaped and not player.dead for player in self.players) - 1)

    def action_mask(self) -> np.ndarray:
        return np.array([self._legal(action) for action in range(self.action_space.n)], dtype=np.int8)

    def _legal(self, action: int) -> bool:
        player = self.players[0]
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
                    not self._card_occupied(row, column, ignore_player=0)
                    or self._can_displace(row, column, 0)
                )
            )
        if action == PIT_CAGE:
            return player.position != PIT_CAGE and (player.position == LORRY or self._movement_depth(player) > 0)
        if action == LORRY:
            return player.position != LORRY and player.position >= 0
        if action == PASS:
            return self.planned_players[0]
        return False

    def _movement_depth(self, player: Player) -> int:
        if player.position == PIT_CAGE:
            return player.cage_index or 0
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

    def _deal_rows(self, depth: int) -> list[list[tuple[int, int]]]:
        deck = [
            (card, backside)
            for backside in (BACKSIDE_A, BACKSIDE_B, BACKSIDE_C)
            for card in ([DRAGON] * 2 + [GEM] * 4 + [DYNAMITE] * 4)
        ]
        self.rng.shuffle(deck)
        return [deck[row * self.n_players : (row + 1) * self.n_players] for row in range(depth)]

    def _opponent_action(self, player_index: int) -> int:
        player = self.players[player_index]
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
        if self.opponent_policy == "simple":
            from rule_policy import deterministic_policy
            return deterministic_policy(self)
        if self.opponent_policy == "cautious" and self.round > 0:
            if movement_depth > 0 and self.rng.random() < 0.35:
                return PIT_CAGE
        if self.opponent_policy == "greedy":
            nearest_cards = [action for action in available_cards if action // MAX_COLUMNS == movement_depth - 1]
            if nearest_cards:
                return int(self.rng.choice(nearest_cards))
        if available_cards:
            return int(self.rng.choice(available_cards))
        return PIT_CAGE if player.position > 0 else LORRY

    def _execute_round(self) -> tuple[float, dict[str, Any]]:
        reward = 0.0
        info: dict[str, Any] = {"round": self.round, "action": self.last_action}
        learner_was_escaped = self.players[0].escaped
        player_order = [
            (self.starting_player + offset) % self.n_players
            for offset in range(self.n_players)
        ]
        for player_index in player_order:
            player = self.players[player_index]
            if player.escaped or player.dead:
                continue
            if player.position == PIT_CAGE:
                player.escaped = True
                continue
            if player.position == LORRY or player.position >= len(self.rows) or player.column is None:
                continue
            card_entry = self.rows[player.position][player.column]
            if card_entry is None:
                continue
            card, _backside = card_entry
            self.rows[player.position][player.column] = None
            player.collected_cards.append((card, _backside))
            if card == DRAGON:
                for remaining_player in self.players:
                    if not remaining_player.escaped:
                        remaining_player.dead = True
                if player_index == 0:
                    reward -= 2.0
                    info["outcome"] = "dragon"
                else:
                    reward += 1.0
                    info["outcome"] = "dragon"
                info["dragon_player"] = player_index
                break
            elif card == GEM:
                if player_index == 0:
                    reward += 0.25
            else:
                if player_index == 0:
                    reward += 0.1
        if self.players[0].escaped and not learner_was_escaped:
            reward += 0.5 + self.players[0].gems
            info["outcome"] = "escaped"
        elif not info.get("outcome"):
            info["outcome"] = "continued"
        return reward, info

    def _game_over(self) -> bool:
        active = [player for player in self.players if not player.escaped and not player.dead]
        learner = self.players[0]
        return (
            learner.dead
            or len(active) <= 1
            or any(player.dead for player in self.players)
        )

    def _observation(self) -> np.ndarray:
        player = self.players[0] if self.players else Player()
        values = [
            self.round, player.position, player.gems, player.dynamite,
            int(player.escaped), int(player.dead), self.n_players,
            sum(not p.escaped and not p.dead for p in self.players),
            int(self.last_action == PIT_CAGE), int(self.last_action == LORRY), len(self.rows),
            player.column if player.column is not None else MAX_COLUMNS,
            self.consecutive_passes, self.planning_turns,
            player.cage_index if player.cage_index is not None else MAX_ROWS,
        ]
        backside_values = [BACKSIDE_UNKNOWN] * 25
        own_front_values = [BACKSIDE_UNKNOWN] * 25
        for row_index, row in enumerate(self.rows[:5]):
            for player_index, card_entry in enumerate(row[:5]):
                if card_entry is not None:
                    backside_values[row_index * 5 + player_index] = card_entry[1]
                    if player_index == 0:
                        own_front_values[row_index * 5 + player_index] = card_entry[0] + 4
        values.extend(backside_values)
        values.extend(own_front_values)
        return np.asarray(values, dtype=np.int8)
