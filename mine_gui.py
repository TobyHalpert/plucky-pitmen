"""Tkinter viewer for the MineEnv simulation."""

from __future__ import annotations

import argparse
import pickle
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np

from game_env import (
    BACKSIDE_A, BACKSIDE_B, BACKSIDE_C, DRAGON, DYNAMITE, GEM, LORRY,
    HAULS_PER_GAME, MineEnv, PASS, PIT_CAGE,
)
from rule_policy import deterministic_policy
from train_q_learning import select_action, train


SPECIAL_ACTION_LABELS = {
    PIT_CAGE: "Leave via pit cage",
    LORRY: "Lorry",
    PASS: "Pass",
}
PLAYER_COLORS = ("#f5c451", "#e05a5a", "#4d8fe8", "#55b979", "#f5f5f5")
BACKSIDE_SYMBOLS = {BACKSIDE_A: "o", BACKSIDE_B: "v", BACKSIDE_C: "~"}
FRONT_LABELS = {GEM: "GEM", DRAGON: "DRAGON", DYNAMITE: "DYNAMITE"}
STRATEGIES = ("simple", "training", "random", "cautious", "greedy")
TRAINING_EPISODES = 1000
TRAINING_CACHE_VERSION = 2
TRAINING_CACHE_DIR = Path(__file__).with_name(".training_cache")
BLAST_ACTION_PREFIX = "Blast: "


class MineViewer:
    """Interactive visual monitor for one simulated game."""

    def __init__(self, root: tk.Tk, opponents: int, policy: str, seed: int) -> None:
        self.root = root
        self.seed = seed
        self.policy = policy
        self.training_q_table = None
        self.training_rng = np.random.default_rng(seed)
        self.env = MineEnv(opponents=opponents, opponent_policy=self._environment_policy())
        self.autoplay = False
        self.autoplay_job: str | None = None

        root.title("Plucky Pitmen - Mine Viewer")
        root.minsize(860, 620)
        root.configure(bg="#101820")

        self.haul_var = tk.StringVar()
        self.stack_var = tk.StringVar()
        self.message_var = tk.StringVar()
        self.action_var = tk.IntVar(value=0)
        self.blast_requested = False
        self._shown_displacement_count = 0
        self.policy_var = tk.StringVar(value=policy)
        self.opponents_var = tk.IntVar(value=opponents)
        self.seed_var = tk.IntVar(value=seed)

        self._build_controls()
        self._build_scene()
        self.reset_game()

    def _build_controls(self) -> None:
        controls = tk.Frame(self.root, bg="#182832", padx=14, pady=12)
        controls.pack(fill="x")

        tk.Label(
            controls, text="PLUCKY PITMEN", fg="#f5c451", bg="#182832",
            font=("Georgia", 18, "bold"),
        ).pack(side="left", padx=(0, 20))
        tk.Label(
            controls, textvariable=self.haul_var, fg="#d7e4e8", bg="#182832",
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left", padx=(14, 0))
        tk.Label(
            controls, textvariable=self.stack_var, fg="#b7c8cc", bg="#182832",
            font=("Segoe UI", 10),
        ).pack(side="left", padx=(14, 0))

        settings = tk.Frame(controls, bg="#182832")
        settings.pack(side="right")
        tk.Label(settings, text="Opponents", fg="#b7c8cc", bg="#182832").grid(row=0, column=0, padx=4)
        ttk.Spinbox(settings, from_=2, to=4, width=3, textvariable=self.opponents_var).grid(row=0, column=1, padx=4)
        tk.Label(settings, text="Policy", fg="#b7c8cc", bg="#182832").grid(row=0, column=2, padx=4)
        ttk.Combobox(
            settings, textvariable=self.policy_var, values=STRATEGIES,
            state="readonly", width=9,
        ).grid(row=0, column=3, padx=4)
        tk.Label(settings, text="Seed", fg="#b7c8cc", bg="#182832").grid(row=0, column=4, padx=4)
        ttk.Entry(settings, textvariable=self.seed_var, width=7).grid(row=0, column=5, padx=4)
        ttk.Button(settings, text="New game", command=self.reset_game).grid(row=0, column=6, padx=(8, 0))

    def _build_scene(self) -> None:
        body = tk.Frame(self.root, bg="#101820", padx=18, pady=14)
        body.pack(fill="both", expand=True)

        self.mine_canvas = tk.Canvas(body, bg="#1b2d35", highlightthickness=0)
        self.mine_canvas.pack(side="left", fill="both", expand=True)
        self.mine_canvas.bind("<Configure>", lambda _event: self.draw_scene())

        side = tk.Frame(body, bg="#14242c", width=235, padx=14, pady=14)
        side.pack(side="right", fill="y", padx=(14, 0))
        side.pack_propagate(False)
        tk.Label(side, text="PLAYERS", fg="#f5c451", bg="#14242c", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.players_text = tk.Text(
            side, height=14, width=25, bg="#0e1a20", fg="#d7e4e8", relief="flat",
            font=("Consolas", 10), padx=8, pady=8, state="disabled",
        )
        self.players_text.configure(tabs=("1.55i",))
        self.players_text.pack(fill="x", pady=(8, 18))
        tk.Label(side, text="YOUR ACTION", fg="#f5c451", bg="#14242c", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.action_menu = ttk.Combobox(
            side, state="readonly", width=20,
        )
        self.action_menu.pack(fill="x", pady=(8, 8))
        self.action_menu.bind("<<ComboboxSelected>>", self._action_label_selected)
        self.step_button = ttk.Button(side, text="Take action", command=self.take_action)
        self.step_button.pack(fill="x")
        self.autoplay_button = ttk.Button(side, text="Autoplay", command=self.toggle_autoplay)
        self.autoplay_button.pack(fill="x", pady=(6, 0))
        self.autoplay_step_button = ttk.Button(side, text="Autoplay step", command=self.autoplay_once)
        self.autoplay_step_button.pack(fill="x", pady=(6, 0))
        message_frame = tk.Frame(side, bg="#14242c")
        message_frame.pack(fill="both", expand=True, pady=(18, 0))
        self.message_text = tk.Text(
            message_frame, bg="#0e1a20", fg="#b7c8cc", relief="flat",
            font=("Segoe UI", 9), padx=8, pady=8, wrap="word", state="disabled",
        )
        self.message_text.pack(side="left", fill="both", expand=True)
        message_scrollbar = ttk.Scrollbar(
            message_frame, orient="vertical", command=self.message_text.yview,
        )
        message_scrollbar.pack(side="right", fill="y")
        self.message_text.configure(yscrollcommand=message_scrollbar.set)
        self.message_var.trace_add("write", self._append_message)

    def _append_message(self, *_args: object) -> None:
        self.message_text.configure(state="normal")
        self.message_text.insert("end", self.message_var.get() + "\n")
        self.message_text.configure(state="disabled")
        self.message_text.see("end")

    def _action_label_selected(self, _event: tk.Event) -> None:
        selected = self.action_menu.get()
        player = self.env.players[0]
        if self.env.tie_breaker_contenders:
            labels = list(self.action_menu["values"])
            if selected in labels:
                self.action_var.set(labels.index(selected))
            return
        if player.blast_pending and player.blast_cards:
            labels = list(self.action_menu["values"])
            if selected in labels:
                self.action_var.set(labels.index(selected))
            return

        self.blast_requested = selected.startswith(BLAST_ACTION_PREFIX)
        if self.blast_requested:
            selected = selected[len(BLAST_ACTION_PREFIX):]
        for action in range(self.env.action_space.n):
            if self._action_label(action) == selected:
                self.action_var.set(action)
                return

    def reset_game(self) -> None:
        self.stop_autoplay()
        self._shown_displacement_count = 0
        self.message_text.configure(state="normal")
        self.message_text.delete("1.0", "end")
        self.message_text.configure(state="disabled")
        try:
            self.policy = self.policy_var.get()
            self.training_q_table = None
            self.env = MineEnv(opponents=self.opponents_var.get(), opponent_policy=self._environment_policy())
            self.env.reset(seed=self.seed_var.get())
            self._advance_opponents_to_p1()
            if self.policy == "training":
                cache_path = self._training_cache_path()
                if cache_path.exists():
                    with cache_path.open("rb") as cache_file:
                        self.training_q_table = pickle.load(cache_file)
                else:
                    self.message_var.set("Training P1 policy...")
                    self.root.update_idletasks()
                    self.training_q_table = train(
                        TRAINING_EPISODES, self.opponents_var.get(), "random", self.seed_var.get()
                    )
                    TRAINING_CACHE_DIR.mkdir(exist_ok=True)
                    with cache_path.open("wb") as cache_file:
                        pickle.dump(self.training_q_table, cache_file)
        except (tk.TclError, ValueError) as error:
            self.message_var.set(f"Could not start game: {error}")
            return
        self._update_action_options()
        self.message_var.set("Choose a destination. Opponents move automatically after your action.")
        self.step_button.state(["!disabled"])
        self.draw_scene()

    def _environment_policy(self) -> str:
        return "random" if self.policy == "training" else self.policy

    def setup_tie_break_test(self) -> None:
        self.env.haul = 3
        self.env.starting_player = 0
        self.env.players[0].score = 5
        self.env.players[1].score = 5
        for player in self.env.players[2:]:
            player.score = 0
        self.env.rows = [[None] * self.env.n_players for _ in range(3)]
        self.env.rows[0][0] = (GEM, BACKSIDE_A)
        self.env.rows[0][1] = (DRAGON, BACKSIDE_B)
        self.env.cards_remaining = [(GEM, BACKSIDE_C)]
        self.env._begin_tie_breaker()
        self._update_action_options()
        self.message_var.set("Tie-break test: Player 1 draws first. Choose a mine card or the deck.")
        self.draw_scene()

    def _advance_opponents_to_p1(self) -> None:
        while self.env.planning_player != 0:
            _observation, _reward, terminated, _truncated, _info = self.env.step_one_player()
            if terminated or self.env.players[0].escaped or self.env.players[0].dead:
                return

    def _training_cache_path(self) -> Path:
        return TRAINING_CACHE_DIR / (
            f"q_table_opponents_{self.opponents_var.get()}_seed_{self.seed_var.get()}_"
            f"episodes_{TRAINING_EPISODES}_v{TRAINING_CACHE_VERSION}.pkl"
        )

    def take_action(self) -> None:
        if self.env.tie_breaker_contenders:
            self._resolve_tie_breaker(int(self.action_var.get()))
            return

        player = self.env.players[0]
        if player.blast_pending and player.blast_cards:
            choice_index = int(self.action_var.get())
            if not 0 <= choice_index < len(player.blast_cards):
                self.message_var.set("Select one of the three blast cards.")
                return
            final_haul = self.env.haul == HAULS_PER_GAME
            chosen = self.env.resolve_blast(0, choice_index)
            if chosen[0] == DRAGON:
                if final_haul:
                    self.message_var.set("Blast choice: DRAGON. Player 1 awakened the dragon. Game over.")
                    self.stop_autoplay()
                    self.step_button.state(["disabled"])
                else:
                    self.message_var.set("Blast choice: DRAGON. Player 1 awakened the dragon. Starting the next haul.")
            else:
                self.message_var.set(f"Blast choice: {FRONT_LABELS[chosen[0]]}. Reward +0.00.")
            self.draw_scene()
            self._update_action_options()
            return

        blast_requested = self.blast_requested
        self.blast_requested = False

        while self.env.planning_player != 0 and not (player.escaped or player.dead):
            _observation, reward, terminated, _truncated, info = self.env.step_one_player()
            if terminated:
                self.draw_scene()
                self._update_action_options()
                self.step_button.state(["disabled"])
                self.message_var.set(
                    self._format_outcome(info, reward) + " Game over. Start a new game to play again."
                )
                return

        if player.escaped or player.dead:
            self.draw_scene()
            self._update_action_options()
            return

        action = self.action_var.get()

        if action >= self.env.action_space.n or not self.env.action_mask()[action]:
            self.message_var.set("That destination is not legal from your current position.")
            return
        if blast_requested:
            try:
                _observation, reward, terminated, _truncated, info = self.env.step_one_player(
                    action, announce_blast=True
                )
            except ValueError as error:
                self.message_var.set(str(error))
                return
        else:
            _observation, reward, terminated, _truncated, info = self.env.step_one_player(action)
        while not terminated and self.env.planning_player != 0:
            _observation, reward, terminated, _truncated, info = self.env.step_one_player()
        self.message_var.set(self._format_outcome(info, reward))
        self.draw_scene()
        self._update_action_options()
        if terminated:
            self.stop_autoplay()
            self.step_button.state(["disabled"])
            self.message_var.set(self.message_var.get() + " Game over. Start a new game to play again.")

    def toggle_autoplay(self) -> None:
        if self.autoplay:
            self.stop_autoplay()
        else:
            self.autoplay = True
            self.autoplay_button.configure(text="Pause")
            self._autoplay_step()

    def stop_autoplay(self) -> None:
        self.autoplay = False
        self.autoplay_button.configure(text="Autoplay")
        if self.autoplay_job is not None:
            self.root.after_cancel(self.autoplay_job)
            self.autoplay_job = None

    def _format_outcome(self, info: dict, reward: float) -> str:
        outcome = info.get("outcome", "continued")
        message = f"Round {info.get('round', self.env.round)}: {outcome}. Reward {reward:+.2f}."
        dragon_player = info.get("dragon_player")
        if outcome == "dragon" and dragon_player is not None:
            message += f" Player {dragon_player + 1} awakened the dragon."
        new_displacements = self.env.displacement_events[self._shown_displacement_count:]
        for displaced_player, displacing_player in new_displacements:
            message += (
                f" Player {displaced_player + 1} stood on his own card, "
                f"Player {displacing_player + 1} displaced him."
            )
        self._shown_displacement_count += len(new_displacements)
        return message

    def _resolve_tie_breaker(self, choice_index: int) -> None:
        options = self._tie_break_options()
        if not 0 <= choice_index < len(options):
            self.message_var.set("Select a tie-break card source.")
            return
        source, row, column = options[choice_index]
        result = self.env.resolve_tie_breaker(source, row, column)
        player_number = result["player"] + 1
        if result.get("game_over"):
            self.message_var.set(
                f"Player {player_number} drew a dragon and was eliminated from the tie-break. "
                f"Player {result['winner'] + 1} wins the tie-break. Game over."
            )
            self.stop_autoplay()
            self.step_button.state(["disabled"])
        elif result["outcome"] == "dragon":
            self.message_var.set(f"Player {player_number} drew a dragon and was eliminated from the tie-break.")
        else:
            self.message_var.set(f"Player {player_number} drew a safe tie-break card.")

        while self.env.tie_breaker_contenders:
            current_player = self.env.tie_breaker_contenders[self.env.tie_breaker_turn]
            if current_player == 0:
                break
            opponent_options = self._tie_break_options()
            if len(opponent_options) == 1:
                opponent_source, opponent_row, opponent_column = opponent_options[0]
            else:
                opponent_source, opponent_row, opponent_column = opponent_options[
                    int(self.env.rng.integers(len(opponent_options)))
                ]
            opponent_result = self.env.resolve_tie_breaker(
                opponent_source, opponent_row, opponent_column,
            )
            opponent_number = opponent_result["player"] + 1
            if opponent_result.get("game_over"):
                self.message_var.set(
                    f"Player {opponent_number} drew a dragon and was eliminated from the tie-break. "
                    f"Player {opponent_result['winner'] + 1} wins the tie-break. Game over."
                )
                self.stop_autoplay()
                self.step_button.state(["disabled"])
                break
            if opponent_result["outcome"] == "dragon":
                self.message_var.set(
                    f"Player {opponent_number} drew a dragon and was eliminated from the tie-break."
                )
            else:
                self.message_var.set(f"Player {opponent_number} drew a safe tie-break card.")
        self.draw_scene()
        self._update_action_options()

    def _tie_break_options(self) -> list[tuple[str, int | None, int | None]]:
        options = [("deck", None, None)] if self.env.cards_remaining else []
        options.extend(
            ("mine", row_index, column_index)
            for row_index, row in enumerate(self.env.rows)
            for column_index, card in enumerate(row)
            if card is not None
        )
        return options

    def _autoplay_step(self) -> None:
        if not self.autoplay:
            return
        self._run_single_player_step()
        if self.autoplay:
            self.autoplay_job = self.root.after(750, self._autoplay_step)

    def autoplay_once(self) -> None:
        if self.autoplay:
            self.stop_autoplay()
        self._run_single_player_step()

    def _run_single_player_step(self) -> None:
        if self.env.tie_breaker_contenders:
            self._resolve_tie_breaker(0)
            return

        player = self.env.players[0]
        if player.blast_pending and player.blast_cards:
            final_haul = self.env.haul == HAULS_PER_GAME
            chosen = self.env.resolve_blast(0, 0)
            if chosen[0] == DRAGON:
                if final_haul:
                    self.message_var.set("Blast choice: DRAGON. Player 1 awakened the dragon. Game over.")
                    self.stop_autoplay()
                    self.step_button.state(["disabled"])
                else:
                    self.message_var.set("Blast choice: DRAGON. Player 1 awakened the dragon. Starting the next haul.")
            else:
                self.message_var.set(f"Blast choice: {FRONT_LABELS[chosen[0]]}. Reward +0.00.")
            self.draw_scene()
            self._update_action_options()
            return

        if self.env.planning_player == 0:
            legal = [int(action) for action, allowed in enumerate(self.env.action_mask()) if allowed]
            if not legal and not (self.env.players[0].escaped or self.env.players[0].dead):
                return
            if not legal:
                _observation, reward, terminated, _truncated, info = self.env.step_one_player()
            elif self.policy == "training" and self.training_q_table is not None:
                action = select_action(self.env, self.training_q_table, 0.0, self.training_rng)
                self.action_var.set(action)
                self.action_menu.set(self._action_label(action))
                _observation, reward, terminated, _truncated, info = self.env.step_one_player(action)
            elif self.policy == "simple":
                action = deterministic_policy(self.env)
                self.action_var.set(action)
                self.action_menu.set(self._action_label(action))
                _observation, reward, terminated, _truncated, info = self.env.step_one_player(action)
            else:
                action = PASS if self.env.players[0].position in (PIT_CAGE, LORRY) and PASS in legal else legal[0]
                self.action_var.set(action)
                self.action_menu.set(self._action_label(action))
                _observation, reward, terminated, _truncated, info = self.env.step_one_player(action)
        else:
            _observation, reward, terminated, _truncated, info = self.env.step_one_player()
        self.message_var.set(self._format_outcome(info, reward))
        self.draw_scene()
        self._update_action_options()
        if terminated:
            self.stop_autoplay()
            self.step_button.state(["disabled"])
            self.message_var.set(self.message_var.get() + " Game over. Start a new game to play again.")

    def draw_scene(self) -> None:
        canvas = self.mine_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 560)
        height = max(canvas.winfo_height(), 480)
        left, top, right, bottom = 34, 112, width - 28, height - 34
        row_width = right - left

        max_columns = max(self.env.n_players, max((len(row) for row in self.env.rows), default=1))
        column_width = row_width / max_columns
        for player_index in range(max_columns):
            x1 = left + player_index * column_width + 8
            x2 = left + (player_index + 1) * column_width - 8
            canvas.create_rectangle(x1, 50, x2, 90, fill=PLAYER_COLORS[player_index], outline="#101820", width=2)
            canvas.create_text((x1 + x2) / 2, 70, text="Lorry", fill="#101820", font=("Segoe UI", 8, "bold"))

        cage_gap = 22
        row_height = max(46, min(70, (bottom - top - (len(self.env.rows) - 1) * cage_gap) // max(1, len(self.env.rows))))
        row_step = row_height + cage_gap
        cage_centers: list[float] = []
        self._draw_pit_cage_row(canvas, left, 93, right, 16)
        cage_centers.append(101)

        card_centers: dict[tuple[int, int], tuple[float, float]] = {}
        for row_index, row in enumerate(self.env.rows):
            y1 = top + row_index * row_step
            y2 = y1 + row_height
            canvas.create_rectangle(left, y1, right, y2, fill="#263f48", outline="#52717a", width=2)
            canvas.create_text(left + 10, y1 + 12, text=f"ROW {row_index}", anchor="w", fill="#9fb7bd", font=("Segoe UI", 9, "bold"))
            for player_index, card_entry in enumerate(row):
                x = left + player_index * column_width + column_width / 2
                card_left = left + player_index * column_width + 8
                card_right = left + (player_index + 1) * column_width - 8
                card_top = y1 + 27
                card_bottom = y2 - 9
                if card_entry is not None:
                    canvas.create_rectangle(card_left, card_top, card_right, card_bottom, fill="#3c6874", outline="#9fc5c7", width=2)
                    self._draw_backside_pattern(canvas, card_entry[1], card_left, card_top, card_right, card_bottom)
                    if player_index == 0:
                        canvas.create_text(
                            card_left + 4,
                            card_top + 8,
                            text=FRONT_LABELS[card_entry[0]],
                            anchor="w",
                            fill="#f5c451",
                            font=("Segoe UI", 7, "bold"),
                        )
                    card_centers[(row_index, player_index)] = (x, (card_top + card_bottom) / 2)

            if row_index < len(self.env.rows) - 1:
                self._draw_pit_cage_row(canvas, left, y2 + 3, right, cage_gap - 6)
                cage_centers.append(y2 + 3 + (cage_gap - 6) / 2)

        exit_y = top + (len(self.env.rows) - 1) * row_step + row_height + 12

        for player_index, player in enumerate(self.env.players):
            if player.escaped:
                continue
            if player.dead:
                location = "DEAD"
            elif player.escaped:
                location = "OUT"
            elif player.position == PIT_CAGE:
                location = "pit cage"
            elif player.position == LORRY:
                location = "lorry"
            else:
                location = f"ROW {player.position}"
            if player.position == PIT_CAGE and cage_centers:
                x = left + player_index * column_width + column_width / 2
                cage_index = player.cage_index if player.cage_index is not None else len(cage_centers) - 1
                y = cage_centers[min(cage_index, len(cage_centers) - 1)]
            elif not (player.dead or player.escaped) and player.column is not None and (player.position, player.column) in card_centers:
                x, y = card_centers[(player.position, player.column)]
            else:
                y = 70 if player.position == LORRY else exit_y + 36
                x = left + player_index * column_width + column_width / 2
            color = PLAYER_COLORS[player_index]
            canvas.create_oval(x - 11, y - 11, x + 11, y + 11, fill=color, outline="#101820", width=2)
            canvas.create_text(x, y, text=str(player_index + 1), fill="#101820", font=("Segoe UI", 9, "bold"))
            # Draw blast indicator while the blast is in progress
            if player.blast_pending:
                canvas.create_oval(x - 16, y - 16, x + 16, y + 16, fill="", outline="#ff6b6b", width=3)

        self.haul_var.set(f"Haul {self.env.haul}/3")
        top_backside = (
            BACKSIDE_SYMBOLS[self.env.cards_remaining[0][1]]
            if self.env.cards_remaining else "-"
        )
        self.stack_var.set(f"Stack: {len(self.env.cards_remaining)}  |  top back: {top_backside}")
        self._update_player_list()

    @staticmethod
    def _draw_pit_cage_row(canvas: tk.Canvas, left: float, y: float, right: float, height: float) -> None:
        canvas.create_rectangle(left, y, right, y + height, fill="#263f48", outline="#52717a", width=1)
        canvas.create_text(right - 8, y + height / 2, text="Pit cage", anchor="e", fill="#9fb7bd", font=("Segoe UI", 7, "bold"))

    @staticmethod
    def _draw_backside_pattern(canvas: tk.Canvas, backside: int, left: float, top: float, right: float, bottom: float) -> None:
        pattern_characters = {
            BACKSIDE_A: "o",
            BACKSIDE_B: "v",
            BACKSIDE_C: "~",
        }
        character = pattern_characters.get(backside, "?")
        canvas.create_text(
            (left + right) / 2,
            (top + bottom) / 2,
            text=(character + " ") * 5,
            fill="#b7d4d2",
            font=("Consolas", 12, "bold"),
        )

    def _update_player_list(self) -> None:
        self.players_text.configure(state="normal")
        self.players_text.delete("1.0", "end")
        for index, player in enumerate(self.env.players):
            starting = " (starting player)" if index == self.env.starting_player else ""
            blast_indicator = " 💥 BLASTING" if player.blast_pending else ""
            status = " OUT" if player.escaped else ""
            collected = ", ".join(BACKSIDE_SYMBOLS[backside] for backside in player.collected_backsides) or "-"
            if index == 0:
                gems = ", ".join(BACKSIDE_SYMBOLS[backside] for front, backside in player.collected_cards if front == GEM) or "-"
                dynamite = ", ".join(BACKSIDE_SYMBOLS[backside] for front, backside in player.collected_cards if front == DYNAMITE) or "-"
                self.players_text.insert("end", f"P{index + 1} (you){status}{starting}{blast_indicator}\t{player.score} VP\n")
                self.players_text.insert("end", f"backs: {collected}\n")
                self.players_text.insert("end", f"gems: {gems}  dynamite: {dynamite}\n\n")
            else:
                self.players_text.insert("end", f"P{index + 1}{status}{starting}{blast_indicator}\t{player.score} VP\nbacks: {collected}\n\n")
        if self.env.used_blast_backsides:
            used_cards = ", ".join(
                BACKSIDE_SYMBOLS[backside] for backside in self.env.used_blast_backsides
            )
            self.players_text.insert("end", f"used cards: {used_cards}\n")
        self.players_text.configure(state="disabled")

    def _update_action_options(self) -> None:
        if self.env.tie_breaker_contenders:
            labels = []
            for source, row, column in self._tie_break_options():
                if source == "deck":
                    labels.append("Draw from deck")
                else:
                    labels.append(f"Draw from mine (row {row}, column {column + 1})")
            self.action_menu.configure(values=labels)
            self.action_var.set(0)
            self.action_menu.set(labels[0])
            return

        player = self.env.players[0]
        if player.blast_pending and player.blast_cards:
            labels = [
                f"Blast choice {index + 1} ({BACKSIDE_SYMBOLS[card[1]]})"
                for index, card in enumerate(player.blast_cards)
            ]
            self.action_menu.configure(values=labels)
            self.action_var.set(0)
            self.action_menu.set(labels[0])
            return

        legal_actions = [
            int(action) for action, allowed in enumerate(self.env.action_mask()) if allowed
        ]
        labels = [self._action_label(action) for action in legal_actions]
        if player.dynamite >= 2:
            movement_depth = self.env._movement_depth(player)
            labels.extend(
                f"{BLAST_ACTION_PREFIX}{self._action_label(action)}"
                for action in legal_actions
                if action < PIT_CAGE and action // 5 < movement_depth
            )
        self.action_menu.configure(values=labels)
        if legal_actions:
            self.action_var.set(legal_actions[0])
            self.action_menu.set(labels[0])
        else:
            self.action_var.set(0)
            self.action_menu.set("")

    def _action_label(self, action: int) -> str:
        if action in SPECIAL_ACTION_LABELS:
            return SPECIAL_ACTION_LABELS[action]
        row, column = divmod(action, 5)
        return f"Card row {row}, column {column + 1}"


def main() -> None:
    parser = argparse.ArgumentParser(description="View MineEnv player progression.")
    parser.add_argument("--opponents", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--policy", choices=STRATEGIES, default="simple")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tie-break-test", action="store_true", help="open directly at a tie-break scenario")
    args = parser.parse_args()
    root = tk.Tk()
    viewer = MineViewer(root, args.opponents, args.policy, args.seed)
    if args.tie_break_test:
        viewer.setup_tie_break_test()
    root.mainloop()


if __name__ == "__main__":
    main()
