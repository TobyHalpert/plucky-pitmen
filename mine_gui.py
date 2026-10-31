"""Tkinter viewer for the MineEnv simulation."""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

from game_env import (
    BACKSIDE_A, BACKSIDE_B, BACKSIDE_C, DRAGON, DYNAMITE, GEM, LORRY,
    MineEnv, PASS, PIT_CAGE,
)
from features import N_BASE_ACTIONS
from gui_controller import (
    GameController,
    StepResult,
    agent_policies,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPECIAL_ACTION_LABELS = {
    PIT_CAGE: "Leave via pit cage",
    LORRY: "Lorry",
    PASS: "Pass",
}
PLAYER_COLORS = ("#f5c451", "#e05a5a", "#4d8fe8", "#55b979", "#f5f5f5")
BACKSIDE_SYMBOLS = {BACKSIDE_A: "o", BACKSIDE_B: "v", BACKSIDE_C: "~"}
FRONT_LABELS = {GEM: "GEM", DRAGON: "DRAGON", DYNAMITE: "DYNAMITE"}
POLICIES = agent_policies()
BLAST_ACTION_PREFIX = "Blast: "
# The action space is the base primitives plus one "and announce a blast" copy
# of each. Derive the split instead of hardcoding it: the grid grows with the
# deepest board a deck can produce.
ANNOUNCE_OFFSET = N_BASE_ACTIONS
AUTOPLAY_DELAY_MS = 750
RESIZE_DEBOUNCE_MS = 100


# Cross-platform font families. Tk's built-in "TkDefaultFont", "TkTextFont",
# and "TkFixedFont" resolve to native defaults on every platform, but explicit
# family names give more consistent metrics for layout math.
if sys.platform == "win32":
    UI_FAMILY = "Segoe UI"
    MONO_FAMILY = "Consolas"
elif sys.platform == "darwin":
    UI_FAMILY = "Helvetica Neue"
    MONO_FAMILY = "Menlo"
else:
    UI_FAMILY = "DejaVu Sans"
    MONO_FAMILY = "DejaVu Sans Mono"


def ui_font(size: int, weight: str = "normal") -> tuple[str, int, str]:
    return (UI_FAMILY, size, weight)


def mono_font(size: int, weight: str = "normal") -> tuple[str, int, str]:
    return (MONO_FAMILY, size, weight)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    """Static layout parameters for the mine canvas."""
    margin_left: int = 34
    margin_top: int = 112
    margin_right: int = 28
    margin_bottom: int = 34
    lorry_y1: int = 50
    lorry_y2: int = 90
    lorry_pad: int = 8
    pit_cage_top: int = 93
    pit_cage_height: int = 16
    cage_gap: int = 22
    min_row_height: int = 46
    max_row_height: int = 70
    card_pad_x: int = 8
    card_pad_top: int = 27
    card_pad_bottom: int = 9
    token_radius: int = 11
    blast_radius: int = 16
    exit_offset: int = 12
    exit_token_offset: int = 36
    min_canvas_width: int = 560
    min_canvas_height: int = 480


LAYOUT = Layout()


# ---------------------------------------------------------------------------
# Renderer — pure canvas drawing
# ---------------------------------------------------------------------------

class MineRenderer:
    """Pure Tkinter canvas renderer. Owns no game state."""

    def __init__(self, canvas: tk.Canvas) -> None:
        self.canvas = canvas

    def render(self, env: MineEnv) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), LAYOUT.min_canvas_width)
        height = max(canvas.winfo_height(), LAYOUT.min_canvas_height)
        left = LAYOUT.margin_left
        top = LAYOUT.margin_top
        right = width - LAYOUT.margin_right
        bottom = height - LAYOUT.margin_bottom
        row_width = right - left

        max_columns = max(env.n_players, max((len(r) for r in env.rows), default=1))
        column_width = row_width / max_columns

        self._draw_lorry_row(left, column_width, max_columns)
        row_height, row_step, cage_centers, card_centers = self._draw_mine_rows(
            env, left, top, bottom, column_width
        )
        exit_y = top + (len(env.rows) - 1) * row_step + row_height + LAYOUT.exit_offset
        self._draw_tokens(env, left, column_width, exit_y, cage_centers, card_centers)

    # -- pieces ------------------------------------------------------------

    def _draw_lorry_row(self, left: int, column_width: float, columns: int) -> None:
        for i in range(columns):
            x1 = left + i * column_width + LAYOUT.lorry_pad
            x2 = left + (i + 1) * column_width - LAYOUT.lorry_pad
            self.canvas.create_rectangle(
                x1, LAYOUT.lorry_y1, x2, LAYOUT.lorry_y2,
                fill=PLAYER_COLORS[i], outline="#101820", width=2,
            )
            self.canvas.create_text(
                (x1 + x2) / 2, (LAYOUT.lorry_y1 + LAYOUT.lorry_y2) / 2,
                text="Lorry", fill="#101820", font=ui_font(8, "bold"),
            )

    def _draw_mine_rows(
        self, env: MineEnv, left: int, top: int, bottom: int, column_width: float,
    ) -> tuple[int, int, list[float], dict[tuple[int, int], tuple[float, float]]]:
        n_rows = len(env.rows)
        row_height = max(
            LAYOUT.min_row_height,
            min(LAYOUT.max_row_height, (bottom - top - (n_rows - 1) * LAYOUT.cage_gap) // max(1, n_rows)),
        )
        row_step = row_height + LAYOUT.cage_gap

        right = self.canvas.winfo_width() - LAYOUT.margin_right
        cage_centers: list[float] = []
        card_centers: dict[tuple[int, int], tuple[float, float]] = {}

        self._draw_pit_cage_row(left, LAYOUT.pit_cage_top, right, LAYOUT.pit_cage_height)
        cage_centers.append(LAYOUT.pit_cage_top + LAYOUT.pit_cage_height / 2)

        for row_index, row in enumerate(env.rows):
            y1 = top + row_index * row_step
            y2 = y1 + row_height
            self.canvas.create_rectangle(left, y1, right, y2, fill="#263f48", outline="#52717a", width=2)
            self.canvas.create_text(
                left + 10, y1 + 12, text=f"ROW {row_index}",
                anchor="w", fill="#9fb7bd", font=ui_font(9, "bold"),
            )
            for col, card_entry in enumerate(row):
                x = left + col * column_width + column_width / 2
                card_left = left + col * column_width + LAYOUT.card_pad_x
                card_right = left + (col + 1) * column_width - LAYOUT.card_pad_x
                card_top = y1 + LAYOUT.card_pad_top
                card_bottom = y2 - LAYOUT.card_pad_bottom
                if card_entry is not None:
                    self.canvas.create_rectangle(
                        card_left, card_top, card_right, card_bottom,
                        fill="#3c6874", outline="#9fc5c7", width=2,
                    )
                    self._draw_backside_pattern(
                        card_entry[1], card_left, card_top, card_right, card_bottom
                    )
                    if col == 0:
                        self.canvas.create_text(
                            card_left + 4, card_top + 8, text=FRONT_LABELS[card_entry[0]],
                            anchor="w", fill="#f5c451", font=ui_font(7, "bold"),
                        )
                    card_centers[(row_index, col)] = (x, (card_top + card_bottom) / 2)

            if row_index < n_rows - 1:
                gap_top = y2 + 3
                gap_height = LAYOUT.cage_gap - 6
                self._draw_pit_cage_row(left, gap_top, right, gap_height)
                cage_centers.append(gap_top + gap_height / 2)

        return row_height, row_step, cage_centers, card_centers

    def _draw_tokens(
        self,
        env: MineEnv,
        left: int,
        column_width: float,
        exit_y: float,
        cage_centers: list[float],
        card_centers: dict[tuple[int, int], tuple[float, float]],
    ) -> None:
        for i, player in enumerate(env.players):
            if player.escaped:
                continue
            if player.position == PIT_CAGE and cage_centers:
                x = left + i * column_width + column_width / 2
                cage_index = player.cage_index if player.cage_index is not None else len(cage_centers) - 1
                y = cage_centers[min(cage_index, len(cage_centers) - 1)]
            elif not (player.dead or player.escaped) and player.column is not None \
                    and (player.position, player.column) in card_centers:
                x, y = card_centers[(player.position, player.column)]
            else:
                y = (LAYOUT.lorry_y1 + LAYOUT.lorry_y2) / 2 if player.position == LORRY \
                    else exit_y + LAYOUT.exit_token_offset
                x = left + i * column_width + column_width / 2

            color = PLAYER_COLORS[i]
            r = LAYOUT.token_radius
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="#101820", width=2)
            self.canvas.create_text(x, y, text=str(i + 1), fill="#101820", font=ui_font(9, "bold"))
            if player.blast_pending:
                br = LAYOUT.blast_radius
                self.canvas.create_oval(x - br, y - br, x + br, y + br, outline="#ff6b6b", width=3)

    def _draw_pit_cage_row(self, left: float, y: float, right: float, height: float) -> None:
        self.canvas.create_rectangle(left, y, right, y + height, fill="#263f48", outline="#52717a", width=1)
        self.canvas.create_text(
            right - 8, y + height / 2, text="Pit cage",
            anchor="e", fill="#9fb7bd", font=ui_font(7, "bold"),
        )

    def _draw_backside_pattern(
        self, backside: int, left: float, top: float, right: float, bottom: float,
    ) -> None:
        character = BACKSIDE_SYMBOLS.get(backside, "?")
        self.canvas.create_text(
            (left + right) / 2, (top + bottom) / 2,
            text=(character + " ") * 5, fill="#b7d4d2", font=mono_font(12, "bold"),
        )


# ---------------------------------------------------------------------------
# Viewer — thin Tkinter shell wiring controller and renderer
# ---------------------------------------------------------------------------

class MineViewer:
    """Interactive visual monitor. Delegates game logic to GameController
    and canvas drawing to MineRenderer."""

    def __init__(
        self,
        root: tk.Tk,
        opponents: int,
        policy: str,
        seed: int,
        all_seats: bool = False,
        deterministic: bool = False,
    ) -> None:
        self.root = root
        self.controller = GameController(
            opponents,
            policy,
            seed,
            all_seats=all_seats,
            deterministic=deterministic,
        )
        self.autoplay = False
        self.autoplay_job: str | None = None
        self._resize_job: str | None = None

        root.title("Plucky Pitmen - Mine Viewer")
        root.minsize(860, 620)
        root.configure(bg="#101820")

        self.haul_var = tk.StringVar()
        self.stack_var = tk.StringVar()
        self.message_var = tk.StringVar()
        self.action_var = tk.IntVar(value=0)
        self.blast_requested = False
        self.policy_var = tk.StringVar(value=policy)
        self.opponents_var = tk.IntVar(value=opponents)
        self.seed_var = tk.IntVar(value=seed)
        self.all_seats_var = tk.BooleanVar(value=all_seats)
        self.deterministic_var = tk.BooleanVar(value=deterministic)

        self._build_controls()
        self._build_scene()
        self.renderer = MineRenderer(self.mine_canvas)
        self.reset_game()

    # -- widget construction ----------------------------------------------

    def _build_controls(self) -> None:
        controls = tk.Frame(self.root, bg="#182832", padx=14, pady=12)
        controls.pack(fill="x")

        tk.Label(controls, text="PLUCKY PITMEN", fg="#f5c451", bg="#182832",
                 font=ui_font(18, "bold")).pack(side="left", padx=(0, 20))
        tk.Label(controls, textvariable=self.haul_var, fg="#d7e4e8", bg="#182832",
                 font=ui_font(11, "bold")).pack(side="left", padx=(14, 0))
        tk.Label(controls, textvariable=self.stack_var, fg="#b7c8cc", bg="#182832",
                 font=ui_font(10)).pack(side="left", padx=(14, 0))

        settings = tk.Frame(controls, bg="#182832")
        settings.pack(side="right")
        tk.Label(settings, text="Opponents", fg="#b7c8cc", bg="#182832",
                 font=ui_font(10)).grid(row=0, column=0, padx=4)
        ttk.Spinbox(settings, from_=2, to=4, width=3,
                    textvariable=self.opponents_var).grid(row=0, column=1, padx=4)
        tk.Label(settings, text="Policy", fg="#b7c8cc", bg="#182832",
                 font=ui_font(10)).grid(row=0, column=2, padx=4)
        ttk.Combobox(settings, textvariable=self.policy_var, values=POLICIES,
                     state="readonly", width=18).grid(row=0, column=3, padx=4)
        tk.Label(settings, text="Seed", fg="#b7c8cc", bg="#182832",
                 font=ui_font(10)).grid(row=0, column=4, padx=4)
        ttk.Entry(settings, textvariable=self.seed_var, width=7).grid(row=0, column=5, padx=4)
        ttk.Checkbutton(settings, text="AI in every seat",
                        variable=self.all_seats_var).grid(row=0, column=6, padx=(10, 0))
        ttk.Checkbutton(settings, text="Greedy",
                        variable=self.deterministic_var).grid(row=1, column=6, padx=(10, 0))
        ttk.Button(settings, text="New game", command=self.reset_game).grid(row=0, column=7, padx=(8, 0))

    def _build_scene(self) -> None:
        body = tk.Frame(self.root, bg="#101820", padx=18, pady=14)
        body.pack(fill="both", expand=True)

        self.mine_canvas = tk.Canvas(body, bg="#1b2d35", highlightthickness=0)
        self.mine_canvas.pack(side="left", fill="both", expand=True)
        self.mine_canvas.bind("<Configure>", self._on_canvas_resize)

        side = tk.Frame(body, bg="#14242c", width=235, padx=14, pady=14)
        side.pack(side="right", fill="y", padx=(14, 0))
        side.pack_propagate(False)

        tk.Label(side, text="PLAYERS", fg="#f5c451", bg="#14242c",
                 font=ui_font(11, "bold")).pack(anchor="w")
        self.players_text = tk.Text(
            side, height=14, width=25, bg="#0e1a20", fg="#d7e4e8", relief="flat",
            font=mono_font(10), padx=8, pady=8, state="disabled",
        )
        self.players_text.configure(tabs=("1.55i",))
        self.players_text.pack(fill="x", pady=(8, 18))

        tk.Label(side, text="YOUR ACTION", fg="#f5c451", bg="#14242c",
                 font=ui_font(11, "bold")).pack(anchor="w")
        self.action_menu = ttk.Combobox(side, state="readonly", width=20)
        self.action_menu.pack(fill="x", pady=(8, 8))
        self.action_menu.bind("<<ComboboxSelected>>", self._action_label_selected)

        self.step_button = ttk.Button(side, text="Take action", command=self.take_action)
        self.step_button.pack(fill="x")
        self.autoplay_button = ttk.Button(side, text="Autoplay", command=self.toggle_autoplay)
        self.autoplay_button.pack(fill="x", pady=(6, 0))
        self.autoplay_step_button = ttk.Button(side, text="Autoplay step", command=self.autoplay_once)
        self.autoplay_step_button.pack(fill="x", pady=(6, 0))

        tk.Label(side, text="AI POLICY", fg="#f5c451", bg="#14242c",
                 font=ui_font(11, "bold")).pack(anchor="w", pady=(18, 0))
        self.policy_text = tk.Text(
            side, height=7, width=25, bg="#0e1a20", fg="#9fd6c0", relief="flat",
            font=mono_font(9), padx=8, pady=8, state="disabled",
        )
        self.policy_text.pack(fill="x", pady=(8, 0))

        message_frame = tk.Frame(side, bg="#14242c")
        message_frame.pack(fill="both", expand=True, pady=(18, 0))
        self.message_text = tk.Text(
            message_frame, bg="#0e1a20", fg="#b7c8cc", relief="flat",
            font=ui_font(9), padx=8, pady=8, wrap="word", state="disabled",
        )
        self.message_text.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(message_frame, orient="vertical", command=self.message_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.message_text.configure(yscrollcommand=scrollbar.set)
        self.message_var.trace_add("write", self._append_message)

    # -- message log -------------------------------------------------------

    def _append_message(self, *_args: object) -> None:
        text = self.message_var.get()
        if not text:
            return
        self.message_text.configure(state="normal")
        self.message_text.insert("end", text + "\n")
        self.message_text.configure(state="disabled")
        self.message_text.see("end")

    def _post_message(self, text: str) -> None:
        if text:
            self.message_var.set(text)

    def _clear_messages(self) -> None:
        self.message_text.configure(state="normal")
        self.message_text.delete("1.0", "end")
        self.message_text.configure(state="disabled")

    # -- resize debouncing -------------------------------------------------

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(RESIZE_DEBOUNCE_MS, self.draw_scene)

    # -- action dropdown ---------------------------------------------------

    def _action_label_selected(self, _event: tk.Event) -> None:
        selected = self.action_menu.get()
        labels = list(self.action_menu["values"])

        if self.controller.in_tie_breaker or self.controller.blast_pending:
            if selected in labels:
                self.action_var.set(labels.index(selected))
            return

        self.blast_requested = selected.startswith(BLAST_ACTION_PREFIX)
        if self.blast_requested:
            selected = selected[len(BLAST_ACTION_PREFIX):]
        for action in range(self.controller.env.action_space.n):
            if self._action_label(action) == selected:
                self.action_var.set(action)
                return

    def _action_label(self, action: int) -> str:
        if action >= ANNOUNCE_OFFSET:
            return f"{BLAST_ACTION_PREFIX}{self._action_label(action - ANNOUNCE_OFFSET)}"
        if action in SPECIAL_ACTION_LABELS:
            return SPECIAL_ACTION_LABELS[action]
        row, col = divmod(action, 5)
        return f"Card row {row}, column {col + 1}"

    # -- lifecycle ---------------------------------------------------------

    def reset_game(self) -> None:
        self.stop_autoplay()
        self._clear_messages()
        try:
            self.controller.reset(
                opponents=self.opponents_var.get(),
                policy=self.policy_var.get(),
                seed=self.seed_var.get(),
                all_seats=self.all_seats_var.get(),
                deterministic=self.deterministic_var.get(),
            )
            self.controller.ensure_training_ready(progress_callback=self._on_training_progress)
        except (tk.TclError, ValueError, FileNotFoundError) as error:
            self._post_message(f"Could not start game: {error}")
            return
        self._update_action_options()
        self._post_message("Choose a destination. Opponents move automatically after your action.")
        self.step_button.state(["!disabled"])
        self.draw_scene()

    def _on_training_progress(self, message: str) -> None:
        self._post_message(message)
        self.root.update_idletasks()

    def setup_tie_break_test(self) -> None:
        self.controller.setup_tie_break_test()
        self._update_action_options()
        self._post_message("Tie-break test: Player 1 draws first. Choose a mine card or the deck.")
        self.draw_scene()

    def setup_blast_test(self) -> None:
        self.controller.setup_blast_test()
        self._update_action_options()
        self._post_message("Blast test: You have 2 Dynamite. Open the 'YOUR ACTION' dropdown and choose a 'Blast: ...' action!")
        self.draw_scene()

    # -- user actions ------------------------------------------------------

    def take_action(self) -> None:
        if self.controller.in_tie_breaker:
            self._process_results(
                self.controller.resolve_tie_break_choice(int(self.action_var.get()))
            )
            self._update_action_options()
            self.draw_scene()
            return

        if self.controller.blast_pending:
            result = self.controller.resolve_player_blast(int(self.action_var.get()))
            self._process_results([result])
            self._update_action_options()
            self.draw_scene()
            return

        blast_requested = self.blast_requested
        self.blast_requested = False
        result = self.controller.take_player_action(int(self.action_var.get()), blast_requested)
        self._process_results([result])
        self._update_action_options()
        self.draw_scene()

    # -- autoplay ----------------------------------------------------------

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

    def _autoplay_step(self) -> None:
        if not self.autoplay:
            return
        try:
            self._run_single_player_step()
        except Exception as exc:
            self._post_message(f"Autoplay error: {exc}")
            self.stop_autoplay()
            return
        if self.autoplay:
            self.autoplay_job = self.root.after(AUTOPLAY_DELAY_MS, self._autoplay_step)

    def autoplay_once(self) -> None:
        if self.autoplay:
            self.stop_autoplay()
        self._run_single_player_step()

    def _run_single_player_step(self) -> None:
        self._process_results([self.controller.step_one_player_auto()])
        self._update_action_options()
        choice = self.controller.last_choice
        if choice is not None and self.controller.last_choice_seat == 0:
            # Show what the agent actually chose, rather than sampling it twice.
            self.action_var.set(int(choice.action))
            self.action_menu.set(self._action_label(int(choice.action)))
        self.draw_scene()

    # -- result handling ---------------------------------------------------

    def _process_results(self, results: list[StepResult]) -> None:
        game_over = False
        for result in results:
            if result.message:
                self._post_message(result.message)
            if result.game_over:
                game_over = True
        if game_over:
            self.stop_autoplay()
            self.step_button.state(["disabled"])
            self._post_message("Game over. Start a new game to play again.")

    # -- rendering ---------------------------------------------------------

    def draw_scene(self) -> None:
        self.renderer.render(self.controller.env)
        env = self.controller.env
        self.haul_var.set(f"Haul {env.haul}/3")
        top_back = BACKSIDE_SYMBOLS[env.cards_remaining[0][1]] if env.cards_remaining else "-"
        self.stack_var.set(f"Stack: {len(env.cards_remaining)}  |  top back: {top_back}")
        self._update_player_list()
        self._update_policy_panel()

    def _update_player_list(self) -> None:
        env = self.controller.env
        self.players_text.configure(state="normal")
        self.players_text.delete("1.0", "end")
        for i, player in enumerate(env.players):
            starting = " (starting player)" if i == env.starting_player else ""
            status = " OUT" if player.escaped else ""
            collected = ", ".join(BACKSIDE_SYMBOLS[b] for b in player.collected_backsides) or "-"
            if i == 0:
                gems = ", ".join(
                    BACKSIDE_SYMBOLS[b] for f, b in player.collected_cards if f == GEM
                ) or "-"
                dyn = ", ".join(
                    BACKSIDE_SYMBOLS[b] for f, b in player.collected_cards if f == DYNAMITE
                ) or "-"
                self.players_text.insert(
                    "end", f"P{i + 1} (you){status}{starting}\t{player.score} VP\n"
                )
                self.players_text.insert("end", f"gems: {gems}  dynamite: {dyn}\n\n")
            else:
                self.players_text.insert(
                    "end", f"P{i + 1}{status}{starting}\t{player.score} VP\nbacks: {collected}\n\n"
                )
        if env.used_blast_backsides:
            used = ", ".join(BACKSIDE_SYMBOLS[b] for b in env.used_blast_backsides)
            self.players_text.insert("end", f"used cards: {used}\n")
        self.players_text.configure(state="disabled")

    def _update_policy_panel(self) -> None:
        """Top action probabilities for the neural seat (feedback aid)."""
        self.policy_text.configure(state="normal")
        self.policy_text.delete("1.0", "end")
        snapshot = self.controller.policy_snapshot(limit=6)
        if not snapshot and not self.controller.is_neural:
            self.policy_text.insert("end", "scripted policy\n(no network)")
        elif not snapshot:
            self.policy_text.insert("end", "no decision pending")
        else:
            chosen = self.controller.last_choice
            chosen_action = int(chosen.action) if chosen is not None else None
            for action, probability in snapshot:
                marker = ">" if action == chosen_action else " "
                self.policy_text.insert(
                    "end",
                    f"{marker} {probability * 100:5.1f}%  {self._action_label(action)}\n",
                )
            if any(
                self.controller.env.players[seat].blast_pending
                for seat in range(self.controller.env.n_players)
            ):
                self.policy_text.insert("end", "\nblast pending\n")
        self.policy_text.configure(state="disabled")

    # -- action menu -------------------------------------------------------

    def _update_action_options(self) -> None:
        if self.controller.in_tie_breaker:
            labels = [
                "Draw from deck" if src == "deck"
                else f"Draw from mine (row {r}, column {c + 1})"
                for src, r, c in self.controller.tie_break_options()
            ]
            self._set_action_labels(labels)
            return

        player = self.controller.player
        if self.controller.blast_pending:
            labels = [
                f"Blast choice {i + 1} ({BACKSIDE_SYMBOLS[card[1]]})"
                for i, card in enumerate(player.blast_cards)
            ]
            self._set_action_labels(labels)
            return

        legal = self.controller.legal_actions()
        labels = [self._action_label(a) for a in legal]
        if player.dynamite >= 2:
            depth = self.controller.movement_depth()
            labels.extend(
                f"{BLAST_ACTION_PREFIX}{self._action_label(a)}"
                for a in legal if a < PIT_CAGE and a // 5 < depth
            )
        self.action_menu.configure(values=labels)
        if legal:
            self.action_var.set(legal[0])
            self.action_menu.set(labels[0])
        else:
            self.action_var.set(0)
            self.action_menu.set("")

    def _set_action_labels(self, labels: list[str]) -> None:
        self.action_menu.configure(values=labels)
        self.action_var.set(0)
        self.action_menu.set(labels[0] if labels else "")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="View MineEnv player progression.")
    parser.add_argument("--opponents", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        default="simple",
        help="scripted policy, or neural:<checkpoint> from checkpoints/",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--all-seats",
        action="store_true",
        help="let the chosen policy (usually neural:<checkpoint>) play every seat",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="play the network's argmax instead of sampling at temperature 0.7",
    )
    parser.add_argument("--blast-test", action="store_true",
                        help="open directly with enough dynamite to blast")
    parser.add_argument("--tie-break-test", action="store_true",
                        help="open directly at a tie-break scenario")
    args = parser.parse_args()
    root = tk.Tk()
    viewer = MineViewer(
        root,
        args.opponents,
        args.policy,
        args.seed,
        all_seats=args.all_seats,
        deterministic=args.deterministic,
    )
    if args.tie_break_test:
        viewer.setup_tie_break_test()
    elif args.blast_test:
        viewer.setup_blast_test()
    root.mainloop()


if __name__ == "__main__":
    main()