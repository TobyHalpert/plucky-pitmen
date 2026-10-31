"""Tk-free game controller for the ``mine_gui`` viewer.

Everything here runs headless, so the interactive viewer can be tested without a
display: the controller owns the environment, decides who acts (a human, a
scripted policy, or a checkpointed self-play network) and returns plain
``StepResult`` messages for the Tk layer to display.

The neural policies are the interesting ones: ``policy="neural:selfplay4"``
loads ``checkpoints/selfplay4.npz`` and lets the trained agent play seat 1,
announcing and resolving blasts through its own policy (the viewer would
otherwise never show them, because blasts are an optional half of the action
space). ``all_seats=True`` puts a copy of the same network in every seat, which
is the cheapest way to watch what the learner actually does.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agents import ActionChoice, NeuralAgent, make_scripted_agent
from features import action_mask, state_action_matrix
from game_env import (
    BACKSIDE_A,
    BACKSIDE_B,
    BACKSIDE_C,
    DRAGON,
    DYNAMITE,
    GEM,
    LORRY,
    PASS,
    PIT_CAGE,
    HAULS_PER_GAME,
    MineEnv,
)
from learning import PolicyNet
from rule_policy import deterministic_policy
from selfplay import _choice_for
from train_q_learning import select_action, train

FRONT_LABELS = {GEM: "GEM", DRAGON: "DRAGON", DYNAMITE: "DYNAMITE"}
SCRIPTED_STRATEGIES = ("simple", "training", "random", "cautious", "greedy")
# The rule bot's movement, but it cashes in every dynamite pair it can. Blasts
# are rare in self-play (the learner seldom holds two dynamite), so this is the
# only way to watch the mechanic — and the ~50% chance that it wakes the dragon.
BLAST_STRATEGY = "blaster"
TRAINING_EPISODES = 1000
TRAINING_CACHE_VERSION = 2
TRAINING_CACHE_DIR = Path(__file__).with_name(".training_cache")
CHECKPOINT_DIR = Path(__file__).with_name("checkpoints")
NEURAL_PREFIX = "neural:"
NEURAL_TEMPERATURE = 0.7


# ---------------------------------------------------------------------------
# Which policies the viewer can offer
# ---------------------------------------------------------------------------

def available_checkpoints(directory: Path | None = None) -> list[Path]:
    """Checkpoints the viewer can load, newest name first."""
    directory = Path(directory if directory is not None else CHECKPOINT_DIR)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.npz"))


def agent_policies(directory: Path | None = None) -> list[str]:
    """Policy names for the viewer's dropdown: scripted ones plus checkpoints."""
    return (
        list(SCRIPTED_STRATEGIES)
        + [BLAST_STRATEGY]
        + [f"{NEURAL_PREFIX}{path.stem}" for path in available_checkpoints(directory)]
    )


def resolve_checkpoint(policy: str, directory: Path | None = None) -> Path | None:
    """Map ``"neural:selfplay4"`` (or ``"neural:/path/net.npz"``) to a file."""
    if not policy.startswith(NEURAL_PREFIX):
        return None
    spec = policy[len(NEURAL_PREFIX):]
    direct = Path(spec)
    if direct.suffix == ".npz" and direct.exists():
        return direct
    for path in available_checkpoints(directory):
        if path.stem == spec:
            return path
    raise FileNotFoundError(
        f"no checkpoint named {spec!r} in "
        f"{Path(directory) if directory is not None else CHECKPOINT_DIR}"
    )


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Outcome of advancing the simulation by one atomic step."""
    message: str = ""
    game_over: bool = False


class GameController:
    """Owns the MineEnv and encapsulates all game-progression logic."""

    def __init__(
        self,
        opponents: int,
        policy: str,
        seed: int,
        *,
        all_seats: bool = False,
        deterministic: bool = False,
    ) -> None:
        self.all_seats = all_seats
        self.deterministic = deterministic
        self._net: PolicyNet | None = None
        self._net_policy: str | None = None
        self.agents: list[Any] = []
        self.last_choice: ActionChoice | None = None
        self.last_choice_seat: int | None = None
        self.opponents = opponents
        self.policy = policy
        self.seed = seed
        self.training_q_table = None
        self.training_rng = np.random.default_rng(seed)
        self.env = MineEnv(opponents=opponents, opponent_policy=self._environment_policy())
        self._shown_displacement_count = 0
        self._build_agents()

    # -- setup -------------------------------------------------------------

    def reset(
        self,
        opponents: int,
        policy: str,
        seed: int,
        *,
        all_seats: bool | None = None,
        deterministic: bool | None = None,
    ) -> None:
        self.opponents = opponents
        self.policy = policy
        self.seed = seed
        if all_seats is not None:
            self.all_seats = bool(all_seats)
        if deterministic is not None:
            self.deterministic = bool(deterministic)
        self.training_rng = np.random.default_rng(seed)
        self.training_q_table = None
        self.last_choice = None
        self.last_choice_seat = None
        self._shown_displacement_count = 0
        self.env = MineEnv(opponents=opponents, opponent_policy=self._environment_policy())
        self.env.reset(seed=seed)
        self._build_agents()
        self._advance_opponents_to_p1()

    # -- policies ----------------------------------------------------------

    @property
    def is_neural(self) -> bool:
        return self.policy.startswith(NEURAL_PREFIX)

    def policy_net(self) -> PolicyNet | None:
        """Load (and cache) the checkpoint behind ``policy``."""
        if not self.is_neural:
            return None
        if self._net is None or self._net_policy != self.policy:
            path = resolve_checkpoint(self.policy)
            if path is None:  # pragma: no cover - resolve_checkpoint raised
                raise FileNotFoundError(self.policy)
            self._net = PolicyNet.load(path)
            self._net_policy = self.policy
        return self._net

    def _build_agents(self) -> None:
        """One agent per seat this controller decides for; ``None`` = env policy."""
        self.agents = [None] * self.env.n_players
        if self.policy == BLAST_STRATEGY:
            for seat in range(self.env.n_players if self.all_seats else 1):
                self.agents[seat] = make_scripted_agent(
                    "rule", seed=self.seed + 101 * seat, blast="always"
                )
            return
        if not self.is_neural:
            return
        net = self.policy_net()
        self.agents[0] = NeuralAgent(
            net,
            seed=self.seed,
            temperature=NEURAL_TEMPERATURE,
            deterministic=self.deterministic,
        )
        if self.all_seats:
            for seat in range(1, self.env.n_players):
                self.agents[seat] = NeuralAgent(
                    net,
                    seed=self.seed + 101 * seat,
                    temperature=NEURAL_TEMPERATURE,
                    deterministic=self.deterministic,
                )

    def ensure_training_ready(self, progress_callback: Callable[[str], None] | None = None) -> None:
        if self.policy != "training" or self.training_q_table is not None:
            return
        cache_path = self.training_cache_path()
        if cache_path.exists():
            with cache_path.open("rb") as cache_file:
                self.training_q_table = pickle.load(cache_file)
            return
        if progress_callback:
            progress_callback("Training P1 policy...")
        self.training_q_table = train(
            TRAINING_EPISODES, self.opponents, "random", self.seed
        )
        TRAINING_CACHE_DIR.mkdir(exist_ok=True)
        with cache_path.open("wb") as cache_file:
            pickle.dump(self.training_q_table, cache_file)

    def training_cache_path(self) -> Path:
        return TRAINING_CACHE_DIR / (
            f"q_table_opponents_{self.opponents}_seed_{self.seed}_"
            f"episodes_{TRAINING_EPISODES}_v{TRAINING_CACHE_VERSION}.pkl"
        )

    def _environment_policy(self) -> str:
        """Built-in opponent policy for the seats the controller does not own."""
        if self.policy == "training":
            return "random"
        if self.policy in SCRIPTED_STRATEGIES:
            return self.policy
        if self.policy == BLAST_STRATEGY:
            return "simple"
        # A neural policy says nothing about the opponents; play them as the
        # deterministic rule bot, which is the benchmark used in BALANCE.md.
        return "simple"

    def _advance_opponents_to_p1(self) -> None:
        while self.env.planning_player != 0:
            _obs, _reward, terminated, _trunc, _info = self.env.step_one_player()
            if terminated or self.env.players[0].escaped or self.env.players[0].dead:
                return

    # -- state queries -----------------------------------------------------

    @property
    def player(self):
        return self.env.players[0]

    @property
    def in_tie_breaker(self) -> bool:
        return bool(self.env.tie_breaker_contenders)

    @property
    def blast_pending(self) -> bool:
        p = self.player
        return p.blast_pending and bool(p.blast_cards)

    def legal_actions(self) -> list[int]:
        return [i for i, ok in enumerate(self.env.action_mask()) if ok]

    def movement_depth(self) -> int:
        # Wraps the env's private helper so callers stay decoupled.
        return self.env._movement_depth(self.player)

    def tie_break_options(self) -> list[tuple[str, int | None, int | None]]:
        options: list[tuple[str, int | None, int | None]] = []
        if self.env.cards_remaining:
            options.append(("deck", None, None))
        options.extend(
            ("mine", r, c)
            for r, row in enumerate(self.env.rows)
            for c, card in enumerate(row)
            if card is not None
        )
        return options

    # -- test scaffolding --------------------------------------------------

    def setup_tie_break_test(self) -> None:
        self.env.haul = 3
        self.env.starting_player = 0
        self.env.players[0].score = 5
        self.env.players[1].score = 5
        for p in self.env.players[2:]:
            p.score = 0
        self.env.rows = [[None] * self.env.n_players for _ in range(3)]
        self.env.rows[0][0] = (GEM, BACKSIDE_A)
        self.env.rows[0][1] = (DRAGON, BACKSIDE_B)
        self.env.cards_remaining = [(GEM, BACKSIDE_C)]
        self.env._begin_tie_breaker()

    def setup_blast_test(self) -> None:
        """Injects 2 dynamite cards into P1's hand so they can blast immediately."""
        self.env.players[0].collected_cards = [
            (DYNAMITE, BACKSIDE_A),
            (DYNAMITE, BACKSIDE_B),
        ]
        self.env.players[0].position = 3
        self.env.players[0].column = None

    # -- message formatting ------------------------------------------------

    def _consume_displacement_messages(self) -> str:
        new = self.env.displacement_events[self._shown_displacement_count:]
        self._shown_displacement_count += len(new)
        return "".join(
            f" Player {d + 1} stood on his own card, Player {s + 1} displaced him."
            for d, s in new
        )

    def _format_outcome(self, info: dict, reward: float) -> str:
        outcome = info.get("outcome", "continued")
        msg = f"Round {info.get('round', self.env.round)}: {outcome}. Reward {reward:+.2f}."
        dragon_player = info.get("dragon_player")
        if outcome == "dragon" and dragon_player is not None:
            msg += f" Player {dragon_player + 1} awakened the dragon."
        msg += self._consume_displacement_messages()
        return msg

    # -- blast resolution --------------------------------------------------

    def resolve_player_blast(self, choice_index: int, player_index: int = 0) -> StepResult:
        player = self.env.players[player_index]
        if not 0 <= choice_index < len(player.blast_cards):
            return StepResult("Select one of the three blast cards.")
        
        final_haul = self.env.haul == HAULS_PER_GAME
        chosen = self.env.resolve_blast(player_index, choice_index)
        player_num = player_index + 1
        
        if chosen[0] == DRAGON:
            if final_haul:
                return StepResult(
                    f"Blast choice: DRAGON. Player {player_num} awakened the dragon. Game over.",
                    game_over=True,
                )
            return StepResult(
                f"Blast choice: DRAGON. Player {player_num} awakened the dragon. Starting the next haul."
            )
        return StepResult(f"Blast choice: {FRONT_LABELS[chosen[0]]}. Reward +0.00.")

    # -- action taking -----------------------------------------------------

    def take_player_action(self, action: int, announce_blast: bool) -> StepResult:
        if action >= self.env.action_space.n or not self.env.action_mask()[action]:
            return StepResult("That destination is not legal from your current position.")

        # First let any pending opponents move.
        p = self.player
        while self.env.planning_player != 0 and not (p.escaped or p.dead):
            _obs, reward, terminated, _trunc, info = self.env.step_one_player()
            if terminated:
                return StepResult(self._format_outcome(info, reward), game_over=True)

        if p.escaped or p.dead:
            return StepResult("")

        try:
            _obs, reward, terminated, _trunc, info = self.env.step_one_player(
                action, announce_blast=announce_blast
            ) if announce_blast else self.env.step_one_player(action)
        except ValueError as error:
            return StepResult(str(error))

        while not terminated and self.env.planning_player != 0:
            # P1 may have left the mine or died with this action; in that case
            # its turn never comes again and the loop would spin forever.
            if p.escaped or p.dead:
                break
            _obs, reward, terminated, _trunc, info = self.env.step_one_player()

        return StepResult(self._format_outcome(info, reward), game_over=terminated)

    # -- autoplay / auto step ---------------------------------------------

    def autoplay_choose_choice(self, seat: int = 0) -> ActionChoice | None:
        """The decision for ``seat``, including whether it wants to blast.

        A scripted autoplay policy never announces a blast (none of the scripted
        baselines do); the neural policies do, through the same action space and
        the masked-softmax head they are trained with.
        """
        agent = self.agents[seat] if seat < len(self.agents) else None
        if agent is not None:
            return _choice_for(agent, self.env.player_action_context(seat))
        if seat != 0:
            return None
        action = self.autoplay_choose_action()
        return None if action is None else ActionChoice(int(action))

    def autoplay_choose_action(self) -> int | None:
        legal = self.legal_actions()
        if not legal:
            return None
        if self.is_neural:
            choice = self.autoplay_choose_choice(0)
            return None if choice is None else int(choice.action)
        if self.policy == "training" and self.training_q_table is not None:
            return int(select_action(self.env, self.training_q_table, 0.0, self.training_rng))
        if self.policy == "simple":
            return int(deterministic_policy(self.env.player_action_context(0)))
        if self.policy == "random":
            return int(self.training_rng.choice(legal))
        # "cautious" and "greedy" fall through to a safe default until
        # dedicated heuristics are wired in.
        if self.player.position in (PIT_CAGE, LORRY) and PASS in legal:
            return PASS
        return legal[0]

    # -- autoplay driver ---------------------------------------------------

    def pending_blast_seat(self) -> int | None:
        """The seat whose blast is waiting on a card choice, if we own it."""
        for seat, player in enumerate(self.env.players):
            if player.blast_pending and player.blast_cards:
                if seat < len(self.agents) and self.agents[seat] is not None:
                    return seat
        return None

    def _auto_blast(self, seat: int) -> StepResult:
        agent = self.agents[seat]
        cards = self.env.players[seat].blast_cards
        ctx = self.env.player_action_context(seat)
        index = int(agent.blast_choice(cards, ctx))
        index = min(max(index, 0), len(cards) - 1)
        return self.resolve_player_blast(index, seat)

    def _auto_tie_break(self) -> StepResult:
        """Resolve one tie-break draw for whoever is on turn."""
        contenders = self.env.tie_breaker_contenders
        current = contenders[self.env.tie_breaker_turn % len(contenders)]
        agent = self.agents[current] if current < len(self.agents) else None
        ctx = self.env.player_action_context(current)
        if agent is not None:
            source, row, column = agent.tie_break(ctx)
        else:
            options = self.tie_break_options()
            source, row, column = (
                options[0]
                if len(options) == 1
                else options[int(self.env.rng.integers(len(options)))]
            )
        try:
            result = self.env.resolve_tie_breaker(source, row, column)
        except ValueError:
            # The agent asked for a card that is not there (an empty draw pile
            # is the usual case). Fall back to any legal source instead of
            # retrying the same illegal choice forever.
            options = self.tie_break_options()
            if not options:
                return StepResult("no tie-break card left", game_over=True)
            source, row, column = options[0]
            result = self.env.resolve_tie_breaker(source, row, column)
        step = self._tie_break_message(result)
        if step.game_over:
            return step
        if not self.env.tie_breaker_contenders:
            step.game_over = True
        return step

    def step_one_player_auto(self) -> StepResult:
        """Advance one decision: a tie-break draw, a blast, or a move.

        Seats this controller owns (seat 1, plus every seat when ``all_seats`` is
        on) act through their agent, so a neural seat both announces and resolves
        its own blasts. The other seats keep playing the environment's built-in
        policy, exactly as before.
        """
        if self.in_tie_breaker:
            return self._auto_tie_break()

        seat = self.pending_blast_seat()
        if seat is not None:
            return self._auto_blast(seat)

        planning = self.env.planning_player
        agent = self.agents[planning] if planning < len(self.agents) else None
        if agent is None and planning != 0:
            # An opponent the environment plays itself.
            _obs, reward, terminated, _trunc, info = self.env.step_one_player()
            return StepResult(self._format_outcome(info, reward), game_over=terminated)

        # Seat 1 is either our agent or the scripted autoplay policy; either way
        # the step has to be taken explicitly, because the environment only
        # auto-plays player 0 when it is the one being driven by a policy.
        choice = self.autoplay_choose_choice(planning)
        if choice is None:
            return StepResult("")
        self.last_choice = choice
        self.last_choice_seat = planning
        try:
            _obs, reward, terminated, _trunc, info = self.env.step_one_player(
                int(choice.action),
                announce_blast=bool(choice.announce_blast),
                force_action=True,
            )
        except ValueError as error:
            return StepResult(str(error))
        message = self._format_outcome(info, reward)
        if choice.announce_blast and not info.get("blast_active"):
            message = f"Player {planning + 1} announced a blast. " + message
        return StepResult(message, game_over=terminated)

    # -- what the network is thinking --------------------------------------

    def policy_snapshot(
        self, seat: int = 0, limit: int = 6
    ) -> list[tuple[int, float]]:
        """Top actions and probabilities for the neural seat, for the viewer.

        Empty for scripted policies, during a tie-break, and while a blast is
        waiting on a card choice (the action space does not describe either).
        """
        net = self.policy_net()
        if net is None or not self.env.players:
            return []
        if self.in_tie_breaker or self.blast_pending:
            return []
        ctx = self.env.player_action_context(seat)
        if not ctx.legal_actions:
            return []
        mask = action_mask(ctx)
        if not mask.any():
            return []
        probs = net.masked_probs(
            state_action_matrix(ctx), mask, temperature=NEURAL_TEMPERATURE
        )
        order = np.argsort(probs)[::-1]
        top = [
            (int(action), float(probs[action]))
            for action in order
            if probs[action] > 0
        ]
        return top[:limit]

    # -- tie-break ---------------------------------------------------------

    def resolve_tie_break_choice(self, choice_index: int) -> list[StepResult]:
        """Resolve P1's tie-break choice and any subsequent opponent draws."""
        options = self.tie_break_options()
        if not 0 <= choice_index < len(options):
            return [StepResult("Select a tie-break card source.")]

        results: list[StepResult] = []
        source, row, column = options[choice_index]
        results.append(self._tie_break_message(self.env.resolve_tie_breaker(source, row, column)))

        while self.env.tie_breaker_contenders:
            current = self.env.tie_breaker_contenders[self.env.tie_breaker_turn]
            if current == 0:
                break
            opts = self.tie_break_options()
            src, r, c = opts[0] if len(opts) == 1 else opts[int(self.env.rng.integers(len(opts)))]
            results.append(self._tie_break_message(self.env.resolve_tie_breaker(src, r, c)))
            if results[-1].game_over:
                break
        return results

    @staticmethod
    def _tie_break_message(result: dict) -> StepResult:
        player_num = result["player"] + 1
        if result.get("game_over"):
            return StepResult(
                f"Player {player_num} drew a dragon and was eliminated from the tie-break. "
                f"Player {result['winner'] + 1} wins the tie-break. Game over.",
                game_over=True,
            )
        if result["outcome"] == "dragon":
            return StepResult(
                f"Player {player_num} drew a dragon and was eliminated from the tie-break."
            )
        return StepResult(f"Player {player_num} drew a safe tie-break card.")


