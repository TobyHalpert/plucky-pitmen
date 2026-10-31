"""Multi-agent game runner for Plucky Pitmen.

The environment's public ``step`` API is single-learner oriented.  This module
drives complete games in which *every* seat is controlled by an agent,
including blast announcements, blast draws and the tie-break, and records one
``learning.Decision`` per move for whichever seats are learning.

Reward signal
-------------
Scores change only at the end of a haul, far too sparse for short games.  Each
player therefore receives a small dense shaping reward for loot collected and
for leaving the mine safely, plus the real score change and a terminal bonus
for winning.  Shaping reads the true card fronts; that is a *training* signal
only - the policy still sees nothing but the cheat-safe context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from agents import ActionChoice
from game_env import DYNAMITE, PASS, PIT_CAGE, PlayerActionContext, MineEnv

MAX_GAME_STEPS = 4000
# A planning phase can be prolonged indefinitely by a policy that keeps
# bouncing between a pit cage and its lorry.  After this many decisions in one
# round, the driver forces a pass whenever passing is legal, which guarantees
# that a round (and therefore a game) always resolves.
MAX_PLANNING_FACTOR = 4

# Dense shaping, kept small relative to the points that decide a haul.
GEM_SHAPING = 0.20
OTHER_CARD_SHAPING = 0.02
ESCAPE_SHAPING = 0.30
DEATH_SHAPING = -0.05
WIN_BONUS = 1.0


@dataclass
class HaulSummary:
    haul: int
    scores: list[int]
    escaped: list[int]
    dead: list[int]
    outcome: str
    dragon_player: int | None = None
    gems: list[int] = field(default_factory=list)
    cards: list[int] = field(default_factory=list)


@dataclass
class GameResult:
    seed: int
    scores: list[int]
    winner: int | None
    turns: int
    hauls: list[HaulSummary] = field(default_factory=list)
    blasts_announced: int = 0
    blast_choices: int = 0
    agent_names: list[str] = field(default_factory=list)
    aborted: bool = False

    @property
    def dragons(self) -> int:
        return sum(1 for haul in self.hauls if "dragon" in haul.outcome)

    @property
    def escapes(self) -> int:
        return sum(len(haul.escaped) for haul in self.hauls)

    @property
    def lorry_standoffs(self) -> int:
        """Hauls that ended because everybody left inside stood on a lorry."""
        return sum(1 for haul in self.hauls if haul.outcome == "lorry")


# ----------------------------------------------------------------------
# Reward attribution
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class _Snapshot:
    haul: int
    scores: tuple[int, ...]
    gems: tuple[int, ...]
    cards: tuple[int, ...]
    escaped: tuple[bool, ...]
    dead: tuple[bool, ...]


def _snapshot(env: MineEnv) -> _Snapshot:
    return _Snapshot(
        haul=env.haul,
        scores=tuple(player.score for player in env.players),
        gems=tuple(player.gems for player in env.players),
        cards=tuple(len(player.collected_cards) for player in env.players),
        escaped=tuple(player.escaped for player in env.players),
        dead=tuple(player.dead for player in env.players),
    )


def _reward_vector(
    env: MineEnv, before: _Snapshot, escape_shaping: float = ESCAPE_SHAPING
) -> list[float]:
    rewards = [0.0] * len(env.players)
    haul_changed = env.haul != before.haul
    for index, player in enumerate(env.players):
        reward = player.score - before.scores[index]
        if not haul_changed:
            gems = player.gems - before.gems[index]
            cards = len(player.collected_cards) - before.cards[index]
            reward += GEM_SHAPING * gems
            reward += OTHER_CARD_SHAPING * (cards - gems)
            if player.escaped and not before.escaped[index]:
                reward += escape_shaping
            if player.dead and not before.dead[index]:
                reward += DEATH_SHAPING
        rewards[index] = float(reward)
    return rewards


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def _choice_for(agent: Any, ctx: PlayerActionContext) -> ActionChoice:
    if hasattr(agent, "decide"):
        choice = agent.decide(ctx)
    else:  # pragma: no cover - defensive, Agent.decide is the normal path
        choice = ActionChoice(int(agent.act(ctx)))
    action = int(choice.action)
    announce = bool(choice.announce_blast)
    if not announce and hasattr(agent, "wants_blast"):
        announce = bool(agent.wants_blast(ctx))
    if announce:
        dynamite = sum(1 for front, _ in ctx.private_cards if front == DYNAMITE)
        if dynamite < 2 or ctx.own_blast_pending or not 0 <= action < PIT_CAGE:
            announce = False
    return ActionChoice(action, announce)


def _tie_break(env: MineEnv, agent: Any, player_index: int) -> dict[str, Any]:
    ctx = env.player_action_context(player_index)
    source, row, column = agent.tie_break(ctx)
    try:
        return env.resolve_tie_breaker(source, row, column)
    except ValueError:
        if env.cards_remaining:
            return env.resolve_tie_breaker("deck")
        for row_index, cards in enumerate(env.rows):
            for column_index, card in enumerate(cards):
                if card is not None:
                    return env.resolve_tie_breaker(
                        "mine", row_index, column_index
                    )
        raise


def play_game(
    agents: Sequence[Any],
    *,
    seed: int | None = None,
    max_steps: int = MAX_GAME_STEPS,
    decisions: list[Any] | None = None,
    reward_shaping: bool = True,
    escape_shaping: float = ESCAPE_SHAPING,
    passes: str = "n-1",
    solo_escape_bonus: bool = True,
    blast_choice_sees_fronts: bool = True,
) -> GameResult:
    """Play one complete 3-haul game with one agent per seat."""
    n_players = len(agents)
    if n_players not in (3, 4, 5):
        raise ValueError("Plucky Pitmen supports 3 to 5 players")

    env = MineEnv(
        opponents=n_players - 1,
        passes=passes,
        solo_escape_bonus=solo_escape_bonus,
        blast_choice_sees_fronts=blast_choice_sees_fronts,
    )
    env.reset(seed=seed)
    learning = decisions is not None
    if decisions is None:
        decisions = []
    last_decision: dict[int, Any] = {}
    blasts_announced = 0
    blast_choices = 0
    winner: int | None = None
    steps = 0
    aborted = False
    round_key = (env.haul, env.round)
    planning_steps = 0

    def flush(player: int, reward: float) -> None:
        if not reward:
            return
        decision = last_decision.get(player)
        if decision is not None:
            decision.reward += float(reward)

    while not env.game_finished and steps < max_steps:
        steps += 1
        before = _snapshot(env) if learning else None

        # 1. Tie-break ---------------------------------------------------
        if env.tie_breaker_contenders:
            contenders = env.tie_breaker_contenders
            player_index = contenders[env.tie_breaker_turn % len(contenders)]
            outcome = _tie_break(env, agents[player_index], player_index)
            if outcome.get("winner") is not None:
                winner = int(outcome["winner"])
            continue

        # 2. Pending blast draws -----------------------------------------
        pending = [
            index for index, player in enumerate(env.players)
            if player.blast_pending and player.blast_cards
        ]
        if pending:
            player_index = pending[0]
            ctx = env.player_action_context(player_index)
            cards = tuple(ctx.own_blast_cards)
            choice = int(agents[player_index].blast_choice(cards, ctx))
            choice = min(max(choice, 0), len(cards) - 1)
            blast_choices += 1
            env.resolve_blast(player_index, choice)
            if learning:
                for index, reward in enumerate(
                    _reward_vector(env, before, escape_shaping)
                ):
                    flush(index, reward)
            continue

        # 3. Planning phase ----------------------------------------------
        if (env.haul, env.round) != round_key:
            round_key = (env.haul, env.round)
            planning_steps = 0
        active = env.active_player_indices()
        if not active:
            break
        if env.planning_player not in active:
            env.planning_player = active[0]
        player_index = env.planning_player

        ctx = env.player_action_context(player_index)
        if not ctx.legal_actions:
            break

        planning_steps += 1
        forced_pass = (
            planning_steps > MAX_PLANNING_FACTOR * n_players
            and PASS in ctx.legal_actions
        )
        count_before = len(decisions)
        choice = (
            ActionChoice(PASS)
            if forced_pass
            else _choice_for(agents[player_index], ctx)
        )

        try:
            _obs, _reward, terminated, _truncated, info = env.step_one_player(
                choice.action,
                announce_blast=choice.announce_blast,
                force_action=True,
            )
        except ValueError:
            # Invalid blast announcement: retry the move without it.
            _obs, _reward, terminated, _truncated, info = env.step_one_player(
                choice.action, force_action=True
            )
            choice = ActionChoice(choice.action, False)

        if choice.announce_blast:
            blasts_announced += 1

        for decision in decisions[count_before:]:
            last_decision[decision.player] = decision

        if "winner" in info and info["winner"] is not None:
            winner = int(info["winner"])

        if learning:
            for index, reward in enumerate(
                _reward_vector(env, before, escape_shaping)
            ):
                flush(index, reward)

        if terminated:
            break

    aborted = not env.game_finished

    if env.game_finished and winner is None:
        best = max(player.score for player in env.players)
        leaders = [i for i, p in enumerate(env.players) if p.score == best]
        winner = leaders[0] if len(leaders) == 1 else None

    if learning:
        if winner is not None:
            flush(winner, WIN_BONUS)
        else:
            best = max(player.score for player in env.players)
            for index, player in enumerate(env.players):
                if player.score == best:
                    flush(index, WIN_BONUS)

    return GameResult(
        seed=-1 if seed is None else int(seed),
        scores=[player.score for player in env.players],
        winner=winner,
        turns=steps,
        hauls=[
            HaulSummary(
                haul=int(record["haul"]),
                scores=list(record["scores"]),
                escaped=list(record["escaped"]),
                dead=list(record["dead"]),
                outcome=str(record["outcome"]),
                dragon_player=record.get("dragon_player"),
                gems=list(record.get("gems", [])),
                cards=list(record.get("cards", [])),
            )
            for record in env.haul_history
        ],
        blasts_announced=blasts_announced,
        blast_choices=blast_choices,
        agent_names=[getattr(agent, "name", type(agent).__name__) for agent in agents],
        aborted=aborted,
    )


# ----------------------------------------------------------------------
# Series with seat rotation
# ----------------------------------------------------------------------
AgentFactory = Callable[[int], Any]


@dataclass
class SeriesGame:
    result: GameResult
    policy_of_seat: list[int]

    def scores_by_policy(self) -> list[int]:
        scores = [0] * len(self.policy_of_seat)
        for seat, policy in enumerate(self.policy_of_seat):
            scores[policy] = self.result.scores[seat]
        return scores

    def winner_policy(self) -> int | None:
        if self.result.winner is None:
            return None
        return self.policy_of_seat[self.result.winner]


def play_series(
    factories: Sequence[AgentFactory],
    games: int,
    *,
    seed: int = 0,
    rotate_seats: bool = True,
    decisions: list[Any] | None = None,
    passes: str = "n-1",
    solo_escape_bonus: bool = True,
    blast_choice_sees_fronts: bool = True,
) -> list[SeriesGame]:
    """Play ``games`` games between policy factories, rotating seats.

    ``factories`` are indexed by *policy*; each game shifts who sits where so
    that no policy can profit from a fixed seat or from always starting.
    """
    series: list[SeriesGame] = []
    n_policies = len(factories)
    for game in range(games):
        game_seed = seed + game
        offset = game % n_policies if rotate_seats else 0
        policy_of_seat = [
            (seat + offset) % n_policies for seat in range(n_policies)
        ]
        agents = [
            factories[policy](game_seed * 1000 + seat)
            for seat, policy in enumerate(policy_of_seat)
        ]
        result = play_game(
            agents,
            seed=game_seed,
            decisions=decisions,
                passes=passes,
            solo_escape_bonus=solo_escape_bonus,
            blast_choice_sees_fronts=blast_choice_sees_fronts,
        )
        series.append(SeriesGame(result, policy_of_seat))
    return series


__all__ = [
    "AgentFactory",
    "GameResult",
    "HaulSummary",
    "MAX_GAME_STEPS",
    "SeriesGame",
    "play_game",
    "play_series",
]
