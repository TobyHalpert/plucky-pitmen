"""Tests for the multi-agent self-play stack."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from agents import (
    NeuralAgent,
    RuleAgent,
    make_scripted_agent,
)
from features import (
    N_ACTION_FEATURES,
    N_ACTIONS,
    N_BASE_ACTIONS,
    N_INPUT_FEATURES,
    N_STATE_FEATURES,
    action_mask,
    split_action,
    state_action_matrix,
    state_features,
)
from game_env import (
    DRAGON,
    DYNAMITE,
    FRONT_UNKNOWN,
    GEM,
    LORRY,
    MAX_COLUMNS,
    PASS,
    PIT_CAGE,
    MineEnv,
)
from learning import Decision, PolicyNet, compute_returns
from selfplay import play_game, play_series


def _scripted_agents(players: int, seed: int = 0):
    policies = ("rule", "greedy", "random", "cautious")
    return [
        make_scripted_agent(policies[index % len(policies)], seed=seed + index)
        for index in range(players)
    ]


@pytest.mark.parametrize("players", [3, 4, 5])
def test_games_terminate_for_every_player_count(players):
    for seed in range(3):
        result = play_game(_scripted_agents(players, seed), seed=seed)
        assert not result.aborted
        assert len(result.hauls) == 3
        assert all(score >= -1 for score in result.scores)
        assert result.turns > 0


def test_driver_forces_a_pass_before_a_round_can_stall():
    """A pathological "always move" policy must still finish the game."""

    class Restless(RuleAgent):
        name = "restless"

        def act(self, ctx):
            cards = [a for a in ctx.legal_actions if a < PIT_CAGE]
            if cards:
                return cards[0]
            if LORRY in ctx.legal_actions:
                return LORRY
            if PIT_CAGE in ctx.legal_actions:
                return PIT_CAGE
            return PASS

    result = play_game([Restless() for _ in range(3)], seed=2)
    assert not result.aborted
    assert len(result.hauls) == 3


def test_blast_announcements_reach_the_environment():
    class Blaster(RuleAgent):
        name = "blaster"

        def wants_blast(self, ctx):
            dynamite = sum(1 for front, _ in ctx.private_cards if front == DYNAMITE)
            return dynamite >= 2

    blasts = 0
    for seed in range(8):
        agents = [Blaster() for _ in range(4)]
        result = play_game(agents, seed=seed)
        blasts += result.blasts_announced
    assert blasts > 0


def test_play_series_rotates_every_policy_through_every_seat():
    factories = [
        lambda seed, policy=policy: make_scripted_agent(policy, seed=seed)
        for policy in ("rule", "greedy", "random")
    ]
    series = play_series(factories, 6, seed=100, rotate_seats=True)
    seats_seen = {policy: set() for policy in range(3)}
    for game in series:
        for seat, policy in enumerate(game.policy_of_seat):
            seats_seen[policy].add(seat)
    assert all(seats == {0, 1, 2} for seats in seats_seen.values())


# ----------------------------------------------------------------------
# Features and cheat safety
# ----------------------------------------------------------------------
def test_feature_dimensions_are_fixed():
    env = MineEnv(opponents=2)
    env.reset(seed=1)
    ctx = env.player_action_context(0)
    state = state_features(ctx)
    assert state.shape == (N_STATE_FEATURES,)
    matrix = state_action_matrix(ctx, state=state)
    assert matrix.shape == (N_ACTIONS, N_INPUT_FEATURES)
    assert N_INPUT_FEATURES == N_STATE_FEATURES + N_ACTION_FEATURES


def test_features_ignore_hidden_opponent_card_fronts():
    env = MineEnv(opponents=2)
    env.reset(seed=5)
    before = state_action_matrix(env.player_action_context(0))

    for row in env.rows:
        for column in range(1, len(row)):
            card = row[column]
            if card is None:
                continue
            front, back = card
            replacement = GEM if front != GEM else DYNAMITE
            row[column] = (replacement, back)

    after = state_action_matrix(env.player_action_context(0))
    assert np.array_equal(before, after)


def test_features_ignore_opponent_collected_card_fronts():
    env = MineEnv(opponents=2)
    env.reset(seed=6)
    env.players[1].collected_cards = [(GEM, 0), (DYNAMITE, 1)]
    before = state_features(env.player_action_context(0))

    # Same backsides, different (private) fronts: must be indistinguishable.
    env.players[1].collected_cards = [(DYNAMITE, 0), (GEM, 1)]
    after = state_features(env.player_action_context(0))
    assert np.array_equal(before, after)


def test_feature_encoder_is_seat_invariant():
    """The same situation encoded for two seats should produce mirrored data."""

    env = MineEnv(opponents=2)
    env.reset(seed=8)
    env.starting_player = 0
    env.planning_player = 0
    ctx0 = env.player_action_context(0)
    ctx1 = env.player_action_context(1)

    state0 = state_features(ctx0)
    state1 = state_features(ctx1)
    # Not identical situations, but both must be finite and in range.
    assert np.isfinite(state0).all() and np.isfinite(state1).all()
    assert state0.min() >= -2.0 and state0.max() <= 2.0


def test_action_mask_matches_environment_legality():
    env = MineEnv(opponents=2)
    env.reset(seed=2)
    for _ in range(6):
        player_index = env.planning_player
        ctx = env.player_action_context(player_index)
        mask = action_mask(ctx)
        for action in range(N_BASE_ACTIONS):
            assert bool(mask[action]) == env._legal(action, player_index)
        dynamite = sum(
            1 for front, _ in ctx.private_cards if front == DYNAMITE
        )
        can_blast = dynamite >= 2 and not ctx.own_blast_pending
        for action in range(N_BASE_ACTIONS, N_ACTIONS):
            primitive, announces = split_action(action)
            assert bool(mask[action]) == (
                bool(mask[primitive]) and primitive < PIT_CAGE and can_blast
            )
        env.step_one_player(env.legal_actions(player_index)[0], force_action=True)


# ----------------------------------------------------------------------
# Learning
# ----------------------------------------------------------------------
def test_play_game_records_decisions_and_finite_returns():
    net = PolicyNet(hidden=16, seed=3)
    decisions: list[Decision] = []
    agents = [
        NeuralAgent(net, seed=index, recorder=decisions) for index in range(3)
    ]
    result = play_game(agents, seed=4, decisions=decisions)
    assert decisions
    compute_returns(decisions)
    assert all(np.isfinite(decision.ret) for decision in decisions)
    for decision in decisions:
        primitive, _ = split_action(decision.action)
        assert primitive in decision.ctx.legal_actions
    assert all(score >= -1 for score in result.scores)


def test_neural_agent_never_selects_an_illegal_action():
    net = PolicyNet(hidden=16, seed=1)
    env = MineEnv(opponents=2)
    env.reset(seed=9)
    for seat in range(env.n_players):
        agent = NeuralAgent(net, seed=seat, epsilon=1.0)
        for _ in range(10):
            ctx = env.player_action_context(seat)
            choice = agent.decide(ctx)
            primitive, announces = split_action(choice.action)
            assert primitive in ctx.legal_actions
            assert not announces or choice.announce_blast


def test_policy_net_save_load_roundtrip(tmp_path):
    net = PolicyNet(hidden=16, value_hidden=8, seed=2, name="roundtrip")
    net.trained_generations = 4
    path = net.save(tmp_path / "net.npz")
    loaded = PolicyNet.load(path)
    assert loaded.hidden == 16
    assert loaded.name == "roundtrip"
    assert loaded.trained_generations == 4
    for key in net.params:
        assert np.allclose(net.params[key], loaded.params[key])


def test_policy_net_can_learn_to_prefer_one_action():
    net = PolicyNet(hidden=16, seed=3, lr=2e-3)
    env = MineEnv(opponents=2)
    env.reset(seed=7)
    decisions: list[Decision] = []
    agents = [
        NeuralAgent(net, seed=index, recorder=decisions) for index in range(3)
    ]
    play_game(agents, seed=8, decisions=decisions)
    assert decisions

    target = int(decisions[0].ctx.legal_actions[0])
    fixed = [
        Decision(
            player=0,
            ctx=decisions[index % len(decisions)].ctx,
            state=state_features(decisions[index % len(decisions)].ctx),
            action=target,
            reward=1.0,
        )
        for index in range(48)
    ]
    ctx = decisions[0].ctx
    mask = action_mask(ctx)
    matrix = state_action_matrix(ctx)
    before = float(net.masked_probs(matrix, mask)[target])
    for _ in range(80):
        net.fit(fixed, epochs=1, batch_size=24, entropy_coef=0.0)
    after = float(net.masked_probs(matrix, mask)[target])
    # The head scores every action of the grid, so an absolute probability
    # margin shrinks with the size of the grid (106 actions now). Assert a
    # relative gain over the uniform-over-legal starting point instead.
    assert after > before * 1.25
    assert after > 1.0 / mask.sum()


def test_lorry_standoff_leaves_the_haul_and_campers_keep_their_loot():
    """The haul is over, so the players still inside leave and take part in scoring."""
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.players[0].escaped = True
    env.players[0].collected_cards = [(GEM, 0), (GEM, 1)]
    env.players[1].position = LORRY
    env.players[1].collected_cards = [(GEM, 0), (GEM, 1), (GEM, 2)]
    env.players[2].position = LORRY
    env.players[2].collected_cards = [(GEM, 0)]

    info: dict = {"outcome": "continued"}
    _reward, terminated = env._finish_round(0.0, info)

    assert not terminated
    assert env.haul == 2
    record = env.haul_history[-1]
    assert record["outcome"] == "lorry"
    # Everybody left the mine, so everybody is scored: two gems beat one.
    assert record["escaped"] == [0, 1, 2]
    assert env.players[1].score == 3
    assert env.players[0].score == 0
    assert env.players[2].score == 0


def test_lorry_standoff_needs_everybody_left_inside_on_a_lorry():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.players[0].position = LORRY
    env.players[1].position = LORRY
    env.players[2].position = 1
    env.players[2].column = 2

    info: dict = {"outcome": "continued"}
    env._finish_round(0.0, info)
    assert info.get("outcome") != "lorry"
    assert env.haul == 1


def test_lorry_standoff_does_not_happen_when_someone_has_left():
    """An escapee is out of the mine, so the rest standing still is still a stall."""
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.players[0].escaped = True
    env.players[1].position = 1
    env.players[1].column = 1
    env.players[2].position = LORRY

    info: dict = {"outcome": "continued"}
    env._finish_round(0.0, info)
    assert info.get("outcome") != "lorry"


def test_pass_rule_variant_changes_the_pass_threshold():
    default = MineEnv(opponents=2)
    default.reset(seed=1)
    assert default._pass_threshold() == 2
    variant = MineEnv(opponents=2, passes="n")
    variant.reset(seed=1)
    assert variant._pass_threshold() == 3
    with pytest.raises(ValueError):
        MineEnv(opponents=2, passes="n+1")


def test_solo_escape_bonus_rule_variant():
    default = MineEnv(opponents=2)
    default.reset(seed=4)
    default.haul = 1
    default.players[0].escaped = True  # no gems
    default._score_haul({"outcome": "continued"})
    assert default.players[0].score == 3

    variant = MineEnv(opponents=2, solo_escape_bonus=False)
    variant.reset(seed=4)
    variant.haul = 1
    variant.players[0].escaped = True
    variant._score_haul({"outcome": "continued"})
    assert variant.players[0].score == 0


def test_blind_blast_choice_hides_the_drawn_fronts():
    open_book = MineEnv(opponents=2)
    open_book.reset(seed=3)
    open_book.players[0].blast_cards = [(GEM, 0), (DYNAMITE, 1), (GEM, 2)]
    open_book.players[0].blast_pending = True
    assert all(
        front != FRONT_UNKNOWN
        for front, _back in open_book.player_action_context(0).own_blast_cards
    )

    blind = MineEnv(opponents=2, blast_choice_sees_fronts=False)
    blind.reset(seed=3)
    blind.players[0].blast_cards = [(GEM, 0), (DYNAMITE, 1), (GEM, 2)]
    blind.players[0].blast_pending = True
    context = blind.player_action_context(0)
    assert all(front == FRONT_UNKNOWN for front, _back in context.own_blast_cards)
    assert [back for _front, back in context.own_blast_cards] == [0, 1, 2]


def test_rusher_no_longer_beats_the_rule_policy():
    """Standing on the lorry ends the haul, so the campers' loot still counts."""
    from agents import RusherAgent

    margins = []
    for seed in range(60):
        result = play_game([RusherAgent(), RuleAgent(), RuleAgent()], seed=seed)
        margins.append(
            result.scores[0] - float(np.mean(result.scores[1:]))
        )
        assert 0 in result.hauls[0].escaped
        assert result.hauls[0].gems[0] == 0

    # The empty-handed escape used to be worth +2.69 a game; it is now worth
    # nothing, because the campers leave with their loot when the haul ends.
    assert float(np.mean(margins)) < 1.0


def test_empty_escape_loses_the_bonus_to_a_camper_holding_gems():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.players[0].escaped = True  # left with nothing
    env.players[1].position = LORRY
    env.players[1].collected_cards = [(GEM, 0), (GEM, 1)]
    env.players[2].position = LORRY

    info: dict = {"outcome": "continued"}
    env._finish_round(0.0, info)

    assert env.haul_history[-1]["outcome"] == "lorry"
    assert env.players[0].score == 0
    assert env.players[1].score == 3


def test_rushing_beats_camping_but_not_collecting():
    """Leaving early still beats doing nothing; it loses to collecting and leaving."""
    from agents import LeaverAgent, RusherAgent

    campers, collecting = [], []
    for seed in range(40):
        result = play_game(
            [RusherAgent(), RuleAgent(), RuleAgent()], seed=seed
        )
        campers.append(result.scores[0])
        result = play_game(
            [RusherAgent(), LeaverAgent(1), LeaverAgent(1)], seed=seed
        )
        collecting.append(result.scores[0])

    assert float(np.mean(campers)) > 0.5
    assert float(np.mean(collecting)) < float(np.mean(campers))


def test_rusher_loses_its_edge_without_the_solo_bonus():
    from agents import RusherAgent

    def mean_score(bonus: bool) -> float:
        scores = [
            play_game(
                [RusherAgent(), RuleAgent(), RuleAgent()],
                seed=seed,
                solo_escape_bonus=bonus,
            ).scores[0]
            for seed in range(30)
        ]
        return float(np.mean(scores))

    assert mean_score(True) > mean_score(False) + 1.0
    assert mean_score(False) < 1.0


def test_leaver_passes_instead_of_walking_past_its_card():
    """Re-planning every turn would move the figure past the card it wants."""
    from agents import ActionChoice, LeaverAgent

    env = MineEnv(opponents=2)
    env.reset(seed=7)
    agent = LeaverAgent(target_gems=1)
    moved = 0
    for _ in range(24):
        if env.game_finished or env.haul > 1:
            break
        index = env.planning_player
        ctx = env.player_action_context(index)
        if not ctx.legal_actions:
            break
        choice = agent.decide(ctx) if index == 0 else ActionChoice(PASS)
        if index == 0 and choice.action < PIT_CAGE:
            moved += 1
        env.step_one_player(
            choice.action, announce_blast=choice.announce_blast, force_action=True
        )
    # One move per planning phase at most; the rest are passes.
    assert moved <= 3


def test_leaver_beats_two_rushers():
    """Collecting and leaving is the counter to leaving empty-handed."""
    from agents import LeaverAgent, RusherAgent

    scores, wins, gems = [], 0, []
    for seed in range(60):
        result = play_game([LeaverAgent(1), RusherAgent(), RusherAgent()], seed=seed)
        scores.append(result.scores[0])
        wins += result.winner == 0
        gems.extend(
            haul.gems[0] for haul in result.hauls if 0 in haul.escaped
        )
    assert wins / 60 > 0.75
    assert float(np.mean(scores)) > 5.0
    assert float(np.mean(gems)) > 0.3


def test_rusher_stops_winning_once_opponents_leave_with_loot():
    """The 'dominant strategy' only pays while the others stay in the mine."""
    from agents import LeaverAgent, RusherAgent

    wins = 0
    for seed in range(60):
        result = play_game([RusherAgent(), LeaverAgent(1), LeaverAgent(1)], seed=seed)
        wins += result.winner == 0
    # Below the 1-in-3 chance rate of a three-player game.
    assert wins / 60 < 0.40


def test_dying_with_a_blast_pending_still_finishes_the_haul():
    """A dead player must not hold the round open.

    Regression: a dragon that killed a player who had a blast pending left the
    flag set. Nothing could act, the round never finished, and the game was
    recorded as ``aborted`` with every score at zero. Four always-blasting
    ``rule`` players hit this in ~60% of games.
    """
    from agents import make_scripted_agent

    for seed in range(40):
        result = play_game(
            [make_scripted_agent("rule", seed=seed * 4 + i, blast="always") for i in range(4)],
            seed=seed,
        )
        assert not result.aborted, (seed, result.abort_reason)
        assert result.abort_reason == ""
        assert len(result.hauls) == 3


def test_pending_blast_from_a_dead_or_escaped_player_does_not_block_the_haul():
    """The round-finish guard ignores players who can no longer blast."""
    env = MineEnv(opponents=3)
    env.reset(seed=11)
    env.players[1].dead = True
    env.players[1].blast_pending = True
    env.players[1].blast_cards = [(0, 0), (0, 1), (0, 2)]
    env.players[2].escaped = True
    env.players[2].blast_pending = True
    env.players[2].blast_cards = [(0, 0), (0, 1), (0, 2)]
    assert not any(
        p.blast_pending and not p.escaped and not p.dead for p in env.players
    )


def test_a_tie_break_always_has_enough_dragons_to_finish():
    """The card economy guarantees the tie-break can end, so nothing is topped up.

    Each haul starts from a fresh 30-card deck with six dragons and a haul ends
    as soon as a dragon is drawn, so at most three are spent before the final
    tie-break, while eliminating ``contenders - 1`` players needs at most four.
    This checks the invariant on real games at both player counts instead of
    trusting the argument, and pins down that the guard never has to fire.
    """
    from agents import make_scripted_agent

    observed: list[tuple[int, int]] = []
    original = MineEnv._begin_tie_breaker

    def recording(env):
        started = original(env)
        if started:
            observed.append(
                (
                    len(env.tie_breaker_contenders) - 1,
                    env._dragons_left_to_draw(),
                )
            )
        return started

    MineEnv._begin_tie_breaker = recording
    try:
        for players, names in (
            (4, ("rule",) * 4),
            (5, ("rule", "rule", "cautious", "random", "leaver")),
        ):
            for seed in range(40):
                agents = [
                    make_scripted_agent(name, seed=seed * 13 + index)
                    for index, name in enumerate(names)
                ]
                result = play_game(agents, seed=seed)
                assert not result.aborted, (players, seed, result.abort_reason)
    finally:
        MineEnv._begin_tie_breaker = original

    assert observed, "no tie-break was exercised"
    assert all(have >= need for need, have in observed), observed


def test_tie_breaker_guard_tops_up_the_deck_if_the_invariant_is_broken():
    """Defensive only: an unreachable state must still settle instead of hanging.

    Constructed by hand, never seen in a game — see the test above. If a rules
    change ever leaves too few dragons, the tie-break must still terminate.
    """
    from game_env import GEM

    env = MineEnv(opponents=2)
    env.reset(seed=9)
    for player in env.players:
        player.score = 4
    env.cards_remaining = [(GEM, 0)] * 10
    env.rows = [[(GEM, 0)] * 5 for _ in range(2)]

    assert env._dragons_left_to_draw() == 0
    assert env._begin_tie_breaker() is True
    assert env.tie_breaker_shortage is True
    assert env._dragons_left_to_draw() >= len(env.tie_breaker_contenders) - 1


def test_threat_agent_retreats_when_a_blast_is_pending():
    """The reaction policy that the trained agent never learns (BALANCE.md §6)."""
    import dataclasses

    from agents import make_scripted_agent

    threat = make_scripted_agent("threat")
    calm = make_scripted_agent("rule")
    env = MineEnv(opponents=2)
    env.reset(seed=3)

    checked = False
    for _ in range(60):
        ctx = env.player_action_context(0)
        if PIT_CAGE in ctx.legal_actions and ctx.opponents and not env.game_finished:
            pending = tuple(
                dataclasses.replace(opponent, blast_pending=True)
                for opponent in ctx.opponents
            )
            threatened = dataclasses.replace(ctx, opponents=pending)
            assert threat.threatened(ctx) is False
            assert threat.threatened(threatened) is True
            # Calm: identical to the plain rule player.
            assert threat.act(ctx) == calm.act(ctx)
            # Threatened: go to the cage, whatever the rule policy wanted.
            assert int(threat.act(threatened)) == PIT_CAGE
            checked = True
            break
        env.step_one_player(PASS, force_action=True)
    assert checked, "no decision with a reachable pit cage was found"


def replaced(ctx, **changes):
    return dataclasses.replace(ctx, **changes)


def _trap_context(*, dragon_row: int | None, own_row: int = 5, opponents=()):
    """A 4-seat context with an own-column dragon at ``dragon_row``."""
    from game_env import GEM, PlayerActionContext

    rows = []
    for row in range(7):
        cells = []
        for column in range(4):
            front = GEM if column == 0 else None
            if dragon_row is not None and row == dragon_row and column == 0:
                front = DRAGON
            cells.append((front, 1) if front is not None else None)
        rows.append(tuple(cells))
    return PlayerActionContext(
        player_index=0,
        legal_actions=(own_row * 4 + 0,),
        public_rows=tuple(rows),
        private_cards=(),
        own_blast_cards=(),
        own_score=0,
        own_position=own_row,
        own_column=0,
        own_cage_index=None,
        own_blast_pending=False,
        opponents=tuple(opponents),
        used_blast_backsides=(),
        haul=0,
        round=0,
        starting_player=0,
        n_players=4,
        cards_remaining_in_deck=10,
        displacement_events=(),
        tie_breaker_contenders=(),
    )


def test_visible_dragon_ahead_only_counts_rows_still_ahead_of_the_figure():
    from agents import visible_dragon_ahead

    assert visible_dragon_ahead(_trap_context(dragon_row=1)) == (1, 1)
    # Below the figure you can never dig again, so it is not a "known ahead".
    assert visible_dragon_ahead(_trap_context(dragon_row=6)) is None
    assert visible_dragon_ahead(_trap_context(dragon_row=None)) is None


def test_trap_blast_mode_needs_a_dynamite_pair_and_a_known_dragon_ahead():
    from agents import should_blast

    pair = ((DYNAMITE, 1), (DYNAMITE, 1), (GEM, 1))
    assert not should_blast("trap", _trap_context(dragon_row=1))  # no dynamite pair
    assert should_blast("trap", replaced(_trap_context(dragon_row=1), private_cards=pair))
    # A dragon already behind the figure is not a reason to blast.
    assert not should_blast("trap", replaced(_trap_context(dragon_row=6), private_cards=pair))


def test_behind_blast_mode_needs_an_opponent_in_front():
    from agents import should_blast
    from game_env import OpponentPublicView

    armed = replaced(
        _trap_context(dragon_row=None),
        private_cards=((DYNAMITE, 1), (DYNAMITE, 1)),
    )
    assert not should_blast("behind", armed)
    rival = OpponentPublicView(
        player_index=1,
        position=4,
        column=1,
        cage_index=None,
        escaped=False,
        dead=False,
        score=3,
        collected_backsides=(),
        card_count=3,
        blast_pending=False,
    )
    assert should_blast("behind", replaced(armed, opponents=(rival,)))


def test_profile_reports_habits_of_a_checkpoint(tmp_path):
    """The behaviour profiler must run on a checkpoint and count real moves."""
    from profile_agent import profile_checkpoint

    net = PolicyNet(hidden=8, value_hidden=4, seed=0, name="tiny")
    path = tmp_path / "tiny.npz"
    net.save(path)

    row = profile_checkpoint(str(path), games=2, players=3, seed=1, opponents="rule")
    assert row["stats"].games == 2
    assert row["decisions"] > 0
    assert row["digs"] > 0
    assert 0.0 <= row["front_row_share"] <= 1.0
    assert 0.0 <= row["own_column_share"] <= 1.0
    assert row["blasts_fired"] >= 0.0


def test_a_face_up_dragon_in_the_own_column_is_recognised():
    from features import digs_face_up_dragon, has_dragon_free_move

    ctx = _trap_context(dragon_row=1)  # own column holds a dragon at row 1
    assert digs_face_up_dragon(ctx, 1 * MAX_COLUMNS + 0)  # column 0 = own
    assert not digs_face_up_dragon(ctx, 1 * MAX_COLUMNS + 1)  # another column
    assert not digs_face_up_dragon(ctx, 0 * MAX_COLUMNS + 0)  # own column, a gem
    assert not digs_face_up_dragon(ctx, int(PASS))
    assert has_dragon_free_move(ctx)


def _rival(player_index: int, position: int):
    from game_env import OpponentPublicView

    return OpponentPublicView(
        player_index=player_index,
        position=position,
        column=1,
        cage_index=None,
        escaped=False,
        dead=False,
        score=0,
        collected_backsides=(),
        card_count=1,
        blast_pending=False,
    )


def test_the_action_mask_keeps_a_visible_dragon_as_bait():
    """A rival deeper in the mine can push the bait into a cage, so it stays (BALANCE §4)."""
    from features import N_BASE_ACTIONS, action_mask, rival_can_push

    ctx = replaced(
        _trap_context(dragon_row=1, own_row=5, opponents=(_rival(1, 4),)),
        legal_actions=(1 * MAX_COLUMNS + 0, 4 * MAX_COLUMNS + 0),
        private_cards=((DYNAMITE, 1), (DYNAMITE, 1)),
    )
    assert rival_can_push(ctx, 1 * MAX_COLUMNS + 0)
    mask = action_mask(ctx)
    assert mask[1 * MAX_COLUMNS + 0], "the trap stays legal"
    assert mask[1 * MAX_COLUMNS + 0 + N_BASE_ACTIONS], "its blast twin stays legal"
    assert mask[4 * MAX_COLUMNS + 0]


def test_the_action_mask_hides_a_visible_dragon_nobody_can_punish():
    """With no rival deep enough, the digger eats the dragon itself: pure loss."""
    from features import action_mask, doomed_dragon_digs, rival_can_push

    ctx = replaced(
        _trap_context(dragon_row=1, own_row=5, opponents=(_rival(1, 0),)),
        legal_actions=(1 * MAX_COLUMNS + 0, 4 * MAX_COLUMNS + 0),
        private_cards=((DYNAMITE, 1), (DYNAMITE, 1)),
    )
    assert not rival_can_push(ctx, 1 * MAX_COLUMNS + 0), "a rival at row 0 cannot reach row 1"
    assert doomed_dragon_digs(ctx) == {1 * MAX_COLUMNS + 0}
    mask = action_mask(ctx)
    assert not mask[1 * MAX_COLUMNS + 0]
    assert mask[4 * MAX_COLUMNS + 0], "another move must stay available"


def test_the_action_mask_never_empties_itself():
    from features import action_mask

    ctx = replaced(
        _trap_context(dragon_row=1, own_row=5, opponents=(_rival(1, 0),)),
        legal_actions=(1 * MAX_COLUMNS + 0,),
    )
    assert action_mask(ctx)[1 * MAX_COLUMNS + 0], "the only legal move always stays"


def test_the_action_mask_stays_legal_when_every_move_is_a_dragon():
    from features import action_mask

    ctx = replaced(
        _trap_context(dragon_row=1, own_row=5),
        legal_actions=(1 * MAX_COLUMNS + 0,),
    )
    mask = action_mask(ctx)
    assert mask[1 * MAX_COLUMNS + 0], "the mask must never be empty"
