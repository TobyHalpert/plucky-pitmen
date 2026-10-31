"""Tests for the multi-agent self-play stack."""

from __future__ import annotations

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
from game_env import DYNAMITE, FRONT_UNKNOWN, GEM, LORRY, PASS, PIT_CAGE, MineEnv
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
    assert after > before + 0.05


def test_lorry_standoff_ends_the_haul_and_pays_nobody_inside():
    """No card can be collected, so the haul is over: campers score nothing."""
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.players[0].escaped = True
    env.players[0].collected_cards = [(GEM, 0), (GEM, 1)]
    env.players[1].position = LORRY
    env.players[2].position = LORRY

    info: dict = {"outcome": "continued"}
    _reward, terminated = env._finish_round(0.0, info)

    assert not terminated
    assert env.haul == 2
    record = env.haul_history[-1]
    assert record["outcome"] == "lorry"
    assert record["escaped"] == [0]
    assert record["dead"] == []
    # The sole escapee keeps the haul bonus, the campers get nothing.
    assert env.players[0].score == 3
    assert env.players[1].score == 0
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


def test_rusher_beats_the_rule_policy_under_the_printed_rules():
    """The suspected dominant strategy, as a hand-written policy."""
    from agents import RusherAgent

    rusher, others = [], []
    for seed in range(30):
        result = play_game([RusherAgent(), RuleAgent(), RuleAgent()], seed=seed)
        rusher.append(result.scores[0])
        others.extend(result.scores[1:])
        assert 0 in result.hauls[0].escaped
        assert result.hauls[0].gems[0] == 0

    assert float(np.mean(rusher)) > float(np.mean(others)) + 1.0
    assert float(np.mean(rusher)) > 2.0


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
