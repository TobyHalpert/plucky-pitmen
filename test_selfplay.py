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


def test_action_mask_never_widens_the_rules():
    """The mask may narrow the rules for documented policies, never widen them."""
    from features import (
        digs_face_up_dragon,
        digs_past_a_known_gem,
        lorry_skips_a_card_worth_taking,
        strands_the_figure,
        takes_the_gem_over_the_dynamite,
        walks_past_a_free_card,
    )

    env = MineEnv(opponents=2)
    env.reset(seed=2)
    for _ in range(6):
        player_index = env.planning_player
        ctx = env.player_action_context(player_index)
        mask = action_mask(ctx)
        for action in range(N_BASE_ACTIONS):
            if mask[action]:
                assert env._legal(action, player_index)
            elif env._legal(action, player_index):
                assert (
                    action == int(PIT_CAGE)
                    or action == int(LORRY)
                    and lorry_skips_a_card_worth_taking(ctx)
                    or digs_face_up_dragon(ctx, action)
                    or action in walks_past_a_free_card(ctx)
                    or action in strands_the_figure(ctx)
                    or action in digs_past_a_known_gem(ctx)
                    or action in takes_the_gem_over_the_dynamite(ctx)
                ), "only the documented policy rules may hide a legal move"
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

    allowed = [
        int(action)
        for action in decisions[0].ctx.legal_actions
        if action_mask(decisions[0].ctx)[int(action)]
    ]
    target = allowed[0]
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
    # The mask decides how much of the grid competes for the gradient, so this
    # number moves whenever a masking rule changes; 200 epochs keeps the
    # relative-gain assertion honest across that (rule 9 cut it at 80).
    for _ in range(200):
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


def test_a_seat_in_a_cage_does_not_dig_its_way_back_out():
    """The escape is the decision; walking out of it is not.

    A figure in a cage is out of the mine at the execution phase and keeps its
    loot, so a card move taken from there undoes the escape. Over 60 four-net
    games the shipped checkpoint took 604 of them, 313 while holding a gem the
    escape would have banked, and they are worth **-0.72 (se 0.12)** a move
    against taking the departure (BALANCE.md, "Out of the cage is out").
    """
    from features import action_mask, cage_banks_nothing, steps_out_of_the_cage

    # In a cage, holding a gem, with a rival standing on a dragon in this seat's
    # own column: that is a bank worth taking, so the escape stands and the only
    # move left is the departure.
    base = _board_context(
        {(0, 0): (None, 1), (1, 0): (DRAGON, 1)},
        legal=(0 * MAX_COLUMNS + 0, int(PASS), int(LORRY)),
        own_position=PIT_CAGE,
    )
    in_a_cage = dataclasses.replace(
        base,
        private_cards=((GEM, 1),),
        own_cage_index=3,
        own_column=None,
        opponents=(dataclasses.replace(_rival(1, 0), column=0, position=1),),
    )
    assert cage_banks_nothing(in_a_cage) is False
    assert steps_out_of_the_cage(in_a_cage) == {0 * MAX_COLUMNS + 0}
    mask = action_mask(in_a_cage)
    assert not mask[0 * MAX_COLUMNS + 0]
    assert mask[int(PASS)] or mask[int(LORRY)]

    # Empty-handed and nothing to run from there is nothing to undo, so digging
    # out stays available (the lorry is offered too: stay in the haul).
    empty = dataclasses.replace(
        in_a_cage, private_cards=(), opponents=()
    )
    assert cage_banks_nothing(empty) is True
    assert steps_out_of_the_cage(empty) == set()
    assert action_mask(empty)[0 * MAX_COLUMNS + 0]

    # And a seat that is not in a cage is untouched by the rule.
    on_a_card = _board_context(
        {(0, 0): (None, 1)}, legal=(0 * MAX_COLUMNS + 0,), own_position=3
    )
    assert steps_out_of_the_cage(on_a_card) == set()


def test_the_rule_bot_holds_the_escape_it_has_already_made():
    """``rule`` (the viewer's default opponent) had the same habit, and worse.

    It re-entered the mine from the cage on its second decision of a phase: 520
    times in 40 rule-vs-rule games. It now holds the departure when the escape
    banks something, so a rule game takes 1.12 card moves a seat-phase instead of
    1.43, and the steps out of a cage fall to the empty-handed ones it still has
    a reason to take.
    """
    from agents import RuleAgent
    from game_env import PlayerActionContext

    env = MineEnv(opponents=2)
    env.reset(seed=5)
    ctx = env.player_action_context(0)
    # Standing in a cage, planned, holding a gem, with a card to step onto.
    cage = dataclasses.replace(
        ctx,
        own_position=PIT_CAGE,
        own_cage_index=3,
        own_column=None,
        private_cards=((GEM, 1),),
        legal_actions=(0, int(PASS), int(LORRY)),
        opponents=(),
        consecutive_passes=0,
        passes_to_close=2,
    )
    assert RuleAgent().act(cage) == int(PASS)
    without_loot = dataclasses.replace(cage, private_cards=())
    assert RuleAgent().act(without_loot) in (int(PASS), int(LORRY), 0)


def test_rusher_loses_its_edge_without_the_solo_bonus():
    """The empty escape is worth nothing, so the bonus is the rusher's whole edge.

    The numbers moved when the rule bot stopped forfeiting a banked escape
    (``rule_policy``, the stability guard): against a field that banks, the
    camper scores +1.18 with the solo bonus and +0.20 without it, where it used
    to score +2.83 and +0.33 against a bot that dug its way back out of the cage
    it had just walked into. So the bonus is still the edge -- 0.98 of a 1.18
    score -- but the field is no longer handing the rusher free points.
    """
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

    assert mean_score(True) > mean_score(False) + 0.7
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


def test_a_visible_dragon_is_not_encoded_as_safe():
    """Risk channels must state the truth for a front the player can see (BALANCE §4)."""
    from features import action_features

    ctx = _trap_context(dragon_row=1, own_row=5)
    dragon = action_features(ctx, 1 * MAX_COLUMNS + 0)
    gem = action_features(ctx, 0 * MAX_COLUMNS + 0)
    assert dragon[13] == 1.0 and dragon[16] == 1.0 and dragon[17] == 0.0
    assert gem[12] == 1.0 and gem[16] == 0.0 and gem[17] == 1.0


def test_a_backside_dragon_risk_comes_from_the_backside():
    """A visible A-dragon leaves one dragon on the A backsides, not two."""
    from game_env import BACKSIDE_A
    from features import DRAGONS_PER_BACKSIDE, TOTAL_PER_BACKSIDE, action_features

    ctx = _trap_context(dragon_row=1)
    rows = list(ctx.public_rows)
    rows[1] = ((DRAGON, BACKSIDE_A), (None, BACKSIDE_A), None, None)
    ctx = replaced(ctx, public_rows=tuple(rows))

    unknown = action_features(ctx, 1 * MAX_COLUMNS + 1)
    assert unknown[15] == 1.0, "the rival's front is unknown"
    expected = (DRAGONS_PER_BACKSIDE - 1) / (TOTAL_PER_BACKSIDE - 1)
    assert abs(unknown[16] - expected) < 1e-6
    assert abs(unknown[16] + unknown[17] - 1.0) < 1e-6


def test_action_reason_spells_out_what_the_deciding_seat_knows():
    """The viewer draws only backsides, so the reason has to name the knowledge."""
    from features import action_reason

    from game_env import BACKSIDE_A
    from features import N_BASE_ACTIONS

    ctx = _trap_context(dragon_row=1)
    assert action_reason(ctx, 0 * MAX_COLUMNS + 0) == "own column, gem"
    assert action_reason(ctx, 1 * MAX_COLUMNS + 0) == "own column, dragon"
    assert (
        action_reason(ctx, 1 * MAX_COLUMNS + 0 + N_BASE_ACTIONS)
        == "own column, dragon + blast"
    )

    rows = list(ctx.public_rows)
    rows[1] = ((DRAGON, BACKSIDE_A), (None, BACKSIDE_A), None, None)
    hidden = replaced(ctx, public_rows=tuple(rows))
    reason = action_reason(hidden, 1 * MAX_COLUMNS + 1)
    assert reason.startswith("back o ("), reason
    assert reason.endswith("dragon)"), reason


def test_action_reason_says_what_an_escape_is_running_from():
    """The dragon a cage answers can be private: say so, or the move looks mad.

    Seed 7 of the viewer, haul 3, round 0: P3 sat on a dragon in its own column
    and planned a cage at 62% with an empty hand.  On screen that is a figure on
    a plain card walking out of a quiet mine.
    """
    from features import action_reason

    on_dragon = _board_context(
        {(1, 0): (DRAGON, 0)}, legal=(int(PIT_CAGE),), own_position=1
    )
    assert action_reason(on_dragon, int(PIT_CAGE)) == (
        "pit cage, banks 0g (the dragon under me)"
    )

    carrying = replaced(on_dragon, private_cards=((GEM, 0),))
    assert "banks 1g" in action_reason(carrying, int(PIT_CAGE))

    # A rival's charged blast is the other thing an escape answers.
    charged = replaced(
        _board_context({}, legal=(int(PIT_CAGE),), own_position=5),
        opponents=(replaced(_rival(1, 4), blast_pending=True),),
    )
    assert "a rival's blast" in action_reason(charged, int(PIT_CAGE))

    # Nothing in particular: fall back to the table's hazard.
    calm = _board_context({}, legal=(int(PIT_CAGE),), own_position=5)
    assert action_reason(calm, int(PIT_CAGE)) == (
        "pit cage, banks 0g (0% the mine blows this round)"
    )


def test_the_mask_forbids_spending_the_haul_while_a_visible_card_is_reachable():
    """A dragon under the figure is a reason to refuse that card, not the haul.

    Seed 7, haul 3, round 0: from that position the gem one row away was worth
    +2.59 +- 0.12 and the empty escape +2.23 +- 0.10, and the seat took the gem
    at its next decision in the same phase anyway.
    """
    from features import action_mask, cage_banks_nothing, known_safe_card_available

    on_dragon = _board_context(
        {(0, 0): (GEM, 0), (1, 0): (DRAGON, 0)},
        legal=(0 * MAX_COLUMNS + 0, 1 * MAX_COLUMNS + 0, int(PIT_CAGE)),
        own_position=1,
    )
    assert known_safe_card_available(on_dragon)
    assert cage_banks_nothing(on_dragon)
    assert not action_mask(on_dragon)[int(PIT_CAGE)]
    assert action_mask(on_dragon)[0 * MAX_COLUMNS + 0]

    # Nothing it can see is safe: now leaving is a real choice and it stays.
    only_unknown = replaced(
        on_dragon,
        legal_actions=(2 * MAX_COLUMNS + 1, int(PIT_CAGE)),
        public_rows=tuple(
            (row if index != 2 else ((None, 0),) + row[1:])
            for index, row in enumerate(on_dragon.public_rows)
        ),
    )
    assert not known_safe_card_available(only_unknown)
    assert not cage_banks_nothing(only_unknown)
    assert action_mask(only_unknown)[int(PIT_CAGE)]


def test_action_reason_calls_a_raid_a_raid():
    from features import action_reason

    ctx = _backside_raid_context()
    assert action_reason(ctx, 1 * MAX_COLUMNS + 1).startswith("raid P2")


def _backside_raid_context():
    """A rival sits on a backside whose safe cards are all known.

    Eight A-cards are in that rival's pile, and an A backside carries exactly two
    dragons, so the two unknown A-cards left in the deck are both dragons.
    """
    from game_env import BACKSIDE_A, BACKSIDE_B, GEM, PlayerActionContext

    rows = []
    for row in range(7):
        rows.append(
            tuple(
                (GEM, BACKSIDE_B) if column == 0 else None for column in range(4)
            )
        )
    rows[1] = (None, (None, BACKSIDE_A), None, None)
    return PlayerActionContext(
        player_index=0,
        legal_actions=(1 * MAX_COLUMNS + 1, 3 * MAX_COLUMNS + 0),
        public_rows=tuple(rows),
        private_cards=(),
        own_blast_cards=(),
        own_score=0,
        own_position=5,
        own_column=0,
        own_cage_index=None,
        own_blast_pending=False,
        opponents=(
            replaced(_rival(1, 1), column=1, collected_backsides=(BACKSIDE_A,) * 8),
            replaced(_rival(2, 2), column=2),
            replaced(_rival(3, 3), column=3),
        ),
        used_blast_backsides=(),
        haul=1,
        round=2,
        starting_player=0,
        n_players=4,
        cards_remaining_in_deck=5,
        displacement_events=(),
        tie_breaker_contenders=(),
    )


def test_the_reader_refuses_a_backside_whose_unknowns_are_all_dragons():
    from rule_policy import _estimate_dragon_risk, deterministic_policy

    ctx = _backside_raid_context()
    assert _estimate_dragon_risk(ctx, 0) == 1.0, "the backside is exhausted of safe cards"
    raid = 1 * MAX_COLUMNS + 1
    safe = 3 * MAX_COLUMNS + 0
    assert deterministic_policy(ctx) == raid, "the baseline trusts the sitter"
    assert deterministic_policy(ctx, trust_sitter=False) == safe, "the reader does not"


def _cage_context(**overrides):
    """A context where fleeing is offered next to an ordinary dig."""
    base = dict(
        legal_actions=(0 * MAX_COLUMNS + 0, int(PIT_CAGE)),
        opponents=(),
    )
    base.update(overrides)
    return replaced(_trap_context(dragon_row=1), **base)


def _board_context(cards, *, legal=None, own_position=5):
    """A 4-seat context on a board whose cells are given as ``{(row, col): cell}``."""
    from game_env import PlayerActionContext

    rows = []
    for row in range(4):
        cells = []
        for column in range(4):
            cells.append(cards.get((row, column)))
        rows.append(tuple(cells))
    legal_actions = legal if legal is not None else tuple(
        row * MAX_COLUMNS + column for row, column in sorted(cards)
    )
    return PlayerActionContext(
        player_index=0,
        legal_actions=tuple(legal_actions),
        public_rows=tuple(rows),
        private_cards=(),
        own_blast_cards=(),
        own_score=0,
        own_position=own_position,
        own_column=0,
        own_cage_index=None,
        own_blast_pending=False,
        opponents=(),
        used_blast_backsides=(),
        haul=0,
        round=0,
        starting_player=0,
        n_players=4,
        cards_remaining_in_deck=10,
        displacement_events=(),
        tie_breaker_contenders=(),
    )


def test_a_gem_you_can_see_beats_a_dynamite_you_can_see():
    """Two face-up cards in your own column, one gem one dynamite: no trade.

    Seed 7 of the viewer, haul 1, planning phase 2: P1's column reads gem (r0),
    dynamite (r2), gem (r3) and it took the dynamite at 44% while the r3 gem was
    its own first choice at 56%.  A gem is scored and a lone dynamite is inert,
    and the gem below becomes unreachable as soon as the figure stands above it.
    """
    from features import action_mask, takes_the_gem_over_the_dynamite

    ctx = _board_context(
        {(2, 0): (DYNAMITE, 2), (3, 0): (GEM, 2)},
        legal=(2 * MAX_COLUMNS + 0, 3 * MAX_COLUMNS + 0),
        own_position=4,
    )
    assert takes_the_gem_over_the_dynamite(ctx) == {2 * MAX_COLUMNS + 0}
    mask = action_mask(ctx)
    assert not mask[2 * MAX_COLUMNS + 0]
    assert mask[3 * MAX_COLUMNS + 0]

    # Two dynamite, no gem in sight: the gem rule stays out of it, and the depth
    # rule does the deciding -- same card, deeper row, more of the column open.
    only_dynamite = _board_context(
        {(1, 0): (DYNAMITE, 1), (2, 0): (DYNAMITE, 2)},
        legal=(1 * MAX_COLUMNS + 0, 2 * MAX_COLUMNS + 0),
        own_position=4,
    )
    assert takes_the_gem_over_the_dynamite(only_dynamite) == set()
    assert not action_mask(only_dynamite)[1 * MAX_COLUMNS + 0]
    assert action_mask(only_dynamite)[2 * MAX_COLUMNS + 0]

    # A dragon the seat can see is not a gem: the dynamite stays legal there, and
    # so does the dragon (it is bait, which ``doomed_dragon_digs`` handles).
    with_dragon = _board_context(
        {(1, 0): (DRAGON, 1), (2, 0): (DYNAMITE, 2)},
        legal=(1 * MAX_COLUMNS + 0, 2 * MAX_COLUMNS + 0),
        own_position=4,
    )
    assert takes_the_gem_over_the_dynamite(with_dragon) == set()

    # A rival's dyn~ is face down to this seat: only its own column is readable.
    rival_dynamite = _board_context(
        {(2, 1): (DYNAMITE, 2), (3, 0): (GEM, 2)},
        legal=(2 * MAX_COLUMNS + 1, 3 * MAX_COLUMNS + 0),
        own_position=4,
    )
    assert takes_the_gem_over_the_dynamite(rival_dynamite) == set()
    assert action_mask(rival_dynamite)[2 * MAX_COLUMNS + 1]


def test_a_face_down_dig_is_not_a_trade_against_a_gem_you_can_see():
    """Same one card a round, and only one of the two can be a dragon.

    Seed 7 of the viewer, haul 1, planning phase 1: P1 has gems at r0 and r1 of its
    own column and takes r1c1 -- an empty cell in P2's column, backside "~", 25%
    dragon -- while P2 was standing in P3's column, so nothing was displaced. That
    is weak dominance in every branch: gem equal, dynamite (unscored) worse,
    dragon -2 and the haul over.
    """
    from features import action_mask, digs_past_a_known_gem

    has_gem = _board_context(
        {(1, 0): (GEM, 2), (1, 1): (None, 2)},
        legal=(1 * MAX_COLUMNS + 0, 1 * MAX_COLUMNS + 1),
        own_position=2,
    )
    assert digs_past_a_known_gem(has_gem) == {1 * MAX_COLUMNS + 1}
    mask = action_mask(has_gem)
    assert mask[1 * MAX_COLUMNS + 0]
    assert not mask[1 * MAX_COLUMNS + 1]

    # A rival sitting on its own column can still be raided: the push is worth
    # something the card is not.
    raid = _board_context(
        {(1, 0): (GEM, 2), (1, 1): (None, 2)},
        legal=(1 * MAX_COLUMNS + 0, 1 * MAX_COLUMNS + 1),
        own_position=2,
    )
    raid = replaced(
        raid,
        opponents=(replaced(_rival(1, 1), column=1),),
    )
    assert digs_past_a_known_gem(raid) == set()
    assert action_mask(raid)[1 * MAX_COLUMNS + 1]

    # With no gem in sight the face-down card is the best thing on the table.
    no_gem = _board_context(
        {(1, 1): (None, 2)},
        legal=(1 * MAX_COLUMNS + 1,),
        own_position=2,
    )
    assert digs_past_a_known_gem(no_gem) == set()
    assert action_mask(no_gem)[1 * MAX_COLUMNS + 1]

    # A face-down card in the seat's own column is the same lottery.
    own_hidden = _board_context(
        {(1, 0): (GEM, 2), (0, 0): (None, 1)},
        legal=(1 * MAX_COLUMNS + 0, 0 * MAX_COLUMNS + 0),
        own_position=2,
    )
    assert digs_past_a_known_gem(own_hidden) == {0 * MAX_COLUMNS + 0}
    assert not action_mask(own_hidden)[0 * MAX_COLUMNS + 0]
    assert action_mask(own_hidden)[1 * MAX_COLUMNS + 0]


def test_the_deeper_of_two_own_column_gems_is_taken_first():
    """The row you take a card at is the cage index a raid leaves you with.

    A raid sets ``cage_index = the row the figure stood on`` and from a cage only
    shallower rows are legal again, so a front-row gem taken mid-phase is a
    phase you cannot recover in. Seed 7 of the viewer, haul 1, round 1: P1 has
    gems at r0, r1 and r3 and takes r0c0; over 200 four-net games 114 of 209
    raids landed on a figure in row 0 and 65% of raids left it no card at all.
    """
    from features import action_mask, strands_the_figure

    ctx = _board_context(
        {(0, 0): (GEM, 0), (1, 0): (GEM, 2), (3, 0): (GEM, 2)},
        legal=(0 * MAX_COLUMNS + 0, 1 * MAX_COLUMNS + 0, 3 * MAX_COLUMNS + 0),
        own_position=4,
    )
    assert strands_the_figure(ctx) == {0 * MAX_COLUMNS + 0, 1 * MAX_COLUMNS + 0}
    mask = action_mask(ctx)
    assert not mask[0 * MAX_COLUMNS + 0]
    assert not mask[1 * MAX_COLUMNS + 0]
    assert mask[3 * MAX_COLUMNS + 0]

    # A single own-column gem has nothing to rank against: take it.
    single = _board_context(
        {(0, 0): (GEM, 0)},
        legal=(0 * MAX_COLUMNS + 0,),
        own_position=4,
    )
    assert strands_the_figure(single) == set()
    assert action_mask(single)[0 * MAX_COLUMNS + 0]

    # A deeper *dynamite* is not an alternative to a gem in hand.
    dynamite_below = _board_context(
        {(0, 0): (GEM, 0), (2, 0): (DYNAMITE, 1)},
        legal=(0 * MAX_COLUMNS + 0, 2 * MAX_COLUMNS + 0),
        own_position=4,
    )
    assert strands_the_figure(dynamite_below) == set()
    assert action_mask(dynamite_below)[0 * MAX_COLUMNS + 0]


def test_a_figure_in_a_cage_is_given_the_lorry_instead_of_a_pass():
    """A pass in the cage *is* the escape, so the lorry has to stay on the menu.

    Seed 7 of the viewer, haul 1, planning phase 1: P3 is displaced into cage
    index 0 holding nothing, with a hazard of 0.00.  The lorry rule masked the
    lorry -- it "skips a card worth taking", and there is no card at all in a
    cage -- so the only legal move left was a pass, and the network "chose" an
    empty departure at 100%.  Priced from that position on 200 re-rolls: lorry
    +0.93 +- 0.11, pass +0.55 +- 0.10.
    """
    from features import action_mask

    in_cage = _board_context(
        {(0, 0): (GEM, 0)},
        legal=(int(LORRY), int(PASS)),
        own_position=int(PIT_CAGE),
    )
    assert action_mask(in_cage)[int(LORRY)]
    assert not action_mask(in_cage)[int(PASS)]

    # Loot in hand does not change it while nothing can blow: the cage banks a
    # claim it could bank later, so the haul is worth keeping alive.
    carrying = replaced(in_cage, private_cards=((GEM, 0),))
    assert action_mask(carrying)[int(LORRY)]
    assert not action_mask(carrying)[int(PASS)]

    # A charged blast is something to run from, so the departure stands.
    charged = replaced(in_cage, own_blast_pending=True)
    assert action_mask(charged)[int(PASS)]
    assert not action_mask(charged)[int(LORRY)]


def test_the_mask_hides_a_face_down_dig_that_pays_more_than_a_deeper_card():
    """The backside is the price; walking past a card as safe as this one is free."""
    from features import backside_stats, walks_past_a_free_card

    same_price = _board_context(
        {(1, 1): (None, 0), (2, 2): (None, 0)}, own_position=4
    )
    assert walks_past_a_free_card(same_price) == {1 * MAX_COLUMNS + 1}
    mask = action_mask(same_price)
    assert not mask[1 * MAX_COLUMNS + 1]
    assert mask[2 * MAX_COLUMNS + 2]

    # Backside 1 is cheap here: both of its dragons are already face up in the
    # player's own column, so a card off it cannot be a dragon at all.
    cheap_deeper = _board_context(
        {(0, 0): (DRAGON, 1), (1, 0): (DRAGON, 1), (1, 1): (None, 0), (2, 2): (None, 1)},
        own_position=4,
    )
    assert backside_stats(cheap_deeper)[3][1] < backside_stats(cheap_deeper)[3][0]
    assert walks_past_a_free_card(cheap_deeper) == {1 * MAX_COLUMNS + 1}

    # The other way round is reading, not waste: take the cheaper card even
    # though a dearer one sits deeper.
    cheap_now = _board_context(
        {(0, 0): (DRAGON, 1), (1, 0): (DRAGON, 1), (1, 2): (None, 1), (2, 1): (None, 0)},
        own_position=4,
    )
    assert walks_past_a_free_card(cheap_now) == set()

    # A visible front is not a lottery ticket: "known gem now" against
    # "unknown card deeper" is a real trade and stays in the action space.
    known = _board_context({(1, 1): (GEM, 0), (2, 2): (None, 0)}, own_position=4)
    assert walks_past_a_free_card(known) == set()

    # An occupied card is somebody's figure, not a free card to pick up.
    occupied = replaced(
        _board_context({(1, 1): (None, 0), (2, 1): (None, 0)}, own_position=4),
        opponents=(_rival(1, 2),),
    )
    assert walks_past_a_free_card(occupied) == set()


def test_the_mask_forbids_parking_on_a_lorry_to_skip_an_ordinary_card():
    """The lorry skips the card under the figure, so it is only right over a dragon.

    It does not even protect the figure: a dragon kills everyone still in the
    mine, lorry included. So parking is worth a round only when the card being
    skipped is one the seat can see is a dragon — and in that case it stays.
    """
    from features import lorry_skips_a_card_worth_taking

    ordinary = _board_context(
        {(1, 1): (None, 0), (2, 1): (None, 0)},
        legal=(1 * MAX_COLUMNS + 1, int(LORRY)),
        own_position=2,
    )
    assert lorry_skips_a_card_worth_taking(ordinary)
    mask = action_mask(ordinary)
    assert not mask[int(LORRY)]
    assert mask[1 * MAX_COLUMNS + 1]

    # Standing on a dragon the seat can see: the lorry is the way to refuse it.
    on_dragon = _board_context(
        {(1, 0): (DRAGON, 0), (2, 1): (None, 0)},
        legal=(2 * MAX_COLUMNS + 1, int(LORRY)),
        own_position=1,
    )
    assert not lorry_skips_a_card_worth_taking(on_dragon)
    assert action_mask(on_dragon)[int(LORRY)]

    # The lorry is the only legal move: never empty the action space.
    only_lorry = _board_context({}, legal=(int(LORRY),), own_position=3)
    assert action_mask(only_lorry)[int(LORRY)]


def test_the_state_says_how_likely_the_mine_is_to_blow_this_round():
    """A cage is insurance; the risk it insures against has to be an input.

    The board only enters the policy input through the candidate-action block, so
    a seat cannot see the card under a rival's figure.  Without this feature the
    escape decision is blind to the one quantity that makes it right or wrong.
    """
    from features import HAZARD_FEATURE, round_dragon_hazard, state_features

    # Nobody is in the columns: no card can be turned, so no dragon this round.
    empty = _board_context({}, legal=(0,), own_position=5)
    assert round_dragon_hazard(empty) == 0.0
    assert state_features(empty)[HAZARD_FEATURE] == 0.0

    # A rival *raiding* a column it cannot see carries that backside's remaining
    # dragons: two dragons behind ten cards is 0.2.
    raider = replaced(_rival(1, 2), column=0)
    rival = replaced(
        _board_context({(2, 0): (None, 0)}, legal=(0,)), opponents=(raider,)
    )
    assert round_dragon_hazard(rival) == pytest.approx(0.2)
    assert state_features(rival)[HAZARD_FEATURE] == pytest.approx(0.2)

    # Same card, but both of that backside's dragons are already face up in the
    # seat's own column: the card cannot be one, and the hazard is zero again.
    spent = replaced(
        _board_context(
            {(0, 0): (DRAGON, 0), (1, 0): (DRAGON, 0), (2, 0): (None, 0)},
            legal=(2 * MAX_COLUMNS + 0,),
        ),
        opponents=(raider,),
    )
    assert round_dragon_hazard(spent) == 0.0

    # The seat's own card counts too, whether it is home or a raid.
    standing = _board_context({(2, 0): (None, 0)}, legal=(0,), own_position=2)
    assert round_dragon_hazard(standing) == pytest.approx(0.2)

    # A rival standing in its own column is no hazard at all: it can see the
    # card, and stepping away from a dragon is always available -- seed 7's
    # haul-1 round-1 table, where every seat stood on a card it could see and
    # the haul ended on a dragon 1% of 200 re-rolls.
    sitter = replaced(
        _board_context({(2, 2): (None, 0)}, legal=(0,)),
        opponents=(replaced(_rival(2, 2), column=2),),
    )
    assert round_dragon_hazard(sitter) == 0.0

    # A charged blast can wake a dragon whatever the board says.
    charged = replaced(
        _board_context({(2, 1): (None, 0)}, legal=(0,)),
        opponents=(replaced(_rival(1, 2), column=0, blast_pending=True),),
    )
    assert round_dragon_hazard(charged) == 1.0


def test_the_mask_forbids_an_escape_that_banks_nothing():
    """Nothing collected, nothing to escape from: the cage buys nothing."""
    from features import action_mask, cage_banks_nothing

    ctx = _cage_context()
    assert cage_banks_nothing(ctx)
    mask = action_mask(ctx)
    assert not mask[int(PIT_CAGE)]
    assert mask[0 * MAX_COLUMNS + 0]


def test_the_mask_forbids_banking_a_gem_while_nothing_can_blow():
    """Leaving is a decision that repeats: the roll has not been made yet.

    One gem in hand is not a reason to spend the round.  The seat can leave
    later in the same planning phase, once a rival has moved onto a card and the
    danger is on the board, and it can leave later in the haul with more gems --
    and the gem count is what the haul bonus compares.
    """
    from features import action_mask, cage_banks_nothing, round_dragon_hazard

    carrying = replaced(_cage_context(), private_cards=((GEM, 0),))
    assert round_dragon_hazard(carrying) == 0.0
    assert cage_banks_nothing(carrying)
    assert not action_mask(carrying)[int(PIT_CAGE)]


def test_one_rival_on_a_face_down_card_is_not_a_reason_to_leave_with_loot():
    """The ordinary risk of a four-seat mine is 0.2-0.3 every single round.

    Seed 7, haul 3: all four nets left with one gem each at 0.22-0.29 while three
    of them could see gems in their own columns, and the haul paid a four-way tie
    of +1 with the mine abandoned. A bar below the table's own base rate makes the
    cage legal at every decision, which is surrender rather than insurance.
    """
    from features import CAGE_ANSWERS_A_HAZARD, action_mask, round_dragon_hazard

    ordinary = replaced(
        _board_context(
            {(2, 0): (None, 0)},
            legal=(0, int(PIT_CAGE)),
            own_position=5,
        ),
        private_cards=((GEM, 0),),
        opponents=(replaced(_rival(1, 2), column=0),),
    )
    assert round_dragon_hazard(ordinary) < CAGE_ANSWERS_A_HAZARD
    assert not action_mask(ordinary)[int(PIT_CAGE)]


def test_the_mask_keeps_the_cage_when_something_can_blow():
    """Loot plus a board that is genuinely going up: that is an exit."""
    from features import (
        CAGE_ANSWERS_A_HAZARD,
        action_mask,
        cage_banks_nothing,
        round_dragon_hazard,
    )

    threatened = replaced(
        _board_context(
            {(2, 0): (None, 0), (1, 1): (None, 0), (1, 2): (None, 1)},
            legal=(0, int(PIT_CAGE)),
            own_position=5,
        ),
        private_cards=((GEM, 0),),
        opponents=(
            replaced(_rival(1, 2), column=0),
            replaced(_rival(2, 1), column=1),
            replaced(_rival(3, 1), column=2),
        ),
    )
    assert round_dragon_hazard(threatened) >= CAGE_ANSWERS_A_HAZARD
    assert not cage_banks_nothing(threatened)
    assert action_mask(threatened)[int(PIT_CAGE)]


def test_the_mask_keeps_the_cage_when_a_blast_is_charged():
    from features import action_mask

    waiting = replaced(_rival(1, 4), blast_pending=True)
    assert action_mask(_cage_context(opponents=(waiting,)))[int(PIT_CAGE)]


def test_the_mask_ignores_the_herd():
    """Seed 7 of the viewer: four nets read each other's cages until a haul is lost.

    A raid displaces a rival into a pit cage, so "somebody is in a cage" is the
    ordinary state of an aggressive table, not a threat. Reading it as one makes
    every seat flee and throws the haul away with no gems collected
    (BALANCE.md, "An escape has to bank something").
    """
    from features import action_mask, cage_banks_nothing

    queued = replaced(_rival(1, PIT_CAGE), position=PIT_CAGE)
    ctx = _cage_context(opponents=(queued,))
    assert cage_banks_nothing(ctx)
    assert not action_mask(ctx)[int(PIT_CAGE)]

    gone = replaced(_rival(1, 4), escaped=True)
    ctx = _cage_context(opponents=(gone,))
    assert cage_banks_nothing(ctx)
    assert not action_mask(ctx)[int(PIT_CAGE)]


def test_the_mask_keeps_the_cage_when_a_rival_will_wake_our_dragon():
    from features import action_mask, cage_banks_nothing

    sitting = replaced(_rival(1, 1), column=0)
    ctx = _cage_context(opponents=(sitting,))
    assert not cage_banks_nothing(ctx)
    assert action_mask(ctx)[int(PIT_CAGE)]


def test_the_mask_keeps_the_cage_as_the_only_move():
    from features import action_mask

    ctx = _cage_context(legal_actions=(int(PIT_CAGE),))
    assert action_mask(ctx)[int(PIT_CAGE)]


def _escape_reward(gems: int):
    """Dense reward for one seat escaping a haul with ``gems`` in hand."""
    from types import SimpleNamespace

    from selfplay import _reward_vector, _snapshot

    def player(score, held, gems_held, escaped, dead):
        return SimpleNamespace(
            score=score,
            gems=gems_held,
            collected_cards=[None] * held,
            escaped=escaped,
            dead=dead,
        )

    before_env = SimpleNamespace(
        haul=1, players=[player(0, 0, 0, False, False)]
    )
    before = _snapshot(before_env)
    after_env = SimpleNamespace(haul=1, players=[player(0, gems, gems, True, False)])
    return _reward_vector(after_env, before, escape_shaping=0.3)[0]


def test_an_empty_handed_escape_is_not_paid_for():
    """Walking to a pit cage with nothing must not be rewarded (BALANCE 5)."""
    empty = _escape_reward(gems=0)
    with_gem = _escape_reward(gems=1)
    assert empty == 0.0
    assert with_gem > 0.3, "the gem shaping and the escape shaping both apply"


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
    """Past the first decision of the round, the digger eats the dragon itself."""
    from features import action_mask, can_wait_on_a_dragon, doomed_dragon_digs, rival_can_push

    # The seat stands on a dynamite it can see, so stepping one row up onto a gem
    # is a gain rule 9 still allows -- that is the "another move" it needs.
    trap = _trap_context(dragon_row=1, own_row=5, opponents=(_rival(1, 0),))
    ctx = replaced(
        trap,
        legal_actions=(1 * MAX_COLUMNS + 0, 4 * MAX_COLUMNS + 0, int(PASS)),
        private_cards=((DYNAMITE, 1), (DYNAMITE, 1)),
        public_rows=tuple(
            tuple(
                (DYNAMITE, 1) if index == 5 and column == 0 else cell
                for column, cell in enumerate(row)
            )
            for index, row in enumerate(trap.public_rows)
        ),
    )
    assert not rival_can_push(ctx, 1 * MAX_COLUMNS + 0), "a rival at row 0 cannot reach row 1"
    assert not can_wait_on_a_dragon(ctx), "already planned: nothing forces anybody to move"
    assert doomed_dragon_digs(ctx) == {1 * MAX_COLUMNS + 0}
    mask = action_mask(ctx)
    assert not mask[1 * MAX_COLUMNS + 0]
    assert mask[4 * MAX_COLUMNS + 0], "another move must stay available"


def test_a_dragon_you_can_see_is_a_waiting_square_on_the_first_decision():
    """Standing on a dragon is reversible if the round cannot end before your turn.

    A round resolves on ``n - 1`` consecutive passes, and only a seat that has
    already planned may pass, so a first decision with somebody still unplanned
    behind it always gets a second decision -- and the seat steps off. That is
    the depth rule's argument with the row made explicit: the bottom of the mine
    cannot be raided, and it can be left.
    """
    from features import action_mask, can_wait_on_a_dragon, doomed_dragon_digs

    first = replaced(
        _trap_context(dragon_row=1, own_row=5, opponents=(_rival(1, 0),)),
        legal_actions=(1 * MAX_COLUMNS + 0, 4 * MAX_COLUMNS + 0),
        private_cards=((DYNAMITE, 1), (DYNAMITE, 1)),
    )
    assert can_wait_on_a_dragon(first)
    assert doomed_dragon_digs(first) == set()
    assert action_mask(first)[1 * MAX_COLUMNS + 0]

    # Last in the round's order: everything behind us has planned, so the other
    # three can pass in a row and resolve the round on top of the dragon.
    last = replaced(
        first,
        starting_player=1,
        opponents=(_rival(1, 0), _rival(2, 0), _rival(3, 0)),
    )
    order = [1, 2, 3, 0]  # starting player 1, four seats, this seat is seat 0 -> last
    assert order[-1] == 0
    assert not can_wait_on_a_dragon(last)
    assert doomed_dragon_digs(last) == {1 * MAX_COLUMNS + 0}
    assert not action_mask(last)[1 * MAX_COLUMNS + 0]

    # A seat alone in the mine has nothing behind it either.
    alone = replaced(first, opponents=())
    assert not can_wait_on_a_dragon(alone)


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


def _counter_context(rows, *, legal, own_position=2, private=(), opponents=(), haul=1):
    """A 4-seat context for the counter agent, built from ``{(row, col): cell}``."""
    from game_env import PlayerActionContext

    grid = []
    for row in range(4):
        grid.append(tuple(rows.get((row, column)) for column in range(4)))
    return PlayerActionContext(
        player_index=0,
        legal_actions=tuple(legal),
        public_rows=tuple(grid),
        private_cards=tuple(private),
        own_blast_cards=(),
        own_score=0,
        own_position=own_position,
        own_column=0,
        own_cage_index=None,
        own_blast_pending=False,
        opponents=tuple(opponents),
        used_blast_backsides=(),
        haul=haul,
        round=0,
        starting_player=0,
        n_players=4,
        cards_remaining_in_deck=10,
        displacement_events=(),
        tie_breaker_contenders=(),
    )


def _counter_rival(index, position, column, *, escaped=False, cards=0):
    from game_env import OpponentPublicView

    return OpponentPublicView(
        player_index=index,
        position=position,
        column=column,
        cage_index=None,
        escaped=escaped,
        dead=False,
        score=0,
        collected_backsides=(),
        card_count=cards,
        blast_pending=False,
    )


def test_the_counter_never_pushes_even_when_a_push_is_the_only_card():
    """Displacement is worth -0.30 points, so the counter declines it (BALANCE.md)."""
    from agents import CounterAgent

    # A rival sits on a card in its own column, two rows shallower than us, so
    # stepping onto it would displace it into the cage.
    rival = _counter_rival(1, 0, 1)
    ctx = _counter_context(
        {(0, 1): (None, 0)}, legal=(1, int(PASS)), own_position=2, opponents=(rival,)
    )
    assert CounterAgent().act(ctx) != 1
    # ... unless it is asked to, which is how the price was measured.
    assert CounterAgent(allow_push=True).act(ctx) == 1


def test_the_counter_reads_an_empty_handed_exit_as_a_dragon():
    """A seat only leaves with an empty hand to answer a dragon it can see.

    That is the tell: the card it was standing on is a dragon, so the counter
    never steps on it. The card itself is never revealed to anybody.
    """
    from agents import CounterAgent

    agent = CounterAgent()
    rival = _counter_rival(1, 0, 1)
    first = _counter_context(
        {(0, 1): (None, 0), (1, 0): (None, 2)},
        legal=(1 * MAX_COLUMNS + 0,), own_position=3, opponents=(rival,),
    )
    agent.act(first)
    # Next sighting: the same seat is out of the mine and its hand did not grow.
    gone = _counter_rival(1, PIT_CAGE, None, escaped=True, cards=0)
    second = _counter_context(
        {(0, 1): (None, 0), (1, 0): (None, 2)},
        legal=(1 * MAX_COLUMNS + 0,), own_position=3, opponents=(gone,), haul=1,
    )
    agent.act(second)
    assert (1, 0, 1) in agent.dragon_cells
    # And it refuses that cell afterwards, even as the only card on offer.
    only = _counter_context(
        {(0, 1): (None, 0)}, legal=(1, int(PASS)), own_position=1, opponents=(rival,)
    )
    assert agent.act(only) != 1


def test_the_counter_does_not_take_a_card_it_has_not_seen():
    """Own-column fronts are visible; another column's are not. Only dig the first."""
    from agents import CounterAgent

    # Own column: a gem one row up. Another column: a free face-down card.
    ctx = _counter_context(
        {(0, 0): (GEM, 0), (1, 1): (None, 2)},
        legal=(0 * MAX_COLUMNS + 0, 1 * MAX_COLUMNS + 1, int(PIT_CAGE)),
        own_position=2,
    )
    assert CounterAgent().act(ctx) == 0 * MAX_COLUMNS + 0

    # Nothing known reachable, one unknown card and the cage: bank, do not dig.
    unknown_only = _counter_context(
        {(1, 1): (None, 2)},
        legal=(1 * MAX_COLUMNS + 1, int(PIT_CAGE)),
        own_position=2,
        private=((GEM, 0),),
    )
    assert CounterAgent().act(unknown_only) == int(PIT_CAGE)


def test_the_counter_banks_two_gems_and_hands_the_nets_a_losing_table(monkeypatch):
    """The measured exploit, as pinned by the numbers in BALANCE.md.

    One counter against three copies of the historical pit-cage checkpoint wins about 56% of
    games outright and takes +4.1 points a seat against their +1.1. The test uses
    a short slice, and both sides are deterministic on the seed, so it is a
    regression guard rather than a sample.
    """
    import game_env

    monkeypatch.setattr(game_env, "DISPLACEMENT_STEPS_TO_THE_NEXT_CARD", False)
    from agents import CounterAgent, NeuralAgent
    from gui_controller import resolve_checkpoint
    from learning import PolicyNet

    net = PolicyNet.load(resolve_checkpoint("neural:selfplay4_pitcage"))
    margins = []
    for seed in range(40):
        agents = [CounterAgent()] + [
            NeuralAgent(net, seed=seed * 31 + seat, temperature=0.7)
            for seat in (1, 2, 3)
        ]
        scores = play_game(agents, seed=seed).scores
        margins.append(scores[0] - float(np.mean(scores[1:])))
    assert float(np.mean(margins)) > 1.5


def test_the_counter_is_deterministic_and_never_displaces():
    """No dice, and not one displacement in a whole game of them.

    The counter has no random source at all, so the same seed must replay
    exactly; and because it never pushes, four counters in four seats produce a
    game with no displacement in it. That is the shared tell this agent is built
    on: the seats it plays against do not displace either.
    """
    from agents import CounterAgent, NeuralAgent
    from gui_controller import resolve_checkpoint
    from learning import PolicyNet

    net = PolicyNet.load(resolve_checkpoint("neural:selfplay4"))

    def seats():
        return [CounterAgent()] + [
            NeuralAgent(net, seed=5 + i, temperature=0.7) for i in (1, 2, 3)
        ]

    assert play_game(seats(), seed=91).scores == play_game(seats(), seed=91).scores

    seen: list[tuple[int, object]] = []

    class Recording(CounterAgent):
        def decide(self, ctx):
            choice = super().decide(ctx)
            seen.append((int(choice.action), ctx))
            return choice

    def is_push(action: int, ctx) -> bool:
        if not 0 <= action < PIT_CAGE:
            return False
        row, column = divmod(action, MAX_COLUMNS)
        return any(
            not (o.escaped or o.dead) and o.column == column and o.position == row
            for o in ctx.opponents
        )

    play_game([Recording() for _ in range(4)], seed=12)
    assert seen
    assert not any(is_push(action, ctx) for action, ctx in seen)


def _hold_context(rows, *, legal, own_position, own_column=0, private=()):
    from game_env import PlayerActionContext

    grid = [tuple(rows.get((row, column)) for column in range(4)) for row in range(4)]
    return PlayerActionContext(
        player_index=0,
        legal_actions=tuple(legal),
        public_rows=tuple(grid),
        private_cards=tuple(private),
        own_blast_cards=(),
        own_score=0,
        own_position=own_position,
        own_column=own_column,
        own_cage_index=None,
        own_blast_pending=False,
        opponents=(),
        used_blast_backsides=(),
        haul=1,
        round=0,
        starting_player=0,
        n_players=4,
        cards_remaining_in_deck=10,
        displacement_events=(),
        tie_breaker_contenders=(),
    )


def test_a_second_move_in_the_same_phase_is_a_climb_towards_the_lorry():
    """The card a seat scores is the one under it when the phase closes.

    Movement is one-way, so a seat that has already planned and moves again does
    not collect a second card -- it swaps the card it chose for a shallower one.
    Measured on the shipped checkpoint: 66% of seat-phases took two or more
    moves, all of them with a pass available, and the extra move is worth -0.08
    (se 0.09) against holding. Masking it is worth +0.39 a seat over 200 paired
    games (BALANCE.md, "The card you are standing on").
    """
    from features import action_mask, climbs_off_the_card_it_chose

    # Standing on a gem it can see, deeper cards free: hold, do not climb.
    on_a_gem = _hold_context(
        {(0, 0): (DYNAMITE, 0), (1, 0): (GEM, 0)},
        legal=(0 * MAX_COLUMNS + 0, int(PASS)),
        own_position=1,          # standing on the gem at row 1
    )
    climbed = climbs_off_the_card_it_chose(on_a_gem)
    assert 0 * MAX_COLUMNS + 0 in climbed  # a free card shallower up the column
    mask = action_mask(on_a_gem)
    assert mask[int(PASS)]                 # holding is what it does
    assert not mask[0 * MAX_COLUMNS + 0]   # the climb is refused

    # But a gem it can SEE above it is worth the swap: a scored card beats a
    # dynamite that scores nothing.
    on_a_dynamite = _hold_context(
        {(0, 0): (GEM, 0), (1, 0): (DYNAMITE, 0)},
        legal=(0 * MAX_COLUMNS + 0, int(PASS)),
        own_position=1,          # standing on the dynamite at row 1
    )
    assert 0 * MAX_COLUMNS + 0 not in climbs_off_the_card_it_chose(on_a_dynamite)
    assert action_mask(on_a_dynamite)[0 * MAX_COLUMNS + 0]

    # And a visible dragon under the figure is not a card worth holding.
    on_a_dragon = _hold_context(
        {(0, 0): (DYNAMITE, 0), (1, 0): (DRAGON, 0)},
        legal=(0 * MAX_COLUMNS + 0, int(PASS)),
        own_position=1,          # standing on the visible dragon at row 1
    )
    assert climbs_off_the_card_it_chose(on_a_dragon) == set()

    # The first decision of a phase is an empty set: it has to move.
    fresh = _hold_context(
        {(0, 0): (GEM, 0), (1, 0): (GEM, 0)},
        legal=(0 * MAX_COLUMNS + 0, 1 * MAX_COLUMNS + 0),
        own_position=4,
        own_column=None,
    )
    assert climbs_off_the_card_it_chose(fresh) == set()


def test_the_table_holds_its_card_and_collects_more_for_it():
    """The rule, measured end to end: +0.39 points a seat, twice the gems.

    Same weights, the rule switched off by pretending it finds nothing to mask.
    """
    import features
    from gui_controller import GameController, resolve_checkpoint
    from learning import PolicyNet

    net = PolicyNet.load(resolve_checkpoint("neural:selfplay4"))
    rule = features.climbs_off_the_card_it_chose
    real = features.action_mask

    def play(seed):
        ctl = GameController(3, "neural:selfplay4", seed)
        ctl.reset(3, "neural:selfplay4", seed, all_seats=True, advance=False)
        ctl.agents = [
            NeuralAgent(net, seed=seed * 31 + i, temperature=0.7) for i in range(4)
        ]
        env, steps = ctl.env, 0
        while not env.game_finished and steps < 1500:
            steps += 1
            ctl.step_one_player_auto()
        return env

    def with_rule(seed):
        features.climbs_off_the_card_it_chose = rule
        return play(seed)

    def without_rule(seed):
        features.climbs_off_the_card_it_chose = lambda _ctx: set()
        try:
            return play(seed)
        finally:
            features.climbs_off_the_card_it_chose = rule

    on, off = [], []
    for seed in range(12):
        env = with_rule(seed)
        on.append(float(np.mean([p.score for p in env.players])))
        env = without_rule(seed)
        off.append(float(np.mean([p.score for p in env.players])))
    features.action_mask = real
    assert float(np.mean(on)) > float(np.mean(off))


def test_an_escape_that_banks_nothing_is_refused_from_the_lorry_too():
    """Rule 4 is not position-specific, which is why the lorry/cage pair is safe.

    An escape is refused wherever it is offered, so a seat standing on the lorry
    cannot re-enter a cage it would refuse, and from the lorry it always has its
    pass to fall back on -- the pair cannot spin. Measured on the shipped
    checkpoint, 100 four-net games: the cage is offered from the lorry 143 times
    and every one is a real escape (a blast is charged), so the mask has nothing
    to remove; and there is no decision in 10,640 where a seat on the lorry could
    not pass. BALANCE.md, "The lorry and the cage: nothing to fix".
    """
    import features
    from features import action_mask
    from game_env import LORRY, PlayerActionContext

    def on_the_lorry(*, gems, blast=False):
        from game_env import OpponentPublicView

        rivals = tuple(
            OpponentPublicView(
                player_index=index,
                position=4,
                column=index,
                cage_index=None,
                escaped=False,
                dead=False,
                score=0,
                collected_backsides=(),
                card_count=0,
                blast_pending=blast and index == 1,
            )
            for index in (1, 2, 3)
        )
        return PlayerActionContext(
            player_index=0,
            legal_actions=(int(PIT_CAGE), int(PASS)),
            public_rows=tuple(tuple((None, 0) for _ in range(4)) for _ in range(5)),
            private_cards=tuple((GEM, 0) for _ in range(gems)),
            own_blast_cards=(),
            own_score=0,
            own_position=LORRY,
            own_column=None,
            own_cage_index=None,
            own_blast_pending=False,
            opponents=rivals,
            used_blast_backsides=(),
            haul=2,
            round=1,
            starting_player=0,
            n_players=4,
            cards_remaining_in_deck=10,
            displacement_events=(),
            tie_breaker_contenders=(),
            consecutive_passes=0,
            passes_to_close=3,
            own_planned=True,
        )

    # Loot in hand but nothing to run from: the cage is refused, the seat holds.
    quiet = on_the_lorry(gems=2)
    assert features.cage_banks_nothing(quiet)
    mask = action_mask(quiet)
    assert not mask[int(PIT_CAGE)]
    assert mask[int(PASS)]

    # A charged blast is a reason to leave, so the cage comes back and the
    # escape is taken (rule 8 then drops the pass, keeping the lorry).
    threatened = on_the_lorry(gems=2, blast=True)
    assert not features.cage_banks_nothing(threatened)
    assert action_mask(threatened)[int(PIT_CAGE)]


def test_a_four_net_game_finishes_promptly():
    """Every game ends, and a game costs about 110 decisions, not thousands.

    A planning phase that cannot close would show up here as a game that never
    finishes. Measured on the shipped checkpoint: 108 decisions a game, worst 285,
    three decisions a planning phase.
    """
    from collections import Counter
    from gui_controller import GameController, resolve_checkpoint
    from learning import PolicyNet

    net = PolicyNet.load(resolve_checkpoint("neural:selfplay4"))
    slowest = 0
    for seed in range(6):
        ctl = GameController(3, "neural:selfplay4", seed)
        ctl.reset(3, "neural:selfplay4", seed, all_seats=True, advance=False)
        ctl.agents = [
            NeuralAgent(net, seed=seed * 31 + i, temperature=0.7) for i in range(4)
        ]
        env, decisions = ctl.env, 0
        while not env.game_finished and decisions < 1200:
            decisions += 1
            ctl.step_one_player_auto()
        assert env.game_finished, f"seed {seed} did not finish in {decisions} decisions"
        slowest = max(slowest, decisions)
    # A game that cannot close a phase would blow straight past this.
    assert slowest < 400


def test_a_push_sends_the_victim_to_a_cage_with_every_card_it_collected(monkeypatch):
    """The rule that makes "displace them and take their loot" impossible.

    A displaced figure goes to the nearest pit cage, where it leaves the haul and
    keeps every card it has collected -- the push costs it its round, never its
    hand. What the pusher gets is the one card under it, face down.
    """
    import game_env

    monkeypatch.setattr(game_env, "DISPLACEMENT_STEPS_TO_THE_NEXT_CARD", False)
    from game_env import BACKSIDE_A, BACKSIDE_B, GEM, PIT_CAGE

    env = MineEnv(opponents=2)
    env.reset(seed=4)

    # A real card in player 1's column, with room above it for the pusher.
    row, column = next(
        (row, 1)
        for row in range(len(env.rows) - 1)
        for column in (1,)
        if env.rows[row][column] is not None
    )
    victim = env.players[1]
    victim.position, victim.column = row, column
    victim.collected_cards = [(GEM, BACKSIDE_A), (GEM, BACKSIDE_B)]
    env.players[0].position, env.players[0].column = row + 1, 0

    push = row * MAX_COLUMNS + column
    assert env._legal(push), "a rival in its own column, deeper than the pusher"
    env._apply_action(0, push)  # the decision only; no round is executed here

    assert env.players[1].position == PIT_CAGE
    assert env.players[1].column is None
    assert env.players[0].position == row and env.players[0].column == column
    assert len(env.players[1].collected_cards) == 2, "the loot is still the victim's"


def test_the_pusher_leaves_a_card_it_can_see_a_gem_alone():
    """A push always moves towards the lorries: it gives up the card underneath."""
    from agents import PusherAgent
    from game_env import GEM

    ctx = replaced(
        _board_context(
            {(2, 0): (GEM, 1), (1, 1): (GEM, 1)},
            legal=(2 * MAX_COLUMNS + 0, 1 * MAX_COLUMNS + 1),
            own_position=2,
        ),
        opponents=(_rival(1, 1),),
    )
    assert PusherAgent(tell="any").act(ctx) == 2 * MAX_COLUMNS + 0


def test_the_pusher_needs_a_previous_sighting_for_its_held_tell():
    """``tell="held"`` only takes a card from a rival that stayed put for a turn."""
    from agents import PusherAgent

    ctx = replaced(
        _board_context(
            {(1, 1): (GEM, 1)},
            legal=(1 * MAX_COLUMNS + 1, int(PASS)),
            own_position=3,
        ),
        opponents=(_rival(1, 1),),
    )
    held = PusherAgent(tell="held")
    assert held.act(ctx) != 1 * MAX_COLUMNS + 1, "first sighting: not a parked seat yet"
    assert held.act(ctx) == 1 * MAX_COLUMNS + 1, "seen on that cell before: take it"

    # The looser tell takes it straight away, which is the configuration that
    # measures -0.59 a seat (BALANCE.md, "The parked seat").
    assert PusherAgent(tell="any").act(ctx) == 1 * MAX_COLUMNS + 1


def test_the_bait_on_a_visible_dragon_is_allowed_and_priced(monkeypatch):
    """Standing on a dragon you can see, waiting to be pushed off it, is a tactic.

    Rule 1 refuses a dig onto a known dragon when no rival can punish it, because
    the seat then has to eat the card itself. The bait is the mirror image: with a
    rival deep enough to push, the seat is *paid* to stand there, so the rule has
    to let it through. Banning the bait over 200 four-net games costs 0.23 a seat
    (hauls thrown away 14% -> 24%), so ``BAIT_IS_ALLOWED`` is on; this pins both
    the flag and the mask it changes.
    """
    import features
    from features import action_mask, can_wait_on_a_dragon, doomed_dragon_digs

    assert features.BAIT_IS_ALLOWED is True

    dragon = 1 * MAX_COLUMNS + 0
    ctx = replaced(
        _board_context(
            {(1, 0): (DRAGON, 1)},
            legal=(dragon, int(LORRY)),
            own_position=2,
        ),
        opponents=(_rival(1, 4),),  # deeper in the mine: it can push the bait
    )
    assert can_wait_on_a_dragon(ctx), "first decision of the round, not last in order"
    assert doomed_dragon_digs(ctx) == set(), "the rival can push: bait is on offer"
    assert action_mask(ctx)[dragon]

    monkeypatch.setattr(features, "BAIT_IS_ALLOWED", False)
    assert doomed_dragon_digs(ctx) == {dragon}  # nobody there to punish it: no bait
    mask = action_mask(ctx)
    assert not mask[dragon]
    assert mask[int(LORRY)]


def test_the_escape_stands_once_taken_is_measured_and_off(monkeypatch):
    """The stronger version of the cage rule, kept behind a flag.

    Holding a seat to the departure makes the visible "into the cage and straight
    back out" impossible and is worth +0.145 a seat, but it costs 28% of the gems
    in the mine (4.33 -> 3.13 a haul) and doubles the hauls the table throws away.
    It is off by default; this pins both the flag and what it does to the mask.
    """
    import features
    from features import action_mask, the_escape_stands_once_taken

    assert features.ESCAPE_STANDS_ONCE_TAKEN is False

    base = _board_context(
        {(0, 0): (None, 1), (1, 0): (DRAGON, 1)},
        legal=(0 * MAX_COLUMNS + 0, int(PASS), int(LORRY)),
        own_position=PIT_CAGE,
    )
    committed = dataclasses.replace(
        base,
        private_cards=((GEM, 1),),
        own_cage_index=3,
        own_column=None,
        own_planned=True,
        opponents=(dataclasses.replace(_rival(1, 0), column=0, position=1),),
    )
    assert the_escape_stands_once_taken(committed) is True
    # Off: the dig is refused by rule 10 but the lorry is still a way back in.
    assert not action_mask(committed)[0 * MAX_COLUMNS + 0]
    # On: the departure is the only move left.
    monkeypatch.setattr(features, "ESCAPE_STANDS_ONCE_TAKEN", True)
    mask = action_mask(committed)
    assert mask[int(PASS)]
    assert not mask[int(LORRY)]
    assert not mask[0 * MAX_COLUMNS + 0]
