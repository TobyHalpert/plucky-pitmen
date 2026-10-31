from collections import Counter

from game_env import (
    BACKSIDE_A,
    BACKSIDE_B,
    BACKSIDE_C,
    BACKSIDE_UNKNOWN,
    DRAGON,
    DYNAMITE,
    GEM,
    DECK_SIZE,
    LORRY,
    MAX_COLUMNS,
    MAX_ROWS,
    MineEnv,
    PASS,
    PIT_CAGE,
)

# The observation is 15 state values, then one backside per mine cell, then one
# front per cell. The mine block is sized by the deepest board a deck can build.
CELLS = MAX_ROWS * MAX_COLUMNS
FRONT_OFFSET = 15 + CELLS


def test_random_game_terminates():
    env = MineEnv(opponents=2)
    observation, _ = env.reset(seed=3)
    env.starting_player = 0
    assert observation.shape == env.observation_space.shape
    for _ in range(300):
        if env._game_over():
            break
        legal = env.action_mask()
        assert legal.any()
        action = PASS if legal[PASS] else int(legal.argmax())
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    assert terminated or truncated


def test_tie_breaker_uses_turn_order_and_eliminates_dragon_drawers():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.starting_player = 1
    env.haul = 3
    env.players[0].score = 5
    env.players[1].score = 5
    env.players[2].score = 1
    env.cards_remaining = [(GEM, BACKSIDE_A)]
    env.rows[0][0] = (DRAGON, BACKSIDE_B)

    assert env._begin_tie_breaker() is True
    assert env.tie_breaker_contenders == [0, 1]
    assert env.tie_breaker_turn == 1

    first = env.resolve_tie_breaker("deck")
    assert first["player"] == 1
    assert first["outcome"] == "safe"
    assert env.players[1].collected_cards[-1] == (GEM, BACKSIDE_A)

    second = env.resolve_tie_breaker("mine", 0, 0)
    assert second["player"] == 0
    assert second["outcome"] == "dragon"
    assert second["winner"] == 1
    assert second["game_over"] is True


def test_seeded_reset_is_reproducible():
    first = MineEnv().reset(seed=12)[0]
    second = MineEnv().reset(seed=12)[0]
    assert (first == second).all()


def test_new_game_randomizes_starter_and_execution_rotates_starter():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    first_starter = env.starting_player
    assert 0 <= first_starter < env.n_players

    env.players[1].dead = True
    env.players[2].dead = True
    env.step(LORRY)
    env.step(PASS)
    assert env.starting_player == (first_starter + 1) % env.n_players
    assert env.haul == 2

    env.reset(seed=4)
    assert env.starting_player == first_starter
    assert env.players[0].score == 0


def test_opponents_before_p1_can_move_before_p1s_first_turn():
    env = MineEnv(opponents=3)
    env.reset(seed=2)
    assert env.starting_player == 3
    assert env.planning_player == 3

    while env.planning_player != 0:
        env.step_one_player()

    assert env.players[3].position < len(env.rows)
    assert env.players[0].position == len(env.rows)


def test_passing_is_not_allowed_on_first_planning_turn():
    env = MineEnv(opponents=2)
    env.reset(seed=4)

    assert not env._legal(PASS)
    env.step(0)
    assert env._legal(PASS)


def test_row_zero_allows_pit_cage_lorry_and_pass_choices():
    env = MineEnv(opponents=2)
    env.reset(seed=4)

    env.step(0)

    legal = env.action_mask()
    assert legal[PIT_CAGE]
    assert legal[LORRY]
    assert legal[PASS]

    env.step(PIT_CAGE)
    assert env.players[0].position == PIT_CAGE
    assert env.players[0].cage_index == 0


def test_p1_cannot_pass_when_an_opponent_started():
    env = MineEnv(opponents=2)
    env.reset(seed=7)
    env.planning_player = 1
    env.players[1].position = LORRY
    env.step_one_player()

    env.planning_player = 0
    assert not env._legal(PASS)


def test_opponent_cannot_move_to_their_current_row():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.planning_player = 1
    env.players[1].position = 2

    legal = env.action_mask(1)

    assert not legal[2 * 5:3 * 5].any()
    assert legal[0 * 5:2 * 5].any()


def test_observation_exposes_backsides_but_not_fronts():
    env = MineEnv(opponents=2)
    observation, _ = env.reset(seed=12)

    visible_backsides = [
            observation[15 + row_index * MAX_COLUMNS + player_index]
        for row_index, row in enumerate(env.rows)
        for player_index, _card in enumerate(row)
    ]
    expected_backsides = [backside for row in env.rows for _front, backside in row]

    assert visible_backsides == expected_backsides
    assert all(value in (BACKSIDE_A, BACKSIDE_B, BACKSIDE_C) for value in visible_backsides)
    assert BACKSIDE_UNKNOWN in observation[[15 + row_index * MAX_COLUMNS + 4 for row_index in range(3)]]


def test_observation_reveals_only_p1_column_fronts():
    env = MineEnv(opponents=2)
    observation, _ = env.reset(seed=12)

    own_fronts = [
        observation[FRONT_OFFSET + row_index * MAX_COLUMNS]
        for row_index in range(3)
    ]
    expected_fronts = [env.rows[row_index][0][0] + 4 for row_index in range(3)]

    assert own_fronts == expected_fronts
    assert observation[15] == env.rows[0][0][1]
    assert all(
        observation[FRONT_OFFSET + row_index * MAX_COLUMNS + column] == BACKSIDE_UNKNOWN
        for row_index in range(3)
        for column in range(1, 3)
    )


def test_player_view_masks_hidden_card_fronts_but_keeps_private_cards_visible():
    env = MineEnv(opponents=2)
    env.reset(seed=12)
    env.rows[0][2] = (DRAGON, BACKSIDE_C)
    env.players[1].collected_cards = [(GEM, BACKSIDE_A)]

    view = env.player_view(1)

    assert view["rows"][0][2] == (None, BACKSIDE_C)
    assert view["private_cards"] == [(GEM, BACKSIDE_A)]
    assert env.rows[0][2] == (DRAGON, BACKSIDE_C)


def test_player_action_context_hides_raw_state_and_exposes_only_public_info():
    env = MineEnv(opponents=2)
    env.reset(seed=12)
    env.rows[0][2] = (DRAGON, BACKSIDE_C)
    env.players[1].collected_cards = [(GEM, BACKSIDE_A)]

    context = env.player_action_context(1)

    assert context.player_index == 1
    assert context.public_rows[0][2] == (None, BACKSIDE_C)
    assert context.private_cards == ((GEM, BACKSIDE_A),)
    assert env.rows[0][2] == (DRAGON, BACKSIDE_C)
    assert isinstance(context.legal_actions, tuple)
    assert context.legal_actions == tuple(int(action) for action, allowed in enumerate(env.action_mask()) if allowed)


def test_players_can_choose_any_free_card_not_only_their_column():
    env = MineEnv(opponents=2)
    env.reset(seed=4)

    other_column_card = 0 * 5 + 2
    assert env._legal(other_column_card)
    env.step(other_column_card)

    assert env.players[0].position == 0
    assert env.players[0].column == 2
    assert not env._legal(other_column_card)


def test_cards_are_collected_only_after_consecutive_passes():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.starting_player = 0
    env.rows[0][2] = (DYNAMITE, BACKSIDE_A)

    env.step(2)
    assert env.players[0].dynamite == 0

    env.step(PASS)
    assert env.players[0].dynamite == 1


def test_blast_choice_draw_and_resolution_are_limited_to_three_cards():
    env = MineEnv(opponents=2)
    env.reset(seed=5)
    env.planning_player = 0
    env.players[0].collected_cards = [(DYNAMITE, BACKSIDE_A), (DYNAMITE, BACKSIDE_B)]
    env.rows[0][0] = (GEM, BACKSIDE_A)
    env.players[0].position = 0
    env.players[0].column = 0
    env.cards_remaining = [
        (GEM, BACKSIDE_A),
        (DYNAMITE, BACKSIDE_B),
        (DRAGON, BACKSIDE_C),
        (GEM, BACKSIDE_C),
        (DYNAMITE, BACKSIDE_A),
        (GEM, BACKSIDE_B),
    ]

    env.begin_blast(0)

    assert env.players[0].blast_pending is True
    assert len(env.cards_remaining) == 6

    env._execute_round()

    assert len(env.players[0].blast_cards) == 3
    assert env.used_blast_backsides == [BACKSIDE_A, BACKSIDE_B]
    assert [card[1] for card in env.players[0].blast_cards] == [BACKSIDE_A, BACKSIDE_B, BACKSIDE_C]
    assert len(env.cards_remaining) == 3

    chosen = env.resolve_blast(0, 1)

    assert chosen[1] == BACKSIDE_B
    assert env.players[0].blast_pending is True
    second = env.resolve_blast(0, 0)
    assert env.players[0].blast_pending is False
    assert env.players[0].collected_cards[-2:] == [chosen, second]


def test_blast_announcement_survives_a_turn_that_reaches_execution():
    env = MineEnv(opponents=2)
    env.reset(seed=5)
    env.planning_player = 0
    env.players[0].collected_cards = [(DYNAMITE, BACKSIDE_A), (DYNAMITE, BACKSIDE_B)]
    env.rows[0][0] = (GEM, BACKSIDE_A)
    env.players[0].position = 2
    env.players[0].column = 0
    env.cards_remaining = [
        (GEM, BACKSIDE_A),
        (DYNAMITE, BACKSIDE_B),
        (GEM, BACKSIDE_C),
        (DYNAMITE, BACKSIDE_A),
        (GEM, BACKSIDE_B),
        (DYNAMITE, BACKSIDE_C),
    ]

    env.step_one_player(0, announce_blast=True)
    env.planning_player = 0
    env.consecutive_passes = 2
    env.step_one_player(PASS)

    assert env.players[0].blast_pending is True
    assert len(env.players[0].blast_cards) == 3


def test_announced_blast_keeps_same_row_out_of_next_turn_choices():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.players[0].collected_cards = [(DYNAMITE, BACKSIDE_A), (DYNAMITE, BACKSIDE_B)]
    env.planning_player = 0

    env.step_one_player(2 * 5)
    env.begin_blast(0)

    legal_cards = [
        action for action, allowed in enumerate(env.action_mask())
        if allowed and action < PIT_CAGE
    ]

    assert legal_cards
    assert all(action // 5 < 2 for action in legal_cards)


def test_opponent_passes_complete_the_pass_sequence():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.starting_player = 0
    env.rows[0][2] = (DYNAMITE, BACKSIDE_A)

    env.step(2)
    assert env.players[0].dynamite == 0
    _observation, reward, terminated, truncated, info = env.step(PASS)

    assert env.players[0].dynamite == 1
    assert info["outcome"] in {"continued", "escaped", "dragon"}
    assert reward != 0.0 or terminated or truncated


def test_woken_dragon_kills_everyone_still_in_the_mine():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.rows[0][0] = (DRAGON, BACKSIDE_A)
    for column in (1, 2):
        env.rows[0][column] = (GEM, BACKSIDE_B)
    for player_index, player in enumerate(env.players):
        player.position = 0
        player.column = player_index

    _reward, info = env._execute_round()

    assert info["outcome"] == "dragon"
    assert all(player.dead for player in env.players)


def test_player_can_displace_occupant_in_that_players_own_column():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.players[0].position = 2
    env.players[0].column = None
    env.players[1].position = 0
    env.players[1].column = 1

    target = 1
    assert env._legal(target)
    env._displace(0, 1, 0)

    assert env.players[1].position == PIT_CAGE
    assert env.players[1].column is None
    assert env.displacement_events == [(1, 0)]


def test_a_seat_can_count_how_often_a_rival_displaces_somebody():
    """Displacement is performed in the open, so it is public history.

    It is also the one readable signal about whether a rival takes bait: a seat
    that never displaces cannot be counted on to push a figure off a dragon it
    cannot see. The learner gets the count as a feature.
    """
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.players[0].position = 2
    env.players[0].column = None
    env.players[1].position = 0
    env.players[1].column = 1
    env._displace(0, 1, 0)

    victim = env.player_action_context(1)
    displacer = [o for o in victim.opponents if o.player_index == 0][0]
    assert displacer.displacements == 1
    assert [o for o in victim.opponents if o.player_index == 2][0].displacements == 0
    assert env.player_view(1)["opponents"][0]["displacements"] == 1

    from features import N_OPPONENT_FEATURES, N_STATE_FEATURES, state_features

    row = state_features(victim)
    assert row.shape == (N_STATE_FEATURES,)
    slot_base = N_STATE_FEATURES - 4 * N_OPPONENT_FEATURES
    ordered = sorted(victim.opponents, key=lambda o: (o.player_index - 1) % 3)
    pusher_slot = [i for i, o in enumerate(ordered) if o.player_index == 0][0]
    quiet_slot = 1 - pusher_slot
    last = N_OPPONENT_FEATURES - 1
    import pytest

    assert row[slot_base + pusher_slot * N_OPPONENT_FEATURES + last] == pytest.approx(1 / 3)
    assert row[slot_base + quiet_slot * N_OPPONENT_FEATURES + last] == 0.0

    env._start_next_haul()
    assert env.players[1].displacements == 0
    assert env.players[0].displacements == 1


def test_pit_cage_is_planned_before_the_player_leaves():
    env = MineEnv(opponents=2)
    env.reset(seed=4)

    env.step(PIT_CAGE)
    assert not env.players[0].escaped
    assert env.players[0].position == PIT_CAGE
    assert env.players[0].cage_index == 2
    assert not env.action_mask()[PIT_CAGE]
    assert any(env.action_mask()[:PIT_CAGE])

    env.step(PASS)
    assert env.haul == 1
    assert env.players[0].escaped


def test_game_scores_three_hauls_and_terminates_after_the_third():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    for player in env.players[1:]:
        player.dead = True

    for haul in (1, 2):
        env.step(PIT_CAGE)
        _observation, _reward, terminated, _truncated, info = env.step(PASS)
        assert not terminated
        assert info["game"] == f"{haul}/3"
        assert env.haul == haul + 1

        for player in env.players[1:]:
            player.dead = True

    env.step(PIT_CAGE)
    _observation, _reward, terminated, _truncated, info = env.step(PASS)
    assert terminated
    assert info["game"] == "3/3"
    assert info["game_over"]
    assert env.players[0].score == 10


def test_player_gets_regular_choices_again_from_pit_cage():
    env = MineEnv(opponents=2)
    env.reset(seed=4)

    env.step(PIT_CAGE)

    legal_card_actions = [
        action for action in range(PIT_CAGE) if env.action_mask()[action]
    ]
    assert legal_card_actions
    assert all(action // 5 < 2 for action in legal_card_actions)


def test_lorry_allows_only_previous_cage_or_pass():
    env = MineEnv(opponents=2)
    env.reset(seed=4)

    env.step(LORRY)

    legal = env.action_mask()
    assert not any(legal[:PIT_CAGE])
    assert legal[PIT_CAGE]
    assert not legal[LORRY]
    assert legal[PASS]

    env.step(PIT_CAGE)
    assert env.players[0].position == PIT_CAGE
    assert env.players[0].cage_index == 0


def test_haul_scores_when_only_one_player_remains_for_next_planning_phase():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.players[1].escaped = True
    env.players[2].escaped = True

    env.step(LORRY)
    _observation, _reward, terminated, _truncated, info = env.step(PASS)

    assert not terminated
    assert info["haul"] == 1
    assert info["next_haul"] == 2
    assert env.haul == 2


def test_surviving_players_keep_unused_dynamite_for_next_haul():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.players[0].collected_cards = [
        (DYNAMITE, BACKSIDE_A),
        (GEM, BACKSIDE_B),
    ]
    env.players[1].collected_cards = [(DYNAMITE, BACKSIDE_C)]
    env.players[2].collected_cards = [(DYNAMITE, BACKSIDE_B)]
    env.players[2].dead = True

    env._start_next_haul()

    assert env.players[0].collected_cards == [(DYNAMITE, BACKSIDE_A)]
    assert env.players[1].collected_cards == [(DYNAMITE, BACKSIDE_C)]
    assert env.players[2].collected_cards == []


def test_collected_cards_disappear_and_new_row_requires_no_departure():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.starting_player = 0
    env.rows[0][2] = (GEM, BACKSIDE_A)
    initial_row_count = len(env.rows)
    top_stack_card = env.cards_remaining[0]

    env.step(2)
    env.step(PASS)

    assert env.rows[0][2] is None
    assert len(env.rows) == initial_row_count + 1
    assert env.rows[-1][0] == top_stack_card
    assert env.players[0].column is None
    assert not env.players[0].collected_cards or env.players[0].collected_cards[-1] != top_stack_card
    assert all(player.position == len(env.rows) for player in env.players)


def test_new_row_deals_cards_in_starting_player_order():
    env = MineEnv(opponents=3)
    env.reset(seed=4)
    env.starting_player = 2
    env.rows[0][2] = (GEM, BACKSIDE_A)
    top_cards = env.cards_remaining[:env.n_players]

    env.step(2)
    env.step(PASS)

    assert env.starting_player == 3
    assert env.rows[-1] == [top_cards[2], top_cards[3], top_cards[0], top_cards[1]]


def test_lorry_player_stays_in_the_mine_for_the_next_round():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.starting_player = 0
    initial_row_count = len(env.rows)

    env.step(LORRY)
    env.step(PASS)

    assert not env.players[0].escaped
    assert len(env.rows) == initial_row_count + 1


def test_everybody_on_the_lorry_ends_the_haul_at_once():
    """Nobody can collect a card, so no new row is dealt and the haul is over."""
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.starting_player = 0
    for player in env.players:
        player.position = LORRY
    env.planning_turns = 1
    env.planned_players = [True] * env.n_players
    rows_before = len(env.rows)

    _observation, _reward, terminated, _truncated, info = env.step(PASS)

    assert not terminated
    assert info["outcome"] == "lorry"
    assert env.haul == 2
    # No extra row is dealt: the haul ended instead of growing the mine.
    assert len(env.haul_history) == 1
    assert rows_before == 3


def test_players_leave_when_no_cards_remain_in_the_mine():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.rows = [[None] * env.n_players for _ in env.rows]
    env.planned_players = [True] * env.n_players
    env.consecutive_passes = 1

    _observation, _reward, terminated, _truncated, info = env.step(PASS)

    assert not terminated
    assert info["mine_empty"] is True
    assert env.haul == 2
    assert all(not player.dead for player in env.players)


def test_players_at_new_mine_start_can_pass_without_collecting_a_card():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.starting_player = 0
    env._set_players_to_starting_positions()
    env.planning_turns = 1

    env.step(PASS)

    assert all(not player.collected_cards for player in env.players)


def test_pass_preserves_previous_reward_during_planning():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.starting_player = 0
    env.last_reward = 0.75
    env.planning_player = 1
    env.players[1].position = LORRY
    env.planning_turns = 3
    env.planned_players[1] = True

    _observation, reward, terminated, _truncated, info = env.step_one_player()

    assert reward == 0.75
    assert not terminated
    assert info["outcome"] == "planning"


def test_planning_rewards_do_not_reveal_card_fronts():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.rows[0][0] = (GEM, BACKSIDE_A)
    env.rows[0][1] = (DRAGON, BACKSIDE_A)

    assert env._planning_reward(0) == env._planning_reward(1) == 0.0
    assert env._planning_reward(0) == env._planning_reward(PIT_CAGE)
    assert env._planning_reward(LORRY) < env._planning_reward(0)


def test_action_selection_prefers_pass_when_q_values_are_tied():
    from train_q_learning import select_action
    import numpy as np

    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.players[0].position = LORRY
    env.planning_turns = 1
    env.planned_players[0] = True

    assert select_action(env, {}, 0.0, np.random.default_rng(4)) == PASS


def test_training_handles_p1_without_legal_actions():
    from train_q_learning import train

    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.players[0].escaped = True
    assert not env.action_mask().any()
    assert isinstance(train(1, 2, "random", 4), dict)


def test_opponents_pass_when_on_lorry_or_pit_cage():
    env = MineEnv(opponents=2)
    env.reset(seed=4)

    env.players[1].position = LORRY
    env.planning_turns = 3
    env.planned_players[1] = True
    assert env._opponent_action(1) == PASS
    env.players[1].position = PIT_CAGE
    assert env._opponent_action(1) == PASS


def test_opponents_can_choose_known_own_column_dragons():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.rows[0][1] = (DRAGON, BACKSIDE_A)
    env.players[1].position = 1

    assert 1 in [
        row * 5 + column
        for row, cards in enumerate(env.rows)
        if row < env._movement_depth(env.players[1])
        for column, card in enumerate(cards)
        if card is not None
    ]


def test_one_player_passes_advance_lorry_and_pit_cage_positions():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.players[0].position = LORRY
    env.players[1].position = PIT_CAGE
    env.players[2].position = PIT_CAGE
    env.planning_turns = 3
    env.planned_players = [True] * env.n_players

    for _ in range(3):
        if env.planning_player == 0:
            env.step_one_player(PASS)
        else:
            env.step_one_player()

    assert env.haul == 2
    assert all(player.position == 3 for player in env.players)


def test_haul_continues_while_someone_is_still_in_the_mine():
    """One escaped player must not end the haul for the two still digging."""
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.starting_player = 0
    env.players[0].position = PIT_CAGE
    env.players[0].escaped = True
    env.players[1].position = 1
    env.players[1].column = 1
    env.players[2].position = 2
    env.players[2].column = 2
    env.planning_player = 1
    env.planning_turns = 2
    env.planned_players = [True] * env.n_players

    info = {"outcome": "planning"}
    terminated = False
    for _ in range(10):
        _observation, _reward, terminated, _truncated, info = env.step_one_player()
        if terminated or info["outcome"] != "planning":
            break

    assert not terminated
    assert env.haul == 1
    assert info["outcome"] == "continued"


def test_escaped_player_is_inactive_while_other_players_continue():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.players[0].escaped = True
    env.players[1].position = LORRY
    env.players[2].position = LORRY
    env.planning_player = 0
    env.planned_players = [False] * env.n_players

    assert not env.action_mask().any()
    _, _, _, _, info = env.step_one_player()
    assert info["player"] == 1
    assert env.planning_player == 2


def test_collected_card_backside_is_publicly_recorded():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.starting_player = 0
    env.rows[0][2] = (GEM, BACKSIDE_C)

    env.step(2)
    env.step(PASS)

    assert env.players[0].collected_backsides == [BACKSIDE_C]


def test_deck_has_thirty_cards_with_three_backside_sets():
    env = MineEnv(opponents=2)
    deck = [card for row in env._deal_rows(10) for card in row]

    assert len(deck) == 30
    assert Counter(backside for _front, backside in deck) == Counter({BACKSIDE_A: 10, BACKSIDE_B: 10, BACKSIDE_C: 10})
    for backside in (BACKSIDE_A, BACKSIDE_B, BACKSIDE_C):
        fronts = Counter(front for front, card_backside in deck if card_backside == backside)
        assert fronts == Counter({DRAGON: 2, GEM: 4, DYNAMITE: 4})


# ----------------------------------------------------------------------
# The action grid must cover every row the deck can produce. A haul grows a row
# every round, so sizing the grid by the three-row opening board left the front
# rows with no action at all: agents could neither see them nor name them and
# had to fall back to a shallower row, abandoning the cards below them.
# ----------------------------------------------------------------------
def _deep_mine(opponents: int = 2, depth: int = 7) -> MineEnv:
    env = MineEnv(opponents=opponents)
    env.reset(seed=0)
    while len(env.rows) < depth:
        env.rows.extend(env._deal_next_rows(1))
        env.round += 1
    for player in env.players:
        player.position = len(env.rows)
        player.cage_index = None
    return env


def test_action_grid_covers_the_deepest_board_any_deck_can_produce():
    for opponents in (2, 3, 4):
        n_players = opponents + 1
        deepest_board = DECK_SIZE // n_players
        assert deepest_board <= MAX_ROWS
        assert n_players <= MAX_COLUMNS
    # The last card action must stay below the special actions, or a card would
    # be indistinguishable from "leave via pit cage".
    assert PIT_CAGE == MAX_ROWS * MAX_COLUMNS
    assert PIT_CAGE > (MAX_ROWS - 1) * MAX_COLUMNS + (MAX_COLUMNS - 1)


def test_front_row_is_visible_and_playable_when_the_mine_grows_past_opening_depth():
    env = _deep_mine(depth=7)
    front = len(env.rows) - 1
    column = (env.players[0].column or 0)
    action = front * MAX_COLUMNS + column

    assert front >= 5, "the test needs a board deeper than the old five-row grid"
    assert env.rows[front][column] is not None
    assert env._legal(action, 0)
    assert action < PIT_CAGE, "a card action must never collide with pit cage"

    view = env.player_view(0)
    assert len(view["rows"]) == len(env.rows)
    assert view["rows"][front][column] is not None
    assert len(env.player_action_context(0).public_rows) == len(env.rows)
    assert env._observation().shape == env.observation_space.shape


def test_every_row_of_a_grown_mine_has_a_reachable_card_action():
    env = _deep_mine(depth=8)
    context = env.player_action_context(0)
    legal = [action for action, allowed in enumerate(env.action_mask(0)) if allowed]
    legal_card_rows = {action // MAX_COLUMNS for action in legal if action < PIT_CAGE}

    assert legal_card_rows == set(range(len(env.rows)))
    assert {
        action // MAX_COLUMNS for action in context.legal_actions if action < PIT_CAGE
    } == legal_card_rows
