from collections import Counter

from game_env import (
    BACKSIDE_A,
    BACKSIDE_B,
    BACKSIDE_C,
    BACKSIDE_UNKNOWN,
    DRAGON,
    DYNAMITE,
    GEM,
    LORRY,
    MineEnv,
    PASS,
    PIT_CAGE,
)


def test_random_game_terminates():
    env = MineEnv(opponents=2)
    observation, _ = env.reset(seed=3)
    env.starting_player = 0
    assert observation.shape == (65,)
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
    assert env.partie == 2

    env.reset(seed=4)
    assert env.starting_player == first_starter
    assert env.players[0].score == 0


def test_passing_is_not_allowed_on_first_planning_turn():
    env = MineEnv(opponents=2)
    env.reset(seed=4)

    assert not env._legal(PASS)
    env.step(0)
    assert env._legal(PASS)


def test_p1_cannot_pass_when_an_opponent_started():
    env = MineEnv(opponents=2)
    env.reset(seed=7)
    env.planning_player = 1
    env.players[1].position = LORRY
    env.step_one_player()

    env.planning_player = 0
    assert not env._legal(PASS)


def test_observation_exposes_backsides_but_not_fronts():
    env = MineEnv(opponents=2)
    observation, _ = env.reset(seed=12)

    visible_backsides = [
            observation[15 + row_index * 5 + player_index]
        for row_index, row in enumerate(env.rows)
        for player_index, _card in enumerate(row)
    ]
    expected_backsides = [backside for row in env.rows for _front, backside in row]

    assert visible_backsides == expected_backsides
    assert all(value in (BACKSIDE_A, BACKSIDE_B, BACKSIDE_C) for value in visible_backsides)
    assert BACKSIDE_UNKNOWN in observation[[15 + row_index * 5 + 4 for row_index in range(3)]]


def test_observation_reveals_only_p1_column_fronts():
    env = MineEnv(opponents=2)
    observation, _ = env.reset(seed=12)

    own_fronts = [
        observation[40 + row_index * 5]
        for row_index in range(3)
    ]
    expected_fronts = [env.rows[row_index][0][0] + 4 for row_index in range(3)]

    assert own_fronts == expected_fronts
    assert observation[15] == env.rows[0][0][1]
    assert all(
        observation[40 + row_index * 5 + column] == BACKSIDE_UNKNOWN
        for row_index in range(3)
        for column in range(1, 3)
    )


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
    assert env.partie == 1
    assert env.players[0].escaped


def test_match_scores_three_parties_and_terminates_after_the_third():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    for player in env.players[1:]:
        player.dead = True

    for partie in (1, 2):
        env.step(PIT_CAGE)
        _observation, _reward, terminated, _truncated, info = env.step(PASS)
        assert not terminated
        assert info["match"] == f"{partie}/3"
        assert env.partie == partie + 1

        for player in env.players[1:]:
            player.dead = True

    env.step(PIT_CAGE)
    _observation, _reward, terminated, _truncated, info = env.step(PASS)
    assert terminated
    assert info["match"] == "3/3"
    assert info["match_over"]
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


def test_collected_cards_disappear_and_new_row_requires_no_departure():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.starting_player = 0
    env.rows[0][2] = (GEM, BACKSIDE_A)
    initial_row_count = len(env.rows)

    env.step(2)
    env.step(PASS)

    assert env.rows[0][2] is None
    assert len(env.rows) == initial_row_count + 1
    assert all(player.position == len(env.rows) for player in env.players)


def test_lorry_player_stays_in_the_mine_for_the_next_round():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.starting_player = 0
    initial_row_count = len(env.rows)

    env.step(LORRY)
    env.step(PASS)

    assert not env.players[0].escaped
    assert len(env.rows) == initial_row_count + 1


def test_lorry_players_can_continue_after_two_additional_rows():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.starting_player = 0
    for player in env.players:
        player.position = LORRY
    env.planning_turns = 1
    env.planned_players = [True] * env.n_players

    for _ in range(2):
        for player in env.players:
            player.position = LORRY
        env.step(PASS)
        env.planning_turns = 1
        env.planned_players = [True] * env.n_players

    assert env.partie == 1
    assert len(env.rows) == 5


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

    assert env._planning_reward(0) == env._planning_reward(1)
    assert env._planning_reward(0) == env._planning_reward(PIT_CAGE)
    assert env._planning_reward(0) > env._planning_reward(LORRY)


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

    assert env.partie == 2
    assert all(player.position == 3 for player in env.players)


def test_pit_cage_player_does_not_end_partie_while_lorry_players_remain():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.starting_player = 0
    env.players[0].position = PIT_CAGE
    env.players[0].escaped = True
    env.players[1].position = LORRY
    env.players[2].position = LORRY
    env.planning_player = 1
    env.planning_turns = 2
    env.planned_players = [True] * env.n_players

    _observation, _reward, terminated, _truncated, info = env.step_one_player()
    if not terminated:
        _observation, _reward, terminated, _truncated, info = env.step_one_player()

    assert not terminated
    assert env.partie == 1
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


def test_ending_on_card_gets_position_reward_without_revealing_front():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.players[0].position = 0
    env.players[0].column = 0
    env.rows[0][0] = (GEM, BACKSIDE_A)

    reward, _info = env._execute_round()

    gem_reward = reward

    env.players[0].position = 0
    env.players[0].column = 1
    env.rows[0][1] = (DRAGON, BACKSIDE_A)
    dragon_reward, _info = env._execute_round()

    assert gem_reward == dragon_reward


def test_ending_on_empty_space_gets_no_position_reward():
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.players[0].position = 0
    env.players[0].column = 0
    env.rows[0][0] = None

    reward, _info = env._execute_round()

    assert reward == 0.0


def test_deck_has_thirty_cards_with_three_backside_sets():
    env = MineEnv(opponents=2)
    deck = [card for row in env._deal_rows(10) for card in row]

    assert len(deck) == 30
    assert Counter(backside for _front, backside in deck) == Counter({BACKSIDE_A: 10, BACKSIDE_B: 10, BACKSIDE_C: 10})
    for backside in (BACKSIDE_A, BACKSIDE_B, BACKSIDE_C):
        fronts = Counter(front for front, card_backside in deck if card_backside == backside)
        assert fronts == Counter({DRAGON: 2, GEM: 4, DYNAMITE: 4})
