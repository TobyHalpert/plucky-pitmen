"""Tests for the matched starting-player study."""

from __future__ import annotations

from agents import make_scripted_agent
from game_env import MineEnv
from selfplay import GameResult, play_game
from starting_player_study import _game_record, _seed_level_records


def test_game_record_computes_split_win_share_delta():
    result = GameResult(
        seed=17,
        scores=[9, 5, 7],
        turns=1,
        winner=None,
        winners=[0, 2],
    )

    record = _game_record(result, starter=0, players=3)

    assert record["starter_win_share"] == 0.5
    assert record["nonstarter_mean_win_share"] == 0.25
    assert record["win_share_delta"] == 0.25
    assert record["starter_win_rate"] == 1.0


def test_seed_summary_requires_all_starter_rotations_and_nonaborted_games():
    records = [
        _game_record(
            GameResult(
                seed=23,
                scores=[3, 2, 1],
                turns=1,
                winner=starter,
                winners=[starter],
            ),
            starter=starter,
            players=3,
        )
        for starter in range(3)
    ]

    complete_seeds, metrics = _seed_level_records(records, players=3)
    assert complete_seeds == [23]
    assert metrics["win_share_delta"] == [1.0]

    records[0]["aborted"] = True
    complete_seeds, metrics = _seed_level_records(records, players=3)
    assert complete_seeds == []
    assert metrics["win_share_delta"] == []


def test_play_game_preserves_forced_starter_in_supplied_environment():
    # Seed 1 naturally selects seat 1; force seat 2 so a reset inside
    # play_game would make this assertion fail.
    env = MineEnv(opponents=2)
    env.reset(seed=1)
    env.starting_player = 2
    env.planning_player = 2

    first_actor: list[int] = []
    step_one_player = env.step_one_player

    def record_first_actor(*args, **kwargs):
        if not first_actor:
            first_actor.append(env.planning_player)
        return step_one_player(*args, **kwargs)

    env.step_one_player = record_first_actor
    agents = [make_scripted_agent("rule", seed=100 + seat) for seat in range(3)]

    play_game(agents, seed=1, env=env)

    assert first_actor == [2]
