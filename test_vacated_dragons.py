"""The departure inference: a rival walking off its own card is a dragon tell.

Each backside colour is ten cards with two dragons, so a dragon in anybody's
column is a dragon that is not in the draw pile.  A seat can see its own column's
fronts; a rival's departure from a card in its own column is the public hint that
it was looking at a dragon it did not want to keep.

These tests pin three things: only *voluntary own-column card-to-card* steps are
recorded, the default (trust 0) leaves every earlier measurement bit-identical,
and trusting the tell lowers that colour's risk for the right arithmetic reason.
"""

from __future__ import annotations

import dataclasses

import numpy as np

import features
from features import backside_stats, suspected_dragon_counts
from game_env import DRAGON, GEM, LORRY, MAX_COLUMNS, PIT_CAGE, MineEnv


def _env_with_dragon_at(row: int, column: int, seed: int = 5) -> MineEnv:
    env = MineEnv(opponents=2)
    env.reset(seed=seed)
    env.rows[row][column] = (DRAGON, 0)
    return env


def test_walking_off_your_own_card_records_its_backside():
    env = _env_with_dragon_at(2, 0)
    env.players[0].position = 2
    env.players[0].column = 0
    env._apply_action(0, 1 * MAX_COLUMNS + 1)  # step to row 1, column 1
    assert len(env.vacated_events) == 1
    player, row, column, back, front = env.vacated_events[0]
    assert (player, row, column, back, front) == (0, 2, 0, 0, DRAGON)
    # ... and the policy sees the colour, never the front.
    ctx = env.player_action_context(0)
    assert ctx.vacated_backsides == (0,)


def test_a_step_inside_someone_elses_column_is_not_recorded():
    env = _env_with_dragon_at(2, 2)
    env.players[0].position = 2
    env.players[0].column = 2  # raiding P3's column, not its own
    env._apply_action(0, 1 * MAX_COLUMNS + 1)
    assert env.vacated_events == []


def test_a_displacement_does_not_count_as_walking_off():
    """Being pushed is involuntary: it says nothing about the card."""
    env = _env_with_dragon_at(1, 1, seed=6)
    env.players[1].position = 1
    env.players[1].column = 1
    env.players[0].position = 2
    env.players[0].column = None
    env._apply_action(0, 1 * MAX_COLUMNS + 1)  # pushes P2 on its own card
    assert env.players[1].position == 1
    assert env.players[1].column == 1
    assert env.players[1].displaced is True
    assert env.vacated_events == []

    # When P2 then takes its mandatory move off (1, 1), that forced move is
    # still involuntary and must not be recorded as walking off a dragon.
    env._apply_action(1, 0 * MAX_COLUMNS + 1)
    assert env.players[1].position == 0
    assert env.players[1].displaced is False
    assert env.vacated_events == []


def test_leaving_for_a_cage_or_lorry_is_not_recorded():
    env = _env_with_dragon_at(2, 0)
    env.players[0].position = 2
    env.players[0].column = 0
    env._apply_action(0, PIT_CAGE)
    env.players[0].position = 2
    env.players[0].column = 0
    env._apply_action(0, LORRY)
    assert env.vacated_events == []


def test_the_default_ignores_the_tell_completely():
    """Trust 0 has to be bit-identical, or every recorded number moves."""
    env = _env_with_dragon_at(2, 0)
    ctx = env.player_action_context(0)
    suspecting = dataclasses.replace(ctx, vacated_backsides=(0, 0))
    assert suspected_dragon_counts(suspecting).tolist() == [2.0, 0.0, 0.0]
    plain = backside_stats(ctx)
    with_evidence = backside_stats(suspecting)
    assert np.array_equal(plain[3], with_evidence[3])
    assert np.array_equal(plain[2], with_evidence[2])


def test_trusting_the_tell_lowers_that_colours_risk():
    env = _env_with_dragon_at(2, 0)
    ctx = env.player_action_context(0)
    one = dataclasses.replace(ctx, vacated_backsides=(0,))
    off = backside_stats(one, trust_suspected=0.0)
    on = backside_stats(one, trust_suspected=1.0)
    # A suspected dragon leaves both the remaining-dragon count and the unknown
    # pool: the card is lying in the mine, not in the pile.
    assert on[2][0] == off[2][0] - 1
    assert on[3][0] < off[3][0]
    assert np.array_equal(on[3][1:], off[3][1:]), "other colours must not move"


def test_two_departures_of_one_colour_make_it_dragon_free():
    env = _env_with_dragon_at(2, 0)
    ctx = env.player_action_context(0)
    both = dataclasses.replace(ctx, vacated_backsides=(0, 0))
    assert backside_stats(both, trust_suspected=1.0)[3][0] == 0.0


def test_the_record_resets_with_the_deck_between_hauls():
    """A new haul is a reshuffled pile: last haul's tells are worthless."""
    env = MineEnv(opponents=2)
    env.reset(seed=8)
    env.vacated_events.append((0, 0, 0, 1, GEM))
    assert env.player_action_context(0).vacated_backsides == (1,)
    env._start_next_haul()
    assert env.vacated_events == []
    assert env.player_action_context(0).vacated_backsides == ()


def test_opponent_repeated_hopping_records_public_tells_and_lowers_blast_risk():
    """Repeatedly hopping over an own-column card marks that cell risky and the pile safer."""
    from agents import preferred_blast_choice
    from features import (
        action_features,
        backside_stats,
        blast_dragon_risk,
        cell_suspicion_weights,
        suspected_dragon_counts,
    )
    from game_env import DYNAMITE, FRONT_UNKNOWN

    env = MineEnv(opponents=3)
    env.reset(seed=21)
    # Place a backside-0 dragon at (2, 1) and gems at (0, 1) in P2's column.
    env.rows[2][1] = (DRAGON, 0)
    env.rows[1][1] = (GEM, 2)
    env.rows[0][1] = (GEM, 2)
    env.players[0].collected_cards = [(DYNAMITE, 1), (DYNAMITE, 2)]

    ctx_before = env.player_action_context(0)
    blast_before = blast_dragon_risk(ctx_before)
    risk0_before = float(backside_stats(ctx_before, trust_suspected=1.0)[3][0])

    # Round 0: P2 hops from front of mine (depth 3) past (2, 1) and (1, 1) to (0, 1).
    env.planning_player = 1
    env._apply_action(1, 0 * MAX_COLUMNS + 1)
    ctx_r0 = env.player_action_context(0)
    assert any(col == 1 and row == 2 and back == 0 for _rnd, col, row, back, _ff in ctx_r0.hopped_cells)
    # Ground-truth card front is never leaked into hopped_cells (5-tuples only).
    assert all(len(ev) == 5 for ev in ctx_r0.hopped_cells)
    assert cell_suspicion_weights(ctx_r0)[(2, 1)] == 0.35

    # Round 1: another row dealt, P2 again starts at front and hops past (2, 1).
    env.round = 1
    env.players[1].position = 3
    env.players[1].column = None
    env._apply_action(1, 1 * MAX_COLUMNS + 1)
    ctx_r1 = env.player_action_context(0)
    assert cell_suspicion_weights(ctx_r1)[(2, 1)] == 0.65
    assert suspected_dragon_counts(ctx_r1)[0] >= 0.65 - 1e-6

    risk0_after = float(backside_stats(ctx_r1, trust_suspected=1.0)[3][0])
    blast_after = blast_dragon_risk(ctx_r1)
    assert risk0_after < risk0_before
    assert blast_after < blast_before
    # Action feature 21 flags cell (2, 1) itself as suspected dragon.
    af = action_features(ctx_r1, 2 * MAX_COLUMNS + 1)
    assert af[21] >= 0.65 - 1e-6

    # Blind blast keeper prefers backside 0 over untouched backside 1 or 2 when equal otherwise.
    env_clean = MineEnv(opponents=3)
    env_clean.reset(seed=22)
    # Make all board backsides balanced and add two hops on backside 0 in P2's column.
    for r in range(3):
        for c in range(4):
            env_clean.rows[r][c] = (GEM, (r + c) % 3)
    env_clean.rows[2][1] = (DRAGON, 0)
    env_clean.hopped_events = [(0, 1, 2, 0, False, DRAGON), (1, 1, 2, 0, False, DRAGON)]
    choice = preferred_blast_choice(
        ((FRONT_UNKNOWN, 1), (FRONT_UNKNOWN, 0), (FRONT_UNKNOWN, 2)),
        env_clean.player_action_context(0),
    )
    assert choice == 1

