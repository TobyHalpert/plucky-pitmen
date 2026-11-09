"""Can a cheap blast cancel make a deliberate step onto a known dragon safe?

The engine's phase order says no, and for two independent reasons:

1. the card under a figure is revealed in phase 2 of the round, and
   ``_prepare_blasts`` runs only when phase 2 woke nothing, so a pending blast is
   never drawn against the card its owner is standing on;
2. any move other than PASS cancels the announcement, so a seat cannot both keep
   a blast alive and leave the card.

What does work is the plain trap: step on, act again, be gone before the reveal.
``probe_dragon_step.py`` measures that in real games; these tests pin the rule
that makes it necessary.
"""

from __future__ import annotations

from game_env import (
    DYNAMITE,
    GEM,
    MAX_COLUMNS,
    PASS,
    PIT_CAGE,
    DRAGON,
    MineEnv,
)

BACKSIDE_A = 0
BACKSIDE_B = 1
BACKSIDE_C = 2


def _two_dynamite(env: MineEnv, seat: int = 0) -> None:
    env.players[seat].collected_cards = [
        (DYNAMITE, BACKSIDE_A),
        (DYNAMITE, BACKSIDE_B),
    ]


def _safe_deck() -> list[tuple[int, int]]:
    return [(GEM, BACKSIDE_A), (GEM, BACKSIDE_B), (GEM, BACKSIDE_C)] * 2


def test_a_pending_blast_is_never_drawn_against_the_card_you_stand_on():
    """The blast cannot cover the card under the figure: the reveal comes first."""
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.rows[0][0] = (DRAGON, BACKSIDE_A)
    for column in (1, 2):
        env.rows[0][column] = (GEM, BACKSIDE_B)
    for seat, player in enumerate(env.players):
        player.position = 0
        player.column = seat
    _two_dynamite(env)
    deck = _safe_deck()
    env.cards_remaining = list(deck)
    env.begin_blast(0)

    _reward, info = env._execute_round()

    assert info["outcome"] == "dragon"
    assert all(player.dead for player in env.players)
    # The blast died with its owner: never drawn, never charged for.
    assert env.players[0].blast_pending is False
    assert env.players[0].blast_cards == []
    assert env.used_blast_backsides == []
    assert len(env.cards_remaining) == len(deck)


def test_stepping_off_the_dragon_cancels_the_announcement_and_saves_the_seat():
    """The trap line: step on, act again, leave -- and the blast is spent on nothing."""
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    # Rows only get shallower: a figure may name any card in a row above it, so
    # the step off needs a shallower card to step onto.
    env.rows[2][0] = (DRAGON, BACKSIDE_A)
    env.rows[1][0] = (GEM, BACKSIDE_B)
    for column in (1, 2):
        env.rows[0][column] = (GEM, BACKSIDE_B)
        env.rows[1][column] = (GEM, BACKSIDE_C)
        env.rows[2][column] = (GEM, BACKSIDE_C)
    env.planning_player = 0
    env.planned_players[0] = True
    env.players[0].position = 4
    env.players[0].column = 0
    _two_dynamite(env)
    env.begin_blast(0)

    # Deeper figures may name any card in a shallower row, so both steps are
    # legal.  ``step_one_player`` always acts for the current planning seat, so
    # the second decision has to give the turn back to seat 0.
    env.step_one_player(2 * MAX_COLUMNS + 0, force_action=True)
    assert (env.players[0].position, env.players[0].column) == (2, 0)
    assert env.players[0].blast_pending is False, "the move itself cancels the blast"

    env.planning_player = 0
    env.step_one_player(1 * MAX_COLUMNS + 0, force_action=True)
    assert (env.players[0].position, env.players[0].column) == (1, 0)

    _reward, info = env._execute_round()

    assert info.get("outcome") != "dragon"
    assert not any(player.dead for player in env.players)
    assert env.used_blast_backsides == []


def test_a_seat_that_does_not_act_again_dies_on_its_own_dragon():
    """Acting again is load-bearing: without it the dig is simply fatal."""
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.rows[0][0] = (DRAGON, BACKSIDE_A)
    for column in (1, 2):
        env.rows[0][column] = (GEM, BACKSIDE_B)
    env.planning_player = 0
    env.planned_players[0] = True
    env.players[0].position = 1
    env.players[0].column = 0

    env.step_one_player(0 * MAX_COLUMNS + 0, force_action=True)
    assert (env.players[0].position, env.players[0].column) == (0, 0)
    env.planning_player = 0
    env.step_one_player(int(PASS), force_action=True)

    _reward, info = env._execute_round()

    assert info["outcome"] == "dragon"
    assert env.players[0].dead is True


def test_the_cage_still_ends_the_haul_for_a_seat_standing_on_its_dragon():
    """Phase 1 escapes run first, so a caged seat is never the one who wakes it."""
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.rows[0][0] = (DRAGON, BACKSIDE_A)
    for column in (1, 2):
        env.rows[0][column] = (GEM, BACKSIDE_B)
    env.players[0].position = PIT_CAGE
    env.players[0].cage_index = 0
    for seat in (1, 2):
        env.players[seat].position = 1
        env.players[seat].column = seat
        env.rows[1][seat] = (GEM, BACKSIDE_B)

    _reward, info = env._execute_round()

    assert env.players[0].escaped is True
    assert env.players[0].dead is False
    assert info["outcome"] != "dragon"
