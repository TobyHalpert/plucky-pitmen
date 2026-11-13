"""Tests for foreign-column dragon-signal dodging, mid-round displacement
unlocking, and the 131-input retrained ``selfplay4`` policy.
"""

from __future__ import annotations

import features
from agents import NeuralAgent
from game_env import (
    DRAGON,
    DYNAMITE,
    GEM,
    LORRY,
    MAX_COLUMNS,
    PASS,
    PIT_CAGE,
    MineEnv,
)
from learning import PolicyNet
from selfplay import play_game


def test_row0_foreign_dragon_signal_unlocks_lorry_and_refuses_pass():
    """When a displaced column owner flees to PIT_CAGE in a calm mine, the
    raider standing on that owner's row-0 card detects the signalled dragon,
    refuses PASS, and dodges via LORRY rather than dying on [PASS].
    """
    checked = 0
    for seed in range(60):
        env = MineEnv(opponents=3)
        env.reset(seed=seed)
        if env.rows[0][1][0] != DRAGON:
            continue
        env.players[1].position, env.players[1].column, env.planned_players[1] = 0, 1, True
        env.players[2].position, env.players[2].column, env.planned_players[2] = 2, 2, True
        env.players[3].position, env.players[3].column, env.planned_players[3] = 2, 3, True
        env.planning_player = 0
        # P0 displaces P1 at (row=0, col=1); P1 flees to PIT_CAGE; P2 and P3 pass.
        env.step_one_player(0 * MAX_COLUMNS + 1, force_action=True)
        env.step_one_player(PIT_CAGE, force_action=True)
        env.step_one_player(PASS, force_action=True)
        env.step_one_player(PASS, force_action=True)

        ctx0 = env.player_action_context(0)
        assert features.own_card_is_signalled_dragon(ctx0)
        assert int(PASS) in features.holds_a_signalled_dragon(ctx0)
        mask = features.action_mask(ctx0)
        assert not mask[int(PASS)]
        assert mask[int(LORRY)]
        checked += 1
    assert checked >= 5


def test_safe_foreign_push_without_cage_signal_keeps_pass_open():
    """When the displaced column owner stays in the mine instead of fleeing to
    PIT_CAGE, the card is not treated as a signalled dragon and PASS stays open.
    """
    for seed in range(40):
        env = MineEnv(opponents=3)
        env.reset(seed=seed)
        if env.rows[1][1][0] == DRAGON:
            continue
        env.players[1].position, env.players[1].column, env.planned_players[1] = 1, 1, True
        env.players[2].position, env.players[2].column, env.planned_players[2] = 2, 2, True
        env.players[3].position, env.players[3].column, env.planned_players[3] = 2, 3, True
        env.planning_player = 0
        env.step_one_player(1 * MAX_COLUMNS + 1, force_action=True)
        # Displaced owner P1 steps onto shallower card (0, 1) rather than PIT_CAGE.
        env.step_one_player(0 * MAX_COLUMNS + 1, force_action=True)
        env.step_one_player(PASS, force_action=True)
        env.step_one_player(PASS, force_action=True)

        ctx0 = env.player_action_context(0)
        assert not features.own_card_is_signalled_dragon(ctx0)
        mask = features.action_mask(ctx0)
        assert mask[int(PASS)]
        break


def test_climbs_off_card_unlocks_midround_displacement_and_leader_row0_push():
    """Mid-round displacements onto rivals in their own column are unlocked when
    not standing on a gem, and row-0 pushes on the score leader are unlocked
    even when standing on a gem.
    """
    env = MineEnv(opponents=3)
    env.reset(seed=3)
    # Force P0 onto own dynamite at (2, 0), P1 onto own card at (0, 1).
    env.rows[2][0] = (DYNAMITE, env.rows[2][0][1])
    env.players[0].position, env.players[0].column, env.planned_players[0] = 2, 0, True
    env.players[1].position, env.players[1].column, env.planned_players[1] = 0, 1, True
    env.players[2].position, env.players[2].column, env.planned_players[2] = 2, 2, True
    env.players[3].position, env.players[3].column, env.planned_players[3] = 2, 3, True
    env.planning_player = 0

    ctx_dyn = env.player_action_context(0)
    doomed_dyn = features.climbs_off_the_card_it_chose(ctx_dyn)
    assert (0 * MAX_COLUMNS + 1) not in doomed_dyn

    # Standing on an own gem: non-leader push is masked, but score-leader row-0 push is open.
    env.rows[2][0] = (GEM, env.rows[2][0][1])
    env.players[0].score = 2
    env.players[1].score = 2
    ctx_gem_tied = env.player_action_context(0)
    assert (0 * MAX_COLUMNS + 1) in features.climbs_off_the_card_it_chose(ctx_gem_tied)

    env.players[1].score = 6
    ctx_gem_trailing = env.player_action_context(0)
    assert (0 * MAX_COLUMNS + 1) not in features.climbs_off_the_card_it_chose(
        ctx_gem_trailing
    )


def test_turn1_last_chance_and_parachute_push_features():
    """Turn-1 pushes track unsettled pushers, pigeonhole overload, early-mover
    priority, and shallower-own-gem / no-own-gem parachutes.
    """
    env = MineEnv(opponents=3)
    env.reset(seed=5)
    # Same-row own gem at row 2 with two unsettled players left: urgency is 0.
    env.rows[0][0] = None
    env.rows[1][0] = None
    env.rows[2][0] = (GEM, 0)
    env.rows[2][1] = (GEM, 0)
    env.players[1].position, env.players[1].column, env.planned_players[1] = 2, 1, True
    env.planning_player = 0
    ctx_same = env.player_action_context(0)
    af_same = features.action_features(ctx_same, 2 * MAX_COLUMNS + 1)
    assert af_same[32] == 0.0
    assert af_same[33] == 0.0

    # Put P0's own gem at row 0 < row 2 (parachute ahead):
    env.rows[0][0] = (GEM, 0)
    env.rows[2][0] = None
    ctx_early = env.player_action_context(0)
    af_early = features.action_features(ctx_early, 2 * MAX_COLUMNS + 1)
    assert 0.25 <= af_early[32] <= 0.55
    assert af_early[33] == 1.0

    # Pigeonhole overload: P1 and P2 both settled at row 2 (K=2), only P3 unsettled (U=1):
    env.players[2].position, env.players[2].column, env.planned_players[2] = 2, 2, True
    ctx_over = env.player_action_context(0)
    af_over = features.action_features(ctx_over, 2 * MAX_COLUMNS + 1)
    assert af_over[32] > af_early[32]

    # Once P3 has also settled (even at row 2 > row 1!), nobody unsettled remains (U=0):
    env.players[1].position, env.players[1].column, env.planned_players[1] = 1, 1, True
    env.players[3].position, env.players[3].column, env.planned_players[3] = 2, 3, True
    ctx_last = env.player_action_context(0)
    af_last = features.action_features(ctx_last, 1 * MAX_COLUMNS + 1)
    assert af_last[32] >= 1.0
    assert af_last[33] == 1.0


def test_retrained_selfplay4_uses_turn1_and_turn2_displacements():
    """Shipped ``selfplay4.npz`` has 133 inputs and displaces on Turn 1 and Turn 2."""
    assert features.N_ACTION_FEATURES == 34
    assert features.N_INPUT_FEATURES == 133
    net = PolicyNet.load("checkpoints/selfplay4.npz")
    assert net.params["W1"].shape == (133, 64)

    turn1_pushes = 0
    turn2_pushes = 0
    mid_mover_t1_pushes = 0
    early_victim_pushes = 0

    class Counter(NeuralAgent):
        def decide(self, ctx):
            choice = super().decide(ctx)
            if 0 <= choice.action < PIT_CAGE:
                r, c = divmod(choice.action, MAX_COLUMNS)
                if c != ctx.player_index and features._occupant(ctx, r, c) == c:
                    nonlocal turn1_pushes, turn2_pushes
                    nonlocal mid_mover_t1_pushes, early_victim_pushes
                    sp = ctx.starting_player
                    if (c - sp) % 4 in (0, 1):
                        early_victim_pushes += 1
                    if not ctx.own_planned:
                        turn1_pushes += 1
                        if (ctx.player_index - sp) % 4 in (1, 2):
                            mid_mover_t1_pushes += 1
                    else:
                        turn2_pushes += 1
            return choice

    for seed in range(12):
        agents = [Counter(net, seed=seed * 31 + i, temperature=0.7) for i in range(4)]
        res = play_game(agents, seed=seed)
        assert not res.aborted
    assert turn1_pushes >= 20
    assert turn2_pushes >= 5
    assert mid_mover_t1_pushes >= 10
    assert early_victim_pushes >= 50
