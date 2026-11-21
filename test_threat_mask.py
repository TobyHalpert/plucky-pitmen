"""Tests for where the threat reaction lives (BALANCE.md §6, probe_threat_mask.py).

A pending rival blast is not only a change in the state: ``round_dragon_hazard``
returns 1.0 for it, and that is what makes an *empty-handed* pit-cage move legal
at all.  The reaction therefore partly sits in the action mask, which is why a
paired counterfactual over masked distributions can read as a reaction that the
policy itself does not make.  These tests pin the mechanism, so the §6 number
cannot silently drift again.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from features import action_mask, round_dragon_hazard, split_action
from game_env import GEM, PASS, PIT_CAGE, MineEnv


def _cage_legal(ctx) -> bool:
    mask = action_mask(ctx)
    return any(
        mask[action] for action in np.flatnonzero(mask)
        if split_action(int(action))[0] == PIT_CAGE
    )


def _threatened(ctx):
    return dataclasses.replace(
        ctx,
        opponents=tuple(
            dataclasses.replace(opponent, blast_pending=True)
            for opponent in ctx.opponents
        ),
    )


def test_a_pending_blast_is_read_as_certain_danger():
    env = MineEnv(opponents=2)
    env.reset(seed=3)
    ctx = env.player_action_context(0)
    assert round_dragon_hazard(ctx) < 1.0
    assert round_dragon_hazard(_threatened(ctx)) == 1.0


def test_a_pending_blast_is_what_makes_an_empty_handed_cage_legal():
    """The mask, not the policy, is where the empty-handed flinch comes from."""
    found = False
    for seed in range(40):
        env = MineEnv(opponents=2)
        env.reset(seed=seed)
        for _ in range(80):
            if env.game_finished:
                break
            ctx = env.player_action_context(0)
            empty_handed = not any(front == GEM for front, _ in ctx.private_cards)
            still_inside = ctx.legal_actions and PIT_CAGE in ctx.legal_actions
            if (
                empty_handed
                and still_inside
                and not _cage_legal(ctx)
            ):
                threatened = _threatened(ctx)
                assert round_dragon_hazard(threatened) >= round_dragon_hazard(ctx)
                assert _cage_legal(threatened), (
                    "a pending blast has to open the cage for a seat with "
                    "nothing banked: that is rule 4 of action_mask"
                )
                found = True
                break
            action = int(PASS) if PASS in ctx.legal_actions else int(ctx.legal_actions[0])
            env.step_one_player(action, force_action=True)
        if found:
            break
    assert found, "no empty-handed decision with a masked cage was found"


def test_the_mask_split_puts_every_pair_in_one_of_two_groups(tmp_path):
    from learning import PolicyNet
    from probe_threat_mask import mask_split

    net = PolicyNet.load("checkpoints/selfplay4.npz")
    report = mask_split(net, players=4, games=2, seed=424242)
    assert report["pairs"] == sum(
        group["pairs"] for group in report["groups"].values()
    )
    assert report["pairs"] > 0
    for group in report["groups"].values():
        assert 0.0 <= group["mean_tv"] <= 1.0
        assert "mean_cage_shift" in group
