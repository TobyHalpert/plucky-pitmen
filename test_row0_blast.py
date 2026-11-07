"""Regression tests for outcome-labelled Row-0 cancellation training."""
from __future__ import annotations

import numpy as np

from features import action_mask, join_action, split_action
from game_env import DRAGON, LORRY, MAX_COLUMNS, PIT_CAGE
from train_row0_blast import (
    bad_action_indices,
    collect_row0,
    row0_candidates,
    sample_row0_hidden,
)
from learning import PolicyNet

NET = PolicyNet.load("checkpoints/selfplay4_last_haul.npz")


def test_row0_hidden_sampler_preserves_own_private_state_and_opponent_blast():
    positions = collect_row0(NET, 20, seed=10000, limit=1000)
    by_kind = {kind: row for row in positions for kind in [row[2]]}
    for source, seat, kind, _game_seed, _action in (
        by_kind["initial"], by_kind["follow_up"]
    ):
        before = source.players[seat]
        env = sample_row0_hidden(source, seat, np.random.default_rng(41))
        acting = env.players[seat]

        assert kind in ("initial", "follow_up")
        assert acting.position == before.position
        assert acting.score == before.score
        assert acting.blast_pending == before.blast_pending
        assert acting.collected_cards == before.collected_cards
        assert acting.blast_cards == before.blast_cards
        for other in range(4):
            if other == seat:
                continue
            dragons = sum(
                card[0] == DRAGON
                for card in (*env.players[other].collected_cards, *env.players[other].blast_cards)
            )
            assert dragons <= 3



def test_row0_candidates_exploit_announce_leave_and_safety_choices():
    source, seat, _kind, _game_seed, action = next(
        row for row in collect_row0(NET, 2, seed=10000, limit=500) if row[2] == "initial"
    )
    ctx = source.player_action_context(seat)
    mask = action_mask(ctx)
    choices = row0_candidates(NET, ctx, "initial")
    assert action in choices
    assert any(
        split_action(candidate) == (action, True)
        and 0 <= action < PIT_CAGE
        and action // MAX_COLUMNS == 0
        for candidate in choices
    )
    for candidate in choices:
        primitive, _blast = split_action(candidate)
        assert bool(mask[primitive])


def test_bad_action_labels_do_not_punish_legal_non_row0_play():
    row0_blast = join_action(1, True)
    row0_plain = join_action(2, False)
    row2_blast = join_action(2 * MAX_COLUMNS + 3, True)

    assert bad_action_indices("initial", [row0_plain, row0_blast, row2_blast]) == [1]
    assert bad_action_indices(
        "follow_up", [int(PIT_CAGE), int(LORRY), int(row0_plain)]
    ) == [0, 1]
