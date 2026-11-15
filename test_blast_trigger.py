"""Tests for the trigger trainer's mechanics, not for its result.

BALANCE §5 reports the result (a committed blast is worth zero, and the distilled
candidate fires less and plays worse). These tests pin the three things that
would silently invalidate it: the candidate pair must be one move with and
without the announcement, the harvested positions must not accumulate copies of
themselves, and the distilled targets must be legal distributions.
"""

from __future__ import annotations

import numpy as np

from features import action_mask, join_action, split_action
from game_env import DYNAMITE, GEM, MAX_COLUMNS, PIT_CAGE, MineEnv
from train_blast_trigger import HarvestEnv, build_examples, trigger_candidates

BACKSIDE_A = 0
BACKSIDE_B = 1


class UniformNet:
    """Stand-in for ``PolicyNet``: uniform over the legal set."""

    def masked_probs(self, matrix, mask, temperature: float = 1.0):
        probs = np.zeros(len(mask), dtype=np.float64)
        probs[np.asarray(mask, dtype=bool)] = 1.0 / max(int(np.sum(mask)), 1)
        return probs


def _blast_legal_env() -> MineEnv:
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    env.rows[0][0] = (GEM, BACKSIDE_A)
    env.rows[1][0] = (GEM, BACKSIDE_B)
    env.players[0].collected_cards = [(DYNAMITE, BACKSIDE_A), (DYNAMITE, BACKSIDE_B)]
    env.players[0].position = 2
    env.players[0].column = 0
    env.planning_player = 0
    return env


def test_the_candidates_are_one_move_with_and_without_the_announcement():
    env = _blast_legal_env()
    ctx = env.player_action_context(0)

    actions = trigger_candidates(UniformNet(), ctx)

    assert len(actions) == 2
    plain, announced = actions
    assert split_action(plain)[1] is False
    assert split_action(announced)[1] is True
    assert join_action(split_action(plain)[0], True) == announced
    mask = action_mask(ctx)
    assert mask[plain] and mask[announced]


def test_no_candidate_pair_without_a_card_to_step_onto():
    env = _blast_legal_env()
    # Everything within reach of a figure at row 2, in every column.
    for row in env.rows[:2]:
        for column in range(len(row)):
            row[column] = None
    assert trigger_candidates(UniformNet(), env.player_action_context(0)) == []


def test_a_harvest_snapshot_does_not_copy_the_positions_already_found():
    """Regression: deepcopying the env copied its own list of collected copies."""
    env = HarvestEnv(opponents=2, blast_choice_sees_fronts=False)
    env.reset(seed=4)
    env.rows[0][0] = (GEM, BACKSIDE_A)
    env.rows[1][0] = (GEM, BACKSIDE_B)
    env.players[0].collected_cards = [(DYNAMITE, BACKSIDE_A), (DYNAMITE, BACKSIDE_B)]
    env.players[0].position = 2
    env.players[0].column = 0
    env.planning_player = 0

    env.player_action_context(0)
    first = env.found[0][0]
    assert len(env.found) == 1
    assert first.found == [], "the snapshot must not carry the harvest list"
    # The live environment keeps its own record untouched.
    assert len(env.found) == 1


def test_distilled_targets_are_legal_distributions_on_the_candidate_pair():
    env = _blast_legal_env()
    ctx = env.player_action_context(0)
    actions = trigger_candidates(UniformNet(), ctx)
    rows = [(ctx, actions, np.array([0.30, 0.10]))]

    examples = build_examples(
        UniformNet(), rows, temperature=0.1, prior_weight=0.35
    )

    assert len(examples) == 1
    target = examples[0][1]
    mask = action_mask(examples[0][0])
    assert np.isclose(target.sum(), 1.0)
    assert np.all(target[~mask] == 0.0)
    assert target[actions[0]] > target[actions[1]], "the better branch must be preferred"
    assert target[actions[1]] > 0.0, "the prior keeps the losing branch alive"


def test_blast_variants_are_only_offered_while_two_dynamite_are_held():
    env = _blast_legal_env()
    assert trigger_candidates(UniformNet(), env.player_action_context(0))
    env.players[0].collected_cards = [(DYNAMITE, BACKSIDE_A), (GEM, BACKSIDE_B)]
    ctx = env.player_action_context(0)
    mask = action_mask(ctx)
    assert not any(split_action(int(a))[1] for a in np.flatnonzero(mask))
    assert trigger_candidates(UniformNet(), ctx) == []
    assert MAX_COLUMNS > 0 and PIT_CAGE > 0  # keep the imports honest
