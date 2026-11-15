"""Information boundaries and learning guards for tactical branch training."""
from collections import Counter

import numpy as np
import pytest

from features import N_ACTIONS, action_mask, state_action_matrix
from game_env import MineEnv, DYNAMITE, DRAGON
from learning import PolicyNet
from train_tactical import sample_hidden, candidates


def inventory(env):
    return Counter([card for row in env.rows for card in row if card is not None]
                   + list(env.cards_remaining)
                   + [card for p in env.players for card in p.collected_cards])


def test_hidden_samples_preserve_acting_observation_and_card_inventory():
    env = MineEnv(opponents=3)
    env.reset(seed=42)
    # Transfer a real safe card into an opponent's collected hand.
    card_index = next(i for i, card in enumerate(env.cards_remaining) if card[0] != DRAGON)
    env.players[1].collected_cards.append(env.cards_remaining.pop(card_index))
    original = env.player_action_context(0)
    changed = False
    for seed in range(10):
        sampled = sample_hidden(env, 0, np.random.default_rng(seed))
        assert sampled.player_action_context(0) == original
        assert inventory(sampled) == inventory(env)
        assert all(c[0] != DRAGON for c in sampled.players[1].collected_cards)
        changed |= sampled.rows != env.rows
    assert changed
    assert env.player_action_context(0) == original


def test_tactical_fit_moves_policy_towards_target_without_touching_value():
    env = MineEnv(opponents=3)
    env.reset(seed=42)
    env.players[0].collected_cards = [(DYNAMITE, 0)] * 2
    ctx = env.player_action_context(0)
    net = PolicyNet(hidden=8, value_hidden=4, seed=1, lr=0.01)
    mask = action_mask(ctx)
    matrix = state_action_matrix(ctx)
    before = net.masked_probs(matrix, mask)
    action = int(np.flatnonzero(mask)[-1])
    target = np.zeros(N_ACTIONS)
    target[action] = 1
    values = {k: v.copy() for k, v in net.params.items() if k.startswith(('V', 'vb'))}
    losses = net.fit_tactical([(ctx, target)], epochs=20)
    assert net.masked_probs(matrix, mask)[action] > before[action]
    assert losses[-1] < losses[0]
    for key, value in values.items():
        np.testing.assert_array_equal(net.params[key], value)
    with pytest.raises(ValueError):
        net.fit_tactical([(ctx, np.full(N_ACTIONS, np.nan))])
    illegal = np.flatnonzero(~mask)
    if len(illegal):
        target[:] = 0
        target[illegal[0]] = 1
        with pytest.raises(ValueError):
            net.fit_tactical([(ctx, target)])


def test_tactical_candidates_are_legal_and_have_policy_coverage():
    env = MineEnv(opponents=3)
    env.reset(seed=42)
    env.players[env.planning_player].collected_cards = [(DYNAMITE, 0)] * 2
    ctx = env.player_action_context(env.planning_player)
    net = PolicyNet(hidden=8, seed=2)
    chosen = candidates(net, ctx)
    assert chosen == sorted(set(chosen))
    assert len(chosen) > 1
    assert all(action_mask(ctx)[a] for a in chosen)
