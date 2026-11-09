"""The colour of a dragon you have seen, in the keeper and in the policy.

Each backside colour is a ten-card set with two dragons, so a dragon a seat can
see in its own column is a dragon that is not in the draw pile.  A face-down
blast keeps two of three drawn cards and wakes the dragon if either is one, which
makes that knowledge worth a lot: one dragon seen takes a colour from 2-in-10 to
1-in-9, both take it to zero.

These tests pin the two halves separately, because only one of them is learned.
"""

from __future__ import annotations

from agents import Agent, NeuralAgent, preferred_blast_choice
from features import backside_stats
from game_env import DRAGON, DYNAMITE, FRONT_UNKNOWN, GEM, MineEnv
from probe_blast_colour import without_the_seen_dragon


def _context_with_dragons(count: int):
    env = MineEnv(opponents=2)
    env.reset(seed=4)
    for row in range(count):
        env.rows[row][0] = (DRAGON, 0)
    return env.player_action_context(0)


def test_a_seen_dragon_lowers_that_colour_s_risk():
    base = backside_stats(_context_with_dragons(0))[3][0]
    one = backside_stats(_context_with_dragons(1))[3][0]
    both = backside_stats(_context_with_dragons(2))[3][0]
    assert base > one > both == 0.0, (base, one, both)


def test_the_blind_keeper_takes_the_colour_with_no_dragons_left():
    """The rule the seat actually plays when the printed rule is on."""
    ctx = _context_with_dragons(2)
    risk = backside_stats(ctx)[3]
    assert risk[0] == 0.0 and risk[1] > 0.0 and risk[2] > 0.0
    cards = ((FRONT_UNKNOWN, 1), (FRONT_UNKNOWN, 0), (FRONT_UNKNOWN, 2))
    choice = preferred_blast_choice(cards, ctx)
    assert cards[choice][1] == 0, "it has to keep the dragon-free colour"
    # ... and the second pick of the same colour is still the safest, which is
    # how "maybe two" of that colour happens.
    assert risk[cards[choice][1]] == 0.0


def test_the_keeper_is_hand_written_not_learned():
    """Every agent, including the trained one, inherits the same keeper."""
    assert NeuralAgent.blast_choice is Agent.blast_choice


def test_peeking_overrides_the_colour_rule():
    """The engine default shows the fronts, and then a known gem wins.

    That is why the colour reasoning is invisible in a default replay: with the
    fronts visible there is no risk to price.  It only binds face down.
    """
    ctx = _context_with_dragons(2)
    cards = ((DYNAMITE, 0), (GEM, 2), (FRONT_UNKNOWN, 1))
    choice = preferred_blast_choice(cards, ctx)
    assert cards[choice] == (GEM, 2)


def test_the_counterfactual_only_rewrites_the_known_dragon():
    ctx = _context_with_dragons(1)
    doctored = without_the_seen_dragon(ctx)
    assert doctored is not None
    before = backside_stats(ctx)[3][0]
    after = backside_stats(doctored)[3][0]
    assert after > before, "turning the dragon into a gem has to make the colour riskier"
    assert len(doctored.public_rows) == len(ctx.public_rows)
    changed = [
        (row_a, row_b)
        for row_a, row_b in zip(ctx.public_rows, doctored.public_rows)
        if row_a != row_b
    ]
    assert len(changed) == 1
    row_before, row_after = changed[0]
    differing = [
        (a, b) for a, b in zip(row_before, row_after) if a != b
    ]
    assert len(differing) == 1
    (front_before, back_before), (front_after, back_after) = differing[0]
    assert front_before == DRAGON and front_after == GEM
    assert back_before == back_after, "the backside has to stay put"
