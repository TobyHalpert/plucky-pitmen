"""Last-haul curriculum generation and teacher/evaluation separation."""
from dataclasses import replace

from features import action_mask
from game_env import DYNAMITE, PASS
from learning import PolicyNet
from train_last_haul import scenario, collect_last, BlastPlanTeacher


def test_late_scenario_has_really_executed_three_rounds():
    valid = [scenario(seed, 3) for seed in range(2300000, 2300010)]
    valid = [entry for entry in valid if entry is not None]
    assert len(valid) >= 5
    for env, seat in valid:
        assert env.haul == 3 and env.round == 3
        assert len(env.rows) == 6
        assert sum(card is None for row in env.rows for card in row) == 12
        assert all(not p.dead and not p.escaped for p in env.players)
        assert all(not p.blast_pending for p in env.players)
        assert env.planning_player == env.starting_player == seat
        assert not any(env.planned_players)
        assert sum(p.score == max(q.score for q in env.players) for p in env.players) == 1


def test_curriculum_varies_scores_and_dynamite_and_keeps_seed_splits_separate():
    starts = [scenario(seed, 0)[0] for seed in range(2300000, 2300020)]
    assert len({tuple(p.score for p in env.players) for env in starts}) > 5
    assert {p.dynamite for env in starts for p in env.players} == {0, 1, 2, 4, 6}
    net = PolicyNet(hidden=8, seed=1)
    train = collect_last(net, 8, 2300000)
    held = collect_last(net, 8, 2400000)
    assert {s for _, _, s in train}.isdisjoint(s for _, _, s in held)
    assert {env.round for env, _, _ in train} == {0, 3}
    assert all(env.haul == 3 for env, _, _ in train + held)
    assert any(env.players[seat].blast_pending for env, seat, _ in train)


def test_plan_teacher_only_keeps_a_policy_allowed_pending_blast():
    env, seat = scenario(2300000, 0)
    env.players[seat].collected_cards = [(DYNAMITE, 0), (DYNAMITE, 1)]
    ctx = env.player_action_context(seat)
    ctx = replace(ctx, legal_actions=tuple(ctx.legal_actions) + (PASS,),
                  own_blast_pending=True, own_planned=True)
    assert action_mask(ctx)[PASS]
    teacher = BlastPlanTeacher(PolicyNet(hidden=8, seed=1), seed=1)
    choice = teacher.decide(ctx)
    assert choice.action == PASS and not choice.announce_blast


def test_retained_dynamite_is_not_resampled_into_gems():
    import numpy as np
    from train_last_haul import sample_last_hidden
    from test_tactical import inventory
    env, seat = scenario(2300000, 0)
    ctx = env.player_action_context(seat)
    for seed in range(5):
        sampled = sample_last_hidden(env, seat, np.random.default_rng(seed))
        assert sampled.player_action_context(seat) == ctx
        assert inventory(sampled) == inventory(env)
        for other, count in enumerate(env.tactical_retained_counts):
            assert sampled.players[other].collected_cards[:count] == env.players[other].collected_cards[:count]
