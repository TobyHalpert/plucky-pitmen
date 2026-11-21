from __future__ import annotations

import probe_row0_gem_exposure as probe
from game_env import DYNAMITE, DRAGON, GEM, MineEnv


class _ChooseActionZero:
    """Small deterministic network stand-in for the instrumentation test."""

    name = "fixed-row0"

    def select_action(self, matrix, mask, rng, **kwargs):
        assert mask[0]
        return 0, 0.0, 0.0


def test_counts_row0_gem_displacement_and_same_pusher_theft():
    env = MineEnv(opponents=3)
    env.reset(seed=41)
    env.rows[0][0] = (GEM, env.rows[0][0][1])
    env.rows[1][0] = (DYNAMITE, env.rows[1][0][1])
    env.rows[2][0] = (DRAGON, env.rows[2][0][1])
    env.planning_player = 0
    for seat in (1, 2, 3):
        env.players[seat].position = 2
        env.players[seat].column = seat
        env.planned_players[seat] = False

    trace = {"game_seed": 41, "seat": 0, "offers": [], "choices": []}
    originals = probe._install_spies()
    probe.ACTIVE_TRACES[id(env)] = trace
    try:
        ctx = env.player_action_context(0)
        agent = probe.Row0AuditAgent(
            _ChooseActionZero(), seed=7, trace=trace, guard=False
        )
        choice = agent.decide(ctx)
        assert choice.action == 0
        assert trace["offers"][0]["all_active_rivals_on_deeper_cards"]
        assert trace["choices"][0]["all_active_rivals_on_deeper_cards"]
        assert not trace["choices"][0]["guard_keeps_action"]

        # The focal net enters its row-0 home Gem; P2 displaces it before reveal.
        env.step_one_player(choice.action, force_action=True)
        env.planning_player = 1
        assert 0 in env.legal_actions(1)
        env.step_one_player(0, force_action=True)
        assert trace["choices"][0]["displaced_before_reveal"]
        assert trace["choices"][0]["pusher"] == 1

        env._execute_round()
        event = trace["choices"][0]
        assert event["reveal_processed"]
        assert event["gem_resolved"]
        assert event["reveal_occupant"] == 1
        assert event["pusher_took_gem"]
        assert env.players[1].gems == 1

        # One attempt is the denominator; the same attempt completes each step.
        stats = probe._condition_stats(
            [{"offers": trace["offers"], "choices": trace["choices"]}],
            "all_active_rivals_on_deeper_cards",
        )
        assert stats["offers"] == 1
        assert stats["choices"] == 1
        assert stats["displaced_before_reveal"] == 1
        assert stats["gem_resolved"] == 1
        assert stats["displaced_and_gem_resolved"] == 1
        assert stats["pusher_took_gem"] == 1
        assert stats["guard_would_block_choice"] == 1
    finally:
        probe.ACTIVE_TRACES.pop(id(env), None)
        probe._restore_spies(originals)
