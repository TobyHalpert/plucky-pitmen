"""Tests for the scripted league composition in ``train_selfplay.py``.

The league is the whole curriculum a self-play learner meets, so what it
contains is a research decision, not an implementation detail: BALANCE.md §10.6
names "seed the league with ``threat`` seats" as the cheap fix for a network
that never reacts to a pending blast.  These tests pin down the two things that
make such a run interpretable:

* with the flag absent the league is *bit-identical* to the historical one (its
  random stream is untouched), so old checkpoints can be reproduced;
* with the threat flag on, that share of scripted seats really is the reaction policy;
* an optional pusher-match draw fills every non-learning seat with the adversarial pusher.
"""

from __future__ import annotations

import argparse

import numpy as np

import train_selfplay as trainer
from agents import AlwaysDisplaceAgent, NeuralAgent, ThreatAgent, make_scripted_agent
from train_selfplay import SCRIPTED_POOL, scripted_league_agent, train


def _args(**kwargs) -> argparse.Namespace:
    base = {
        "league_threat_prob": 0.0,
        "league_blast_mode": "rich",
        "league_blast_prob": 0.15,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_the_historical_league_draws_the_same_numbers_without_the_flag():
    """`--league-threat-prob 0` must not consume an extra random draw."""
    with_flag = np.random.default_rng(11)
    without_flag = np.random.default_rng(11)

    for _ in range(40):
        agent = scripted_league_agent(with_flag, _args(), game_seed=1, seat=0)
        # The historical draw: one policy index, then (only for a blasting
        # mode) one blast share.  Reproduce it by hand on a second stream.
        policy = SCRIPTED_POOL[int(without_flag.integers(len(SCRIPTED_POOL)))]
        blasts = without_flag.random() < 0.15
        expected = make_scripted_agent(
            policy, seed=1, blast="rich" if blasts else "never"
        )
        assert agent.name == expected.name
        # ``cautious`` has no blast trigger at all; compare what exists.
        assert getattr(agent, "blast", "never") == getattr(
            expected, "blast", "never"
        )

    assert with_flag.random() == without_flag.random()


def test_the_flag_seeds_the_league_with_threat_seats():
    rng = np.random.default_rng(5)
    agents = [
        scripted_league_agent(
            rng, _args(league_threat_prob=1.0), game_seed=1, seat=seat
        )
        for seat in range(20)
    ]
    assert all(isinstance(agent, ThreatAgent) for agent in agents)


def test_threat_seats_still_take_the_blasting_mode():
    """A threat seat can threaten on its own account too."""
    rng = np.random.default_rng(6)
    agent = scripted_league_agent(
        rng,
        _args(league_threat_prob=1.0, league_blast_mode="always", league_blast_prob=1.0),
        game_seed=1,
        seat=0,
    )
    assert isinstance(agent, ThreatAgent)
    assert agent.blast == "always"


def test_a_partial_share_mixes_threat_seats_into_the_pool():
    rng = np.random.default_rng(7)
    names = [
        scripted_league_agent(
            rng, _args(league_threat_prob=0.5), game_seed=1, seat=0
        ).name
        for _ in range(400)
    ]
    share = names.count("threat") / len(names)
    assert 0.4 < share < 0.6, share
    assert set(names) - {"threat"}, "the ordinary pool must survive the flag"


def test_a_train_step_runs_with_no_periodic_save(tmp_path):
    """`--save-every 0` used to divide by zero after the first generation."""
    out = tmp_path / "tiny.npz"
    summary = train(
        argparse.Namespace(
            generations=1,
            games_per_generation=1,
            players=3,
            hidden=8,
            value_hidden=4,
            lr=5e-4,
            gamma=0.99,
            entropy_coef=0.01,
            value_coef=0.5,
            epochs=1,
            batch_size=64,
            temperature=1.0,
            epsilon=0.0,
            mirror_prob=0.0,
            league_prob=0.0,
            league_size=0,
            snapshot_every=0,
            save_every=0,
            eval_every=5,
            eval_games=0,
            eval_seed=1,
            eval_opponent="rule",
            deterministic_eval=False,
            log_every=0,
            blind_blast_choice=False,
            pass_rule="n-1",
            league_blast_mode="rich",
            league_blast_prob=0.15,
            league_threat_prob=0.0,
            escape_shaping=0.0,
            solo_escape_bonus=True,
            seed=3,
            name="tiny",
            resume=None,
            out=str(out),
        )
    )
    assert out.exists()
    assert summary["generations"] == 1


def test_pusher_match_curriculum_fills_nonlearner_seats(tmp_path, monkeypatch):
    """A pusher-match draw puts all three scripted opponents on that policy."""
    original_play_game = trainer.play_game
    observed = []

    def record_agents(agents, *args, **kwargs):
        observed.append(tuple(agents))
        return original_play_game(agents, *args, **kwargs)

    monkeypatch.setattr(trainer, "play_game", record_agents)
    args = trainer.build_parser().parse_args([
        "--generations", "1",
        "--games-per-generation", "1",
        "--players", "4",
        "--hidden", "8",
        "--value-hidden", "4",
        "--mirror-prob", "0",
        "--pusher-match-prob", "1",
        "--league-prob", "0",
        "--snapshot-every", "0",
        "--save-every", "0",
        "--eval-games", "0",
        "--epsilon", "0",
        "--escape-shaping", "0",
        "--seed", "19",
        "--name", "push-curriculum-test",
        "--out", str(tmp_path / "push.npz"),
    ])
    summary = trainer.train(args)

    assert summary["pusher_match_games"] == 1
    assert len(observed) == 1
    assert sum(isinstance(agent, AlwaysDisplaceAgent) for agent in observed[0]) == 3
    assert sum(isinstance(agent, NeuralAgent) for agent in observed[0]) == 1
