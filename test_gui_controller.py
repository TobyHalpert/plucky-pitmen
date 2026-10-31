"""Headless tests for the GUI controller, including the neural policies.

The Tk layer cannot run in CI (no display), but everything that decides a move
lives in :mod:`gui_controller`, so the interesting part is testable: does the
trained network load, act, announce and resolve its own blasts, and does a game
actually finish?
"""

from __future__ import annotations

import pytest

from gui_controller import (
    GameController,
    agent_policies,
    available_checkpoints,
    resolve_checkpoint,
)
from learning import PolicyNet


@pytest.fixture()
def tiny_net(tmp_path):
    """A small checkpoint on disk, so the tests never need checkpoints/."""
    net = PolicyNet(hidden=8, value_hidden=4, seed=0, name="tiny")
    path = tmp_path / "tiny.npz"
    net.save(path)
    return path


def play_to_the_end(controller: GameController, step_limit: int = 4000):
    for step in range(step_limit):
        result = controller.step_one_player_auto()
        if result.game_over:
            return step + 1
    raise AssertionError("the game did not finish")


def test_scripted_autoplay_still_completes():
    controller = GameController(2, "simple", seed=11)
    controller.reset(2, "simple", seed=11)
    steps = play_to_the_end(controller)
    assert steps > 10
    assert controller.env.game_finished
    assert not controller.is_neural
    assert controller.policy_snapshot() == []


def test_neural_policy_loads_announces_and_finishes_at_four_players(tiny_net):
    policy = f"neural:{tiny_net}"
    controller = GameController(3, policy, seed=5)
    controller.reset(3, policy, seed=5)

    assert controller.env.n_players == 4
    assert controller.is_neural
    assert controller.agents[0] is not None
    assert all(agent is None for agent in controller.agents[1:])

    steps = play_to_the_end(controller)
    assert steps > 10
    assert controller.env.game_finished


def test_policy_snapshot_is_a_distribution_over_legal_actions(tiny_net):
    policy = f"neural:{tiny_net}"
    controller = GameController(2, policy, seed=5)
    controller.reset(2, policy, seed=5)

    snapshot = controller.policy_snapshot(limit=56)
    assert snapshot, "the opening position should be a decision point"
    total = sum(probability for _action, probability in snapshot)
    assert 0.99 <= total <= 1.01
    legal = set(controller.legal_actions())
    allowed = legal | {action + 28 for action in legal}
    assert {action for action, _probability in snapshot} <= allowed
    assert all(probability > 0 for _action, probability in snapshot)


def test_all_seats_neural_plays_itself(tiny_net):
    policy = f"neural:{tiny_net}"
    controller = GameController(3, policy, seed=2, all_seats=True)
    controller.reset(3, policy, seed=2, all_seats=True)

    assert all(agent is not None for agent in controller.agents)
    assert controller.env.opponent_policy == "simple"
    steps = play_to_the_end(controller)
    assert steps > 10
    assert controller.env.game_finished


def test_autoplay_picks_and_resolves_a_blast_card(tiny_net):
    """The viewer must never wait for a click when an AI seat is blasting."""
    policy = f"neural:{tiny_net}"
    controller = GameController(2, policy, seed=4)
    controller.reset(2, policy, seed=4)
    controller.setup_blast_test()

    # Announcing schedules the blast; the three cards are dealt when the
    # planning phase resolves (`_prepare_blasts`).
    player = controller.player
    assert player.dynamite >= 2
    player.blast_pending = True
    controller.env._prepare_blasts()
    assert len(player.blast_cards) == 3
    assert controller.pending_blast_seat() == 0

    # A blast reveals its three cards one at a time (keep two, return the
    # third), so the driver has to keep resolving until the blast is over.
    messages = []
    for _ in range(4):
        result = controller.step_one_player_auto()
        messages.append(result.message)
        if not player.blast_pending or result.game_over:
            break
    assert any(message.startswith("Blast choice:") for message in messages)
    assert controller.pending_blast_seat() is None


def test_an_announced_blast_can_fire_through_the_viewer_driver(tiny_net):
    """End-to-end: announce in a real game, then let autoplay drive to the end.

    Whether the blast actually fires depends on the round resolving before a
    dragon does, so this searches a few seeds and requires that at least one of
    them reaches the "choose one of three" state without any human input.
    """
    policy = f"neural:{tiny_net}"
    resolved = 0
    for seed in range(12):
        controller = GameController(2, policy, seed=seed)
        controller.reset(2, policy, seed=seed)
        controller.setup_blast_test()
        move = next(
            action for action in controller.legal_actions() if action < 25
        )
        controller.env.step_one_player(move, announce_blast=True)
        for _ in range(60):
            result = controller.step_one_player_auto()
            if result.message.startswith("Blast choice:"):
                resolved += 1
                break
            if result.game_over:
                break
    assert resolved >= 1


def test_autoplay_handles_a_tie_break(tiny_net):
    policy = f"neural:{tiny_net}"
    controller = GameController(2, policy, seed=6)
    controller.reset(2, policy, seed=6)
    controller.setup_tie_break_test()

    assert controller.in_tie_breaker
    before = len(controller.env.tie_breaker_contenders)
    result = controller.step_one_player_auto()
    assert "tie-break" in result.message
    assert result.game_over or len(controller.env.tie_breaker_contenders) <= before


def test_policy_list_includes_checkpoints_on_disk(tmp_path, tiny_net):
    assert available_checkpoints(tmp_path)
    policies = agent_policies(tmp_path)
    assert "simple" in policies
    assert "neural:tiny" in policies
    assert resolve_checkpoint("neural:tiny", tmp_path) == tiny_net
    assert resolve_checkpoint("simple", tmp_path) is None
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint("neural:missing", tmp_path)


def test_neural_seats_agree_bit_for_bit_on_the_same_seed(tiny_net):
    """Same seed, same game: the viewer must be reproducible while debugging."""
    policy = f"neural:{tiny_net}"

    def play():
        controller = GameController(2, policy, seed=8)
        controller.reset(2, policy, seed=8)
        play_to_the_end(controller)
        return [player.score for player in controller.env.players]

    assert play() == play()


def test_viewer_only_calls_controller_attributes_that_exist():
    """The Tk layer cannot run headless, so at least keep it in sync.

    Guards the split between ``mine_gui`` (widgets) and ``gui_controller``
    (decisions): if the viewer calls ``self.controller.<something>`` that the
    controller no longer has, this fails instead of the reader's first click.
    """
    import ast
    from pathlib import Path as _Path

    source = _Path("mine_gui.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    referenced = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
        and node.value.attr == "controller"
    }
    assert referenced, "the viewer should talk to the controller"
    controller = GameController(2, "simple", seed=1)
    controller.reset(2, "simple", seed=1)
    missing = sorted(name for name in referenced if not hasattr(controller, name))
    assert not missing, missing


def test_policy_snapshot_is_quiet_after_the_game_ends(tiny_net):
    """The viewer redraws after every step, including the last one."""
    policy = f"neural:{tiny_net}"
    controller = GameController(2, policy, seed=9)
    controller.reset(2, policy, seed=9)
    play_to_the_end(controller)
    assert controller.policy_snapshot() == []
