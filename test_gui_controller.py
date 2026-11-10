"""Headless tests for the GUI controller, including the neural policies.

The Tk layer cannot run in CI (no display), but everything that decides a move
lives in :mod:`gui_controller`, so the interesting part is testable: does the
trained network load, act, announce and resolve its own blasts, and does a game
actually finish?
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from features import N_INPUT_FEATURES, join_action
from gui_controller import (
    GameController,
    agent_policies,
    available_checkpoints,
    checkpoint_hint,
    checkpoint_problem,
    parse_policy,
    resolve_checkpoint,
    stale_checkpoints,
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
    from features import N_BASE_ACTIONS

    legal = set(controller.legal_actions())
    allowed = legal | {action + N_BASE_ACTIONS for action in legal}
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


def test_the_viewer_names_who_plays_each_seat(tiny_net):
    """"P3 walks to the cage" is unreadable unless the table says who P3 is.

    The viewer is a mixed table by default: the chosen policy drives P1 only and
    the rest are the environment's built-in opponent.  Reading a rule bot's move
    as the trained network's is exactly how a fixed behaviour keeps getting
    reported, so the seat list carries the policy.
    """
    policy = f"neural:{tiny_net}"
    mixed = GameController(3, policy, seed=7)
    assert mixed.seat_policies() == [policy, "simple", "simple", "simple"]

    seated = GameController(3, policy, seed=7, all_seats=True)
    assert seated.seat_policies() == [policy] * 4

    scripted = GameController(3, "simple", seed=7)
    assert scripted.seat_policies() == ["simple"] * 4


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


def test_the_viewer_says_how_to_train_a_network_when_there_is_none(tmp_path):
    """``checkpoints/`` is git-ignored, so a fresh clone offers no ``neural:`` entry.

    The dropdown then simply has no network in it, and "there is no
    neural:selfplay4 to choose" is the expected outcome rather than a bug. The
    viewer has to say what to run instead of leaving the user guessing.
    """
    assert checkpoint_hint(tmp_path) is not None
    hint = checkpoint_hint(tmp_path)
    assert "train_selfplay.py" in hint and "checkpoints/selfplay4.npz" in hint
    assert not [name for name in agent_policies(tmp_path) if name.startswith("neural:")]


def test_a_usable_checkpoint_silences_the_hint(tiny_net):
    directory = tiny_net.parent
    assert checkpoint_hint(directory) is None
    assert available_checkpoints(directory) == [tiny_net]
    assert checkpoint_problem(tiny_net) is None
    assert stale_checkpoints(directory) == []


def test_a_checkpoint_from_an_older_feature_layout_is_named_with_its_fix(tmp_path):
    """Offering it in the dropdown only to fail on "New game" wastes the click."""
    stale = tmp_path / "selfplay4_legacy.npz"
    np.savez(
        stale,
        meta=json.dumps({"hidden": 8, "value_hidden": 4, "name": "legacy"}),
        W1=np.zeros((N_INPUT_FEATURES - 4, 8), dtype=np.float32),
    )

    problem = checkpoint_problem(stale)
    assert problem is not None
    assert "migrate_checkpoint.py" in problem and "selfplay4_legacy.npz" in problem
    assert "neural:selfplay4_legacy" not in agent_policies(tmp_path)
    assert stale_checkpoints(tmp_path) == [problem]


def test_every_seat_ahead_of_p1_answers_through_its_own_agent(tiny_net):
    """The starting seat used to be played by the environment's built-in policy.

    ``reset`` steps the seats ahead of P1 so the viewer opens on P1's decision.
    That loop used ``env.step_one_player()`` with no action, which hands the turn
    to the environment's ``simple`` bot — so in "AI in every seat" mode the first
    move of a game came from a bot while every other move came from the network.
    Any call into the built-in policy from that loop is now an error.
    """
    policy = f"neural:{tiny_net}"
    for seed in range(6):
        controller = GameController(3, policy, seed)
        controller.reset(3, policy, seed, all_seats=True)  # loads the four seats

        def refuse(*_args, **_kwargs):
            raise AssertionError("the environment played a seat the controller owns")

        controller.env._opponent_action = refuse
        controller.env._agent_action = refuse
        controller.reset(3, policy, seed, all_seats=True)


def test_the_command_line_accepts_a_checkpoint_that_exists(tiny_net):
    assert parse_policy("neural:tiny", tiny_net.parent) == "neural:tiny"
    # A bare path is the same request, and is normalised to the viewer's form.
    assert parse_policy(str(tiny_net)) == f"neural:{tiny_net}"
    assert parse_policy("simple") == "simple"


def test_the_command_line_explains_a_missing_checkpoint_instead_of_listing_names(tmp_path):
    """"invalid choice: neural:selfplay4" is the message this replaces."""
    with pytest.raises(argparse.ArgumentTypeError) as caught:
        parse_policy("neural:selfplay4", tmp_path)
    message = str(caught.value)
    assert "no checkpoint named 'selfplay4'" in message
    assert "train_selfplay.py" in message, "it has to say how to get one"

    with pytest.raises(argparse.ArgumentTypeError) as caught:
        parse_policy("nonsense")
    assert "unknown policy 'nonsense'" in caught.value.args[0]
    assert "neural:<name>" in caught.value.args[0]


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


def test_deterministic_mode_takes_the_argmax(tiny_net):
    """Feedback runs should show the policy, not one sample of it."""
    policy = f"neural:{tiny_net}"
    controller = GameController(2, policy, seed=3, deterministic=True)
    controller.reset(2, policy, seed=3, deterministic=True)

    checked = 0
    for _ in range(400):
        if (
            not controller.in_tie_breaker
            and not controller.blast_pending
            and controller.env.planning_player == 0
        ):
            expected = controller.policy_snapshot(limit=1)[0][0]
            result = controller.step_one_player_auto()
            if controller.last_choice_seat == 0 and controller.last_choice is not None:
                # The snapshot names the action the way the policy does (a blast
                # is a separate index); the choice carries the blast as a flag.
                chosen = join_action(
                    int(controller.last_choice.action),
                    bool(controller.last_choice.announce_blast),
                )
                assert chosen == expected
                checked += 1
        else:
            result = controller.step_one_player_auto()
        if result.game_over:
            break
    assert checked >= 3


def test_deterministic_mode_is_reproducible(tiny_net):
    policy = f"neural:{tiny_net}"

    def play() -> list[int]:
        controller = GameController(2, policy, seed=8, deterministic=True)
        controller.reset(2, policy, seed=8, deterministic=True)
        play_to_the_end(controller)
        return [player.score for player in controller.env.players]

    assert play() == play()


def test_tie_break_falls_back_when_the_requested_pile_is_empty(tiny_net):
    """A tie-break draw the agent cannot make must not freeze the viewer.

    With four neural seats in the mine, both piles can run out at once: the
    contenders are all escaped, the mine is empty, and the agent still asks for
    a card. Before the fallback the driver retried the same illegal draw
    forever, which looked like a hang in the window.
    """
    policy = f"neural:{tiny_net}"
    controller = GameController(3, policy, seed=1, all_seats=True)
    controller.reset(3, policy, seed=1, all_seats=True)
    controller.env.tie_breaker_contenders = [0, 1]
    controller.env.tie_breaker_turn = 0
    controller.env.cards_remaining = []

    class MineOnly:
        """Always asks for a card the mine no longer has."""

        def tie_break(self, ctx):
            return ("mine", 0, 0)

        def decide(self, ctx):  # pragma: no cover - not reached here
            raise AssertionError("no planning steps expected")

    controller.agents[0] = MineOnly()
    result = controller.step_one_player_auto()
    assert "tie-break" in result.message or result.game_over


def test_blaster_policy_announces_and_fires_blasts():
    """The viewer needs a baseline that actually blasts; the learner does not."""
    announced = fired = 0
    for seed in range(6):
        controller = GameController(3, "blaster", seed=seed)
        controller.reset(3, "blaster", seed=seed)
        for _ in range(4000):
            before = sum(len(p.blast_cards) for p in controller.env.players)
            result = controller.step_one_player_auto()
            if controller.last_choice is not None and controller.last_choice.announce_blast:
                announced += 1
            if sum(len(p.blast_cards) for p in controller.env.players) > before:
                fired += 1
            if result.game_over:
                break
        else:  # pragma: no cover - the driver must terminate
            raise AssertionError("the game did not finish")
    assert announced > 0
    assert fired > 0


def test_every_seat_neural_finishes_across_seeds(tiny_net):
    """Four neural seats must always reach the end of the third haul.

    Two seeds used to stall at the game-ending tie-break; a hung window is the
    worst failure mode this viewer has, so this walks a handful of seeds.
    """
    policy = f"neural:{tiny_net}"
    for seed in range(8):
        controller = GameController(3, policy, seed=seed, all_seats=True)
        controller.reset(3, policy, seed=seed, all_seats=True)
        steps = play_to_the_end(controller)
        assert steps > 0
        assert controller.env.game_finished


def test_the_watchable_blast_rules_all_finish_and_trap_blasts_less():
    """"Blast when a dragon is waiting for me" must be watchable, not a hang.

    It is a losing rule (BALANCE.md §5), which is exactly why the viewer offers
    it: the announced counts should sit between never and always.
    """
    announced = {}
    for policy in ("blaster:behind", "blaster:trap", "blaster"):
        announced[policy] = 0
        for seed in range(4):
            controller = GameController(3, policy, seed=seed)
            controller.reset(3, policy, seed=seed)
            for _ in range(4000):
                result = controller.step_one_player_auto()
                if controller.last_choice is not None and controller.last_choice.announce_blast:
                    announced[policy] += 1
                if result.game_over:
                    break
            else:  # pragma: no cover - the driver must terminate
                raise AssertionError("the game did not finish")
    assert announced["blaster"] > announced["blaster:trap"] > 0


def test_the_viewer_names_what_the_deciding_seat_knows(tiny_net):
    """Every offered move carries a reason, and the standing plan says if it is locked."""
    from game_env import BACKSIDE_A, MAX_COLUMNS

    policy = f"neural:{tiny_net}"
    controller = GameController(3, policy, seed=7)
    controller.reset(3, policy, seed=7)

    snapshot = controller.policy_snapshot(limit=6)
    assert snapshot
    reasons = [controller.reason_for(action) for action, _probability in snapshot]
    assert all(reason for reason in reasons)
    assert any("own column" in reason or "back " in reason for reason in reasons)

    plan = controller.plan_summary(0)
    assert "P1:" in plan
    assert "tentative" in plan or "locked" in plan

    # The reason has to describe the *deciding* seat's knowledge, not seat 0's.
    card = 0 * MAX_COLUMNS + 2
    controller.env.rows[0][2] = (None, BACKSIDE_A)
    assert "back " in controller.reason_for(card, 2)


def test_a_deep_mine_still_offers_its_front_rows_to_the_viewer():
    """The action grid must cover every row a deck can deal (BALANCE 7)."""
    from game_env import BACKSIDE_A, MAX_COLUMNS, PIT_CAGE

    controller = GameController(3, "simple", seed=3)
    controller.reset(3, "simple", seed=3)
    controller.env.rows = [
        [(None, BACKSIDE_A) for _ in range(4)] for _ in range(6)
    ]
    for player in controller.env.players:
        player.position = 6
        player.column = None

    legal = controller.legal_actions()
    front_row = [action for action in legal if action // MAX_COLUMNS == 0]
    assert front_row, "the front row must be nameable"
    assert max(action // MAX_COLUMNS for action in legal if action < PIT_CAGE) == 5


def test_dropdown_labels_a_legal_push_as_a_push(tmp_path):
    """Occupied cards must not look like ordinary destinations."""
    pytest.importorskip("tkinter")
    from game_env import BACKSIDE_A, GEM, MAX_COLUMNS
    from mine_gui import MineViewer

    controller = GameController(3, "simple", seed=1)
    controller.reset(3, "simple", seed=1)
    controller.env.rows[0][1] = (GEM, BACKSIDE_A)
    controller.env.players[0].position = 2
    controller.env.players[0].column = None
    controller.env.players[1].position = 0
    controller.env.players[1].column = 1

    action = 0 * MAX_COLUMNS + 1
    assert action in controller.legal_actions()
    assert controller.card_action_label(action).startswith("Displace Player 2")

    # An occupant not standing in its own column cannot be displaced, and the
    # card stays occupied, so it is not a legal destination at all.  Note the
    # occupant must be a *different* seat: column 1 belongs to P2, so a
    # non-displaceable figure there has to be P3 raiding it.  Moving P2 itself
    # to column 0 would simply vacate the card and make it a free-card move.
    controller.env.players[1].position = 5
    controller.env.players[1].column = None
    controller.env.players[2].position = 0
    controller.env.players[2].column = 1

    assert controller.env._card_occupant(0, 1) == 2
    assert action not in controller.legal_actions()
    assert not controller.card_action_label(action).startswith("Displace")

    choices = [("Displace Player 2: card row 0, column 2", action, False)]

    class Menu:
        def __init__(self):
            self.values = []
            self.value = ""
        def configure(self, **kwargs):
            self.values = kwargs["values"]
        def get(self):
            return self.value
        def set(self, value):
            self.value = value

    class IntVar:
        def __init__(self, value=0):
            self._value = value
        def set(self, value):
            self._value = value
        def get(self):
            return self._value

    viewer = MineViewer.__new__(MineViewer)
    viewer.action_menu = Menu()
    viewer.action_var = IntVar(-1)
    viewer.blast_requested = False
    viewer._action_choice_by_label = {}
    MineViewer._apply_action_choices(viewer, choices)
    assert viewer.action_var.get() == action
    assert viewer.blast_requested is False

    # A displayed-but-no-longer-offered entry must not keep its hidden action.
    viewer.action_menu.set("row 9, column 9")
    viewer._update_action_options = lambda preferred_label=None: MineViewer._apply_action_choices(
        viewer, [("Pass", action, False)], preferred_label=preferred_label
    )
    MineViewer._action_label_selected(viewer, None)
    assert viewer.action_var.get() == action
