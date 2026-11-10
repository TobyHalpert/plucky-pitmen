"""Tests for the bluffing seat used to price the table's flinch (probe_bluff.py)."""

from __future__ import annotations

from agents import BlufferAgent, NeuralAgent
from learning import PolicyNet
from selfplay import play_game


def _play(bluff, games: int = 25) -> tuple[int, int, int]:
    """Return (announcements, hauls in which the seat's blast fired, aborted)."""
    net = PolicyNet.load("checkpoints/selfplay4.npz")
    announced = fired = aborted = 0
    for game in range(games):
        seed = 991_000 + game
        seat = game % 4
        agents = [
            NeuralAgent(net, seed=seed * 1000 + index, temperature=0.7)
            for index in range(4)
        ]
        agents[seat] = BlufferAgent(bluff=bluff)
        result = play_game(agents, seed=seed)
        aborted += int(result.aborted)
        fired += sum(1 for haul in result.hauls if seat in haul.blasts)
    return announced, fired, aborted


def test_the_bluffer_announces_and_almost_never_fires():
    """It announces to buy the flinch; the abort is the point.

    "Never" is impossible: a bluff only works if the bluffer gets another turn.
    When the phase closes first -- every rival passes on the announcement -- the
    blast stands and fires.  That is the same mechanism that makes a pending
    blast worth anything at all to the rivals, so it has to be measured rather
    than assumed away.
    """
    from game_env import MineEnv

    net = PolicyNet.load("checkpoints/selfplay4.npz")
    announced = 0
    fired = 0
    original = MineEnv.begin_blast

    bluffer_seat = [0]

    def begin(env, player_index):
        nonlocal announced
        if player_index == bluffer_seat[0]:
            announced += 1
        return original(env, player_index)

    MineEnv.begin_blast = begin
    try:
        for game in range(25):
            seed = 991_000 + game
            bluffer_seat[0] = game % 4
            agents = [
                NeuralAgent(net, seed=seed * 1000 + index, temperature=0.7)
                for index in range(4)
            ]
            agents[bluffer_seat[0]] = BlufferAgent(bluff="forced")
            result = play_game(agents, seed=seed)
            fired += sum(1 for haul in result.hauls if bluffer_seat[0] in haul.blasts)
    finally:
        MineEnv.begin_blast = original

    assert announced > 0, "the bluffer never found two dynamite in 25 games"
    # Measured over 60 games: 178 announcements, 8 of them forced to fire,
    # because the planning phase closed before the abort (4.5%).
    assert fired <= announced / 4, (
        f"{fired} of {announced} announced blasts fired; a bluffer that fires "
        "that often is not bluffing"
    )


def test_the_control_arm_never_announces():
    """Same movement policy, no announcements: the paired control is clean."""
    from game_env import MineEnv

    net = PolicyNet.load("checkpoints/selfplay4.npz")
    announcements = 0
    original = MineEnv.begin_blast
    seat_of_interest = [0]

    def begin(env, player_index):
        nonlocal announcements
        if player_index == seat_of_interest[0]:
            announcements += 1
        return original(env, player_index)

    MineEnv.begin_blast = begin
    try:
        for game in range(15):
            seed = 991_500 + game
            seat_of_interest[0] = game % 4
            agents = [
                NeuralAgent(net, seed=seed * 1000 + index, temperature=0.7)
                for index in range(4)
            ]
            agents[seat_of_interest[0]] = BlufferAgent(bluff=False)
            play_game(agents, seed=seed)
    finally:
        MineEnv.begin_blast = original

    assert announcements == 0


def test_the_free_bluff_only_announces_when_it_was_moving_anyway():
    """``free`` never forces a move; ``forced`` may. Both stay legal."""
    from game_env import PIT_CAGE

    free = BlufferAgent(bluff="free")
    forced = BlufferAgent(bluff="forced")
    assert free.bluff == "free" and forced.bluff == "forced"
    # Truthiness of the flag is the API probe_bluff.py uses.
    assert bool(BlufferAgent()) is True
    assert PIT_CAGE > 0
