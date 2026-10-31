"""What does a checkpointed agent actually *do*?

The leaderboards say how often a policy wins; they do not say how it plays.
This prints the habits next to the results, so two training runs can be compared
without re-reading the code: how deep it digs, how much of its digging is the
front row it can see, how often it banks loot, how often it passes, retreats or
leaves, and how much it blasts.

    python profile_agent.py --checkpoints checkpoints/selfplay4_g060.npz,checkpoints/selfplay4.npz

Each checkpoint plays one seat against ``--opponents`` scripted seats, seats
rotated, so the habits are not confounded with a lucky seat.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field


from agents import NeuralAgent, _card_actions, make_scripted_agent, owns_dynamite_pair
from evaluate_balance import summarise
from features import N_BASE_ACTIONS, digs_face_up_dragon
from game_env import LORRY, MAX_COLUMNS, PASS, PIT_CAGE
from learning import PolicyNet
from selfplay import play_series


@dataclass
class Habits:
    """Per-decision counters for one seat."""

    decisions: int = 0
    digs: int = 0
    front_row_digs: int = 0
    own_column_digs: int = 0
    known_dragon_digs: int = 0
    passes: int = 0
    cage_retreats: int = 0
    lorry_exits: int = 0
    announcements: int = 0
    blastable: int = 0
    rows_sum: int = 0
    depth_sum: int = 0
    haul_rows: int = 0

    def add(self, ctx, choice) -> None:
        primitive = int(choice.action)
        announces = bool(choice.announce_blast)
        self.decisions += 1
        self.announcements += int(announces)
        self.blastable += int(
            owns_dynamite_pair(ctx) and bool(_card_actions(ctx))
            and not ctx.own_blast_pending
        )
        rows = len(ctx.public_rows)
        self.haul_rows += rows
        if ctx.own_position not in (PIT_CAGE, LORRY):
            self.depth_sum += int(ctx.own_position)
        if primitive == PASS:
            self.passes += 1
            return
        if primitive == PIT_CAGE:
            self.cage_retreats += 1
            return
        if primitive == LORRY:
            self.lorry_exits += 1
            return
        if primitive >= N_BASE_ACTIONS:
            return
        row, column = divmod(int(primitive), MAX_COLUMNS)
        self.digs += 1
        self.rows_sum += row
        self.front_row_digs += int(row == rows - 1)
        self.own_column_digs += int(column == ctx.player_index)
        # Only the player's own column shows fronts, so only a dig there can be
        # a dragon the player can *see*.
        self.known_dragon_digs += int(digs_face_up_dragon(ctx, primitive))


@dataclass
class Profile:
    habits: Habits = field(default_factory=Habits)
    blasts_fired: int = 0
    blast_dragons: int = 0
    dragons: int = 0
    revealed: int = 0
    hauls: int = 0
    rounds: int = 0


def profile_checkpoint(path: str, games: int, players: int, seed: int, opponents: str) -> dict:
    net = PolicyNet.load(path)
    profile = Profile()

    def factory(agent_seed: int):
        inner = NeuralAgent(net, seed=agent_seed, temperature=0.7, name="agent")

        class Watched:
            name = "agent"

            def decide(self, ctx):
                choice = inner.decide(ctx)
                profile.habits.add(ctx, choice)
                return choice

            def wants_blast(self, ctx):
                return inner.wants_blast(ctx)

            def blast_choice(self, cards, ctx):
                return inner.blast_choice(cards, ctx)

            def tie_break(self, ctx):
                return inner.tie_break(ctx)

        return Watched()

    others = [
        (lambda agent_seed: make_scripted_agent(opponents, seed=agent_seed))
        for _ in range(players - 1)
    ]
    series = play_series([factory, *others], games, seed=seed, rotate_seats=True)
    stats = summarise(series, path.split("/")[-1])

    for game in series:
        seat = game.policy_of_seat.index(0)
        for haul in game.result.hauls:
            profile.hauls += 1
            profile.revealed += int(haul.dragon_player == seat)
            profile.rounds += int(getattr(haul, "rounds", 0) or 0)
            profile.blasts_fired += len(haul.blasts)
            profile.blast_dragons += int(haul.outcome == "blast_dragon")
            profile.dragons += int("dragon" in haul.outcome)

    h = profile.habits
    return {
        "stats": stats,
        "games": games,
        "decisions": h.decisions / games,
        "digs": h.digs / games,
        "front_row_share": h.front_row_digs / h.digs if h.digs else 0.0,
        "own_column_share": h.own_column_digs / h.digs if h.digs else 0.0,
        "known_dragon_digs": h.known_dragon_digs / games,
        "mean_dig_row": h.rows_sum / h.digs if h.digs else 0.0,
        "mean_board_rows": h.haul_rows / h.decisions if h.decisions else 0.0,
        "passes": h.passes / games,
        "cage_retreats": h.cage_retreats / games,
        "lorry_exits": h.lorry_exits / games,
        "announcements": h.announcements / games,
        "blastable": h.blastable / games,
        "blasts_fired": profile.blasts_fired / games,
        "blast_dragons": profile.blast_dragons / games,
        "dragons": profile.dragons / profile.hauls if profile.hauls else 0.0,
        "revealed": profile.revealed / games,
        "rounds": profile.rounds / profile.hauls if profile.hauls else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--opponents", default="rule")
    args = parser.parse_args()

    paths = [part.strip() for part in args.checkpoints.split(",") if part.strip()]
    rows = [profile_checkpoint(path, args.games, args.players, args.seed, args.opponents)
            for path in paths]

    print(f"{args.players} players, {args.games} games per checkpoint, one seat each, "
          f"seats rotated, opponents = {args.opponents}.\n")
    print("| checkpoint | win % | margin | score | escapes (0 gems) | gems/escape | "
          "points when rivals bank loot / when they stay |")
    print("| :-- | --: | --: | --: | --: | --: | :-- |")
    for path, row in zip(paths, rows):
        s = row["stats"]
        print(
            f"| {path.split('/')[-1]} | {100 * s.wins / s.games:.0f}% | "
            f"{s.mean_margin:+.2f} | {s.mean_score:+.2f} | "
            f"{100 * s.escape_rate:.0f}% ({100 * s.zero_gem_escape_rate:.0f}%) | "
            f"{s.mean_gems_on_escape:.2f} | "
            f"{s.points_when_rivals_loot:+.2f} / {s.points_when_rivals_stay:+.2f} |"
        )

    print("\n| habit | " + " | ".join(f"{p.split('/')[-1]}" for p in paths) + " |")
    print("| :-- | " + " | ".join("--:" for _ in paths) + " |")
    habits = [
        ("planning decisions / game", lambda r: f"{r['decisions']:.0f}"),
        ("digs / game", lambda r: f"{r['digs']:.1f}"),
        ("...into the front row", lambda r: f"{100 * r['front_row_share']:.0f}%"),
        ("...into its own column", lambda r: f"{100 * r['own_column_share']:.0f}%"),
        ("...into a face-up dragon", lambda r: f"{r['known_dragon_digs']:.2f}"),
        ("mean row dug (0 = top)", lambda r: f"{r['mean_dig_row']:.2f}"),
        ("mean rows on the board", lambda r: f"{r['mean_board_rows']:.1f}"),
        ("passes / game", lambda r: f"{r['passes']:.2f}"),
        ("pit-cage retreats / game", lambda r: f"{r['cage_retreats']:.2f}"),
        ("lorry exits / game", lambda r: f"{r['lorry_exits']:.2f}"),
        ("decisions where a blast was possible", lambda r: f"{r['blastable']:.2f}"),
        ("blasts announced / game", lambda r: f"{r['announcements']:.2f}"),
        ("blasts fired / game", lambda r: f"{r['blasts_fired']:.2f}"),
        ("blasts that woke a dragon", lambda r: f"{r['blast_dragons']:.2f}"),
        ("hauls ended by a dragon", lambda r: f"{100 * r['dragons']:.0f}%"),
        ("dragon revealed by this seat", lambda r: f"{r['revealed']:.2f}"),
        ("rounds per haul", lambda r: f"{r['rounds']:.1f}"),
    ]
    for label, getter in habits:
        print(f"| {label} | " + " | ".join(getter(row) for row in rows) + " |")


if __name__ == "__main__":
    main()
