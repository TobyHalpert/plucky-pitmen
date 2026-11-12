"""Audit and benchmark foreign dragon-signal dodging and displacement play in
the shipped 131-input ``checkpoints/selfplay4.npz``.

Measures:
1. Row-0 foreign dragon signal audit: when a raider pushes a column owner off a
   row-0 dragon and the owner flees to ``PIT_CAGE`` in a calm mine, ``action_mask``
   refuses ``PASS`` and unlocks ``LORRY`` ``100%`` of the time (eliminating the
   pre-fix ``100%`` ``[PASS]`` suicide).
2. Multi-opponent tournament comparing the retrained ``selfplay4.npz`` against
   the pre-retrain silo baseline (``W1[99+24:99+32] == 0``), ``RuleAgent``,
   ``CounterAgent``, and ``PusherAgent``.

Usage:
    python probe_raider_constrictions.py --games 200
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import features
from agents import CounterAgent, NeuralAgent, PusherAgent, RuleAgent
from game_env import (
    DRAGON,
    LORRY,
    MAX_COLUMNS,
    PASS,
    PIT_CAGE,
    MineEnv,
)
from learning import PolicyNet
from selfplay import play_game


class TrackedNeuralAgent(NeuralAgent):
    """Standard ``NeuralAgent`` that records card-move selectivity."""

    def __init__(self, net: PolicyNet, *, seed: int = 0, temperature: float = 1.0):
        super().__init__(net, seed=seed, temperature=temperature)
        self.stats = {"own": 0, "ffree": 0, "disp": 0, "dodges": 0}

    def decide(self, ctx):
        if features.own_card_is_signalled_dragon(ctx):
            self.stats["dodges"] += 1
        choice = super().decide(ctx)
        prim = choice.action
        if 0 <= prim < PIT_CAGE:
            rr, cc = divmod(prim, MAX_COLUMNS)
            if cc == ctx.player_index:
                self.stats["own"] += 1
            elif features._occupant(ctx, rr, cc) not in (None, ctx.player_index):
                self.stats["disp"] += 1
            else:
                self.stats["ffree"] += 1
        return choice


def audit_row0_signal_dodge(seeds: int = 300) -> tuple[int, int, int]:
    """Verify ``action_mask`` dodges ``100%`` of signalled row-0 foreign dragons."""
    forced_suicides = 0
    lorry_unlocked = 0
    total = 0
    for seed in range(seeds):
        env = MineEnv(opponents=3)
        env.reset(seed=seed)
        if env.rows[0][1][0] != DRAGON:
            continue
        env.players[1].position, env.players[1].column, env.planned_players[1] = 0, 1, True
        env.players[2].position, env.players[2].column, env.planned_players[2] = 2, 2, True
        env.players[3].position, env.players[3].column, env.planned_players[3] = 2, 3, True
        env.planning_player = 0
        env.step_one_player(0 * MAX_COLUMNS + 1, force_action=True)
        env.step_one_player(PIT_CAGE, force_action=True)
        env.step_one_player(PASS, force_action=True)
        env.step_one_player(PASS, force_action=True)
        ctx0 = env.player_action_context(0)
        mask = features.action_mask(ctx0)
        total += 1
        if [i for i in range(len(mask)) if mask[i]] == [int(PASS)]:
            forced_suicides += 1
        if mask[int(LORRY)] and not mask[int(PASS)]:
            lorry_unlocked += 1
    return total, forced_suicides, lorry_unlocked


_SHIPPED: PolicyNet | None = None
_LEGACY: PolicyNet | None = None


def _run_one(args: tuple[str, str, int, str, float]):
    global _SHIPPED, _LEGACY
    hero_spec, opp_spec, seed, ckpt, temp = args
    if _SHIPPED is None:
        _SHIPPED = PolicyNet.load(ckpt)
        _LEGACY = copy.deepcopy(_SHIPPED)
        _LEGACY.params["W1"][features.N_STATE_FEATURES + 24 :] = 0.0
    seat = seed % 4

    def mk(spec: str, s: int):
        if spec == "shipped":
            return TrackedNeuralAgent(_SHIPPED, seed=s, temperature=temp)
        if spec == "legacy":
            return TrackedNeuralAgent(_LEGACY, seed=s, temperature=temp)
        if spec == "rule":
            return RuleAgent(blast="safe")
        if spec == "counter":
            return CounterAgent()
        if spec == "pusher":
            return PusherAgent(tell="any", before_planning=False)
        raise ValueError(spec)

    ags = [mk(opp_spec, seed * 31 + i) for i in range(4)]
    hero = mk(hero_spec, seed * 31 + seat)
    ags[seat] = hero
    res = play_game(ags, seed=seed)
    sc = res.scores[seat]
    riv = float(np.mean([res.scores[i] for i in range(4) if i != seat]))
    top = max(res.scores)
    win_sh = (1.0 / res.scores.count(top)) if sc == top else 0.0
    st = getattr(hero, "stats", {"own": 0, "ffree": 0, "disp": 0, "dodges": 0})
    return sc, sc - riv, win_sh, st


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/selfplay4.npz")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=12000)
    args = parser.parse_args()

    total, suicides, fixed = audit_row0_signal_dodge(300)
    print(
        f"Row-0 dragon push signal audit (n={total}): "
        f"forced [PASS] suicides = {suicides}/{total} ({100*suicides/max(1,total):.1f}%); "
        f"shipped action_mask dodges via LORRY = {fixed}/{total} ({100*fixed/max(1,total):.1f}%)."
    )

    matchups = [
        ("1x retrained vs 3x legacy (invade)", "shipped", "legacy"),
        ("1x legacy vs 3x retrained (defend)", "legacy", "shipped"),
        ("4x retrained mirror (equilibrium) ", "shipped", "shipped"),
        ("1x retrained vs 3x RuleAgent      ", "shipped", "rule"),
        ("1x retrained vs 3x CounterAgent   ", "shipped", "counter"),
        ("1x retrained vs 3x PusherAgent    ", "shipped", "pusher"),
    ]
    for label, hero, opp in matchups:
        tasks = [
            (hero, opp, args.seed + s, args.checkpoint, args.temperature)
            for s in range(args.games)
        ]
        with ProcessPoolExecutor(max_workers=2) as ex:
            rows = list(ex.map(_run_one, tasks))
        sc = float(np.mean([r[0] for r in rows]))
        mg = np.array([r[1] for r in rows])
        ws = float(np.mean([r[2] for r in rows]))
        own_c = sum(r[3]["own"] for r in rows)
        ff_c = sum(r[3]["ffree"] for r in rows)
        dp_c = sum(r[3]["disp"] for r in rows)
        tot_c = max(1, own_c + ff_c + dp_c)
        mix = (
            f" | own={100*own_c/tot_c:4.1f}% ffree={100*ff_c/tot_c:4.1f}% disp={100*dp_c/tot_c:4.1f}%"
            if hero == "shipped"
            else ""
        )
        print(
            f"{label}: sc={sc:.2f} | margin={np.mean(mg):+.3f}+-{np.std(mg)/np.sqrt(args.games):.3f} "
            f"| win_sh={100*ws:5.1f}%{mix}"
        )


if __name__ == "__main__":
    main()
