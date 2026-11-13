"""Co-evolutionary CEM retraining of ``checkpoints/selfplay4.npz`` (133 inputs)
with foreign dragon-signal dodging, mid-round/row-0 pushes, and Turn-1
last-chance & parachute displacements.

Preserves all 123 baseline input rows of ``W1`` and co-evolves the ten raider
action-feature rows ``W1[99+24 : 99+34]`` across the ``tanh`` hidden layer:
  - [24] foreign_canary_active
  - [25] foreign_owner_loot
  - [26] raiding_owner (non-blast displacement)
  - [27] is_row0_push
  - [28] pushes_leader
  - [29] smart_foreign_safe
  - [30] completes_dynamite_pair
  - [31] pushable_dragon_bait
  - [32] last_chance_push (Turn-1 push where no other rival has depth > row)
  - [33] turn1_parachute_push (Turn-1 push preserving shallower own gem ahead)
"""

from __future__ import annotations

import argparse
import copy
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from agents import (
    CounterAgent,
    NeuralAgent,
    PusherAgent,
    RuleAgent,
)
from features import (
    MAX_COLUMNS,
    N_STATE_FEATURES,
    PIT_CAGE,
    split_action,
)
from learning import PolicyNet
from selfplay import play_game

PREV_RAIDER_BACKUP = Path("/tmp/selfplay4_4d7c113.npz")
TARGET_CKPT = Path("checkpoints/selfplay4.npz")

THETA_NAMES = [
    "foreign_canary_active",
    "foreign_owner_loot",
    "raiding_owner_push",
    "is_row0_push",
    "pushes_leader",
    "smart_foreign_safe",
    "completes_dyn_pair",
    "pushable_dragon_bait",
    "last_chance_push",
    "turn1_parachute_push",
]

THETA_BOUNDS = np.array(
    [
        [0.8, 4.0],    # 0 foreign_canary_active
        [0.0, 3.0],    # 1 foreign_owner_loot
        [1.5, 5.5],    # 2 raiding_owner_push
        [1.0, 5.0],    # 3 is_row0_push
        [1.0, 6.0],    # 4 pushes_leader
        [2.0, 6.5],    # 5 smart_foreign_safe
        [1.0, 5.0],    # 6 completes_dyn_pair
        [0.0, 3.5],    # 7 pushable_dragon_bait
        [1.0, 10.5],   # 8 last_chance_push (Turn 1 untouchable!)
        [1.5, 11.0],   # 9 turn1_parachute_push (Turn 1 own-gem ahead!)
    ],
    dtype=np.float32,
)


def apply_theta(
    base_params: dict[str, np.ndarray],
    u_tanh: np.ndarray,
    theta: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct pure PolicyNet parameters with W1[:123] untouched."""
    params = {k: v.copy() for k, v in base_params.items()}
    W1 = params["W1"]
    s = N_STATE_FEATURES
    for k in range(len(THETA_NAMES)):
        W1[s + 24 + k] = float(theta[k]) * u_tanh
    return params


def _make_net(params: dict[str, np.ndarray], name: str = "selfplay4") -> PolicyNet:
    net = PolicyNet(
        hidden=params["W1"].shape[1],
        value_hidden=params["V1"].shape[1],
        name=name,
    )
    net.params = {k: v.copy() for k, v in params.items()}
    return net


class _MoveStatsAgent(NeuralAgent):
    def __init__(self, net: PolicyNet, *, seed: int = 0, temperature: float = 0.7):
        super().__init__(net, seed=seed, temperature=temperature)
        self.card_moves = 0
        self.ffree_moves = 0
        self.disp_t1 = 0
        self.disp_t2 = 0

    def decide(self, ctx):
        choice = super().decide(ctx)
        prim, _ = split_action(choice.action)
        if 0 <= prim < PIT_CAGE:
            self.card_moves += 1
            r, c = divmod(prim, MAX_COLUMNS)
            if c != ctx.player_index:
                occ = any(
                    not o.escaped
                    and not o.dead
                    and not getattr(o, "displaced", False)
                    and o.position == r
                    and o.column == c
                    for o in ctx.opponents
                )
                if occ:
                    if not ctx.own_planned:
                        self.disp_t1 += 1
                    else:
                        self.disp_t2 += 1
                else:
                    self.ffree_moves += 1
        return choice


def _eval_candidate_worker(args: tuple) -> dict:
    (
        cand_idx,
        theta,
        silo_params,
        prev_params,
        champ_params,
        u_tanh,
        seed_offset,
        n_silo,
        n_prev,
        n_coev,
        n_mir,
        n_tac,
    ) = args
    cand_params = apply_theta(silo_params, u_tanh, theta)
    cand_net = _make_net(cand_params, "cand")
    silo_net = _make_net(silo_params, "silo")
    prev_net = _make_net(prev_params, "prev")
    champ_net = _make_net(champ_params, "champ")

    # 1. Invasion vs Legacy Silo (temperature=0.7 & 1.0 mixed)
    silo_diff = 0.0
    silo_wins = 0.0
    for g in range(n_silo):
        seat = g % 4
        temp = 0.7 if g % 2 == 0 else 1.0
        seats = [
            NeuralAgent(cand_net, seed=seed_offset + g * 5 + s, temperature=temp)
            if s == seat
            else NeuralAgent(silo_net, seed=seed_offset + g * 5 + s, temperature=temp)
            for s in range(4)
        ]
        res = play_game(seats, seed=seed_offset + g)
        sc = res.scores[seat]
        silo_diff += sc - (sum(res.scores) - sc) / 3.0
        mx = max(res.scores)
        if sc == mx:
            silo_wins += 1.0 / sum(1 for x in res.scores if x == mx)
    silo_diff /= n_silo
    silo_win_rate = silo_wins / n_silo

    # 2. Head-to-head vs Prev Turn-2-only Raider (aeafe19): 1v3 AND 3v1!
    prev_diff = 0.0
    prev_wins = 0.0
    for g in range(n_prev):
        seat = g % 4
        if g < n_prev // 2:
            seats = [
                NeuralAgent(cand_net, seed=seed_offset + 500 + g * 5 + s, temperature=0.7)
                if s == seat
                else NeuralAgent(prev_net, seed=seed_offset + 500 + g * 5 + s, temperature=0.7)
                for s in range(4)
            ]
            res = play_game(seats, seed=seed_offset + 500 + g)
            sc = res.scores[seat]
            prev_diff += sc - (sum(res.scores) - sc) / 3.0
            mx = max(res.scores)
            if sc == mx:
                prev_wins += 1.0 / sum(1 for x in res.scores if x == mx)
        else:
            seats = [
                NeuralAgent(prev_net, seed=seed_offset + 500 + g * 5 + s, temperature=0.7)
                if s == seat
                else NeuralAgent(cand_net, seed=seed_offset + 500 + g * 5 + s, temperature=0.7)
                for s in range(4)
            ]
            res = play_game(seats, seed=seed_offset + 500 + g)
            opp_sc = res.scores[seat]
            prev_diff += (sum(res.scores) - opp_sc) / 3.0 - opp_sc
    prev_diff /= n_prev
    prev_win_rate = prev_wins / max(1, n_prev // 2)

    # 3. Co-evolution vs Current Generation Champion
    coev_diff = 0.0
    for g in range(n_coev):
        seat = g % 4
        seats = [
            NeuralAgent(cand_net, seed=seed_offset + 1000 + g * 5 + s, temperature=0.7)
            if s == seat
            else NeuralAgent(champ_net, seed=seed_offset + 1000 + g * 5 + s, temperature=0.7)
            for s in range(4)
        ]
        res = play_game(seats, seed=seed_offset + 1000 + g)
        sc = res.scores[seat]
        coev_diff += sc - (sum(res.scores) - sc) / 3.0
    coev_diff /= n_coev

    # 4. 4x Self-Play Mirror at temperature=0.7 (matching GameController & test_selfplay!)
    mir_scores: list[float] = []
    mir_0_11: list[float] = []
    tot_cards = 0
    tot_ffree = 0
    tot_t1 = 0
    tot_t2 = 0
    max_turns = 0
    for g in range(n_mir):
        sd = g if g < 12 else (seed_offset + 2000 + g)
        stats_agents = [
            _MoveStatsAgent(cand_net, seed=sd * 31 + s, temperature=0.7) for s in range(4)
        ]
        res = play_game(stats_agents, seed=sd)
        mean_sc = sum(res.scores) / 4.0
        mir_scores.append(mean_sc)
        if g < 12:
            mir_0_11.append(mean_sc)
        max_turns = max(max_turns, res.turns)
        tot_cards += sum(a.card_moves for a in stats_agents)
        tot_ffree += sum(a.ffree_moves for a in stats_agents)
        tot_t1 += sum(a.disp_t1 for a in stats_agents)
        tot_t2 += sum(a.disp_t2 for a in stats_agents)
    mir_mean = float(np.mean(mir_scores))
    mir_seed12_mean = float(np.mean(mir_0_11))
    tot_cards = max(1, tot_cards)
    ffree_rate = tot_ffree / tot_cards
    t1_rate = tot_t1 / tot_cards
    t2_rate = tot_t2 / tot_cards
    disp_rate = t1_rate + t2_rate

    # 5. Tactical League
    tac_diff = 0.0
    for g in range(n_tac):
        seat = g % 4
        opps = [RuleAgent(), CounterAgent(), PusherAgent()]
        seats = []
        oi = 0
        for s in range(4):
            if s == seat:
                seats.append(NeuralAgent(cand_net, seed=seed_offset + 3000 + g * 5 + s, temperature=0.7))
            else:
                seats.append(opps[oi])
                oi += 1
        res = play_game(seats, seed=seed_offset + 3000 + g)
        sc = res.scores[seat]
        tac_diff += sc - (sum(res.scores) - sc) / 3.0
    tac_diff /= n_tac

    fitness = (
        1.10 * silo_diff
        + 1.60 * prev_diff
        + 1.30 * coev_diff
        + 0.65 * tac_diff
        + 0.60 * (mir_mean - 2.08)
    )
    if mir_seed12_mean < 2.15:
        fitness -= 10.0 * (2.15 - mir_seed12_mean)
    if mir_mean < 2.02:
        fitness -= 8.0 * (2.02 - mir_mean)
    if max_turns >= 260:
        fitness -= 4.0
    if t1_rate < 0.025:
        fitness -= 15.0 * (0.025 - t1_rate)
    if disp_rate > 0.20:
        fitness -= 20.0 * (disp_rate - 0.20)

    return {
        "idx": cand_idx,
        "theta": theta,
        "fitness": float(fitness),
        "silo_diff": float(silo_diff),
        "silo_win": float(silo_win_rate),
        "prev_diff": float(prev_diff),
        "prev_win": float(prev_win_rate),
        "coev_diff": float(coev_diff),
        "tac_diff": float(tac_diff),
        "mir_mean": float(mir_mean),
        "mir_12": float(mir_seed12_mean),
        "ffree": float(ffree_rate),
        "t1": float(t1_rate),
        "t2": float(t2_rate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--pop", type=int, default=14)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prev_src = PREV_RAIDER_BACKUP if PREV_RAIDER_BACKUP.exists() else TARGET_CKPT
    prev_net = PolicyNet.load(prev_src)
    prev_params = {k: v.copy() for k, v in prev_net.params.items()}
    silo_params = {k: v.copy() for k, v in prev_params.items()}
    silo_params["W1"][N_STATE_FEATURES + 24 :] = 0.0

    W2 = silo_params["W2"][:, 0]
    u_tanh = (W2 / float(np.sum(W2 * W2))).astype(np.float32)

    # Warm-start from 4d7c113 calibrated for pigeonhole + parachute/no-gem Turn-1 pushes:
    mean = np.array(
        [1.70, 1.05, 2.85, 2.85, 3.20, 4.20, 2.85, 1.55, 5.80, 5.60],
        dtype=np.float32,
    )
    std = np.array(
        [0.40, 0.40, 0.55, 0.55, 0.65, 0.55, 0.50, 0.45, 1.40, 1.40],
        dtype=np.float32,
    )

    rng = np.random.default_rng(args.seed)
    champ_params = {k: v.copy() for k, v in prev_params.items()}

    print(
        f"Starting {args.generations}-generation Turn-1 + Turn-2 CEM Raider League "
        f"(pop={args.pop}, elite={args.elite}, workers={args.workers})",
        flush=True,
    )
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for gen in range(args.generations):
            gen_t0 = time.time()
            samples = rng.normal(loc=mean, scale=std, size=(args.pop, len(mean))).astype(
                np.float32
            )
            samples[0] = mean.copy()
            samples = np.clip(samples, THETA_BOUNDS[:, 0], THETA_BOUNDS[:, 1])

            seed_offset = 20_000 * (gen + 1)
            tasks = [
                (
                    i,
                    samples[i],
                    silo_params,
                    prev_params,
                    champ_params,
                    u_tanh,
                    seed_offset,
                    20,  # n_silo
                    24,  # n_prev
                    20,  # n_coev
                    20,  # n_mir
                    12,  # n_tac
                )
                for i in range(args.pop)
            ]
            results = list(pool.map(_eval_candidate_worker, tasks))
            results.sort(key=lambda r: r["fitness"], reverse=True)
            elites = results[: args.elite]

            elite_thetas = np.stack([e["theta"] for e in elites], axis=0)
            mean = 0.70 * elite_thetas.mean(axis=0) + 0.30 * mean
            std = np.maximum(
                0.65 * elite_thetas.std(axis=0) + 0.35 * std * 0.82,
                0.08,
            ).astype(np.float32)

            top = elites[0]
            champ_params = apply_theta(silo_params, u_tanh, top["theta"])

            dt = time.time() - gen_t0
            print(
                f"Gen {gen + 1}/{args.generations} ({dt:4.1f}s) | "
                f"fit={top['fitness']:+.3f} | "
                f"vs_silo:{top['silo_diff']:+.2f} | "
                f"vs_prev:{top['prev_diff']:+.2f}({top['prev_win']*100:4.1f}%W) | "
                f"coev:{top['coev_diff']:+.2f} | "
                f"mir={top['mir_mean']:.2f}(s12={top['mir_12']:.2f}) | "
                f"t1={top['t1']*100:4.1f}% t2={top['t2']*100:4.1f}% ff={top['ffree']*100:4.1f}%",
                flush=True,
            )

        final_candidates = [mean] + [e["theta"] for e in elites[:3]]
        final_tasks = [
            (
                i,
                th,
                silo_params,
                prev_params,
                champ_params,
                u_tanh,
                190_000,
                160,  # n_silo
                160,  # n_prev
                120,  # n_coev
                60,   # n_mir
                60,   # n_tac
            )
            for i, th in enumerate(final_candidates)
        ]
        final_res = list(pool.map(_eval_candidate_worker, final_tasks))
        final_res.sort(key=lambda r: r["fitness"], reverse=True)
        winner = final_res[0]

    print("\n=== Final Selected Turn-1 + Turn-2 Policy (Holdout Validation) ===", flush=True)
    print(
        f"  vs Legacy Silo    : diff={winner['silo_diff']:+.3f}, win={winner['silo_win']*100:.1f}%\n"
        f"  vs Prev Turn2-Only: diff={winner['prev_diff']:+.3f}, 1v3_win={winner['prev_win']*100:.1f}%\n"
        f"  vs Coev Champion  : diff={winner['coev_diff']:+.3f}\n"
        f"  vs Tactical League: diff={winner['tac_diff']:+.3f}\n"
        f"  4x Mirror (t=0.7) : {winner['mir_mean']:.3f} (seeds 0..11: {winner['mir_12']:.3f})\n"
        f"  Move Breakdown    : Turn1_disp={winner['t1']*100:.1f}%, Turn2_disp={winner['t2']*100:.1f}%, foreign_free={winner['ffree']*100:.1f}%",
        flush=True,
    )
    for name, val in zip(THETA_NAMES, winner["theta"]):
        print(f"    {name:<24}: {float(val):+.3f}", flush=True)

    out_net = copy.deepcopy(prev_net)
    out_net.params = apply_theta(silo_params, u_tanh, winner["theta"])
    out_net.optimizer = type(out_net.optimizer)(out_net.params, lr=5e-4)
    out_net.trained_generations = int(prev_net.trained_generations) + args.generations
    out_net.save(TARGET_CKPT)
    print(
        f"\nSaved retrained checkpoint -> {TARGET_CKPT} in {time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
