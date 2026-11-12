"""Co-evolutionary CEM retraining of ``checkpoints/selfplay4.npz`` with unlocked
foreign-column probing, row-0/leader displacement, and dragon-signal dodging.

Preserves all 123 baseline input rows of ``W1`` (keeping own-column card choice,
cage/lorry/pass calibration, and blast-risk reading intact) and co-evolves the
eight new action-feature rows ``W1[99+24 : 99+32]`` across the ``tanh`` hidden
layer against:
  - Invasion: 1x Candidate vs 3x Baseline ``selfplay4``
  - Defense : 3x Candidate vs 1x Baseline ``selfplay4``
  - Self-play Nash co-evolution: 1x Candidate vs 3x Current Champion
  - 4x Candidate Self-Play Mirror (enforcing mean table score >= 2.15)
  - Tactical League: ``[RuleAgent, CounterAgent, PusherAgent]``
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

BASELINE_BACKUP = Path("/tmp/selfplay4_131_baseline.npz")
TARGET_CKPT = Path("checkpoints/selfplay4.npz")

# Theta parameterizes the 8 new action-feature rows W1[99+24 : 99+32] along the
# tanh output direction u = W2 / ||W2||^2:
#   0: [24] foreign_canary_active
#   1: [25] foreign_owner_loot
#   2: [26] raiding_owner (non-blast displacement)
#   3: [27] is_row0_push
#   4: [28] pushes_leader
#   5: [29] smart_foreign_safe
#   6: [30] completes_dynamite_pair
#   7: [31] pushable_dragon_bait
THETA_NAMES = [
    "foreign_canary_active",
    "foreign_owner_loot",
    "raiding_owner_push",
    "is_row0_push",
    "pushes_leader",
    "smart_foreign_safe",
    "completes_dyn_pair",
    "pushable_dragon_bait",
]

THETA_BOUNDS = np.array(
    [
        [0.5, 5.0],   # 0 foreign_canary_active
        [0.0, 3.5],   # 1 foreign_owner_loot
        [1.5, 6.5],   # 2 raiding_owner_push
        [0.5, 5.0],   # 3 is_row0_push
        [0.5, 6.5],   # 4 pushes_leader
        [1.5, 7.0],   # 5 smart_foreign_safe
        [0.5, 5.0],   # 6 completes_dyn_pair
        [0.0, 4.0],   # 7 pushable_dragon_bait
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
    for k in range(8):
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
    def __init__(self, net: PolicyNet, *, seed: int = 0):
        super().__init__(net, seed=seed)
        self.card_moves = 0
        self.ffree_moves = 0
        self.disp_moves = 0

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
                    self.disp_moves += 1
                else:
                    self.ffree_moves += 1
        return choice


def _eval_candidate_worker(args: tuple) -> dict:
    (
        cand_idx,
        theta,
        base_params,
        champ_params,
        u_tanh,
        seed_offset,
        n_inv,
        n_def,
        n_coev,
        n_mir,
        n_tac,
    ) = args
    cand_params = apply_theta(base_params, u_tanh, theta)
    cand_net = _make_net(cand_params, "cand")
    old_net = _make_net(base_params, "old")
    champ_net = _make_net(champ_params, "champ")

    # 1. Invasion: 1x Candidate vs 3x Old selfplay4
    inv_diff = 0.0
    inv_wins = 0.0
    for g in range(n_inv):
        seat = g % 4
        seats = [
            NeuralAgent(cand_net, seed=seed_offset + g * 5 + s)
            if s == seat
            else NeuralAgent(old_net, seed=seed_offset + g * 5 + s)
            for s in range(4)
        ]
        res = play_game(seats, seed=seed_offset + g)
        sc = res.scores[seat]
        inv_diff += sc - (sum(res.scores) - sc) / 3.0
        mx = max(res.scores)
        if sc == mx:
            inv_wins += 1.0 / sum(1 for x in res.scores if x == mx)
    inv_diff /= n_inv
    inv_win_rate = inv_wins / n_inv

    # 2. Defense: 3x Candidate vs 1x Old selfplay4
    def_diff = 0.0
    for g in range(n_def):
        seat = g % 4
        seats = [
            NeuralAgent(old_net, seed=seed_offset + 500 + g * 5 + s)
            if s == seat
            else NeuralAgent(cand_net, seed=seed_offset + 500 + g * 5 + s)
            for s in range(4)
        ]
        res = play_game(seats, seed=seed_offset + 500 + g)
        opp_sc = res.scores[seat]
        def_diff += (sum(res.scores) - opp_sc) / 3.0 - opp_sc
    def_diff /= n_def

    # 3. Co-evolution: 1x Candidate vs 3x Current Champion
    coev_diff = 0.0
    coev_wins = 0.0
    for g in range(n_coev):
        seat = g % 4
        seats = [
            NeuralAgent(cand_net, seed=seed_offset + 1000 + g * 5 + s)
            if s == seat
            else NeuralAgent(champ_net, seed=seed_offset + 1000 + g * 5 + s)
            for s in range(4)
        ]
        res = play_game(seats, seed=seed_offset + 1000 + g)
        sc = res.scores[seat]
        coev_diff += sc - (sum(res.scores) - sc) / 3.0
        mx = max(res.scores)
        if sc == mx:
            coev_wins += 1.0 / sum(1 for x in res.scores if x == mx)
    coev_diff /= n_coev
    coev_win_rate = coev_wins / n_coev

    # 4. 4x Candidate Self-Play Mirror (first 12 games match test_selfplay.py:2001!)
    mir_scores: list[float] = []
    mir_0_11: list[float] = []
    tot_cards = 0
    tot_ffree = 0
    tot_disp = 0
    max_turns = 0
    for g in range(n_mir):
        sd = g if g < 12 else (seed_offset + 2000 + g)
        stats_agents = [
            _MoveStatsAgent(cand_net, seed=sd * 31 + s) for s in range(4)
        ]
        for a in stats_agents:
            a.temperature = 0.7
        res = play_game(stats_agents, seed=sd)
        mean_sc = sum(res.scores) / 4.0
        mir_scores.append(mean_sc)
        if g < 12:
            mir_0_11.append(mean_sc)
        max_turns = max(max_turns, res.turns)
        tot_cards += sum(a.card_moves for a in stats_agents)
        tot_ffree += sum(a.ffree_moves for a in stats_agents)
        tot_disp += sum(a.disp_moves for a in stats_agents)
    mir_mean = float(np.mean(mir_scores))
    mir_seed12_mean = float(np.mean(mir_0_11))
    tot_cards = max(1, tot_cards)
    ffree_rate = tot_ffree / tot_cards
    disp_rate = tot_disp / tot_cards

    # 5. Tactical League: Candidate vs [RuleAgent, CounterAgent, PusherAgent]
    tac_diff = 0.0
    for g in range(n_tac):
        seat = g % 4
        opps = [RuleAgent(), CounterAgent(), PusherAgent()]
        seats = []
        oi = 0
        for s in range(4):
            if s == seat:
                seats.append(NeuralAgent(cand_net, seed=seed_offset + 3000 + g * 5 + s))
            else:
                seats.append(opps[oi])
                oi += 1
        res = play_game(seats, seed=seed_offset + 3000 + g)
        sc = res.scores[seat]
        tac_diff += sc - (sum(res.scores) - sc) / 3.0
    tac_diff /= n_tac

    # Balanced Nash + League Fitness with strict mirror-quality floor (>= 2.15)
    fitness = (
        1.25 * inv_diff
        + 1.25 * (inv_win_rate - 0.25)
        + 0.90 * def_diff
        + 1.35 * coev_diff
        + 1.35 * (coev_win_rate - 0.25)
        + 0.65 * tac_diff
        + 0.55 * (mir_mean - 2.15)
    )
    if mir_seed12_mean < 2.15:
        fitness -= 8.0 * (2.15 - mir_seed12_mean)
    if mir_mean < 2.12:
        fitness -= 7.0 * (2.12 - mir_mean)
    if max_turns >= 260:
        fitness -= 4.0
    # Selectivity band: meaningful displacement & canary raiding without chaos
    if disp_rate < 0.05:
        fitness -= 15.0 * (0.05 - disp_rate)
    elif disp_rate > 0.17:
        fitness -= 20.0 * (disp_rate - 0.17)
    if ffree_rate > 0.11:
        fitness -= 15.0 * (ffree_rate - 0.11)

    return {
        "idx": cand_idx,
        "theta": theta,
        "fitness": float(fitness),
        "inv_diff": float(inv_diff),
        "inv_win": float(inv_win_rate),
        "def_diff": float(def_diff),
        "coev_diff": float(coev_diff),
        "coev_win": float(coev_win_rate),
        "tac_diff": float(tac_diff),
        "mir_mean": float(mir_mean),
        "mir_12": float(mir_seed12_mean),
        "ffree": float(ffree_rate),
        "disp": float(disp_rate),
        "max_turns": int(max_turns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--pop", type=int, default=14)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = BASELINE_BACKUP if BASELINE_BACKUP.exists() else TARGET_CKPT
    base_net = PolicyNet.load(src)
    base_params = {k: v.copy() for k, v in base_net.params.items()}
    W2 = base_params["W2"][:, 0]
    u_tanh = (W2 / float(np.sum(W2 * W2))).astype(np.float32)

    # Selective warm start below calib_A ([3.2, 1.0, 5.2, 2.5, 4.0, 6.2, 3.0, 1.5])
    # so 4x mirror stays >= 2.15 while 1v3 & 3v1 crush Old selfplay4:
    mean = np.array(
        [
            2.30,  # 0 foreign_canary_active
            0.90,  # 1 foreign_owner_loot
            4.10,  # 2 raiding_owner_push
            2.10,  # 3 is_row0_push
            3.80,  # 4 pushes_leader
            4.90,  # 5 smart_foreign_safe
            2.90,  # 6 completes_dyn_pair
            1.70,  # 7 pushable_dragon_bait
        ],
        dtype=np.float32,
    )
    std = np.array(
        [0.65, 0.50, 0.80, 0.70, 0.85, 0.80, 0.65, 0.65],
        dtype=np.float32,
    )

    rng = np.random.default_rng(args.seed)
    champ_params = {k: v.copy() for k, v in base_params.items()}

    print(
        f"Starting {args.generations}-generation CEM Raider League "
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

            seed_offset = 10_000 * (gen + 1)
            tasks = [
                (
                    i,
                    samples[i],
                    base_params,
                    champ_params,
                    u_tanh,
                    seed_offset,
                    20,  # n_inv
                    16,  # n_def
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
            champ_params = apply_theta(base_params, u_tanh, top["theta"])

            dt = time.time() - gen_t0
            print(
                f"Gen {gen + 1}/{args.generations} ({dt:4.1f}s) | "
                f"fit={top['fitness']:+.3f} | "
                f"1v3:{top['inv_diff']:+.2f}({top['inv_win']*100:4.1f}%W) | "
                f"3v1:{top['def_diff']:+.2f} | "
                f"coev:{top['coev_diff']:+.2f}({top['coev_win']*100:4.1f}%W) | "
                f"tac:{top['tac_diff']:+.2f} | "
                f"mir={top['mir_mean']:.2f}(s12={top['mir_12']:.2f}) | "
                f"ff={top['ffree']*100:4.1f}% dp={top['disp']*100:4.1f}%",
                flush=True,
            )

        # Final head-to-head holdout validation (160 games per matchup)
        final_candidates = [mean] + [e["theta"] for e in elites[:3]]
        final_tasks = [
            (
                i,
                th,
                base_params,
                champ_params,
                u_tanh,
                90_000,
                160,  # n_inv
                120,  # n_def
                120,  # n_coev
                60,   # n_mir
                60,   # n_tac
            )
            for i, th in enumerate(final_candidates)
        ]
        final_res = list(pool.map(_eval_candidate_worker, final_tasks))
        final_res.sort(key=lambda r: r["fitness"], reverse=True)
        winner = final_res[0]

    print("\n=== Final Selected Policy (Holdout Validation) ===", flush=True)
    print(
        f"  1v3 Old selfplay4 : diff={winner['inv_diff']:+.3f}, win={winner['inv_win']*100:.1f}%\n"
        f"  3v1 Old selfplay4 : diff={winner['def_diff']:+.3f}\n"
        f"  1v3 Coev Champion : diff={winner['coev_diff']:+.3f}, win={winner['coev_win']*100:.1f}%\n"
        f"  vs Tactical League: diff={winner['tac_diff']:+.3f}\n"
        f"  4x Mirror Score   : {winner['mir_mean']:.3f} (seeds 0..11: {winner['mir_12']:.3f})\n"
        f"  Move Selectivity  : foreign_free={winner['ffree']*100:.1f}%, displacements={winner['disp']*100:.1f}%",
        flush=True,
    )
    for name, val in zip(THETA_NAMES, winner["theta"]):
        print(f"    {name:<24}: {float(val):+.3f}", flush=True)

    out_net = copy.deepcopy(base_net)
    out_net.params = apply_theta(base_params, u_tanh, winner["theta"])
    out_net.optimizer = type(out_net.optimizer)(out_net.params, lr=5e-4)
    out_net.trained_generations = int(base_net.trained_generations) + args.generations
    out_net.save(TARGET_CKPT)
    print(
        f"\nSaved retrained checkpoint -> {TARGET_CKPT} in {time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
