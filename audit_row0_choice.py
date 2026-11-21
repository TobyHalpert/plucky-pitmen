"""Measure what the policy chooses in unchanged Row-0 announcement situations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from features import action_mask, split_action, state_action_matrix
from game_env import LORRY, MAX_COLUMNS, PASS, PIT_CAGE
from learning import PolicyNet
from train_row0_blast import collect_row0


def masses(net: PolicyNet, positions):
    rows = []
    for source, seat, kind, _game_seed, _action in positions:
        ctx = source.player_action_context(seat)
        probs = net.masked_probs(state_action_matrix(ctx), action_mask(ctx), temperature=0.7)

        def primitive_row(action: int):
            primitive = split_action(int(action))[0]
            return primitive // MAX_COLUMNS if 0 <= primitive < PIT_CAGE else None

        row0_blast = sum(float(probs[a]) for a in np.flatnonzero(action_mask(ctx))
                         if split_action(int(a))[1] and primitive_row(int(a)) == 0)
        row1_blast = sum(float(probs[a]) for a in np.flatnonzero(action_mask(ctx))
                         if split_action(int(a))[1] and primitive_row(int(a)) == 1)
        row2_blast = sum(float(probs[a]) for a in np.flatnonzero(action_mask(ctx))
                         if split_action(int(a))[1] and primitive_row(int(a)) == 2)
        nearby_blast = row1_blast + row2_blast
        leave = sum(float(probs[a]) for a in (int(PIT_CAGE), int(LORRY)) if action_mask(ctx)[a])
        pass_mass = float(probs[int(PASS)]) if action_mask(ctx)[int(PASS)] else 0.0
        rows.append({
            "kind": kind,
            "row0_blast": row0_blast,
            "row1_blast": row1_blast,
            "row2_blast": row2_blast,
            "nearby_blast": nearby_blast,
            "pass": pass_mass,
            "leave": leave,
        })

    summary = {
        "positions": len(rows),
        "initial_positions": sum(row["kind"] == "initial" for row in rows),
        "follow_up_positions": sum(row["kind"] == "follow_up" for row in rows),
        "row0_blast": float(np.mean([row["row0_blast"] for row in rows])),
        "row2_blast": float(np.mean([row["row2_blast"] for row in rows])),
        "nearby_blast": float(np.mean([row["nearby_blast"] for row in rows])),
        "follow_up_pass": float(np.mean([row["pass"] for row in rows if row["kind"] == "follow_up"] or [0])),
        "follow_up_leave": float(np.mean([row["leave"] for row in rows if row["kind"] == "follow_up"] or [0])),
    }
    initial = [row for row in rows if row["kind"] == "initial"]
    if initial:
        summary.update({
            "initial_row0_blast": float(np.mean([row["row0_blast"] for row in initial])),
            "initial_row2_blast": float(np.mean([row["row2_blast"] for row in initial])),
            "initial_nearby_blast": float(np.mean([row["nearby_blast"] for row in initial])),
            "initial_row2_preferred_rate": float(np.mean([
                row["row2_blast"] > row["row0_blast"] for row in initial
            ])),
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="checkpoints/selfplay4_last_haul.npz")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--positions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3600000)
    parser.add_argument("--out", default="reports/row0_choice_audit.json")
    args = parser.parse_args()

    positions = collect_row0(
        PolicyNet.load(args.baseline), args.positions, args.seed, limit=args.positions * 80
    )
    report = {"source": args.baseline, "collected_positions": len(positions)}
    for path in args.candidates:
        if not Path(path).exists():
            continue
        report[Path(path).name] = masses(PolicyNet.load(path), positions)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
