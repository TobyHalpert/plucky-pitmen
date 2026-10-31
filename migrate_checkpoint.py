"""Widen a legacy checkpoint so it still loads after the feature layout changed.

Checkpoints store their own weight shapes, so a network trained before the
per-opponent *displacement* feature was added has an input width of 78 (25 state
+ 9 backside + 4 opponent slots x 11) instead of today's 82, and ``PolicyNet.load``
refuses it. This tool inserts a zero column at the new position, which is
exactly what an untrained feature looks like: the migrated network makes the
same decisions as the original until it is trained again.

    python migrate_checkpoint.py checkpoints/selfplay4.npz
    python migrate_checkpoint.py old.npz new.npz     # explicit output path
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from learning import PolicyNet

BASE = 34  # 25 state + 9 backside features come first
SLOTS = 4  # MAX_TRACKED_OPPONENTS
ACTION_FEATURES = 21


def remap_state_index(index: int, old_opponent_features: int, new_opponent_features: int) -> int:
    """Map a state-feature index in the old layout onto the new one."""
    if index < BASE:
        return index
    slot, offset = divmod(index - BASE, old_opponent_features)
    if offset >= old_opponent_features:  # pragma: no cover - defensive
        raise ValueError("offset outside the opponent block")
    if slot >= SLOTS:
        raise ValueError("index outside the tracked-opponent block")
    return BASE + slot * new_opponent_features + offset


def migrate(
    source: str | Path,
    destination: str | Path,
    old_opponent_features: int,
    new_opponent_features: int,
) -> Path:
    # Read the arrays directly: ``PolicyNet.load`` rejects the legacy width.
    with np.load(Path(source), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        raw = {
            key: np.asarray(data[key], dtype=np.float32)
            for key in data.files
            if key != "meta"
        }
    net = PolicyNet(
        hidden=int(meta.get("hidden", 64)),
        value_hidden=int(meta.get("value_hidden", 32)),
        name=str(meta.get("name", Path(source).stem)),
    )
    net.trained_generations = int(meta.get("trained_generations", 0))
    net.metadata = {
        key: value
        for key, value in meta.items()
        if key not in {"hidden", "value_hidden", "name", "trained_generations"}
    }
    old_state = BASE + SLOTS * old_opponent_features
    new_state = BASE + SLOTS * new_opponent_features
    width = raw["W1"].shape[1]
    if raw["W1"].shape[0] == new_state + ACTION_FEATURES:
        raise SystemExit(f"{source} already has today's feature layout")
    if raw["W1"].shape[0] != old_state + ACTION_FEATURES:
        raise SystemExit(
            f"{source} has {raw['W1'].shape[0]} input features, expected "
            f"{old_state + ACTION_FEATURES} for --from-opponent-features "
            f"{old_opponent_features}"
        )

    W1 = np.zeros((new_state + ACTION_FEATURES, width), dtype=np.float32)
    V1 = np.zeros((new_state, raw["V1"].shape[1]), dtype=np.float32)
    for old in range(old_state):
        new = remap_state_index(old, old_opponent_features, new_opponent_features)
        W1[new] = raw["W1"][old]
        V1[new] = raw["V1"][old]
    for old in range(old_state, old_state + ACTION_FEATURES):
        W1[new_state + old - old_state] = raw["W1"][old]

    params = dict(raw)
    params["W1"] = W1
    params["V1"] = V1
    net.params = params
    net.optimizer = type(net.optimizer)(params, lr=5e-4)
    return net.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination", nargs="?")
    parser.add_argument(
        "--from-opponent-features",
        type=int,
        default=11,
        help="opponent features per slot in the old checkpoint (11 before the "
        "displacement feature, 12 today)",
    )
    args = parser.parse_args()

    from features import N_OPPONENT_FEATURES

    destination = args.destination
    if destination is None:
        destination = str(Path(args.source).with_name(Path(args.source).stem + "_migrated.npz"))
    path = migrate(
        args.source,
        destination,
        args.from_opponent_features,
        int(N_OPPONENT_FEATURES),
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
