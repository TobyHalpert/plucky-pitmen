"""Widen a legacy checkpoint so it still loads after the feature layout changed.

Checkpoints store their own weight shapes, so a network trained before the
per-opponent *displacement* feature was added has an input width of 78 (25 state
+ 9 backside + 4 opponent slots x 11) instead of today's 82, and ``PolicyNet.load``
refuses it. This tool inserts a zero column at the new position, which is
exactly what an untrained feature looks like: the migrated network makes the
same decisions as the original until it is trained again.

    python migrate_checkpoint.py checkpoints/selfplay4.npz
    python migrate_checkpoint.py old.npz new.npz     # explicit output path

A feature *appended after* the tracked-opponent block is a different shape: the
new index is the last state feature instead of something inside a slot, and the
action block moves up by one. ``--append-hazard`` handles that case: it inserts
a zero row for the table-hazard feature, which an untrained input is worth.

    python migrate_checkpoint.py old.npz new.npz --append-hazard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from features import N_ACTION_FEATURES, N_OPPONENT_FEATURES, N_STATE_FEATURES
from learning import PolicyNet

BASE = 34  # 25 state + 9 backside features come first
SLOTS = 4  # MAX_TRACKED_OPPONENTS
ACTION_FEATURES = int(N_ACTION_FEATURES)


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


def append_state_feature(
    source: str | Path,
    destination: str | Path,
    old_state: int,
) -> Path:
    """Insert one zero row per weight matrix for a feature appended at the end."""
    with np.load(Path(source), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        raw = {
            key: np.asarray(data[key], dtype=np.float32)
            for key in data.files
            if key != "meta"
        }
    new_state = int(N_STATE_FEATURES)
    if old_state >= new_state:
        raise SystemExit(f"{source} already has today's feature layout")
    if raw["W1"].shape[0] != old_state + ACTION_FEATURES:
        raise SystemExit(
            f"{source} has {raw['W1'].shape[0]} input features, expected "
            f"{old_state + ACTION_FEATURES}"
        )
    if raw["V1"].shape[0] != old_state:
        raise SystemExit(
            f"{source} has a value net of width {raw['V1'].shape[0]}, expected {old_state}"
        )
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
    width = raw["W1"].shape[1]
    W1 = np.zeros((new_state + ACTION_FEATURES, width), dtype=np.float32)
    V1 = np.zeros((new_state, raw["V1"].shape[1]), dtype=np.float32)
    W1[:old_state] = raw["W1"][:old_state]
    W1[new_state:] = raw["W1"][old_state:]
    V1[:old_state] = raw["V1"][:old_state]
    params = dict(raw)
    params["W1"] = W1
    params["V1"] = V1
    net.params = params
    net.optimizer = type(net.optimizer)(params, lr=5e-4)
    return net.save(destination)


def add_phase_clock(source: str | Path, destination: str | Path) -> Path:
    """Widen a checkpoint to the planning-clock layout.

    The tracked-opponent block goes from 12 features to 13 (*has this rival
    planned*), and two state features are appended after the hazard (*how close
    the phase is to closing*, *how many seats are still forced to move*). Both
    new inputs start at zero, which is what an untrained feature is worth, so the
    migrated network makes the same decisions as the original.
    """
    with np.load(Path(source), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        raw = {
            key: np.asarray(data[key], dtype=np.float32)
            for key in data.files
            if key != "meta"
        }
    old_opponent_features = N_OPPONENT_FEATURES - 1
    old_state = BASE + SLOTS * old_opponent_features + 1  # hazard was appended last
    new_state = int(N_STATE_FEATURES)
    if raw["V1"].shape[0] == new_state:
        raise SystemExit(f"{source} already has today's feature layout")
    if raw["V1"].shape[0] != old_state:
        raise SystemExit(
            f"{source} has a value net of width {raw['V1'].shape[0]}, expected "
            f"{old_state} for the layout before the planning clock"
        )
    if raw["W1"].shape[0] != old_state + ACTION_FEATURES:
        raise SystemExit(
            f"{source} has {raw['W1'].shape[0]} input features, expected "
            f"{old_state + ACTION_FEATURES}"
        )
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
    width = raw["W1"].shape[1]
    W1 = np.zeros((new_state + ACTION_FEATURES, width), dtype=np.float32)
    V1 = np.zeros((new_state, raw["V1"].shape[1]), dtype=np.float32)
    for old in range(old_state):
        if old < BASE:
            new = old
        elif old < BASE + SLOTS * old_opponent_features:
            slot, offset = divmod(old - BASE, old_opponent_features)
            new = BASE + slot * N_OPPONENT_FEATURES + offset
        else:
            new = old + SLOTS  # trailing state features, hazard first
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


def add_blast_risk(source: str | Path, destination: str | Path) -> Path:
    """Widen a 110-input (89 state + 21 action) checkpoint to 113 inputs (90 state + 23 action).

    Appends one zero row after ``FORCED_SEATS_FEATURE`` (row 89 = ``BLAST_RISK_FEATURE``)
    in both ``W1`` and ``V1``, and two zero rows after ``action_features[20]``
    (rows 111..112 = ``[announces_blast * blast_risk, announces_blast * (1 - blast_risk)]``).
    """
    raw = dict(np.load(source, allow_pickle=False))
    meta = json.loads(str(raw.pop("meta")))
    old_state = 89
    old_action = 21
    new_state = int(N_STATE_FEATURES)
    new_action = int(ACTION_FEATURES)
    if raw["W1"].shape[0] != old_state + old_action:
        raise SystemExit(
            f"{source} has {raw['W1'].shape[0]} input features, expected "
            f"{old_state + old_action}"
        )
    if raw["V1"].shape[0] != old_state:
        raise SystemExit(
            f"{source} has {raw['V1'].shape[0]} value-head features, expected {old_state}"
        )
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
    W1 = np.zeros((new_state + new_action, raw["W1"].shape[1]), dtype=np.float32)
    V1 = np.zeros((new_state, raw["V1"].shape[1]), dtype=np.float32)
    W1[:old_state] = raw["W1"][:old_state]
    V1[:old_state] = raw["V1"][:old_state]
    W1[new_state : new_state + old_action] = raw["W1"][old_state : old_state + old_action]
    params = dict(raw)
    params["W1"] = W1
    params["V1"] = V1
    net.params = params
    net.optimizer = type(net.optimizer)(params, lr=5e-4)
    return net.save(destination)


def add_raider_features(source: str | Path, destination: str | Path) -> Path:
    """Widen a 123- or 131-input checkpoint to ``N_STATE_FEATURES + N_ACTION_FEATURES`` (133).

    Appends zero rows at ``action_features[24..33]`` (foreign canary/loot/push,
    row-0/leader push, smart foreign safe, dynamite pair, dragon bait, and
    Turn-1 last-chance / parachute pushes), preserving exact forward-pass
    logits until retrained.
    """
    raw = dict(np.load(source, allow_pickle=False))
    meta = json.loads(str(raw.pop("meta")))
    old_state = 99
    old_total = int(raw["W1"].shape[0])
    new_state = int(N_STATE_FEATURES)
    new_action = int(ACTION_FEATURES)
    if old_total not in {99 + 24, 99 + 32}:
        raise SystemExit(
            f"{source} has {old_total} input features, expected 123 or 131"
        )
    if raw["V1"].shape[0] != old_state:
        raise SystemExit(
            f"{source} has {raw['V1'].shape[0]} value-head features, expected {old_state}"
        )
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
    W1 = np.zeros((new_state + new_action, raw["W1"].shape[1]), dtype=np.float32)
    W1[:old_total] = raw["W1"]
    params = dict(raw)
    params["W1"] = W1
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
    parser.add_argument(
        "--add-phase-clock",
        action="store_true",
        help="the checkpoint predates the planning-clock inputs (a 13th "
        "per-opponent flag plus two appended state features)",
    )
    parser.add_argument(
        "--add-blast-risk",
        action="store_true",
        help="the checkpoint predates BLAST_RISK_FEATURE and the two blast-risk "
        "action features (widens 89+21 -> 90+23 with zero rows)",
    )
    parser.add_argument(
        "--add-raider-features",
        action="store_true",
        help="the checkpoint predates the eight raider/push/canary action features "
        "(widens 99+24 -> 99+32 with zero rows)",
    )
    parser.add_argument(
        "--append-hazard",
        action="store_true",
        help="the checkpoint predates the table-hazard state feature, which was "
        "appended after the tracked-opponent block",
    )
    args = parser.parse_args()

    destination = args.destination
    if destination is None:
        destination = str(Path(args.source).with_name(Path(args.source).stem + "_migrated.npz"))
    if args.add_raider_features:
        print(f"wrote {add_raider_features(args.source, destination)}")
        return
    if args.add_blast_risk:
        print(f"wrote {add_blast_risk(args.source, destination)}")
        return
    if args.add_phase_clock:
        print(f"wrote {add_phase_clock(args.source, destination)}")
        return
    if args.append_hazard:
        old_state = int(N_STATE_FEATURES) - 1
        path = append_state_feature(args.source, destination, old_state)
        print(f"wrote {path}")
        return
    path = migrate(
        args.source,
        destination,
        args.from_opponent_features,
        int(N_OPPONENT_FEATURES),
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
