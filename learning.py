"""A small numpy policy-gradient learner for Plucky Pitmen.

The policy scores every candidate action with a shared MLP over
``[state features | action features]`` and samples from a masked softmax.
Training is REINFORCE with a learned value baseline (A2C-style) and an entropy
bonus.  No deep-learning framework is required, which keeps the project's
dependency list at numpy + gymnasium.

The network is intentionally small: Plucky Pitmen decisions are short,
interpretable and highly stochastic, so a compact model trained against a
league of opponents is both faster and less prone to overfitting than a large
one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from features import (
    N_ACTIONS,
    N_INPUT_FEATURES,
    N_STATE_FEATURES,
    action_mask,
    state_action_matrix,
    state_features,
)


# ----------------------------------------------------------------------
# Experience
# ----------------------------------------------------------------------
@dataclass
class Decision:
    """One planning decision of one seat, with its (later filled) return."""

    player: int
    ctx: object
    state: np.ndarray
    action: int
    reward: float = 0.0
    ret: float = 0.0
    logp: float = 0.0
    value: float = 0.0
    generation: int = 0


def compute_returns(decisions: Sequence[Decision], gamma: float = 0.99) -> None:
    """Fill ``ret`` with the discounted return-to-go for each player."""
    by_player: dict[int, list[Decision]] = {}
    for decision in decisions:
        by_player.setdefault(decision.player, []).append(decision)
    for sequence in by_player.values():
        running = 0.0
        for decision in reversed(sequence):
            running = decision.reward + gamma * running
            decision.ret = running


# ----------------------------------------------------------------------
# Adam
# ----------------------------------------------------------------------
class Adam:
    def __init__(
        self,
        params: dict[str, np.ndarray],
        lr: float = 5e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {key: np.zeros_like(value) for key, value in params.items()}
        self.v = {key: np.zeros_like(value) for key, value in params.items()}
        self.t = 0

    def step(
        self,
        params: dict[str, np.ndarray],
        grads: dict[str, np.ndarray],
        clip: float = 5.0,
    ) -> float:
        norm = float(np.sqrt(sum(float((g * g).sum()) for g in grads.values())))
        scale = 1.0
        if clip and norm > clip:
            scale = clip / (norm + 1e-12)
        self.t += 1
        for key, grad in grads.items():
            grad = grad * scale
            if self.weight_decay:
                grad = grad + self.weight_decay * params[key]
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad * grad
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return norm


# ----------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------
class PolicyNet:
    """Masked softmax policy over ``2 * 28`` actions with a value baseline."""

    def __init__(
        self,
        hidden: int = 64,
        value_hidden: int = 32,
        seed: int = 0,
        lr: float = 5e-4,
        name: str = "policy",
    ) -> None:
        rng = np.random.default_rng(seed)
        self.hidden = int(hidden)
        self.value_hidden = int(value_hidden)
        self.name = name
        self.params: dict[str, np.ndarray] = {}

        if self.hidden > 0:
            self.params["W1"] = (
                rng.standard_normal((N_INPUT_FEATURES, self.hidden))
                / np.sqrt(N_INPUT_FEATURES)
            ).astype(np.float32)
            self.params["b1"] = np.zeros(self.hidden, dtype=np.float32)
            self.params["W2"] = (
                rng.standard_normal((self.hidden, 1)) / np.sqrt(self.hidden)
            ).astype(np.float32)
            self.params["b2"] = np.zeros(1, dtype=np.float32)
        else:
            self.params["W2"] = (
                rng.standard_normal((N_INPUT_FEATURES, 1))
                / np.sqrt(N_INPUT_FEATURES)
            ).astype(np.float32)
            self.params["b2"] = np.zeros(1, dtype=np.float32)

        if self.value_hidden > 0:
            self.params["V1"] = (
                rng.standard_normal((N_STATE_FEATURES, self.value_hidden))
                / np.sqrt(N_STATE_FEATURES)
            ).astype(np.float32)
            self.params["vb1"] = np.zeros(self.value_hidden, dtype=np.float32)
            self.params["V2"] = (
                rng.standard_normal((self.value_hidden, 1)) / np.sqrt(self.value_hidden)
            ).astype(np.float32)
            self.params["vb2"] = np.zeros(1, dtype=np.float32)
        else:
            self.params["V2"] = np.zeros((N_STATE_FEATURES, 1), dtype=np.float32)
            self.params["vb2"] = np.zeros(1, dtype=np.float32)

        self.optimizer = Adam(self.params, lr=lr)
        self.trained_generations = 0
        self.metadata: dict[str, object] = {}

    # -- forward / backward -------------------------------------------
    def _policy_forward(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        if self.hidden > 0:
            z1 = X @ self.params["W1"] + self.params["b1"]
            h = np.tanh(z1)
            scores = (h @ self.params["W2"] + self.params["b2"]).ravel()
            return scores, (X, h)
        scores = (X @ self.params["W2"] + self.params["b2"]).ravel()
        return scores, (X,)

    def _policy_backward(
        self, cache: tuple[np.ndarray, ...], dscores: np.ndarray
    ) -> dict[str, np.ndarray]:
        dscores = dscores.reshape(-1, 1)
        if self.hidden > 0:
            X, h = cache
            grads = {
                "W2": h.T @ dscores,
                "b2": dscores.sum(axis=0),
            }
            dh = dscores @ self.params["W2"].T
            dz1 = dh * (1.0 - h * h)
            grads["W1"] = X.T @ dz1
            grads["b1"] = dz1.sum(axis=0)
            return grads
        (X,) = cache
        return {"W2": X.T @ dscores, "b2": dscores.sum(axis=0)}

    def _value_forward(
        self, states: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        if self.value_hidden > 0:
            zv = states @ self.params["V1"] + self.params["vb1"]
            hv = np.tanh(zv)
            values = (hv @ self.params["V2"] + self.params["vb2"]).ravel()
            return values, (states, hv, zv)
        values = (states @ self.params["V2"] + self.params["vb2"]).ravel()
        return values, (states,)

    def _value_backward(
        self, cache: tuple[np.ndarray, ...], dvalues: np.ndarray
    ) -> dict[str, np.ndarray]:
        dvalues = dvalues.reshape(-1, 1)
        if self.value_hidden > 0:
            states, hv, _zv = cache
            grads = {
                "V2": hv.T @ dvalues,
                "vb2": dvalues.sum(axis=0),
            }
            dhv = dvalues @ self.params["V2"].T
            dzv = dhv * (1.0 - hv * hv)
            grads["V1"] = states.T @ dzv
            grads["vb1"] = dzv.sum(axis=0)
            return grads
        (states,) = cache
        return {"V2": states.T @ dvalues, "vb2": dvalues.sum(axis=0)}

    # -- inference -----------------------------------------------------
    def logits(self, matrix: np.ndarray) -> np.ndarray:
        return self._policy_forward(np.asarray(matrix, dtype=np.float32))[0]

    def value(self, state: np.ndarray) -> float:
        values, _ = self._value_forward(
            np.asarray(state, dtype=np.float32)[None, :]
        )
        return float(values[0])

    def masked_probs(
        self,
        matrix: np.ndarray,
        mask: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        logits = self.logits(matrix) / max(temperature, 1e-6)
        logits = np.where(mask, logits, -1e30)
        logits = logits - logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()

    def select_action(
        self,
        matrix: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        epsilon: float = 0.0,
    ) -> tuple[int, float, float]:
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            return 0, 0.0, 0.0
        values, _ = self._value_forward(
            np.asarray(matrix[0, :N_STATE_FEATURES], dtype=np.float32)[None, :]
        )
        probabilities = self.masked_probs(matrix, mask, temperature)
        if epsilon > 0 and rng.random() < epsilon:
            action = int(rng.choice(legal))
        elif deterministic:
            action = int(np.argmax(probabilities))
        else:
            action = int(rng.choice(N_ACTIONS, p=probabilities))
        logp = float(np.log(probabilities[action] + 1e-12))
        return action, logp, float(values[0])

    # -- learning ------------------------------------------------------
    def fit(
        self,
        decisions: Sequence[Decision],
        *,
        epochs: int = 1,
        batch_size: int = 256,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        rng: np.random.Generator | None = None,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        if not decisions:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        rng = rng or np.random.default_rng(0)
        compute_returns(decisions, gamma=gamma)
        returns = np.asarray([d.ret for d in decisions], dtype=np.float32)

        order = rng.permutation(len(decisions))
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0
        for _epoch in range(max(1, epochs)):
            for start in range(0, len(order), batch_size):
                if max_batches is not None and updates >= max_batches:
                    break
                index = order[start : start + batch_size]
                batch = [decisions[int(i)] for i in index]
                self._fit_batch(
                    batch,
                    returns[index],
                    entropy_coef=entropy_coef,
                    value_coef=value_coef,
                    stats=stats,
                )
                updates += 1
        if updates:
            for key in stats:
                stats[key] /= updates
        return stats

    def _fit_batch(
        self,
        batch: Sequence[Decision],
        returns: np.ndarray,
        *,
        entropy_coef: float,
        value_coef: float,
        stats: dict[str, float],
    ) -> None:
        count = len(batch)
        matrices = np.stack(
            [
                state_action_matrix(decision.ctx, decision.state)
                for decision in batch
            ]
        )
        masks = np.stack([action_mask(decision.ctx) for decision in batch])
        actions = np.asarray([decision.action for decision in batch], dtype=np.int64)
        states = np.asarray([decision.state for decision in batch], dtype=np.float32)

        flat = matrices.reshape(count * N_ACTIONS, N_INPUT_FEATURES)
        flat_scores, cache = self._policy_forward(flat)
        scores = flat_scores.reshape(count, N_ACTIONS)

        masked = np.where(masks, scores, -1e30)
        masked = masked - masked.max(axis=1, keepdims=True)
        exp = np.exp(masked)
        probs = exp / exp.sum(axis=1, keepdims=True)
        log_probs = np.log(probs + 1e-12)

        values, value_cache = self._value_forward(states)
        advantage = returns - values
        std = float(advantage.std())
        if std > 1e-6:
            advantage = advantage / (std + 1e-6)

        chosen = log_probs[np.arange(count), actions]
        entropy = -(probs * log_probs).sum(axis=1)

        policy_loss = -float((chosen * advantage).mean())
        value_loss = 0.5 * float(((values - returns) ** 2).mean())

        # dL/dlogits for REINFORCE + entropy bonus.
        dlogits = probs.copy()
        dlogits[np.arange(count), actions] -= 1.0
        dlogits *= (advantage / count)[:, None]
        entropy_grad = probs * (log_probs + entropy[:, None])
        dlogits -= (entropy_coef / count) * entropy_grad

        grads = self._policy_backward(cache, dlogits.ravel())

        dvalues = value_coef * (values - returns) / count
        grads.update(self._value_backward(value_cache, dvalues))

        self.optimizer.step(self.params, grads)

        stats["policy_loss"] += policy_loss
        stats["value_loss"] += value_loss
        stats["entropy"] += float(entropy.mean())

    # -- persistence ---------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "hidden": self.hidden,
            "value_hidden": self.value_hidden,
            "name": self.name,
            "trained_generations": self.trained_generations,
            **self.metadata,
        }
        with open(path, "wb") as handle:
            np.savez(handle, meta=json.dumps(meta), **self.params)
        return path

    @classmethod
    def load(cls, path: str | Path, *, lr: float = 5e-4) -> "PolicyNet":
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            net = cls(
                hidden=int(meta.get("hidden", 64)),
                value_hidden=int(meta.get("value_hidden", 32)),
                lr=lr,
                name=str(meta.get("name", Path(path).stem)),
            )
            for key in net.params:
                net.params[key] = np.asarray(data[key], dtype=np.float32)
            net.trained_generations = int(meta.get("trained_generations", 0))
            net.metadata = {
                key: value
                for key, value in meta.items()
                if key not in {"hidden", "value_hidden", "name", "trained_generations"}
            }
        return net

    def clone(self, name: str | None = None) -> "PolicyNet":
        clone = PolicyNet(
            hidden=self.hidden,
            value_hidden=self.value_hidden,
            lr=self.optimizer.lr,
            name=name or self.name,
        )
        for key in self.params:
            clone.params[key] = self.params[key].copy()
        clone.trained_generations = self.trained_generations
        clone.metadata = dict(self.metadata)
        return clone


__all__ = [
    "Adam",
    "Decision",
    "PolicyNet",
    "compute_returns",
    "state_features",
]
