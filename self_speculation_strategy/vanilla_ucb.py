"""Global UCB arm selection using empirical per-round throughput."""

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class UCBSpecTPSOptimizer:
    """BanditSpec-style global UCB with per-round TPS as the reward."""

    def __init__(
        self,
        arms: Optional[Sequence[Any]] = None,
        L: float = 256.0,
        delta: float = 0.1,
        arm_labels: Optional[Sequence[str]] = None,
        eps: float = 1e-9,
    ):
        if not math.isfinite(L) or L <= 0.0:
            raise ValueError("L must be finite and positive")
        if not math.isfinite(delta) or not 0.0 < delta <= 1.0:
            raise ValueError("delta must be in (0, 1]")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be finite and positive")

        self.L = float(L)
        self.delta = float(delta)
        self.eps = float(eps)
        self.arm_labels = list(arm_labels) if arm_labels is not None else None
        self.log_filename = "ucbspec_per_prompt_results.csv"
        self.arm_set_id = -1
        self.arms: List[Any] = []
        self.history: List[Dict[str, object]] = []
        self.last_arm: Optional[int] = None
        self._clear_statistics(0)
        if arms is not None and len(arms) > 0:
            self.reset(arms)

    def _clear_statistics(self, num_arms: int) -> None:
        self.K = int(num_arms)
        self.n_arms = self.K
        self.n = np.zeros(self.K, dtype=np.float64)
        self.sum_y = np.zeros(self.K, dtype=np.float64)
        self.sum_c = np.zeros(self.K, dtype=np.float64)
        self.sum_draft_len = np.zeros(self.K, dtype=np.float64)
        self.sum_tps = np.zeros(self.K, dtype=np.float64)
        self.values = np.zeros(self.K, dtype=np.float64)
        self.t = 0

    @property
    def has_arms(self) -> bool:
        return bool(self.arms)

    @property
    def total_pulls(self) -> int:
        return self.t

    def reset(self, arms: Sequence[Any]) -> None:
        if len(arms) == 0:
            raise ValueError("UCBSpec requires at least one arm")
        self.arms = list(arms)
        self._clear_statistics(len(self.arms))
        self.last_arm = None
        self.history = []
        self.arm_set_id += 1

    def confidence_radius(self, arm_idx: int) -> float:
        arm_idx = int(arm_idx)
        self._validate_arm_index(arm_idx)
        pulls = float(self.n[arm_idx])
        if pulls <= 0.0:
            return float("inf")

        t = max(float(self.t), 1.0)
        log_arg = (
            float(self.K)
            * (t ** 2)
            * math.sqrt(1.0 + pulls)
            / max(self.delta, self.eps)
        )
        log_arg = max(log_arg, 1.0 + self.eps)
        return (self.L / 2.0) * math.sqrt(
            ((1.0 + pulls) / (pulls ** 2))
            * (1.0 + 2.0 * math.log(log_arg))
        )

    def ucb(self, arm_idx: int) -> float:
        arm_idx = int(arm_idx)
        self._validate_arm_index(arm_idx)
        if self.n[arm_idx] <= 0.0:
            return float("inf")
        return float(self.values[arm_idx]) + self.confidence_radius(arm_idx)

    def select_arm(self) -> int:
        if not self.arms:
            raise RuntimeError("UCBSpec has no arms")

        # Preserve arm-builder observations: only initialize arms that have
        # not already been pulled while constructing the current arm set.
        unpulled = np.flatnonzero(self.n == 0.0)
        if unpulled.size:
            arm_idx = int(unpulled[0])
        else:
            indices = self.values + np.array(
                [self.confidence_radius(idx) for idx in range(self.K)],
                dtype=np.float64,
            )
            arm_idx = int(np.argmax(indices))

        self.last_arm = arm_idx
        return arm_idx

    def update(
        self,
        arm_idx: int,
        reward_tokens: float,
        time_cost: float,
        draft_len: float = 0.0,
    ) -> None:
        arm_idx = int(arm_idx)
        self._validate_arm_index(arm_idx)
        reward_tokens = float(reward_tokens)
        time_cost = float(time_cost)
        draft_len = float(draft_len)
        if not math.isfinite(reward_tokens) or reward_tokens < 0.0:
            raise ValueError("reward_tokens must be finite and non-negative")
        if not math.isfinite(time_cost) or time_cost <= 0.0:
            raise ValueError("time_cost must be finite and positive")
        if not math.isfinite(draft_len) or draft_len < 0.0:
            raise ValueError("draft_len must be finite and non-negative")

        reward_tps = reward_tokens / max(time_cost, self.eps)
        self.n[arm_idx] += 1.0
        self.sum_y[arm_idx] += reward_tokens
        self.sum_c[arm_idx] += time_cost
        self.sum_draft_len[arm_idx] += draft_len
        self.sum_tps[arm_idx] += reward_tps
        self.values[arm_idx] = self.sum_tps[arm_idx] / self.n[arm_idx]
        self.t += 1
        self.history.append(
            {
                "round": self.t,
                "arm_index": int(arm_idx),
                "reward": reward_tokens,
                "reward_tokens": reward_tokens,
                "elapsed_cost": time_cost,
                "reward_tps": reward_tps,
                "draft_len": draft_len,
            }
        )

    def snapshot(self) -> Dict[str, object]:
        total_tokens = float(self.sum_y.sum())
        total_cost = float(self.sum_c.sum())
        return {
            "algorithm": "vanilla_ucb_tps",
            "reward": "committed_tokens / empirical_round_time",
            "arm_set_id": self.arm_set_id,
            "L": self.L,
            "delta": self.delta,
            "total_pulls": self.total_pulls,
            "total_reward_tokens": total_tokens,
            "total_cost": total_cost,
            "aggregate_tps": total_tokens / total_cost if total_cost > 0.0 else None,
            "arms": [self._arm_snapshot(idx, arm) for idx, arm in enumerate(self.arms)],
            "history": list(self.history),
        }

    def _arm_snapshot(self, arm_idx: int, arm: Any) -> Dict[str, object]:
        if hasattr(arm, "to_dict"):
            payload = dict(arm.to_dict())
        else:
            payload = {
                "budget": getattr(arm, "budget", None),
                "skip_set": list(getattr(arm, "skip_set", [])),
            }
        pulls = int(self.n[arm_idx])
        payload.update(
            {
                "index": arm_idx,
                "label": (
                    self.arm_labels[arm_idx]
                    if self.arm_labels is not None and arm_idx < len(self.arm_labels)
                    else None
                ),
                "pulls": pulls,
                "reward_sum": float(self.sum_y[arm_idx]),
                "tps_reward_sum": float(self.sum_tps[arm_idx]),
                "cost_sum": float(self.sum_c[arm_idx]),
                "draft_length_sum": float(self.sum_draft_len[arm_idx]),
                "mean_tps": float(self.values[arm_idx]) if pulls else None,
                "empirical_tpt": (
                    float(self.sum_y[arm_idx] / self.sum_c[arm_idx])
                    if self.sum_c[arm_idx] > 0.0
                    else None
                ),
                "empirical_throughput": (
                    float(self.sum_y[arm_idx] / self.sum_c[arm_idx])
                    if self.sum_c[arm_idx] > 0.0
                    else None
                ),
                "confidence_radius": self.confidence_radius(arm_idx) if pulls else None,
                "ucb": self.ucb(arm_idx) if pulls else None,
            }
        )
        return payload

    def _validate_arm_index(self, arm_idx: int) -> None:
        if not 0 <= int(arm_idx) < self.K:
            raise IndexError(f"arm index {arm_idx} is out of range")
