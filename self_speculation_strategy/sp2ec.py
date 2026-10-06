from dataclasses import dataclass
import math
from typing import Any, Dict, List, Sequence


@dataclass
class _ArmStatistics:
    pulls: int = 0
    reward_sum: float = 0.0
    cost_sum: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.pulls if self.pulls else 0.0

    @property
    def mean_cost(self) -> float:
        return self.cost_sum / self.pulls if self.pulls else 0.0

    @property
    def throughput(self) -> float:
        return self.reward_sum / self.cost_sum if self.cost_sum > 0.0 else 0.0


def select_top_tpt_arms(arms: Sequence[Any], num_arms: int) -> List[Any]:
    """Keep the top distinct Eq. (4) candidates, ordered by DP budget."""
    if num_arms < 1:
        raise ValueError("num_arms must be at least 1")

    ranked = sorted(arms, key=lambda arm: arm.estimated_tpt, reverse=True)
    selected = []
    seen_masks = set()
    for arm in ranked:
        mask_key = tuple(arm.skip_set)
        if mask_key in seen_masks:
            continue
        seen_masks.add(mask_key)
        arm.tpt_rank = len(selected) + 1
        selected.append(arm)
        if len(selected) == num_arms:
            break

    return sorted(selected, key=lambda arm: arm.budget)


class SinglePeakArmSelector:
    """Single-Peak Speculative Decoding from SP2EC Algorithm 2."""

    def __init__(self, max_draft_length: int, beta: float = 0.1):
        if max_draft_length < 1:
            raise ValueError("max_draft_length must be at least 1")
        if not math.isfinite(beta) or beta < 0.0:
            raise ValueError("beta must be finite and non-negative")
        self.max_draft_length = int(max_draft_length)
        self.beta = float(beta)
        self.arms: List[Any] = []
        self.stats: List[_ArmStatistics] = []
        self.leader_counts: List[int] = []
        self.last_selected_arm = None
        self.arm_set_id = -1
        self.history: List[Dict[str, object]] = []

    @property
    def has_arms(self) -> bool:
        return bool(self.arms)

    @property
    def total_pulls(self) -> int:
        return sum(stat.pulls for stat in self.stats)

    def reset(self, arms: Sequence[Any]) -> None:
        if not arms:
            raise ValueError("SP2EC requires at least one arm")

        # The single-peak neighborhood is defined over increasing DP budget.
        self.arms = sorted(arms, key=lambda arm: arm.budget)
        self.stats = [_ArmStatistics() for _ in self.arms]
        self.leader_counts = [0 for _ in self.arms]
        self.last_selected_arm = None
        self.arm_set_id += 1
        self.history = []

    def select_arm(self) -> int:
        if not self.arms:
            raise RuntimeError("No SP2EC arms are available")

        t = self.total_pulls
        num_arms = len(self.arms)

        # Algorithm 2 initializes every arm exactly once.
        if t <= num_arms - 1:
            arm_idx = t
            self.last_selected_arm = arm_idx
            return arm_idx

        leader = max(range(num_arms), key=lambda idx: self.stats[idx].throughput)
        self.leader_counts[leader] += 1
        leader_count = self.leader_counts[leader]

        if leader_count % 3 == 1:
            arm_idx = leader
        else:
            left = max(0, leader - 1)
            right = min(num_arms - 1, leader + 1)
            neighborhood = range(left, right + 1)
            arm_idx = max(
                neighborhood,
                key=lambda idx: self._ucb(idx, leader_count),
            )

        self.last_selected_arm = arm_idx
        return arm_idx

    def update(self, arm_idx: int, reward: int, elapsed_cost: float) -> None:
        if not 0 <= arm_idx < len(self.arms):
            raise IndexError(f"arm index {arm_idx} is out of range")
        if reward < 0:
            raise ValueError("reward must be non-negative")
        if elapsed_cost <= 0.0:
            raise ValueError("elapsed_cost must be positive")

        stat = self.stats[arm_idx]
        stat.pulls += 1
        stat.reward_sum += float(reward)
        stat.cost_sum += float(elapsed_cost)
        self.history.append(
            {
                "round": self.total_pulls,
                "arm_index": arm_idx,
                "reward": int(reward),
                "elapsed_cost": float(elapsed_cost),
            }
        )

    def snapshot(self) -> Dict[str, object]:
        total_reward = sum(stat.reward_sum for stat in self.stats)
        total_cost = sum(stat.cost_sum for stat in self.stats)
        return {
            "arm_set_id": self.arm_set_id,
            "beta": self.beta,
            "total_pulls": self.total_pulls,
            "total_reward": total_reward,
            "total_cost": total_cost,
            "empirical_tpt": total_reward / total_cost if total_cost > 0.0 else None,
            "arms": [
                self._arm_snapshot(idx, arm, stat)
                for idx, (arm, stat) in enumerate(zip(self.arms, self.stats))
            ],
            "history": list(self.history),
        }

    def _confidence_radius(self, arm_idx: int, leader_count: int) -> float:
        stat = self.stats[arm_idx]
        if stat.pulls == 0:
            return float("inf")

        pulls = stat.pulls
        log_argument = 3.0 * (leader_count ** 2) * math.sqrt(1.0 + pulls)
        return (self.max_draft_length / 2.0) * math.sqrt(
            ((1.0 + pulls) / (pulls ** 2))
            * (1.0 + 2.0 * math.log(log_argument))
        )

    def _ucb(self, arm_idx: int, leader_count: int) -> float:
        stat = self.stats[arm_idx]
        if stat.pulls == 0 or stat.mean_cost <= 0.0:
            return float("inf")
        return (
            stat.mean_reward
            + self.beta * self._confidence_radius(arm_idx, leader_count)
        ) / stat.mean_cost

    def _arm_snapshot(self, arm_idx, arm, stat) -> Dict[str, object]:
        if hasattr(arm, "to_dict"):
            payload = dict(arm.to_dict())
        else:
            payload = {
                "budget": arm.budget,
                "skip_set": list(arm.skip_set),
                "estimated_tpt": arm.estimated_tpt,
            }
        payload.update(
            {
                "index": arm_idx,
                "pulls": stat.pulls,
                "reward_sum": stat.reward_sum,
                "cost_sum": stat.cost_sum,
                "mean_reward": stat.mean_reward if stat.pulls else None,
                "mean_cost": stat.mean_cost if stat.pulls else None,
                "empirical_tpt": stat.throughput if stat.pulls else None,
                "empirical_throughput": stat.throughput if stat.pulls else None,
                "leader_count": self.leader_counts[arm_idx],
            }
        )
        return payload


# Keep the descriptive alias available for callers outside this repository.
SP2ECArmSelector = SinglePeakArmSelector
