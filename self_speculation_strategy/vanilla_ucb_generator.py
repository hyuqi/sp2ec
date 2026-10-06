"""Vanilla global-UCB variants over the existing SP2EC arm builders."""

from typing import Optional

from utils import Env

from .sp2ec_basic_generator import SP2ECBasicGenerator
from .sp2ec_knapspec_generator import SP2ECKnapspecGenerator
from .vanilla_ucb import UCBSpecTPSOptimizer


class _VanillaUCBSelectorMixin:
    ucb_reward_bound: float
    ucb_delta: float

    def _new_arm_selector(self) -> UCBSpecTPSOptimizer:
        return UCBSpecTPSOptimizer(
            L=self.ucb_reward_bound,
            delta=self.ucb_delta,
        )


class VanillaUCBBaseGenerator(_VanillaUCBSelectorMixin, SP2ECBasicGenerator):
    """Sequential block-importance arms selected by global TPS-UCB."""

    def __init__(
        self,
        env: Env,
        gamma: int = 4,
        optimize_interval: int = 512,
        coefficients: Optional[tuple] = None,
        tree: bool = False,
        ucb_reward_bound: float = 256.0,
        ucb_delta: float = 0.1,
        draft_confidence_threshold: float = 0.7,
        dynamic_draft_stopping: bool = True,
        max_skip_fraction: Optional[float] = None,
        include_zero_skip_arm: bool = False,
        min_skipped_blocks: Optional[int] = None,
        max_skipped_blocks: Optional[int] = None,
    ):
        self.ucb_reward_bound = float(ucb_reward_bound)
        self.ucb_delta = float(ucb_delta)
        super().__init__(
            env=env,
            gamma=gamma,
            optimize_interval=optimize_interval,
            coefficients=coefficients,
            tree=tree,
            draft_confidence_threshold=draft_confidence_threshold,
            dynamic_draft_stopping=dynamic_draft_stopping,
            max_skip_fraction=max_skip_fraction,
            include_zero_skip_arm=include_zero_skip_arm,
            min_skipped_blocks=min_skipped_blocks,
            max_skipped_blocks=max_skipped_blocks,
        )


class VanillaUCBKnapspecGenerator(_VanillaUCBSelectorMixin, SP2ECKnapspecGenerator):
    """KnapSpec DP arms selected by global TPS-UCB."""

    def __init__(
        self,
        env: Env,
        gamma: int = 4,
        skip_budget_M: int = 8,
        optimize_interval: int = 64,
        coefficients: Optional[tuple] = None,
        sim_threshold: float = 0.5,
        num_arms: int = 5,
        tree: bool = False,
        ucb_reward_bound: float = 256.0,
        ucb_delta: float = 0.1,
        draft_confidence_threshold: float = 0.7,
        dynamic_draft_stopping: bool = True,
        scoring_mode: str = "legacy",
        dp_budget_fraction: float = 0.5,
        include_zero_skip_arm: bool = False,
        arm_index_start: Optional[int] = None,
        arm_index_end: Optional[int] = None,
    ):
        self.ucb_reward_bound = float(ucb_reward_bound)
        self.ucb_delta = float(ucb_delta)
        super().__init__(
            env=env,
            gamma=gamma,
            skip_budget_M=skip_budget_M,
            optimize_interval=optimize_interval,
            coefficients=coefficients,
            sim_threshold=sim_threshold,
            num_arms=num_arms,
            tree=tree,
            draft_confidence_threshold=draft_confidence_threshold,
            dynamic_draft_stopping=dynamic_draft_stopping,
            scoring_mode=scoring_mode,
            dp_budget_fraction=dp_budget_fraction,
            include_zero_skip_arm=include_zero_skip_arm,
            arm_index_start=arm_index_start,
            arm_index_end=arm_index_end,
        )
