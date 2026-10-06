from typing import Optional

from utils import Env

from .knapspec_generator import KnapspecGenerator


class SP2ECKnapspecGenerator(KnapspecGenerator):
    """KnapSpec candidate generation with SP2EC Algorithm 2 arm selection."""

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
        beta: float = 0.1,
        draft_confidence_threshold: float = 0.7,
        dp_budget_fraction: float = 0.5,
        dynamic_draft_stopping: bool = True,
        include_zero_skip_arm: bool = False,
        arm_index_start: Optional[int] = None,
        arm_index_end: Optional[int] = None,
        scoring_mode: str = "legacy",
    ):
        super().__init__(
            env=env,
            gamma=gamma,
            skip_budget_M=skip_budget_M,
            optimize_interval=optimize_interval,
            coefficients=coefficients,
            sim_threshold=sim_threshold,
            num_arms=num_arms,
            enable_sp2ec=True,
            tree=tree,
            beta=beta,
            draft_confidence_threshold=draft_confidence_threshold,
            dp_budget_fraction=dp_budget_fraction,
            dynamic_draft_stopping=dynamic_draft_stopping,
            include_zero_skip_arm=include_zero_skip_arm,
            arm_index_start=arm_index_start,
            arm_index_end=arm_index_end,
            scoring_mode=scoring_mode,
        )
