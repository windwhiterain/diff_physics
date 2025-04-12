from copy import deepcopy
from typing import override
from diff_physics.common.entity import Frame, Mask
from diff_physics.common.util import norm_sqr
from diff_physics.objective.base import Objective as BaseObjective
from diff_physics.common.util import substract
from taichi_hint.wrap.linear_algbra import Vec
from dataclasses import dataclass


class Objective(BaseObjective):
    def __init__(self, frame: Frame, mask: Mask) -> None:
        super().__init__()
        self.frame = frame
        self.mask = mask

    @override
    def update(self, frame: Frame) -> tuple[Frame, float]:
        grad_frame = deepcopy(self.frame)
        if not self.mask.position:
            grad_frame.positions.fill(Vec(0))
        if not self.mask.velocity:
            grad_frame.velocities.fill(Vec(0))
        if self.mask.position:
            substract(grad_frame.positions, frame.positions)
        if self.mask.velocity:
            substract(grad_frame.velocities, frame.velocities)
        return (
            grad_frame,
            norm_sqr(grad_frame.positions) + norm_sqr(grad_frame.velocities),
        )
