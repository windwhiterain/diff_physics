from copy import deepcopy
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Literal, override
from diff_physics.common.entity import Frame, Mask
from diff_physics.common.system import System
from diff_physics.common.util import add, multiply, norm_sqr, sum
from diff_physics.editor import SaveImage
from diff_physics.energy.base import Energy
from diff_physics.objective.base import Objective
from taichi_hint.wrap.linear_algbra import Vec
from taichi_hint.wrap.ndarray import NDArray


@dataclass
class SolverData:
    energies: list[Energy]
    time_delta: float
    time: float = 1
    num_iteration: int = 4
    num_frame_cached_max: int = 10
    num_iteration_back_propagation: int = 4
    objectives: list[Objective] = field(default_factory=lambda: [])
    height_ground: float = 0
    num_iteration_contact_force: int = 4
    num_iteration_contact_force_grad: int = 4
    gravity: float = -0.01


@dataclass
class Data:
    num: int
    frame: Frame
    masses: NDArray[float, Literal[1]]
    solver: SolverData


@dataclass
class CachedFrame:
    frame: Frame
    id_frame_saved: int


class Solver(System):
    data: Data

    def set_data(self, data: Data) -> None:
        super().set_data(data)
        self.grad_frame = Frame(
            deepcopy(data.frame.positions), deepcopy(data.frame.velocities)
        )
        self.grad_frame.positions.fill(Vec(0))
        self.grad_frame.velocities.fill(Vec(0))
        self.num_frame = ceil(data.solver.time / data.solver.time_delta)
        self.refresh_cache()

        self.id_frame = 0

    def refresh_cache(self):
        self.frames_saved: list[Frame | None] = [None] * ceil(
            self.num_frame / self.data.solver.num_frame_cached_max
        )

        def cached_frames():
            for i in range(self.data.solver.num_frame_cached_max - 1):
                yield None

        self.frames_cached = list[CachedFrame | None](cached_frames())

        self.frames_saved[0] = Frame(
            deepcopy(self.data.frame.positions), deepcopy(self.data.frame.velocities)
        )

    def step(self): ...

    def evaluate(self, id_frame: int):
        if id_frame == self.id_frame:
            return
        id_frame_saved = id_frame // self.data.solver.num_frame_cached_max
        id_frame_cached = id_frame % self.data.solver.num_frame_cached_max

        def find_in_cache(
            frame_saved: Frame, id_frame_saved: int, id_frame_cached: int
        ):
            while True:
                if id_frame_cached == 0:
                    self.data.frame.positions.copy_from(frame_saved.positions)
                    self.data.frame.velocities.copy_from(frame_saved.velocities)
                    self.id_frame = (
                        id_frame_saved * self.data.solver.num_frame_cached_max
                    )
                    break
                if frame_cached := self.frames_cached[id_frame_cached - 1]:
                    if frame_cached.id_frame_saved == id_frame_saved:
                        self.data.frame.positions.copy_from(
                            frame_cached.frame.positions
                        )
                        self.data.frame.velocities.copy_from(
                            frame_cached.frame.velocities
                        )
                        self.id_frame = (
                            id_frame_saved * self.data.solver.num_frame_cached_max
                            + id_frame_cached
                        )
                        break
                id_frame_cached -= 1

        if frame_saved := self.frames_saved[id_frame_saved]:
            find_in_cache(frame_saved, id_frame_saved, id_frame_cached)
        else:
            while True:
                id_frame_saved -= 1
                if frame_saved := self.frames_saved[id_frame_saved]:
                    find_in_cache(
                        frame_saved,
                        id_frame_saved,
                        self.data.solver.num_frame_cached_max - 1,
                    )
                    break

        while self.id_frame < id_frame:
            self.step()
            self.id_frame += 1
            id_frame_saved = self.id_frame // self.data.solver.num_frame_cached_max
            id_frame_cached = self.id_frame % self.data.solver.num_frame_cached_max
            if id_frame_cached == 0:

                if frame_saved := self.frames_saved[id_frame_saved]:
                    pass
                else:
                    self.frames_saved[id_frame_saved] = Frame(
                        deepcopy(self.data.frame.positions),
                        deepcopy(self.data.frame.velocities),
                    )
            else:
                if frame_cached := self.frames_cached[id_frame_cached - 1]:
                    if frame_cached.id_frame_saved != id_frame_saved:
                        frame_cached.frame.positions.copy_from(
                            self.data.frame.positions
                        )
                        frame_cached.frame.velocities.copy_from(
                            self.data.frame.velocities
                        )
                        frame_cached.id_frame_saved = id_frame_saved
                else:
                    self.frames_cached[id_frame_cached - 1] = CachedFrame(
                        Frame(
                            deepcopy(self.data.frame.positions),
                            deepcopy(self.data.frame.velocities),
                        ),
                        id_frame_saved,
                    )
            yield

    def back_propagation(self): ...

    def optimize(self, mask: Mask, scale_descent: float = 0.5):
        num_iteration = -1
        while True:
            num_iteration += 1

            yield SaveImage(f"begin_{num_iteration}")
            self.grad_frame.positions.fill(Vec(0))
            self.grad_frame.velocities.fill(Vec(0))
            for _ in self.evaluate(self.num_frame - 1):
                yield
            yield SaveImage(f"end_{num_iteration}")

            loss = 0
            for objective in self.data.solver.objectives:
                grad_frame, _loss = objective.update(self.data.frame)
                add(self.grad_frame.positions, grad_frame.positions)
                add(self.grad_frame.velocities, grad_frame.velocities)
                loss += _loss
            if loss > 0:
                while self.id_frame > 0:
                    self.back_propagation()
                    for _ in self.evaluate(self.id_frame - 1):
                        yield
                    yield
                grad_norm_sqr = 0
                if mask.position:
                    grad_norm_sqr += norm_sqr(self.grad_frame.positions)
                if mask.velocity:
                    grad_norm_sqr += norm_sqr(self.grad_frame.velocities)
                step_scale = loss / grad_norm_sqr * scale_descent

                if mask.position:
                    add(
                        self.data.frame.positions,
                        multiply(self.grad_frame.positions, Vec(step_scale)),
                    )
                if mask.velocity:
                    add(
                        self.data.frame.velocities,
                        multiply(self.grad_frame.velocities, Vec(step_scale)),
                    )
                self.refresh_cache()
                yield
            else:
                break
