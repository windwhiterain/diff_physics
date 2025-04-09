from copy import deepcopy
from dataclasses import dataclass
from math import ceil
from typing import Any, Literal, override
from diff_physics.common.system import System
from diff_physics.energy.base import Energy
from taichi_hint.wrap.linear_algbra import Vec
from taichi_hint.wrap.ndarray import NDArray


@dataclass
class SolverData:
    energies: list[Energy]
    time_delta: float
    num_frame: int = 100
    num_iteration: int = 4
    num_frame_cached_max: int = 10


@dataclass
class Data:
    num: int
    positions: NDArray[Vec, Literal[1]]
    velocities: NDArray[Vec, Literal[1]]
    masses: NDArray[float, Literal[1]]
    solver: SolverData


@dataclass
class Frame:
    positions: NDArray[Vec, Literal[1]]
    velocities: NDArray[Vec, Literal[1]]


@dataclass
class CachedFrame:
    frame: Frame
    id_frame_saved: int


class Solver(System):
    data: Data

    def set_data(self, data: Data) -> None:
        super().set_data(data)
        self.frames_saved: list[Frame | None] = [None] * ceil(
            data.solver.num_frame / data.solver.num_frame_cached_max
        )

        def cached_frames():
            for i in range(data.solver.num_frame_cached_max - 1):
                yield None

        self.frames_cached = list[CachedFrame | None](cached_frames())

        self.frames_saved[0] = Frame(
            deepcopy(self.data.positions), deepcopy(self.data.velocities)
        )
        self.id_frame = 0

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
                    self.data.positions.copy_from(frame_saved.positions)
                    self.data.velocities.copy_from(frame_saved.velocities)
                    self.id_frame = (
                        id_frame_saved * self.data.solver.num_frame_cached_max
                    )
                    break
                if frame_cached := self.frames_cached[id_frame_cached - 1]:
                    if frame_cached.id_frame_saved == id_frame_saved:
                        self.data.positions.copy_from(frame_cached.frame.positions)
                        self.data.velocities.copy_from(frame_cached.frame.velocities)
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

        count = 0
        while self.id_frame < id_frame:
            count += 1
            self.step()
            self.id_frame += 1
            id_frame_saved = self.id_frame // self.data.solver.num_frame_cached_max
            id_frame_cached = self.id_frame % self.data.solver.num_frame_cached_max
            if id_frame_cached == 0:

                if frame_saved := self.frames_saved[id_frame_saved]:
                    pass
                else:
                    self.frames_saved[id_frame_saved] = Frame(
                        deepcopy(self.data.positions), deepcopy(self.data.velocities)
                    )
            else:
                if frame_cached := self.frames_cached[id_frame_cached - 1]:
                    if frame_cached.id_frame_saved != id_frame_saved:
                        frame_cached.frame.positions.copy_from(self.data.positions)
                        frame_cached.frame.velocities.copy_from(self.data.velocities)
                        frame_cached.id_frame_saved = id_frame_saved
                else:
                    self.frames_cached[id_frame_cached - 1] = CachedFrame(
                        Frame(
                            deepcopy(self.data.positions),
                            deepcopy(self.data.velocities),
                        ),
                        id_frame_saved,
                    )
        print(count)
