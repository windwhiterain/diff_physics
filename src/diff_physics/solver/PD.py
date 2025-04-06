from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any, Literal, override

import taichi
import taichi.lang
import taichi.linalg.sparse_solver
from diff_physics.common.util import (
    Vec2I,
    add,
    add_diag,
    add_vec,
    multiply,
    get_vec,
    set_vec,
    substract,
)
from diff_physics.energy.base import Energy, System
from taichi_hint.argpack import ArgPack
from taichi_hint.scope import kernel
from taichi_hint.util import repr_iterable
from taichi_hint.wrap import  wrap
from taichi_hint.wrap.linear_algbra import Vec
from taichi_hint.wrap.ndarray import NDArray


@dataclass
class Data:
    num: int
    positions: NDArray[Vec, Literal[1]]
    velocities: NDArray[Vec, Literal[1]]
    masses: NDArray[float, Literal[1]]


@wrap
@dataclass
class Arg(ArgPack):
    num: int
    positions: NDArray[Vec, Literal[1]]
    # solver
    time_delta: float
    velocities: NDArray[Vec, Literal[1]]
    masses: NDArray[float, Literal[1]]
    # linear eq
    x: NDArray[float, Literal[1]]
    b: NDArray[float, Literal[1]]
    vec: NDArray[float, Literal[1]]
    positions_prev: NDArray[Vec, Literal[1]]
    x_prev: NDArray[float, Literal[1]]


class Solver(System):
    data: Data
    arg: Arg
    b_dim: int
    A_num: int
    A: taichi.linalg.SparseMatrix
    ATA: taichi.linalg.SparseMatrix
    AT: taichi.linalg.SparseMatrix
    sparse_solver: taichi.linalg.SparseSolver

    def __init__(
        self, energies: list[Energy], time_delta: float, iteration: int
    ) -> None:
        super().__init__()
        self.energies = energies
        self.time_delta = time_delta
        self.iteration = iteration

    @override
    def set_data(self, data: Data) -> None:
        self.data = data
        self.b_dim = 0
        self.A_num = 0
        positions_prev = deepcopy(data.positions)
        for energy in self.energies:
            data_energy = copy(data)
            data_energy.positions = positions_prev
            energy.set_data(data)
            self.b_dim += energy.b_dim()
            self.A_num += energy.A_num()
        self.arg = Arg(
            data.num,
            data.positions,
            self.time_delta,
            self.data.velocities,
            self.data.masses,
            NDArray[float, Literal[1]].zero(data.num * 3),
            NDArray[float, Literal[1]].zero(self.b_dim),
            NDArray[float, Literal[1]].zero(data.num * 3),
            deepcopy(data.positions),
            NDArray[float, Literal[1]].zero(data.num * 3),
        )
        self.update_x()
        self.arg.x_prev.copy_from(self.arg.x)
        A_buider = taichi.linalg.SparseMatrixBuilder(
            self.b_dim, data.num * 3, self.A_num
        )
        offset = 0
        for energy in self.energies:
            energy.build_A(A_buider, offset)
            offset += energy.b_dim()
        self.A = A_buider.build()
        self.AT = self.A.transpose()
        self.ATA = self.AT @ self.A
        m_builder = taichi.linalg.SparseMatrixBuilder(
            self.arg.num*3, self.arg.num*3, self.arg.num*3)
        self.build_m(m_builder)
        m = m_builder.build()
        dt = self.arg.time_delta
        mat = m * (dt ** (-2)) + self.A.transpose() @ self.A
        self.sparse_solver = taichi.linalg.SparseSolver(solver_type="LLT")
        self.sparse_solver.analyze_pattern(mat)
        self.sparse_solver.factorize(mat)

    def step(self) -> None:
        for i in range(self.iteration):
            offset = 0
            for energy in self.energies:
                energy.fill_b(self.arg.b, offset)
                offset += energy.b_dim()
            self.update_vec()
            x_delta_next = self.sparse_solver.solve(self.arg.vec)
            self.update_x_positions_prev(x_delta_next)
        self.update_position_velocity_x(x_delta_next)

    def update_x_positions_prev(self, x_delta_next: NDArray[float, Literal[1]]):
        self._update_x_positions_prev(self.arg, x_delta_next)

    @kernel
    def _update_x_positions_prev(
        self, arg: Arg, x_delta_next: NDArray[float, Literal[1]]
    ):
        for i in range(arg.num * 3):
            arg.x_prev[i] = arg.x[i] + x_delta_next[i]
        for i in range(arg.num):
            arg.positions_prev[i] = arg.positions[i] + get_vec(x_delta_next, 3, i * 3)

    def update_position_velocity_x(
        self, x_delta_next: NDArray[float, Literal[1]]
    ) -> None:
        self._update_position_velocity(self.arg, x_delta_next)
        self.update_x()

    @kernel
    def _update_position_velocity(self, arg: Arg, dx: NDArray[float, Literal[1]]):
        for i in range(arg.num):
            dx_idx = i*3
            position_delta = get_vec(dx, 3, dx_idx)
            arg.positions[i] += position_delta
            arg.velocities[i] = position_delta/self.arg.time_delta

    def update_vec(self) -> None:
        ATAx = self.ATA @ self.arg.x_prev
        ATb = self.AT @ self.arg.b
        substract(ATb, ATAx)
        self.arg.vec.copy_from(ATb)
        self._update_vec(self.arg)

    @kernel
    def _update_vec(self, arg: Arg):
        for i in range(arg.num):
            mass = arg.masses[i]
            velocity = arg.velocities[i]
            vec_idx = i*3
            add_vec(arg.vec, 3, vec_idx,
                    velocity*mass*(arg.time_delta ** (-1)))

    def update_x(self) -> None:
        self._update_x(self.arg)

    @kernel
    def _update_x(self, arg: Arg):
        for i in range(arg.num):
            position = arg.positions[i]
            x_idx = i*3
            set_vec(arg.x, 3, x_idx, position)

    def build_m(self, m_builder: taichi.types.sparse_matrix_builder()) -> None:
        self._build_m(self.arg, m_builder)

    @kernel
    def _build_m(self, arg: Arg, m_builder: taichi.types.sparse_matrix_builder()):
        for i in range(arg.num):
            mass = arg.masses[i]
            m_idx = i*3
            add_diag(m_builder, 3, Vec2I(m_idx, m_idx), mass)
