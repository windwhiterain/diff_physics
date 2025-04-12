from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any, Literal, override

import taichi
import taichi.lang
import taichi.linalg.sparse_solver
from diff_physics.common.util import (
    Vec2I,
    add,
    add_b,
    add_diag,
    add_vec,
    devide,
    flatten,
    fold,
    multiply,
    get_vec,
    set_vec,
    substract,
    substract_b,
)
from diff_physics.energy.base import Energy, System
from diff_physics.solver.base import Data as BaseData
from taichi_hint.argpack import ArgPack
from taichi_hint.scope import kernel
from taichi_hint.util import repr_iterable
from taichi_hint.wrap import  wrap
from taichi_hint.wrap.linear_algbra import Vec
from taichi_hint.wrap.ndarray import NDArray
from diff_physics.solver.base import Solver as BaseSolver


@dataclass
class Data(BaseData):
    pass


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
    positions_iter: NDArray[Vec, Literal[1]]
    x_iter: NDArray[float, Literal[1]]


class Solver(BaseSolver):
    data: Data  # type: ignore
    A: taichi.linalg.SparseMatrix
    ATA: taichi.linalg.SparseMatrix
    AT: taichi.linalg.SparseMatrix
    GB: taichi.linalg.SparseMatrix
    sparse_solver: taichi.linalg.SparseSolver

    def set_data(self, data: Data) -> None:  # type: ignore
        super().set_data(data)
        self.dim_b = 0
        self.num_A = 0
        self.num_grad_b = 0
        frame_iter = deepcopy(data.frame)
        for energy in self.data.solver.energies:
            data_energy = copy(data)
            data_energy.frame = frame_iter
            energy.set_data(data_energy)
            self.dim_b += energy.dim_b()
            self.num_A += energy.num_A()
            self.num_grad_b += energy.num_grad_b()
        self.arg = Arg(
            data.num,
            data.frame.positions,
            self.data.solver.time_delta,
            self.data.frame.velocities,
            self.data.masses,
            NDArray[float, Literal[1]].zero(data.num * 3),
            NDArray[float, Literal[1]].zero(self.dim_b),
            NDArray[float, Literal[1]].zero(data.num * 3),
            frame_iter.positions,
            NDArray[float, Literal[1]].zero(data.num * 3),
        )
        self.update_x()
        self.arg.x_iter.copy_from(self.arg.x)
        A_buider = taichi.linalg.SparseMatrixBuilder(
            self.dim_b, data.num * 3, self.num_A
        )
        offset = 0
        for energy in self.data.solver.energies:
            energy.build_A(A_buider, offset)
            offset += energy.dim_b()
        self.A = A_buider.build()
        self.AT = self.A.transpose()
        self.ATA = self.AT @ self.A
        m_builder = taichi.linalg.SparseMatrixBuilder(
            self.arg.num*3, self.arg.num*3, self.arg.num*3)
        self.build_m(m_builder)
        M = m_builder.build()
        dt = self.arg.time_delta
        self.M_ti2 = M * (dt ** (-2))
        mat = self.M_ti2 + self.A.transpose() @ self.A
        self.sparse_solver = taichi.linalg.SparseSolver(solver_type="LLT")
        self.sparse_solver.analyze_pattern(mat)
        self.sparse_solver.factorize(mat)

    def step(self) -> None:
        for i in range(self.data.solver.num_iteration):
            offset = 0
            for energy in self.data.solver.energies:
                energy.fill_b(self.arg.b, offset)
                offset += energy.dim_b()
            self.update_vec()
            x_delta_next = self.sparse_solver.solve(self.arg.vec)
            self.update_x_positions_iter(x_delta_next)
        self.update_position_velocity_x(x_delta_next)

    @override
    def back_propagation(self) -> None:
        builder_grad_b = taichi.linalg.SparseMatrixBuilder(
            self.dim_b, self.data.num * 3, self.num_grad_b
        )
        offset = 0
        for energy in self.data.solver.energies:
            energy.build_grad_b(builder_grad_b, offset)
            offset += energy.dim_b()
        grad_b = builder_grad_b.build()
        ATGB = self.AT @ grad_b
        l_x = NDArray[Vec, Literal[1]].zero(self.data.num * 3)
        grad_x = flatten(self.grad_frame.positions)
        for i in range(self.data.solver.num_iteration_back_propagation):
            vector = add_b(grad_x, ATGB @ l_x)
            l_x = self.sparse_solver.solve(vector)
        l_dx = NDArray[Vec, Literal[1]].zero(self.data.num * 3)
        grad_dx = multiply(
            flatten(self.grad_frame.velocities), self.data.solver.time_delta
        )
        for i in range(self.data.solver.num_iteration_back_propagation):
            vector = add_b(grad_dx, ATGB @ l_dx)
            l_dx = self.sparse_solver.solve(vector)
        self.grad_frame.positions = fold(
            add(self.M_ti2 @ l_x, (self.ATA + ATGB) @ l_dx)
        )
        self.grad_frame.velocities = fold(
            devide(
                add(self.M_ti2 @ l_x, self.M_ti2 @ l_dx), self.data.solver.time_delta
            )
        )

    def update_x_positions_iter(self, x_delta_next: NDArray[float, Literal[1]]):
        self._update_x_positions_iter(self.arg, x_delta_next)

    @kernel
    def _update_x_positions_iter(
        self, arg: Arg, x_delta_next: NDArray[float, Literal[1]]
    ):
        for i in range(arg.num * 3):
            arg.x_iter[i] = arg.x[i] + x_delta_next[i]
        for i in range(arg.num):
            arg.positions_iter[i] = arg.positions[i] + get_vec(x_delta_next, i * 3)

    def update_position_velocity_x(
        self, x_delta_next: NDArray[float, Literal[1]]
    ) -> None:
        self._update_position_velocity(self.arg, x_delta_next)
        self.update_x()

    @kernel
    def _update_position_velocity(self, arg: Arg, dx: NDArray[float, Literal[1]]):
        for i in range(arg.num):
            dx_idx = i*3
            position_delta = get_vec(dx, dx_idx)
            arg.positions[i] += position_delta
            arg.velocities[i] = position_delta/self.arg.time_delta

    def update_vec(self) -> None:
        ATAx = self.ATA @ self.arg.x_iter
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
            add_vec(arg.vec, vec_idx, velocity * mass * (arg.time_delta ** (-1)))

    def update_x(self) -> None:
        self._update_x(self.arg)

    @kernel
    def _update_x(self, arg: Arg):
        for i in range(arg.num):
            position = arg.positions[i]
            x_idx = i*3
            set_vec(arg.x, x_idx, position)

    def build_m(self, m_builder: taichi.types.sparse_matrix_builder()) -> None:
        self._build_m(self.arg, m_builder)

    @kernel
    def _build_m(self, arg: Arg, m_builder: taichi.types.sparse_matrix_builder()):
        for i in range(arg.num):
            mass = arg.masses[i]
            m_idx = i*3
            add_diag(m_builder, Vec2I(m_idx, m_idx), mass)
