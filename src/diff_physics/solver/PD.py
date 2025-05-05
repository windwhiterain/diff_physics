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
    get_reduce_ret,
    maximul_element,
    multiply,
    get_vec,
    norm_sqr,
    set_vec,
    substract,
    substract_b,
    sum_sqr,
)
from diff_physics.energy.base import Energy, System
from diff_physics.solver.base import Data as BaseData
from taichi_hint.argpack import ArgPack
from taichi_hint.scope import func, kernel
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
    time: float
    velocities: NDArray[Vec, Literal[1]]
    masses: NDArray[float, Literal[1]]
    # linear eq
    x: NDArray[float, Literal[1]]
    b: NDArray[float, Literal[1]]
    vec: NDArray[float, Literal[1]]
    positions_iter: NDArray[Vec, Literal[1]]
    velocities_iter: NDArray[Vec, Literal[1]]
    x_iter: NDArray[float, Literal[1]]
    # contact
    force_norm_contact: NDArray[float, Literal[1]]

class Solver(BaseSolver):
    data: Data  # type: ignore
    A: taichi.linalg.SparseMatrix
    ATA: taichi.linalg.SparseMatrix
    AT: taichi.linalg.SparseMatrix
    GB: taichi.linalg.SparseMatrix
    sparse_solver: taichi.linalg.SparseSolver

    @override
    def set_data(self, data: Data) -> None:  # type: ignore
        super().set_data(data)
        self.dim_b = 0
        self.num_A = 0
        self.num_grad_b = 0
        self.frame_iter = deepcopy(data.frame)
        self.force_norm_contact = NDArray[float, Literal[1]].zero(data.num)
        for energy in self.data.solver.energies:
            data_energy = copy(data)
            data_energy.frame = self.frame_iter
            energy.set_data(data_energy)
            self.dim_b += energy.dim_b()
            self.num_A += energy.num_A()
            self.num_grad_b += energy.num_grad_b()
        self.arg = Arg(
            data.num,
            data.frame.positions,
            self.data.solver.time_delta,
            0.0,
            self.data.frame.velocities,
            self.data.masses,
            NDArray[float, Literal[1]].zero(data.num * 3),
            NDArray[float, Literal[1]].zero(self.dim_b),
            NDArray[float, Literal[1]].zero(data.num * 3),
            self.frame_iter.positions,
            self.frame_iter.velocities,
            NDArray[float, Literal[1]].zero(data.num * 3),
            self.force_norm_contact,
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
        mat = self.M_ti2 + self.AT @ self.A
        self.sparse_solver = taichi.linalg.SparseSolver(solver_type="LLT")
        self.sparse_solver.analyze_pattern(mat)
        self.sparse_solver.factorize(mat)

    @override
    def step(self) -> None:
        self.arg.time = self.arg.time_delta * self.id_frame
        self.frame_iter.positions.copy_from(self.data.frame.positions)
        self.frame_iter.velocities.copy_from(self.data.frame.velocities)
        for i in range(self.data.solver.num_iteration_contact_force):
            for j in range(self.data.solver.num_iteration):
                offset = 0
                for energy in self.data.solver.energies:
                    energy.fill_b(self.arg.b, offset)
                    offset += energy.dim_b()
                self.update_vec()
                x_iter_next = self.sparse_solver.solve(self.arg.vec)
                self.update_frame_iter(x_iter_next)
            loss_contact = self.loss_contact()
            if loss_contact > 0:
                loss_contact_grad_force_contact = self.loss_contact_gard_force_contact(
                    loss=loss_contact
                )
                loss_contact_grad_force_norm_contact = (
                    self.loss_contact_grad_force_norm_contact(
                        loss_contact_grad_force_contact
                    )
                )
                force_sum_sqr_contact = sum_sqr(loss_contact_grad_force_norm_contact)
                if force_sum_sqr_contact > 0:
                    substract(
                        self.force_norm_contact,
                        multiply(
                            devide(
                                loss_contact_grad_force_norm_contact,
                                force_sum_sqr_contact,
                            ),
                            loss_contact,
                        ),
                    )
                maximul_element(self.force_norm_contact, 0)
            else:
                break
        self.update_frame()

    @override
    def back_propagation(self) -> None:
        self.frame_iter.positions.copy_from(self.data.frame.positions)
        self.frame_iter.velocities.copy_from(self.data.frame.velocities)
        grad_b = self.grad_b()
        ATGB = self.AT @ grad_b
        l_x = NDArray[Vec, Literal[1]].zero(self.data.num * 3)
        grad_x = flatten(self.grad_frame.positions)
        for i in range(self.data.solver.num_iteration_back_propagation):
            vector = add_b(grad_x, ATGB @ l_x)
            l_x = self.sparse_solver.solve(vector)
        l_dx = NDArray[Vec, Literal[1]].zero(self.data.num * 3)
        grad_dx = devide(
            flatten(self.grad_frame.velocities), self.data.solver.time_delta
        )
        for i in range(self.data.solver.num_iteration_back_propagation):
            vector = add_b(grad_dx, ATGB @ l_dx)
            l_dx = self.sparse_solver.solve(vector)
        self.grad_frame.positions = fold(
            add(self.M_ti2 @ l_x, (ATGB - self.ATA) @ l_dx)
        )
        self.grad_frame.velocities = fold(
            multiply(
                add(self.M_ti2 @ l_x, self.M_ti2 @ l_dx), self.data.solver.time_delta
            )
        )

    def grad_b(self):
        builder_grad_b = taichi.linalg.SparseMatrixBuilder(
            self.dim_b, self.data.num * 3, self.num_grad_b
        )
        offset = 0
        for energy in self.data.solver.energies:
            energy.build_grad_b(builder_grad_b, offset)
            offset += energy.dim_b()
        return builder_grad_b.build()

    def loss_contact(self) -> float:
        ret = get_reduce_ret()
        self._loss_contact(self.arg, ret)
        return ret[None]

    @kernel
    def _loss_contact(self, arg: Arg, reduce_ret: taichi.template()):
        reduce_ret[None] = 0.0
        for i in range(arg.num):
            loss = 0.0
            if (
                arg.positions_iter[i][2] - self.data.solver.height_ground < 0
                and arg.velocities_iter[i][2] < 0
                or arg.force_norm_contact[i] > 0
            ):
                loss = (arg.positions_iter[i][2] - self.data.solver.height_ground) ** 2
            reduce_ret[None] += loss

    @func
    def loss_contact_grad_x(
        self,
        positions: NDArray[Vec, Literal[1]],
        velocities: NDArray[Vec, Literal[1]],
        force_norm: NDArray[float, Literal[1]],
        index: int,
    ) -> Vec:
        ret = Vec(0)
        if (
            positions[index][2] - self.data.solver.height_ground < 0
            and velocities[index][2] < 0
            or force_norm[index] > 0
        ):
            ret = Vec(0, 0, 2 * (positions[index][2] - self.data.solver.height_ground))
        return ret

    def loss_contact_gard_force_contact(self, loss: float):
        ret = NDArray[float, Literal[1]].zero(self.arg.num * 3)
        grad_b = self.grad_b()
        ATGB = self.AT @ grad_b
        for _ in range(self.data.solver.num_iteration_contact_force_grad):
            vector = self.vector_contact(ret, ATGB, loss)
            ret.copy_from(self.sparse_solver.solve(vector))
        return ret

    def vector_contact(self, solution, ATGB, loss: float):
        vector = ATGB @ solution
        self._vector_contact(self.arg, vector, loss)
        return vector

    @kernel
    def _vector_contact(
        self, arg: Arg, vector: NDArray[float, Literal[1]], loss: float
    ):
        for i in range(arg.num):
            add_vec(
                vector,
                i * 3,
                self.loss_contact_grad_x(
                    arg.positions_iter, arg.velocities_iter, arg.force_norm_contact, i
                ),
            )

    def loss_contact_grad_force_norm_contact(
        self, loss_contact_grad_force_contact: NDArray[float, Literal[1]]
    ):
        ret = NDArray[float, Literal[1]].zero(self.data.num)
        self._loss_contact_grad_force_norm_contact(
            self.arg, loss_contact_grad_force_contact, ret
        )
        return ret

    @kernel
    def _loss_contact_grad_force_norm_contact(
        self,
        arg: Arg,
        loss_contact_grad_force_contact: NDArray[float, Literal[1]],
        ret: NDArray[float, Literal[1]],
    ):
        for i in range(arg.num):
            if (
                arg.positions_iter[i][2] - self.data.solver.height_ground < 0
                and arg.velocities_iter[i][2] < 0
                or arg.force_norm_contact[i] > 0
            ):
                ret[i] = Vec(0, 0, 1).dot(
                    get_vec(loss_contact_grad_force_contact, i * 3)
                )

    def update_frame_iter(self, x_iter_next: NDArray[float, Literal[1]]):
        self.arg.x_iter.copy_from(x_iter_next)
        self._update_frame_iter(self.arg, x_iter_next)

    @kernel
    def _update_frame_iter(self, arg: Arg, x_iter_next: NDArray[float, Literal[1]]):
        for i in range(arg.num):
            position_iter_next = get_vec(x_iter_next, i * 3)
            arg.velocities_iter[i] = (
                position_iter_next - arg.positions[i]
            ) / arg.time_delta
            arg.positions_iter[i] = position_iter_next

    def update_frame(self) -> None:
        self.data.frame.positions.copy_from(self.frame_iter.positions)
        self.data.frame.velocities.copy_from(self.frame_iter.velocities)

    def update_vec(self) -> None:
        ATb = self.AT @ self.arg.b
        self.arg.vec.copy_from(ATb)
        self._update_vec(self.arg)
        self.add_force_gravity(self.arg)
        self.add_force_contact(self.arg)

    @kernel
    def _update_vec(self, arg: Arg):
        for i in range(arg.num):
            mass = arg.masses[i]
            position = arg.positions[i]
            velocity = arg.velocities[i]
            vec_idx = i*3
            add_vec(
                arg.vec,
                vec_idx,
                mass
                * (
                    arg.time_delta ** (-1) * velocity
                    + arg.time_delta ** (-2) * position
                ),
            )

    @kernel
    def add_force_gravity(self, arg: Arg):
        for i in range(arg.num):
            add_vec(arg.vec, i * 3, Vec(0, 0, self.data.solver.gravity) * arg.masses[i])

    @kernel
    def add_force_contact(self, arg: Arg):
        for i in range(arg.num):
            add_vec(arg.vec, i * 3, Vec(0, 0, 1) * arg.force_norm_contact[i])

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
