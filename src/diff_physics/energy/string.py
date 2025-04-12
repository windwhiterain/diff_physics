from dataclasses import dataclass
from turtle import position
from typing import Any, Literal, override
import taichi

from diff_physics.common.entity import Frame
from diff_physics.common.util import add_diag, add_mat, set_mat, get_vec, set_vec
from taichi_hint.argpack import ArgPack
from taichi_hint.scope import kernel
from taichi_hint.wrap.linear_algbra import Mat, Vec, Vec2I
from taichi_hint.wrap.ndarray import NDArray
from taichi_hint.wrap import wrap
from diff_physics.energy.base import Energy as BaseEnergy


@dataclass
class StringData:
    num: int
    point_pairs: NDArray[Vec2I, Literal[1]]
    rest_len: NDArray[float, Literal[1]]


@dataclass
class Data:
    frame: Frame
    string: StringData


@wrap
@dataclass
class Arg(ArgPack):
    positions: NDArray[Vec, Literal[1]]
    # energy
    num: int
    point_pairs: NDArray[Vec2I, Literal[1]]
    rest_lens: NDArray[float, Literal[1]]


class Energy(BaseEnergy):
    data: Data
    arg: Arg
    _b_dim: int
    _A_num: int
    _GB_num: int

    @override
    def set_data(self, data: Data) -> None:
        super().set_data(data)
        self.arg = Arg(
            data.frame.positions,
            data.string.num,
            data.string.point_pairs,
            data.string.rest_len,
        )
        self._b_dim = self.data.string.num * 3
        self._A_num = self.data.string.num * 2 * 3
        self._GB_num = self.data.string.num * 2 * 9

    @override
    def dim_b(self) -> int:
        return self._b_dim

    @override
    def num_A(self) -> int:
        return self._A_num

    @override
    def num_grad_b(self) -> int:
        return self._GB_num

    @override
    def build_A(
        self,
        A: taichi.types.sparse_matrix_builder(),
        offset: int,
    ) -> None:
        self._build_A(self.arg, A, offset)

    @kernel
    def _build_A(
        self,
        arg: Arg,
        A: taichi.types.sparse_matrix_builder(),
        offset: int,
    ):
        for i in range(arg.num):
            mat_idx0 = i*3 + offset
            point_pair = arg.point_pairs[i]
            add_diag(A, Vec2I(mat_idx0, point_pair[0] * 3), -1)
            add_diag(A, Vec2I(mat_idx0, point_pair[1] * 3), 1)

    @override
    def A_forward(self, input: NDArray[float, Literal[1]], input_offset: int, output: NDArray[float, Literal[1]], output_offset: int):
        self._A_forward(self.arg, input, input_offset, output, output_offset)

    @kernel
    def _A_forward(self, arg: Arg, input: NDArray[float, Literal[1]], input_offset: int, output: NDArray[float, Literal[1]], output_offset: int):
        for i in range(arg.num):
            point_pair = arg.point_pairs[i]
            input_idx_pair = point_pair*3
            output_idx = i*3
            set_vec(
                output,
                output_offset + output_idx,
                -get_vec(input, input_offset + input_idx_pair[0])
                + get_vec(input, input_offset + input_idx_pair[1]),
            )

    @override
    def fill_b(
        self,
        b: NDArray[float, Literal[1]],
        offset: int,
    ) -> None:
        self._fill_b(self.arg, b, offset)

    @override
    def build_grad_b(
        self,
        grad_b: taichi.linalg.SparseMatrix,
        offset: int,
    ) -> None:
        self._build_grad_b(self.arg, grad_b, offset)

    @kernel
    def _fill_b(
        self,
        arg: Arg,
        b: NDArray[float, Literal[1]],
        offset: int,
    ):
        for i in range(arg.num):
            point_pair = arg.point_pairs[i]
            rest_len = arg.rest_lens[i]
            position0, position1 = arg.positions[point_pair[0]
                                                 ], arg.positions[point_pair[1]]
            position_delta = position1 - position0
            position_delta_normalized = position_delta.normalized()
            position_delta_proj = position_delta_normalized * rest_len
            array_idx = i * 3 + offset
            set_vec(b, array_idx, position_delta_proj)

    @kernel
    def _build_grad_b(
        self,
        arg: Arg,
        grad_b: taichi.types.sparse_matrix_builder(),
        offset: int,
    ):
        for i in range(arg.num):
            point_pair = arg.point_pairs[i]
            rest_len = arg.rest_lens[i]
            position0, position1 = (
                arg.positions[point_pair[0]],
                arg.positions[point_pair[1]],
            )
            position_delta = position1 - position0
            position_delta_norm = position_delta.norm()
            position_delta_normalized = position_delta.normalized()
            array_idx = i * 3 + offset
            mat = (
                rest_len
                / position_delta_norm
                * (
                    taichi.Matrix.identity(float, 3)
                    - position_delta_normalized.outer_product(position_delta_normalized)
                )
            )
            add_mat(grad_b, Vec2I(array_idx, point_pair[1] * 3), mat)
            add_mat(grad_b, Vec2I(array_idx, point_pair[0] * 3), -mat)
