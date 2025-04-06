from dataclasses import dataclass
from turtle import position
from typing import Any, Literal, override
import taichi

from diff_physics.common.util import add_diag, get_vec, set_vec
from taichi_hint.argpack import ArgPack
from taichi_hint.scope import kernel
from taichi_hint.wrap.linear_algbra import Vec, Vec2I
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
    positions: NDArray[Vec, Literal[1]]
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
    arg: Arg
    _b_dim: int
    _A_num: int

    @override
    def set_data(self, data: Data) -> None:
        self.data = data.string
        self.arg = Arg(data.positions,
                       data.string.num,
                       data.string.point_pairs, data.string.rest_len)
        self._b_dim = self.data.num*3
        self._A_num = self._b_dim*2

    @override
    def b_dim(self) -> int:
        return self._b_dim

    @override
    def A_num(self) -> int:
        return self._A_num

    @override
    def build_A(self, A: taichi.types.sparse_matrix_builder(), offset: int) -> None:
        self._build_A(self.arg, A, offset)

    @kernel
    def _build_A(self, arg: Arg, A: taichi.types.sparse_matrix_builder(), offset: int):
        for i in range(arg.num):
            mat_idx0 = i*3 + offset
            point_pair = arg.point_pairs[i]
            add_diag(A, 3, Vec2I(mat_idx0, point_pair[0]*3), -1)
            add_diag(A, 3, Vec2I(mat_idx0, point_pair[1]*3), 1)

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
                3,
                output_offset + output_idx,
                -get_vec(input, 3, input_offset + input_idx_pair[0])
                + get_vec(input, 3, input_offset + input_idx_pair[1]),
            )

    @override
    def fill_b(self,  b: NDArray[float, Literal[1]], offset: int) -> None:
        self._fill_b(self.arg, b, offset)

    @kernel
    def _fill_b(self, arg: Arg,  b: NDArray[float, Literal[1]], offset: int):
        for i in range(arg.num):
            point_pair = arg.point_pairs[i]
            rest_len = arg.rest_lens[i]
            position0, position1 = arg.positions[point_pair[0]
                                                 ], arg.positions[point_pair[1]]
            position_delta = position1 - position0

            position_delta_proj = position_delta.normalized()*rest_len
            array_idx = i * 3 + offset
            set_vec(b, 3, array_idx, position_delta_proj)
