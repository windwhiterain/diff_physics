from abc import abstractmethod
from typing import Any, Literal
import taichi
from taichi.linalg.sparse_matrix import SparseMatrix
from diff_physics.common.system import System
from taichi_hint.wrap.ndarray import NDArray


@taichi.data_oriented
class Energy(System):
    @abstractmethod
    def dim_b(self) -> int: ...

    @abstractmethod
    def num_A(self) -> int: ...
    @abstractmethod
    def num_grad_b(self) -> int: ...

    @abstractmethod
    def build_A(self, A: taichi.types.sparse_matrix_builder(), offset: int): ...

    @abstractmethod
    def A_forward(
        self,
        input: NDArray[float, Literal[1]],
        input_offset: int,
        output: NDArray[float, Literal[1]],
        output_offset: int,
    ): ...

    @abstractmethod
    def A_backward(
        self,
        input: NDArray[float, Literal[1]],
        input_offset: int,
        output: NDArray[float, Literal[1]],
        output_offset: int,
    ): ...

    @abstractmethod
    def fill_b(self, b: NDArray[float, Literal[1]], offset: int): ...

    @abstractmethod
    def build_grad_b(
        self, grad_b: taichi.types.sparse_matrix_builder(), offset: int
    ): ...
