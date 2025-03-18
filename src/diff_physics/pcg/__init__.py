from abc import abstractmethod
from dataclasses import dataclass
import struct
from typing import Iterable, Literal, override
import taichi as ti

from taichi_hint.scope import func, kernel, pyfunc
from taichi_hint.specialize import specializable
from taichi_hint.wrap.linear_algbra import Algbra, Number, Vec, Vec2, Vec2I
from taichi_hint.wrap.ndarray import NDArray
from taichi_lib.common import Box2I


class Topology[Idx]():
    @abstractmethod
    def index(self, input: Idx) -> int: ...
    @abstractmethod
    def iter(self) -> Iterable[Idx]: ...
    @abstractmethod
    def num(self) -> int: ...


@ti.data_oriented
class Grid2Prim0(Topology[Vec2I]):
    def __init__(self, bound: Box2I) -> None:
        self.shape = bound.shape()
        self.bound = bound

    @pyfunc
    @override
    def index(self, input: Vec2I) -> int:
        local = input - self.bound.min
        return local[0] * self.shape[1] + local[1]

    @override
    @pyfunc
    def iter(self) -> Iterable[Vec2I]:
        return self.bound.iter()

    @override
    @pyfunc
    def num(self) -> int:
        return self.bound.size()


@ti.data_oriented
class Grid2Prim1:
    def __init__(self, prim0: Grid2Prim0) -> None:
        self.prim0 = prim0
        self.hx_grid = Grid2Prim0(
            Box2I(Vec2I(0), Vec2I(prim0.shape[0] - 1, prim0.shape[1]))
        )
        self.hy_grid = Grid2Prim0(
            Box2I(Vec2I(0), Vec2I(prim0.shape[0], prim0.shape[1] - 1))
        )
        self.cross_grid = Grid2Prim0(Box2I(Vec2I(0), prim0.shape - 1))
        self.num = self.hx_grid.num() + self.hy_grid.num() + self.cross_grid.num() * 2
        self.indices = NDArray[Vec2I, Literal[1]].zero(self.num)
        self.update_indices(self.indices)

    @kernel
    def update_indices(self, indices: NDArray[Vec2I, Literal[1]]):
        for idx in self.hx_grid.iter():
            indices_idx = self.hx_grid.index(idx)
            indices[indices_idx] = Vec2I(
                self.prim0.index(idx), self.prim0.index(idx + Vec2I(1, 0))
            )
        for idx in self.hy_grid.iter():
            indices_idx = self.hx_grid.num() + self.hy_grid.index(idx)
            indices[indices_idx] = Vec2I(
                self.prim0.index(idx), self.prim0.index(idx + Vec2I(0, 1))
            )
        for idx in self.cross_grid.iter():
            indices_idx = (
                self.hx_grid.num() + self.hy_grid.num() + self.cross_grid.index(idx) * 2
            )
            indices[indices_idx] = Vec2I(
                self.prim0.index(idx), self.prim0.index(idx + Vec2I(1, 1))
            )
            indices[indices_idx + 1] = Vec2I(
                self.prim0.index(idx + Vec2I(1, 0)),
                self.prim0.index(idx + Vec2I(0, 1)),
            )


@specializable
class Unary[Inp, Oup]:
    @abstractmethod
    def forward(self, input: Inp) -> Oup: ...


@ti.data_oriented
class X0Y(Unary[Vec2, Vec]):
    @override
    @func
    def forward(self, input: Vec2) -> Vec:
        return Vec(input[0], 0, input[1])


@ti.data_oriented
@specializable
class Attribute[Item: Algbra]:
    def __init__(self, topology: Topology, unaries: list[Unary]):
        self.topology = topology
        self.unaries = unaries
        self.array = NDArray[self.__specialization__.Item, Literal[1]].zero(
            topology.num()
        )
        self.update_array(self.array)

    @kernel
    def update_array(self, array: NDArray[Item, Literal[1]]):
        for i in self.topology.iter():
            value = i
            for unary in ti.static(self.unaries):
                value: ti.hide = unary.forward(value)
            array[self.topology.index(i)] = value
