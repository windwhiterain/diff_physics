from dataclasses import dataclass
import taichi as ti

from taichi_hint.scope import kernel, pyfunc
from taichi_hint.wrap.linear_algbra import Vec2I, VecI
from taichi_hint.wrap.ndarray import NDArray
from taichi_lib.common import Box2I, BoxI
from tests import Literal

@ti.data_oriented
class Grid2Prim0:
    def __init__(self, bound: Box2I) -> None:
        self.num = bound.size()
        self.shape = bound.shape()
        self.bound = bound

    @pyfunc
    def sample(self, input: Vec2I) -> int:
        local = input - self.bound.min
        return local[0]*self.shape[1] + local[1]


@ti.data_oriented
class Grid2Prim1:
    def __init__(self, prim0: Grid2Prim0) -> None:
        self.prim0 = prim0
        self.hx_grid = Grid2Prim0(
            Box2I(Vec2I(0), Vec2I(prim0.shape[0]-1, prim0.shape[1])))
        self.hy_grid = Grid2Prim0(
            Box2I(Vec2I(0), Vec2I(prim0.shape[0]-1, prim0.shape[1])))
        self.hx_cross = Grid2Prim0(
            Box2I(Vec2I(0), prim0.shape-1))
        self.num = self.hx_grid.num + self.hy_grid.num + self.hx_cross.num*2
        self.indices = NDArray[Literal[1], Vec2I].zero(self.num)
    def update_indices(self, nd):...




