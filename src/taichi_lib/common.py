from dataclasses import dataclass
from typing import Iterable, Literal, get_args
import taichi as ti
import taichi.math as tm

from taichi_hint.wrap import wrap
from taichi_hint.scope import pyfunc, func
from taichi_hint.struct import Struct
from taichi_hint.wrap.linear_algbra import LinearAlgbra, Number, cumprod


@wrap
@dataclass
class Bound[Dim, Item: Number](Struct):
    min: LinearAlgbra[Literal[1], Dim, Item]
    max: LinearAlgbra[Literal[1], Dim, Item]

    @pyfunc
    def shape(self) -> LinearAlgbra[Literal[1], Dim, Item]:
        return self.max - self.min

    @pyfunc
    def size(self) -> Item:
        return cumprod(self.shape())

    @func
    def iter(self) -> Iterable[LinearAlgbra[Literal[1], Dim, Item]]:
        dim = ti.static(self.min.n)
        subscript = [(0, 0)] * dim
        for i in ti.static(range(dim)):
            subscript[i] = (self.min[i], self.max[i])
        return ti.grouped(ti.ndrange(*subscript))  # type: ignore


Box2 = Bound[Literal[2], float]
Box = Bound[Literal[3], float]
Box2I = Bound[Literal[2], int]
BoxI = Bound[Literal[3], int]
