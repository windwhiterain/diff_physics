from dataclasses import dataclass
from typing import Literal, get_args
import taichi as ti
import taichi.math as tm

from taichi_hint.wrap import wrap
from taichi_hint.scope import pyfunc, func
from taichi_hint.struct import Struct
from taichi_hint.wrap.linear_algbra import LinearAlgbra


@wrap
@dataclass
class Bound[Dim, Item](Struct):
    min: LinearAlgbra[Literal[1], Dim, Item]
    max: LinearAlgbra[Literal[1], Dim, Item]

    @pyfunc
    def shape(self) -> LinearAlgbra[Literal[1], Dim, Item]:
        return self.max-self.min

    @pyfunc
    def size(self) -> Item:
        return self.shape().cumprod()

    @func
    def ndrange(self):
        dim = ti.static(self.min.n)
        subscript = [(0,0)]*dim
        for i in ti.static(range(dim)):
            subscript[i] = (self.min[i], self.max[i])
        return ti.grouped(ti.ndrange(*subscript))


Box2 = Bound[Literal[2], float]
Box = Bound[Literal[3], float]
Box2I = Bound[Literal[2], int]
BoxI = Bound[Literal[3], int]
