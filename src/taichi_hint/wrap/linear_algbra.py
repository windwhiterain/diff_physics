from typing import Any, Literal, get_args, override

import taichi
from taichi_hint.scope import pyfunc
from taichi_hint.util import is_solid_types
from taichi_hint.wrap.common import Wrap, wrap

Number = float | int


@wrap
class LinearAlgbra[Dim, Shape, Item: Number](Wrap):
    def __init__(self, *args):
        pass

    @staticmethod
    @override
    def solidize(specialization: Any) -> Any:
        dim_l, shape_l, item = (
            specialization.Dim,
            specialization.Shape,
            specialization.Item,
        )
        if not is_solid_types(dim_l, shape_l, item):
            return None
        shape = get_args(shape_l)
        dim = get_args(dim_l)[0]
        if dim == 1:
            ret = taichi.types.vector(shape[0], item)
            ret.cumprod = cumprod
            return ret
        elif dim == 2:
            return taichi.types.matrix(
                get_args(shape[0])[0], get_args(shape[1])[0], item
            )
        else:
            raise Exception()

    def __add__(
        self, o: "LinearAlgbra[Dim, Shape, Item] | Item"
    ) -> "LinearAlgbra[Dim, Shape, Item]": ...

    def __sub__(
        self, o: "LinearAlgbra[Dim, Shape, Item] | Item"
    ) -> "LinearAlgbra[Dim, Shape, Item]": ...

    def __mul__(self, o: Item) -> "LinearAlgbra[Dim, Shape, Item]": ...

    def __rmul__(self, o: Item) -> "LinearAlgbra[Dim, Shape, Item]": ...

    def __truediv__(self, o: Item) -> "LinearAlgbra[Dim, Shape, Item]": ...

    def __neg__(self) -> "LinearAlgbra[Dim, Shape, Item]": ...

    def __getitem__(
        self, index: "LinearAlgbra[Literal[1], Dim, int] | int"
    ) -> Item: ...
    def __setitem__(
        self, index: "LinearAlgbra[Literal[1], Dim, int] | int", value: Item
    ): ...

    def norm(self) -> float: ...

    def normalized(self) -> "LinearAlgbra[Dim, Shape, Item]": ...

    def cross(
        self, o: "LinearAlgbra[Dim, Shape, Item]"
    ) -> "LinearAlgbra[Dim, Shape, Item]": ...


Algbra = Number | LinearAlgbra


@pyfunc
def cumprod[Item: Number, Dim](vec: LinearAlgbra[Literal[1], Dim, Item]) -> Item:
    ret = vec[0]
    for i in taichi.static(range(1, vec.n)):
        ret *= vec[i]
    return ret  # type: ignore


Vec2I = LinearAlgbra[Literal[1], Literal[2], int]
VecI = LinearAlgbra[Literal[1], Literal[3], int]
Vec2 = LinearAlgbra[Literal[1], Literal[2], float]
Vec = LinearAlgbra[Literal[1], Literal[3], float]
Mat = LinearAlgbra[Literal[2], tuple[Literal[3], Literal[3]], float]
