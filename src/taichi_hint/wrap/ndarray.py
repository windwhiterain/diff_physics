from typing import Any, Iterable, Literal, TypeVar, get_args, override

import numpy
import taichi
from taichi_hint.util import is_solid_type
from taichi_hint.wrap.common import Wrap, wrap
from taichi_hint.wrap.linear_algbra import Algbra, LinearAlgbra, Number


@wrap
class NDArray[Item: Algbra, Dim](Wrap):
    shape: list[int]

    annotation_only = True

    @classmethod
    def zero(cls, shape: Any) -> "NDArray[Item, Dim]":
        return taichi.ndarray(cls.Item, shape)  # type: ignore

    @staticmethod
    @override
    def solidize(specialization: Any) -> Any:
        item, dim_l = specialization.Item, specialization.Dim
        if not is_solid_type(item):
            item = None
        if not is_solid_type(dim_l):
            dim = None
        else:
            (dim,) = get_args(dim_l)
        return taichi.types.ndarray(item, dim)

    def __getitem__(self, index: LinearAlgbra[Literal[1], Dim, int] | int) -> Item: ...
    def __setitem__(
        self, index: LinearAlgbra[Literal[1], Dim, int] | int, value: Item
    ): ...

    def fill(self, value: Item) -> None: ...

    def to_numpy(self) -> numpy.ndarray: ...
    def copy_from(self, other: "NDArray[Item, Dim]") -> None: ...
