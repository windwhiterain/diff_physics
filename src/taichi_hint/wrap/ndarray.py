from typing import Any, Literal, TypeVar, get_args, override

import numpy
import taichi
from taichi_hint.wrap.common import Wrap, wrap
from taichi_hint.wrap.linear_algbra import Algbra, LinearAlgbra, Number


@wrap
class NDArray[Item: Algbra, Dim](Wrap):
    annotation_only = True

    @classmethod
    def zero(cls, shape: Any) -> "NDArray[Item, Dim]":
        return taichi.ndarray(cls.Item, shape)  # type: ignore

    @staticmethod
    @override
    def value(specialization: Any) -> Any:
        item, dim_l = specialization.Item, specialization.Dim
        if isinstance(item, TypeVar):
            item = None
        if isinstance(dim_l, TypeVar):
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
