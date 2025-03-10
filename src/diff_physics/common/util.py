from typing import Literal
import taichi
from taichi_hint.template import Templated
from taichi_hint.wrap.linear_algbra import Vec, Vec2I
from taichi_hint.scope import func, kernel
from taichi_hint.wrap.ndarray import NDArray


@func
def add_diag(mat: taichi.template(), dim: Templated[int], index: Vec2I, item: float):
    for i in taichi.static(range(dim)):
        item_index = index+i
        mat[item_index[0], item_index[1]] += item


@func
def set_vec(array: NDArray[float, Literal[1]], dim: Templated[int], index: int, vec: Vec):
    for i in taichi.static(range(dim)):
        array[index + i] = vec[i]


@func
def read_vec(array: NDArray[float, Literal[1]], dim: Templated[int], index: int) -> Vec:
    ret = Vec(0)
    for i in taichi.static(range(dim)):
        ret[i] = array[index + i]
    return ret


@func
def add_vec(array: NDArray[float, Literal[1]], dim: Templated[int], index: int, vec: Vec):
    for i in taichi.static(range(dim)):
        array[index + i] += vec[i]


@kernel
def substract[Item, Dim](a: NDArray[Item, Dim], b: NDArray[Item, Dim]):
    for i in taichi.grouped(a):
        a[i] -= b[i]


@kernel
def multiply[Item, Dim](a: NDArray[Item, Dim], b: Item):
    for i in taichi.grouped(a):
        a[i] *= b
