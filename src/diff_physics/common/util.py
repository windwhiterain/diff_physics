from typing import Literal
import taichi
from diff_physics.pcg import LinearAlgbra
from taichi_hint.template import Templated
from taichi_hint.wrap.linear_algbra import Algbra, Mat, Vec, Vec2I
from taichi_hint.scope import func, kernel
from taichi_hint.wrap.ndarray import NDArray


@func
def add_diag(mat: taichi.template(), index: Vec2I, item: float):
    for i in taichi.static(range(3)):
        item_index = index+i
        mat[item_index[0], item_index[1]] += item


@func
def add_mat(mat: taichi.template(), index: Vec2I, item: Mat):
    for I in taichi.static(taichi.grouped(taichi.ndrange(3, 3))):
        item_index = index + I
        mat[item_index[0], item_index[1]] += item[I]


def set_mat(mat: taichi.linalg.SparseMatrix, index: Vec2I, item: Mat):
    for I in taichi.static(taichi.grouped(taichi.ndrange(3, 3))):
        item_index = index + I
        mat[item_index[0], item_index[1]] = item[I]


@func
def set_vec(array: NDArray[float, Literal[1]], index: int, vec: Vec):
    for i in taichi.static(range(3)):
        array[index + i] = vec[i]


@func
def get_vec(array: NDArray[float, Literal[1]], index: int) -> Vec:
    ret = Vec(0)
    for i in taichi.static(range(3)):
        ret[i] = array[index + i]
    return ret


@func
def add_vec(array: NDArray[float, Literal[1]], index: int, vec: Vec):
    for i in taichi.static(range(3)):
        array[index + i] += vec[i]


def add[Item: Algbra, Dim](
    a: NDArray[Item, Dim], b: NDArray[Item, Dim]
) -> NDArray[Item, Dim]:
    _add(a, b)
    return a


@kernel
def _add[Item: Algbra, Dim](a: NDArray[Item, Dim], b: NDArray[Item, Dim]):
    for i in taichi.grouped(a):
        a[i] += b[i]


def add_b[Item: Algbra, Dim](
    a: NDArray[Item, Dim], b: NDArray[Item, Dim]
) -> NDArray[Item, Dim]:
    _add_b(a, b)
    return b


@kernel
def _add_b[Item: Algbra, Dim](a: NDArray[Item, Dim], b: NDArray[Item, Dim]):
    for i in taichi.grouped(a):
        b[i] += a[i] + b[i]


def add_element[Item: Algbra, Dim](
    a: NDArray[Item, Dim], b: Item
) -> NDArray[Item, Dim]:
    _add_element(a, b)
    return a


@kernel
def _add_element[Item: Algbra, Dim](a: NDArray[Item, Dim], b: Item):
    for i in taichi.grouped(a):
        a[i] += b


def substract[Item: Algbra, Dim](
    a: NDArray[Item, Dim], b: NDArray[Item, Dim]
) -> NDArray[Item, Dim]:
    _substract(a, b)
    return a


@kernel
def _substract[Item: Algbra, Dim](a: NDArray[Item, Dim], b: NDArray[Item, Dim]):
    for i in taichi.grouped(a):
        a[i] -= b[i]


def substract_b[Item: Algbra, Dim](
    a: NDArray[Item, Dim], b: NDArray[Item, Dim]
) -> NDArray[Item, Dim]:
    _substract_b(a, b)
    return b


@kernel
def _substract_b[Item: Algbra, Dim](a: NDArray[Item, Dim], b: NDArray[Item, Dim]):
    for i in taichi.grouped(a):
        b[i] = a[i] - b[i]


def multiply[Item: Algbra, Dim](a: NDArray[Item, Dim], b: Item) -> NDArray[Item, Dim]:
    _multiply(a, b)
    return a


@kernel
def _multiply[Item: Algbra, Dim](a: NDArray[Item, Dim], b: Item):
    for i in taichi.grouped(a):
        a[i] *= b


def devide[Item: Algbra, Dim](a: NDArray[Item, Dim], b: Item) -> NDArray[Item, Dim]:
    _devide(a, b)
    return a


@kernel
def _devide[Item: Algbra, Dim](a: NDArray[Item, Dim], b: Item):
    for i in taichi.grouped(a):
        a[i] /= b


def maximul_element[Item: Algbra, Dim](
    a: NDArray[Item, Dim], b: Item
) -> NDArray[Item, Dim]:
    _maximul_element(a, b)
    return a


@kernel
def _maximul_element[Item: Algbra, Dim](a: NDArray[Item, Dim], b: Item):
    for i in taichi.grouped(a):
        taichi.max(a[i], b)


def flatten(a: NDArray[Vec, Literal[1]]) -> NDArray[float, Literal[1]]:
    ret = NDArray[float, Literal[1]].zero(a.shape[0] * 3)
    _flatten(a, ret)
    return ret


@kernel
def _flatten(a: NDArray[Vec, Literal[1]], b: NDArray[float, Literal[1]]):
    for i in range(a.shape[0]):
        set_vec(b, i * 3, a[i])


def fold(a: NDArray[float, Literal[1]]) -> NDArray[Vec, Literal[1]]:
    ret = NDArray[Vec, Literal[1]].zero(a.shape[0] // 3)
    _fold(a, ret)
    return ret


@kernel
def _fold(a: NDArray[float, Literal[1]], b: NDArray[Vec, Literal[1]]):
    for i in range(b.shape[0]):
        b[i] = get_vec(a, i * 3)


reduce_ret = None


def get_reduce_ret():
    global reduce_ret
    if reduce_ret is None:
        reduce_ret = taichi.field(float, shape=())
    return reduce_ret


def sum[Item: LinearAlgbra](a: NDArray[Item, Literal[1]]) -> float:
    get_reduce_ret()[None] = 0
    _sum(a, reduce_ret)
    return get_reduce_ret()[None]


@kernel
def _sum[Item: LinearAlgbra](a: NDArray[Item, Literal[1]], b: taichi.template()):
    for i in range(a.shape[0]):
        b[None] += a[i].sum()


def norm_sqr[Item: LinearAlgbra](a: NDArray[Item, Literal[1]]) -> float:
    reduce_ret = get_reduce_ret()
    _norm_sqr(a, reduce_ret)
    return reduce_ret[None]


@kernel
def _norm_sqr[Item: LinearAlgbra](
    a: NDArray[Item, Literal[1]], reduce_ret: taichi.template()
):
    reduce_ret[None] = 0
    for i in range(a.shape[0]):
        reduce_ret[None] += a[i].norm_sqr()


def sum_sqr(a: NDArray[float, Literal[1]]) -> float:
    reduce_ret = get_reduce_ret()
    _sum_sqr(a, reduce_ret)
    return reduce_ret[None]


@kernel
def _sum_sqr(a: NDArray[float, Literal[1]], reduce_ret: taichi.template()):
    reduce_ret[None] = 0
    for i in range(a.shape[0]):
        reduce_ret[None] += a[i] ** 2
