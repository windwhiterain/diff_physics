
from typing import get_args
from numpy import ndarray
import taichi
from taichi.lang.util import taichi_scope

from taichi_hint.scope import kernel
from taichi_hint.template import Templated
from taichi_hint.wrap.linear_algbra import Vec2I, VecI
from taichi_lib.common import Box2I, BoxI

taichi.init()

@taichi.func
def grouped(x:taichi.template()):
    ret = [(0,0)]*x.n
    for i in taichi.static(range(x.n)):
        ret[i] = (0,x[i])
    return taichi.grouped(taichi.ndrange(*ret))


@kernel
def tt(x: Templated[BoxI]):
    for i in x.ndrange():
        print(i)


tt(BoxI(VecI(1,1,1), VecI(2,3,4)))
