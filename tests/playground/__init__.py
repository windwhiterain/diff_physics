import taichi

from diff_physics.common.util import substract, substract_b
from taichi_hint.wrap.linear_algbra import Vec

taichi.init()

a = taichi.ndarray(float, 16)
a.fill(1)
b = taichi.ndarray(float, 16)
b.fill(2)

c = substract_b(a, b)

print(c.to_numpy())
