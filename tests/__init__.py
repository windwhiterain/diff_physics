from dataclasses import dataclass
from typing import Literal

import taichi
import diff_physics
from diff_physics.common.util import Vec2I
import diff_physics.energy
import diff_physics.energy.string
import diff_physics.solver
import diff_physics.solver.PD
from taichi_hint.wrap.linear_algbra import Vec
from taichi_hint.wrap.ndarray import NDArray

taichi.init()


@dataclass
class Data:
    num: int
    positions: NDArray[Vec, Literal[1]]
    solver: diff_physics.solver.PD.Data
    string: diff_physics.energy.string.Data


point_pairs = NDArray[Vec2I, Literal[1]].zero(1)
point_pairs[0] = Vec2I(0, 1)
rest_len = NDArray[float, Literal[1]].zero(1)
rest_len[0] = float(1)
string_data = diff_physics.energy.string.Data(1, point_pairs, rest_len)

masses = NDArray[float, Literal[1]].zero(2)
masses.fill(1)
solver_data = diff_physics.solver.PD.Data(
    0.02, NDArray[Vec, Literal[1]].zero(2), masses, [diff_physics.energy.string.Energy()])

positions = NDArray[Vec, Literal[1]].zero(2)
positions[0] = Vec(-1, 0, 0)
positions[1] = Vec(1, 0, 0)
data = Data(2, positions, solver_data, string_data)

solver = diff_physics.solver.PD.Solver()
solver.set_data(data)

for i in range(16):
    solver.step()
