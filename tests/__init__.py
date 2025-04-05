from dataclasses import dataclass
from typing import Literal

import taichi
import taichi._kernels
import diff_physics
from diff_physics.common.util import Vec2I, multiply
from diff_physics.editor import Edges, Editor, Renderable
import diff_physics.energy
import diff_physics.energy.string
from diff_physics.pcg import X0Y, Attribute, Grid2Prim0, Grid2Prim1, Norm
import diff_physics.solver
import diff_physics.solver.PD
from taichi_hint.wrap.linear_algbra import Vec
from taichi_hint.wrap.ndarray import NDArray
from taichi_lib.common import Box2I

taichi.init()

g2p0 = Grid2Prim0(Box2I(Vec2I(0, 0), Vec2I(4, 4)))
num = g2p0.num
positions = Attribute[Vec](g2p0, [X0Y()]).array
g2p1 = Grid2Prim1(g2p0)
rest_lengths = Norm[Literal[3]](g2p1, positions).norms
multiply(rest_lengths, 0.9)

@dataclass
class Data:
    num: int
    positions: NDArray[Vec, Literal[1]]
    solver: diff_physics.solver.PD.Data
    string: diff_physics.energy.string.Data


edges = g2p1.indices
string_data = diff_physics.energy.string.Data(1, edges, rest_lengths)

masses = NDArray[float, Literal[1]].zero(num)
masses.fill(1)
solver_data = diff_physics.solver.PD.Data(
    0.02,
    NDArray[Vec, Literal[1]].zero(num),
    masses,
    [diff_physics.energy.string.Energy()],
)

data = Data(num, positions, solver_data, string_data)

solver = diff_physics.solver.PD.Solver()
solver.set_data(data)


def run():
    solver.step()


editor = Editor()
editor.renderables.append(Edges(positions=positions, indices=g2p1.indices))
editor.run(run)
