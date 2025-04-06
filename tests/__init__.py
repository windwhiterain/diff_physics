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
positions = Attribute[Vec](g2p0, [X0Y()]).array
g2p1 = Grid2Prim1(g2p0)
rest_lengths = Norm[Literal[3]](g2p1, positions).norms
multiply(rest_lengths, 0.9)


@dataclass
class Data(diff_physics.energy.string.Data, diff_physics.solver.PD.Data):
    pass


edges = g2p1.indices
string_data = diff_physics.energy.string.StringData(g2p1.num, edges, rest_lengths)

masses = NDArray[float, Literal[1]].zero(g2p0.num)
masses.fill(1)

data = Data(
    g2p0.num, positions, NDArray[Vec, Literal[1]].zero(g2p0.num), masses, string_data
)

solver = diff_physics.solver.PD.Solver([diff_physics.energy.string.Energy()], 0.02, 4)
solver.set_data(data)


def run():
    solver.step()


editor = Editor()
editor.renderables.append(Edges(positions=positions, indices=g2p1.indices))
editor.run(run)
