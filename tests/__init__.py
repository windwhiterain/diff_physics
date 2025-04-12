from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import taichi
import diff_physics as dp
from diff_physics.common.entity import Frame
from diff_physics.common.util import Vec2I, add, add_element, multiply
from diff_physics.editor import Edges, Editor, Renderable
import diff_physics.energy
import diff_physics.energy.string
from diff_physics.pcg import X0Y, Attribute, Grid2Prim0, Grid2Prim1, Norm
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
class Data(dp.energy.string.Data, dp.solver.PD.Data):
    pass


edges = g2p1.indices
data_string = diff_physics.energy.string.StringData(g2p1.num, edges, rest_lengths)

masses = NDArray[float, Literal[1]].zero(g2p0.num)
masses.fill(1)

frame = Frame(positions, NDArray[Vec, Literal[1]].zero(g2p0.num))
target_frame = deepcopy(frame)
multiply(target_frame.positions, Vec(0.5))
add_element(target_frame.positions, Vec(-1, 0, -1))
data_solver = dp.solver.base.SolverData(
    [diff_physics.energy.string.Energy()],
    0.02,
    objectives=[
        dp.objective.equal.Objective(target_frame, dp.common.entity.Mask(True, False))
    ],
    num_frame=50,
)

data = Data(
    g2p0.num,
    frame,
    masses,
    data_solver,
    data_string,
)


solver = diff_physics.solver.PD.Solver()
assert data.frame.positions is positions
solver.set_data(data)


assert data.frame.positions is positions
optimize = solver.optimize(diff_physics.common.entity.Mask(False, True))

def run():

    next(optimize)
    print(f"frame {solver.id_frame}")


editor = Editor()
editor.renderables.append(Edges(positions=positions, indices=g2p1.indices))
editor.run(run)
