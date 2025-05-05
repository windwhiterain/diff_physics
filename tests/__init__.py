from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import taichi
import diff_physics as dp
from diff_physics.common.entity import Frame
from diff_physics.common.util import Vec2I, add, add_element, multiply
from diff_physics.editor import Edges, Editor, Renderable
from diff_physics.editor.renderable import Points, Vectors
import diff_physics.energy
import diff_physics.energy.string
from diff_physics.pcg import X0Y, Add, Attribute, Grid2Prim0, Grid2Prim1, Norm
from taichi_hint.scope import kernel
from taichi_hint.wrap.linear_algbra import Vec, Vec4
from taichi_hint.wrap.ndarray import NDArray
from taichi_lib.common import Box2I

taichi.init(arch="gpu")

g2p0 = Grid2Prim0(Box2I(Vec2I(0, 0), Vec2I(12, 20)))
positions = Attribute[Vec](g2p0, [X0Y(), Add(Vec(-5, 0, -9))]).array
g2p1 = Grid2Prim1(g2p0)
rest_lengths = Norm[Literal[3]](g2p1, positions).norms
# multiply(rest_lengths, 0.7)


@dataclass
class Data(dp.energy.string.Data, dp.solver.PD.Data):
    pass


edges = g2p1.indices
data_string = diff_physics.energy.string.StringData(g2p1.num, edges, rest_lengths)

masses = NDArray[float, Literal[1]].zero(g2p0.num)
masses.fill(1)

frame = Frame(positions, NDArray[Vec, Literal[1]].zero(g2p0.num))
target_frame = deepcopy(frame)


@kernel
def update_target_frame(positions: NDArray[Vec, Literal[1]]):
    for i in range(g2p0.num):
        position = positions[i]
        positions[i] = (
            taichi.math.rotation3d(0.0, 0.0, taichi.math.pi / 2) @ Vec4(position, 1)
        ).xyz


update_target_frame(target_frame.positions)

data_solver = dp.solver.base.SolverData(
    [diff_physics.energy.string.Energy()],
    0.2,
    objectives=[
        dp.objective.equal.Objective(target_frame, dp.common.entity.Mask(True, False))
    ],
    time=4,
    height_ground=-100,
    gravity=0,
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
optimize = solver.optimize(dp.common.entity.Mask(False, True), scale_descent=0.5)


def run():
    for action in optimize:
        print(f"frame {solver.id_frame}")
        yield action
    # solver.step()
    # solver.id_frame += 1


editor = Editor(Path(__file__).parent)
editor.renderables.append(Points(positions, Vec(1, 0, 0)))
editor.renderables.append(Edges(positions, g2p1.indices, Vec(0, 0, 1)))
editor.renderables.append(Vectors(positions, frame.velocities, 0.8, Vec(0.5, 0.5, 0)))
editor.run(run())
