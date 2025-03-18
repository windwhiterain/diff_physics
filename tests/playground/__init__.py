from typing import Annotated, Any, overload, override
import numpy
import taichi
import taichi.lang
import taichi.lang.ast
import taichi.lang.ast.hint
from diff_physics.editor import Editor
from diff_physics.editor.renderable import Edges
from diff_physics.pcg import X0Y, Attribute, Grid2Prim0, Grid2Prim1, kernel
from taichi_hint.wrap.linear_algbra import Vec, Vec2, Vec2I
from taichi_hint.wrap.ndarray import NDArray
from taichi_lib.common import Box2I

taichi.init(taichi.lang.gpu)

g2p0 = Grid2Prim0(Box2I(Vec2I(0, 0), Vec2I(4, 4)))
positions = Attribute[Vec](g2p0, [X0Y()]).array
print(positions.to_numpy())
g2p1 = Grid2Prim1(g2p0)

editor = Editor()
editor.renderables.append(Edges(positions, g2p1.indices))
editor.run()
