from dataclasses import dataclass
from typing import Literal

from taichi_hint.wrap.linear_algbra import Vec, Vec2I
from taichi_hint.wrap.ndarray import NDArray


@dataclass
class Renderable: ...


@dataclass
class Points(Renderable):
    positions: NDArray[Vec, Literal[1]]


@dataclass
class Edges(Points):
    indices: NDArray[Vec2I, Literal[1]]
