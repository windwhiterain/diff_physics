from dataclasses import dataclass
from typing import Literal

from diff_physics.common.util import Vec
from taichi_hint.wrap.ndarray import NDArray


@dataclass
class Mask:
    position: bool
    velocity: bool


@dataclass
class Frame:
    positions: NDArray[Vec, Literal[1]]
    velocities: NDArray[Vec, Literal[1]]
