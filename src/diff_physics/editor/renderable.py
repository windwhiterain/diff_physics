from dataclasses import dataclass
from turtle import pos
from typing import Literal, override

import taichi

from taichi_hint.scope import kernel
from taichi_hint.wrap.linear_algbra import Vec, Vec2I
from taichi_hint.wrap.ndarray import NDArray


class Renderable:
    color: Vec

    def __init__(self, color: Vec) -> None:
        self.color = color

    def update(self): ...


class Points(Renderable):

    def __init__(self, positions: NDArray[Vec, Literal[1]], color: Vec):
        super().__init__(color)
        self._positions = positions

    def positions(self) -> NDArray[Vec, Literal[1]]:
        return self._positions


class Edges(Points):

    def __init__(
        self,
        positions: NDArray[Vec, Literal[1]],
        indices: NDArray[Vec2I, Literal[1]],
        color: Vec,
    ):
        super().__init__(positions, color)
        self._indices = indices

    def indices(self) -> NDArray[Vec2I, Literal[1]]:
        return self._indices


@taichi.data_oriented
class Vectors(Edges):

    def __init__(
        self,
        positions: NDArray[Vec, Literal[1]],
        vectors: NDArray[Vec, Literal[1]],
        scale: float,
        color: Vec,
    ):
        Renderable.__init__(self, color)
        self.scale = scale
        self.vector_positions = positions
        self.vectors = vectors
        self._positions = NDArray[Vec, Literal[1]].zero(positions.shape[0] * 2)
        self._indices = NDArray[Vec2I, Literal[1]].zero(positions.shape[0])
        self.update_indices(self._indices)

    @kernel
    def update_indices(self, indices: NDArray[Vec2I, Literal[1]]):
        for i in range(self._indices.shape[0]):
            indices[i] = Vec2I(i * 2, i * 2 + 1)

    @override
    def update(self):
        self._update(self._positions, self.vector_positions, self.vectors)

    @kernel
    def _update(
        self,
        positions: NDArray[Vec, Literal[1]],
        vector_positions: NDArray[Vec, Literal[1]],
        vectors: NDArray[Vec, Literal[1]],
    ):
        for i in range(vector_positions.shape[0]):
            positions[i * 2] = vector_positions[i]
            positions[i * 2 + 1] = vector_positions[i] + vectors[i] * self.scale
