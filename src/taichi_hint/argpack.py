from abc import abstractmethod
from copy import copy
from typing import Any, override

import taichi
from taichi_hint.transform_annotations import solidize_annotations
from taichi_hint.wrap import Wrap, wrap


class ArgPack(Wrap):
    @staticmethod
    @override
    def solidize(specialization: Any) -> Any:
        annotations = copy(specialization.__annotations__)
        solidize_annotations(annotations)
        return taichi.types.argpack(**annotations)
