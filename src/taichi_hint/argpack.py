from abc import abstractmethod
from copy import copy
from typing import Any, override

import taichi
from taichi_hint.transform_annotations import transform_annotations
from taichi_hint.wrap import Wrap, wrap


class ArgPack(Wrap):
    @staticmethod
    @override
    def value(specialization: Any) -> Any:
        annotations = copy(specialization.__annotations__)
        transform_annotations(annotations)
        return taichi.types.argpack(**annotations)
