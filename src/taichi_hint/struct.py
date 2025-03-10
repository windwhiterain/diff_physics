from copy import copy
from typing import Any, override

import taichi
from taichi_hint.transform_annotations import transform_annotations
from taichi_hint.wrap import Wrap


class Struct(Wrap):
    @staticmethod
    @override
    def value(specialization: Any) -> Any:
        return taichi.dataclass(specialization)
