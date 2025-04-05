from copy import copy
from typing import Any, override

import taichi
from taichi_hint.transform_annotations import solidize_annotations
from taichi_hint.wrap import Wrap


class Struct(Wrap):
    @staticmethod
    @override
    def solidize(specialization: Any) -> Any:
        return taichi.dataclass(specialization)
