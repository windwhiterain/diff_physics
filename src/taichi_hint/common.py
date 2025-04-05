from typing import Any, override

from taichi_hint.util import is_dunder


class Object:
    __specialization__: object = None

    @override
    def __getattribute__(self, name: str) -> Any:
        if not is_dunder(name):
            if specialization := object.__getattribute__(self, "__specialization__"):
                if ret := getattr(specialization, name, None):
                    return ret
        return super().__getattribute__(name)
