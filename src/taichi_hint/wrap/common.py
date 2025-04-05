from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import FunctionType
from typing import Any

from taichi_hint.specialize import specializable


class Wrap(ABC):
    annotation_only = False

    @staticmethod
    @abstractmethod
    def solidize(specialization: Any) -> Any: ...


wrap_data_name = "__wrap_data__"


@dataclass
class WrapData:
    value: Any


def wrap[T:type[Wrap]](cls: T) -> T:

    def post_proc(specialization: Any) -> Any:
        if cls.annotation_only:
            return specialization
        if wrap_data := getattr(specialization, wrap_data_name, None):
            return wrap_data.value
        else:
            value = cls.solidize(specialization)
            if value is None:
                return specialization
            wrap_data = WrapData(value)
            setattr(specialization, wrap_data_name, wrap_data)
            return value

    return specializable(cls, post_proc)

@dataclass
class signature[T:type[FunctionType]]:
    value: FunctionType

    def __call__(self, fn: T) -> T:
        return self.value  # type: ignore
