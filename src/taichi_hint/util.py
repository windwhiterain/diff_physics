from types import GenericAlias, UnionType
from typing import Any, Iterable, TypeAliasType, TypeVar, get_origin


def get_type_alias(annotation: Any) -> Any:
    if isinstance(annotation, GenericAlias):
        return annotation.__reduce__()[1][0]
    return None


def de_type_alias(annotation: Any) -> Any:
    while isinstance(annotation, TypeAliasType):
        annotation = annotation.__value__
    return annotation


def de_generic_alias(annotation: Any) -> Any:
    while origin := get_origin(annotation):
        annotation = origin
    return annotation


def de_alias(annotation: Any) -> Any:
    return de_generic_alias(de_type_alias(annotation))


def is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__") and len(name) > 4


def is_solid_type(type) -> bool:
    if isinstance(type, (TypeVar, UnionType)):
        return False
    if type_args := getattr(type, "__args__", None):
        for type_arg in type_args:
            if not is_solid_type(type_arg):
                return False
    return True


def is_solid_types(*types) -> bool:
    for type in types:
        if not is_solid_type(type):
            return False
    return True


def repr_iterable(iterable: Iterable) -> str:
    return f"[{",".join(iterable)}]"
