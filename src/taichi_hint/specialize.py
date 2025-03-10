import inspect
from types import FunctionType, MethodType
from typing import Any, Callable, TypeVar, get_args

from taichi_hint.util import de_generic_alias, is_dunder


def specialize(generic_alias: Any, type_map: dict[TypeVar, Any]) -> Any:
    if generic_alias in type_map:
        return type_map[generic_alias]
    generic = de_generic_alias(generic_alias)
    if class_getitem := getattr(generic, "__class_getitem__", None):
        type_args_new = tuple(specialize(i, type_map)
                              for i in generic_alias.__args__)
        return class_getitem(type_args_new)
    else:
        return generic_alias


specialize_data_name = "__specialize_data__"


class SpecializeData:
    pass


def specializable(cls: type, post_proc: Callable[[Any,], Any] = lambda x: x) -> Any:
    type_params = cls.__type_params__
    if len(type_params) > 0:
        class_getitem = cls.__class_getitem__  # type: ignore

        def class_getitem_new(*args, **kargs) -> Any:
            generic_alias = class_getitem(*args, **kargs)
            if not (wrap_data := getattr(generic_alias, specialize_data_name, None)):
                wrap_data = SpecializeData()
                setattr(generic_alias, specialize_data_name, wrap_data)
                # type_param
                type_params = cls.__type_params__
                type_args = get_args(generic_alias)
                type_map = dict[TypeVar, Any]()
                type_arg_num = len(type_params)
                if type_arg_num == len(type_args):
                    for i in range(type_arg_num):
                        type_var = type_params[i]
                        if isinstance(type_var, TypeVar):
                            type_arg = type_args[i]
                            type_map[type_var] = type_arg
                            object.__setattr__(
                                generic_alias, type_var.__name__, type_arg)
                else:
                    raise Exception()
                # annotation
                generic_alias.__annotations__ = {k: specialize(
                    v, type_map) for k, v in cls.__annotations__.items()}

                # function
                for k, v in inspect.getmembers(cls, lambda x: inspect.isfunction(x) or inspect.ismethod(x)):
                    if is_dunder(k):
                        continue
                    is_method = inspect.ismethod(v)
                    if is_method:
                        func = v.__func__
                    else:
                        func = v
                    func.__annotations__ = {k: specialize(
                        v, type_map) for k, v in func.__annotations__.items()}
                    # classmethod
                    if is_method:
                        def bounded(*args, **kargs):
                            return func(generic_alias, *args, *kargs)
                        object.__setattr__(generic_alias, k, bounded)
            return post_proc(generic_alias)

        cls.__class_getitem__ = class_getitem_new  # type: ignore
        return cls
    else:
        return post_proc(cls)
