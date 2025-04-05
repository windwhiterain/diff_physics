import taichi
import taichi._kernels
import taichi.lang
import taichi.lang.kernel_impl

from taichi_hint.transform_annotations import solidize_annotations_scope


def kernel[T:object](fn: T) -> T:
    solidize_annotations_scope(fn.__annotations__)
    return taichi.lang.kernel_impl._kernel_impl(fn, 3)  # type: ignore


def func[T:object](fn: T) -> T:
    solidize_annotations_scope(fn.__annotations__)
    return taichi.func(fn)  # type: ignore


def pyfunc[T:object](fn: T) -> T:
    solidize_annotations_scope(fn.__annotations__)
    return taichi.pyfunc(fn)  # type: ignore
