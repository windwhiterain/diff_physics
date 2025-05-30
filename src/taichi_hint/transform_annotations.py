from inspect import isclass
from typing import Any, TypeVar
import taichi

from taichi_hint.util import (
    de_generic_alias,
    de_type_alias,
    get_type_alias,
    is_solid_type,
    is_solid_types,
)
from taichi_hint.template import Templated, Template
from taichi_hint.wrap import Wrap


def solidize_annotations(annotations: dict[str, Any], post_proc=lambda x: x):
    for k, annotation in annotations.items():
        annotation_new = None
        generic_alias = de_type_alias(annotation)
        generic = de_generic_alias(generic_alias)
        if isclass(generic):
            if issubclass(generic, Wrap):
                annotation_new = generic.solidize(generic_alias)
                annotation_new = post_proc(annotation_new)
        if annotation_new is not None:
            annotations[k] = annotation_new


def solidize_annotations_scope(annotations: dict[str, Any]):
    def post_proc(annotation_new):
        if annotation_new is None:
            annotation_new = taichi.template()
        return annotation_new

    solidize_annotations(annotations, post_proc)
    for k, annotation in annotations.items():
        annotation_new = None
        if not is_solid_type(annotation):
            annotation_new = taichi.template()
        elif get_type_alias(annotation) is Templated:
            annotation_new = taichi.template()
        else:
            generic_alias = de_type_alias(annotation)
            generic = de_generic_alias(generic_alias)
            if isclass(generic):
                if issubclass(generic, Template):
                    annotation_new = taichi.template()
        if annotation_new is not None:
            annotations[k] = annotation_new
