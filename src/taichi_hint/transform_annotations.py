from inspect import isclass
from typing import Any, TypeVar
import taichi

from taichi_hint.util import de_generic_alias, de_type_alias, get_type_alias
from taichi_hint.template import Templated, Template
from taichi_hint.wrap import Wrap


def transform_annotations(annotations: dict[str, Any], post_proc=lambda x: x):
    for k, annotation in annotations.items():
        annotation_new = None
        if isinstance(annotation, TypeVar):
            annotation_new = post_proc(annotation_new)
        else:
            generic_alias = de_type_alias(annotation)
            generic = de_generic_alias(generic_alias)
            if isclass(generic):
                if issubclass(generic, Wrap):
                    if generic.annotation_only:
                        annotation_new = generic.value(generic_alias)
                        annotation_new = post_proc(annotation_new)

        if annotation_new is not None:
            annotations[k] = annotation_new


def transform_annotations_scope(annotations: dict[str, Any]):
    def post_proc(annotation_new):
        if annotation_new is None:
            annotation_new = taichi.template()
        return annotation_new
    transform_annotations(annotations, post_proc)
    for k, annotation in annotations.items():
        annotation_new = None
        if get_type_alias(annotation) is Templated:
            annotation_new = taichi.template()
        else:
            generic_alias = de_type_alias(annotation)
            generic = de_generic_alias(generic_alias)
            if isclass(generic):
                if issubclass(generic, Template):
                    annotation_new = taichi.template()
        if annotation_new is not None:
            annotations[k] = annotation_new
