"""
This module provides various decorators and utility functions for common programming patterns.
It includes decorators for applying functions to list elements, deep copying arguments,
and passing through information in a pipeline.
"""

from pydantic.dataclasses import dataclass

from typing import Callable

@dataclass
class ExList[T]: ...

def apply_to_list[T, R](
    func: Callable[[T], R],
) -> Callable[[T | ExList[T]], R | ExList[R]]: ...
def safe_deepcopy(obj: object) -> object: ...
def deepcopy_args[R](func: Callable[..., R]) -> Callable[..., R]: ...
def info_pass_through[T, R, U](
    func: Callable[[T], R],
) -> Callable[[tuple[U, T]], tuple[U, R]]: ...
