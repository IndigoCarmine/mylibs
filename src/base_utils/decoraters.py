"""
This module provides various decorators and utility functions for common programming patterns.
It includes decorators for applying functions to list elements, deep copying arguments,
and passing through information in a pipeline.
"""

import copy
import functools
from pydantic.dataclasses import dataclass
from functools import wraps
from typing import Callable


@dataclass
class ExList[T]:
    """
    A wrapper class for lists, used to apply functions to each element within the list.
    """

    inner: list[T]


def apply_to_list[T, R](
    func: Callable[[T], R],
) -> Callable[[T | ExList[T]], R | ExList[R]]:
    """
    A decorator that applies a given function to elements within a list or a single item.
    If the input is an ExList, the function is applied to each item in its inner list.
    Otherwise, the function is applied directly to the single input item.
    """

    @wraps(func)
    def wrapper[U](arg: T | ExList[T]) -> R | ExList[R]:
        if isinstance(arg, ExList):
            return ExList[R]([func(item) for item in arg.inner])  # type: ignore
        else:
            return func(arg)

    return wrapper


def safe_deepcopy(obj: object) -> object:
    """
    Performs a deep copy of an object. If deepcopy fails (e.g., due to unpickleable objects),
    it returns the original object instead of raising an error.
    """
    try:
        return copy.deepcopy(obj)
    except TypeError:
        return obj  # Return the original object if deepcopy fails


def deepcopy_args[R](func: Callable[..., R]) -> Callable[..., R]:
    """
    A decorator that deep copies all arguments and keyword arguments before passing them
    to the decorated function. This prevents unintended modifications to mutable inputs.
    """

    @functools.wraps(func)
    def wrapper(*args: ..., **kwargs: ...) -> R:
        copied_args = tuple(safe_deepcopy(arg) for arg in args)
        copied_kwargs = {k: safe_deepcopy(v) for k, v in kwargs.items()}
        return func(*copied_args, **copied_kwargs)

    return wrapper


def info_pass_through[T, R, U](
    func: Callable[[T], R],
) -> Callable[[tuple[U, T]], tuple[U, R]]:
    """
    A decorator that allows an additional piece of information (U) to be passed through
    a function, returning it alongside the function's result.
    Useful for maintaining context in a processing pipeline.
    """

    @wraps(func)
    def wrapper(arg: tuple[U, T]):
        u = arg[0]
        t = arg[1]
        return (u, func(t))

    return wrapper
