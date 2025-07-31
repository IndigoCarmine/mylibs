"""
This module provides various decorators and utility functions for common programming patterns.
It includes decorators for applying functions to list elements, deep copying arguments,
and passing through information in a pipeline.
"""
import copy
import functools
from pydantic.dataclasses import dataclass
from functools import wraps
from typing import Callable, Concatenate, List, TypeVar, Union

T = TypeVar("T")  # 入力の型
R = TypeVar("R")  # 出力の型
U = TypeVar("U")


@dataclass
class ExList[T]:
    """
    A wrapper class for lists, used to apply functions to each element within the list.
    """
    inner: list[T]


def apply_to_list(
    func: Callable[[T], R],
) -> Callable[[Union[T, ExList[T]]], Union[R, ExList[R]]]:
    """
    A decorator that applies a given function to elements within a list or a single item.
    If the input is an ExList, the function is applied to each item in its inner list.
    Otherwise, the function is applied directly to the single input item.
    """
    @wraps(func)
    def wrapper(arg: Union[T, List[T]]) -> Union[R, List[R]]:
        if isinstance(arg, ExList):
            return ExList(
                [func(item) for item in arg.inner]
            )  # リストの場合、各要素に関数を適用
        else:
            return func(arg)  # 単一の値の場合、関数をそのまま適用

    return wrapper


def safe_deepcopy(obj):
    """
    Performs a deep copy of an object. If deepcopy fails (e.g., due to unpickleable objects),
    it returns the original object instead of raising an error.
    """
    try:
        return copy.deepcopy(obj)
    except TypeError:
        return obj  # deepcopyできない場合はそのまま使う


def deepcopy_args(func):
    """
    A decorator that deep copies all arguments and keyword arguments before passing them
    to the decorated function. This prevents unintended modifications to mutable inputs.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        copied_args = tuple(safe_deepcopy(arg) for arg in args)
        copied_kwargs = {k: safe_deepcopy(v) for k, v in kwargs.items()}
        return func(*copied_args, **copied_kwargs)

    return wrapper


def info_pass_through(func: Callable[[T], R]) -> Callable[[tuple[U, T]], tuple[U, R]]:
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


@dataclass
class Result[I, R]:
    """
    A dataclass to encapsulate the input (i) and result (result) of a pipeline step.
    """
    i: I
    result: R


InputType = TypeVar("InputType")
P = TypeVar("P")


def pipline(
    func: Callable[Concatenate[InputType, ...], R],
) -> Callable[[InputType], Result[InputType, R]]:
    """
    A decorator that wraps a method to return its input and result as a Result object.
    Designed for use in a pipeline where the original input needs to be preserved.
    """
    @wraps(func)
    def wrapper(self: type, args=None) -> Result[InputType, R]:
        if args is None:
            return Result(self, func(self))
        else:
            return Result(self, func(self, args))

    return wrapper


@dataclass
class Test:
    """
    A simple test class demonstrating the usage of the pipline decorator.
    """
    val: int

    @pipline
    def add(self, v: int):
        """
        Adds a value to the 'val' attribute.
        """
        self.val = self.val + v

    @pipline
    def a(self, a):
        """
        A placeholder method returning 1.
        """
        return 1

    @pipline
    def print(self):
        """
        Prints the current value of the 'val' attribute.
        """
        print(self.val)


if __name__ == "__main__":
    teete = Test(10)
    print(teete.add(1))
