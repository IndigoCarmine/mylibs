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
    inner: list[T]


def apply_to_list(
    func: Callable[[T], R],
) -> Callable[[Union[T, ExList[T]]], Union[R, ExList[R]]]:
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
    try:
        return copy.deepcopy(obj)
    except TypeError:
        return obj  # deepcopyできない場合はそのまま使う


def deepcopy_args(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        copied_args = tuple(safe_deepcopy(arg) for arg in args)
        copied_kwargs = {k: safe_deepcopy(v) for k, v in kwargs.items()}
        return func(*copied_args, **copied_kwargs)

    return wrapper


def info_pass_through(func: Callable[[T], R]) -> Callable[[tuple[U, T]], tuple[U, R]]:
    @wraps(func)
    def wrapper(arg: tuple[U, T]):
        u = arg[0]
        t = arg[1]
        return (u, func(t))

    return wrapper


@dataclass
class Result[I, R]:
    i: I
    result: R


InputType = TypeVar("InputType")
P = TypeVar("P")


def pipline(
    func: Callable[Concatenate[InputType, ...], R],
) -> Callable[[InputType], Result[InputType, R]]:
    @wraps(func)
    def wrapper(self: type, args=None) -> Result[InputType, R]:
        if args is None:
            return Result(self, func(self))
        else:
            return Result(self, func(self, args))

    return wrapper


@dataclass
class Test:
    val: int

    @pipline
    def add(self, v: int):
        self.val = self.val + v

    @pipline
    def a(self, a):
        return 1

    @pipline
    def print(self):
        print(self.val)


if __name__ == "__main__":
    teete = Test(10)
    print(teete.add(1))
