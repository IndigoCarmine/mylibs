from pydantic.dataclasses import dataclass
from functools import wraps
from typing import Callable, Concatenate, List, TypeVar, Union

T = TypeVar("T")  # 入力の型
R = TypeVar("R")  # 出力の型
U = TypeVar("U")


def apply_to_list(
    func: Callable[[T], R]
) -> Callable[[Union[T, List[T]]], Union[R, List[R]]]:
    @wraps(func)
    def wrapper(arg: Union[T, List[T]]) -> Union[R, List[R]]:
        if isinstance(arg, list):
            return [func(item) for item in arg]  # リストの場合、各要素に関数を適用
        else:
            return func(arg)  # 単一の値の場合、関数をそのまま適用

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


I = TypeVar("I")
P = TypeVar("P")


def pipline(func: Callable[Concatenate[I, ...], R]) -> Callable[[I], Result[I, R]]:
    @wraps(func)
    def wrapper(self: type, args=None) -> Result[I, R]:
        if args == None:
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
