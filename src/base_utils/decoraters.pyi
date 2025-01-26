from typing import Callable, List, TypeVar, Union

T = TypeVar("T")  # 入力の型
R = TypeVar("R")  # 出力の型
U = TypeVar("U")


def apply_to_list(
    func: Callable[[T], R]
) -> Callable[[Union[T, List[T]]], Union[R, List[R]]]: ...


def info_pass_through(
    func: Callable[[T], R]
) -> Callable[[tuple[U, T]], tuple[U, R]]: ...
