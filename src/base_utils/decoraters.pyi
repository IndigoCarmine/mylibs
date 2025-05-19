from typing import Callable, List, TypeVar, Union
from dataclasses import dataclass

T = TypeVar("T")  # 入力の型
R = TypeVar("R")  # 出力の型
U = TypeVar("U")

@dataclass
class ExList[T]:
    inner: list[T]

def apply_to_list(
    func: Callable[[T], R],
) -> Callable[[Union[T, ExList[T]]], Union[R, ExList[R]]]: ...
def info_pass_through(
    func: Callable[[T], R],
) -> Callable[[tuple[U, T]], tuple[U, R]]: ...
