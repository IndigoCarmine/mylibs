"""
This module provides utility functions for building and managing data processing pipelines.
It includes functions for joining multiple functions, flattening nested lists,
and finding files based on various criteria.
"""


from typing import Callable, Any

# basic pipeline functions #

def join(funcs: list[Callable[..., Any]]) -> Callable[..., Any]: ...

# file related functions #

def find_all_file(path: str) -> list[str]: ...
def find_all_file_by_suffix(path: str, suffix: str) -> list[str]: ...
def get_subfolders(path: str) -> list[tuple[str, str]]: ...
