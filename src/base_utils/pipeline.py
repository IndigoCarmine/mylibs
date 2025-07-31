"""
This module provides utility functions for building and managing data processing pipelines.
It includes functions for joining multiple functions, flattening nested lists,
and finding files based on various criteria.
"""
import os
from typing import Iterable, Callable
from warnings import warn
from base_utils.typecheck import type_check


# basic pipeline functions #


def deprecated(message: str):
    """
    Decorator to mark a function as deprecated.
    When the decorated function is called, a DeprecationWarning is issued with the provided message.
    """

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            warn(f"{func.__name__} is deprecated: {message}", DeprecationWarning)
            return func(*args, **kwargs)

        return wrapper

    return decorator


@type_check
def join(funcs: list[Callable]) -> Callable:
    """
    Joins multiple functions into a single callable.
    The output of each function becomes the input of the next function in the list.
    """

    def new_func(*args):
        temp_args = args
        for func in funcs:
            if isinstance(temp_args, tuple):
                temp_args = func(*temp_args)
            else:
                temp_args = func(temp_args)
            print("Result: ", temp_args)

        return temp_args

    return lambda *args: new_func(*args)


@type_check
@deprecated("Use numpy flatten instead")
def flatten(data: Iterable[Iterable | object]) -> list:
    """
    Recursively flattens a nested iterable (list, tuple, etc.) into a single-level list.
    Note: This function is deprecated; consider using NumPy's flatten for better performance.
    """
    result = []
    for item in data:
        if isinstance(item, Iterable):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


# file related functions #


@type_check
def find_all_file(path: str) -> list[str]:
    """
    Recursively finds all files within a given directory and its subdirectories.
    Returns a list of absolute paths to the files.
    """
    import os

    path = os.path.abspath(path)
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


@type_check
def find_all_file_by_suffix(path: str, suffix: str) -> list[str]:
    """
    Finds all files within a given directory and its subdirectories that end with a specified suffix.
    Returns a list of absolute paths to the matching files.
    """

    all_files = find_all_file(path)
    return [file for file in all_files if file.endswith(suffix)]


@type_check
def find_all_file_by_prefix(path: str, prefix: str) -> list[str]:
    """
    Finds all files within a given directory and its subdirectories whose base name starts with a specified prefix.
    Returns a list of absolute paths to the matching files.
    """

    all_files = find_all_file(path)
    return [file for file in all_files if os.path.basename(file).startswith(prefix)]


@type_check
def get_subfolders(path: str) -> list[tuple[str, str]]:
    """
    Retrieves a list of all immediate subfolders within a given path.
    Returns a list of tuples, where each tuple contains the subfolder name and its absolute path.
    """
    import os

    return [
        (subfolder, os.path.join(path, subfolder))
        for subfolder in os.listdir(path)
        if os.path.isdir(os.path.join(path, subfolder))
    ]
