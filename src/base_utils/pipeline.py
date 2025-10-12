"""
This module provides utility functions for building and managing data processing pipelines.
It includes functions for joining multiple functions, flattening nested lists,
and finding files based on various criteria.
"""

import os
from typing import Callable, Any


# basic pipeline functions #


def join(funcs: list[Callable[..., Any]]) -> Callable[..., Any]:
    """
    Joins multiple functions into a single callable.
    The output of each function becomes the input of the next function in the list.
    """

    def new_func(*args: Any) -> Any:
        temp_args = args
        for func in funcs:
            if isinstance(temp_args, tuple):
                temp_args = func(*temp_args)
            else:
                temp_args = func(temp_args)
            print("Result: ", temp_args)

        return temp_args

    return lambda *args: new_func(*args)


# file related functions #


def find_all_file(path: str) -> list[str]:
    """
    Recursively finds all files within a given directory and its subdirectories.
    Returns a list of absolute paths to the files.
    """
    import os

    path = os.path.abspath(path)
    all_files: list[str] = []
    for root, _, files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def find_all_file_by_suffix(path: str, suffix: str) -> list[str]:
    """
    Finds all files within a given directory and its subdirectories that end with a specified suffix.
    Returns a list of absolute paths to the matching files.
    """

    all_files = find_all_file(path)
    return [file for file in all_files if file.endswith(suffix)]


def find_all_file_by_prefix(path: str, prefix: str) -> list[str]:
    """
    Finds all files within a given directory and its subdirectories whose base name starts with a specified prefix.
    Returns a list of absolute paths to the matching files.
    """

    all_files = find_all_file(path)
    return [file for file in all_files if os.path.basename(file).startswith(prefix)]


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
