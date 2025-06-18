import os
from typing import Iterable, Callable
from warnings import warn
from base_utils.typecheck import type_check


# basic pipeline functions #


def deprecated(message: str):
    """
    Decorator to mark a function as deprecated.
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
    Join multiple functions
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
    Flatten nested list
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
    Find all files in the path
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
    Find all files in the path by suffix
    """

    all_files = find_all_file(path)
    return [file for file in all_files if file.endswith(suffix)]


@type_check
def find_all_file_by_prefix(path: str, prefix: str) -> list[str]:
    """
    Find all files in the path by prefix
    """

    all_files = find_all_file(path)
    return [file for file in all_files if os.path.basename(file).startswith(prefix)]


@type_check
def get_subfolders(path: str) -> list[tuple[str, str]]:
    """
    Get all subfolders in the path
    """
    import os

    return [
        (subfolder, os.path.join(path, subfolder))
        for subfolder in os.listdir(path)
        if os.path.isdir(os.path.join(path, subfolder))
    ]
