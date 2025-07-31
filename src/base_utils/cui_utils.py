"""
This module provides utility functions for command-line interface (CLI) interactions.
It includes functions for inputting file paths, formatting strings, and displaying colored messages.
"""
from base_utils import typecheck


@typecheck.type_check
def input_path(message: str = "Enter a file name: ") -> str:
    """
    Read a string from standard input and get file path from user.
    easy to drag and drop file path.
    """
    return input(message).replace('"', "").replace("'", "").replace("& ", "")


@typecheck.type_check
def format_return_char(string: str) -> str:
    """
    Format return character.
    all return characters are replaced with \\n
    """
    return string.replace("\r\n", "\n").replace("\r", "\n")


@typecheck.type_check
def add_suffix(file_path: str, suffix: str) -> str:
    """
    Add suffix to file path.
    e.g. add_suffix('test.txt', '_new') -> 'test_new.txt'
    """
    p = file_path.split(".")
    return ".".join(p[:-1]) + suffix + "." + p[-1]


def warning(*values: object):
    """
    Print warning message. It is just a print function with red color.
    """
    print("\033[31m", end="")
    print("".join([str(val) for val in values]), end="")
    print("\033[0m")


def notice(*values: object):
    """
    Print notice message. It is just a print function with yellow color.
    """
    print("\033[33m", end="")
    print("".join([str(val) for val in values]), end="")
    print("\033[0m")
