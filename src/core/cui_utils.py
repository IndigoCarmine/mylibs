from core import typecheck

@typecheck.type_check
def input_path(message: str = "Enter a file name: ") -> str:
    """
    Read a string from standard input and get file path from user.
    easy to drag and drop file path.
    """
    return input(message).replace(
        '"', '').replace("'", '').replace("& ", '')


@typecheck.type_check
def format_return_char(string: str) -> str:
    """
    Format return character.
    all return characters are replaced with \n
    """
    return string.replace("\r\n", "\n").replace("\r", "\n")


@typecheck.type_check
def add_suffix(file_path: str, suffix: str) -> str:
    """
    Add suffix to file path.
    e.g. add_suffix('test.txt', '_new') -> 'test_new.txt'
    """
    p = file_path.split('.')
    return '.'.join(p[:-1]) + suffix + '.' + p[-1]
