from core import typecheck

@typecheck.type_check
def input_path(message: str = "Enter a file name: ") -> str:
    """
    Get file path from user.
    easy to drag and drop file path.
    """
    return input(message).replace(
        '"', '').replace("'", '').replace("& ", '')

