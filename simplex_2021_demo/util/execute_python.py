import runpy
from inspect import ismodule
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict


def execute_code(
    code: str,
    init_globals=None,
):
    """
    Runs Python code from a string as if it was an imported module (.py file).

    I wanted the imports to behave as in modules, so I didn't use `eval()` or `exec()`.

    Args:
        code:
            Python source code.
        init_globals:
            Global variables.

    Returns:
        The module.
    """
    with TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "code.py"
        path.write_text(code)
        return runpy.run_path(str(path), init_globals=init_globals)


def run_code_and_get_variables(
    code: str,
    init_globals: Dict = None,
) -> Dict:
    """Runs code and returns a dict with meaningful variables, found by some heuristics."""
    result = execute_code(
        code=code,
        init_globals=init_globals,
    )
    result = {
        name: value
        for name, value in result.items()
        if not (name.startswith("__") or ismodule(value))
    }
    return result
