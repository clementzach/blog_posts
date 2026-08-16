import re
import subprocess
import sys


def _last_defined_function(code: str) -> str | None:
    matches = re.findall(r"^def (\w+)\s*\(", code, re.MULTILINE)
    return matches[-1] if matches else None


def tested_function(code: str, test_list: list[str]) -> str | None:
    """The function from `code` that `test_list` actually calls.

    `_last_defined_function` picks the last ``def`` in the file, which breaks
    when the canonical solution defines a helper after the function under
    test (e.g. MBPP task 223 defines `is_majority` then a `binary_search`
    helper, but the tests call `is_majority`). Falls back to the last
    defined function if none of the defined names appear in the tests.
    """
    defined = re.findall(r"^def (\w+)\s*\(", code, re.MULTILINE)
    tests_blob = "\n".join(test_list)
    called = [name for name in defined if re.search(rf"\b{re.escape(name)}\s*\(", tests_blob)]
    if called:
        return called[0]
    return defined[-1] if defined else None


def _normalize_tests(code: str, test_list: list[str], expected_func: str | None) -> list[str]:
    """Replace expected_func in tests with the LLM's actual last-defined function name."""
    actual = _last_defined_function(code)
    if not actual or not expected_func or actual == expected_func:
        return test_list
    return [re.sub(rf"\b{re.escape(expected_func)}\b", actual, t) for t in test_list]


def _execute(full_code: str, timeout_sec: int) -> tuple[bool, str | None]:
    """Returns (passed, error_string). error_string is None on success."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            timeout=timeout_sec,
        )
        if proc.returncode == 0:
            return True, None
        return False, proc.stderr.decode(errors="replace").strip()
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired"


def run_test(
    code: str,
    test_list: list[str],
    expected_func: str | None = None,
    test_imports: list[str] | None = None,
    timeout_sec: int = 60,
) -> tuple[bool, str | None]:
    """Run the original (3-7 assertion) ``test_list``. Superseded by
    ``run_test_plus`` for problems that have MBPP+'s augmented ``test``
    field, but kept for ad hoc/quick checks."""
    test_list = _normalize_tests(code, test_list, expected_func)
    imports = "\n".join(test_imports) + "\n" if test_imports else ""
    full_code = imports + code + "\n" + "\n".join(test_list)
    return _execute(full_code, timeout_sec)


def entry_point_from_test(test: str) -> str | None:
    """The candidate function name MBPP+'s compiled ``test`` field calls.

    The script always invokes the candidate as ``assertion(<entry_point>(*inp),
    ...)``; for problems with nondeterministic output (e.g. dict/set/Counter
    ordering) the reference solution is embedded and aliased to `ref_func`,
    which appears as the *second* argument to `assertion(`, never immediately
    after it -- so anchoring on `assertion(` avoids picking up `ref_func`.
    """
    matches = re.findall(r"assertion\((\w+)\(\*inp\)", test)
    return matches[0] if matches else None


def run_test_plus(
    code: str,
    test: str,
    test_imports: list[str] | None = None,
    timeout_sec: int = 60,
) -> tuple[bool, str | None]:
    """Run MBPP+'s augmented ``test`` field against `code`.

    Dozens to hundreds of test cases per problem (vs. 3-7 in ``test_list``),
    generated and cross-validated by the evalplus authors against multiple
    independent reference solutions. ``test`` already declares its own
    imports (numpy, math) and defines its own ``assertion`` helper; it calls
    the candidate under the canonical solution's function name, so if the
    model defined its function under a different name we alias it in rather
    than rewriting the (sometimes very large) test script.
    """
    entry_point = entry_point_from_test(test)
    actual = _last_defined_function(code)
    imports = "\n".join(test_imports) + "\n" if test_imports else ""
    alias = (
        f"\n{entry_point} = {actual}\n"
        if entry_point and actual and entry_point != actual
        else "\n"
    )
    full_code = imports + code + alias + test
    return _execute(full_code, timeout_sec)
