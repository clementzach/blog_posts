import re
import subprocess
import sys


def _last_defined_function(code: str) -> str | None:
    matches = re.findall(r"^def (\w+)\s*\(", code, re.MULTILINE)
    return matches[-1] if matches else None


def _normalize_tests(code: str, test_list: list[str], expected_func: str | None) -> list[str]:
    """Replace expected_func in tests with the LLM's actual last-defined function name."""
    actual = _last_defined_function(code)
    if not actual or not expected_func or actual == expected_func:
        return test_list
    return [re.sub(rf"\b{re.escape(expected_func)}\b", actual, t) for t in test_list]


def run_test(
    code: str, test_list: list[str], expected_func: str | None = None, timeout_sec: int = 60
) -> tuple[bool, str | None]:
    """Returns (passed, error_string). error_string is None on success."""
    test_list = _normalize_tests(code, test_list, expected_func)
    full_code = code + "\n" + "\n".join(test_list)
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
