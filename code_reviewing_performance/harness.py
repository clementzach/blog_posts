import subprocess
import sys


def run_test(code: str, test_list: list[str], timeout_sec: int = 5) -> bool:
    """Returns True if code passes all test assertions, False otherwise."""
    full_code = code + "\n" + "\n".join(test_list)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            timeout=timeout_sec,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False
