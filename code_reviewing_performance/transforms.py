"""Deterministic variable-name obfuscation transforms.

Three transforms are defined (STUDY_DESIGN.md section 4):

* ``original``     - code unchanged
* ``hash_append``  - ``name`` -> ``name_<sha256[:8]>``
* ``hash_replace`` - ``name`` -> ``<sha256[:8]>``

The identifier *set* is derived from the AST (``ast.Name`` / ``ast.arg`` /
function names); the rename is applied by targeted replacement of NAME tokens
via ``tokenize`` so that formatting, comments, and string literals are
preserved. Because every occurrence of an identifier is renamed consistently,
both transforms preserve program semantics -- only the surface form changes.
"""

import ast
import builtins
import hashlib
import io
import keyword
import tokenize

TRANSFORMS = ("original", "hash_append", "hash_replace")

_BUILTINS = set(dir(builtins))
_KEYWORDS = set(keyword.kwlist) | set(keyword.softkwlist)


def hash8(name: str) -> str:
    """Stable 8-hex-char hash; same identifier -> same hash everywhere."""
    return hashlib.sha256(name.encode()).hexdigest()[:8]


def _imported_names(tree: ast.AST) -> set[str]:
    """Names bound by import statements (excluded from renaming)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def extract_identifiers(code: str) -> set[str]:
    """User-defined identifiers eligible for obfuscation.

    Includes variable references/bindings (``ast.Name``), parameters
    (``ast.arg``), and function names. Excludes keywords, builtins, imported
    names, and names shorter than 2 characters. Attribute names (e.g.
    ``.append``) are intentionally left untouched.
    """
    tree = ast.parse(code)
    excluded = _BUILTINS | _KEYWORDS | _imported_names(tree)

    idents: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            idents.add(node.id)
        elif isinstance(node, ast.arg):
            idents.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            idents.add(node.name)

    return {n for n in idents if len(n) >= 2 and n not in excluded}


def _rename(name: str, transform: str) -> str:
    if transform == "hash_append":
        return f"{name}_{hash8(name)}"
    if transform == "hash_replace":
        # Leading underscore guarantees a valid identifier even when the hash
        # begins with a digit (e.g. "98c1eb4e").
        return f"_{hash8(name)}"
    raise ValueError(f"Unknown transform: {transform!r}")


def apply_transform(code: str, transform: str) -> str:
    """Return ``code`` with the given transform applied.

    ``original`` returns the code unchanged. Renaming operates on NAME tokens
    only, replacing from the end of the file backwards so earlier column
    offsets stay valid.
    """
    if transform == "original":
        return code

    idents = extract_identifiers(code)
    if not idents:
        return code

    mapping = {name: _rename(name, transform) for name in idents}

    # Collect (start, end, replacement) for NAME tokens we should rename.
    edits: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    for tok in tokens:
        if tok.type == tokenize.NAME and tok.string in mapping:
            edits.append((tok.start, tok.end, mapping[tok.string]))

    lines = code.splitlines(keepends=True)
    # Apply from the end so (row, col) positions remain valid.
    for (srow, scol), (erow, ecol), new_text in sorted(edits, reverse=True):
        # NAME tokens never span multiple lines.
        line = lines[srow - 1]
        lines[srow - 1] = line[:scol] + new_text + line[ecol:]

    return "".join(lines)
