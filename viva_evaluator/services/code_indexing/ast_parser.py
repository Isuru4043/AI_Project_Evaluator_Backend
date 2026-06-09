"""
AST parser — walks a code repo and extracts function/class units.

WEEK 3 SCOPE:
    - Python: native `ast` module.
    - JavaScript / TypeScript: regex-based fallback (good enough for FYP).
    - Java / others: file-level chunks (treat each file as one unit).

JS/Java AST proper coverage → Week 8 polish.

OUTPUT:
    [
        {
            'file_path':  'auth/views.py',     # relative to repo root
            'unit_type':  'function',          # or 'class' / 'file'
            'name':       'verify_token',
            'line_start': 42,
            'line_end':   78,
            'imports':    ['jwt', 'datetime', 'rest_framework'],
            'source':     '<raw source code of this unit>',
        },
        ...
    ]
"""

import ast
import logging
import re
from pathlib import Path
from typing import List, Dict, Set

logger = logging.getLogger(__name__)


# Skip auto-generated / vendored / test files in code indexing.
SKIP_FILE_PATTERNS = re.compile(
    r'(?:^|/)(?:test_|_test\.py|migrations/|node_modules/|dist/|build/|\.min\.)',
    re.IGNORECASE,
)


# =============================================================================
# Public API
# =============================================================================

def parse_repo(repo_path: str) -> List[Dict]:
    """
    Walk a cloned repo and return every function/class unit found.

    Args:
        repo_path: filesystem path to the (already extracted) source code.

    Returns:
        List of unit dicts (see module docstring). Empty list if parsing fails
        across the board.
    """
    from code_analysis.services.repo_service import iter_code_files

    units: List[Dict] = []
    repo_root = Path(repo_path)

    for file_path in iter_code_files(repo_path):
        rel_path = str(file_path.relative_to(repo_root)).replace('\\', '/')

        if SKIP_FILE_PATTERNS.search(rel_path):
            continue

        suffix = file_path.suffix.lower()
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue

        if not content.strip():
            continue

        if suffix == '.py':
            units.extend(_parse_python_file(rel_path, content))
        elif suffix in ('.js', '.jsx', '.ts', '.tsx'):
            units.extend(_parse_js_file_regex(rel_path, content))
        else:
            units.append(_file_as_unit(rel_path, content))

    logger.info('AST parser: extracted %d units from %s', len(units), repo_path)
    return units


# =============================================================================
# Python parser — full AST.
# =============================================================================

def _parse_python_file(rel_path: str, content: str) -> List[Dict]:
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        logger.debug('Python parse failed for %s: %s — falling back to file unit', rel_path, exc)
        return [_file_as_unit(rel_path, content)]

    imports = sorted(_extract_python_imports(tree))
    source_lines = content.splitlines()
    units: List[Dict] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            unit_type = 'class' if isinstance(node, ast.ClassDef) else 'function'

            line_start = node.lineno
            line_end = getattr(node, 'end_lineno', line_start) or line_start
            source = '\n'.join(source_lines[line_start - 1:line_end])

            # Skip trivial units (single-line stubs, dunders that just pass)
            if len(source.strip()) < 30:
                continue

            units.append({
                'file_path':  rel_path,
                'unit_type':  unit_type,
                'name':       node.name,
                'line_start': line_start,
                'line_end':   line_end,
                'imports':    imports,
                'source':     source,
            })

    if not units:
        # If nothing extracted (e.g., only top-level code), fall back to file unit
        units.append(_file_as_unit(rel_path, content))

    return units


def _extract_python_imports(tree: ast.AST) -> Set[str]:
    """Return the top-level module names imported in this file."""
    imports: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split('.')[0]
                imports.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split('.')[0]
                imports.add(top)
    return imports


# =============================================================================
# JS/TS regex fallback — not as accurate as a real AST, but sufficient.
# =============================================================================

_JS_FUNCTION_RE = re.compile(
    r'^[ \t]*(?:export\s+)?(?:async\s+)?'
    r'(?:function\s+(\w+)\s*\([^)]*\)|'                  # function name(...)
    r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>|'  # const name = (...) =>
    r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function)',       # const name = function
    re.MULTILINE,
)

_JS_CLASS_RE = re.compile(r'^[ \t]*(?:export\s+)?class\s+(\w+)', re.MULTILINE)

_JS_IMPORT_RE = re.compile(
    r'(?:import\s+[^;]+from\s+["\']([^"\']+)["\']|require\(["\']([^"\']+)["\']\))',
)


def _parse_js_file_regex(rel_path: str, content: str) -> List[Dict]:
    """
    Approximate JS/TS extraction. Catches function declarations, arrow
    function const declarations, and classes. Misses some edge cases —
    that's acceptable for FYP scope.
    """
    imports = _extract_js_imports(content)
    lines = content.splitlines()
    units: List[Dict] = []

    # Functions
    for match in _JS_FUNCTION_RE.finditer(content):
        name = match.group(1) or match.group(2) or match.group(3) or 'anonymous'
        line_start = content.count('\n', 0, match.start()) + 1
        line_end = _find_js_block_end(lines, line_start - 1)
        source = '\n'.join(lines[line_start - 1:line_end])

        if len(source.strip()) < 30:
            continue

        units.append({
            'file_path':  rel_path,
            'unit_type':  'function',
            'name':       name,
            'line_start': line_start,
            'line_end':   line_end,
            'imports':    imports,
            'source':     source,
        })

    # Classes
    for match in _JS_CLASS_RE.finditer(content):
        name = match.group(1)
        line_start = content.count('\n', 0, match.start()) + 1
        line_end = _find_js_block_end(lines, line_start - 1)
        source = '\n'.join(lines[line_start - 1:line_end])

        if len(source.strip()) < 30:
            continue

        units.append({
            'file_path':  rel_path,
            'unit_type':  'class',
            'name':       name,
            'line_start': line_start,
            'line_end':   line_end,
            'imports':    imports,
            'source':     source,
        })

    if not units:
        units.append(_file_as_unit(rel_path, content, imports=imports))

    return units


def _extract_js_imports(content: str) -> List[str]:
    """Pull module names from import / require statements."""
    seen: Set[str] = set()
    for match in _JS_IMPORT_RE.finditer(content):
        module = match.group(1) or match.group(2) or ''
        if module:
            # Normalize: keep top-level package name, drop relative paths
            if module.startswith('.') or module.startswith('/'):
                continue
            top = module.split('/')[0]
            if top.startswith('@') and '/' in module:
                # scoped: @scope/pkg → @scope/pkg
                top = '/'.join(module.split('/')[:2])
            seen.add(top)
    return sorted(seen)


def _find_js_block_end(lines: List[str], start_idx: int) -> int:
    """Find the matching closing brace by tracking depth from start_idx."""
    depth = 0
    started = False
    for i in range(start_idx, len(lines)):
        for ch in lines[i]:
            if ch == '{':
                depth += 1
                started = True
            elif ch == '}':
                depth -= 1
                if started and depth <= 0:
                    return i + 1
    return min(start_idx + 50, len(lines))  # cap fallback


# =============================================================================
# File-as-unit fallback
# =============================================================================

def _file_as_unit(rel_path: str, content: str, imports: List[str] = None) -> Dict:
    """When AST parsing fails or for unsupported languages."""
    return {
        'file_path':  rel_path,
        'unit_type':  'file',
        'name':       Path(rel_path).stem,
        'line_start': 1,
        'line_end':   content.count('\n') + 1,
        'imports':    imports or [],
        'source':     content[:5000],   # cap so prompts stay bounded
    }
