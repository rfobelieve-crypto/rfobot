#!/usr/bin/env python3
"""Pre-commit secret scanner.

Scans staged files for credential patterns and aborts the commit if any
match.  Mostly belt-and-suspenders — Railway env vars + .gitignore should
already prevent leaks, but this catches "I accidentally pasted a key into
a .py for debugging".

Usage:
  scripts/install_security_hooks.sh  # links this to .git/hooks/pre-commit
  git commit ...                     # runs automatically

Exit codes:
  0 = clean, allow commit
  1 = suspect content found, abort
"""
from __future__ import annotations

import re
import subprocess
import sys

# ── Patterns ────────────────────────────────────────────────────────────
# Each pattern is (name, regex, allowlist_regex_to_skip).
#
# Tune carefully:  false positives are annoying but tolerable, false
# negatives can leak real credentials.  Better to be slightly noisy.

PATTERNS: list[tuple[str, str, str | None]] = [
    # OKX API key format (UUID v4-ish)
    ("OKX-API-KEY-UUID",
     r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
     r"^(\s*#|\s*//|\s*\*)"),  # allow in comments (rare false-positive)

    # OKX-style header references with values
    ("OKX-HEADER-INLINE",
     r"OK-ACCESS-(?:KEY|SIGN|PASSPHRASE|TIMESTAMP)['\"\s:]+[A-Za-z0-9+/=]{16,}",
     None),

    # Generic api_key = "..." in code (excluding our env-var lookups)
    ("HARDCODED-api_key",
     r"""(?i)(api[_-]?key|api_secret|passphrase)\s*[=:]\s*['\"][A-Za-z0-9+/=_\-]{20,}['\"]""",
     None),

    # Anthropic API key
    ("ANTHROPIC-KEY",
     r"\bsk-ant-[A-Za-z0-9_\-]{40,}\b",
     None),

    # OpenAI API key (legacy and project)
    ("OPENAI-KEY",
     r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}\b",
     None),

    # AWS access key
    ("AWS-AKID",
     r"\bAKIA[0-9A-Z]{16}\b",
     None),

    # GitHub personal access token
    ("GITHUB-PAT",
     r"\bghp_[A-Za-z0-9]{36}\b",
     None),

    # Telegram bot token
    ("TELEGRAM-TOKEN",
     r"\b\d{8,11}:[A-Za-z0-9_\-]{35}\b",
     None),

    # Long base64-ish secret-looking strings as bare values
    # (heuristic: keeps false positives low by requiring var-name context)
    ("BARE-BASE64-VALUE",
     r"""(?i)(secret|token|password|passphrase|api[_-]?key)\s*[=:]\s*['\"]?[A-Za-z0-9+/]{40,}=*['\"]?""",
     r"(?i)example|placeholder|<.*>|XXXX|YYYY|ZZZZ|REDACTED|TODO|FIXME"),
]


# Files we never scan (vendored / generated / docs explicitly about format)
SKIP_PATHS = (
    ".git/",
    "node_modules/",
    "__pycache__/",
    ".venv/",
    "venv/",
    "docs/api_key_management.md",       # has format examples by design
    "scripts/pre_commit_secret_scan.py",  # this file describes patterns
)


def get_staged_files() -> list[str]:
    """Files about to be committed (Added / Modified / Renamed)."""
    out = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=AMR"],
        text=True,
    )
    files = [f for f in out.splitlines() if f]
    return [f for f in files if not any(s in f for s in SKIP_PATHS)]


def scan_file(path: str) -> list[tuple[str, int, str, str]]:
    """Returns list of (pattern_name, line_no, line_text, matched_substring)."""
    try:
        # Read staged version (not working tree, in case of partial stage)
        content = subprocess.check_output(
            ["git", "show", f":{path}"], text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    hits = []
    for pname, pregex, allow in PATTERNS:
        regex = re.compile(pregex)
        allow_re = re.compile(allow) if allow else None
        for i, line in enumerate(content.splitlines(), 1):
            if allow_re and allow_re.search(line):
                continue
            m = regex.search(line)
            if m:
                hits.append((pname, i, line.rstrip(), m.group(0)))
    return hits


def main() -> int:
    staged = get_staged_files()
    if not staged:
        return 0

    all_hits: list[tuple[str, str, int, str, str]] = []
    for f in staged:
        for hit in scan_file(f):
            all_hits.append((f, *hit))

    if not all_hits:
        return 0

    print("\n" + "=" * 70, file=sys.stderr)
    print("PRE-COMMIT BLOCKED: suspect credential pattern in staged files",
          file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    for path, pname, lineno, line, match in all_hits:
        print(f"\n  {path}:{lineno}", file=sys.stderr)
        print(f"    pattern: {pname}", file=sys.stderr)
        print(f"    matched: {match[:80]}", file=sys.stderr)
        # show line truncated
        snippet = line if len(line) <= 200 else line[:200] + "..."
        print(f"    line:    {snippet}", file=sys.stderr)

    print("\n" + "-" * 70, file=sys.stderr)
    print("If this is a FALSE POSITIVE:", file=sys.stderr)
    print("  - Reword the line (e.g. break the literal across lines)",
          file=sys.stderr)
    print("  - Or commit with --no-verify if you're 100% sure",
          file=sys.stderr)
    print("    (don't make --no-verify a habit — it defeats the point)",
          file=sys.stderr)
    print("\nIf this is REAL:", file=sys.stderr)
    print("  - Remove the value from the file", file=sys.stderr)
    print("  - Move it to an env var (Railway dashboard or local .env)",
          file=sys.stderr)
    print("  - If already pushed: REVOKE the key on the provider's site",
          file=sys.stderr)
    print("    (history scan won't help — assume it's leaked)",
          file=sys.stderr)
    print("=" * 70 + "\n", file=sys.stderr)

    return 1


if __name__ == "__main__":
    sys.exit(main())
