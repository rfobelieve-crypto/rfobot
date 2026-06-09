"""One-shot: remap hardcoded Cyberpunk-cyan colors in dashboard_tabs/*.py to the
Nansen palette. Opacity-bucketed so accent (cyan) -> green only where it was a
real accent; mid/low opacity (labels/borders) -> neutral grey. Run once."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = sorted((ROOT / "indicator" / "dashboard_tabs").glob("*.py"))

# Exact-string replacements (order: specific first). Case-insensitive for hex.
EXACT = [
    # cyan by opacity bucket
    ("rgba(0,240,255,0.85)", "rgba(232,234,237,0.92)"),  # strong text -> near-white
    ("rgba(0,240,255,0.6)",  "rgba(154,160,166,0.85)"),  # label
    ("rgba(0,240,255,0.55)", "rgba(154,160,166,0.8)"),
    ("rgba(0,240,255,0.5)",  "rgba(154,160,166,0.8)"),
    ("rgba(0,240,255,0.4)",  "rgba(154,160,166,0.6)"),
    ("rgba(0,240,255,0.3)",  "rgba(154,160,166,0.5)"),   # dim label/border
    ("rgba(0,240,255,0.08)", "rgba(255,255,255,0.06)"),  # subtle border/bg
    ("rgba(0,240,255,0.05)", "rgba(255,255,255,0.04)"),
    ("rgba(0,204,128,0.6)",  "rgba(52,224,160,0.6)"),    # green variant
    ("rgba(255,51,102,0.7)", "rgba(255,95,109,0.7)"),    # red variant
    ("rgba(255,51,102,0.6)", "rgba(255,95,109,0.6)"),
    ("rgba(204,68,68,0.7)",  "rgba(217,96,106,0.7)"),
    ("rgba(204,68,68,0.05)", "rgba(217,96,106,0.06)"),
]
# Hex (case-insensitive)
HEX = [
    ("00f0ff", "34e0a0"),   # cyan accent -> Nansen green
    ("00cc80", "34e0a0"),   # positive green -> Nansen green
    ("ff3366", "ff5f6d"),   # red -> Nansen red
    ("cc4444", "d9606a"),   # muted red
    ("ffb400", "f5b544"),   # amber -> Nansen warn
    ("1a1a2e", "23262c"),   # old border/bg -> Nansen surface line
]

total = {}
for f in FILES:
    s = f.read_text(encoding="utf-8")
    before = s
    for a, b in EXACT:
        s = s.replace(a, b)
    for a, b in HEX:
        s = re.sub("#" + a, "#" + b, s, flags=re.IGNORECASE)
    # catch-all: any remaining cyan rgba(0,240,255,X) -> neutral grey
    s = re.sub(r"rgba\(0,\s*240,\s*255,\s*([0-9.]+)\)",
               lambda m: f"rgba(154,160,166,{m.group(1)})", s)
    if s != before:
        f.write_text(s, encoding="utf-8")
    total[f.name] = (before != s)

print("Rewrote:", [n for n, ch in total.items() if ch])
# verify no cyan left
leftover = 0
for f in FILES:
    s = f.read_text(encoding="utf-8")
    leftover += len(re.findall(r"0,\s*240,\s*255", s))
    leftover += len(re.findall(r"#00f0ff", s, flags=re.IGNORECASE))
print("Cyan leftovers (should be 0):", leftover)
