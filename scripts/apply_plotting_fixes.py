#!/usr/bin/env python3
# coding: utf-8
"""
apply_plotting_fixes.py

Description (中文):
脚本用于在本仓库的 examples/ 和 BayesianTripleCollocation-master 等示例脚本中，自动替换分散的 Matplotlib 中文字体设置（如 plt.rcParams['font.sans-serif'] = [...] 和 plt.rcParams['axes.unicode_minus'] = False），改为统一导入并调用 src.utils.plotting.ensure_chinese_font()，并引入 save_fig 以便后续替换保存行为。

Usage:
    python scripts/apply_plotting_fixes.py --dry-run
    python scripts/apply_plotting_fixes.py --apply

注意：脚本会备份被修改的文件（.bak），建议先用 --dry-run 查看将要修改的文件列表和变更预览，然后再使用 --apply 执行变更。

This file will be committed to branch refactor/docs-examples-fonts.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

# 列出要扫描并替换的相对路径（示例列表，可根据需要扩展）
TARGET_FILES = [
    "examples/example_fusion.py",
    "examples/etcc_example.py",
    "examples/comprehensive_comparison.py",
    "examples/example_all_methods.py",
    "examples/simple_average_example.py",
    "examples/covariance_method_comparison.py",
    "examples/bayesian_tch_example.py",
    "examples/robust_collocation_scenarios.py",
    "examples/vogel_menard_generalized_example.py",
    # 包含第三方子目录的示例脚本
    "BayesianTripleCollocation-master/simulation_scripts/plotting.py",
    "BayesianTripleCollocation-master/simulation_scripts/plot_Q2_params.py",
    "collocation/bayesian_tch.py",
]

# 正则：匹配 plt.rcParams['font.sans-serif'] = ... 和 plt.rcParams["axes.unicode_minus"] = ...
RE_FONT_SANS = re.compile(r"^\s*plt\.rcParams\[['\"]font\.sans-serif['\"]\]\s*=.*$", flags=re.M)
RE_UNICODE_MINUS = re.compile(r"^\s*plt\.rcParams\[['\"]axes\.unicode_minus['\"]\]\s*=.*$", flags=re.M)
RE_PLT_RC_FONT = re.compile(r"^\s*plt\.rc\(['\"]font['\"],.*\)$", flags=re.M)

# 插入代码块：导入并调用 ensure_chinese_font
INSERT_BLOCK = (
    "from src.utils.plotting import ensure_chinese_font, save_fig\n"
    "ensure_chinese_font()\n"
)

def process_file(path: Path) -> Tuple[bool, str]:
    """Process a single file. Returns (changed, diff_preview).

    - If file contains the target rcParams lines, they will be removed and the INSERT_BLOCK
      will be inserted after the top module docstring or after the last import block.
    - A backup file with suffix .bak will be written when applying changes.
    """
    text = path.read_text(encoding="utf-8")
    original = text

    # Detect occurrences
    found_font = bool(RE_FONT_SANS.search(text) or RE_PLT_RC_FONT.search(text))
    found_minus = bool(RE_UNICODE_MINUS.search(text))
    if not (found_font or found_minus):
        return False, "No matches"

    # Remove matched lines
    text = RE_FONT_SANS.sub("", text)
    text = RE_UNICODE_MINUS.sub("", text)
    # Also remove plt.rc('font', ...) patterns (keep other rc settings)
    text = RE_PLT_RC_FONT.sub(lambda m: _remove_font_family_from_rc(m.group(0)), text)

    # Insert INSERT_BLOCK after module docstring if present, else after imports
    new_text = _insert_after_imports_or_docstring(text, INSERT_BLOCK)

    # Generate a short diff-like preview
    preview = _diff_preview(original, new_text)

    return True, preview

def _remove_font_family_from_rc(line: str) -> str:
    """Attempt to remove only family=... from a plt.rc(...) call. If uncertain, return original line unchanged.
    This is conservative: if family present we remove the family arg, else keep line.
    """
    # Very simple heuristic: remove family=...'...' or family=[...]
    new = re.sub(r"family\s*=\s*\[?[^"]*\]?\s*,?", "", line)
    return new

def _insert_after_imports_or_docstring(text: str, insert_block: str) -> str:
    # If there is a module-level docstring, insert after it
    m = re.match(r"^(\s*('''|\"\"\").*?\2\s*)", text, flags=re.S)
    insert_pos = 0
    if m:
        insert_pos = m.end()
    else:
        # Find last import occurrence to append after imports
        imports = list(re.finditer(r"^\s*(import\s+\w+|from\s+\w+\s+import\s+.+)$", text, flags=re.M))
        if imports:
            insert_pos = imports[-1].end()
    # Avoid duplicate insertion: if insert_block already present, skip
    if insert_block.strip() in text:
        return text
    return text[:insert_pos] + "\n" + insert_block + text[insert_pos:]

def _diff_preview(a: str, b: str, nlines: int = 20) -> str:
    """Return a small unified diff-style preview (not full diff) for quick inspection."""
    import difflib

    a_lines = a.splitlines()
    b_lines = b.splitlines()
    diff = difflib.unified_diff(a_lines, b_lines, lineterm="", n=3)
    out = "\n".join(list(diff)[: nlines])
    return out or "(diff too large or no changes)"

def apply_changes(paths: List[Path], dry_run: bool = True) -> None:
    changed_files = []
    for p in paths:
        if not p.exists():
            print(f"[SKIP] Not found: {p}")
            continue
        changed, preview = process_file(p)
        if not changed:
            print(f"[OK] No inline font rcParams in: {p}")
            continue
        print(f"[MOD] {p}\n")
        print(preview)
        changed_files.append(p)
        if not dry_run:
            # backup
            backup = p.with_suffix(p.suffix + ".bak")
            backup.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
            # write new content
            new_text = _insert_after_imports_or_docstring(RE_FONT_SANS.sub("", RE_UNICODE_MINUS.sub("", p.read_text(encoding="utf-8"))), INSERT_BLOCK)
            p.write_text(new_text, encoding="utf-8")
            print(f"  -> Applied (backup: {backup})")
    if not changed_files:
        print("No files changed.")
    else:
        print(f"Processed {len(changed_files)} file(s).")

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply plotting font unification across repository examples")
    parser.add_argument("--apply", action="store_true", help="Actually write changes. If omitted, runs in dry-run mode and prints previews.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    paths = [repo_root.joinpath(p) for p in TARGET_FILES]
    apply_changes(paths, dry_run=not args.apply)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())