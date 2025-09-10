#!/usr/bin/env python3
"""
Automated Fix Script for Common Issues
Automatically fixes common code quality and formatting issues
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Run a command and return success status and output"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {description}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False, e.stderr
    except FileNotFoundError:
        print(f"⚠️  {description} skipped: tool not found")
        return False, "Tool not found"


def fix_formatting():
    """Fix code formatting issues"""
    print("🎨 Fixing code formatting...")

    # Black formatter
    success, _ = run_command(
        ["black", "src/", "tests/", "examples/", "scripts/", "launch.py"],
        "Running Black formatter",
    )

    # isort import sorting
    success, _ = run_command(
        ["isort", "src/", "tests/", "examples/", "scripts/", "launch.py"],
        "Sorting imports with isort",
    )

    return success


def fix_linting_issues():
    """Fix auto-fixable linting issues"""
    print("🔧 Fixing linting issues...")

    # Ruff auto-fix
    success, _ = run_command(
        [
            "ruff",
            "check",
            "--fix",
            "src/",
            "tests/",
            "examples/",
            "scripts/",
            "launch.py",
        ],
        "Auto-fixing with Ruff",
    )

    # autopep8 for additional PEP8 fixes
    success, _ = run_command(
        [
            "autopep8",
            "--in-place",
            "--recursive",
            "--aggressive",
            "src/",
            "tests/",
            "examples/",
            "scripts/",
            "launch.py",
        ],
        "Applying autopep8 fixes",
    )

    return success


def fix_line_endings():
    """Fix line ending issues (Windows/Unix)"""
    print("📝 Fixing line endings...")

    # Find all Python files
    python_files = []
    for pattern in [
        "src/**/*.py",
        "tests/**/*.py",
        "examples/**/*.py",
        "scripts/**/*.py",
        "*.py",
    ]:
        python_files.extend(Path(".").glob(pattern))

    fixed_count = 0
    for py_file in python_files:
        try:
            with open(py_file, "rb") as f:
                content = f.read()

            # Convert CRLF to LF
            if b"\r\n" in content:
                content = content.replace(b"\r\n", b"\n")
                with open(py_file, "wb") as f:
                    f.write(content)
                fixed_count += 1
        except Exception as e:
            print(f"⚠️  Could not fix line endings in {py_file}: {e}")

    if fixed_count > 0:
        print(f"✅ Fixed line endings in {fixed_count} files")
    else:
        print("✅ No line ending issues found")

    return True


def fix_trailing_whitespace():
    """Remove trailing whitespace"""
    print("🧹 Removing trailing whitespace...")

    python_files = []
    for pattern in [
        "src/**/*.py",
        "tests/**/*.py",
        "examples/**/*.py",
        "scripts/**/*.py",
        "*.py",
    ]:
        python_files.extend(Path(".").glob(pattern))

    fixed_count = 0
    for py_file in python_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Remove trailing whitespace
            new_lines = [line.rstrip() + "\n" for line in lines]

            # Remove trailing newlines at end of file, but ensure one newline
            while (
                len(new_lines) > 1 and new_lines[-1] == "\n" and new_lines[-2] == "\n"
            ):
                new_lines.pop()

            # Ensure file ends with exactly one newline
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"

            # Write back if changed
            if new_lines != lines:
                with open(py_file, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                fixed_count += 1

        except Exception as e:
            print(f"⚠️  Could not fix whitespace in {py_file}: {e}")

    if fixed_count > 0:
        print(f"✅ Fixed trailing whitespace in {fixed_count} files")
    else:
        print("✅ No trailing whitespace issues found")

    return True


def install_missing_tools():
    """Install missing development tools"""
    print("📦 Installing missing development tools...")

    tools = [
        "black",
        "isort",
        "flake8",
        "mypy",
        "ruff",
        "bandit",
        "autopep8",
        "pre-commit",
    ]

    for tool in tools:
        try:
            subprocess.run([tool, "--version"], capture_output=True, check=True)
            print(f"✅ {tool} is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"📦 Installing {tool}...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", tool], check=True
                )
                print(f"✅ {tool} installed successfully")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {tool}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fix common code issues automatically")
    parser.add_argument(
        "--install-tools", action="store_true", help="Install missing development tools"
    )
    parser.add_argument(
        "--format-only", action="store_true", help="Only run formatting fixes"
    )
    parser.add_argument(
        "--lint-only", action="store_true", help="Only run linting fixes"
    )

    args = parser.parse_args()

    print("🔧 Automated Fix Script for Qwen-Image Project")
    print("=" * 50)

    if args.install_tools:
        install_missing_tools()
        return

    success_count = 0
    total_count = 0

    if not args.lint_only:
        # Formatting fixes
        total_count += 1
        if fix_formatting():
            success_count += 1

        total_count += 1
        if fix_line_endings():
            success_count += 1

        total_count += 1
        if fix_trailing_whitespace():
            success_count += 1

    if not args.format_only:
        # Linting fixes
        total_count += 1
        if fix_linting_issues():
            success_count += 1

    print("\n" + "=" * 50)
    print(
        f"🏁 Fix Summary: {success_count}/{total_count} operations completed successfully"
    )

    if success_count == total_count:
        print("🎉 All fixes applied successfully!")
        print("💡 Run 'scripts/lint.sh' to verify all issues are resolved")
    else:
        print("⚠️  Some fixes may have failed. Check the output above.")

    print("=" * 50)


if __name__ == "__main__":
    main()
