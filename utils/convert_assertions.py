"""
Convert unittest assertions to modern assert statements in test files.

Written by Claude.
"""

import argparse
import re
from pathlib import Path
from typing import Tuple


class AssertionConverter:
    """Converts unittest assertions to modern assert statements."""

    def __init__(self):
        # Define conversion patterns: (regex_pattern, replacement_function)
        self.conversions = [
            # self.assertTrue(x) -> assert x
            (r"self\.assertTrue\(([^)]+)\)", lambda m: f"assert {m.group(1)}"),
            # self.assertFalse(x) -> assert not x
            (r"self\.assertFalse\(([^)]+)\)", lambda m: f"assert not ({m.group(1)})"),
            # self.assertIsNone(x) -> assert x is None
            (r"self\.assertIsNone\(([^)]+)\)", lambda m: f"assert {m.group(1)} is None"),
            # self.assertIsNotNone(x) -> assert x is not None
            (r"self\.assertIsNotNone\(([^)]+)\)", lambda m: f"assert {m.group(1)} is not None"),
            # self.assertIs(a, b) -> assert a is b
            (r"self\.assertIs\(([^,]+),\s*([^)]+)\)", lambda m: f"assert {m.group(1)} is {m.group(2)}"),
            # self.assertIsNot(a, b) -> assert a is not b
            (r"self\.assertIsNot\(([^,]+),\s*([^)]+)\)", lambda m: f"assert {m.group(1)} is not {m.group(2)}"),
            # self.assertIn(a, b) -> assert a in b
            (r"self\.assertIn\(([^,]+),\s*([^)]+)\)", lambda m: f"assert {m.group(1)} in {m.group(2)}"),
            # self.assertNotIn(a, b) -> assert a not in b
            (r"self\.assertNotIn\(([^,]+),\s*([^)]+)\)", lambda m: f"assert {m.group(1)} not in {m.group(2)}"),
            # self.assertGreater(a, b) -> assert a > b
            (r"self\.assertGreater\(([^,]+),\s*([^)]+)\)", lambda m: f"assert {m.group(1)} > {m.group(2)}"),
            # self.assertGreaterEqual(a, b) -> assert a >= b
            (r"self\.assertGreaterEqual\(([^,]+),\s*([^)]+)\)", lambda m: f"assert {m.group(1)} >= {m.group(2)}"),
            # self.assertLess(a, b) -> assert a < b
            (r"self\.assertLess\(([^,]+),\s*([^)]+)\)", lambda m: f"assert {m.group(1)} < {m.group(2)}"),
            # self.assertLessEqual(a, b) -> assert a <= b
            (r"self\.assertLessEqual\(([^,]+),\s*([^)]+)\)", lambda m: f"assert {m.group(1)} <= {m.group(2)}"),
            # self.assertIsInstance(a, b) -> assert isinstance(a, b)
            (
                r"self\.assertIsInstance\(([^,]+),\s*([^)]+)\)",
                lambda m: f"assert isinstance({m.group(1)}, {m.group(2)})",
            ),
            # self.assertNotIsInstance(a, b) -> assert not isinstance(a, b)
            (
                r"self\.assertNotIsInstance\(([^,]+),\s*([^)]+)\)",
                lambda m: f"assert not isinstance({m.group(1)}, {m.group(2)})",
            ),
            # self.assertAlmostEqual(a, b) -> assert abs(a - b) < 1e-7
            (
                r"self\.assertAlmostEqual\(([^,]+),\s*([^)]+)\)",
                lambda m: f"assert abs({m.group(1)} - {m.group(2)}) < 1e-7",
            ),
            # self.assertNotAlmostEqual(a, b) -> assert abs(a - b) >= 1e-7
            (
                r"self\.assertNotAlmostEqual\(([^,]+),\s*([^)]+)\)",
                lambda m: f"assert abs({m.group(1)} - {m.group(2)}) >= 1e-7",
            ),
        ]

    def _split_args(self, args_str: str) -> list:
        """Split function arguments, handling nested parentheses."""
        args = []
        current_arg = ""
        paren_count = 0

        for char in args_str:
            if char == "(":
                paren_count += 1
                current_arg += char
            elif char == ")":
                paren_count -= 1
                current_arg += char
            elif char == "," and paren_count == 0:
                args.append(current_arg.strip())
                current_arg = ""
            else:
                current_arg += char

        if current_arg.strip():
            args.append(current_arg.strip())

        return args

    def convert_line(self, line: str) -> str:
        """Convert a single line from unittest assertions to assert statements."""
        # Handle assertEqual and assertNotEqual with proper argument parsing
        # Use a more sophisticated approach to find the matching closing parenthesis
        equal_pattern = r"self\.assert(Not)?Equal\("
        match = re.search(equal_pattern, line)
        if match:
            is_not = match.group(1) is not None
            start_pos = match.end()

            # Find the matching closing parenthesis
            paren_count = 0
            end_pos = start_pos
            for i, char in enumerate(line[start_pos:], start_pos):
                if char == "(":
                    paren_count += 1
                elif char == ")":
                    if paren_count == 0:
                        end_pos = i
                        break
                    paren_count -= 1

            if end_pos > start_pos:
                args_str = line[start_pos:end_pos]
                args = self._split_args(args_str)
                if len(args) == 2:
                    operator = "!=" if is_not else "=="
                    replacement = f"assert {args[0]} {operator} {args[1]}"
                    line = line[: match.start()] + replacement + line[end_pos + 1 :]
                    return line

        # Handle other patterns
        for pattern, replacement_func in self.conversions:
            line = re.sub(pattern, replacement_func, line)
        return line

    def convert_file(self, filepath: Path, dry_run: bool = False) -> Tuple[bool, int]:
        """
        Convert a single file.

        Returns:
            Tuple of (was_modified, num_conversions)
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return False, 0

        converted_lines = []
        num_conversions = 0

        for line in lines:
            converted_line = self.convert_line(line)
            if converted_line != line:
                num_conversions += 1
            converted_lines.append(converted_line)

        if num_conversions > 0:
            if dry_run:
                # In dry-run mode, just report the changes without writing
                print(f"[DRY RUN] {filepath}: {num_conversions} conversions (no file modified)")
                return True, num_conversions
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.writelines(converted_lines)
                return True, num_conversions
            except Exception as e:
                print(f"Error writing {filepath}: {e}")
                return False, 0

        return False, 0

    def convert_directory(self, test_dir: Path, recursive: bool = True, dry_run: bool = False) -> None:
        """Convert all Python test files in a directory."""
        if not test_dir.exists():
            print(f"Directory {test_dir} does not exist")
            return

        if not test_dir.is_dir():
            print(f"{test_dir} is not a directory")
            return

        pattern = "**/*.py" if recursive else "*.py"
        python_files = list(test_dir.glob(pattern))

        if not python_files:
            print(f"No Python files found in {test_dir}")
            return

        total_files_modified = 0
        total_conversions = 0

        for filepath in python_files:
            # Skip if it doesn't look like a test file
            if not self._is_test_file(filepath):
                continue

            was_modified, num_conversions = self.convert_file(filepath, dry_run=dry_run)

            if was_modified:
                total_files_modified += 1
                total_conversions += num_conversions
                if not dry_run:
                    print(f"✓ {filepath}: {num_conversions} conversions")
            else:
                if not dry_run:
                    print(f"- {filepath}: no changes needed")

        print("\nSummary:")
        print(f"Files processed: {len([f for f in python_files if self._is_test_file(f)])}")
        print(f"Files modified: {total_files_modified}")
        print(f"Total conversions: {total_conversions}")

    def _is_test_file(self, filepath: Path) -> bool:
        """Check if a file appears to be a test file."""
        filename = filepath.name.lower()
        return filename.startswith("test_") or filename.endswith("_test.py") or "test" in filename


def main():
    parser = argparse.ArgumentParser(description="Convert unittest assertions to modern assert statements")
    parser.add_argument(
        "test_dir", type=Path, default="tests", nargs="?", help="Directory containing test files (default: tests)"
    )
    parser.add_argument("--no-recursive", action="store_true", help="Don't search subdirectories recursively")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")

    args = parser.parse_args()

    converter = AssertionConverter()

    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")

    converter.convert_directory(args.test_dir, recursive=not args.no_recursive, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
