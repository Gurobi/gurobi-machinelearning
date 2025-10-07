import json
import sys
from datetime import datetime
from pathlib import Path

expected_license = Path("copyright.txt").read_text().rstrip()


def _allowed_first_line(line: str, base_first_line: str) -> bool:
    """
    Allow first line to either be:
      - a single current year (YYYY), or
      - a range START-YYYY where START is in [2023, YYYY] and YYYY is the current year.

    The rest of the text (before and after the year section) must match the base template.
    """
    import re

    m = re.search(r"\b(\d{4})\b", base_first_line)
    if not m:
        # If no year in base, require exact match
        return line == base_first_line

    pre = base_first_line[: m.start()]
    post = base_first_line[m.end() :]

    if not (line.startswith(pre) and line.endswith(post)):
        return False

    middle = line[len(pre) : len(line) - len(post)]
    current_year = datetime.utcnow().year

    # Accept single current year
    if re.fullmatch(r"\d{4}", middle):
        return int(middle) == current_year

    # Accept range START-CURRENT_YEAR
    m2 = re.fullmatch(r"(\d{4})-(\d{4})", middle)
    if not m2:
        return False

    start, end = int(m2.group(1)), int(m2.group(2))
    if end != current_year:
        return False
    if start < 2023 or start > end:
        return False
    return True


def license_header_matches(text: str) -> bool:
    base_lines = expected_license.splitlines()
    lines = text.splitlines()
    if len(lines) < len(base_lines):
        return False

    # Compare first line with flexible year
    if not _allowed_first_line(lines[0], base_lines[0]):
        return False

    # Compare remaining lines exactly
    for i in range(1, len(base_lines)):
        if lines[i] != base_lines[i]:
            return False

    return True


def python_check(content: str, cur_file: str) -> str:
    if not license_header_matches(content):
        y = datetime.utcnow().year
        return (
            f"'{cur_file}' did not start with copyright.txt content. "
            f"First line must use either '{y}' or 'START-{y}' with START>=2023."
        )
    return ""


def notebook_check(content: str, cur_file: str) -> str:
    j = json.loads(content)
    try:
        full_text = j["metadata"]["license"]["full_text"]
    except KeyError:
        return (
            f"'{cur_file}' does not have the copyright.txt content as metadata. "
            f"This should go into metadata.license.full_text"
        )

    if not isinstance(full_text, str) or not license_header_matches(full_text):
        y = datetime.utcnow().year
        return (
            f"'{cur_file}' license metadata does not match expected text. "
            f"First line must use either '{y}' or 'START-{y}' with START>=2023."
        )
    return ""


def check_file(cur_file: str) -> str:
    file_content = Path(cur_file).read_text()
    if cur_file.endswith(".py"):
        return python_check(file_content, cur_file)
    elif cur_file.endswith(".ipynb"):
        return notebook_check(file_content, cur_file)
    else:
        return f"Incorrect file passed: '{cur_file}', expecting '.py' or '.ipynb'"


exit_code = 0
for cur in sys.argv[1:]:
    if error := check_file(cur):
        print(f"License check FAILED for: '{cur}': {error}")
        exit_code = 1
    else:
        print(f"License check passed for: '{cur}'")

exit(exit_code)
