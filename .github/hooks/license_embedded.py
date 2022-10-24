import json
import sys
from pathlib import Path

cur_file = sys.argv[1]

file_content = Path(cur_file).read_text()
expected_license = Path("copyright.txt").read_text().rstrip()


def python_check(content: str):
    if not content.startswith(expected_license):
        print(f"'{cur_file}' did not start with copyright.txt content")
        exit(1)


def notebook_check(content: str):
    j = json.loads(content)
    if not j["metadata"]["license"]["full_text"] == expected_license:
        print(
            f"'{cur_file}' does not have the copyright.txt content as metadata. "
            f"This should go into metadata.license.full_text"
        )
        exit(1)


if cur_file.endswith(".py"):
    python_check(file_content)
elif cur_file.endswith(".ipynb"):
    notebook_check(file_content)
else:
    print(f"Incorrect file passed: '{cur_file}', expecting '.py' or '.ipynb'")
    exit(1)

print(f"License check passed for: '{cur_file}'")
exit(0)
