import json
import sys
from pathlib import Path

expected_license = Path("copyright.txt").read_text().rstrip()


def python_check(content: str, cur_file: str) -> int:
    if not content.startswith(expected_license):
        print(f"'{cur_file}' did not start with copyright.txt content")
        return 1
    else:
        return 0


def notebook_check(content: str, cur_file: str) -> int:
    j = json.loads(content)
    code = 0
    try:
        if not j["metadata"]["license"]["full_text"] == expected_license:
            code = 1
    except KeyError:
        code = 1

    if code == 1:
        print(
            f"'{cur_file}' does not have the copyright.txt content as metadata. "
            f"This should go into metadata.license.full_text"
        )
    return code


def check_file(cur_file: str) -> int:
    file_content = Path(cur_file).read_text()
    if cur_file.endswith(".py"):
        return python_check(file_content, cur_file)
    elif cur_file.endswith(".ipynb"):
        return notebook_check(file_content, cur_file)
    else:
        print(f"Incorrect file passed: '{cur_file}', expecting '.py' or '.ipynb'")
        return 1


exit_code = 0
for cur in sys.argv[1:]:
    exit_code = max(exit_code, check_file(cur))
exit(exit_code)
