import json
import sys
from pathlib import Path

expected_license = Path("copyright.txt").read_text().rstrip()


def python_check(content: str, cur_file: str):
    if not content.startswith(expected_license):
        print(f"'{cur_file}' did not start with copyright.txt content")
        exit(1)


def notebook_check(content: str, cur_file: str):
    j = json.loads(content)
    try:
        if not j["metadata"]["license"]["full_text"] == expected_license:
            print(
                f"'{cur_file}' does not have the copyright.txt content as metadata. "
                f"This should go into metadata.license.full_text"
            )
            exit(1)
    except KeyError:
        print(
            f"'{cur_file}' does not have the copyright.txt content as metadata. "
            f"This should go into metadata.license.full_text"
        )
        exit(1)


def check_file(cur_file: str):
    file_content = Path(cur_file).read_text()
    if cur_file.endswith(".py"):
        python_check(file_content, cur_file)
    elif cur_file.endswith(".ipynb"):
        notebook_check(file_content, cur_file)
    else:
        print(f"Incorrect file passed: '{cur_file}', expecting '.py' or '.ipynb'")
        exit(1)


for cur in sys.argv[1:]:
    check_file(cur)
