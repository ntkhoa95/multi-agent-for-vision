import os

import autopep8


def format_code_in_directory(directory):
    """Format all Python files in the given directory and its subdirectories."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                print(f"Formatting {file_path}")
                options = autopep8.parse_args(
                    ["--aggressive", "--aggressive", "--in-place", file_path]
                )
                autopep8.fix_file(file_path, options=options)


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    format_code_in_directory(current_directory)
