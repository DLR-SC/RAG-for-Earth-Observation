from pathlib import Path


def count_lines_dir(filepath: str | Path, filetype: str) -> int:
    """Count number of lines in a directory."""
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.is_dir():
        msg = "Provided filepath is not a directory"
        raise ValueError(msg)

    total = 0
    for file in filepath.rglob(f"*.{filetype}"):
        total += sum(1 for _ in file.open("rb"))

    return total

def count_lines_file(filepath: str | Path) -> int:
    """Count number of lines in a file."""
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.is_file():
        msg = "Provided filepath is not a file"
        raise ValueError(msg)

    return sum(1 for _ in filepath.open("rb"))
