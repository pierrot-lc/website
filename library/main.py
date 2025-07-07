from pathlib import Path

import yaml


def extract_yaml_from_markdown(content: str) -> dict[str, str] | None:
    """Return the YAML part of the given markdown content, if any.

    The markdown file is expected to follow the format:

    ```md
    ---
    some: yaml
    file: value
    ---

    Some markdown content here.
    ```
    """
    if not content.startswith("---\n"):
        return None
    content = content.split("---\n")[1]
    return yaml.safe_load(content)


def list_items(directory: Path) -> list[tuple[Path, str, str]]:
    files = [filepath for filepath in directory.glob("*.md")]
    files += [
        filepath
        for subdir in directory.iterdir()
        if subdir.is_dir()
        for filepath in subdir.glob("*.md")
    ]
    files = [
        (filepath, extract_yaml_from_markdown(filepath.read_text()))
        for filepath in files
    ]
    files = [(filepath, data) for filepath, data in files if data is not None]
    files = [
        (filepath, data["title"], data["date"])
        for filepath, data in files
        if "title" in data and "date" in data
    ]
    return files


def write_file(files: list[tuple[Path, str, str]], out: Path):
    """Process the data and write their content into the file.

    1. Sort files by their date.
    2. Rename to their relative path w.r.t. out.
    3. Rename their type to be .html.
    """
    files = list(sorted(files, key=lambda f: f[2], reverse=True))
    files = [(f[0].relative_to(out.parent), f[1], f[2]) for f in files]
    files = [(f[0].with_suffix(".html"), f[1], f[2]) for f in files]
    content = [f"- [{title}]({filepath}) - *{date}*" for filepath, title, date in files]
    content = "\n".join(content)
    out.write_text(content)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=Path,
        help="Directory containing the markdown files",
        required=True,
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Where to write the list",
        required=True,
    )

    args = parser.parse_args()
    files = list_items(args.directory)
    write_file(files, args.out)
