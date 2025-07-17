from pathlib import Path

import yaml

type Title = str
type Date = str
type Item = tuple[Path, Title, Date]


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


def list_items(directory: Path) -> list[Item]:
    """Search for valid items within the directory and its subdirectories."""
    items: list[Item] = []
    paths: list[Path] = [f for f in directory.glob("*.md")]
    paths += [f for subdir in directory.iterdir() for f in subdir.glob("*.md")]

    for filepath in paths:
        data = extract_yaml_from_markdown(filepath.read_text())

        if data is None or "title" not in data or "date" not in data:
            continue

        items.append((filepath, data["title"], data["date"]))

    return items


def write_file(files: list[Item], out: Path):
    """Process the data and write their content into the file.

    1. Sort files by their date.
    2. Rename to their relative path w.r.t. out.
    3. Replace "/posts.html" by "/".
    4. Rename their type to be .html.
    """
    content: list[str] = []

    for filepath, title, date in sorted(files, key=lambda f: f[2], reverse=True):
        filepath = filepath.relative_to(out.parent)
        filepath = filepath.with_suffix(".html")
        filepath = str(filepath).replace("/post.html", "/")
        content.append(f"- [{title}]({filepath}) - *{date}*")

    out.write_text("\n".join(content))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory", type=Path, help="Where the markdown files are", required=True
    )
    parser.add_argument(
        "--out", type=Path, help="Where to write the list", required=True
    )
    args = parser.parse_args()

    items = list_items(args.directory)
    write_file(items, args.out)
