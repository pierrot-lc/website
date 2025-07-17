from html.parser import HTMLParser
from xml.sax.saxutils import escape as e
from urllib.parse import urljoin
from pathlib import Path
from typing import override
import time

import yaml

type Date = str
type Title = str
type Description = str

type Item = tuple[Path, Title, Date]
type Channel = tuple[Title, Description]


class FeedParser(HTMLParser):
    """Read webpage's metadata.

    https://docs.python.org/3/library/html.parser.html
    """

    published_date: Date | None
    title: Title | None

    def __init__(self):
        super().__init__()

        self.published_date = None
        self.title = None

    def handle_meta(self, attrs: list[tuple[str, str | None]]):
        data = {key: value for key, value in attrs}
        match data.get("name", None):
            case "published-date":
                self.published_date = data.get("content", None)
            case "og:title":
                self.title = data.get("content", None)
            case _:
                pass

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        match tag:
            case "meta":
                self.handle_meta(attrs)
            case _:
                pass


def write_rss(channel: Channel, items: list[Item], root: Path, config: Path, out: Path):
    """Generate the XML file of the RSS feed.

    ---
    Sources:
        https://www.rssboard.org/rss-profile
        https://www.rssboard.org/rss-specification
        https://www.rssboard.org/files/sample-rss-2.xml
        https://www.rssboard.org/rss-validator/
    """
    config: dict[str, str] = yaml.safe_load(config.read_text())

    for prop in ["author", "base-url", "mail"]:
        assert prop in config, f"Missing '{prop}' in YAML config file"

    rss_link = urljoin(config["base-url"], str(out.relative_to(root)))

    rss = '<?xml version="1.0"?>\n'
    rss += '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">\n'

    # Channel specification.
    rss += " <channel>\n"
    rss += f"  <title>{e(channel[0])}</title>\n"
    rss += f"  <link>{e(config['base-url'])}</link>\n"
    rss += f"  <description>{e(channel[1])}</description>\n"
    rss += f'  <atom:link href="{rss_link}" rel="self" type="application/rss+xml" />\n'

    # Items.
    for filepath, title, date in items:
        link = urljoin(config["base-url"], str(filepath.relative_to(root)))
        link = link.replace("/post.html", "/")
        date = time.strptime(date, "%Y-%m-%d")
        date = time.strftime("%a, %d %b %Y 00:00:00 +0000", date)

        rss += "  <item>\n"
        rss += f"   <title>{e(title)}</title>\n"
        rss += f"   <link>{e(link)}</link>\n"
        rss += f"   <description>{e(title)}</description>\n"
        rss += f"   <author>{e(config['mail'])} ({e(config['author'])})</author>\n"
        rss += f"   <pubDate>{e(date)}</pubDate>\n"
        rss += f"   <guid>{e(link)}</guid>\n"
        rss += "  </item>\n"

    rss += " </channel>\n"
    rss += "</rss>"
    out.write_text(rss)


def list_items(directory: Path) -> list[Item]:
    """Search for valid items within the directory and its subdirectories."""
    items: list[Item] = []
    paths: list[Path] = [f for f in directory.glob("*.html")]
    paths += [f for subdir in directory.iterdir() for f in subdir.glob("*.html")]

    for filepath in paths:
        parser = FeedParser()
        parser.feed(filepath.read_text())

        if parser.published_date is None or parser.title is None:
            continue

        items.append((filepath, parser.title, parser.published_date))

    return items


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, help="Website's root", required=True)
    parser.add_argument("--config", type=Path, help="Config file", required=True)
    parser.add_argument("--name", type=str, help="Channel name", required=True)
    parser.add_argument(
        "--description", type=str, help="Channel description", required=True
    )
    parser.add_argument("--out", type=Path, help="Output file", required=True)
    parser.add_argument(
        "directories", type=Path, nargs="+", help="Where the HTML files are"
    )
    args = parser.parse_args()

    items = []
    for directory in args.directories:
        items += list_items(directory)
    items = list(sorted(items, key=lambda i: i[2], reverse=True))
    write_rss((args.name, args.description), items, args.root, args.config, args.out)
