# pierrot-lc.dev

Hi!

Here are all the files used to produce my blog at [pierrot-lc.dev](https://pierrot-lc.dev). This
projects work exactly as other static site generators work. I convert markdown files to HTML and
then use a simple static server pointing to the directory where HTML files can be found.

The repo is organized as follows:

- `supply/` contains all markdown files, as well as other static files such as the css files
- `factory/` is the program converting the markdown files into HTML files
- `delivery/` is the web server

The whole project can be build using `nix`. `flake.nix` will tie everything together. It builds the
`factory` and `delivery` program as well as some JS dependencies and uses `factory` to convert my
markdown files and place them into a new static directory.

Use `nix run .#website` to build everything and launch a local server.
