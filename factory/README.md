# Factory

Convert a markdown file to HTML.

It also reads yaml metadata, from which you can specify a few informations for the converter. Have a
look at `supply/config.yaml` to know what configurations are expected. The added metadata on top of
a markdown file will add and override any configuration present in yaml configuration.

Uses `tree-sitter` to parse the markdown and yaml contents.

The converter does not support all the markdown specifications. I implement the specifications as I
need them for the blog.

## Build

Uses `nix`:

- `nix develop` to enter the development shell
- `nix build .#factory` to build the `factory` binary
- `meson test -C builddir -v` to run the tests

## Usage

```sh
./factory --config [yaml-path] --md [md-path]
```
