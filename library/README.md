# Library

Read all markdown files of a directory and produce a new file that list them all based on their
title and date. Their title and date are obtained using their YAML content.

## Build

Uses `nix`:

- `nix develop` to enter the development shell
- `nix build .#library` to build the `library` binary

## Usage

```sh
./library --directory [directory-path] --out [out-file]
```
