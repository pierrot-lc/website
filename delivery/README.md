# Delivery

A simple web server to deliver static webpages. Uses gleam 💜!

## Build

- `nix develop` to enter in a development shell
- `nix build` to build the `delivery` binary

## Usage

```sh
./delivery [static-dir]                      # HTTP
./delivery [static-dir] [certfile] [keyfile] # HTTPS
```

You can replace `./delivery` by `gleam run` if you're in the development shell.
