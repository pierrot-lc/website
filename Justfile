BUILD_DIR := "delivery/priv"

build:
    nix build .#website --out-link "{{ BUILD_DIR }}"
