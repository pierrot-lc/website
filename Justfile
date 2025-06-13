BUILD_DIR := "delivery/priv"

build:
    nix build .#pages --out-link "{{ BUILD_DIR }}"
