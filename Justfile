BUILD_DIR := "delivery/priv"

build:
    nix build .#pages --out-link "{{ BUILD_DIR }}"

phd-entry:
    touch supply/phd-journal/$(date +%Y-%m-%d).md
