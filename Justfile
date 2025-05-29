BUILD_DIR := "delivery/priv"

build:
    nix build .#pages --out-link "{{ BUILD_DIR }}"

docker-build:
    docker build --network host --tag pierrotlc/website --target run .
