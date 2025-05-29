FROM nixos/nix:latest

ARG CERTFILE
ARG KEYFILE

WORKDIR /website
COPY . /website
RUN nix build .#website --experimental-features "nix-command flakes"

CMD nix run .#website --experimental-features "nix-command flakes" -- $CERTFILE $KEYFILE
