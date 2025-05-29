FROM nixos/nix:latest AS builder

WORKDIR /build
COPY . /build

RUN echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf
RUN nix build .#pages && mkdir static && cp -r result/* static/
RUN nix bundle --bundler github:DavHau/nix-portable -o bundle ./delivery

FROM bash AS run

WORKDIR /app
COPY --from=builder /build/static ./static
COPY --from=builder /build/bundle/bin/delivery .

ARG CERTFILE
ARG KEYFILE
CMD ./delivery /app/static $CERTFILE $KEYFILE
