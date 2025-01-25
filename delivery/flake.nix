{
  description = "Pierrot's Blog";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {inherit system;};
  in {
    devShells.${system}.default = pkgs.mkShell {
      name = "blog";
      packages = [
        pkgs.gleam
        pkgs.just
        pkgs.erlang_nox
        pkgs.rebar3
        pkgs.marksman
      ];
      shellHook = ''
        export SHELL="/run/current-system/sw/bin/bash"
      '';
    };
  };
}
