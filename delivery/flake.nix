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

    shell = pkgs.mkShell {
      name = "delivery";
      packages = [
        pkgs.erlang_nox
        pkgs.gleam
        pkgs.just
        pkgs.rebar3
      ];
    };
  in {
    devShells.${system}.default = shell;
  };
}
