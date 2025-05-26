{
  description = "Pierrot's Blog";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    shell = system: let
      pkgs = import nixpkgs {inherit system;};
    in
      pkgs.mkShell {
        name = "delivery";
        buildInputs = [
          pkgs.erlang_nox
          pkgs.gleam
          pkgs.just
          pkgs.rebar3
        ];
      };
  in {
    devShells."x86_64-linux".default = shell "x86_64-linux";
    devShells."aarch64-linux".default = shell "aarch64-linux";
  };
}
