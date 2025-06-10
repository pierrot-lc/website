{
  description = "Pierrot's Blog";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    nix-gleam.url = "github:arnarg/nix-gleam";
  };

  outputs = {
    self,
    nix-gleam,
    nixpkgs,
  }: let
    shell = system: let
      pkgs = import nixpkgs {inherit system;};
    in
      pkgs.mkShell {
        name = "delivery";
        buildInputs = [
          pkgs.beamMinimalPackages.erlang
          pkgs.gleam
          pkgs.just
          pkgs.rebar3
        ];
      };
    delivery = system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [nix-gleam.overlays.default];
      };
    in
      pkgs.buildGleamApplication {
        pname = "delivery";
        version = "git";

        src = ./.;
      };
  in {
    devShells."x86_64-linux".default = shell "x86_64-linux";
    packages."x86_64-linux".default = delivery "x86_64-linux";

    devShells."aarch64-linux".default = shell "aarch64-linux";
    packages."aarch64-linux".default = delivery "aarch64-linux";
  };
}
