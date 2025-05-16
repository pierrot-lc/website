{
  description = "Website";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    prism = {
      url = "github:PrismJS/prism";
      flake = false;
    };
  };

  outputs = inputs @ {
    self,
    nixpkgs,
    ...
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {inherit system;};
    lib = pkgs.lib;

    katex = pkgs.stdenv.mkDerivation {
      name = "katex";
      unpackPhase = "true";

      buildInputs = [pkgs.nodePackages.katex];

      installPhase = ''
        mkdir -p $out/katex
        cp -r ${pkgs.nodePackages.katex}/lib/node_modules/katex/dist/{katex.min.css,katex.min.js,fonts} $out/katex
      '';
    };

    # https://github.com/NixOS/nixpkgs/blob/master/doc/languages-frameworks/javascript.section.md
    prism = pkgs.buildNpmPackage {
      pname = "prism";
      version = "git";

      src = inputs.prism;
      npmDeps = pkgs.importNpmLock {
        npmRoot = inputs.prism;
      };

      npmConfigHook = pkgs.importNpmLock.npmConfigHook;
      npmBuildScript = "build";
    };
  in {
    packages.${system} = {
      katex = katex;
      prism = prism;
    };
  };
}
