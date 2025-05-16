{
  description = "Website";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    highlightjs = {
      url = "github:highlightjs/highlight.js";
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
        mkdir -p $out
        cp -r ${pkgs.nodePackages.katex}/lib/node_modules/katex/dist/{katex.min.css,katex.min.js,fonts} $out
      '';
    };

    # https://github.com/NixOS/nixpkgs/blob/master/doc/languages-frameworks/javascript.section.md
    highlightjs = pkgs.buildNpmPackage {
      pname = "highlightjs";
      version = "git";

      nativeBuildInputs = [pkgs.git];

      src = inputs.highlightjs;
      patches = [./fix_rev.patch];
      npmDeps = pkgs.importNpmLock {
        npmRoot = inputs.highlightjs;
      };

      npmConfigHook = pkgs.importNpmLock.npmConfigHook;
      dontNpmBuild = true;

      buildPhase = ''
        node tools/build.js --no-esm --target browser python
      '';
      installPhase = ''
        mkdir -p $out
        cp ./build/highlight.min.js $out
        cp ./src/styles/default.css $out/highlight.css
      '';
    };
  in {
    packages.${system} = {
      katex = katex;
      highlightjs = highlightjs;
    };
  };
}
