{
  description = "Website";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";

    factory.url = "./factory";
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

    factory = inputs.factory.packages.${system}.default;

    katex = pkgs.stdenv.mkDerivation {
      pname = "katex";
      version = pkgs.nodePackages.katex.version;
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
      version = inputs.highlightjs.shortRev;
      src = inputs.highlightjs;

      npmDeps = pkgs.importNpmLock {npmRoot = inputs.highlightjs;};
      npmConfigHook = pkgs.importNpmLock.npmConfigHook;

      patchPhase = ''
        # Fix the automatic git rev detection.
        sed -i '20,22c\git_sha: "${inputs.highlightjs.shortRev}"' tools/build_browser.js
      '';
      buildPhase = ''
        node tools/build.js --no-esm --target browser python
      '';
      installPhase = ''
        mkdir -p $out
        cp ./build/highlight.min.js $out
        cp ./src/styles/default.css $out/highlight.css
      '';
    };

    website = pkgs.stdenv.mkDerivation {
      pname = "website";
      version = "git";
      src = ./supply;

      builInputs = [
        factory
        highlightjs
        katex
        pkgs.nodePackages.prettier
      ];

      buildPhase = ''
        cp ${katex}/katex.min.css ./styles/
        cp -r ${katex}/fonts ./styles/
        cp ${katex}/katex.min.js ./scripts/

        cp ${highlightjs}/highlight.css ./styles/
        cp ${highlightjs}/highlight.min.js ./scripts/

        find "." -name "*md" -type f | while read -r file; do
            ${factory}/bin/factory --config "./config.yaml" --md "''${file}" > "''${file%.md}.html"
        done

        find "." -name "*html" -type f | while read -r file; do
            ${pkgs.nodePackages.prettier}/bin/prettier --write "''${file}"
        done
      '';

      installPhase = ''
        mkdir $out
        cp -r . $out/
      '';
    };
  in {
    packages.${system} = {
      highlightjs = highlightjs;
      katex = katex;
      website = website;
    };
  };
}
