{
  description = "Website";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";

    factory.url = "./factory";
    delivery.url = "./delivery";
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
    katex = system: let
      pkgs = import nixpkgs {inherit system;};
    in
      pkgs.stdenv.mkDerivation {
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
    highlightjs = system: let
      pkgs = import nixpkgs {inherit system;};
    in
      pkgs.buildNpmPackage {
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

    pages = system: let
      pkgs = import nixpkgs {inherit system;};
      factory_pkg = inputs.factory.packages.${system}.default;
      highlightjs_pkg = highlightjs system;
      katex_pkg = katex system;
    in
      pkgs.stdenv.mkDerivation {
        pname = "pages";
        version = "git";
        src = ./supply;

        builInputs = [
          factory_pkg
          highlightjs_pkg
          katex_pkg
          pkgs.imagemagick
          pkgs.nodePackages.prettier
          pkgs.ubuntu-sans
        ];

        buildPhase = ''
          ${pkgs.imagemagick}/bin/magick favicon.png -resize 32x32 favicon.ico

          mkdir -p ./styles/fonts

          cp ${katex_pkg}/katex.min.css ./styles/
          cp ${katex_pkg}/fonts/* ./styles/fonts/
          cp ${katex_pkg}/katex.min.js ./scripts/

          cp ${highlightjs_pkg}/highlight.css ./styles/
          cp ${highlightjs_pkg}/highlight.min.js ./scripts/

          cp ${pkgs.ubuntu-sans}/share/fonts/truetype/ubuntu-sans/* ./styles/fonts

          find "." -name "*md" -type f | while read -r file; do
              ${factory_pkg}/bin/factory --config "./config.yaml" --md "''${file}" > "''${file%.md}.html"
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

    website = system: let
      pkgs = import nixpkgs {inherit system;};
      delivery_pkg = inputs.delivery.packages.${system}.default;
      pages_pkg = pages system;
    in
      pkgs.stdenv.mkDerivation {
        pname = "website";
        version = "git";
        unpackPhase = "true";

        builInputs = [
          delivery_pkg
          pages_pkg
        ];

        buildPhase = ''
          cp -r ${pages_pkg} pages
          cp -r ${delivery_pkg} delivery
        '';

        installPhase = ''
          mkdir -p $out/bin $out/lib/delivery

          # Patch the delivery's lib locations so that the `priv` directory is
          # found during runtime.
          sed "s|${delivery_pkg}|''${out}|g" delivery/bin/delivery > $out/bin/website
          cp -r delivery/lib/* $out/lib/
          cp -r pages $out/lib/delivery/priv

          chmod u+x $out/bin/website
        '';
      };
  in {
    packages."x86_64-linux" = {
      highlightjs = highlightjs "x86_64-linux";
      katex = katex "x86_64-linux";
      pages = pages "x86_64-linux";
      website = website "x86_64-linux";
    };

    packages."aarch64-linux" = {
      highlightjs = highlightjs "aarch64-linux";
      katex = katex "aarch64-linux";
      pages = pages "aarch64-linux";
      website = website "aarch64-linux";
    };
  };
}
