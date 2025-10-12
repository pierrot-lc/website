{
  description = "Website";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";

    delivery.url = "./delivery";
    factory.url = "./factory";
    library.url = "./library";
    highlightjs = {
      url = "github:highlightjs/highlight.js";
      flake = false;
    };
    modern-normalize = {
      url = "github:sindresorhus/modern-normalize";
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
          node tools/build.js --no-esm --target browser python nix ini
        '';
        installPhase = ''
          mkdir -p $out
          cp ./build/highlight.min.js $out
          cp ./src/styles/default.css $out/highlight.css
        '';
      };

    modern-normalize = system: let
      pkgs = import nixpkgs {inherit system;};
    in
      pkgs.stdenv.mkDerivation {
        pname = "modern-normalize";
        version = inputs.modern-normalize.shortRev;
        src = inputs.modern-normalize;

        buildPhase = ''
        '';
        installPhase = ''
          mkdir -p $out
          cp ./modern-normalize.css $out
        '';
      };

    pages = system: let
      pkgs = import nixpkgs {inherit system;};
      factory_pkg = inputs.factory.packages.${system}.default;
      highlightjs_pkg = highlightjs system;
      katex_pkg = katex system;
      modern-normalize_pkg = modern-normalize system;
      library_pkg = inputs.library.packages.${system}.default;
    in
      pkgs.stdenv.mkDerivation {
        pname = "pages";
        version = "git";
        src = ./supply;

        builInputs = [
          factory_pkg
          highlightjs_pkg
          katex_pkg
          library_pkg
          modern-normalize_pkg
          pkgs.imagemagick
          pkgs.jetbrains-mono
        ];

        buildPhase = ''
          # List the different articles into a dedicated page.
          ${library_pkg}/bin/listing --directory ./posts --out ./posts.md
          ${library_pkg}/bin/listing --directory ./phd-journal --out ./journal.md
          ${library_pkg}/bin/listing --directory ./paper-reviews --out ./reviews.md

          # Merge posts and paper reviews.
          echo "## Posts" > ./combined.md
          cat ./posts.md >> ./combined.md
          echo "" >> ./combined.md
          echo "## Paper Reviews" >> ./combined.md
          cat ./reviews.md >> ./combined.md

          ${pkgs.imagemagick}/bin/magick favicon.png -resize 32x32 favicon.ico

          mkdir -p ./styles/fonts

          cp ${katex_pkg}/katex.min.css ./styles/
          cp ${katex_pkg}/fonts/* ./styles/fonts/
          cp ${katex_pkg}/katex.min.js ./scripts/

          cp ${highlightjs_pkg}/highlight.css ./styles/
          cp ${highlightjs_pkg}/highlight.min.js ./scripts/

          cp ${modern-normalize_pkg}/modern-normalize.css ./styles/

          cp ${pkgs.jetbrains-mono}/share/fonts/truetype/* ./styles/fonts

          find "." -name "*md" -type f | while read -r file; do
              ${factory_pkg}/bin/factory --config "./config.yaml" --md "''${file}" > "''${file%.md}.html"
          done

          ${library_pkg}/bin/feed --name "Pierrot's Posts" --description "All my posts" --root . --config ./config.yaml --out ./feed.xml ./posts ./paper-reviews
          ${library_pkg}/bin/feed --name "Pierrot's Journal" --description "Weekly PhD Journal" --root . --config ./config.yaml --out ./phd-journal/feed.xml ./phd-journal
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
          mkdir -p $out/bin

          # Add the priv directory as first argument.
          sed "s|-extra|-extra \"''$out/priv\"|g" delivery/bin/delivery > $out/bin/website
          chmod u+x $out/bin/website
          cp -r pages $out/priv
        '';
      };
  in {
    packages."x86_64-linux" = {
      highlightjs = highlightjs "x86_64-linux";
      katex = katex "x86_64-linux";
      pages = pages "x86_64-linux";
      website = website "x86_64-linux";
      default = website "x86_64-linux";
    };

    packages."aarch64-linux" = {
      highlightjs = highlightjs "aarch64-linux";
      katex = katex "aarch64-linux";
      pages = pages "aarch64-linux";
      website = website "aarch64-linux";
      default = website "aarch64-linux";
    };
  };
}
