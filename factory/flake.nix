{
  description = "Generate static web pages from markdown";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {inherit system;};

    packages = [
      pkgs.clang
      pkgs.just
      pkgs.marksman
      pkgs.tree-sitter
    ];

    tree-sitter-grammars = pkgs.tree-sitter.withPlugins (p: [
      p.tree-sitter-json
      p.tree-sitter-markdown
    ]);

    tree-sitter-libs = pkgs.stdenv.mkDerivation {
      name = "tree-sitter-libs";
      unpackPhase = "true";

      buildInputs = [tree-sitter-grammars];
      buildPhase = ''
        mkdir -p $out/lib
        for file in "${tree-sitter-grammars}"/*.so; do
          cp "$file" "$out/lib/lib$(basename $file)"
        done
      '';
    };

    libs = [
      pkgs.tree-sitter
      tree-sitter-libs
    ];

    shell = pkgs.mkShell {
      name = "factory";
      inherit packages;
      env = {
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
        TS_GRAMMARS_DIR = "${tree-sitter-libs}/lib";
        TS_INCLUDE_DIR = "${pkgs.tree-sitter}/include";
      };
      shellHook = ''
        export SHELL="/run/current-system/sw/bin/bash"
      '';
    };
  in {
    devShells.${system}.default = shell;
  };
}
