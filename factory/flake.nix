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

    # Compile tree-sitter grammars and generate a proper directory with the
    # resulting libraries such that pkg-config can find them.
    tree-sitter-grammars = let
      grammars = pkgs.tree-sitter.withPlugins (p: [
        p.tree-sitter-markdown
        p.tree-sitter-markdown-inline
      ]);
      pkgConfigFile = pkgs.writeText "tree-sitter-grammars.pc" ''
        libdir=''${prefix}/lib

        Name: tree-sitter-grammars
        Description: Grammars implemented for tree-sitter
        URL: ${pkgs.tree-sitter.meta.homepage}
        Version: ${pkgs.tree-sitter.version}
        Libs: -L''${libdir}
      '';
    in
      pkgs.stdenv.mkDerivation {
        name = "tree-sitter-grammars";
        unpackPhase = "true"; # No src.

        buildPhase = ''
          mkdir -p $out/lib/pkgconfig

          # Import the draft of pkg-config file and define prefix.
          cp ${pkgConfigFile} $out/lib/pkgconfig/tree-sitter-grammars.pc
          sed -i "1i prefix=$out" $out/lib/pkgconfig/tree-sitter-grammars.pc

          for file in "${grammars}"/*.so; do
            # Fetch grammar and rename it as a C lib.
            cp "$file" "$out/lib/lib$(basename $file)"

            # Add the lib to pkg-config.
            sed -i "$ s/$/ -l$(basename -s .so $file)/" $out/lib/pkgconfig/tree-sitter-grammars.pc
          done
        '';
      };

    # NOTE: Do not use clangd from `pkgs.clang` as it is poorly integrated with
    # NixOS libc.
    buildInputs = [
      pkgs.just
      pkgs.llvmPackages_latest.clang-tools
      pkgs.meson
      pkgs.ninja
      pkgs.pkg-config
      pkgs.tree-sitter
      tree-sitter-grammars
    ];

    # NOTE: `pkgs.pkg-config` automatically generate the environment variables
    # in the shell based on the `buildInputs` packages. As long as a package
    # have a 'lib/pkgconfig' directory it gets added to the pkg-config path.
    shell = pkgs.mkShell {
      name = "factory";
      inherit buildInputs;
      shellHook = ''
        just setup
        just tests
      '';
    };
  in {
    devShells.${system}.default = shell;
  };
}
