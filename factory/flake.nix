{
  description = "Generate static web pages from markdown";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    # Compile tree-sitter grammars and generate a proper directory with the
    # resulting libraries such that pkg-config can find them.
    tree-sitter-grammars = system: let
      pkgs = import nixpkgs {inherit system;};

      grammars = pkgs.tree-sitter.withPlugins (p: [
        p.tree-sitter-markdown
        p.tree-sitter-markdown-inline
        p.tree-sitter-yaml
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
    buildInputs = system: let
      pkgs = import nixpkgs {inherit system;};
    in [
      pkgs.just
      pkgs.lldb
      pkgs.llvmPackages_latest.clang-tools
      pkgs.meson
      pkgs.ninja
      pkgs.pkg-config
      pkgs.tree-sitter
      (tree-sitter-grammars system)
    ];

    # NOTE: `pkgs.pkg-config` automatically generate the environment variables
    # in the shell based on the `buildInputs` packages. As long as a package
    # have a 'lib/pkgconfig' directory it gets added to the pkg-config path.
    shell = system: let
      pkgs = import nixpkgs {inherit system;};
    in
      pkgs.mkShell {
        name = "factory";
        buildInputs = buildInputs system;
        shellHook = ''
          just setup
          just tests
        '';
      };

    factory = system: let
      pkgs = import nixpkgs {inherit system;};
      fs = pkgs.lib.fileset;
    in
      pkgs.stdenv.mkDerivation {
        pname = "factory";
        version = "git";
        doCheck = true;

        buildInputs = buildInputs system;

        # From https://nix.dev/tutorials/working-with-local-files
        src = fs.toSource {
          root = ./.;
          fileset = fs.difference ./. (fs.maybeMissing ./result);
        };

        configurePhase = ''
          meson setup builddir
        '';

        buildPhase = ''
          meson compile -C builddir
        '';

        checkPhase = ''
          meson test -C builddir -v
        '';

        installPhase = ''
          mkdir -p $out/bin
          cp ./builddir/factory $out/bin/factory
        '';
      };
  in {
    devShells."x86_64-linux".default = shell "x86_64-linux";
    packages."x86_64-linux".default = factory "x86_64-linux";

    devShells."aarch64-linux".default = shell "aarch64-linux";
    packages."aarch64-linux".default = factory "aarch64-linux";
  };
}
