{
  description = "Library";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
    ...
  }: let
    python = system: let
      pkgs = import nixpkgs {inherit system;};
    in
      pkgs.python313.withPackages (ps: [ps.pyyaml]);

    shell = system: let
      pkgs = import nixpkgs {inherit system;};
      python_pkg = python system;
    in
      pkgs.mkShell {
        name = "python-devshell";
        packages = [python_pkg];
      };

    library = system: let
      pkgs = import nixpkgs {inherit system;};
      fs = pkgs.lib.fileset;
      python_pkg = python system;
    in
      pkgs.stdenv.mkDerivation {
        pname = "library";
        version = "git";

        propagatedBuildInputs = [python_pkg];

        # From https://nix.dev/tutorials/working-with-local-files
        src = fs.toSource {
          root = ./.;
          fileset = fs.difference ./. (fs.maybeMissing ./result);
        };

        buildPhase = ''
          echo "#!${python_pkg}/bin/python3" > library
          cat ./main.py >> library
          chmod u+x library
        '';

        installPhase = ''
          mkdir -p $out/bin
          cp ./library $out/bin/library
        '';
      };
  in {
    devShells."x86_64-linux".default = shell "x86_64-linux";
    packages."x86_64-linux".default = library "x86_64-linux";

    devShells."aarch64-linux".default = shell "aarch64-linux";
    packages."aarch64-linux".default = library "aarch64-linux";
  };
}
