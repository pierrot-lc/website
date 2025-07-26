{
  nixConfig = {
    extra-substituters = ["https://nix-community.cachix.org"];
    extra-trusted-public-keys = ["nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="];
  };

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };

    cudaPackages = pkgs.cudaPackages_12_8;

    packages = [
      cudaPackages.cudatoolkit
      pkgs.just
      pkgs.uv
      pkgs.python313Packages.venvShellHook
    ];

    libs = [
      cudaPackages.cudatoolkit
      cudaPackages.cudnn
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib

      # Where your local "lib/libcuda.so" lives. If you're not on NixOS,
      # you should provide the right path (likely another one).
      "/run/opengl-driver"
    ];

    shell = pkgs.mkShell {
      name = "associative-scan";
      inherit packages;

      env = {
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
        XLA_FLAGS = "--xla_gpu_cuda_data_dir=${cudaPackages.cudatoolkit}";
      };

      venvDir = "./.venv";
      postShellHook = ''
        uv sync
        just device-check
      '';
    };
  in {
    devShells.${system}.default = shell;
  };
}
