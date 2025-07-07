---
title: Using NixOS for Deep Learning Projects
date: 2025-07-07
tags: >-
  2025-07-07
---

[NixOS][nix-pills] has changed the way I think about software. You get a smooth linux experience,
once everything is setup. *Once everything is setup.*

When I first installed NixOS, I had only two constraints: use my [neovim configuration][nvim-nix]
and run my deep learning projects. I could do with the rest, at least while properly learning `nix`
and NixOS, but those two I couldn't do without them.

And as usual I guess I am not the only one that wants to run some deep learning code on their
machine. I will also take this as an opportunity to share how I manage my python projects with
`nix`.

This post covers both a [PyTorch][torch] and [JAX][jax] **development environments**. I want an
environment that is easy to use, and that does not enforce the usage of `nix` as much as possible.

## Python Dev

I want my python environment to automatically install the dependencies if needed. Also, I want it to
use standard python tools so that using my project on a non-NixOS machine is seemless.

Here's the flake:

```nix
{
  description = "Python devshell";

  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
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
    };

    packages = [
      pkgs.just
      pkgs.python313Packages.venvShellHook
      pkgs.uv
    ];

    libs = [
      # Numpy external dependency.
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
    ];

    shell = pkgs.mkShell {
      name = "python-devshell";
      inherit packages;

      env = {
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
      };

      venvDir = "./.venv";
      postShellHook = ''
        uv sync
        just tests
      '';
    };
  in {
    devShells.${system}.default = shell;
  };
}
```

If you've never seen a flake before, you might want to look at [this][nix-flakes] and
[this][nix-shells] explanations first.

`pkgs.python313Packages.venvShellHook` automatically creates a new `.venv` directory with python
3.13 if the directory does not already exist.

`postShellHook` is a shell script that is executed everytime you enter the dev shell. I use it to
always install (if needed) the dependencies of my project and run the tests. Note that I use `uv` to
manage my python dependencies! This means that I can easily use the same project with `uv` on
another machine seemlessly, even if that machine does not have `nix`.

## LD_LIBRARY_PATH

The downside of using `uv` is that you will probably encounter missing libraries errors. For
example, if you need `numpy`, you will get an error saying something like:

```
libstdc++.so.6: cannot open shared object file: No such file or directory
```

This is probably what you encounter the most when starting with NixOS. Because NixOS refuses to
expose external libraries, any program that requires such a library will crash. The proper way to
fix it is to explicitely patch the program to point it to its own library with a version that
satisfy the program requirements. This is what allows NixOS to run many versions of any dependency
without conflicts.

Missing libraries is what prevents you from using standard python tools such as `pip` and `uv`. They
will install libraries that won't natively work on NixOS. To allow numpy to find the library it
needs, I populate my devshell's `LD_LIBRARY_PATH` with the libraries needed by my project. That's
not the [ideal solution][avoid-ld-library-path], but I feel it's more balanced than going full `nix`
and having to maintain another setup for other types of machines.

With this, you still always install the same version of each library, but they are shared among all
of your project dependencies, so some conflicts may happen. `uv` with its lockfile will install the
right version of each dependency. Your overall project is still pure.

Most of your problems can be solved by downloading the right library and putting it in your
`LD_LIBRARY_PATH`.

## PyTorch Dev

We have almost everything we need to setup our PyTorch environment. You are able to install most of
your project dependencies, except PyTorch itself. Here's the new thing to add to your flake's
outputs:

```nix
  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };

    packages = [
      pkgs.gnumake
      pkgs.just
      pkgs.python313Packages.venvShellHook
      pkgs.uv
    ];

    libs = [
      pkgs.cudaPackages.cudatoolkit
      pkgs.cudaPackages.cudnn
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib

      # Where your local "lib/libcuda.so" lives. If you're not on NixOS,
      # you should provide the right path (likely another one).
      "/run/opengl-driver"
    ];

    shell = pkgs.mkShell {
      name = "torch";
      inherit packages;

      env = {
        # General libs for PyTorch and Numpy.
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;

        # Specifics for PyTorch's compilation.
        CC = "${pkgs.gcc}/bin/gcc";
        TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
      };

      venvDir = "./.venv";
      postShellHook = ''
        uv sync
        just tests
      '';
    };
  in {
    devShells.${system}.default = shell;
  };
```

It should be pretty easy to follow. I added all required libraries to LD_LIBRARY_PATH, and also
setup some more environment variables to point PyTorch's compiler to the right paths (if you ever
use `torch.compile` in your project).

Because the path to the nvidia drivers are hardcoded (`/run/opengl-driver`), this flake will not
work on a machine that is not running NixOS. On another distribution, your nvidia drivers might get
installed somewhere else. If you use `nix` on a non-NixOS machine, you will have to modify this
path.

With this, once you enter the development shell (`nix develop`), you will be able to use `uv add
torch` to install torch into your `.venv` and it will properly work.

## JAX Dev

Everything we did for PyTorch applies to JAX. Here's the flake's outputs:

```nix
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
    pythonPackages = pkgs.python313Packages;

    packages = [
      pkgs.just
      pkgs.uv
      pythonPackages.venvShellHook
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
      name = "jax";
      inherit packages;

      env = {
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
        XLA_FLAGS = "--xla_gpu_cuda_data_dir=${cudaPackages.cudatoolkit}";
      };

      venvDir = "./.venv";
      postShellHook = ''
        export PATH="$PATH:${cudaPackages.cudatoolkit}/bin"  # Add ptxas to PATH.

        uv sync --group jax-local
        just tests
      '';
    };
  in {
    devShells.${system}.default = shell;
  };
```

The main difference here are the environment variables and the python library to install. For this
to work, you have to declare that you install jax with cuda already installed locally: `uv add
jax[cuda12-local]`. It means that if you happen to run your code on another machine that will
require jax with cuda, you have to use optional dependencies with uv like so:

```toml
# pyproject.toml

#...

[dependency-groups]
jax-local = [
    "jax[cuda12-local]>=0.6.1",
]
jax-cuda = [
    "jax[cuda12]>=0.6.1",
]

[tool.uv]
conflicts = [
  [
    { group = "jax-cuda" },
    { group = "jax-local" },
  ],
]
```

And then you specify which group you want to install with `uv`: `uv sync --group
[jax-local|jax-cuda]`.

Hope this will help new comers on their NixOS journey, as it helped mine!

[avoid-ld-library-path]:    https://discourse.nixos.org/t/what-is-the-nix-way-to-specify-ld-library-path/6407/11
[jax]:                      https://docs.jax.dev/en/latest/index.html
[nix-flakes]:               https://www.youtube.com/watch?v=JCeYq72Sko0
[nix-pills]:                https://nixos.org/guides/nix-pills/01-why-you-should-give-it-a-try.html
[nix-shells]:               https://www.youtube.com/watch?v=0YBWhSNTgV8
[nvim-nix]:                 https://docs.jax.dev/en/latest/index.html
[torch]:                    https://pytorch.org/
