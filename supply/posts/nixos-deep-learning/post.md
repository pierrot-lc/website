---
title: NixOS & Deep Learning
date: 2025-07-08
---

[NixOS][nix-pills] has changed the way I think about software. Once everything is setup, you get a
smooth linux experience. *Once everything is setup.* When I first installed NixOS, I had only two
constraints: [neovim][nvim-nix] and CUDA. I could deal with the rest, at least while properly
learning nix and NixOS.

This post covers both [PyTorch][torch] and [JAX][jax] **development environments**. I want an
environment that is easy to use, and that does not enforce the usage of nix as much as possible.
This last part is important as I also have to run my code on a server where I don't have access to
nix (sadly).

I've often seen that people struggle with python development on NixOS. I will also use this post as
an opportunity to share how I manage my python projects with nix.

## Python & NixOS

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
      '';
    };
  in {
    devShells.${system}.default = shell;
  };
}
```

If you've never seen a flake before, you might want to look for [some][nix-flakes]
[explanations][nix-shells] first.

`pkgs.python313Packages.venvShellHook` automatically creates a new `.venv` directory with python
3.13, if the directory does not already exist (not pure!).

`postShellHook` is a shell script that is executed every time you enter the dev shell. I use it to
install the dependencies of my project (if needed) and run the tests. Note that I use uv to manage
my python dependencies! This means that I can easily use the same project with uv on another machine
seemlessly, even if that machine does not have nix.

When I create my project, I enter the naked development shell, run `uv init` and start populating
the `pyproject.toml` file with uv. This will ensure that uv is using my exact python version
specified in my flake. Running `uv sync` on a new machine will download the right version of python
and the project dependencies.

## LD_LIBRARY_PATH

The downside of using uv is that you will probably encounter missing libraries errors. For
example, if you need numpy, you will get an error saying something like:

```
libstdc++.so.6: cannot open shared object file: No such file or directory
```

This is probably what you encounter the most when starting with NixOS. Because NixOS refuses to
expose external libraries, any program that requires such a library will crash. The proper way to
fix it is to explicitly patch the program and point it to its dedicated libraries. This is what
allows NixOS to run many versions of any dependency without conflicts.

Missing libraries is what prevents you from using standard python tools such as pip and uv. They
will install libraries that won't natively work on NixOS. Hence, I populate my devshell's
`LD_LIBRARY_PATH` to allow my dependencies to find the libraries they need. That's not the [ideal
solution][avoid-ld-library-path], but I feel it's more balanced than going full nix and having to
maintain another setup for other types of machines.

With this, you have a unique version of each library that are shared among all of your project
dependencies. Some conflicts may happen, but that's unlikely. uv with its lockfile will install the
right version of each dependency. Your overall project is still pure.

## PyTorch & NixOS

Most of your python problems can be solved by downloading the right library and putting it in your
`LD_LIBRARY_PATH`. But some complex dependencies might require some more tweaks in your flake. We
have almost everything we need to setup our PyTorch environment. Here's the new thing to add to your
*flake's outputs*:

```nix
  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      # CUDA is sadly not free.
      config.allowUnfree = true;  # mark-line
    };

    packages = [
      # For PyTorch's compilation.
      pkgs.gnumake  # mark-line
      pkgs.python313Packages.venvShellHook
      pkgs.uv
    ];

    libs = [
      pkgs.cudaPackages.cudatoolkit  # mark-line
      pkgs.cudaPackages.cudnn  # mark-line
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib

      # Where your local "lib/libcuda.so" lives. If you're not on NixOS,
      # you should provide the right path (likely another one).
      "/run/opengl-driver"  # mark-line
    ];

    shell = pkgs.mkShell {
      name = "torch";
      inherit packages;

      env = {
        # General libs for PyTorch and NumPy.
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;

        # For PyTorch's compilation.
        CC = "${pkgs.gcc}/bin/gcc";  # mark-line
        TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";  # mark-line
      };

      venvDir = "./.venv";
      postShellHook = ''
        uv sync
      '';
    };
  in {
    devShells.${system}.default = shell;
  };
```

It should be pretty easy to follow. I added all required libraries to LD_LIBRARY_PATH, and also
setup some more environment variables to point PyTorch's compiler to the right paths (if you ever
use `torch.compile` in your project).

Because the path to the nvidia drivers are hardcoded (`"/run/opengl-driver"`), this flake will not
work on a machine that is not running NixOS. On another distribution, your nvidia drivers might get
installed somewhere else. If you use nix on a non-NixOS machine, you will have to modify this path.

With this, once you enter the development shell, you will be able to use `uv add torch` to install
torch into your `.venv`.

## JAX & NixOS

Everything we did for PyTorch applies to JAX. Here's the flake's outputs:

```nix
  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;  # mark-line
    };

    cudaPackages = pkgs.cudaPackages_12_8;  # mark-line

    packages = [
      # Add ptxas to PATH.
      cudaPackages.cudatoolkit  # mark-line
      pkgs.python313Packages.venvShellHook
      pkgs.uv
    ];

    libs = [
      cudaPackages.cudatoolkit  # mark-line
      cudaPackages.cudnn  # mark-line
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib

      "/run/opengl-driver"  # mark-line
    ];

    shell = pkgs.mkShell {
      name = "jax";
      inherit packages;

      env = {
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
        XLA_FLAGS = "--xla_gpu_cuda_data_dir=${cudaPackages.cudatoolkit}";  # mark-line
      };

      venvDir = "./.venv";
      postShellHook = ''
        uv sync --group jax-local
      '';
    };
  in {
    devShells.${system}.default = shell;
  };
```

For this to work, you have to install jax and specify that CUDA is already installed locally: `uv
add jax[cuda12-local]`. On a non-NixOS machine, you probably will prefer the `jax[cuda12]`
dependency that comes with CUDA. You can use optional dependencies with uv like so:

```toml
# pyproject.toml

# ...

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

And then you specify which group you want to install with uv: `uv sync --group
[jax-local|jax-cuda]`.

## Final Thoughts

As said at the beginning, this post covers only the python development shell. The story is
completely different if what you need is to package your python project with nix.

Using flakes not only allows someone to run the python project on NixOS but also makes it really
easy to choose your CUDA dependencies. At some point I had to specifically install the beta version
of my nvidia drivers with the latest version of CUDA. When your whole setup is declarative, those
kind of dependencies is only two lines updates.

Hope this will help new comers on their NixOS journey, as it helped mine!

[avoid-ld-library-path]:    https://discourse.nixos.org/t/what-is-the-nix-way-to-specify-ld-library-path/6407/11
[jax]:                      https://docs.jax.dev/en/latest/index.html
[nix-flakes]:               https://www.youtube.com/watch?v=JCeYq72Sko0
[nix-pills]:                https://nixos.org/guides/nix-pills/01-why-you-should-give-it-a-try.html
[nix-shells]:               https://www.youtube.com/watch?v=0YBWhSNTgV8
[nvim-nix]:                 https://github.com/pierrot-lc/nvim-nix
[torch]:                    https://pytorch.org/
