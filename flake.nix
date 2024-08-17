{
  description = "Pierrot's Website with Jekyll";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {inherit system;};
    rubyPackages = ps: with ps; [
      # jekyll
      # webrick
    ];
  in {
    devShells.${system}.default = pkgs.mkShell {
      name = "website";
      buildInputs = [
        (pkgs.ruby.withPackages rubyPackages)
        pkgs.bundix
        pkgs.just
        pkgs.marksman
      ];
    };
  };
}
