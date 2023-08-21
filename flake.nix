{
  nixConfig = { };
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-stable.url = "github:NixOS/nixpkgs/nixos-23.05";
    flake-parts.url = "github:hercules-ci/flake-parts";
    devenv = {
      url = "github:cachix/devenv";
      inputs = { nixpkgs.follows = "nixpkgs"; };
    };
    llvm-src = {
      url = "github:llvm/llvm-project";
      flake = false;
    };
  };

  outputs = inputs@{ flake-parts, self, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ inputs.devenv.flakeModule ];
      systems = [ "x86_64-linux" "aarch64-darwin" ];
      perSystem = { config, self', inputs', pkgs, system, ... }:
        let
          self-mlir = config.packages.mlir;
          # nixfmt depends on ghc, so take it from nixpkgs-stable to have a
          # better chance of a nix cache hit.
          inherit (inputs.nixpkgs-stable.legacyPackages.${system}) nixfmt;
        in {
          packages = {
            # for now, assertions are always on
            mlir = pkgs.callPackage ./nix/mlir {
              inherit (inputs) llvm-src;
              release_version = "18.0.0";
              doCheck = true;
              enableAssertions = true;
              debugVersion = false;
            };
            # Checks are expensive to run and build, so we prefer mlir-unchecked
            # for development shells.
            mlir-unchecked = self-mlir.override { doCheck = false; };
            mlir-debug = self-mlir.override {
              debugVersion = true;
              doCheck = true;
            };
            mlir-debug-unchecked = self-mlir.override {
              debugVersion = true;
              doCheck = false;
            };
          };
          apps = {
            # A tool for working with github actions. https://github.com/nektos/act
            act.program = "${pkgs.act}/bin/act";

            # Runs nixfmt on all .nix files under source control
            lint-nixfmt.program = toString
              (pkgs.writeShellScript "lint-nixfmt" ''
                PATH=${nixfmt}/bin:''${PATH+:$PATH}
                exec ${./scripts/lint-nixfmt.sh} "$@"
              '');
            lint-clang-format.program = toString
              (pkgs.writeShellScript "lint-nixfmt" ''
                PATH=${pkgs.clang-tools}/bin:''${PATH+:$PATH}
                exec ${./scripts/lint-clang-format.sh} "$@"
              '');
          };
          checks = let
            check-mlir = suffix: mlir:
              pkgs.writeShellScript "check-mlir${suffix}" ''
                set -eu
                mlir="${mlir}"
                "$mlir"/bin/llvm-config --version
                "$mlir"/bin/mlir-opt --version
              '';
          in {
            # We sanity-check that our mlir derivations work
            check-mlir = check-mlir "" self-mlir;

            # TODO For now let's not build this in CI, prioritising CI speed.
            # check-mlir-debug = check-mlir "-debug" config.packages.mlir-debug;
          };
          devenv = {
            shells = let

              # inputs needed to build (checked) mlir. There is probably a better way to do this.
              # Including these means the shell can build LLVM in local
              # development workflows excluding mlir from the shell environment.
              mlir-inputs = self-mlir.buildInputs ++ self-mlir.nativeBuildInputs
                ++ self-mlir.propagatedBuildInputs
                ++ self-mlir.propagatedNativeBuildInputs;

              lint-inputs = [ nixfmt ];

              mkShell = shell-mlir:
                { config, pkgs, ... }:
                let
                  rustPlatform = pkgs.makeRustPlatform {
                    inherit (config.languages.rust.toolchain) rustc cargo;
                  };
                in {
                  packages = mlir-inputs ++ lint-inputs
                    ++ pkgs.lib.optional (shell-mlir != null) shell-mlir
                    ++ [ pkgs.clang-tools ]; # clangd and clang-format

                  languages.cplusplus = { enable = true; };

                  env = {
                    CMAKE_PREFIX_PATH =
                      pkgs.lib.concatMapStringsSep ":" toString mlir-inputs;
                  };

                  #  workaround https://github.com/cachix/devenv/issues/760
                  containers = pkgs.lib.mkForce { };
                };
              yes-mlir = mkShell config.packages.mlir-unchecked;
            in {
              inherit yes-mlir;
              default = yes-mlir;
              no-mlir = mkShell null;
              debug-mlir = mkShell config.packages.mlir-debug-unchecked;
              checked-mlir = mkShell config.packages.mlir;
            };
          };
        };
    };
}
