# Builds an LLVM distribution based on nixpkgs.llvmPackages_git.llvm and a
# commit of github:llvm/llvm-project. The distribution includes MLIR and static
# libraries, may be either Debug or Release, and may or may not have assertions
# enabled.
# TODO: Ensure we can provide RelWithDebInfo
#
{ llvmPackages_git, lib, python3Packages, runCommandNoCC, llvm-src
, release_version, debugVersion ? false, enableAssertions ? true
, doCheck ? false }:
let
  inherit (llvm-src) rev;
  monorepoSrc = llvm-src.outPath;
  rev-version = "git-${builtins.substring 0 7 rev}";

  # These dependencies are found by inspecting the nixpkgs derivation and from
  # mlir/python/requirements.txt
  # TODO: Add a parameter `mkPythonPackages ? pythonPackages: []` allowing the
  #       caller to add packages to the python3 that is used to build
  python3 = python3Packages.python.withPackages
    (ps: [ ps.numpy ps.pybind11 ps.psutil ps.pyyaml ps.lit ]);

  llvmPackages = llvmPackages_git.override (old: {
    inherit monorepoSrc python3;

    officialRelease = null;
    gitRelease = {
      version = release_version;
      inherit rev rev-version;
    };
  });

  llvmTools = llvmPackages.tools.extend (self: super: {
    mlir = (self.llvm.override (old: {
      inherit debugVersion doCheck;
      enablePolly = false;
      enableGoldPlugin = false;
      enablePFM = false;
      enableSharedLibraries = false;
    })).overrideAttrs (old: rec {
      pname = "mlir";
      # This is adapted from nixpkgs, it lowers the store footprint
      src =
        runCommandNoCC "${pname}-src-${release_version}-${rev-version}" { } ''
          mkdir -p "$out"
          cp -r ${monorepoSrc}/cmake "$out"
          cp -r ${monorepoSrc}/mlir "$out"
          cp -r ${monorepoSrc}/llvm "$out"
          cp -r ${monorepoSrc}/third-party "$out"
          # making the source dir writable is convenient for testing
          chmod u+w -R "$out"
        '';

      sourceRoot = "${src.name}/llvm";

      # HACK: The llvm derivation we are overriding adds 'python.withPackages ...'
      # to it's nativeBuildInputs. Unfortunately, this removes the packages we
      # included in `python3`.
      nativeBuildInputs =
        let go = x: if x.name == python3.name then python3 else x;
        in map go old.nativeBuildInputs;

      # This overrides the python that is provided by the base derivation
      passthru = old.passthru or { } // { python = python3; };

      # These are patches from nixpkgs, adapted to the newer llvm we are
      # building against. Links below were to nixpkgs commit 572baf84bf0417b37b13743cc7dbabf850373a96
      # If it becomes annoying to maintain these for multiple llvm versions we
      # can take these as parameters.
      patches = [
        # Adapted from https://github.com/NixOS/nixpkgs/blob/master/pkgs/development/compilers/llvm/git/llvm/gnu-install-dirs.patch
        # Use CMake's GNUInstallDirs to support multiple outputs.
        ./updated-gnu-install-dirs.patch

        # Adapted from https://github.com/NixOS/nixpkgs/blob/master/pkgs/development/compilers/llvm/git/llvm/llvm-lit-cfg-add-libs-to-dylib-path.patch
        # Explanation extracted from https://github.com/NixOS/nixpkgs/blob/master/pkgs/development/compilers/llvm/git/llvm/default.nix:
        #
        # Running the tests involves invoking binaries (like `opt`) that depend on
        # the LLVM dylibs and reference them by absolute install path (i.e. their
        # nix store path).
        #
        # Because we have not yet run the install phase (we're running these tests
        # as part of `checkPhase` instead of `installCheckPhase`) these absolute
        # paths do not exist yet; to work around this we point the loader (`ld` on
        # unix, `dyld` on macOS) at the `lib` directory which will later become this
        # package's `lib` output.
        #
        # Previously we would just set `LD_LIBRARY_PATH` to include the build `lib`
        # dir but:
        #   - this doesn't generalize well to other platforms; `lit` doesn't forward
        #     `DYLD_LIBRARY_PATH` (macOS):
        #     + https://github.com/llvm/llvm-project/blob/0d89963df354ee309c15f67dc47c8ab3cb5d0fb2/llvm/utils/lit/lit/TestingConfig.py#L26
        #   - even if `lit` forwarded this env var, we actually cannot set
        #     `DYLD_LIBRARY_PATH` in the child processes `lit` launches because
        #     `DYLD_LIBRARY_PATH` (and `DYLD_FALLBACK_LIBRARY_PATH`) is cleared for
        #     "protected processes" (i.e. the python interpreter that runs `lit`):
        #     https://stackoverflow.com/a/35570229
        #   - other LLVM subprojects deal with this issue by having their `lit`
        #     configuration set these env vars for us; it makes sense to do the same
        #     for LLVM:
        #     + https://github.com/llvm/llvm-project/blob/4c106cfdf7cf7eec861ad3983a3dd9a9e8f3a8ae/clang-tools-extra/test/Unit/lit.cfg.py#L22-L31
        #
        ./updated-llvm-lit-cfg-add-libs-to-dylib-path.patch

        # Adapted from https://github.com/NixOS/nixpkgs/blob/master/pkgs/development/compilers/llvm/git/llvm/lit-shell-script-runner-set-dyld-library-path.patch
        # Explanation extracted from https://github.com/NixOS/nixpkgs/blob/master/pkgs/development/compilers/llvm/git/llvm/default.nix:
        #
        # `lit` has a mode where it executes run lines as a shell script which is
        # constructs; this is problematic for macOS because it means that there's
        # another process in between `lit` and the binaries being tested. As noted
        # above, this means that `DYLD_LIBRARY_PATH` is cleared which means that our
        # tests fail with dyld errors.
        #
        # To get around this we patch `lit` to reintroduce `DYLD_LIBRARY_PATH`, when
        # present in the test configuration.
        #
        # It's not clear to me why this isn't an issue for LLVM developers running
        # on macOS (nothing about this _seems_ nix specific)..
        ./updated-lit-shell-script-runner-set-dyld-library-path.patch
      ];

      # We could build less of LLVM, in particular we don't need any codegen
      # targets (i.e. one might pass "-DLLVM_TARGETS_TO_BUILD="). We choose
      # this simpler approach so that all tests can be expected to pass.
      cmakeFlags = let boolToString = x: if x then "On" else "Off";
      in old.cmakeFlags or [ ] ++ [
        "-DLLVM_ENABLE_PROJECTS=mlir"
        "-DLLVM_TARGETS_TO_BUILD=host"
        "-DLLVM_INCLUDE_EXAMPLES=Off"
        "-DLLVM_BUILD_EXAMPLES=Off"
        "-DLLVM_ENABLE_ZLIB=Off"
        "-DLLVM_ENABLE_TERMINFO=Off"
        "-DLLVM_ENABLE_ASSERTIONS=${boolToString enableAssertions}"
        "-DLLVM_INCLUDE_TESTS=${boolToString doCheck}"
        "-DMLIR_ENABLE_BINDINGS_PYTHON=On"
        "-DLLVM_BUILD_LLVM_DYLIB=Off"
        "-DLLVM_LINK_LLVM_DYLIB=Off"
        "-DMLIR_BUILD_MLIR_C_DYLIB=On"
        "-DBUILD_SHARED_LIBS=On"
      ];
    });
  });

in llvmTools.mlir
