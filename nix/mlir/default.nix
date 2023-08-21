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
    (ps: [ ps.numpy ps.pybind11 ps.psutil ps.pyyaml ]);

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
      # building against.
      # If it becomes annoying to maintain these for multiple llvm versions we
      # can take these as parameters.
      patches = [
        ./updated-gnu-install-dirs.patch
        ./updated-llvm-lit-cfg-add-libs-to-dylib-path.patch
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
        "-DLLVM_ENABLE_ASSERTIONS=${boolToString enableAssertions}"
        "-DLLVM_INCLUDE_TESTS=${boolToString doCheck}"
        "-DMLIR_ENABLE_BINDINGS_PYTHON=On"
      ];
    });
  });

in llvmTools.mlir
