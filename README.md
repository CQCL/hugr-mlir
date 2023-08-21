# hugr-mlir

A prototype for integrating (hugr)[https://github.com/CQCL-DEV/hugr] and (mlir)[https://mlir.llvm.org].

## Building and Dependencies

* A recent build of LLVM including MLIR
* A python installation including libraries: TODO

All dependencies are available through nix, to enter a shell with all dependencies available:

``` sh
$ nix develop --impure
$ llvm-config --version
$ mlir-opt --version
```

This shell includes a customised LLVM install (see nix/mlir) with MLIR, assertions enabled, and static libraries. CI runs against this install.

Alternatively, you may wish to bring your own LLVM build or install tree(say, `$prefix`), in
which case you can use `nix develop .#no-mlir`. This shell does include all LLVM
dependencies. To point cmake at your LLVM tree, pass
`-DMLIR_DIR=$prefix/lib/cmake/mlir` to cmake.

The `.#debug-mlir` shell includes a debug build of our LLVM.

Note that shells that include LLVM will take a considerable time to build (30-60
minutes, depending on your machine) the first time they are used. These builds
will be cached while `llvm-src` in `flake.lock` and the `mlir` derivation in
`flake.nix` are unchanged.

It is not required to use a nix shell to build, cmake will pick up all
dependencies from it's environment.

### Configure

From root of repo:

```sh
$ cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install
```

### Build

```
$ ninja -C build

```

## Components

For now, a cmake project in the root that pulls in MLIR and builds nothing.


