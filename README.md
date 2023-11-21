# hugr-mlir

A prototype for integrating (hugr)[https://github.com/CQCL-DEV/hugr] and (mlir)[https://mlir.llvm.org].

## Building and Dependencies

* C++17 compiler and toolchain
* A recent build of LLVM including MLIR
* A python installation including libraries: TODO

It is not required to use a nix shell to build, cmake will pick up all
dependencies from it's environment.

To point cmake at your LLVM tree, pass
`-DMLIR_DIR=$prefix/lib/cmake/mlir` to cmake.

### Nix environment

For convenience, all dependencies are available through nix, to enter a shell with all dependencies available:

``` sh
$ nix develop --impure --extra-experimental-features "nix-command flakes"
$ llvm-config --version
$ mlir-opt --version
```

This shell includes a customised LLVM install (see ./nix/mlir) with MLIR, assertions enabled, and static libraries. CI runs against this install. 

Alternatively, you may wish to bring your own LLVM build or install tree(say, `$prefix`), in
which case you can use `nix develop .#no-mlir`. This shell does include all LLVM
dependencies. To point cmake at your LLVM tree, pass
`-DMLIR_DIR=$prefix/lib/cmake/mlir` to cmake.

The `.#debug-mlir` shell includes a debug build of our LLVM.

Note that shells that include LLVM will take a considerable time to build (30-60
minutes, depending on your machine) the first time they are used. These builds
will be cached while `llvm-src` in `flake.lock` and the `mlir` derivation in
`flake.nix` are unchanged.

#### direnv

[direnv](https://direnv.net/) is a convenient way to configure your shell and editor to transparently use nix provided dependencies.

The following `.envrc` will suffice:
```sh
use flake --impure .#
```


### rust

At present we have a single rust crate, `hugr-rs-bridge`. We will be using at
least `guppy`. We have a cargo workspace in the root of the repo, which we use
to share a `Cargo.lock` and to patched dependencies.  The cmake build system
generates a `/.cargo/config.toml` cargo configuration file with values for
various environment variables. In particular the target directory is configured
to be `$build/hugr-rs-bridge/target`. This means cargo commands run within the
workspace pick up these environment variables and work as usual.

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

### mlir

An mlir extension. Includes a dialect `hugr`and associated tooling.

Python bindings to hugr.

### hugr-rs-bridge

A rust library allowing C++ to call into the `hugr` crate.

## mlir <--> hugr mapping

Our goal is to create a bidirectional mapping between mlir and hugr. We will likely not be able to exactly round trip everything, but it would be good to get close.

We begin by defining a subset of mlir as a domain for this mapping.

We can map an arbitrary no-successor, no-region,  mlir op into hugr by creating an mlir hugr extension with an op "mlir_op" and a type "mlir_type".

We map types in mlir into hugr by special casing a whitelist of types(which include all types in the hugr dialect), and mapping the remainder to `mlir.mlir_type<{dialect}.{name}.{str}>` where `str` is the string serialisation fo the mlir type.

We map ops similarly, special casing a whitelist of ops(which include all ops in the hugr dialect) and mapping the remainder to `mlir.mlir_op` with a signature is defined by applying the type mapping to each of it's arguments and results. Attributes on the op are mapped to node weights on `mlir_op`. Attributes can be mapped to strings in the same way as types. 

mlir locations will be mapped into metadata on the hugr ops.

We will define a dialect Attribute "HugrMetadata" that can be applied to any mlir op and will map to metadata in hugr. We will reserve at least one key in the metadata dictionary, i.e. location, so that a "HugrMetadata" attribute may not contain a key e.g. "org.quantinuum.hugr.mlir.location". Note that dialect Attributes may be discarded by passes (much the same as metadata in hugr).

Note that any mlir region ops, or successor ops, must be whitelisted and special cased. We expect to do this initially for the `ControlFlow`, `Func`, `arith`, `index`, `llvm` dialects. We may extend in the future to `StructuredControlFlow`, `Affine`, `Tensor`, `LinearAlgebra`.

