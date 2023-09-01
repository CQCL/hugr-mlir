{ mlir, cmake, ninja, stdenv, lib }:
stdenv.mkDerivation {
  pname = "hugr-mlir";
  version = "0";
  nativeBuildInputs = [ cmake ninja ];
  buildInputs = [ mlir ];
}
