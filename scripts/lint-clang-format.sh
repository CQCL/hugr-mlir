#!/usr/bin/env bash

set -euo pipefail
set -x

git ls-tree --full-tree --name-only -r HEAD | grep -E '^.*\.(cpp|cc|c|h|hpp)$' | xargs clang-format "$@"
