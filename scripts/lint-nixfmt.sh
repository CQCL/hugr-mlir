#!/usr/bin/env bash

set -euo pipefail
set -x

git ls-tree --full-tree --name-only -r HEAD | grep -E '^.*\.nix$' | xargs nixfmt "$@"
