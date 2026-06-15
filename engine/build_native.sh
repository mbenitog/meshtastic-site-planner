#!/usr/bin/env bash
# Build the native engine CLI (golden generation / parity testing).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p engine/build
if [[ -f splat/itwom3.0.cpp ]]; then
  clang++ -O2 -std=gnu++11 -w \
    -o engine/build/splat_cli \
    engine/driver.cpp engine/native/main.cpp splat/itwom3.0.cpp
  echo "built engine/build/splat_cli"
else
  echo "skipped engine/build/splat_cli: splat/itwom3.0.cpp not found"
fi
clang++ -O2 -std=gnu++11 -w \
  -o engine/build/ultra_cli \
  engine/native/ultra_main.cpp
echo "built engine/build/ultra_cli"
