#!/usr/bin/env bash
# Build the OpenCL tooling for the ultra GPU backend.
#
#   probe_devices  - standalone device-capability diagnostic (Phase 1)
#
# Works on macOS (system OpenCL.framework, OpenCL 1.2) and Linux
# (libOpenCL from ocl-icd-opencl-dev / vendor ICD loader).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p engine/build

CXX="${CXX:-}"
EXTRA_LDFLAGS=""

if [[ -z "$CXX" ]]; then
  if command -v clang++ >/dev/null 2>&1; then CXX=clang++
  elif command -v g++ >/dev/null 2>&1; then CXX=g++
  else echo "no C++ compiler found (clang++/g++)" >&2; exit 1; fi
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  EXTRA_LDFLAGS="-framework OpenCL"
else
  EXTRA_LDFLAGS="-lOpenCL"
fi

"$CXX" -O2 -std=gnu++11 -w \
  $EXTRA_LDFLAGS \
  -o engine/build/opencl_probe \
  engine/opencl/probe_devices.cpp
echo "built engine/build/opencl_probe"

if [[ -f splat/itwom3.0.cpp ]]; then
  "$CXX" -O2 -std=gnu++11 -w \
    $EXTRA_LDFLAGS \
    -o engine/build/itm_validate \
    engine/opencl/itm_validate.cpp splat/itwom3.0.cpp
  echo "built engine/build/itm_validate"
else
  echo "skipped engine/build/itm_validate: splat/itwom3.0.cpp not found"
fi

# GPU CLI binary (no splat dependency — the kernel is in itm_kernel.cl)
"$CXX" -O2 -std=gnu++11 -w \
  $EXTRA_LDFLAGS \
  -o engine/build/ultra_cli_opencl \
  engine/opencl/ultra_cli_opencl.cpp
echo "built engine/build/ultra_cli_opencl"
