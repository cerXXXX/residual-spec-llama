#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
gcc -O3 -march=native -fopenmp -shared -fPIC -o libresidual.so residual_cpu.c
