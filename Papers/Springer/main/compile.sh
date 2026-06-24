#!/usr/bin/env bash
# Compile main.tex into main.pdf using tectonic.
set -euo pipefail

cd "$(dirname "$0")"
tectonic main.tex "$@"
