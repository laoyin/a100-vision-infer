#!/usr/bin/env bash
# Compatibility entry point: read-only checks; no installation or environment activation.
set -euo pipefail
cd "$(dirname "$0")/.."
echo 'Using current Python environment. No packages will be installed.'
command -v python >/dev/null
python tools/check_build_env.py
