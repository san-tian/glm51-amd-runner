#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

bash -n "${ROOT}/bootstrap.sh"

find "${ROOT}/reference/glm5-fp8-deploy/scripts" -type f -name "*.sh" -print0 |
  while IFS= read -r -d '' script; do
    bash -n "$script"
  done

echo "ok"
