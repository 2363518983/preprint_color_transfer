#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
chmod +x "$ROOT/scripts/setup_env.sh" "$ROOT/scripts/install_autostart.sh" "$ROOT/scripts/start_app.sh"
"$ROOT/scripts/setup_env.sh" "$@"
"$ROOT/scripts/install_autostart.sh"
