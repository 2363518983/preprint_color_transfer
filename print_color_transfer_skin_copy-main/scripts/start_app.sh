#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/lib_python.sh"
cd "$ROOT"
mkdir -p "$ROOT/logs"

if ! PYTHON="$(resolve_app_python "$ROOT")"; then
  echo "找不到 Python。请先运行 ./scripts/setup_env.sh 或 ./scripts/setup_env.sh --conda 环境名"
  exit 1
fi

export PYTHONUNBUFFERED=1
export PATH="$(dirname "$PYTHON"):/usr/bin:/bin:/usr/sbin:/sbin"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 使用 $PYTHON 启动 app.py"
exec "$PYTHON" "$ROOT/app.py" --host 0.0.0.0 --port 5001
