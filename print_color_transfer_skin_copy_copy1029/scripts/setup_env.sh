#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/lib_python.sh"
cd "$ROOT"

USE_CONDA=0
CONDA_NAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda)
      USE_CONDA=1
      if [[ $# -ge 2 && "$2" != -* ]]; then
        CONDA_NAME="$2"
        shift 2
      else
        CONDA_NAME="${CONDA_DEFAULT_ENV:-}"
        shift
      fi
      ;;
    --conda=*)
      USE_CONDA=1
      CONDA_NAME="${1#*=}"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
用法:
  ./scripts/setup_env.sh                新建项目 .venv（默认）
  ./scripts/setup_env.sh --conda        使用当前已激活的 conda 环境
  ./scripts/setup_env.sh --conda py39   使用已有 conda 环境 py39

Mac 上已经有 conda 时，用 --conda，不要再另建 .venv。
EOF
      exit 0
      ;;
    *)
      echo "未知参数：$1"
      echo "可用：./scripts/setup_env.sh --conda 环境名"
      exit 1
      ;;
  esac
done

PYTHON=""
if [[ "$USE_CONDA" -eq 1 ]]; then
  if [[ -n "$CONDA_NAME" ]]; then
    if ! PYTHON="$(conda_env_python "$CONDA_NAME")"; then
      echo "找不到 conda 环境：$CONDA_NAME"
      echo "先查看已有环境：conda env list"
      exit 1
    fi
  elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON="$CONDA_PREFIX/bin/python"
    CONDA_NAME="${CONDA_DEFAULT_ENV:-current}"
  else
    echo "未指定 conda 环境名，当前也没有激活环境。"
    echo "示例：./scripts/setup_env.sh --conda py39"
    echo "或先：conda activate 你的环境 && ./scripts/setup_env.sh --conda"
    echo "查看环境：conda env list"
    exit 1
  fi
  echo "使用已有 conda 环境：${CONDA_NAME}"
else
  if ! command -v python3 >/dev/null 2>&1; then
    echo "未找到 python3。已有 conda 时请改用：./scripts/setup_env.sh --conda 环境名"
    exit 1
  fi
  PYTHON="$(command -v python3)"
  echo "未指定 conda，将新建项目虚拟环境 .venv"
  echo "使用解释器：$("$PYTHON" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
  "$PYTHON" -m venv "$ROOT/.venv"
  PYTHON="$ROOT/.venv/bin/python"
fi

echo "Python：$("$PYTHON" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
"$PYTHON" -m pip install -U pip
"$PYTHON" -m pip install -r "$ROOT/requirements.txt"
save_python_cmd "$ROOT" "$("$PYTHON" -c 'import sys; print(sys.executable)')"
require_model_files "$ROOT"

echo
echo "已记录解释器：$PYTHON"
echo "可先手动试跑：  $PYTHON $ROOT/app.py"
echo "再安装登录自启： $ROOT/scripts/install_autostart.sh"
