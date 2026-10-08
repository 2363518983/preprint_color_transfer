# 解析本工程要用的 Python。由其它脚本 source，不要直接执行。
# 优先级：
#   1. PRINT_COLOR_PYTHON
#   2. 工程目录 .python-cmd（setup_env.sh --conda 会写入）
#   3. CONDA_ENV 环境变量
#   4. 工程 .venv
#   5. conda base
#   6. python3

find_conda_bin() {
  local c
  for c in \
    "${CONDA_EXE:-}" \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "$HOME/miniforge3/bin/conda" \
    "$HOME/mambaforge/bin/conda" \
    /opt/homebrew/Caskroom/miniconda/base/bin/conda \
    /usr/local/Caskroom/miniconda/base/bin/conda
  do
    if [[ -n "$c" && -x "$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  return 1
}

conda_env_python() {
  local env_name="$1"
  local conda_bin python_path
  conda_bin="$(find_conda_bin)" || return 1
  python_path="$("$conda_bin" run -n "$env_name" python -c 'import sys; print(sys.executable)' 2>/dev/null)" || return 1
  python_path="${python_path##*$'\n'}"
  if [[ -z "$python_path" || ! -x "$python_path" ]]; then
    return 1
  fi
  echo "$python_path"
}

save_python_cmd() {
  local root="$1"
  local python_path="$2"
  printf '%s\n' "$python_path" > "$root/.python-cmd"
}

resolve_app_python() {
  local root="$1"
  local python_path conda_bin

  if [[ -n "${PRINT_COLOR_PYTHON:-}" && -x "$PRINT_COLOR_PYTHON" ]]; then
    echo "$PRINT_COLOR_PYTHON"
    return 0
  fi

  if [[ -f "$root/.python-cmd" ]]; then
    python_path="$(<"$root/.python-cmd")"
    python_path="${python_path%%$'\n'}"
    if [[ -x "$python_path" ]]; then
      echo "$python_path"
      return 0
    fi
  fi

  if [[ -n "${CONDA_ENV:-}" ]]; then
    if python_path="$(conda_env_python "$CONDA_ENV")"; then
      echo "$python_path"
      return 0
    fi
  fi

  if [[ -x "$root/.venv/bin/python" ]]; then
    echo "$root/.venv/bin/python"
    return 0
  fi

  conda_bin="$(find_conda_bin || true)"
  if [[ -n "$conda_bin" ]]; then
    python_path="$(dirname "$conda_bin")/python"
    if [[ -x "$python_path" ]]; then
      echo "$python_path"
      return 0
    fi
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  return 1
}

require_model_files() {
  local root="$1"
  local f
  for f in \
    checkpoints/curve_pred_best.pth \
    checkpoints/curves_rgb.pt \
    utils/PSOcoated_v3.icc
  do
    if [[ ! -f "$root/$f" ]]; then
      echo "缺少必要文件：$f"
      echo "请把完整工程（含 checkpoints 和 ICC）拷到这台 Mac 后再部署。"
      return 1
    fi
  done
}
