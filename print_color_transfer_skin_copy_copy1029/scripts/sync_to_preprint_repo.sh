#!/bin/zsh
# 将本仓库内容同步到 fireflycsq/preprint_color_transfer 的子目录并推送。
# 用法: ./scripts/sync_to_preprint_repo.sh
# 需已配置 GitHub SSH（推荐）或 HTTPS 凭据。

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBDIR="print_color_transfer_skin_copy_copy1029"
WORKDIR="${ROOT}/.push_preprint"
REPO="git@github.com:fireflycsq/preprint_color_transfer.git"

cd "$ROOT"
if [[ ! -d "$WORKDIR/.git" ]]; then
  git clone --depth 1 "$REPO" "$WORKDIR"
else
  git -C "$WORKDIR" pull --ff-only origin main
fi

DEST="$WORKDIR/$SUBDIR"
mkdir -p "$DEST"
rsync -a --delete \
  --exclude '.git' \
  --exclude '.push_preprint' \
  --exclude '.merge_upstream_src' \
  --exclude 'web_data' \
  --exclude 'mask_cache' \
  --exclude 'cache' \
  --exclude 'logs' \
  --exclude 'runs' \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  --exclude 'train_*.log' \
  "$ROOT/" "$DEST/"

cd "$WORKDIR"
git add "$SUBDIR"
if git diff --staged --quiet; then
  echo "无变更，跳过提交。"
  exit 0
fi
git commit -m "Sync ${SUBDIR} from print_color_transfer_skin_copy ($(date '+%Y-%m-%d'))."
git push origin main
echo "已推送到 https://github.com/fireflycsq/preprint_color_transfer/tree/main/${SUBDIR}"
