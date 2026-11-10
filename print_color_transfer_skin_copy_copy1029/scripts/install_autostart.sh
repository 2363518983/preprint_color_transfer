#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/lib_python.sh"
LABEL="com.chromawork.print-color"
DST="$HOME/Library/LaunchAgents/${LABEL}.plist"
UID_NUM="$(id -u)"

chmod +x "$ROOT/scripts/start_app.sh" "$ROOT/scripts/setup_env.sh" "$ROOT/scripts/deploy_mac.sh"
mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/logs"

if ! PYTHON="$(resolve_app_python "$ROOT")"; then
  echo "尚未配置 Python。"
  echo "已有 conda 时： $ROOT/scripts/setup_env.sh --conda 环境名"
  echo "没有 conda 时： $ROOT/scripts/setup_env.sh"
  exit 1
fi

cat > "$DST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>${LABEL}</string>
	<key>ProgramArguments</key>
	<array>
		<string>/bin/zsh</string>
		<string>${ROOT}/scripts/start_app.sh</string>
	</array>
	<key>WorkingDirectory</key>
	<string>${ROOT}</string>
	<key>RunAtLoad</key>
	<true/>
	<key>KeepAlive</key>
	<true/>
	<key>StandardOutPath</key>
	<string>${ROOT}/logs/app.out.log</string>
	<key>StandardErrorPath</key>
	<string>${ROOT}/logs/app.err.log</string>
	<key>EnvironmentVariables</key>
	<dict>
		<key>PYTHONUNBUFFERED</key>
		<string>1</string>
		<key>PRINT_COLOR_PYTHON</key>
		<string>${PYTHON}</string>
	</dict>
</dict>
</plist>
EOF

if launchctl print "gui/${UID_NUM}/${LABEL}" >/dev/null 2>&1; then
  launchctl bootout "gui/${UID_NUM}" "$DST" >/dev/null 2>&1 || true
fi
launchctl bootstrap "gui/${UID_NUM}" "$DST"
launchctl enable "gui/${UID_NUM}/${LABEL}" >/dev/null 2>&1 || true
launchctl kickstart -k "gui/${UID_NUM}/${LABEL}" >/dev/null 2>&1 || true

echo "已安装开机/登录自启：$DST"
echo "Python：$PYTHON"
echo "工程目录：$ROOT"
echo "服务地址：http://127.0.0.1:5001"
echo "日志：$ROOT/logs/app.out.log"
