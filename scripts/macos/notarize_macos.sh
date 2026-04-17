#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

APP_NAME="Signal-Linear-System-WuDazheng"
APP_PATH="dist/${APP_NAME}.app"
ZIP_PATH="dist/${APP_NAME}.zip"

if [[ ! -d "${APP_PATH}" ]]; then
  echo "未找到 ${APP_PATH}，请先执行 scripts/macos/build_macos.sh"
  exit 1
fi

if [[ -z "${APPLE_KEYCHAIN_PROFILE:-}" ]]; then
  echo "请先设置 APPLE_KEYCHAIN_PROFILE，例如："
  echo "xcrun notarytool store-credentials SIGNAL_WUDAZHENG_PROFILE --apple-id ... --team-id ... --password ..."
  echo "export APPLE_KEYCHAIN_PROFILE=SIGNAL_WUDAZHENG_PROFILE"
  exit 1
fi

echo "[1/3] 重新打包 ZIP 供公证提交"
rm -f "${ZIP_PATH}"
ditto -c -k --sequesterRsrc --keepParent "${APP_PATH}" "${ZIP_PATH}"

echo "[2/3] 提交 Apple 公证"
xcrun notarytool submit "${ZIP_PATH}" --keychain-profile "${APPLE_KEYCHAIN_PROFILE}" --wait

echo "[3/3] 写入公证票据"
xcrun stapler staple "${APP_PATH}"

echo "公证流程完成：${APP_PATH}"
