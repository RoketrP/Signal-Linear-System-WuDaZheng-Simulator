#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

APP_NAME="Signal-Linear-System-WuDazheng"
BUNDLE_ID="${BUNDLE_ID:-com.roketr.signal-linear-system-wudazheng}"
TARGET_ARCH="${TARGET_ARCH:-$(python3 - <<'PY'
import platform
print(platform.machine())
PY
)}"
SIGN_IDENTITY="${APPLE_DEVELOPER_IDENTITY:-}"
NOTARIZE="${NOTARIZE:-0}"
ZIP_PATH="dist/${APP_NAME}.zip"
APP_PATH="dist/${APP_NAME}.app"
EXECUTABLE_PATH="${APP_PATH}/Contents/MacOS/${APP_NAME}"
PLIST_TEMPLATE="packaging/macos/Info.plist"
export PYINSTALLER_CONFIG_DIR="${PROJECT_ROOT}/.pyinstaller"

echo "[1/8] 检查 Xcode Command Line Tools"
xcode-select -p >/dev/null

echo "[2/8] 准备虚拟环境"
if [[ ! -d "venv" ]]; then
  python3 -m venv venv
fi
source venv/bin/activate

echo "[3/8] 安装打包依赖"
python -m pip install -r requirements-macos-build.txt

echo "[4/8] 清理旧构建目录"
python - <<'PY'
from pathlib import Path
import shutil

for folder in (Path("build"), Path("dist")):
    if folder.exists():
        shutil.rmtree(folder)
PY

echo "[5/8] 使用 PyInstaller 生成 macOS .app 包"
pyinstaller \
  --noconfirm \
  --clean \
  --windowed \
  --onedir \
  --name "${APP_NAME}" \
  --osx-bundle-identifier "${BUNDLE_ID}" \
  --target-architecture "${TARGET_ARCH}" \
  --add-data "app.py:." \
  --collect-all streamlit \
  --collect-all matplotlib \
  --collect-all plotly \
  --collect-all scipy \
  --collect-all numpy \
  macos_launcher.py

echo "[6/8] 写入自定义 Info.plist"
cp "${PLIST_TEMPLATE}" "${APP_PATH}/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier ${BUNDLE_ID}" "${APP_PATH}/Contents/Info.plist" || true

echo "[7/8] 代码签名"
if [[ -n "${SIGN_IDENTITY}" ]]; then
  codesign --force --deep --options runtime --sign "${SIGN_IDENTITY}" "${APP_PATH}"
else
  codesign --force --deep --sign - "${APP_PATH}"
fi

echo "[8/8] 生成 ZIP 归档"
ditto -c -k --sequesterRsrc --keepParent "${APP_PATH}" "${ZIP_PATH}"

if [[ "${NOTARIZE}" == "1" ]]; then
  if [[ -z "${APPLE_KEYCHAIN_PROFILE:-}" ]]; then
    echo "NOTARIZE=1 时必须设置 APPLE_KEYCHAIN_PROFILE 环境变量。"
    exit 1
  fi
  xcrun notarytool submit "${ZIP_PATH}" --keychain-profile "${APPLE_KEYCHAIN_PROFILE}" --wait
  xcrun stapler staple "${APP_PATH}"
fi

cat <<EOF
构建完成:
  APP: ${APP_PATH}
  可执行文件: ${EXECUTABLE_PATH}
  ZIP: ${ZIP_PATH}
  架构: ${TARGET_ARCH}
  签名身份: ${SIGN_IDENTITY:-ad-hoc}
EOF
