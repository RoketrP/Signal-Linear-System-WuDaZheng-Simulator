# macOS 原生打包与安装说明

本文档说明如何将 `Signal-Linear-System-WuDazheng` 打包为 macOS 原生 `.app` 应用，并完成签名、公证与安装。

## 1. 前置条件

- macOS 14 或更高版本的构建机
- 已安装 Xcode Command Line Tools
- 已安装 Python 3.10+
- 项目根目录已存在 `venv` 虚拟环境，或允许脚本自动创建

可通过以下命令检查：

```bash
xcode-select -p
python3 --version
```

## 2. 本机构建

执行一键构建脚本：

```bash
bash scripts/macos/build_macos.sh
```

默认产物：

- `.app` 包：`dist/Signal-Linear-System-WuDazheng.app`
- Mach-O 可执行文件：`dist/Signal-Linear-System-WuDazheng.app/Contents/MacOS/Signal-Linear-System-WuDazheng`
- ZIP 包：`dist/Signal-Linear-System-WuDazheng.zip`

## 3. 指定目标架构

默认情况下，脚本会使用当前 Python 解释器对应的架构。

如果要显式指定：

```bash
TARGET_ARCH=arm64 bash scripts/macos/build_macos.sh
```

```bash
TARGET_ARCH=x86_64 bash scripts/macos/build_macos.sh
```

```bash
TARGET_ARCH=universal2 bash scripts/macos/build_macos.sh
```

注意：

- `universal2` 需要使用支持 `universal2` 的 Python 解释器与兼容依赖
- 当前仅有 `arm64` 解释器时，不能真实产出双架构通用二进制

## 4. 正式签名

若已安装 Apple Developer ID 证书，可通过环境变量传入签名身份：

```bash
export APPLE_DEVELOPER_IDENTITY="Developer ID Application: Your Name (TEAMID)"
bash scripts/macos/build_macos.sh
```

若未提供签名身份，构建脚本会执行 ad-hoc 签名，便于本地分发和验证，但不能替代正式发布签名。

## 5. Apple 公证

先将 notarytool 凭据保存到本机：

```bash
xcrun notarytool store-credentials SIGNAL_WUDAZHENG_PROFILE \
  --apple-id "your-apple-id@example.com" \
  --team-id "TEAMID" \
  --password "app-specific-password"
```

然后执行：

```bash
export APPLE_KEYCHAIN_PROFILE=SIGNAL_WUDAZHENG_PROFILE
bash scripts/macos/notarize_macos.sh
```

## 6. 安装方式

### 方式一：直接运行 `.app`

双击：

```text
dist/Signal-Linear-System-WuDazheng.app
```

应用会自动启动内置 `Streamlit` 服务，并在默认浏览器中打开页面。

### 方式二：终端内运行 Mach-O

```bash
./dist/Signal-Linear-System-WuDazheng.app/Contents/MacOS/Signal-Linear-System-WuDazheng
```

## 7. Gatekeeper 与首次打开

如果是 ad-hoc 签名或未公证版本，首次打开时可能被 Gatekeeper 拦截。可以在“系统设置 -> 隐私与安全性”中允许打开，或使用右键“打开”。

## 8. 发布建议

- 正式发布前，务必使用 Developer ID 证书签名
- 将公证后的 `.app` 或 `.zip` 作为最终发布产物
- 在 Intel 与 Apple Silicon 机器上分别做一次实际运行验证
