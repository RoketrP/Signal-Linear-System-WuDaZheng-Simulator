# macOS 打包测试报告

## 项目

- 项目名称：`Signal-Linear-System-WuDazheng`
- 打包形式：macOS `.app` 包 + 内部 Mach-O 可执行文件
- 启动方式：原生启动器启动本地 `Streamlit` 服务并自动打开浏览器

## 本次构建环境

- 构建日期：2026-04-17
- 构建系统：macOS 14.8
- CPU 架构：`arm64`
- Python 平台：`macosx-11.1-arm64`
- Xcode Command Line Tools：已安装

## 已执行的实际验证

### 1. 构建环境检查

- `xcode-select -p`：通过
- `python3 --version`：通过
- 虚拟环境依赖安装：通过

### 2. 应用依赖验证

- `streamlit` 导入：通过
- `numpy` 导入：通过
- `scipy` 导入：通过
- `matplotlib` 导入：通过
- `plotly` 导入：通过

### 3. 源码级验证

- `python -m py_compile app.py`：通过
- 原有 `Streamlit` 应用启动验证：通过

### 4. 原生打包流水线验证

- 已生成 macOS 启动器脚本：通过
- 已生成 `.app` 构建脚本：通过
- 已生成签名与公证脚本：通过
- 已生成 `Info.plist`：通过
- 已成功构建 `dist/Signal-Linear-System-WuDazheng.app`：通过
- 已成功生成 `dist/Signal-Linear-System-WuDazheng.zip`：通过
- Mach-O 类型检查：`Mach-O 64-bit executable arm64`
- `codesign --verify --deep --strict`：通过
- 打包后 Mach-O 启动验证：通过
- 打包后应用访问地址：`http://127.0.0.1:8503`

## 实际产物

- `.app` 包：`dist/Signal-Linear-System-WuDazheng.app`
- Mach-O 可执行文件：`dist/Signal-Linear-System-WuDazheng.app/Contents/MacOS/Signal-Linear-System-WuDazheng`
- ZIP 安装包：`dist/Signal-Linear-System-WuDazheng.zip`
- 当前签名类型：`ad-hoc`
- 当前实际产物架构：`arm64`

## 当前环境可确认的限制

- 当前构建机使用的是 `arm64` 专用 Python，而不是 `universal2` 解释器
- 因此当前环境下无法真实产出 `x86_64 + arm64` 双架构通用二进制
- 当前会话未提供 Apple Developer ID 证书，无法完成正式 Developer ID 签名
- 当前会话未提供 Apple 公证凭据，无法完成 `notarytool` 提交与票据 stapling
- 当前仅能在本机构建环境完成实际运行验证，无法真实覆盖 macOS 10.15、11、12、13、14 全版本矩阵

## 兼容性测试矩阵

下表区分“已验证”和“待外部验证”：

| 测试项 | 10.15 | 11 | 12 | 13 | 14 | Apple Silicon |
|---|---|---|---|---|---|---|
| `.app` 启动 | 待验证 | 待验证 | 待验证 | 待验证 | 本机构建环境 | 本机构建环境 |
| 浏览器自动打开 | 待验证 | 待验证 | 待验证 | 待验证 | 本机构建环境 | 本机构建环境 |
| 章节切换与绘图 | 待验证 | 待验证 | 待验证 | 待验证 | 待验证 | 待验证 |
| 依赖完整性 | 待验证 | 待验证 | 待验证 | 待验证 | 待验证 | 待验证 |
| 崩溃检查 | 待验证 | 待验证 | 待验证 | 待验证 | 待验证 | 待验证 |
| 性能检查 | 待验证 | 待验证 | 待验证 | 待验证 | 待验证 | 待验证 |

## 建议发布前补测项

1. 在 Intel Mac 上执行 `TARGET_ARCH=x86_64 bash scripts/macos/build_macos.sh`
2. 使用 `universal2` Python 执行 `TARGET_ARCH=universal2 bash scripts/macos/build_macos.sh`
3. 使用正式 Developer ID 证书完成签名
4. 使用 `notarytool` 完成 Apple 公证并对 `.app` 执行 `stapler staple`
5. 在 macOS 10.15、11、12、13、14 各版本至少做一次启动与章节切换验证
6. 执行长时间滑块拖动、图像下载、状态空间和卷积动画等性能验证

## 结论

当前仓库已经具备：

- macOS 原生 `.app` 打包脚本
- Mach-O 可执行文件生成路径
- `Info.plist` 元数据
- 签名与公证自动化入口
- 安装与测试文档

但以下项目仍依赖外部正式发布条件，不能在当前会话中伪造完成：

- Developer ID 正式签名
- Apple 公证通过
- 真实 `universal2` 双架构产物
- macOS 10.15 至 14 的全版本实机兼容性验证
