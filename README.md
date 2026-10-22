# Signal-Linear-System-WuDazheng

基于 `Python + Streamlit + NumPy + SciPy + Matplotlib + Plotly` 开发的《信号与线性系统》交互式网页可视化学习工具，按吴大正《信号与线性系统》第五版目录顺序组织，面向电气、自动化、通信专业本科生、期末复习与考研专业课备考。

## 项目特点

- 严格按教材章节顺序组织侧边栏导航
- 每个小节提供独立参数面板与实时波形/频谱/零极点可视化
- 支持快速示例、参数重置、图像下载
- 仿真计算在本地完成，默认关闭 Streamlit 使用统计；首次安装依赖需要联网
- 单文件主程序，结构清晰，便于二次开发和教学演示

## 教材适配说明

当前版本覆盖以下 8 章全部目录型小节：

1. 第一章 信号与系统
2. 第二章 连续系统的时域分析
3. 第三章 离散系统的时域分析
4. 第四章 傅里叶变换和系统的频域分析
5. 第五章 连续系统的 s 域分析
6. 第六章 离散系统的 z 域分析
7. 第七章 系统函数
8. 第八章 系统的状态变量分析

## 本地一键运行

### 推荐方式：先创建虚拟环境

建议先在项目根目录创建独立 Python 虚拟环境，避免依赖安装到系统全局环境。

#### 1. 确认 Python 版本

本项目要求 `Python 3.10+` 和 `Streamlit 1.51+`。

```bash
python3 --version
```

如果你的系统中 `python` 命令已经指向 Python 3，也可以使用：

```bash
python --version
```

#### 2. 进入项目根目录

```bash
cd /path/to/Signal-Linear-System-WuDaZheng-Simulator
```

#### 3. 创建名为 `venv` 的虚拟环境

```bash
python3 -m venv venv
```

如果你的环境中 `python` 已绑定到 Python 3，也可以执行：

```bash
python -m venv venv
```

#### 4. 激活虚拟环境

macOS / Linux:

```bash
source venv/bin/activate
```

Windows `cmd`:

```bat
venv\Scripts\activate
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

激活成功后，命令行前通常会出现类似 `(venv)` 的前缀。

#### 5. 验证当前 Python 与 pip 已指向虚拟环境

```bash
python --version
which python
pip --version
```

Windows 中用 `where.exe python` 代替 `which python`。

如果输出路径位于项目目录下的 `venv/` 中，说明当前依赖会安装到虚拟环境，而不是系统全局环境。

#### 6. 安装项目依赖

```bash
pip install -r requirements.txt
```

#### 7. 启动项目

```bash
python -m streamlit run app.py
```

#### 8. 退出虚拟环境

```bash
deactivate
```

### 方式一：命令行启动

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

启动后浏览器会自动打开本地网页；如未自动打开，可手动访问命令行显示的本地地址。

### 方式二：Windows 双击启动

直接双击项目根目录下的 `一键启动.bat`。脚本会创建并复用项目内的 `.venv`，通过该环境安装依赖和启动应用，不会把依赖安装到系统全局环境。安装失败会停止并显示错误。

默认仅监听 `127.0.0.1`。如默认端口被占用，可指定其他端口：

```bash
python -m streamlit run app.py --server.port 8502
```

## 计算约定与验证

- 连续傅里叶变换使用 `X(f)=∫x(t)e^(-j2πft)dt`，FFT 包含采样间隔和时间原点修正；DFT 小节仍使用离散定义。
- 连续卷积保留完整长度，输出时间轴从两路输入起点之和开始；有限采样窗会产生截断和数值积分误差。
- 逆拉普拉斯与逆 Z 变换支持重复极点，展示部分分式中分母的幂次。
- 方波功率谱展示双边复指数系数的平方；三角波原信号与正弦级数采用相同相位。
- 能量与平均功率数值标明有限观测窗，避免与全时域理论值混淆。

运行回归检查（无需额外测试依赖）：

```bash
python -m unittest discover -s tests -v
```

测试覆盖解析解、全部 43 个小节、快速示例、参数重置、所有下拉分支、采样临界频率及 PNG 文件可读性。GitHub Actions 在 Windows / Linux 和 Python 3.10 / 3.13 上自动执行。详细修复依据见 [维护说明](docs/maintenance-2026-10.md)。

PNG 导出优先使用系统已安装的微软雅黑、苹方、Noto CJK 等中文字体。若 Linux 导出图中文字缺失，请安装中文字体（例如 `fonts-noto-cjk`）后重启应用。

## macOS 原生打包

项目已补充 macOS 原生 `.app` 打包流水线，适合本地分发、课程演示和后续正式签名发布。

### 一键构建

```bash
bash scripts/macos/build_macos.sh
```

默认生成：

- `dist/Signal-Linear-System-WuDazheng.app`
- `dist/Signal-Linear-System-WuDazheng.app/Contents/MacOS/Signal-Linear-System-WuDazheng`
- `dist/Signal-Linear-System-WuDazheng.zip`

### 签名与公证

- 若设置 `APPLE_DEVELOPER_IDENTITY`，构建脚本会执行正式签名
- 若未设置证书，则会执行 `ad-hoc` 签名，便于本机测试
- 公证脚本：

```bash
bash scripts/macos/notarize_macos.sh
```

### 相关文档

- 构建与安装说明：`docs/macos-build-and-install.md`
- 测试报告：`docs/macos-test-report.md`

## 项目结构

```text
Signal-Linear-System-WuDaZheng-Simulator/
├── app.py
├── macos_launcher.py
├── requirements-macos-build.txt
├── requirements.txt
├── README.md
├── docs/
├── packaging/
├── scripts/
├── tests/
├── .github/workflows/tests.yml
├── .streamlit/config.toml
├── .gitignore
└── 一键启动.bat
```

## 功能截图占位

后续可在此处补充 GitHub 展示截图：

- [ ] 首页与侧边栏导航截图
- [ ] 卷积积分动态演示截图
- [ ] 傅里叶频谱分析截图
- [ ] s 域 / z 域零极点图截图
- [ ] 状态变量分析截图

## 开发与扩展建议

- 可继续补充更细颗粒度教材公式推导过程
- 可新增课堂例题模式与题后练习模式
- 可为每章增加知识点总览页与考点速记页
- 可按课程实验需求补充更复杂电路或通信系统场景

## 开源免责声明

- 本项目仅用于《信号与线性系统》课程学习、教学演示与可视化理解
- 项目内容不能替代教材原文、教师课堂讲授与正式实验指导
- 教材名称与章节组织仅用于学习适配说明，不构成教材内容数字化转载
- 使用者应自行核对公式、符号与课程要求后再用于作业、实验或教学展示

## License

本仓库保留原有 `LICENSE` 文件，请结合仓库许可证使用。

## 开发贡献人员
- Shiyu Chen
- Xinqi Guo
