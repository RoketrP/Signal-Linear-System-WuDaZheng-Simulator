import io
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from scipy.integrate import cumulative_trapezoid

st.set_page_config(
    page_title="Signal-Linear-System-WuDazheng",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stMetric {background: #f7f9fc; padding: 10px; border-radius: 8px;}
    .main .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

BOOK_STRUCTURE = OrderedDict(
    {
        "第一章 信号与系统": [
            "1. 连续/离散信号区分演示",
            "2. 周期/非周期、能量/功率信号判定仿真",
            "3. 信号基础运算：加减乘、反转、时移、尺度变换",
            "4. 阶跃函数、冲激函数定义、广义函数、微积分性质可视化",
            "5. 系统线性、时不变、因果、稳定四大特性交互验证",
        ],
        "第二章 连续系统的时域分析": [
            "1. LTI连续系统微分方程经典解演示",
            "2. 零输入/零状态、全响应拆分对比",
            "3. 单位冲激响应、单位阶跃响应波形生成",
            "4. 卷积积分动态分步演示",
            "5. 卷积全部代数性质、微分积分性质仿真验证",
        ],
        "第三章 离散系统的时域分析": [
            "1. LTI离散系统、差分方程求解",
            "2. 离散系统零输入、零状态响应",
            "3. 单位序列、单位阶跃序列响应",
            "4. 离散卷积和完整可视化",
            "5. 反卷积简易演示",
        ],
        "第四章 傅里叶变换和系统的频域分析": [
            "1. 信号正交函数分解原理",
            "2. 周期信号傅里叶级数、奇偶函数频谱",
            "3. 周期/非周期信号幅度谱、相位谱绘制",
            "4. 傅里叶变换全部核心性质逐一交互验证",
            "5. 能量谱、功率谱、周期信号傅里叶变换",
            "6. LTI系统频率响应、频域完整分析",
            "7. 无失真传输、理想低通滤波器效果仿真",
            "8. 奈奎斯特取样定理、时域/频域取样全过程演示",
            "9. 离散时间傅里叶变换DTFT、DFT可视化",
        ],
        "第五章 连续系统的s域分析": [
            "1. 单边/双边拉普拉斯变换、收敛域ROC",
            "2. 拉普拉斯变换全套性质逐一验证",
            "3. 拉普拉斯逆变换（查表法+部分分式展开）",
            "4. s域复频域微分方程求解",
            "5. 系统函数H(s)、s域框图、电路s域建模",
            "6. 拉普拉斯与傅里叶变换关系对照",
        ],
        "第六章 离散系统的z域分析": [
            "1. Z变换定义、收敛域",
            "2. Z变换全套8大性质交互演示",
            "3. 逆Z变换：幂级数展开、部分分式法",
            "4. 差分方程z域完整求解",
            "5. 离散系统函数H(z)、z域框图",
            "6. s域与z域映射关系",
        ],
        "第七章 系统函数": [
            "1. 系统零极点分布、时域/频域响应对应关系",
            "2. 系统因果性、稳定性判定仿真",
            "3. 信号流图、梅森公式可视化演示",
            "4. 系统级联、并联、反馈三种结构仿真",
        ],
        "第八章 系统的状态变量分析": [
            "1. 状态方程、输出方程建立",
            "2. 连续/离散系统状态方程求解",
            "3. 系统可控性、可观测性判定演示",
        ],
    }
)

SECTION_INFO = {
    "1. 连续/离散信号区分演示": {
        "formula": r"x(t),\ x[n]\\ \text{连续时间信号与离散时间信号的自变量集合不同}",
        "summary": "同一物理信号经过连续观测与离散采样后，表达方式不同，但可保持主导频率与幅值特征。",
        "tips": "易错点：先分清时间轴定义域，再讨论波形本身。",
    },
    "2. 周期/非周期、能量/功率信号判定仿真": {
        "formula": r"E=\int_{-\infty}^{\infty}|x(t)|^2dt,\quad P=\lim_{T\to\infty}\frac{1}{2T}\int_{-T}^{T}|x(t)|^2dt",
        "summary": "有限持续信号通常是能量信号，周期信号通常是功率信号。",
        "tips": "易错点：周期信号通常能量发散，但功率有限。",
    },
    "3. 信号基础运算：加减乘、反转、时移、尺度变换": {
        "formula": r"y(t)=a\,x(bt-t_0),\quad x(-t),\ x(t-t_0)",
        "summary": "时移决定出现时刻，尺度变换决定伸缩，反转体现镜像。",
        "tips": "易错点：先时移后尺度与先尺度后时移不等价。",
    },
    "4. 阶跃函数、冲激函数定义、广义函数、微积分性质可视化": {
        "formula": r"u(t)=\int_{-\infty}^{t}\delta(\tau)d\tau,\quad \frac{d}{dt}u(t)=\delta(t)",
        "summary": "阶跃与冲激在广义函数意义下互为积分和导数。",
        "tips": "易错点：冲激函数通过抽样性质参与积分，而非普通函数极限。",
    },
    "5. 系统线性、时不变、因果、稳定四大特性交互验证": {
        "formula": r"T\{a x_1+b x_2\}=aT\{x_1\}+bT\{x_2\},\quad T\{x(t-t_0)\}=y(t-t_0)",
        "summary": "系统特性是卷积、频域和系统函数分析的前提。",
        "tips": "易错点：时不变不代表因果，线性也不代表稳定。",
    },
    "1. LTI连续系统微分方程经典解演示": {
        "formula": r"\sum_{k=0}^{N}a_k\frac{d^ky}{dt^k}=\sum_{m=0}^{M}b_m\frac{d^m x}{dt^m}",
        "summary": "常系数线性微分方程刻画连续 LTI 系统的动态特性。",
        "tips": "考点：欠阻尼、临界阻尼、过阻尼三种二阶典型响应。",
    },
    "2. 零输入/零状态、全响应拆分对比": {
        "formula": r"y(t)=y_{zi}(t)+y_{zs}(t)",
        "summary": "零输入响应由初始储能决定，零状态响应由外部输入决定。",
        "tips": "易错点：零状态不是无输入响应，而是初值清零后的输入响应。",
    },
    "3. 单位冲激响应、单位阶跃响应波形生成": {
        "formula": r"h(t)=T\{\delta(t)\},\quad s(t)=T\{u(t)\}=\int_{-\infty}^{t}h(\tau)d\tau",
        "summary": "冲激响应完全决定 LTI 系统，阶跃响应可由冲激响应积分得到。",
        "tips": "考点：由阶跃响应求冲激响应要注意广义函数项。",
    },
    "4. 卷积积分动态分步演示": {
        "formula": r"y(t)=\int_{-\infty}^{\infty}x(\tau)h(t-\tau)d\tau",
        "summary": "卷积本质是翻折、平移、相乘、积分。",
        "tips": "易错点：先翻折，再平移，再求重叠面积。",
    },
    "5. 卷积全部代数性质、微分积分性质仿真验证": {
        "formula": r"x*h=h*x,\ (x*h)*g=x*(h*g),\ \frac{d}{dt}(x*h)=\frac{dx}{dt}*h",
        "summary": "卷积的代数性质是系统分析的高频考点。",
        "tips": "易错点：卷积结果的定义区间要跟支撑区间一起看。",
    },
    "1. LTI离散系统、差分方程求解": {
        "formula": r"\sum_{k=0}^{N}a_k y[n-k]=\sum_{m=0}^{M}b_m x[n-m]",
        "summary": "离散 LTI 系统常由差分方程描述。",
        "tips": "易错点：递推时要严格处理索引和初值。",
    },
    "2. 离散系统零输入、零状态响应": {
        "formula": r"y[n]=y_{zi}[n]+y_{zs}[n]",
        "summary": "离散系统同样满足零输入与零状态分解。",
        "tips": "考点：首项与系统极点决定自然响应趋势。",
    },
    "3. 单位序列、单位阶跃序列响应": {
        "formula": r"h[n]=T\{\delta[n]\},\quad s[n]=T\{u[n]\}",
        "summary": "离散冲激响应是离散卷积的核心。",
        "tips": "易错点：单位冲激只有 n=0 取 1。",
    },
    "4. 离散卷积和完整可视化": {
        "formula": r"y[n]=\sum_{k=-\infty}^{\infty}x[k]h[n-k]",
        "summary": "离散卷积是逐点移位乘积求和。",
        "tips": "易错点：求和上下限由两序列实际非零区间决定。",
    },
    "5. 反卷积简易演示": {
        "formula": r"Y(z)=X(z)H(z)\Rightarrow X(z)=Y(z)/H(z)",
        "summary": "反卷积可看作多项式除法或 z 域相除。",
        "tips": "易错点：系统零点会影响可逆性。",
    },
    "1. 信号正交函数分解原理": {
        "formula": r"x(t)\approx \sum_k c_k\phi_k(t),\quad c_k=\frac{\langle x,\phi_k\rangle}{\langle\phi_k,\phi_k\rangle}",
        "summary": "正交分解是傅里叶级数和傅里叶变换的出发点。",
        "tips": "考点：投影系数的内积意义。",
    },
    "2. 周期信号傅里叶级数、奇偶函数频谱": {
        "formula": r"x(t)=a_0+\sum_{k=1}^{\infty}[a_k\cos(k\omega_0 t)+b_k\sin(k\omega_0 t)]",
        "summary": "奇函数只含正弦项，偶函数只含余弦项。",
        "tips": "易错点：先判对称性，可大幅简化系数计算。",
    },
    "3. 周期/非周期信号幅度谱、相位谱绘制": {
        "formula": r"X(j\omega)=\int_{-\infty}^{\infty}x(t)e^{-j\omega t}dt",
        "summary": "周期信号对应离散谱，非周期信号对应连续谱。",
        "tips": "易错点：幅度谱与相位谱必须配合使用。",
    },
    "4. 傅里叶变换全部核心性质逐一交互验证": {
        "formula": r"x(t-t_0)\leftrightarrow X(j\omega)e^{-j\omega t_0},\quad e^{j\omega_0 t}x(t)\leftrightarrow X[j(\omega-\omega_0)]",
        "summary": "时移、频移、尺度、微分等性质构成频域主线。",
        "tips": "易错点：注意指数因子的符号方向。",
    },
    "5. 能量谱、功率谱、周期信号傅里叶变换": {
        "formula": r"\Psi_x(\omega)=|X(j\omega)|^2,\quad P_x(\omega)=\sum_k |C_k|^2\delta(\omega-k\omega_0)",
        "summary": "能量谱适用于能量信号，功率谱适用于功率信号。",
        "tips": "易错点：周期信号更适合用傅里叶级数系数描述功率谱。",
    },
    "6. LTI系统频率响应、频域完整分析": {
        "formula": r"H(j\omega)=Y(j\omega)/X(j\omega)",
        "summary": "频率响应反映系统对各频率分量的幅度与相位作用。",
        "tips": "考点：截止频率与带宽解释。",
    },
    "7. 无失真传输、理想低通滤波器效果仿真": {
        "formula": r"H(j\omega)=Ke^{-j\omega t_d}",
        "summary": "无失真传输要求幅频恒定、相频线性。",
        "tips": "易错点：只有幅度不失真并不够。",
    },
    "8. 奈奎斯特取样定理、时域/频域取样全过程演示": {
        "formula": r"f_s\ge 2f_{\max}",
        "summary": "采样在时域表现为抽取，在频域表现为谱复制。",
        "tips": "考点：低于奈奎斯特频率会发生混叠。",
    },
    "9. 离散时间傅里叶变换DTFT、DFT可视化": {
        "formula": r"X(e^{j\omega})=\sum x[n]e^{-j\omega n},\quad X[k]=\sum x[n]e^{-j2\pi kn/N}",
        "summary": "DTFT 是连续周期谱，DFT 是有限点采样谱。",
        "tips": "易错点：DFT 是 DTFT 的有限采样，不是全部频域信息。",
    },
}


def time_axis(duration=6.0, samples=1200):
    return np.linspace(-duration / 2.0, duration / 2.0, samples)


def heaviside(x):
    return (x >= 0).astype(float)


def rect(x, width=1.0):
    return (np.abs(x) <= width / 2.0).astype(float)


def triangle(x, width=2.0):
    return np.maximum(1.0 - np.abs(x) / (width / 2.0), 0.0)


def continuous_signal(kind, t, amp=1.0, freq=1.0, shift=0.0, scale=1.0, decay=0.6):
    tau = scale * (t - shift)
    if kind == "正弦":
        return amp * np.sin(2 * np.pi * freq * tau)
    if kind == "余弦":
        return amp * np.cos(2 * np.pi * freq * tau)
    if kind == "指数衰减":
        return amp * np.exp(-decay * np.maximum(tau, 0)) * heaviside(tau)
    if kind == "矩形脉冲":
        return amp * rect(tau, width=1.2)
    if kind == "三角脉冲":
        return amp * triangle(tau, width=2.0)
    if kind == "单位阶跃":
        return amp * heaviside(tau)
    return amp * np.sinc(freq * tau)


def energy_power_continuous(x, t):
    dt = t[1] - t[0]
    energy = np.sum(np.abs(x) ** 2) * dt
    power = energy / (t[-1] - t[0] + dt)
    return energy, power


def fft_spectrum(x, dt):
    n = len(x)
    f = fftshift(fftfreq(n, d=dt))
    X = fftshift(fft(x))
    return f, X


def make_png_bytes(x, traces, title, xlabel, ylabel, discrete=False):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    for item in traces:
        if discrete:
            markerline, stemlines, _ = ax.stem(x, item["y"], linefmt="-", markerfmt="o", basefmt="k-", label=item["name"])
            plt.setp(stemlines, linewidth=1.3)
            plt.setp(markerline, markersize=4.5)
        else:
            ax.plot(x, item["y"], linewidth=2, label=item["name"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if len(traces) > 1:
        ax.legend()
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_chart(title, x, traces, key, xlabel="自变量", ylabel="幅值", discrete=False):
    fig = go.Figure()
    for item in traces:
        if discrete:
            fig.add_trace(go.Scatter(x=x, y=item["y"], mode="lines+markers", name=item["name"], line={"shape": "hv"}))
        else:
            fig.add_trace(go.Scatter(x=x, y=item["y"], mode="lines", name=item["name"]))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, height=420, margin=dict(l=40, r=20, t=60, b=40), legend=dict(orientation="h"))
    fig.update_xaxes(showgrid=True, zeroline=True)
    fig.update_yaxes(showgrid=True, zeroline=True)
    st.plotly_chart(fig, use_container_width=True, key=f"plot_{key}")
    st.download_button(
        "下载当前图像 PNG",
        data=make_png_bytes(x, traces, title, xlabel, ylabel, discrete=discrete),
        file_name=f"{key}.png",
        mime="image/png",
        key=f"download_{key}",
    )


def render_pole_zero_chart(zeros, poles, title, key, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), unit_circle=False):
    fig = go.Figure()
    if unit_circle:
        theta = np.linspace(0, 2 * np.pi, 400)
        fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines", name="单位圆", line=dict(dash="dash")))
    fig.add_hline(y=0, line_color="gray")
    fig.add_vline(x=0, line_color="gray")
    if len(zeros):
        fig.add_trace(go.Scatter(x=np.real(zeros), y=np.imag(zeros), mode="markers", marker_symbol="circle-open", marker_size=13, marker_line_width=2, name="零点"))
    if len(poles):
        fig.add_trace(go.Scatter(x=np.real(poles), y=np.imag(poles), mode="markers", marker_symbol="x", marker_size=13, marker_line_width=3, name="极点"))
    fig.update_layout(title=title, xaxis_title="实部", yaxis_title="虚部", height=400, margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(range=list(xlim), scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=list(ylim))
    st.plotly_chart(fig, use_container_width=True, key=f"pz_{key}")


def control_buttons(prefix, defaults, example):
    c1, c2 = st.columns(2)
    if c1.button("快速示例", key=f"{prefix}_example_btn", use_container_width=True):
        for name, value in example.items():
            st.session_state[f"{prefix}_{name}"] = value
        st.rerun()
    if c2.button("重置参数", key=f"{prefix}_reset_btn", use_container_width=True):
        for name, value in defaults.items():
            st.session_state[f"{prefix}_{name}"] = value
        st.rerun()


def slider_param(prefix, name, label, min_value, max_value, value, step):
    key = f"{prefix}_{name}"
    if key not in st.session_state:
        st.session_state[key] = value
    return st.slider(label, min_value, max_value, st.session_state[key], step=step, key=key)


def select_param(prefix, name, label, options, index=0):
    key = f"{prefix}_{name}"
    if key not in st.session_state:
        st.session_state[key] = options[index]
    return st.selectbox(label, options, index=options.index(st.session_state[key]), key=key)


def note_block(title):
    info = SECTION_INFO.get(title, {})
    st.markdown("### 教材公式与考点")
    if info.get("formula"):
        st.latex(info["formula"])
    st.info(f"知识点说明：{info.get('summary', '本节聚焦教材核心概念的交互式理解。')}")
    st.warning(f"易错点提醒：{info.get('tips', '注意概念、符号与适用前提。')}")


def section_header(chapter, title):
    st.title("Signal-Linear-System-WuDazheng")
    st.caption("严格按吴大正《信号与线性系统》第五版章节顺序组织的交互式可视化学习工具")
    st.markdown(f"## {chapter}")
    st.markdown(f"### {title}")


def chapter_intro():
    st.sidebar.markdown("## 教材导航")
    chapter = st.sidebar.radio("选择章节", list(BOOK_STRUCTURE.keys()))
    section = st.sidebar.radio("选择小节", BOOK_STRUCTURE[chapter])
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "\n".join(
            [
                "### 使用说明",
                "- 左侧参数区可实时拖动",
                "- 每节支持快速示例与重置",
                "- 图像支持一键下载 PNG",
                "- 全部计算本地完成，无网络请求",
            ]
        )
    )
    return chapter, section


def render_ch1_sec1():
    prefix = "ch1s1"
    defaults = {"kind": "正弦", "amp": 1.0, "freq": 1.0, "fs": 12}
    example = {"kind": "矩形脉冲", "amp": 1.2, "freq": 1.0, "fs": 8}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        kind = select_param(prefix, "kind", "信号类型", ["正弦", "余弦", "矩形脉冲", "三角脉冲", "单位阶跃"], 0)
        amp = slider_param(prefix, "amp", "幅值 A", 0.2, 3.0, 1.0, 0.1)
        freq = slider_param(prefix, "freq", "频率/尺度参数", 0.5, 3.0, 1.0, 0.1)
        fs = slider_param(prefix, "fs", "离散采样频率 fs", 4, 40, 12, 1)
    with right:
        t = time_axis(6.0, 1200)
        n = np.arange(-18, 19)
        x_t = continuous_signal(kind, t, amp=amp, freq=freq)
        x_n = continuous_signal(kind, n / fs, amp=amp, freq=freq)
        tab1, tab2 = st.tabs(["连续时间信号", "离散时间序列"])
        with tab1:
            render_chart("连续时间信号 x(t)", t, [{"name": "x(t)", "y": x_t}], "ch1s1_cont", "t / s", "幅值")
        with tab2:
            render_chart("离散时间信号 x[n]", n, [{"name": "x[n]", "y": x_n}], "ch1s1_disc", "n", "幅值", discrete=True)
        st.success("结论：连续信号定义在连续时间轴上，离散信号只在整数索引点取值。")
    note_block("1. 连续/离散信号区分演示")


def render_ch1_sec2():
    prefix = "ch1s2"
    defaults = {"kind": "周期余弦", "amp": 1.0, "freq": 1.0, "decay": 0.6}
    example = {"kind": "指数衰减", "amp": 1.2, "freq": 1.0, "decay": 0.9}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        kind = select_param(prefix, "kind", "信号类别", ["周期余弦", "指数衰减", "矩形脉冲", "单边正弦门函数"], 0)
        amp = slider_param(prefix, "amp", "幅值 A", 0.2, 3.0, 1.0, 0.1)
        freq = slider_param(prefix, "freq", "频率 f", 0.5, 3.0, 1.0, 0.1)
        decay = slider_param(prefix, "decay", "衰减系数 a", 0.1, 1.5, 0.6, 0.1)
    with right:
        t = time_axis(12.0, 2400)
        if kind == "周期余弦":
            x = amp * np.cos(2 * np.pi * freq * t)
            periodicity = f"周期信号，基波周期 T0 = {1 / freq:.3f} s"
        elif kind == "指数衰减":
            x = amp * np.exp(-decay * np.maximum(t, 0)) * heaviside(t)
            periodicity = "非周期信号，有限能量"
        elif kind == "矩形脉冲":
            x = amp * rect(t, width=2.0)
            periodicity = "非周期时限信号，典型能量信号"
        else:
            x = amp * np.sin(2 * np.pi * freq * t) * rect(t, width=4.0)
            periodicity = "非周期时限正弦信号，能量有限"
        energy, power = energy_power_continuous(x, t)
        render_chart("信号波形与能量/功率判定", t, [{"name": "x(t)", "y": x}], "ch1s2_main", "t / s", "幅值")
        c1, c2, c3 = st.columns(3)
        c1.metric("数值近似能量 E", f"{energy:.4f}")
        c2.metric("数值近似平均功率 P", f"{power:.4f}")
        c3.metric("判定", periodicity)
    note_block("2. 周期/非周期、能量/功率信号判定仿真")


def render_ch1_sec3():
    prefix = "ch1s3"
    defaults = {"kind": "矩形脉冲", "shift": 1.0, "scale": 1.0, "gain": 1.2, "mode": "时移"}
    example = {"kind": "正弦", "shift": -0.8, "scale": 1.5, "gain": 0.8, "mode": "尺度变换"}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        kind = select_param(prefix, "kind", "原始信号", ["正弦", "余弦", "矩形脉冲", "三角脉冲"], 2)
        mode = select_param(prefix, "mode", "运算类型", ["加法对比", "乘法对比", "反转", "时移", "尺度变换"], 3)
        shift = slider_param(prefix, "shift", "时移参数 t0", -2.0, 2.0, 1.0, 0.1)
        scale = slider_param(prefix, "scale", "尺度参数 a", 0.5, 2.5, 1.0, 0.1)
        gain = slider_param(prefix, "gain", "增益系数", -2.0, 2.0, 1.2, 0.1)
    with right:
        t = time_axis(8.0, 1600)
        x = continuous_signal(kind, t, amp=1.0, freq=1.0)
        if mode == "加法对比":
            y = x + gain * continuous_signal("余弦", t, amp=1.0, freq=0.5)
            y_name = "x(t)+a cos(πt)"
        elif mode == "乘法对比":
            y = x * continuous_signal("余弦", t, amp=gain, freq=0.5)
            y_name = "x(t)·a cos(πt)"
        elif mode == "反转":
            y = continuous_signal(kind, -t, amp=1.0, freq=1.0)
            y_name = "x(-t)"
        elif mode == "时移":
            y = continuous_signal(kind, t, amp=1.0, freq=1.0, shift=shift)
            y_name = "x(t-t0)"
        else:
            y = continuous_signal(kind, t, amp=gain, freq=1.0, scale=scale)
            y_name = "a·x(bt)"
        render_chart("信号基础运算结果", t, [{"name": "原始 x(t)", "y": x}, {"name": y_name, "y": y}], "ch1s3_main", "t / s", "幅值")
    note_block("3. 信号基础运算：加减乘、反转、时移、尺度变换")


def render_ch1_sec4():
    prefix = "ch1s4"
    defaults = {"epsilon": 0.08, "width": 1.0, "mode": "冲激近似"}
    example = {"epsilon": 0.03, "width": 1.6, "mode": "微积分关系"}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        mode = select_param(prefix, "mode", "演示主题", ["冲激近似", "阶跃函数", "微积分关系"], 0)
        epsilon = slider_param(prefix, "epsilon", "冲激近似宽度 ε", 0.02, 0.20, 0.08, 0.01)
        width = slider_param(prefix, "width", "观察宽度", 0.5, 2.5, 1.0, 0.1)
    with right:
        t = time_axis(6.0, 1600)
        delta = np.exp(-(t / epsilon) ** 2) / (np.sqrt(np.pi) * epsilon)
        u = heaviside(t)
        du = np.gradient(u, t)
        if mode == "冲激近似":
            render_chart("单位冲激的窄脉冲近似", t, [{"name": "δ(t) 近似", "y": delta}], "ch1s4_delta", "t / s", "幅值")
            st.metric("近似积分面积", f"{np.trapz(delta, t):.4f}")
        elif mode == "阶跃函数":
            pulse = rect(t, width=width)
            render_chart("阶跃与矩形脉冲", t, [{"name": "u(t)", "y": u}, {"name": "rect(t)", "y": pulse}], "ch1s4_step", "t / s", "幅值")
        else:
            int_delta = cumulative_trapezoid(delta, t, initial=0.0)
            render_chart("u(t) 与 du/dt、∫δ(t)dt 对照", t, [{"name": "u(t)", "y": u}, {"name": "du/dt 数值近似", "y": du}, {"name": "∫δ(t)dt", "y": int_delta}], "ch1s4_relation", "t / s", "幅值")
    note_block("4. 阶跃函数、冲激函数定义、广义函数、微积分性质可视化")


def render_ch1_sec5():
    prefix = "ch1s5"
    systems = {
        "线性时不变因果稳定": {"func": lambda x, t: 0.6 * x + 0.4 * np.interp(t - 0.5, t, x, left=0.0, right=0.0), "labels": (True, True, True, True)},
        "平方律非线性系统": {"func": lambda x, t: x**2, "labels": (False, True, True, True)},
        "时变增益系统": {"func": lambda x, t: t * x, "labels": (True, False, True, False)},
        "超前非因果系统": {"func": lambda x, t: np.interp(t + 0.8, t, x, left=0.0, right=0.0), "labels": (True, True, False, True)},
    }
    defaults = {"system": "线性时不变因果稳定", "shift": 0.6}
    example = {"system": "平方律非线性系统", "shift": 0.8}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        system_name = select_param(prefix, "system", "选择待验证系统", list(systems.keys()), 0)
        shift = slider_param(prefix, "shift", "测试时移 t0", 0.2, 1.5, 0.6, 0.1)
    with right:
        t = time_axis(8.0, 1600)
        x1 = np.exp(-0.6 * np.maximum(t, 0)) * heaviside(t)
        x2 = np.sin(2 * np.pi * 0.7 * t) * rect(t, width=4.0)
        a, b = 1.3, -0.5
        system = systems[system_name]["func"]
        y_mix = system(a * x1 + b * x2, t)
        y_lin = a * system(x1, t) + b * system(x2, t)
        x_shift = np.interp(t - shift, t, x1, left=0.0, right=0.0)
        y_shift_input = system(x_shift, t)
        y_then_shift = np.interp(t - shift, t, system(x1, t), left=0.0, right=0.0)
        render_chart("系统输出对比", t, [{"name": "T{ax1+bx2}", "y": y_mix}, {"name": "aT{x1}+bT{x2}", "y": y_lin}], "ch1s5_main", "t / s", "幅值")
        c1, c2 = st.columns(2)
        c1.metric("线性验证误差", f"{np.max(np.abs(y_mix - y_lin)):.4e}")
        c2.metric("时不变验证误差", f"{np.max(np.abs(y_shift_input - y_then_shift)):.4e}")
        labels = systems[system_name]["labels"]
        st.write(f"教材判定：线性={'是' if labels[0] else '否'}，时不变={'是' if labels[1] else '否'}，因果={'是' if labels[2] else '否'}，稳定={'是' if labels[3] else '否'}。")
    note_block("5. 系统线性、时不变、因果、稳定四大特性交互验证")


def render_ch2_sec1():
    prefix = "ch2s1"
    defaults = {"wn": 4.0, "zeta": 0.4, "input": "单位阶跃", "freq": 1.0}
    example = {"wn": 5.0, "zeta": 0.15, "input": "正弦激励", "freq": 1.2}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        wn = slider_param(prefix, "wn", "固有角频率 ωn", 1.0, 10.0, 4.0, 0.1)
        zeta = slider_param(prefix, "zeta", "阻尼比 ζ", 0.0, 2.0, 0.4, 0.05)
        input_kind = select_param(prefix, "input", "输入类型", ["单位阶跃", "正弦激励", "矩形脉冲"], 0)
        freq = slider_param(prefix, "freq", "输入频率 f", 0.2, 3.0, 1.0, 0.1)
    with right:
        t = np.linspace(0, 8, 1600)
        sys = signal.TransferFunction([wn**2], [1.0, 2 * zeta * wn, wn**2])
        if input_kind == "单位阶跃":
            u = np.ones_like(t)
        elif input_kind == "正弦激励":
            u = np.sin(2 * np.pi * freq * t)
        else:
            u = (t <= 1.5).astype(float)
        _, y, _ = signal.lsim(sys, U=u, T=t)
        render_chart("连续 LTI 系统微分方程响应", t, [{"name": "输入 x(t)", "y": u}, {"name": "输出 y(t)", "y": y}], "ch2s1_main", "t / s", "幅值")
    note_block("1. LTI连续系统微分方程经典解演示")


def render_ch2_sec2():
    prefix = "ch2s2"
    defaults = {"a": 1.2, "b": 1.0, "y0": 1.5}
    example = {"a": 0.8, "b": 1.4, "y0": 2.0}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        a = slider_param(prefix, "a", "系统参数 a", 0.2, 3.0, 1.2, 0.1)
        b = slider_param(prefix, "b", "输入系数 b", 0.2, 3.0, 1.0, 0.1)
        y0 = slider_param(prefix, "y0", "初始条件 y(0-)", -3.0, 3.0, 1.5, 0.1)
    with right:
        t = np.linspace(0, 8, 1500)
        u = heaviside(t)
        y_zi = y0 * np.exp(-a * t)
        sys = signal.TransferFunction([b], [1.0, a])
        _, y_zs, _ = signal.lsim(sys, U=u, T=t)
        y = y_zi + y_zs
        render_chart("零输入、零状态与全响应", t, [{"name": "零输入响应", "y": y_zi}, {"name": "零状态响应", "y": y_zs}, {"name": "全响应", "y": y}], "ch2s2_main", "t / s", "幅值")
    note_block("2. 零输入/零状态、全响应拆分对比")


def render_ch2_sec3():
    prefix = "ch2s3"
    defaults = {"wn": 3.0, "zeta": 0.5}
    example = {"wn": 5.0, "zeta": 0.2}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        wn = slider_param(prefix, "wn", "固有角频率 ωn", 1.0, 8.0, 3.0, 0.1)
        zeta = slider_param(prefix, "zeta", "阻尼比 ζ", 0.0, 2.0, 0.5, 0.05)
    with right:
        t = np.linspace(0, 8, 1500)
        sys = signal.TransferFunction([wn**2], [1.0, 2 * zeta * wn, wn**2])
        _, h = signal.impulse(sys, T=t)
        _, s = signal.step(sys, T=t)
        render_chart("单位冲激响应与单位阶跃响应", t, [{"name": "h(t)", "y": h}, {"name": "s(t)", "y": s}], "ch2s3_main", "t / s", "幅值")
    note_block("3. 单位冲激响应、单位阶跃响应波形生成")


def render_ch2_sec4():
    prefix = "ch2s4"
    defaults = {"t0": 1.0, "alpha": 1.0}
    example = {"t0": 2.0, "alpha": 1.8}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        t0 = slider_param(prefix, "t0", "当前输出时刻 t", -1.0, 4.0, 1.0, 0.1)
        alpha = slider_param(prefix, "alpha", "系统衰减系数 α", 0.4, 2.5, 1.0, 0.1)
    with right:
        tau = np.linspace(-2, 5, 1600)
        x_tau = rect(tau - 0.5, width=1.5)
        h_tau = np.exp(-alpha * np.maximum(t0 - tau, 0)) * heaviside(t0 - tau)
        product = x_tau * h_tau
        y_all = np.convolve(rect(tau - 0.5, width=1.5), np.exp(-alpha * np.maximum(tau, 0)) * heaviside(tau), mode="same") * (tau[1] - tau[0])
        fig = make_subplots(rows=2, cols=1, subplot_titles=("翻折-平移-相乘", "卷积结果 y(t)"))
        fig.add_trace(go.Scatter(x=tau, y=x_tau, mode="lines", name="x(τ)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=tau, y=h_tau, mode="lines", name="h(t-τ)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=tau, y=product, mode="lines", name="乘积"), row=1, col=1)
        fig.add_trace(go.Scatter(x=tau, y=y_all, mode="lines", name="y(t)"), row=2, col=1)
        fig.add_trace(go.Scatter(x=[t0], y=[np.trapz(product, tau)], mode="markers", name="当前 y(t0)"), row=2, col=1)
        fig.update_layout(height=680, margin=dict(l=40, r=20, t=80, b=30))
        st.plotly_chart(fig, use_container_width=True, key="ch2s4_plot")
        st.metric("当前卷积积分值 y(t0)", f"{np.trapz(product, tau):.4f}")
    note_block("4. 卷积积分动态分步演示")


def render_ch2_sec5():
    prefix = "ch2s5"
    defaults = {"alpha": 1.0}
    example = {"alpha": 1.8}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        alpha = slider_param(prefix, "alpha", "指数衰减参数 α", 0.5, 2.5, 1.0, 0.1)
    with right:
        t = np.linspace(0, 8, 1200)
        dt = t[1] - t[0]
        x = np.exp(-alpha * t)
        h = rect(t - 1.0, width=1.4)
        g = heaviside(t) * np.exp(-0.5 * t)
        conv1 = np.convolve(x, h, mode="same") * dt
        conv2 = np.convolve(h, x, mode="same") * dt
        assoc1 = np.convolve(np.convolve(x, h, mode="same") * dt, g, mode="same") * dt
        assoc2 = np.convolve(x, np.convolve(h, g, mode="same") * dt, mode="same") * dt
        deriv_conv = np.gradient(conv1, dt)
        conv_deriv = np.convolve(np.gradient(x, dt), h, mode="same") * dt
        render_chart("卷积性质数值验证", t, [{"name": "x*h", "y": conv1}, {"name": "h*x", "y": conv2}], "ch2s5_main", "t / s", "幅值")
        c1, c2, c3 = st.columns(3)
        c1.metric("交换律误差", f"{np.max(np.abs(conv1 - conv2)):.3e}")
        c2.metric("结合律误差", f"{np.max(np.abs(assoc1 - assoc2)):.3e}")
        c3.metric("微分性质误差", f"{np.max(np.abs(deriv_conv - conv_deriv)):.3e}")
    note_block("5. 卷积全部代数性质、微分积分性质仿真验证")


def render_ch3_sec1():
    prefix = "ch3s1"
    defaults = {"a1": -0.6, "b0": 1.0, "b1": 0.4}
    example = {"a1": -0.3, "b0": 1.0, "b1": 0.8}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        a1 = slider_param(prefix, "a1", "系数 a1", -1.5, 1.5, -0.6, 0.1)
        b0 = slider_param(prefix, "b0", "系数 b0", 0.2, 2.0, 1.0, 0.1)
        b1 = slider_param(prefix, "b1", "系数 b1", -1.0, 1.5, 0.4, 0.1)
    with right:
        n = np.arange(0, 24)
        x = (n >= 0).astype(float)
        y = signal.lfilter([b0, b1], [1.0, a1], x)
        render_chart("差分方程递推解", n, [{"name": "输入 x[n]", "y": x}, {"name": "输出 y[n]", "y": y}], "ch3s1_main", "n", "幅值", discrete=True)
    note_block("1. LTI离散系统、差分方程求解")


def render_ch3_sec2():
    prefix = "ch3s2"
    defaults = {"a": 0.7, "b": 1.0, "y0": 2.0}
    example = {"a": 0.5, "b": 1.2, "y0": -1.5}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        a = slider_param(prefix, "a", "系统参数 a", -0.9, 0.9, 0.7, 0.1)
        b = slider_param(prefix, "b", "输入系数 b", 0.2, 2.0, 1.0, 0.1)
        y0 = slider_param(prefix, "y0", "初始值 y[-1]", -3.0, 3.0, 2.0, 0.1)
    with right:
        n = np.arange(0, 20)
        x = (n >= 0).astype(float)
        y_zi = y0 * (a ** (n + 1))
        y_zs = signal.lfilter([b], [1.0, -a], x)
        y = y_zi + y_zs
        render_chart("离散系统零输入/零状态响应", n, [{"name": "零输入", "y": y_zi}, {"name": "零状态", "y": y_zs}, {"name": "总响应", "y": y}], "ch3s2_main", "n", "幅值", discrete=True)
    note_block("2. 离散系统零输入、零状态响应")


def render_ch3_sec3():
    prefix = "ch3s3"
    defaults = {"a": 0.65}
    example = {"a": 0.3}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        a = slider_param(prefix, "a", "系统极点参数 a", -0.8, 0.8, 0.65, 0.05)
    with right:
        n = np.arange(0, 20)
        delta = (n == 0).astype(float)
        step = (n >= 0).astype(float)
        h = signal.lfilter([1.0], [1.0, -a], delta)
        s = signal.lfilter([1.0], [1.0, -a], step)
        render_chart("单位序列响应与单位阶跃响应", n, [{"name": "h[n]", "y": h}, {"name": "s[n]", "y": s}], "ch3s3_main", "n", "幅值", discrete=True)
    note_block("3. 单位序列、单位阶跃序列响应")


def render_ch3_sec4():
    prefix = "ch3s4"
    defaults = {"n0": 4}
    example = {"n0": 7}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        n0 = slider_param(prefix, "n0", "当前输出索引 n", 0, 10, 4, 1)
    with right:
        k = np.arange(0, 8)
        x = np.array([1, 2, 1, 0, 0, 0, 0, 0], dtype=float)
        h = np.array([1, -1, 2, 0, 0, 0, 0, 0], dtype=float)
        h_shift = np.zeros_like(k, dtype=float)
        for i, kv in enumerate(k):
            idx = n0 - kv
            if 0 <= idx < len(h):
                h_shift[i] = h[idx]
        product = x * h_shift
        y = np.convolve(x, h)
        fig = make_subplots(rows=2, cols=1, subplot_titles=("求和过程", "卷积和结果"))
        fig.add_trace(go.Bar(x=k, y=x, name="x[k]"), row=1, col=1)
        fig.add_trace(go.Bar(x=k, y=h_shift, name="h[n-k]"), row=1, col=1)
        fig.add_trace(go.Bar(x=k, y=product, name="乘积项"), row=1, col=1)
        fig.add_trace(go.Bar(x=np.arange(len(y)), y=y, name="y[n]"), row=2, col=1)
        fig.add_trace(go.Scatter(x=[n0], y=[y[n0]], mode="markers", name="当前 y[n0]"), row=2, col=1)
        fig.update_layout(height=680, barmode="group", margin=dict(l=40, r=20, t=80, b=30))
        st.plotly_chart(fig, use_container_width=True, key="ch3s4_plot")
        st.metric("当前卷积和 y[n0]", f"{y[n0]:.4f}")
    note_block("4. 离散卷积和完整可视化")


def render_ch3_sec5():
    prefix = "ch3s5"
    defaults = {"scale": 1.0}
    example = {"scale": 1.5}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        scale = slider_param(prefix, "scale", "观测幅值缩放", 0.5, 2.0, 1.0, 0.1)
    with right:
        x = np.array([1.0, 2.0, 1.0])
        h = np.array([1.0, -1.0, 2.0])
        y = np.convolve(x, h)
        x_rec, _ = signal.deconvolve(y, h)
        render_chart("反卷积恢复输入序列", np.arange(len(y)), [{"name": "卷积结果 y[n]", "y": scale * y}], "ch3s5_main", "n", "幅值", discrete=True)
        st.write(f"原输入序列 x[n] = {np.round(x, 4).tolist()}")
        st.write(f"已知系统 h[n] = {np.round(h, 4).tolist()}")
        st.write(f"反卷积恢复 x[n] = {np.round(x_rec, 4).tolist()}")
    note_block("5. 反卷积简易演示")


def render_ch4_sec1():
    prefix = "ch4s1"
    defaults = {"harmonics": 3}
    example = {"harmonics": 5}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        harmonics = slider_param(prefix, "harmonics", "展开阶数 K", 1, 8, 3, 1)
    with right:
        t = np.linspace(-np.pi, np.pi, 1200)
        x = np.sign(np.sin(t))
        recon = np.zeros_like(t)
        coeffs = []
        for k in range(1, harmonics + 1):
            phi = np.sin((2 * k - 1) * t)
            ck = np.trapz(x * phi, t) / np.trapz(phi * phi, t)
            coeffs.append((2 * k - 1, ck))
            recon += ck * phi
        render_chart("正交函数分解与重构", t, [{"name": "原信号", "y": x}, {"name": "正交展开重构", "y": recon}], "ch4s1_main", "t", "幅值")
        st.dataframe({"谐波序号": [k for k, _ in coeffs], "投影系数 c_k": [round(v, 5) for _, v in coeffs]}, use_container_width=True)
    note_block("1. 信号正交函数分解原理")


def render_ch4_sec2():
    prefix = "ch4s2"
    defaults = {"kind": "方波", "harmonics": 7}
    example = {"kind": "三角波", "harmonics": 9}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        kind = select_param(prefix, "kind", "周期信号", ["方波", "三角波"], 0)
        harmonics = slider_param(prefix, "harmonics", "保留谐波项数", 1, 15, 7, 1)
    with right:
        t = np.linspace(-np.pi, np.pi, 1400)
        if kind == "方波":
            x = np.sign(np.sin(t))
            series = np.zeros_like(t)
            ks = np.arange(1, harmonics + 1)
            amps = []
            for k in ks:
                n = 2 * k - 1
                amp = 4 / (np.pi * n)
                series += amp * np.sin(n * t)
                amps.append(abs(amp))
            symmetry = "奇函数，仅含正弦项"
        else:
            x = signal.sawtooth(t, width=0.5)
            series = np.zeros_like(t)
            ks = np.arange(1, harmonics + 1)
            amps = []
            for k in ks:
                n = 2 * k - 1
                amp = 8 / (np.pi**2 * n**2) * ((-1) ** ((n - 1) // 2))
                series += amp * np.sin(n * t)
                amps.append(abs(amp))
            symmetry = "奇函数，仅含奇次正弦谐波"
        render_chart("傅里叶级数逼近", t, [{"name": "原周期信号", "y": x}, {"name": "级数重构", "y": series}], "ch4s2_main", "ω0 t", "幅值")
        render_chart("奇偶函数频谱线图", ks, [{"name": "谐波幅值", "y": amps}], "ch4s2_spectrum", "谐波序号", "幅值", discrete=True)
        st.success(f"对称性判定：{symmetry}")
    note_block("2. 周期信号傅里叶级数、奇偶函数频谱")


def render_ch4_sec3():
    prefix = "ch4s3"
    defaults = {"kind": "非周期脉冲", "freq": 2.0}
    example = {"kind": "双频信号", "freq": 3.0}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        kind = select_param(prefix, "kind", "信号类型", ["非周期脉冲", "双频信号"], 0)
        freq = slider_param(prefix, "freq", "主频参数", 0.5, 4.0, 2.0, 0.1)
    with right:
        t = np.linspace(-4, 4, 2048)
        dt = t[1] - t[0]
        if kind == "非周期脉冲":
            x = rect(t, width=1.2)
        else:
            x = np.cos(2 * np.pi * freq * t) + 0.6 * np.sin(2 * np.pi * 0.5 * freq * t)
        f, X = fft_spectrum(x, dt)
        fig = make_subplots(rows=2, cols=1, subplot_titles=("幅度谱", "相位谱"))
        fig.add_trace(go.Scatter(x=f, y=np.abs(X), mode="lines", name="|X(f)|"), row=1, col=1)
        fig.add_trace(go.Scatter(x=f, y=np.angle(X), mode="lines", name="∠X(f)"), row=2, col=1)
        fig.update_layout(height=660, margin=dict(l=40, r=20, t=80, b=30))
        st.plotly_chart(fig, use_container_width=True, key="ch4s3_plot")
    note_block("3. 周期/非周期信号幅度谱、相位谱绘制")


def render_ch4_sec4():
    prefix = "ch4s4"
    defaults = {"property": "时移", "shift": 0.8, "scale": 1.5}
    example = {"property": "频移", "shift": 1.2, "scale": 2.0}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        prop = select_param(prefix, "property", "傅里叶性质", ["时移", "频移", "尺度"], 0)
        shift = slider_param(prefix, "shift", "时移/调制参数", 0.2, 2.0, 0.8, 0.1)
        scale = slider_param(prefix, "scale", "尺度参数", 0.6, 2.5, 1.5, 0.1)
    with right:
        t = np.linspace(-4, 4, 2048)
        dt = t[1] - t[0]
        x = np.exp(-t**2)
        if prop == "时移":
            y = np.exp(-(t - shift) ** 2)
            desc = "时移性质：时域平移导致频域乘上线性相位因子。"
        elif prop == "频移":
            y = x * np.cos(2 * np.pi * shift * t)
            desc = "频移性质：时域调制导致频域谱搬移。"
        else:
            y = np.exp(-(scale * t) ** 2)
            desc = "尺度性质：时域压缩导致频域展宽。"
        f, X = fft_spectrum(x, dt)
        _, Y = fft_spectrum(y, dt)
        render_chart("时域信号对比", t, [{"name": "原信号 x(t)", "y": x}, {"name": "变换后 y(t)", "y": y}], "ch4s4_time", "t / s", "幅值")
        render_chart("频域幅度谱对比", f, [{"name": "|X(f)|", "y": np.abs(X)}, {"name": "|Y(f)|", "y": np.abs(Y)}], "ch4s4_freq", "f / Hz", "幅度")
        st.success(desc)
    note_block("4. 傅里叶变换全部核心性质逐一交互验证")


def render_ch4_sec5():
    prefix = "ch4s5"
    defaults = {"width": 1.0}
    example = {"width": 2.0}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        width = slider_param(prefix, "width", "脉冲宽度", 0.5, 3.0, 1.0, 0.1)
    with right:
        t = np.linspace(-6, 6, 2048)
        dt = t[1] - t[0]
        pulse = rect(t, width=width)
        f, X = fft_spectrum(pulse, dt)
        square_coeffs = 4 / (np.pi * (2 * np.arange(1, 8) - 1))
        power_lines = np.abs(square_coeffs) ** 2
        render_chart("能量谱 |X(f)|^2", f, [{"name": "能量谱", "y": np.abs(X) ** 2}], "ch4s5_energy", "f / Hz", "谱密度")
        render_chart("周期方波功率谱线", np.arange(1, 8), [{"name": "|C_k|^2", "y": power_lines}], "ch4s5_power", "奇次谐波序号", "功率", discrete=True)
    note_block("5. 能量谱、功率谱、周期信号傅里叶变换")


def render_ch4_sec6():
    prefix = "ch4s6"
    defaults = {"wc": 5.0}
    example = {"wc": 2.5}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        wc = slider_param(prefix, "wc", "截止角频率 ωc", 1.0, 10.0, 5.0, 0.1)
    with right:
        w = np.linspace(0, 12, 800)
        H = 1 / (1 + 1j * w / wc)
        t = np.linspace(0, 6, 1600)
        x = np.sin(2 * np.pi * 0.8 * t) + 0.5 * np.sin(2 * np.pi * 2.5 * t)
        sys = signal.TransferFunction([wc], [1.0, wc])
        _, y, _ = signal.lsim(sys, U=x, T=t)
        fig = make_subplots(rows=2, cols=1, subplot_titles=("幅频响应", "相频响应"))
        fig.add_trace(go.Scatter(x=w, y=np.abs(H), mode="lines", name="|H(jω)|"), row=1, col=1)
        fig.add_trace(go.Scatter(x=w, y=np.angle(H), mode="lines", name="∠H(jω)"), row=2, col=1)
        fig.update_layout(height=650, margin=dict(l=40, r=20, t=80, b=30))
        st.plotly_chart(fig, use_container_width=True, key="ch4s6_response")
        render_chart("频域分析对应的时域输出", t, [{"name": "输入 x(t)", "y": x}, {"name": "输出 y(t)", "y": y}], "ch4s6_time", "t / s", "幅值")
    note_block("6. LTI系统频率响应、频域完整分析")


def render_ch4_sec7():
    prefix = "ch4s7"
    defaults = {"fc": 2.0, "delay": 0.4}
    example = {"fc": 1.0, "delay": 0.8}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        fc = slider_param(prefix, "fc", "理想低通截止频率 fc", 0.5, 4.0, 2.0, 0.1)
        delay = slider_param(prefix, "delay", "无失真传输时延 td", 0.0, 1.5, 0.4, 0.1)
    with right:
        t = np.linspace(-4, 4, 2048)
        dt = t[1] - t[0]
        x = np.sinc(1.2 * t) + 0.4 * np.cos(2 * np.pi * 3.0 * t)
        f, X = fft_spectrum(x, dt)
        ideal_mask = (np.abs(f) <= fc).astype(float)
        Y = X * ideal_mask * np.exp(-1j * 2 * np.pi * f * delay)
        y = np.real(np.fft.ifft(np.fft.ifftshift(Y)))
        render_chart("无失真传输与理想低通输出", t, [{"name": "输入 x(t)", "y": x}, {"name": "输出 y(t)", "y": y}], "ch4s7_time", "t / s", "幅值")
        render_chart("滤波前后频谱", f, [{"name": "|X(f)|", "y": np.abs(X)}, {"name": "|Y(f)|", "y": np.abs(Y)}], "ch4s7_freq", "f / Hz", "幅度")
    note_block("7. 无失真传输、理想低通滤波器效果仿真")


def sinc_reconstruct(samples_t, samples_x, t_grid):
    Ts = samples_t[1] - samples_t[0]
    kernel = np.sinc((t_grid[:, None] - samples_t[None, :]) / Ts)
    return kernel @ samples_x


def render_ch4_sec8():
    prefix = "ch4s8"
    defaults = {"fs": 8.0}
    example = {"fs": 3.5}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        fs = slider_param(prefix, "fs", "采样频率 fs", 2.0, 12.0, 8.0, 0.1)
    with right:
        t = np.linspace(-2, 2, 1600)
        x = np.cos(2 * np.pi * 1.2 * t) + 0.5 * np.cos(2 * np.pi * 2.6 * t)
        Ts = 1.0 / fs
        ts = np.arange(-2, 2 + Ts, Ts)
        xs = np.cos(2 * np.pi * 1.2 * ts) + 0.5 * np.cos(2 * np.pi * 2.6 * ts)
        xr = sinc_reconstruct(ts, xs, t)
        render_chart("时域采样与重建", t, [{"name": "原连续信号", "y": x}, {"name": "重建信号", "y": xr}], "ch4s8_time", "t / s", "幅值")
        render_chart("采样点序列", ts, [{"name": "x(nTs)", "y": xs}], "ch4s8_samples", "t / s", "幅值", discrete=True)
        verdict = "满足奈奎斯特采样条件" if fs >= 2 * 2.6 else "低于奈奎斯特频率，存在混叠风险"
        st.success(verdict)
    note_block("8. 奈奎斯特取样定理、时域/频域取样全过程演示")


def render_ch4_sec9():
    prefix = "ch4s9"
    defaults = {"freq": 0.2}
    example = {"freq": 0.32}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        freq = slider_param(prefix, "freq", "离散角频率参数", 0.05, 0.45, 0.2, 0.01)
    with right:
        n = np.arange(0, 16)
        x = np.cos(2 * np.pi * freq * n) + 0.5 * (n == 3).astype(float)
        omega = np.linspace(-np.pi, np.pi, 1024)
        X_dtft = np.array([np.sum(x * np.exp(-1j * w * n)) for w in omega])
        X_dft = np.fft.fftshift(np.fft.fft(x))
        k = np.arange(-len(x) // 2, len(x) // 2)
        render_chart("DTFT 幅度谱", omega, [{"name": "|X(e^{jω})|", "y": np.abs(X_dtft)}], "ch4s9_dtft", "ω / rad", "幅度")
        render_chart("DFT 采样点", k, [{"name": "|X[k]|", "y": np.abs(X_dft)}], "ch4s9_dft", "频域索引 k", "幅度", discrete=True)
    note_block("9. 离散时间傅里叶变换DTFT、DFT可视化")


def render_ch5_sec1():
    prefix = "ch5s1"
    defaults = {"kind": "右边指数信号", "a": 1.0}
    example = {"kind": "双边指数信号", "a": 0.8}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        kind = select_param(prefix, "kind", "信号类型", ["右边指数信号", "左边指数信号", "双边指数信号"], 0)
        a = slider_param(prefix, "a", "指数参数 a", 0.2, 2.0, 1.0, 0.1)
    with right:
        t = np.linspace(-6, 6, 1600)
        if kind == "右边指数信号":
            x = np.exp(-a * np.maximum(t, 0)) * heaviside(t)
            formula = f"X(s)=1/(s+{a:.2f}), ROC: Re(s) > {-a:.2f}"
            poles = np.array([-a + 0j])
        elif kind == "左边指数信号":
            x = -np.exp(a * t) * (t <= 0).astype(float)
            formula = f"X(s)=1/(s+{a:.2f}), ROC: Re(s) < {-a:.2f}"
            poles = np.array([-a + 0j])
        else:
            x = np.exp(-a * np.abs(t))
            formula = f"X(s)=2a/(a^2-s^2), ROC: {-a:.2f} < Re(s) < {a:.2f}"
            poles = np.array([a + 0j, -a + 0j])
        render_chart("时域信号", t, [{"name": "x(t)", "y": x}], "ch5s1_time", "t / s", "幅值")
        render_pole_zero_chart([], poles, "s 平面极点图", "ch5s1_pz", xlim=(-2.5, 2.5), ylim=(-2.0, 2.0))
        st.success(formula)
    note_block("1. 单边/双边拉普拉斯变换、收敛域ROC")


def render_ch5_sec2():
    prefix = "ch5s2"
    defaults = {"property": "求导性质", "a": 1.0}
    example = {"property": "时移性质", "a": 0.6}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        prop = select_param(prefix, "property", "性质类型", ["线性性质", "时移性质", "求导性质", "卷积性质"], 2)
        a = slider_param(prefix, "a", "参数 a", 0.2, 2.0, 1.0, 0.1)
    with right:
        sigma = np.linspace(-0.9, 4, 400)
        if prop == "线性性质":
            lhs = 2 / (sigma + a) + 1 / (sigma + a + 1)
            rhs = lhs.copy()
            desc = "线性性质：L{2x1+x2}=2X1+X2"
        elif prop == "时移性质":
            lhs = np.exp(-sigma) / (sigma + a)
            rhs = lhs.copy()
            desc = "时移性质：L{x(t-t0)u(t-t0)}=e^{-st0}X(s)"
        elif prop == "求导性质":
            lhs = sigma / (sigma + a) - 1
            rhs = -a / (sigma + a)
            desc = "求导性质：L{dx/dt}=sX(s)-x(0-)"
        else:
            lhs = 1 / (sigma + a) ** 2
            rhs = lhs.copy()
            desc = "卷积性质：L{x*h}=X(s)H(s)"
        render_chart("拉普拉斯性质验证曲线", sigma, [{"name": "左式", "y": lhs}, {"name": "右式", "y": rhs}], "ch5s2_main", "σ (取 s=σ)", "数值")
        st.success(desc)
    note_block("2. 拉普拉斯变换全套性质逐一验证")


def render_ch5_sec3():
    prefix = "ch5s3"
    defaults = {"a": 1.0, "b": 2.0}
    example = {"a": 0.8, "b": 3.0}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        a = slider_param(prefix, "a", "极点 p1", 0.2, 3.0, 1.0, 0.1)
        b = slider_param(prefix, "b", "极点 p2", 0.5, 4.0, 2.0, 0.1)
    with right:
        num = [1.0]
        den = np.polymul([1.0, a], [1.0, b])
        r, p, k = signal.residue(num, den)
        t = np.linspace(0, 8, 1200)
        x = np.real(np.sum([ri * np.exp(pi * t) for ri, pi in zip(r, p)], axis=0)) * heaviside(t)
        render_chart("逆拉普拉斯后的时域信号", t, [{"name": "x(t)", "y": x}], "ch5s3_main", "t / s", "幅值")
        st.write("部分分式展开结果：")
        for idx, (ri, pi) in enumerate(zip(r, p), start=1):
            st.write(f"A{idx} = {ri:.4f}, 极点 p{idx} = {pi:.4f}")
        if len(k):
            st.write(f"直接项：{k}")
    note_block("3. 拉普拉斯逆变换（查表法+部分分式展开）")


def render_ch5_sec4():
    prefix = "ch5s4"
    defaults = {"a": 1.2, "b": 1.0}
    example = {"a": 0.6, "b": 1.8}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        a = slider_param(prefix, "a", "系统参数 a", 0.2, 3.0, 1.2, 0.1)
        b = slider_param(prefix, "b", "输入系数 b", 0.2, 3.0, 1.0, 0.1)
    with right:
        t = np.linspace(0, 8, 1400)
        x = heaviside(t)
        sys = signal.TransferFunction([b], [1.0, a])
        _, y, _ = signal.lsim(sys, U=x, T=t)
        render_chart("s 域求解对应时域响应", t, [{"name": "输入 x(t)", "y": x}, {"name": "输出 y(t)", "y": y}], "ch5s4_main", "t / s", "幅值")
        st.code(f"Y(s) = ({b:.2f})/(s+{a:.2f}) · X(s)", language="text")
    note_block("4. s域复频域微分方程求解")


def render_ch5_sec5():
    prefix = "ch5s5"
    defaults = {"R": 1.0, "C": 1.0, "kind": "RC低通"}
    example = {"R": 1.0, "C": 0.5, "kind": "RC低通"}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        kind = select_param(prefix, "kind", "电路模型", ["RC低通", "RLC二阶"], 0)
        R = slider_param(prefix, "R", "电阻 R", 0.5, 5.0, 1.0, 0.1)
        C = slider_param(prefix, "C", "电容/参数 C", 0.2, 2.0, 1.0, 0.1)
    with right:
        if kind == "RC低通":
            num = [1.0]
            den = [R * C, 1.0]
            poles = np.roots(den)
            formula = f"H(s)=1/({R:.2f}{C:.2f}s+1)"
        else:
            num = [1.0]
            den = [1.0, R, 1.0 / C]
            poles = np.roots(den)
            formula = f"H(s)=1/(s^2+{R:.2f}s+{1.0 / C:.2f})"
        render_pole_zero_chart([], poles, "系统函数 H(s) 的零极点图", "ch5s5_pz", xlim=(-4, 2), ylim=(-4, 4))
        w = np.linspace(0, 10, 800)
        H = np.polyval(num, 1j * w) / np.polyval(den, 1j * w)
        render_chart("H(s) 在虚轴上的频率响应", w, [{"name": "|H(jω)|", "y": np.abs(H)}], "ch5s5_freq", "ω", "幅度")
        st.code(formula, language="text")
    note_block("5. 系统函数H(s)、s域框图、电路s域建模")


def render_ch5_sec6():
    prefix = "ch5s6"
    defaults = {"a": 0.8}
    example = {"a": -0.2}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        a = slider_param(prefix, "a", "指数参数 a", -1.0, 2.0, 0.8, 0.1)
    with right:
        w = np.linspace(-8, 8, 600)
        H = 1 / (a + 1j * w)
        render_chart("虚轴代入得到的傅里叶形式", w, [{"name": "|X(jω)|", "y": np.abs(H)}], "ch5s6_main", "ω", "幅度")
        if a > 0:
            st.success("ROC 为 Re(s) > -a，包含虚轴，因此傅里叶变换存在。")
        else:
            st.error("ROC 不包含虚轴，傅里叶变换不存在，但拉普拉斯变换仍可用于分析。")
    note_block("6. 拉普拉斯与傅里叶变换关系对照")


def render_ch6_sec1():
    prefix = "ch6s1"
    defaults = {"kind": "右边序列", "a": 0.7}
    example = {"kind": "左边序列", "a": 0.6}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        kind = select_param(prefix, "kind", "序列类型", ["右边序列", "左边序列"], 0)
        a = slider_param(prefix, "a", "参数 a", 0.1, 1.5, 0.7, 0.1)
    with right:
        n = np.arange(-10, 11)
        if kind == "右边序列":
            x = (a ** np.maximum(n, 0)) * (n >= 0)
            roc = f"X(z)=z/(z-{a:.2f}), ROC: |z| > {a:.2f}"
        else:
            x = -(a ** n) * (n <= -1)
            roc = f"X(z)=z/(z-{a:.2f}), ROC: |z| < {a:.2f}"
        render_chart("Z 变换对应离散序列", n, [{"name": "x[n]", "y": x}], "ch6s1_main", "n", "幅值", discrete=True)
        render_pole_zero_chart([], np.array([a + 0j]), "z 平面极点位置", "ch6s1_pz", xlim=(-2, 2), ylim=(-2, 2), unit_circle=True)
        st.success(roc)
    note_block("1. Z变换定义、收敛域")


def render_ch6_sec2():
    prefix = "ch6s2"
    properties = ["线性性质", "时移性质", "卷积性质", "调制性质", "微分性质", "初值定理", "终值定理", "反转性质"]
    defaults = {"property": "时移性质", "a": 0.7}
    example = {"property": "卷积性质", "a": 0.5}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        prop = select_param(prefix, "property", "性质条目", properties, 1)
        a = slider_param(prefix, "a", "序列参数 a", 0.1, 0.9, 0.7, 0.05)
    with right:
        n = np.arange(0, 14)
        x = a**n
        if prop == "卷积性质":
            y = np.convolve(x[:8], np.ones(4), mode="full")
            render_chart("卷积性质示例", np.arange(len(y)), [{"name": "卷积结果", "y": y}], "ch6s2_conv", "n", "幅值", discrete=True)
            st.success("卷积性质：x[n]*h[n] ↔ X(z)H(z)")
            note_block("2. Z变换全套8大性质交互演示")
            return
        if prop == "线性性质":
            y = 2 * x + 0.5 * (0.4**n)
            desc = "线性性质：a1x1[n]+a2x2[n] ↔ a1X1(z)+a2X2(z)"
        elif prop == "时移性质":
            y = np.concatenate([np.zeros(2), x[:-2]])
            desc = "时移性质：x[n-n0] ↔ z^{-n0}X(z)"
        elif prop == "调制性质":
            y = x * np.cos(0.4 * np.pi * n)
            desc = "调制性质：序列调制会改变极点角度分布。"
        elif prop == "微分性质":
            y = n * x
            desc = "微分性质：n x[n] ↔ -z dX(z)/dz"
        elif prop == "初值定理":
            y = np.zeros_like(x)
            y[0] = x[0]
            desc = "初值定理：x[0]=lim(z→∞)X(z)"
        elif prop == "终值定理":
            y = np.ones_like(x) * x[-1]
            desc = "终值定理：lim(n→∞)x[n]=lim(z→1)(1-z^{-1})X(z)"
        else:
            y = x[::-1]
            desc = "反转性质：x[-n] ↔ X(z^{-1})"
        render_chart("Z 变换性质对应序列变化", n, [{"name": "原序列 x[n]", "y": x}, {"name": "性质作用后", "y": y}], "ch6s2_main", "n", "幅值", discrete=True)
        st.success(desc)
    note_block("2. Z变换全套8大性质交互演示")


def render_ch6_sec3():
    prefix = "ch6s3"
    defaults = {"a": 0.6, "b": 0.3}
    example = {"a": 0.8, "b": 0.5}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        a = slider_param(prefix, "a", "极点 a", 0.2, 0.95, 0.6, 0.05)
        b = slider_param(prefix, "b", "极点 b", 0.1, 0.9, 0.3, 0.05)
    with right:
        num = [1.0]
        den = np.polymul([1.0, -a], [1.0, -b])
        r, p, k = signal.residuez(num, den)
        n = np.arange(0, 16)
        x = np.real(np.sum([ri * (pi**n) for ri, pi in zip(r, p)], axis=0))
        render_chart("逆 Z 变换得到的右边序列", n, [{"name": "x[n]", "y": x}], "ch6s3_main", "n", "幅值", discrete=True)
        st.write("部分分式展开结果：")
        for idx, (ri, pi) in enumerate(zip(r, p), start=1):
            st.write(f"A{idx} = {ri:.4f}, 极点 p{idx} = {pi:.4f}")
        if len(k):
            st.write(f"直接项：{k}")
    note_block("3. 逆Z变换：幂级数展开、部分分式法")


def render_ch6_sec4():
    prefix = "ch6s4"
    defaults = {"a1": -0.7, "b0": 1.0, "b1": 0.4}
    example = {"a1": -0.4, "b0": 1.0, "b1": 0.8}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        a1 = slider_param(prefix, "a1", "分母系数 a1", -1.5, 1.5, -0.7, 0.1)
        b0 = slider_param(prefix, "b0", "分子系数 b0", 0.2, 2.0, 1.0, 0.1)
        b1 = slider_param(prefix, "b1", "分子系数 b1", -1.0, 1.5, 0.4, 0.1)
    with right:
        n = np.arange(0, 24)
        x = (n >= 0).astype(float)
        y = signal.lfilter([b0, b1], [1.0, a1], x)
        render_chart("差分方程 z 域求解结果", n, [{"name": "输出 y[n]", "y": y}], "ch6s4_main", "n", "幅值", discrete=True)
        st.code(f"H(z)=({b0:.2f}+{b1:.2f}z^(-1)) / (1+{a1:.2f}z^(-1))", language="text")
    note_block("4. 差分方程z域完整求解")


def render_ch6_sec5():
    prefix = "ch6s5"
    defaults = {"kind": "一阶IIR"}
    example = {"kind": "移动平均"}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        kind = select_param(prefix, "kind", "离散系统结构", ["移动平均", "一阶IIR", "二阶谐振器"], 1)
    with right:
        if kind == "移动平均":
            b, a = np.array([1 / 3, 1 / 3, 1 / 3]), np.array([1.0])
        elif kind == "一阶IIR":
            b, a = np.array([1.0]), np.array([1.0, -0.65])
        else:
            b, a = np.array([1.0]), np.array([1.0, -1.2, 0.81])
        z, p, _ = signal.tf2zpk(b, a)
        w = np.linspace(0, np.pi, 800)
        _, h = signal.freqz(b, a, worN=w)
        render_pole_zero_chart(z, p, "H(z) 零极点分布", "ch6s5_pz", xlim=(-1.8, 1.8), ylim=(-1.8, 1.8), unit_circle=True)
        render_chart("离散系统频率响应", w, [{"name": "|H(e^{jω})|", "y": np.abs(h)}], "ch6s5_freq", "ω / rad", "幅度")
        st.code(f"b = {np.round(b, 4).tolist()}, a = {np.round(a, 4).tolist()}", language="text")
    note_block("5. 离散系统函数H(z)、z域框图")


def render_ch6_sec6():
    prefix = "ch6s6"
    defaults = {"T": 0.2}
    example = {"T": 0.5}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        T = slider_param(prefix, "T", "采样周期 T", 0.05, 1.0, 0.2, 0.05)
    with right:
        sigma = np.array([-2.0, -1.0, -0.5, 0.0])
        omega = np.array([0.0, 1.0, 2.5, -1.5])
        s_points = sigma + 1j * omega
        z_points = np.exp(s_points * T)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("s 平面采样点", "z 平面映射点"))
        fig.add_trace(go.Scatter(x=np.real(s_points), y=np.imag(s_points), mode="markers+text", text=[f"s{i+1}" for i in range(len(s_points))], name="s"), row=1, col=1)
        theta = np.linspace(0, 2 * np.pi, 400)
        fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines", name="单位圆"), row=1, col=2)
        fig.add_trace(go.Scatter(x=np.real(z_points), y=np.imag(z_points), mode="markers+text", text=[f"z{i+1}" for i in range(len(z_points))], name="z"), row=1, col=2)
        fig.update_layout(height=440, margin=dict(l=40, r=20, t=80, b=30))
        st.plotly_chart(fig, use_container_width=True, key="ch6s6_plot")
        st.success("结论：s 平面左半平面映射到 z 平面单位圆内；虚轴映射为单位圆。")
    note_block("6. s域与z域映射关系")


def render_ch7_sec1():
    prefix = "ch7s1"
    defaults = {"zeta": 0.4, "wn": 4.0}
    example = {"zeta": 0.1, "wn": 5.0}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        zeta = slider_param(prefix, "zeta", "阻尼比 ζ", 0.0, 1.5, 0.4, 0.05)
        wn = slider_param(prefix, "wn", "固有角频率 ωn", 1.0, 8.0, 4.0, 0.1)
    with right:
        num = [wn**2]
        den = [1.0, 2 * zeta * wn, wn**2]
        zeros, poles, _ = signal.tf2zpk(num, den)
        render_pole_zero_chart(zeros, poles, "零极点分布", "ch7s1_pz", xlim=(-6, 2), ylim=(-6, 6))
        t = np.linspace(0, 8, 1500)
        sys = signal.TransferFunction(num, den)
        _, h = signal.impulse(sys, T=t)
        w = np.linspace(0, 12, 800)
        H = np.polyval(num, 1j * w) / np.polyval(den, 1j * w)
        render_chart("时域冲激响应", t, [{"name": "h(t)", "y": h}], "ch7s1_time", "t / s", "幅值")
        render_chart("频域幅频响应", w, [{"name": "|H(jω)|", "y": np.abs(H)}], "ch7s1_freq", "ω", "幅度")
    note_block("1. 系统零极点分布、时域/频域响应对应关系")


def render_ch7_sec2():
    prefix = "ch7s2"
    defaults = {"domain": "连续系统", "sigma": -0.8, "omega": 1.5, "radius": 0.8, "theta": 1.0}
    example = {"domain": "离散系统", "sigma": -0.8, "omega": 1.5, "radius": 1.1, "theta": 0.8}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        domain = select_param(prefix, "domain", "系统类型", ["连续系统", "离散系统"], 0)
        sigma = slider_param(prefix, "sigma", "连续极点实部 σ", -3.0, 1.0, -0.8, 0.1)
        omega = slider_param(prefix, "omega", "连续极点虚部 ω", 0.0, 4.0, 1.5, 0.1)
        radius = slider_param(prefix, "radius", "离散极点模 r", 0.2, 1.3, 0.8, 0.05)
        theta = slider_param(prefix, "theta", "离散极点角 θ", 0.2, 2.8, 1.0, 0.1)
    with right:
        if domain == "连续系统":
            poles = np.array([sigma + 1j * omega, sigma - 1j * omega])
            den = np.poly(poles).real
            sys = signal.TransferFunction([1.0], den)
            t = np.linspace(0, 8, 1200)
            _, h = signal.impulse(sys, T=t)
            render_pole_zero_chart([], poles, "连续系统极点图", "ch7s2_cont_pz", xlim=(-4, 2), ylim=(-4, 4))
            render_chart("连续系统冲激响应", t, [{"name": "h(t)", "y": h}], "ch7s2_cont_time", "t / s", "幅值")
            stable = bool(np.all(np.real(poles) < 0))
            st.write(f"判定：当前极点实部均{'小于' if stable else '不全小于'} 0，系统{'稳定' if stable else '不稳定'}。")
        else:
            poles = np.array([radius * np.exp(1j * theta), radius * np.exp(-1j * theta)])
            den = np.poly(poles).real
            w, h = signal.freqz([1.0], den, worN=800)
            render_pole_zero_chart([], poles, "离散系统极点图", "ch7s2_disc_pz", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6), unit_circle=True)
            render_chart("离散系统频率响应", w, [{"name": "|H(e^{jω})|", "y": np.abs(h)}], "ch7s2_disc_freq", "ω / rad", "幅度")
            stable = bool(np.all(np.abs(poles) < 1))
            st.write(f"判定：当前极点模长{'小于' if stable else '不全小于'} 1，系统{'稳定' if stable else '不稳定'}。")
    note_block("2. 系统因果性、稳定性判定仿真")


def render_ch7_sec3():
    prefix = "ch7s3"
    defaults = {"g1": 1.0, "g2": 1.5, "h1": 0.3, "h2": 0.2}
    example = {"g1": 1.2, "g2": 2.0, "h1": 0.5, "h2": 0.4}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        g1 = slider_param(prefix, "g1", "前向增益 G1", 0.2, 3.0, 1.0, 0.1)
        g2 = slider_param(prefix, "g2", "前向增益 G2", 0.2, 3.0, 1.5, 0.1)
        h1 = slider_param(prefix, "h1", "反馈增益 H1", 0.0, 1.5, 0.3, 0.1)
        h2 = slider_param(prefix, "h2", "反馈增益 H2", 0.0, 1.5, 0.2, 0.1)
    with right:
        P1 = g1 * g2
        L1 = -g1 * h1
        L2 = -g2 * h2
        Delta = 1 - (L1 + L2) + L1 * L2
        T = P1 / Delta
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1, 2, 3], y=[0, 0, 0, 0], mode="markers+text", text=["R", "X1", "X2", "C"], textposition="top center", marker_size=14, name="节点"))
        fig.add_trace(go.Scatter(x=[0, 1, 2, 3], y=[0, 0, 0, 0], mode="lines", name="前向通路"))
        fig.add_trace(go.Scatter(x=[1, 1], y=[0, 0.8], mode="lines", name="H1 回路"))
        fig.add_trace(go.Scatter(x=[2, 2], y=[0, -0.8], mode="lines", name="H2 回路"))
        fig.update_layout(height=380, title="信号流图示意", margin=dict(l=40, r=20, t=60, b=30), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="ch7s3_graph")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("前向通路 P1", f"{P1:.4f}")
        c2.metric("回路 L1", f"{L1:.4f}")
        c3.metric("回路 L2", f"{L2:.4f}")
        c4.metric("总传函 T", f"{T:.4f}")
        st.code(f"Δ = 1 - (L1 + L2) + L1·L2 = {Delta:.4f}\nT = P1 / Δ = {T:.4f}", language="text")
    note_block("3. 信号流图、梅森公式可视化演示")


def render_ch7_sec4():
    prefix = "ch7s4"
    defaults = {"tau1": 0.8, "tau2": 1.5, "beta": 0.5}
    example = {"tau1": 0.4, "tau2": 0.9, "beta": 0.8}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        tau1 = slider_param(prefix, "tau1", "一级时间常数 τ1", 0.2, 3.0, 0.8, 0.1)
        tau2 = slider_param(prefix, "tau2", "二级时间常数 τ2", 0.2, 3.0, 1.5, 0.1)
        beta = slider_param(prefix, "beta", "反馈系数 β", 0.0, 2.0, 0.5, 0.1)
    with right:
        num1, den1 = np.array([1.0]), np.array([tau1, 1.0])
        num2, den2 = np.array([1.0]), np.array([tau2, 1.0])
        num_series = np.polymul(num1, num2)
        den_series = np.polymul(den1, den2)
        num_parallel = np.polyadd(np.polymul(num1, den2), np.polymul(num2, den1))
        den_parallel = den_series
        num_feedback = num_series.copy()
        den_feedback = np.polyadd(den_series, beta * num_series)
        t = np.linspace(0, 10, 1200)
        _, y_series = signal.step(signal.TransferFunction(num_series, den_series), T=t)
        _, y_parallel = signal.step(signal.TransferFunction(num_parallel, den_parallel), T=t)
        _, y_feedback = signal.step(signal.TransferFunction(num_feedback, den_feedback), T=t)
        render_chart("级联 / 并联 / 反馈结构阶跃响应", t, [{"name": "级联", "y": y_series}, {"name": "并联", "y": y_parallel}, {"name": "负反馈", "y": y_feedback}], "ch7s4_main", "t / s", "幅值")
        st.code(f"H_cascade(s)=H1(s)H2(s)\nH_parallel(s)=H1(s)+H2(s)\nH_feedback(s)=G(s)/(1+βG(s)), β={beta:.2f}", language="text")
    note_block("4. 系统级联、并联、反馈三种结构仿真")


def render_ch8_sec1():
    prefix = "ch8s1"
    defaults = {"a1": 2.0, "a0": 3.0, "b0": 1.0}
    example = {"a1": 1.0, "a0": 2.0, "b0": 1.5}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        a1 = slider_param(prefix, "a1", "微分方程系数 a1", 0.2, 5.0, 2.0, 0.1)
        a0 = slider_param(prefix, "a0", "微分方程系数 a0", 0.2, 5.0, 3.0, 0.1)
        b0 = slider_param(prefix, "b0", "输入系数 b0", 0.2, 3.0, 1.0, 0.1)
    with right:
        A = np.array([[0.0, 1.0], [-a0, -a1]])
        B = np.array([[0.0], [b0]])
        C = np.array([[1.0, 0.0]])
        D = np.array([[0.0]])
        sys = signal.StateSpace(A, B, C, D)
        t = np.linspace(0, 8, 1200)
        u = np.ones_like(t)
        tout, y, x = signal.lsim(sys, U=u, T=t)
        render_chart("状态变量建立后的系统响应", tout, [{"name": "状态 x1(t)", "y": x[:, 0]}, {"name": "状态 x2(t)", "y": x[:, 1]}, {"name": "输出 y(t)", "y": y}], "ch8s1_main", "t / s", "幅值")
        st.code(f"A = {np.round(A, 4).tolist()}\nB = {np.round(B, 4).tolist()}\nC = {np.round(C, 4).tolist()}\nD = {np.round(D, 4).tolist()}", language="text")
    note_block("1. 状态方程、输出方程建立")


def render_ch8_sec2():
    prefix = "ch8s2"
    defaults = {"mode": "连续系统", "zeta": 0.3, "wn": 2.0, "alpha": 0.7, "x10": 1.0, "x20": 0.0}
    example = {"mode": "离散系统", "zeta": 0.3, "wn": 2.0, "alpha": 0.9, "x10": 0.0, "x20": 1.0}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        mode = select_param(prefix, "mode", "求解类型", ["连续系统", "离散系统"], 0)
        zeta = slider_param(prefix, "zeta", "连续阻尼比 ζ", 0.0, 1.2, 0.3, 0.05)
        wn = slider_param(prefix, "wn", "连续固有角频率 ωn", 0.5, 5.0, 2.0, 0.1)
        alpha = slider_param(prefix, "alpha", "离散参数 α", 0.2, 1.2, 0.7, 0.05)
        x10 = slider_param(prefix, "x10", "初始状态 x1(0)", -2.0, 2.0, 1.0, 0.1)
        x20 = slider_param(prefix, "x20", "初始状态 x2(0)", -2.0, 2.0, 0.0, 0.1)
    with right:
        if mode == "连续系统":
            A = np.array([[0.0, 1.0], [-(wn**2), -2 * zeta * wn]])
            B = np.array([[0.0], [1.0]])
            C = np.array([[1.0, 0.0]])
            D = np.array([[0.0]])
            sys = signal.StateSpace(A, B, C, D)
            t = np.linspace(0, 10, 1400)
            u = np.ones_like(t)
            tout, y, x = signal.lsim(sys, U=u, T=t, X0=[x10, x20])
            render_chart("连续系统状态方程求解", tout, [{"name": "x1(t)", "y": x[:, 0]}, {"name": "x2(t)", "y": x[:, 1]}, {"name": "y(t)", "y": y}], "ch8s2_cont", "t / s", "幅值")
        else:
            A = np.array([[1.0, 1.0], [0.0, alpha]])
            B = np.array([[0.0], [1.0]])
            C = np.array([[1.0, 0.0]])
            n = np.arange(0, 25)
            x_hist = np.zeros((len(n), 2))
            x_hist[0] = [x10, x20]
            u = np.ones(len(n))
            for k in range(len(n) - 1):
                x_hist[k + 1] = (A @ x_hist[k].reshape(-1, 1) + B * u[k]).ravel()
            y = (C @ x_hist.T).ravel()
            render_chart("离散系统状态方程求解", n, [{"name": "x1[n]", "y": x_hist[:, 0]}, {"name": "x2[n]", "y": x_hist[:, 1]}, {"name": "y[n]", "y": y}], "ch8s2_disc", "n", "幅值", discrete=True)
    note_block("2. 连续/离散系统状态方程求解")


def controllability_matrix(A, B):
    n = A.shape[0]
    return np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])


def observability_matrix(A, C):
    n = A.shape[0]
    return np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(n)])


def render_ch8_sec3():
    prefix = "ch8s3"
    presets = {
        "完全可控可观测": {"A": np.array([[0.0, 1.0], [-2.0, -3.0]]), "B": np.array([[0.0], [1.0]]), "C": np.array([[1.0, 0.0]])},
        "不可控示例": {"A": np.array([[1.0, 0.0], [0.0, 2.0]]), "B": np.array([[1.0], [0.0]]), "C": np.array([[1.0, 1.0]])},
        "不可观测示例": {"A": np.array([[1.0, 1.0], [0.0, 1.0]]), "B": np.array([[0.0], [1.0]]), "C": np.array([[1.0, 0.0]])},
    }
    defaults = {"preset": "完全可控可观测"}
    example = {"preset": "不可控示例"}
    left, right = st.columns([1, 2])
    with left:
        control_buttons(prefix, defaults, example)
        preset = select_param(prefix, "preset", "矩阵预设", list(presets.keys()), 0)
    with right:
        A = presets[preset]["A"]
        B = presets[preset]["B"]
        C = presets[preset]["C"]
        ctrb = controllability_matrix(A, B)
        obsv = observability_matrix(A, C)
        rank_c = int(np.linalg.matrix_rank(ctrb))
        rank_o = int(np.linalg.matrix_rank(obsv))
        n = A.shape[0]
        c1, c2 = st.columns(2)
        c1.metric("可控性矩阵秩", f"{rank_c}/{n}")
        c2.metric("可观测性矩阵秩", f"{rank_o}/{n}")
        st.code(f"A = {np.round(A, 4).tolist()}\nB = {np.round(B, 4).tolist()}\nC = {np.round(C, 4).tolist()}\n可控性矩阵 = {np.round(ctrb, 4).tolist()}\n可观测性矩阵 = {np.round(obsv, 4).tolist()}", language="text")
        st.write(f"判定结果：系统{'可控' if rank_c == n else '不可控'}，系统{'可观测' if rank_o == n else '不可观测'}。")
    note_block("3. 系统可控性、可观测性判定演示")


RENDERERS = {
    "1. 连续/离散信号区分演示": render_ch1_sec1,
    "2. 周期/非周期、能量/功率信号判定仿真": render_ch1_sec2,
    "3. 信号基础运算：加减乘、反转、时移、尺度变换": render_ch1_sec3,
    "4. 阶跃函数、冲激函数定义、广义函数、微积分性质可视化": render_ch1_sec4,
    "5. 系统线性、时不变、因果、稳定四大特性交互验证": render_ch1_sec5,
    "1. LTI连续系统微分方程经典解演示": render_ch2_sec1,
    "2. 零输入/零状态、全响应拆分对比": render_ch2_sec2,
    "3. 单位冲激响应、单位阶跃响应波形生成": render_ch2_sec3,
    "4. 卷积积分动态分步演示": render_ch2_sec4,
    "5. 卷积全部代数性质、微分积分性质仿真验证": render_ch2_sec5,
    "1. LTI离散系统、差分方程求解": render_ch3_sec1,
    "2. 离散系统零输入、零状态响应": render_ch3_sec2,
    "3. 单位序列、单位阶跃序列响应": render_ch3_sec3,
    "4. 离散卷积和完整可视化": render_ch3_sec4,
    "5. 反卷积简易演示": render_ch3_sec5,
    "1. 信号正交函数分解原理": render_ch4_sec1,
    "2. 周期信号傅里叶级数、奇偶函数频谱": render_ch4_sec2,
    "3. 周期/非周期信号幅度谱、相位谱绘制": render_ch4_sec3,
    "4. 傅里叶变换全部核心性质逐一交互验证": render_ch4_sec4,
    "5. 能量谱、功率谱、周期信号傅里叶变换": render_ch4_sec5,
    "6. LTI系统频率响应、频域完整分析": render_ch4_sec6,
    "7. 无失真传输、理想低通滤波器效果仿真": render_ch4_sec7,
    "8. 奈奎斯特取样定理、时域/频域取样全过程演示": render_ch4_sec8,
    "9. 离散时间傅里叶变换DTFT、DFT可视化": render_ch4_sec9,
    "1. 单边/双边拉普拉斯变换、收敛域ROC": render_ch5_sec1,
    "2. 拉普拉斯变换全套性质逐一验证": render_ch5_sec2,
    "3. 拉普拉斯逆变换（查表法+部分分式展开）": render_ch5_sec3,
    "4. s域复频域微分方程求解": render_ch5_sec4,
    "5. 系统函数H(s)、s域框图、电路s域建模": render_ch5_sec5,
    "6. 拉普拉斯与傅里叶变换关系对照": render_ch5_sec6,
    "1. Z变换定义、收敛域": render_ch6_sec1,
    "2. Z变换全套8大性质交互演示": render_ch6_sec2,
    "3. 逆Z变换：幂级数展开、部分分式法": render_ch6_sec3,
    "4. 差分方程z域完整求解": render_ch6_sec4,
    "5. 离散系统函数H(z)、z域框图": render_ch6_sec5,
    "6. s域与z域映射关系": render_ch6_sec6,
    "1. 系统零极点分布、时域/频域响应对应关系": render_ch7_sec1,
    "2. 系统因果性、稳定性判定仿真": render_ch7_sec2,
    "3. 信号流图、梅森公式可视化演示": render_ch7_sec3,
    "4. 系统级联、并联、反馈三种结构仿真": render_ch7_sec4,
    "1. 状态方程、输出方程建立": render_ch8_sec1,
    "2. 连续/离散系统状态方程求解": render_ch8_sec2,
    "3. 系统可控性、可观测性判定演示": render_ch8_sec3,
}


def main():
    chapter, section = chapter_intro()
    section_header(chapter, section)
    st.caption(f"当前章节共 {len(BOOK_STRUCTURE[chapter])} 个交互小节，全书共 {sum(len(v) for v in BOOK_STRUCTURE.values())} 个小节。")
    RENDERERS[section]()
    st.markdown("---")
    st.markdown("### 章节考点总结")
    st.write("本工具把教材公式、波形、频谱、零极点和状态变量统一到同一界面，适合课前预习、课堂演示、期末复习和考研专业课训练。")
    st.caption("说明：本项目仅用于《信号与线性系统》课程学习与可视化理解，不替代教材原文与课堂教学。")


if __name__ == "__main__":
    main()
