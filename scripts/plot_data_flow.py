#!/usr/bin/env python
"""scripts/plot_data_flow.py — 相控阵寿命预测数据流链路图

对标电池组件 fig09_data_flow, 区分三层:
  蓝 = 在轨可观测层 (BIT/定标/遥测, 在线推理输入)
  绿 = 模型预测管线 (特征提取 → HI 派生 → encoder → RUL)
  红 = 仿真真值层 (仅训练/评估标签, 在轨不可直接观测)

核心: 器件参数真值 (RDS, IDSS, gm) 在轨不可直接观测 → 用红色虚线框,
z_true/RUL_true 仅作标签 → 堵住标签泄漏质疑。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_OBS = "#2563eb"
C_MODEL = "#16a34a"
C_TRUE = "#dc2626"


def draw_box(ax, x, y, w, h, text, color, alpha=0.12, fontsize=8.5,
             linestyle="-", linewidth=2):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                         facecolor=color, alpha=alpha, edgecolor=color,
                         linewidth=linewidth, linestyle=linestyle)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#1a1a1a", linespacing=1.4)


def draw_arrow(ax, x1, y1, x2, y2, color="#666", style="-|>", lw=1.3):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color,
                            linewidth=lw, mutation_scale=14)
    ax.add_patch(arrow)


fig, ax = plt.subplots(figsize=(13.5, 8.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10.5)
ax.axis("off")

# ---- 列标题 ----
ax.text(2.5, 10.0, "在轨可观测层\n(BIT / 定标 / 遥测)", ha="center", fontsize=11,
        color=C_OBS, fontweight="bold", linespacing=1.3)
ax.text(7, 10.0, "模型预测管线", ha="center", fontsize=11,
        color=C_MODEL, fontweight="bold", linespacing=1.3)
ax.text(11.5, 10.0, "仿真真值层\n(仅训练/评估标签)", ha="center", fontsize=11,
        color=C_TRUE, fontweight="bold", linespacing=1.3)

# ---- 左列: 在线遥测 ----
draw_box(ax, 0.3, 8.0, 4.4, 1.3,
         "BIT 内定标环路\n通道幅相 (±0.3 dB / ±3°)\n每 24 h 全阵定标一次", C_OBS)
draw_box(ax, 0.3, 6.5, 4.4, 1.0, "子阵温度传感器 (16 点)", C_OBS)
draw_box(ax, 0.3, 5.2, 4.4, 1.0, "子阵直流电流 (16 路)", C_OBS)
draw_box(ax, 0.3, 3.9, 4.4, 1.0, "地面信标反演 EIRP\n(轨次级精度)", C_OBS)
draw_box(ax, 0.3, 2.6, 4.4, 1.0, "工程观测模型 → HI$_{\\text{obs}}$\n(幅相量化 + 测量噪声\n+ 慢变偏置 + 定标周期)", C_OBS, alpha=0.2)

# ---- 中列: 模型管线 ----
draw_box(ax, 5.3, 8.0, 3.4, 1.3,
         "特征提取\n(幅相漂移 → HI$_{\\text{obs}}$)", C_MODEL)
draw_box(ax, 5.3, 5.8, 3.4, 1.7,
         "TCN / GRU Encoder\n(HI 动力学层)\n源域预训练 → 冻结/微调", C_MODEL)
draw_box(ax, 5.3, 3.5, 3.4, 1.5,
         "RUL 预测头\n+ 删失 hinge 损失", C_MODEL)
draw_box(ax, 5.3, 1.5, 3.4, 1.5,
         "在线 RUL 输出\nRMSE ≈ 235 天\n(8 年任务期 8%)", C_MODEL, alpha=0.22)

# ---- 右列: 仿真真值 (仅标签) ----
draw_box(ax, 9.3, 7.5, 4.4, 1.5,
         "器件参数真值\nR$_{\\text{DS}}$, I$_{\\text{DSS}}$, g$_m$\n(!) 在轨不可直接观测", C_TRUE,
         alpha=0.08, linestyle="--")
draw_box(ax, 9.3, 5.5, 4.4, 1.3,
         "z$_{\\text{true}}$ = max(ΔR/δR, ΔI/δI, Δg/δg)\n(退化指标真值)", C_TRUE,
         alpha=0.08, linestyle="--")
draw_box(ax, 9.3, 3.5, 4.4, 1.3,
         "RUL$_{\\text{true}}$\n(仿真器物理前推)", C_TRUE,
         alpha=0.08, linestyle="--")
draw_box(ax, 9.3, 1.5, 4.4, 1.3,
         "一致性验证\n(仿真器 <-> 物理孪生\n逐字一致)", C_TRUE,
         alpha=0.08, linestyle="--")

# ---- 箭头: 观测 → 模型 ----
draw_arrow(ax, 4.7, 8.65, 5.3, 8.65, color=C_OBS)
draw_arrow(ax, 4.7, 3.1, 5.3, 8.2, color=C_OBS)   # HI_obs → 特征提取

# ---- 箭头: 模型内部 ----
draw_arrow(ax, 7.0, 8.0, 7.0, 7.5, color=C_MODEL)
draw_arrow(ax, 7.0, 5.8, 7.0, 5.0, color=C_MODEL)
draw_arrow(ax, 7.0, 3.5, 7.0, 3.0, color=C_MODEL)

# ---- 箭头: 真值 → 模型 (训练监督) ----
draw_arrow(ax, 9.3, 6.0, 8.7, 6.5, color=C_TRUE)
ax.text(9.2, 6.8, "训练监督\n(仅离线)", fontsize=7, color=C_TRUE, ha="center",
        style="italic")
draw_arrow(ax, 9.3, 4.0, 8.7, 4.2, color=C_TRUE)

# ---- 分隔线 ----
ax.axvline(x=4.95, ymin=0.08, ymax=0.88, color="#94a3b8", linestyle=":", alpha=0.4, linewidth=1)
ax.axvline(x=9.05, ymin=0.08, ymax=0.88, color="#94a3b8", linestyle=":", alpha=0.4, linewidth=1)

# 底部注释
ax.text(7, 0.5, "核心保证：在线推理（蓝→绿）仅使用在轨可观测量；"
        "器件参数真值（红）仅在训练阶段提供监督标签，不进入在线推理路径",
        ha="center", fontsize=8.5, color="#475569", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f1f5f9", edgecolor="#cbd5e1"))

ax.set_title("相控阵天线寿命预测数据流 — 在线观测 vs 仿真真值严格分离",
             fontsize=13, fontweight="bold", pad=15)

fig.tight_layout()
fig.savefig("pa_data_flow.png", dpi=150, bbox_inches="tight")
print("-> pa_data_flow.png")
