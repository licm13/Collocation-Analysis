import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------
# 0. 模拟数据 (基于 Vogel & Ménard 2023, Sect. 5)
# -----------------------------------------------------------------
# 我们将模拟标量方差，以便于理解，尽管原论文处理的是完整的协方差矩阵
# [cite: 4072]
np.random.seed(42)
n_samples = 100000

# 真实值 (Truth)
true_signal = np.random.normal(5.0, 2.0, n_samples)

# 定义真实的误差方差 (C_i)
true_C = {
    1: 1.0,  # C_1
    2: 0.8,  # C_2
    3: 1.2,  # C_3
    4: 0.6   # C_4
}

# 定义真实的误差依赖项 (D_ij)
# 我们模拟 Experiment 2/3 的情景 [cite: 4365, 4608]
true_D = {
    (1, 2): 0.0,
    (1, 3): 0.0,
    (1, 4): 0.0,
    (2, 3): 0.4,  # !!! 关键：基础三角形(1,2,3)中的依赖项，但会被我们忽略
    (2, 4): 0.3,  # 待估计
    (3, 4): 0.2   # 待估计
}

# 生成相关的随机误差
# 这是一个简化的生成过程，仅为匹配预设的协方差/依赖项
# 真实的误差 (e_i)
errors = {}
errors[1] = np.random.normal(0, np.sqrt(true_C[1]), n_samples)
errors[2] = np.random.normal(0, np.sqrt(true_C[2]), n_samples)
errors[3] = np.random.normal(0, np.sqrt(true_C[3]), n_samples)
errors[4] = np.random.normal(0, np.sqrt(true_C[4]), n_samples)

# 引入相关性 (简化处理)
# D(2,3) = 0.4
errors[2] += errors[3] * (true_D[(2, 3)] / (2 * true_C[3])) 
# D(2,4) = 0.3
errors[2] += errors[4] * (true_D[(2, 4)] / (2 * true_C[4]))
# D(3,4) = 0.2
errors[3] += errors[4] * (true_D[(3, 4)] / (2 * true_C[4]))

# 重新调整方差以接近目标 C_i (这是一个粗略的模拟)
for i in range(1, 5):
    errors[i] = (errors[i] - np.mean(errors[i])) / np.std(errors[i]) * np.sqrt(true_C[i])

# 生成 4 个观测数据集
datasets = {i: true_signal + errors[i] for i in range(1, 5)}

print("--- 真实误差统计 (Ture Error Statistics) ---")
print(f"C_1: {np.var(errors[1]):.4f} (Target: {true_C[1]})")
print(f"C_2: {np.var(errors[2]):.4f} (Target: {true_C[2]})")
print(f"C_3: {np.var(errors[3]):.4f} (Target: {true_C[3]})")
print(f"C_4: {np.var(errors[4]):.4f} (Target: {true_C[4]})")

# D_ij = E[e_i*e_j] + E[e_j*e_i] (在标量中 E[e_i*e_j] * 2)
# C_ij = E[e_i*e_j] (协方差)
D_23_sim = 2 * np.cov(errors[2], errors[3])[0, 1]
D_24_sim = 2 * np.cov(errors[2], errors[4])[0, 1]
D_34_sim = 2 * np.cov(errors[3], errors[4])[0, 1]
print(f"D(2,3): {D_23_sim:.4f} (Target: {true_D[(2,3)]})")
print(f"D(2,4): {D_24_sim:.4f} (Target: {true_D[(2,4)]})")
print(f"D(3,4): {D_34_sim:.4f} (Target: {true_D[(3,4)]})")


# -----------------------------------------------------------------
# 1. 计算已知的残差协方差 (Gamma_ij)
# -----------------------------------------------------------------
# 在实际应用中，这是我们唯一拥有的信息 [cite: 4085]
Gamma = {}
for i in range(1, 5):
    for j in range(i + 1, 5):
        # Gamma_ij = Var(d_i - d_j)
        Gamma[(i, j)] = np.var(datasets[i] - datasets[j])
        Gamma[(j, i)] = Gamma[(i, j)]

print("\n--- 可观测的残差方差 (Observed Residual Variances) ---")
print(f"Gamma(1,2): {Gamma[(1,2)]:.4f}")
print(f"Gamma(1,3): {Gamma[(1,3)]:.4f}")
print(f"Gamma(1,4): {Gamma[(1,4)]:.4f}")
print(f"Gamma(2,3): {Gamma[(2,3)]:.4f}")
print(f"Gamma(2,4): {Gamma[(2,4)]:.4f}")
print(f"Gamma(3,4): {Gamma[(3,4)]:.4f}")

# -----------------------------------------------------------------
# 2. 应用广义顺序估计 (Apply Generalized Sequential Estimation)
# -----------------------------------------------------------------
print("\n--- 开始顺序估计 (Sequential Estimation) ---")
# 我们遵循论文中的 Setup 1 
# 1. 定义基础三角形 (Basic Triangle): (1, 2, 3) [cite: 4090]
#    我们假设 D(1,2)=0, D(1,3)=0, D(2,3)=0
#    (注意：我们故意违反了 D(2,3)=0 的假设) 
#
# 2. 定义顺序估计 (Sequential Estimation):
#    数据集 4 以数据集 1 为参考 (ref(4) = 1) 
#    我们假设 D(1,4)=0

estimated_C = {}

# --- 步骤 2a: 估计基础三角形 (1, 2, 3) ---
# 使用标准 TC 公式 (Eq. 39) 
C1_est = 0.5 * (Gamma[(1, 2)] + Gamma[(1, 3)] - Gamma[(2, 3)])
C2_est = 0.5 * (Gamma[(1, 2)] + Gamma[(2, 3)] - Gamma[(1, 3)])
C3_est = 0.5 * (Gamma[(1, 3)] + Gamma[(2, 3)] - Gamma[(1, 2)])

estimated_C[1] = C1_est
estimated_C[2] = C2_est
estimated_C[3] = C3_est

print(f"[基础] 估计 C_1: {C1_est:.4f} (vs 真值: {true_C[1]:.4f})")
print(f"[基础] 估计 C_2: {C2_est:.4f} (vs 真值: {true_C[2]:.4f})")
print(f"[基础] 估计 C_3: {C3_est:.4f} (vs 真值: {true_C[3]:.4f})")

# --- 步骤 2b: 顺序估计数据集 4 (ref(4)=1) ---
# 使用顺序公式 (Eq. 49): C_i = Gamma(i, ref(i)) - C_ref(i) 
C4_est = Gamma[(1, 4)] - C1_est  # C1_est 是上一步计算得到的 
estimated_C[4] = C4_est

print(f"[顺序] 估计 C_4: {C4_est:.4f} (vs 真值: {true_C[4]:.4f})")


# --- 步骤 3: 估计误差依赖项 D_ij (Error Dependencies) ---
# 我们可以估计所有*未被假设为零*的依赖项 [cite: 4099]
# 在本例中，我们假设了 (1,2), (1,3), (2,3), (1,4) 为零
# 我们可以估计 (2,4) 和 (3,4)
#
# 使用公式 (Eq. 51): D_ij = C_i + C_j - Gamma(i,j) [cite: 3992]

estimated_D = {}

# 估计 D(2,4)
D_24_est = estimated_C[2] + estimated_C[4] - Gamma[(2, 4)]
estimated_D[(2, 4)] = D_24_est
print(f"[依赖] 估计 D(2,4): {D_24_est:.4f} (vs 真值: {D_24_sim:.4f})")

# 估计 D(3,4)
D_34_est = estimated_C[3] + estimated_C[4] - Gamma[(3, 4)]
estimated_D[(3, 4)] = D_34_est
print(f"[依赖] 估计 D(3,4): {D_34_est:.4f} (vs 真值: {D_34_sim:.4f})")


# -----------------------------------------------------------------
# 4. 讨论分析 (Discussion of Analysis)
# -----------------------------------------------------------------
print("\n--- 结果分析与讨论 (Analysis and Discussion) ---")
#
# 为什么估计值与真值有偏差？
# 来源：我们对 "基础三角形" (1,2,3) 的假设是错误的。
# 我们假设了 D(2,3) = 0，但实际上 D(2,3) = 0.4。
#
# 误差传播 (Error Propagation) (Sect. 4.1.3 & 4.2.4):
#
# 1. 基础三角形的误差 (Eq. 45)[cite: 3928]:
#    被忽略的 D(2,3) (Delta_D_23 = 0.4) 会导致：
#    - C_1 误差: +0.5 * (-Delta_D_23) = -0.2  [cite: 4375]
#    - C_2 误差: +0.5 * (+Delta_D_23) = +0.2  [cite: 4374]
#    - C_3 误差: +0.5 * (+Delta_D_23) = +0.2  [cite: 4374]
#
print(f"C_1 误差: {C1_est - true_C[1]:.4f} (理论: -0.2)")
print(f"C_2 误差: {C2_est - true_C[2]:.4f} (理论: +0.2)")
print(f"C_3 误差: {C3_est - true_C[3]:.4f} (理论: +0.2)")
#
# 2. 顺序估计的误差 (Eq. 54)[cite: 4011]:
#    C_4 的误差 (Delta_C_4) 取决于其参考 C_1 的误差 (Delta_C_1)。
#    Delta_C_4 = Delta_D_14 - Delta_C_1 = 0 - Delta_C_1
#    C_4 误差: -(-0.2) = +0.2 [cite: 4378]
#
print(f"C_4 误差: {C4_est - true_C[4]:.4f} (理论: +0.2)")
#
# 3. 依赖项估计的误差 (Eq. 57)[cite: 4026]:
#    Delta_D_24 = Delta_C_2 + Delta_C_4 = (+0.2) + (+0.2) = +0.4 [cite: 4381]
#    Delta_D_34 = Delta_C_3 + Delta_C_4 = (+0.2) + (+0.2) = +0.4 [cite: 4381]
#
print(f"D(2,4) 误差: {D_24_est - D_24_sim:.4f} (理论: +0.4)")
print(f"D(3,4) 误差: {D_34_est - D_34_sim:.4f} (理论: +0.4)")
#
# 结论：
# 如此示例所示（与论文中的图 3 一致）[cite: 4601]，
# 在 "基础多边形" (Basic Polygon) 上的一个错误假设（例如忽略 D(2,3)）
# 会系统性地传播到所有后续的估计中，包括顺序估计的 C_i 
# 和计算得到的 D_ij。
#
# 这凸显了 Vogel & Ménard (2023) 的核心观点：
# 必须仔细选择假设（即选择哪些数据集对假设为独立），
# 将最可靠的独立假设用于 "基础多边形" [cite: 4909, 4921]。

# -----------------------------------------------------------------
# 5. 可视化与保存
# -----------------------------------------------------------------
# 保存到脚本同目录的 figures 文件夹
script_dir = Path(__file__).resolve().parent
fig_dir = script_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
script_stem = Path(__file__).stem

# 5.1 误差方差 C_i 真值 vs 估计
labels_c = ['C1', 'C2', 'C3', 'C4']
true_c_vals = [true_C[1], true_C[2], true_C[3], true_C[4]]
est_c_vals = [estimated_C[1], estimated_C[2], estimated_C[3], estimated_C[4]]

plt.figure(figsize=(7, 4))
x = np.arange(len(labels_c))
width = 0.35
plt.bar(x - width/2, true_c_vals, width, label='True C', color='#4C78A8')
plt.bar(x + width/2, est_c_vals, width, label='Estimated C', color='#F58518')
plt.xticks(x, labels_c)
plt.ylabel('Variance')
plt.title('Error Variance Comparison (C_i)')
plt.legend()
plt.tight_layout()
out1 = fig_dir / f"{script_stem}_variance_comparison.png"
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()

# 5.2 依赖项 D(2,4), D(3,4) 真值 vs 估计
labels_d = ['D(2,4)', 'D(3,4)']
true_d_vals = [D_24_sim, D_34_sim]
est_d_vals = [estimated_D[(2, 4)], estimated_D[(3, 4)]]

plt.figure(figsize=(7, 4))
x = np.arange(len(labels_d))
plt.bar(x - width/2, true_d_vals, width, label='True D', color='#54A24B')
plt.bar(x + width/2, est_d_vals, width, label='Estimated D', color='#E45756')
plt.xticks(x, labels_d)
plt.ylabel('Dependency')
plt.title('Error Dependency Comparison (D_ij)')
plt.legend()
plt.tight_layout()
out2 = fig_dir / f"{script_stem}_dependency_comparison.png"
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()

# 5.3 残差差分方差 Gamma 矩阵热图
Gamma_mat = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        if i == j:
            Gamma_mat[i, j] = 0.0
        else:
            a, b = min(i+1, j+1), max(i+1, j+1)
            Gamma_mat[i, j] = Gamma[(a, b)]

plt.figure(figsize=(5.5, 4.5))
im = plt.imshow(Gamma_mat, cmap='viridis')
plt.colorbar(im, fraction=0.046, pad=0.04, label='Var(d_i - d_j)')
plt.xticks(range(4), ['1', '2', '3', '4'])
plt.yticks(range(4), ['1', '2', '3', '4'])
plt.xlabel('j')
plt.ylabel('i')
plt.title('Observed Residual Variances Γ(i,j)')
plt.tight_layout()
out3 = fig_dir / f"{script_stem}_gamma_heatmap.png"
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ Figures saved to: {fig_dir}")
print(f"  - {out1.name}")
print(f"  - {out2.name}")
print(f"  - {out3.name}")