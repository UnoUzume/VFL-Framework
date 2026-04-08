import matplotlib.pyplot as plt
import numpy as np

# 1. 初始化 20x20 的坐标网格
N = 20
X, Y = np.meshgrid(np.arange(N), np.arange(N))

# 2. 生成窄山峰（非对称衰减）
# 靠近原点的一侧（<3）使用极大的衰减系数 2.0，使其在轴上降至 e^(-18) ≈ 0，几乎绝对为 0
# 远离原点的一侧（>=3）保持原来 0.2 的平缓衰减
decay_y = np.where(Y < 3, 0.5, 0.2)
decay_x = np.where(X < 3, 0.5, 0.2)

ridge_x = np.where(X >= 3, ((X - 3) / 25) ** 0.4, 0) * np.exp(-decay_y * (Y - 3) ** 2)
ridge_y = np.where(Y >= 3, ((Y - 3) / 25) ** 0.4, 0) * np.exp(-decay_x * (X - 3) ** 2)
Z = ridge_x + ridge_y

# 3. 添加随机抖动，并使用掩码完全替代原有的强制清零
noise = np.abs(np.random.normal(loc=0.0, scale=0.5, size=(N, N)))

# 创建合法区域掩码：X 不为 0，Y 不为 0，且不等于对角线
# True 会在乘法中转换为 1，False 转换为 0
# valid_area = (X != 0) & (Y != 0) & (X != Y)

# 仅在合法区域保留地形和噪声
Z += noise * 0.1

# ---------------- 第 1 张图：热力图可视化与保存 ----------------
fig1, ax1 = plt.subplots(figsize=(8, 6))
heatmap = ax1.imshow(Z, cmap='viridis', origin='lower')

cbar = fig1.colorbar(heatmap)
cbar.set_label('Elevation (Z)', rotation=270, labelpad=15)

ax1.set_title('20x20 Matrix Heatmap')
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_xticks(np.arange(0, N, 2))
ax1.set_yticks(np.arange(0, N, 2))

output_filename1 = 'matrix_heatmap.png'
plt.savefig(output_filename1, dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f'热力图已成功保存为：{output_filename1}')

# ---------------- 第 2 张图：3D 图可视化与保存 (从 20,20 向原点看) ----------------
fig2 = plt.figure(figsize=(10, 8))
# 添加 3D 坐标轴
ax2 = fig2.add_subplot(111, projection='3d')

# 绘制 3D 表面
surf = ax2.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# 【关键修改点】调整视角
# elev=30: 仰角设为 30 度（类似于俯视的高度）
# azim=45: 方位角设为 45 度，将摄像机放在对角线 X=Y 的正方向尽头，也就是 (20,20) 附近看向原点
ax2.view_init(elev=30, azim=45)

ax2.set_title('3D View from (20, 20) towards Origin (0, 0)')
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('Elevation (Z)')

output_filename2 = 'matrix_3d_view.png'
plt.savefig(output_filename2, dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f'3D 图已成功保存为：{output_filename2}')
