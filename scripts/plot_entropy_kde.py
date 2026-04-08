import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde


def plot_entropy_distribution(
	file_path: str, epoch_to_eval: int, target_class: int = 0, optimal_m1: float = 0.02
) -> None:
	print(f'Loading data from: {file_path}')
	all_data = torch.load(file_path)

	if epoch_to_eval not in all_data:
		print(f'Epoch {epoch_to_eval} not found in {file_path}')
		return

	# 1. 提取指定 Epoch 的数据
	data = all_data[epoch_to_eval]

	# 2. 数据清洗：屏蔽目标类别
	valid_mask = data['labels'] != target_class
	valid_entropy = data['entropy'][valid_mask]  # [N, 4]
	valid_is_attack = data['is_attack'][valid_mask]  # [N]
	tMeanEntropy = data['mean_entropy']  # * (nParty)

	# 3. 锁定攻击方 (Party 0) 并划分干净样本与投毒样本
	# 将 Tensor 转为 numpy 数组方便 seaborn 绘图
	entropy_party_0 = valid_entropy[:, 0].numpy()
	is_attack_np = valid_is_attack.numpy()

	clean_entropy = entropy_party_0[is_attack_np == 0] + 1e-8
	poisoned_entropy = entropy_party_0[is_attack_np == 1] + 1e-8

	# 4. 计算真实环境下的绝对物理阈值
	# 根据你前文微观热力图的最佳参数 m1=0.02，结合全局均值还原出真实的分割线
	absolute_threshold = optimal_m1 * tMeanEntropy[0]
	print(f'Global Entropy Mean: {tMeanEntropy[0]:.4f}')
	print(f'Absolute Cut-off Threshold (m1={optimal_m1}): {absolute_threshold:.4f}')

	# 1. 计算干净样本的 KDE
	kde_clean = gaussian_kde(clean_entropy)
	# 生成 500 个均匀分布的 X 点 (范围从 0 到 2.5)
	x_clean = np.linspace(0, 2.5, 500)
	y_clean = kde_clean(x_clean)

	# 保存为 txt，第一列 X，第二列 Y
	np.savetxt(
		'kde_clean.txt', np.column_stack((x_clean, y_clean)), fmt='%.5f', header='x y', comments=''
	)

	# 2. 计算投毒样本的 KDE (坍缩的针)
	kde_poisoned = gaussian_kde(poisoned_entropy)
	# 针的范围很窄，X 取 0 到 0.05 即可，保证分辨率
	x_poisoned = np.linspace(0, 0.05, 500)
	y_poisoned = kde_poisoned(x_poisoned)

	np.savetxt(
		'kde_poisoned.txt',
		np.column_stack((x_poisoned, y_poisoned)),
		fmt='%.5f',
		header='x y',
		comments='',
	)

	print('KDE coordinates saved for TikZ!')

	# ==========================================
	# 5. 面向对象的高级学术绘图
	# ==========================================
	# 显式创建 Figure 和 Axes
	fig, ax1 = plt.subplots(figsize=(10, 7), dpi=300)

	# 创建一个共用 X 轴，但具有独立 Y 轴的坐标系
	ax2 = ax1.twinx()

	# --- 绘制正常样本 (对应左侧 ax1) ---
	sns.kdeplot(
		clean_entropy,
		color='#1f77b4',  # 学术蓝
		fill=True,  # 曲线下方填充颜色
		cut=0,  # 核心参数：告诉算法不要把高斯尾巴延伸到数据范围之外
		clip=(0, None),  # 核心参数：严格将曲线限制在 [0, 正无穷] 之间
		alpha=0.4,  # 透明度
		linewidth=2.5,  # 边框线宽
		label='Clean Samples (Origin Branch)',
		ax=ax1,
	)

	# 设置左侧 ax1 轴标签样式
	ax1.set_ylabel('Density (Clean)', color='#1f77b4', fontsize=12)
	ax1.tick_params(axis='y', labelcolor='#1f77b4')  # Y 轴刻度数字也变蓝

	# --- 绘制攻击样本 (对应右侧 ax2) ---
	sns.kdeplot(
		poisoned_entropy,
		color='#d62728',  # 警示红
		fill=True,
		cut=0,  # 核心参数：告诉算法不要把高斯尾巴延伸到数据范围之外
		clip=(0, None),  # 核心参数：严格将曲线限制在 [0, 正无穷] 之间
		alpha=0.6,
		linewidth=2.5,
		label='Poisoned Samples (Attack Branch)',
		ax=ax2,
	)

	# 设置右侧 ax2 轴标签样式
	ax2.set_ylabel('Density (Poisoned)', color='#d62728', fontsize=12)
	ax2.tick_params(axis='y', labelcolor='#d62728')  # Y 轴刻度数字也变红

	# --- 添加完美的判定边界 (在共用的 X 轴上画即可) ---
	ax1.axvline(
		x=absolute_threshold, color='black', linestyle='--', linewidth=2.5, label='Optimal Boundary'
	)

	# --- 图表排版美化 ---
	# 添加你在右图里写的标题 (更专业，包含了 Epoch 信息)
	ax1.set_title(
		f'Prediction Entropy Density Distribution (Epoch {epoch_to_eval})', fontsize=14, pad=15
	)
	ax1.set_xlabel('Entropy Value ($\\mathcal{H}$)', fontsize=12)

	# 动态调整 X 轴范围 (聚焦核心冲突区)
	# 这里我们截取到干净分布主体，切掉右侧没有意义的长尾
	# max_x_plot = np.percentile(clean_entropy, 95)
	# ax1.set_xlim(-0.02, max_x_plot)

	# 极简学术风格：去顶框
	ax1.spines['top'].set_visible(False)
	ax2.spines['top'].set_visible(False)

	# --- 合并两个坐标系的图例 ---
	# 由于 ax1 和 ax2 各自管理自己的图例，这里需要提取并手动合并
	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()

	# 将图例放在左上角空白处，防止遮挡曲线
	# frameon=False 极其重要，学术界非常推崇极简风格
	ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False, fontsize=11)

	# ==========================================
	# 增加：画中画 (局部放大投毒样本的细节)
	# ==========================================
	# 在 ax1 内部的右上角 (loc='upper right') 开辟一块子区域，大小为主图的 35% x 35%
	ax_inset = inset_axes(
		ax1,
		width='35%',
		height='35%',
		loc='upper right',
		bbox_to_anchor=(0, -0.1, 1, 1),
		bbox_transform=ax1.transAxes,
	)

	# 在子图中仅绘制投毒样本 (红色)
	sns.kdeplot(poisoned_entropy, color='#d62728', fill=True, alpha=0.6, linewidth=2, ax=ax_inset)

	# 画上那条阈值虚线
	ax_inset.axvline(x=absolute_threshold, color='black', linestyle='--', linewidth=2)

	# 【核心】强制放大 X 轴靠近 0 的极小区域
	# 根据你的数据，这里可能需要调成 0 到 0.05，具体看红针有多宽
	ax_inset.set_xlim(-0.001, 0.10)

	# 子图美化：去掉不需要的文字，使其像一个纯粹的“放大镜”
	ax_inset.set_xlabel('Zoom-in: Poisoned', fontsize=10)
	ax_inset.set_ylabel('')
	ax_inset.set_yticks([])  # 隐藏子图的 Y 轴刻度，因为我们只关心分布形状

	# 画连接线 (可选)：在大图和子图之间画框框，表示这是哪里的放大
	# 如果觉得画面太乱可以把这两行删掉
	from mpl_toolkits.axes_grid1.inset_locator import mark_inset

	mark_inset(ax1, ax_inset, loc1=3, loc2=4, fc='none', ec='gray', linestyle=':')

	fig.tight_layout()
	# 保存文件名也自动包含 Epoch
	save_img_path = file_path.replace('.pt', f'_kde_dual_y_perfect_ep{epoch_to_eval}.png')
	fig.savefig(save_img_path, bbox_inches='tight')
	plt.close(fig)  # 循环画图时极其重要！
	print(f'\n[Done] Perfect Dual-Y Plot saved to: {save_img_path}\n')


if __name__ == '__main__':
	# 替换为你实际的数据文件路径
	FILE = 'data/logs/sgba_cifar10/lightning_logs/version_81/dEpochRecords.pt'

	# 假设你之前跑微观热力图发现 0.02 效果最好，这里直接传入
	if pathlib.Path(FILE).exists():
		plot_entropy_distribution(FILE, epoch_to_eval=39, target_class=0, optimal_m1=0.02)
	else:
		print('Data file not found!')
